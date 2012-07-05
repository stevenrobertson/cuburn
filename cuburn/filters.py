import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.driver as cuda
import pycuda.compiler
from pycuda.gpuarray import vec

import code.filters
from code.util import ClsMod, argset, launch2

def set_blur_width(mod, pool, stdev=1, stream=None):
    coefs = pool.allocate((7,), f32)
    coefs[:] = np.exp(np.float32(np.arange(-3, 4))**2/(-2*stdev**2))
    coefs /= np.sum(coefs)
    ptr, size = mod.get_global('gauss_coefs')
    cuda.memcpy_htod_async(ptr, coefs, stream)

def mktref(mod, n):
    tref = mod.get_texref(n)
    tref.set_filter_mode(cuda.filter_mode.POINT)
    tref.set_address_mode(0, cuda.address_mode.WRAP)
    tref.set_address_mode(1, cuda.address_mode.WRAP)
    return tref

def mkdsc(dim, ch):
    return argset(cuda.ArrayDescriptor(), height=dim.ah,
                  width=dim.astride, num_channels=ch,
                  format=cuda.array_format.FLOAT)

class Filter(object):
    # Set to True if the filter requires a full 4-channel side buffer
    full_side = False
    def apply(self, fb, gprof, params, dim, tc, stream=None):
        """
        Queue the application of this filter. When the live stream finishes
        executing the last item enqueued by this method, the result must be
        live in the buffer pointed to by ``fb.d_front`` at the return of this
        function.
        """
        raise NotImplementedError()

class Bilateral(Filter, ClsMod):
    lib = code.filters.bilaterallib
    radius = 15
    directions = 8

    def apply(self, fb, gprof, params, dim, tc, stream=None):
        # Helper variables and functions to keep it clean
        sb = 16 * dim.astride
        bs = sb * dim.ah

        dsc = mkdsc(dim, 4)
        tref = mktref(self.mod, 'chan4_src')
        grad_dsc = mkdsc(dim, 1)
        grad_tref = mktref(self.mod, 'chan1_src')
        set_blur_width(self.mod, fb.pool, stream=stream)

        for pattern in range(self.directions):
            # Scale spatial parameter so that a "pixel" is equivalent to an
            # actual pixel at 1080p
            sstd = params.spatial_std(tc) * dim.w / 1920.

            tref.set_address_2d(fb.d_front, dsc, sb)

            # Blur density two octaves along sampling vector, ultimately
            # storing in the side buffer
            launch2('den_blur', self.mod, stream, dim,
                    fb.d_back, i32(pattern), i32(0), texrefs=[tref])
            grad_tref.set_address_2d(fb.d_back, grad_dsc, sb / 4)
            launch2('den_blur_1c', self.mod, stream, dim,
                    fb.d_side, i32(pattern), i32(1), texrefs=[grad_tref])
            grad_tref.set_address_2d(fb.d_side, grad_dsc, sb / 4)

            launch2('bilateral', self.mod, stream, dim,
                    fb.d_back, i32(pattern), i32(self.radius),
                    f32(sstd), f32(params.color_std(tc)),
                    f32(params.density_std(tc)), f32(params.density_pow(tc)),
                    f32(params.gradient(tc)),
                    texrefs=[tref, grad_tref])
            fb.flip()

class Logscale(Filter, ClsMod):
    lib = code.filters.logscalelib
    def apply(self, fb, gprof, params, dim, tc, stream=None):
        """Log-scale in place."""
        k1 = f32(params.brightness(tc) * 268 / 256)
        # Old definition of area is (w*h/(s*s)). Since new scale 'ns' is now
        # s/w, new definition is (w*h/(s*s*w*w)) = (h/(s*s*w))
        area = dim.h / (params.scale(tc) ** 2 * dim.w)
        k2 = f32(1.0 / (area * gprof.spp(tc)))
        launch2('logscale', self.mod, stream, dim,
                fb.d_front, fb.d_front, k1, k2)

class HaloClip(Filter, ClsMod):
    lib = code.filters.halocliplib
    def apply(self, fb, gprof, params, dim, tc, stream=None):
        gam = f32(1 / gprof.filters.colorclip.gamma(tc) - 1)

        dsc = mkdsc(dim, 1)
        tref = mktref(self.mod, 'chan1_src')

        set_blur_width(self.mod, fb.pool, stream=stream)
        launch2('apply_gamma', self.mod, stream, dim,
                fb.d_side, fb.d_front, f32(0.1))
        tref.set_address_2d(fb.d_side, dsc, 4 * dim.astride)
        launch2('den_blur_1c', self.mod, stream, dim,
               fb.d_back, i32(2), i32(0), texrefs=[tref])
        tref.set_address_2d(fb.d_back, dsc, 4 * dim.astride)
        launch2('den_blur_1c', self.mod, stream, dim,
               fb.d_side, i32(3), i32(0), texrefs=[tref])

        launch2('haloclip', self.mod, stream, dim,
                fb.d_front, fb.d_side, gam)

def calc_lingam(params, tc):
    gam = f32(1 / params.gamma(tc))
    lin = f32(params.gamma_threshold(tc))
    lingam = f32(lin ** (gam-1.0) if lin > 0 else 0)
    return gam, lin, lingam

class SmearClip(Filter, ClsMod):
    full_side = True
    lib = code.filters.smearcliplib
    def apply(self, fb, gprof, params, dim, tc, stream=None):
        gam, lin, lingam = calc_lingam(gprof.filters.colorclip, tc)
        dsc = mkdsc(dim, 4)
        tref = mktref(self.mod, 'chan4_src')

        set_blur_width(self.mod, fb.pool, params.width(tc), stream)
        launch2('apply_gamma_full_hi', self.mod, stream, dim,
                fb.d_side, fb.d_front, f32(gam-1))
        tref.set_address_2d(fb.d_side, dsc, 16 * dim.astride)
        launch2('full_blur', self.mod, stream, dim,
               fb.d_back, i32(2), i32(0), texrefs=[tref])
        tref.set_address_2d(fb.d_back, dsc, 16 * dim.astride)
        launch2('full_blur', self.mod, stream, dim,
               fb.d_side, i32(3), i32(0), texrefs=[tref])
        tref.set_address_2d(fb.d_side, dsc, 16 * dim.astride)
        launch2('full_blur', self.mod, stream, dim,
               fb.d_back, i32(0), i32(0), texrefs=[tref])
        tref.set_address_2d(fb.d_back, dsc, 16 * dim.astride)
        launch2('full_blur', self.mod, stream, dim,
               fb.d_side, i32(1), i32(0), texrefs=[tref])
        launch2('smearclip', self.mod, stream, dim,
                fb.d_front, fb.d_side, f32(gam-1), lin, lingam)

class ColorClip(Filter, ClsMod):
    lib = code.filters.colorcliplib
    def apply(self, fb, gprof, params, dim, tc, stream=None):
        vib = f32(params.vibrance(tc))
        hipow = f32(params.highlight_power(tc))
        gam, lin, lingam = calc_lingam(params, tc)

        launch2('colorclip', self.mod, stream, dim,
                fb.d_front, vib, hipow, gam, lin, lingam)

# Ungainly but practical.
filter_map = dict(bilateral=Bilateral, logscale=Logscale, haloclip=HaloClip,
                  colorclip=ColorClip, smearclip=SmearClip)
def create(gprof):
    return [filter_map[f]() for f in gprof.filter_order]
