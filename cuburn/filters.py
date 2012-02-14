import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.driver as cuda
import pycuda.compiler
from pycuda.gpuarray import vec

import code.filters
from code.util import ClsMod, argset, launch

class Filter(object):
    def apply(self, fb, gnm, dim, tc, stream=None):
        """
        Queue the application of this filter. When the live stream finishes
        executing the last item enqueued by this method, the result must be
        live in the buffer pointed to by ``fb.d_front`` at the return of this
        function.
        """
        raise NotImplementedError()

class Bilateral(Filter, ClsMod):
    lib = code.filters.bilaterallib
    def __init__(self, directions=8, r=15, sstd=6, cstd=0.05,
                 dstd=1.5, dpow=0.8, gspeed=4.0):
        # TODO: expose these parameters on the genome, or at least on the
        # profile, and set them by a less ugly mechanism
        for n in 'directions r sstd cstd dstd dpow gspeed'.split():
            setattr(self, n, locals()[n])
        super(Bilateral, self).__init__()

    def apply(self, fb, gnm, dim, tc, stream=None):
        # Helper variables and functions to keep it clean
        sb = 16 * dim.astride
        bs = sb * dim.ah
        bl, gr = (32, 8, 1), (dim.astride / 32, dim.ah / 8)

        mkdsc = lambda c: argset(cuda.ArrayDescriptor(), height=dim.ah,
                                 width=dim.astride, num_channels=c,
                                 format=cuda.array_format.FLOAT)
        def mktref(n):
            tref = self.mod.get_texref(n)
            tref.set_filter_mode(cuda.filter_mode.POINT)
            tref.set_address_mode(0, cuda.address_mode.WRAP)
            tref.set_address_mode(1, cuda.address_mode.WRAP)
            return tref

        dsc = mkdsc(4)
        tref = mktref('bilateral_src')
        grad_dsc = mkdsc(1)
        grad_tref = mktref('blur_src')

        for pattern in range(self.directions):
            # Scale spatial parameter so that a "pixel" is equivalent to an
            # actual pixel at 1080p
            sstd = self.sstd * dim.w / 1920.

            tref.set_address_2d(fb.d_front, dsc, sb)

            # Blur density two octaves along sampling vector, ultimately
            # storing in the side buffer
            launch('den_blur', self.mod, stream, bl, gr,
                    fb.d_back, i32(pattern), i32(0), texrefs=[tref])
            grad_tref.set_address_2d(fb.d_back, grad_dsc, sb / 4)
            launch('den_blur_1c', self.mod, stream, bl, gr,
                    fb.d_side, i32(pattern), i32(1), texrefs=[grad_tref])
            grad_tref.set_address_2d(fb.d_side, grad_dsc, sb / 4)

            launch('bilateral', self.mod, stream, bl, gr,
                    fb.d_back, i32(pattern), i32(self.r),
                    f32(sstd), f32(self.cstd), f32(self.dstd),
                    f32(self.dpow), f32(self.gspeed),
                    texrefs=[tref, grad_tref])
            fb.flip()

class Logscale(Filter, ClsMod):
    lib = code.filters.logscalelib
    def apply(self, fb, gnm, dim, tc, stream=None):
        """Log-scale in place."""
        k1 = f32(gnm.color.brightness(tc) * 268 / 256)
        # Old definition of area is (w*h/(s*s)). Since new scale 'ns' is now
        # s/w, new definition is (w*h/(s*s*w*w)) = (h/(s*s*w))
        area = dim.h / (gnm.camera.scale(tc) ** 2 * dim.w)
        k2 = f32(1.0 / (area * gnm.spp(tc)))
        nbins = dim.ah * dim.astride
        launch('logscale', self.mod, stream, 256, nbins/256,
                fb.d_front, fb.d_front, k1, k2)

class ColorClip(Filter, ClsMod):
    lib = code.filters.colorcliplib
    def apply(self, fb, gnm, dim, tc, stream=None):
        # TODO: implement integration over cubic splines?
        gam = f32(1 / gnm.color.gamma(tc))
        vib = f32(gnm.color.vibrance(tc))
        hipow = f32(gnm.color.highlight_power(tc))
        lin = f32(gnm.color.gamma_threshold(tc))
        lingam = f32(lin ** (gam-1.0) if lin > 0 else 0)
        bkgd = vec.make_float3(
                gnm.color.background.r(tc),
                gnm.color.background.g(tc),
                gnm.color.background.b(tc))

        nbins = dim.ah * dim.astride
        blocks = int(np.ceil(np.sqrt(nbins / 256.)))
        launch('colorclip', self.mod, stream, 256, (blocks, blocks),
                fb.d_front, gam, vib, hipow, lin, lingam, bkgd, i32(nbins))
