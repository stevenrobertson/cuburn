import sys
import math
import re
import time
from itertools import cycle, repeat, chain, izip
from ctypes import *
from cStringIO import StringIO
import numpy as np
from scipy import ndimage

from fr0stlib import pyflam3
from fr0stlib.pyflam3._flam3 import *
from fr0stlib.pyflam3.constants import *

import pycuda.compiler
import pycuda.driver as cuda

from cuburn import affine
from cuburn.code import util, mwc, iter, filtering

def _chunk(l, cs):
    """
    Yield the contents of list ``l`` in chunks of size no more than ``cs``.
    """
    for i in range(0, len(l), cs):
        yield l[i:i+cs]

class Genome(object):
    """
    Normalizes and precalculates some properties of a Genome. Assumes that
    Genome argument passed in will not change.
    """
    # Fix the ctypes ugliness since switching to __getattribute__ in 2.7.
    # There are more elegant ways to do this, but I can't be bothered.
    def __getattr__(self, name):
        return getattr(self.cp, name)

    def __init__(self, ctypes_genome):
        self.cp = ctypes_genome
        self.xforms = [self.xform[i] for i in range(self.num_xforms)]
        dens = np.array([x.density for i, x in enumerate(self.xforms)
                         if i != self.final_xform_index])
        dens /= np.sum(dens)
        self.norm_density = [np.sum(dens[:i+1]) for i in range(len(dens))]
        self.camera_transform = self.calc_camera_transform()

    scale = property(lambda cp: 2.0 ** cp.zoom)
    adj_density = property(lambda cp: cp.sample_density * (cp.scale ** 2))
    ppu = property(lambda cp: cp.pixels_per_unit * cp.scale)

    def calc_camera_transform(cp):
        """
        An affine matrix which will transform IFS coordinates to image width
        and height. Assumes that width and height are constant.
        """
        g = Features.gutter
        if cp.estimator:
            # The filter shifts by this amount as a side effect of it being
            # written in a confusing and sloppy manner
            # TODO: this will be weird in an animation where one endpoint has
            # a radius of 0, and the other does not
            g -= Features.gutter / 2 - 1
        return ( affine.translate(0.5 * cp.width + g, 0.5 * cp.height + g)
               * affine.scale(cp.ppu, cp.ppu)
               * affine.translate(-cp._center[0], -cp._center[1])
               * affine.rotate(cp.rotate * 2 * np.pi / 360,
                               cp.rot_center[0],
                               cp.rot_center[1]) )

class Animation(object):
    """
    Control structure for rendering a series of frames.

    Each animation will dynamically generate a kernel that includes only the
    code necessary to render the genomes provided. The process of generating
    and uploading the kernel takes a small but finite amount of time. In
    general, the kernel generated for all genomes resulting from interpolating
    between two control points will have identical performance, so it is
    wasteful to create more than one animation for any interpolated sequence.

    However, genome sequences interpolated from three or more control points
    with different features enabled will have the code needed to render all
    genomes enabled for every frame. Doing this can hurt performance.

    In other words, it's best to use exactly one Animation for each
    interpolated sequence between one or two genomes.
    """
    def __init__(self, ctypes_genome_array):
        self._g_arr = type(ctypes_genome_array)()
        libflam3.flam3_align(self._g_arr, ctypes_genome_array, len(ctypes_genome_array))
        self.genomes = map(Genome, self._g_arr)
        self.features = Features(self.genomes)
        self._iter = self._de = self.src = self.cubin = self.mod = None

    def compile(self, keep=False,
                cmp_options=('-use_fast_math', '-maxrregcount', '32')):
        """
        Compile a kernel capable of rendering every frame in this animation.
        The resulting compiled kernel is stored in the ``cubin`` property;
        the source is available as ``src``, and is also returned for
        inspection and display.

        This operation is idempotent, and has no side effects outside of
        setting properties on this instance (unless there's a compiler error,
        which is a bug); it should therefore be threadsafe as well.
        It is, however, rather slow.
        """
        self._iter = iter.IterCode(self.features)
        self._de = filtering.DensityEst(self.features, self.genomes[0])
        # TODO: make choice of filtering explicit
        # TODO: autoload dependent modules?
        self.src = util.assemble_code(util.BaseCode, mwc.MWC, self._iter.packer,
                                      self._iter, filtering.ColorClip, self._de)
        self.cubin = pycuda.compiler.compile(self.src, keep=keep,
                                             options=list(cmp_options))
        return self.src

    def copy(self):
        """
        Return a copy of this animation without any references to the current
        CUDA context. This can be used to load an animation in multiple CUDA
        contexts without recompiling, so that rendering can proceed across
        multiple devices - but managing that is up to you.
        """
        import copy
        new = copy.copy(self)
        new.mod = None
        return new

    def load(self, jit_options=[]):
        """
        Replace the currently loaded CUDA module in the active CUDA context
        with the compiled code's module. A reference is kept to the module,
        meaning that rendering should henceforth only be called from the
        thread and context in which this function was called.
        """
        if self.cubin is None:
            self.compile()
        self.mod = cuda.module_from_buffer(self.cubin, jit_options)

    def render_frames(self, times=None):
        """
        Render a flame for each genome in the iterable value 'genomes'.
        Returns a Python generator object which will yield one NumPy array
        for each rendered image.

        This method produces a considerable amount of side effects, and should
        not be used lightly. Things may go poorly for you if this method is not
        allowed to run until completion (by exhausting all items in the
        generator object).

        A performance note: while any ready tasks will be scheduled on the GPU
        before yielding a result, spending a lot of time before returning
        control to this function can allow the GPU to become idle. It's best
        to hand the resulting array to another thread after grabbing it from
        the renderer for handling.

        ``times`` is a sequence of center times at which to render, or ``None``
        to render one frame for each genome used to create the animation.
        """
        # Don't see this changing, but empirical tests could prove me wrong
        NRENDERERS = 2
        # TODO: under a slightly modified sequencing, certain buffers can be
        # shared (though this may be unimportant if a good AA technique which
        # doesn't require full SS can be found)
        rdrs = [_AnimRenderer(self) for i in range(NRENDERERS)]

        # Zip up each genome with an alternating renderer, plus enough empty
        # genomes at the end to flush all pending tasks
        times = times if times is not None else [cp.time for cp in self.genomes]
        exttimes = chain(times, repeat(None, NRENDERERS))
        for rdr, time in izip(cycle(rdrs), exttimes):
            if rdr.wait():
                yield rdr.get_result()
            if time is not None:
                rdr.render(time)

    def _interp(self, time, cp):
        flam3_interpolate(self._g_arr, len(self._g_arr), time, 0, byref(cp))



class _AnimRenderer(object):
    # Large launches lock the display for a considerable period and may be
    # killed due to a device timeout; small launches are harder to load-balance
    # on the GPU and incur overhead. This empirical value is multiplied by the
    # number of SMs on the device to determine how many blocks should be in
    # each launch. Extremely high quality, high resolution renders may still
    # encounter a device timeout, and no workaround is in place for that yet.
    SM_FACTOR = 8

    # Currently, palette interpolation is done independently of animation
    # interpolation, so that the process is not biased and so we only need to
    # mess about with one texture per renderer. This many steps will always be
    # used, no matter the number of time steps.
    PAL_HEIGHT = 16


    def __init__(self, anim):
        self.anim = anim
        self.pending = False
        self.stream = cuda.Stream()

        self._nsms = cuda.Context.get_device().multiprocessor_count
        self.cps_per_block = self._nsms * self.SM_FACTOR
        self.ncps = anim.features.max_cps
        self.nblocks = int(math.ceil(self.ncps / float(self.cps_per_block)))

        # These are stored to avoid leaks, not to be stateful in method calls
        # TODO: ensure proper cleanup is done
        self._dst_cp = pyflam3.Genome()
        memset(byref(self._dst_cp), 0, sizeof(self._dst_cp))
        self._cen_cp = pyflam3.Genome()
        memset(byref(self._cen_cp), 0, sizeof(self._cen_cp))

        self.nbins = anim.features.acc_height * anim.features.acc_stride
        self.d_den = cuda.mem_alloc(4 * self.nbins)
        self.d_accum = cuda.mem_alloc(16 * self.nbins)
        self.d_out = cuda.mem_alloc(16 * self.nbins)
        self.d_infos = cuda.mem_alloc(anim._iter.packer.align * self.ncps)
        # Defer allocation until first needed
        self.d_seeds = [None] * self.nblocks

    def render(self, cen_time):
        assert not self.pending, "Tried to render with results pending!"
        self.pending = True
        a = self.anim

        cen_cp = self._cen_cp
        a._interp(cen_time, cen_cp)
        palette = self._interp_colors(cen_time, cen_cp)

        util.BaseCode.zero_dptr(a.mod, self.d_den, self.nbins,
                                self.stream)
        util.BaseCode.zero_dptr(a.mod, self.d_accum, 4 * self.nbins,
                                self.stream)

        # ------------------------------------------------------------
        # TODO WARNING TODO WARNING TODO WARNING TODO WARNING TODO
        # This will replace the palette while it's in use by the other
        # rendering function. Need to pass palette texref in function
        # invocation.
        # ------------------------------------------------------------
        dpal = cuda.make_multichannel_2d_array(palette, 'C')
        tref = a.mod.get_texref('palTex')
        tref.set_array(dpal)
        tref.set_format(cuda.array_format.UNSIGNED_INT8, 4)
        tref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        tref.set_filter_mode(cuda.filter_mode.LINEAR)

        cp = self._dst_cp
        packer = a._iter.packer

        iter_fun = a.mod.get_function("iter")
        iter_fun.set_cache_config(cuda.func_cache.PREFER_L1)

        # Must be accumulated over all CPs
        gam, vib = 0, 0

        # This is gross, but there are a lot of fiddly corner cases with any
        # index-based iteration scheme.
        times = list(enumerate(self._mk_dts(cen_time, cen_cp, self.ncps)))
        for b, block_times in enumerate(_chunk(times, self.cps_per_block)):
            infos = []
            if len(a.genomes) > 1:
                for n, t in block_times:
                    a._interp(t, cp)
                    frac = float(n) / cen_cp.ntemporal_samples
                    info = packer.pack(cp=Genome(cp), cp_step_frac=frac)
                    infos.append(info)
                    gam += cp.gamma
                    vib += cp.vibrancy
            else:
                # Can't interpolate normally; just pack copies
                # TODO: this still packs the genome 20 times or so instead of
                # once
                packed = packer.pack(cp=a.genomes[0], cp_step_frac=0)
                infos = [packed] * len(block_times)
                gam += a.genomes[0].gamma * len(block_times)
                vib += a.genomes[0].vibrancy * len(block_times)

            infos = np.concatenate(infos)
            offset = b * packer.align * self.cps_per_block
            # TODO: portable across 32/64-bit arches?
            d_info_off = int(self.d_infos) + offset
            cuda.memcpy_htod(d_info_off, infos)

            if not self.d_seeds[b]:
                seeds = mwc.MWC.make_seeds(iter.IterCode.NTHREADS *
                                           self.cps_per_block)
                self.d_seeds[b] = cuda.to_device(seeds)

            # TODO: get block config from IterCode
            # TODO: print timing information
            iter_fun(self.d_seeds[b], np.uint64(d_info_off),
                     self.d_accum, self.d_den,
                     block=(32, 16, 1), grid=(len(block_times), 1),
                     stream=self.stream)

        # MAJOR TODO: for now, we kill almost all parallelism by forcing the
        # stream here. Later, once we've decided on a density-buffer prefilter,
        # we will move it to the GPU, allowing it to be embedded in the stream
        # and letting the remaining code be asynchronous.
        self.stream.synchronize()
        dbuf_dim = (a.features.acc_height, a.features.acc_stride)
        dbuf = cuda.from_device(self.d_den, dbuf_dim, np.float32)
        dbuf = ndimage.filters.gaussian_filter(dbuf, 0.6)
        cuda.memcpy_htod(self.d_den, dbuf)

        util.BaseCode.zero_dptr(a.mod, self.d_out, 4 * self.nbins,
                                self.stream)
        self.stream.synchronize()
        a._de.invoke(a.mod, Genome(cen_cp),
                     self.d_accum, self.d_out, self.d_den,
                     self.stream)
        self.stream.synchronize()

        f = np.float32
        n = f(self.ncps)
        gam = f(n / gam)
        vib = f(vib / n)
        hipow = f(cen_cp.highlight_power)
        lin = f(cen_cp.gam_lin_thresh)
        lingam = f(math.pow(cen_cp.gam_lin_thresh, gam-1.0) if lin > 0 else 0)
        print gam, vib, lin, lingam, cen_cp.gamma

        # TODO: get block size from colorclip class? It actually does not
        # depend on that being the case
        color_fun = a.mod.get_function("colorclip")
        color_fun(self.d_out, gam, vib, hipow, lin, lingam,
                  block=(256, 1, 1), grid=(self.nbins / 256, 1),
                  stream=self.stream)

    def _interp_colors(self, cen_time, cen_cp):
        # TODO: any visible difference between uint8 and richer formats?
        pal = np.empty((self.PAL_HEIGHT, 256, 4), dtype=np.uint8)
        a = self.anim

        if len(a.genomes) > 1:
            # The typical case; applying real motion blur
            cp = self._dst_cp
            times = self._mk_dts(cen_time, cen_cp, self.PAL_HEIGHT)
            for n, t in enumerate(times):
                a._interp(t, cp)
                for i, e in enumerate(cp.palette.entries):
                    pal[n][i] = np.uint8(np.array(e.color) * 255.0)
        else:
            # Cannot call any interp functions on a single genome; rather than
            # have alternate code-paths, just copy the same colors everywhere
            for i, e in enumerate(a.genomes[0].palette.entries):
                # TODO: This triggers a RuntimeWarning
                pal[0][i] = np.uint8(np.array(e.color) * 255.0)
            pal[1:] = pal[0]
        return pal

    def wait(self):
        if self.pending:
            self.stream.synchronize()
            self.pending = False
            return True
        return False

    def get_result(self):
        a = self.anim
        g = a.features.gutter
        obuf_dim = (a.features.acc_height, a.features.acc_stride, 4)
        out = cuda.from_device(self.d_out, obuf_dim, np.float32)
        # TODO: performance?
        g = a.features.gutter
        out = np.delete(out, np.s_[:g], axis=0)
        out = np.delete(out, np.s_[:g], axis=1)
        out = np.delete(out, np.s_[-g:], axis=0)
        out = np.delete(out, np.s_[a.features.width:], axis=1)
        return out

    @staticmethod
    def _mk_dts(cen_time, cen_cp, ncps):
        w = cen_cp.temporal_filter_width
        return [cen_time + w * (t / (ncps - 1.0) - 0.5) for t in range(ncps)]

class Features(object):
    """
    Determine features and constants required to render a particular set of
    genomes. The values of this class are fixed before compilation begins.
    """
    # Constant parameters which control handling of out-of-frame samples:
    # Number of iterations to iterate without write after new point
    fuse = 20
    # Maximum consecutive out-of-bounds points before picking new point
    max_oob = 10
    max_nxforms = 12

    # Height of the texture pallete which gets uploaded to the GPU (assuming
    # that palette-from-texture is enabled). For most genomes, this doesn't
    # need to be very large at all. However, since only an easily-cached
    # fraction of this will be accessed per SM, larger values shouldn't hurt
    # performance too much. Power-of-two, please.
    palette_height = 16

    # Maximum width of DE and other spatial filters, and thus in turn the
    # amount of padding applied. Note that, for now, this must not be changed!
    # The filtering code makes deep assumptions about this value.
    gutter = 16

    def __init__(self, genomes):
        any = lambda l: bool(filter(None, map(l, genomes)))
        self.max_ntemporal_samples = max(
                [cp.nbatches * cp.ntemporal_samples for cp in genomes])
        self.non_box_temporal_filter = genomes[0].temporal_filter_type
        self.palette_mode = genomes[0].palette_mode and "linear" or "nearest"

        assert len(set([len(cp.xforms) for cp in genomes])) == 1, ("genomes "
            "must have same number of xforms! (use flam3-genome first)")
        self.nxforms = len(genomes[0].xforms)
        self.xforms = [XFormFeatures([cp.xforms[i] for cp in genomes], i)
                       for i in range(self.nxforms)]
        if any(lambda cp: cp.final_xform_enable):
            if not all([cp.final_xform_index == genomes[0].final_xform_index
                        for cp in genomes]):
                raise ValueError("Differing final xform indexes")
            self.final_xform_index = genomes[0].final_xform_index
        else:
            self.final_xform_index = None

        self.max_cps = max([cp.ntemporal_samples for cp in genomes])

        self.width = genomes[0].width
        self.height = genomes[0].height
        self.acc_width = genomes[0].width + 2 * self.gutter
        self.acc_height = genomes[0].height + 2 * self.gutter
        self.acc_stride = 32 * int(math.ceil(self.acc_width / 32.))

class XFormFeatures(object):
    def __init__(self, xforms, xform_id):
        self.id = xform_id
        any = lambda l: bool(filter(None, map(l, xforms)))
        self.has_post = any(lambda xf: getattr(xf, 'post', None))
        self.vars = set()
        for x in xforms:
            self.vars = (
                self.vars.union(set([i for i, v in enumerate(x.var) if v])))


