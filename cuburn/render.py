import os
import sys
import math
import re
import time as timemod
import tempfile
from itertools import cycle, repeat, chain, izip
from ctypes import *
from cStringIO import StringIO
import numpy as np
from scipy import ndimage

from fr0stlib import pyflam3
from fr0stlib.pyflam3._flam3 import *
from fr0stlib.pyflam3.constants import *

import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda
import pycuda.tools
from pycuda.gpuarray import vec

import cuburn.genome
from cuburn import affine
from cuburn.code import util, mwc, iter, filtering

class Renderer(object):
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

    cmp_options = ('-use_fast_math', '-maxrregcount', '42')
    keep = False

    def __init__(self, info):
        self.info = info
        self._iter = self._de = self.src = self.cubin = self.mod = None
        self.packed_genome = None

        # Ensure class options don't get contaminated on an instance
        self.cmp_options = list(self.cmp_options)

    def compile(self, keep=None, cmp_options=None):
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
        keep = self.keep if keep is None else keep
        cmp_options = self.cmp_options if cmp_options is None else cmp_options

        self._iter = iter.IterCode(self.info)
        self._de = filtering.DensityEst(self.info)
        cclip = filtering.ColorClip(self.info)
        self._iter.packer.finalize()
        self.src = util.assemble_code(util.BaseCode, mwc.MWC, self._iter.packer,
                                      self._iter, cclip, self._de)
        with open(os.path.join(tempfile.gettempdir(), 'kernel.cu'), 'w') as fp:
            fp.write(self.src)
        self.cubin = pycuda.compiler.compile(
                self.src, keep=keep, options=cmp_options,
                cache_dir=False if keep else None)
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


    def render(self, times):
        """
        Render a flame for each genome in the iterable value 'genomes'.
        Returns a Python generator object which will yield a 2-tuple of
        ``(time, buf)``, where ``time`` is the start time of the frame and
        ``buf`` is a 3D (width, height, channel) NumPy array containing
        [0,1]-valued RGBA components.

        This method produces a considerable amount of side effects, and should
        not be used lightly. Things may go poorly for you if this method is not
        allowed to run until completion (by exhausting all items in the
        generator object).

        ``times`` is a sequence of (start, stop) times defining the temporal
        range to be rendered for each frame. This will change to be more
        frame-centric in the future, allowing for interpolated temporal width.
        """
        if times == []:
            return

        info = self.info
        iter_stream = cuda.Stream()
        filt_stream = cuda.Stream()

        nbins = info.acc_height * info.acc_stride
        d_accum = cuda.mem_alloc(16 * nbins)
        d_out = cuda.mem_alloc(16 * nbins)

        num_sm = cuda.Context.get_device().multiprocessor_count
        cps_per_block = 1024

        genome_times, genome_knots = self._iter.packer.pack()
        d_genome_times = cuda.to_device(genome_times)
        d_genome_knots = cuda.to_device(genome_knots)
        info_size = 4 * len(self._iter.packer) * cps_per_block
        d_infos = cuda.mem_alloc(info_size)

        pals = info.genome.color.palette
        if isinstance(pals, basestring):
            pals = [0.0, pals, 1.0, pals]
        palint_times = np.empty(len(genome_times[0]), np.float32)
        palint_times.fill(100.0)
        palint_times[:len(pals)/2] = pals[::2]
        d_palint_times = cuda.to_device(palint_times)
        d_palint_vals = cuda.to_device(
                np.concatenate(map(info.db.palettes.get, pals[1::2])))
        d_palmem = cuda.mem_alloc(256 * info.palette_height * 4)

        # The '+1' avoids more situations where the 'smid' value is larger
        # than the number of enabled SMs on a chip, which is warned against in
        # the docs but not seen in the wild. Things could get nastier on
        # subsequent silicon, but I doubt they'd ever kill more than 1 SM
        nslots = pycuda.autoinit.device.max_threads_per_multiprocessor * \
                (pycuda.autoinit.device.multiprocessor_count + 1)

        d_points = cuda.mem_alloc(nslots * 16)
        seeds = mwc.MWC.make_seeds(nslots)
        d_seeds = cuda.to_device(seeds)

        h_out = cuda.pagelocked_empty((info.acc_height, info.acc_stride, 4),
                                      np.float32)

        filter_done_event = None

        packer_fun = self.mod.get_function("interp_iter_params")
        palette_fun = self.mod.get_function("interp_palette_hsv")
        iter_fun = self.mod.get_function("iter")
        #iter_fun.set_cache_config(cuda.func_cache.PREFER_L1)

        util.BaseCode.fill_dptr(self.mod, d_accum, 4 * nbins, filt_stream)

        last_time = times[0][0]

        for start, stop in times:
            cen_cp = cuburn.genome.HacketyGenome(info.genome, (start+stop)/2)

            if filter_done_event:
                iter_stream.wait_for_event(filter_done_event)

            width = np.float32((stop-start) / info.palette_height)
            palette_fun(d_palmem, d_palint_times, d_palint_vals,
                        np.float32(start), width,
                        block=(256,1,1), grid=(info.palette_height,1),
                        stream=iter_stream)

            tref = self.mod.get_texref('palTex')
            array_info = cuda.ArrayDescriptor()
            array_info.height = info.palette_height
            array_info.width = 256
            array_info.array_format = cuda.array_format.UNSIGNED_INT8
            array_info.num_channels = 4
            tref.set_address_2d(d_palmem, array_info, 1024)

            tref.set_format(cuda.array_format.UNSIGNED_INT8, 4)
            tref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
            tref.set_filter_mode(cuda.filter_mode.LINEAR)

            width = np.float32((stop-start) / cps_per_block)
            packer_fun(d_infos, d_genome_times, d_genome_knots,
                       np.float32(start), width, d_seeds,
                       block=(256,1,1), grid=(cps_per_block/256,1),
                       stream=iter_stream)

            # TODO: if we only do this once per anim, does quality improve?
            util.BaseCode.fill_dptr(self.mod, d_points, 4 * nslots,
                                    iter_stream, np.float32(np.nan))

            # Get interpolated control points for debugging
            #iter_stream.synchronize()
            #d_temp = cuda.from_device(d_infos,
                    #(cps_per_block, len(self._iter.packer)), np.float32)
            #for i, n in zip(d_temp[5], self._iter.packer.packed):
                #print '%60s %g' % ('_'.join(n), i)

            nsamps = info.density * info.width * info.height / cps_per_block
            iter_fun(np.uint64(d_accum), d_seeds, d_points,
                     d_infos, np.int32(nsamps),
                     block=(32, self._iter.NTHREADS/32, 1),
                     grid=(cps_per_block, 1),
                     texrefs=[tref], stream=iter_stream)

            iter_stream.synchronize()
            if filter_done_event:
                while not filt_stream.is_done():
                    timemod.sleep(0.01)
                filt_stream.synchronize()
                yield last_time, self._trim(h_out)
                last_time = start

            util.BaseCode.fill_dptr(self.mod, d_out, 4 * nbins, filt_stream)
            self._de.invoke(self.mod, cen_cp, d_accum, d_out, filt_stream)
            util.BaseCode.fill_dptr(self.mod, d_accum, 4 * nbins, filt_stream)
            filter_done_event = cuda.Event().record(filt_stream)

            f32 = np.float32
            # TODO: implement integration over cubic splines?
            gam = f32(1 / cen_cp.color.gamma)
            vib = f32(cen_cp.color.vibrancy)
            hipow = f32(cen_cp.color.highlight_power)
            lin = f32(cen_cp.color.gamma_threshold)
            lingam = f32(math.pow(lin, gam-1.0) if lin > 0 else 0)
            bkgd = vec.make_float3(
                    cen_cp.color.background.r,
                    cen_cp.color.background.g,
                    cen_cp.color.background.b)

            color_fun = self.mod.get_function("colorclip")
            blocks = int(np.ceil(np.sqrt(nbins / 256)))
            color_fun(d_out, gam, vib, hipow, lin, lingam, bkgd, np.int32(nbins),
                      block=(256, 1, 1), grid=(blocks, blocks),
                      stream=filt_stream)
            cuda.memcpy_dtoh_async(h_out, d_out, filt_stream)

        filt_stream.synchronize()
        yield start, self._trim(h_out)

    def _trim(self, result):
        g = self.info.gutter
        return result[g:-g,g:-g].copy()

