import os
import sys
import re
import time as timemod
import tempfile
from collections import namedtuple
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
import pycuda.tools

import cuburn.genome
from cuburn import affine
from cuburn.code import util, mwc, iter, filtering, sort

RenderedImage = namedtuple('RenderedImage', 'buf idx gpu_time')

def _sync_stream(dst, src):
    dst.wait_for_event(cuda.Event(cuda.event_flags.DISABLE_TIMING).record(src))

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
        self._iter = self.src = self.cubin = self.mod = None
        self.packed_genome = None

        # Ensure class options don't get contaminated on an instance
        self.cmp_options = list(self.cmp_options)

    def compile(self, keep=None, cmp_options=None, jit_options=[]):
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
        self._iter.packer.finalize()
        self.src = util.assemble_code(util.BaseCode, mwc.MWC, self._iter.packer,
                                      self._iter)
        with open(os.path.join(tempfile.gettempdir(), 'kernel.cu'), 'w') as fp:
            fp.write(self.src)
        self.cubin = pycuda.compiler.compile(
                self.src, keep=keep, options=cmp_options,
                cache_dir=False if keep else None)
        self.mod = cuda.module_from_buffer(self.cubin, jit_options)
        with open('/tmp/iter_kern.cubin', 'wb') as fp:
            fp.write(self.cubin)
        return self.src

    def render(self, times):
        """
        Render a flame for each genome in the iterable value 'genomes'.
        Returns a RenderedImage object with the rendered buffer in the
        requested format (3D RGBA ndarray only for now).

        This method produces a considerable amount of side effects, and should
        not be used lightly. Things may go poorly for you if this method is not
        allowed to run until completion (by exhausting all items in the
        generator object).

        ``times`` is a sequence of (idx, start, stop) times, where index is
        the logical frame number (though it can be any value) and 'start' and
        'stop' together define the time range to be rendered for each frame.
        """
        if times == []:
            return

        filt = filtering.Filtering()

        reset_rb_fun = self.mod.get_function("reset_rb")
        packer_fun = self.mod.get_function("interp_iter_params")
        palette_fun = self.mod.get_function("interp_palette_hsv")
        iter_fun = self.mod.get_function("iter")
        write_fun = self.mod.get_function("write_shmem")

        info = self.info

        # The synchronization model is messy. See helpers/task_model.svg.
        iter_stream = cuda.Stream()
        filt_stream = cuda.Stream()
        if info.acc_mode == 'deferred':
            write_stream = cuda.Stream()
        else:
            write_stream = iter_stream

        # These events fire when the corresponding buffer is available for
        # reading on the host (i.e. the copy is done). On the first pass, 'a'
        # will be ignored, and subsequently moved to 'b'.
        event_a = cuda.Event().record(filt_stream)
        event_b = None

        nbins = info.acc_height * info.acc_stride
        d_accum = cuda.mem_alloc(16 * nbins)
        d_out = cuda.mem_alloc(16 * nbins)

        acc_size = np.array([info.acc_width, info.acc_height, info.acc_stride])
        d_acc_size = self.mod.get_global('acc_size')[0]
        cuda.memcpy_htod_async(d_acc_size, np.uint32(acc_size), write_stream)

        if info.acc_mode == 'deferred':
            # Having a fixed, power-of-two log size makes things much easier
            log_size = 64 << 20
            d_log = cuda.mem_alloc(log_size * 4)
            d_log_sorted = cuda.mem_alloc(log_size * 4)
            sorter = sort.Sorter(log_size)

            # Shared accumulators take care of the lowest 12 bits, but due to
            # a quirk of the sort implementation, asking the sort to handle
            # fewer bits than it is compiled for will make it considerably
            # slower (and it can't be compiled for <7b), so we actually dig in
            # to the accumulator's SHAB window for those cases.
            SHAB = np.int32(12)
            address_bits = np.int32(np.ceil(np.log2(nbins+1)))
            start_bit = address_bits - sorter.radix_bits
            log_shift = np.int32(SHAB - start_bit)
            nwriteblocks = int(np.ceil(nbins / (1<<SHAB)))
            print start_bit, log_shift, nwriteblocks

        # Calculate 'nslots', the number of simultaneous running threads that
        # can be active on the GPU during iteration (and thus the number of
        # slots for loading and storing RNG and point context that will be
        # prepared on the device), 'rb_size' (the number of blocks in
        # 'nslots'), and determine a number of temporal samples
        # likely to load-balance effectively
        iter_threads_per_block = 256
        dev_data = pycuda.tools.DeviceData()
        occupancy = pycuda.tools.OccupancyRecord(
                dev_data, iter_threads_per_block,
                iter_fun.shared_size_bytes, iter_fun.num_regs)
        nsms = cuda.Context.get_device().multiprocessor_count
        rb_size = occupancy.warps_per_mp * nsms / (iter_threads_per_block / 32)
        nslots = iter_threads_per_block * rb_size

        # Reset the ringbuffer info for the slots
        reset_rb_fun(np.int32(rb_size), block=(1,1,1))

        d_points = cuda.mem_alloc(nslots * 16)
        seeds = mwc.MWC.make_seeds(nslots)
        d_seeds = cuda.to_device(seeds)

        # We used to auto-calculate this to a multiple of the number of SMs on
        # the device, but since we now use shorter launches and, to a certain
        # extent, allow simultaneous occupancy, that's not as important. The
        # 1024 is a magic constant, though: FUSE
        ntemporal_samples = 1024
        genome_times, genome_knots = self._iter.packer.pack()
        d_genome_times = cuda.to_device(genome_times)
        d_genome_knots = cuda.to_device(genome_knots)
        info_size = 4 * len(self._iter.packer) * ntemporal_samples
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

        pal_array_info = cuda.ArrayDescriptor()
        pal_array_info.height = info.palette_height
        pal_array_info.width = 256
        pal_array_info.array_format = cuda.array_format.UNSIGNED_INT8
        pal_array_info.num_channels = 4

        h_out_a = cuda.pagelocked_empty((info.acc_height, info.acc_stride, 4),
                                        np.float32)
        h_out_b = cuda.pagelocked_empty((info.acc_height, info.acc_stride, 4),
                                        np.float32)
        last_idx = None

        for idx, start, stop in times:
            width = np.float32((stop-start) / info.palette_height)
            palette_fun(d_palmem, d_palint_times, d_palint_vals,
                        np.float32(start), width,
                        block=(256,1,1), grid=(info.palette_height,1),
                        stream=write_stream)

            # TODO: do we need to do this each time in order to reset cache?
            tref = self.mod.get_texref('palTex')
            tref.set_address_2d(d_palmem, pal_array_info, 1024)
            tref.set_format(cuda.array_format.UNSIGNED_INT8, 4)
            tref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
            tref.set_filter_mode(cuda.filter_mode.LINEAR)

            width = np.float32((stop-start) / ntemporal_samples)
            packer_fun(d_infos, d_genome_times, d_genome_knots,
                       np.float32(start), width, d_seeds,
                       np.int32(ntemporal_samples), block=(256,1,1),
                       grid=(int(np.ceil(ntemporal_samples/256.)),1),
                       stream=iter_stream)

            # Reset points so that they will be FUSEd
            util.BaseCode.fill_dptr(self.mod, d_points, 4 * nslots,
                                    iter_stream, np.float32(np.nan))

            # Get interpolated control points for debugging
            #stream.synchronize()
            #d_temp = cuda.from_device(d_infos,
                    #(ntemporal_samples, len(self._iter.packer)), np.float32)
            #for i, n in zip(d_temp[5], self._iter.packer.packed):
                #print '%60s %g' % ('_'.join(n), i)

            util.BaseCode.fill_dptr(self.mod, d_accum, 4 * nbins, write_stream)
            nrounds = ( (info.density * info.width * info.height)
                      / (ntemporal_samples * 256 * 256) ) + 1
            if info.acc_mode == 'deferred':
                for i in range(nrounds):
                    iter_fun(np.uint64(d_log), d_seeds, d_points, d_infos,
                             block=(32, self._iter.NTHREADS/32, 1),
                             grid=(ntemporal_samples, 1), stream=iter_stream)
                    _sync_stream(write_stream, iter_stream)
                    sorter.sort(d_log_sorted, d_log, log_size, start_bit, True,
                                stream=write_stream)
                    _sync_stream(iter_stream, write_stream)
                    write_fun(d_accum, d_log_sorted, sorter.dglobal, log_shift,
                              block=(1024, 1, 1), grid=(nwriteblocks, 1),
                              texrefs=[tref], stream=write_stream)
            else:
                iter_fun(np.uint64(d_accum), d_seeds, d_points, d_infos,
                         block=(32, self._iter.NTHREADS/32, 1),
                         grid=(ntemporal_samples, nrounds),
                         texrefs=[tref], stream=iter_stream)

            util.BaseCode.fill_dptr(self.mod, d_out, 4 * nbins, filt_stream)
            _sync_stream(filt_stream, write_stream)
            filt.de(d_out, d_accum, info, start, stop, filt_stream)
            _sync_stream(write_stream, filt_stream)
            filt.colorclip(d_out, info, start, stop, filt_stream)
            cuda.memcpy_dtoh_async(h_out_a, d_out, filt_stream)

            if event_b:
                while not event_a.query():
                    timemod.sleep(0.01)
                gpu_time = event_a.time_since(event_b)
                yield RenderedImage(self._trim(h_out_b), last_idx, gpu_time)

            event_a, event_b = cuda.Event().record(filt_stream), event_a
            h_out_a, h_out_b = h_out_b, h_out_a
            last_idx = idx

        while not event_a.query():
            timemod.sleep(0.001)
        gpu_time = event_a.time_since(event_b)
        yield RenderedImage(self._trim(h_out_b), last_idx, gpu_time)

    def _trim(self, result):
        g = self.info.gutter
        return result[g:-g,g:-g].copy()

