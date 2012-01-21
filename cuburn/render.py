import os
import sys
import re
import time as timemod
import tempfile
from collections import namedtuple
from itertools import cycle, repeat, chain, izip, imap, ifilter
from ctypes import *
from cStringIO import StringIO
import numpy as np
from numpy import float32 as f32, int32 as i32, uint32 as u32, uint64 as u64
from scipy import ndimage

import pycuda.compiler
import pycuda.driver as cuda
import pycuda.tools

import cuburn.genome
from cuburn import affine
from cuburn.code import util, mwc, iter, interp, filtering, sort

RenderedImage = namedtuple('RenderedImage', 'buf idx gpu_time')
Dimensions = namedtuple('Dimensions', 'w h aw ah astride')

def _sync_stream(dst, src):
    dst.wait_for_event(cuda.Event(cuda.event_flags.DISABLE_TIMING).record(src))

class Renderer(object):
    """
    Control structure for rendering a series of frames.
    """

    # Number of iterations to iterate without write after generating a new
    # point. This number is currently fixed pretty deeply in the set of magic
    # constants which govern buffer sizes; changing the value here won't
    # actually change the code on the device to do something different.
    fuse = 256

    # The palette texture/surface covers the color coordinate from [0,1] with
    # (for now, a fixed 256) equidistant horizontal samples, and spans the
    # temporal range of the frame linearly with this many rows. Increasing
    # this value increases the number of uniquely-dithered samples when using
    # pre-dithered surfaces.
    palette_height = 64

    # Palette color interpolation mode (see code.interp.Palette)
    palette_interp_mode = 'yuv'

    # Maximum width of DE and other spatial filters, and thus in turn the
    # amount of padding applied. Note that, for now, this must not be changed!
    # The filtering code makes deep assumptions about this value.
    gutter = 10

    # Accumulation mode. Leave it at 'atomic' for now.
    acc_mode = 'atomic'

    # TODO
    chaos_used = False

    cmp_options = ('-use_fast_math', '-maxrregcount', '42')
    keep = False

    def __init__(self):
        self._iter = self.pal = self.src = self.cubin = self.mod = None

        # Ensure class options don't get contaminated on an instance
        self.cmp_options = list(self.cmp_options)

    def compile(self, genome, keep=None, cmp_options=None):
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

        self._iter = iter.IterCode(self, genome)
        self._iter.packer.finalize()
        self.pal = interp.Palette(self.palette_interp_mode)
        self.src = util.assemble_code(util.BaseCode, mwc.MWC, self._iter.packer,
                                      self.pal, self._iter)
        with open(os.path.join(tempfile.gettempdir(), 'kernel.cu'), 'w') as fp:
            fp.write(self.src)
        self.cubin = pycuda.compiler.compile(
                self.src, keep=keep, options=cmp_options,
                cache_dir=False if keep else None)

    def load(self, genome, jit_options=[]):
        if not self.cubin:
            self.compile(genome)
        self.mod = cuda.module_from_buffer(self.cubin, jit_options)
        with open('/tmp/iter_kern.cubin', 'wb') as fp:
            fp.write(self.cubin)
        return self.src

    def render(self, genome, times, width, height, blend=True):
        """
        Render a frame for each timestamp in the iterable value ``times``. This
        function returns a generator that will yield a RenderedImage object
        containing a shared reference to the output buffer for each specified
        frame.

        The returned buffer is page-locked host memory. Between the time a
        buffer is yielded and the time the next frame's results are requested,
        the buffer will not be modified. Thereafter, however, it will be
        overwritten by an asynchronous DMA operation coming from the CUDA
        device. If you hang on to it for longer than one frame, copy it.

        ``genome`` is the genome to be rendered. Successive calls to the
        `render()` method on one ``Renderer`` object must use genomes which
        produce identical compiled code, and this will not be verified by the
        renderer. In practice, this means you can alter genome parameter
        values, but the full set of keys must remain identical between runs on
        the same renderer.

        ``times`` is a list of (idx, cen_time) tuples, where ``idx`` is passed
        unmodified in the RenderedImage return value and ``cen_time`` is the
        central time of the current frame in spline-time units. (Any
        clock-time or frame-time units in the genome should be preconverted.)

        If ``blend`` is False, the output buffer will contain unclipped,
        premultiplied RGBA data, without vibrancy, highlight power, or the
        alpha elbow applied.
        """
        r = self.render_gen(genome, width, height, blend=blend)
        next(r)
        return ifilter(None, imap(r.send, chain(times, [None])))

    def render_gen(self, genome, width, height, blend=True):
        """
        Render frames. This method is wrapped by the ``render()`` method; see
        its docstring for warnings and details.

        Instead of passing frame times as an iterable, they are passed
        individually via the ``generator.send()`` method. There is an
        internal pipeline latency of one frame, so the first call to the
        ``send()`` method will return None, the second call will return the
        first frame's result, and so on. To retrieve the last frame in a
        sequence, send ``None``.

        Direct use of this method is useful for implementing render servers.
        """

        last_idx = None
        next_frame = yield
        if next_frame is None:
            return

        if not self.mod:
            self.load(genome)

        filt = filtering.Filtering()

        reset_rb_fun = self.mod.get_function("reset_rb")
        packer_fun = self.mod.get_function("interp_iter_params")
        iter_fun = self.mod.get_function("iter")

        # The synchronization model is messy. See helpers/task_model.svg.
        iter_stream = cuda.Stream()
        filt_stream = cuda.Stream()
        if self.acc_mode == 'deferred':
            write_stream = cuda.Stream()
            write_fun = self.mod.get_function("write_shmem")
        else:
            write_stream = iter_stream

        # These events fire when the corresponding buffer is available for
        # reading on the host (i.e. the copy is done). On the first pass, 'a'
        # will be ignored, and subsequently moved to 'b'.
        event_a = cuda.Event().record(filt_stream)
        event_b = None

        awidth = width + 2 * self.gutter
        aheight = 32 * int(np.ceil((height + 2 * self.gutter) / 32.))
        astride = 32 * int(np.ceil(awidth / 32.))
        dim = Dimensions(width, height, awidth, aheight, astride)
        d_acc_size = self.mod.get_global('acc_size')[0]
        cuda.memcpy_htod_async(d_acc_size, u32(list(dim)), write_stream)

        nbins = astride * aheight
        # Extra padding in accum helps with write_shmem overruns
        d_accum = cuda.mem_alloc(16 * nbins + (1<<16))
        d_out = cuda.mem_alloc(16 * aheight * astride)
        if self.acc_mode == 'atomic':
            d_atom = cuda.mem_alloc(8 * nbins)
            flush_fun = self.mod.get_function("flush_atom")

        obuf_copy = util.argset(cuda.Memcpy2D(),
            src_y=self.gutter, src_x_in_bytes=16*self.gutter,
            src_pitch=16*astride, dst_pitch=16*width,
            width_in_bytes=16*width, height=height)
        obuf_copy.set_src_device(d_out)
        h_out_a = cuda.pagelocked_empty((height, width, 4), f32)
        h_out_b = cuda.pagelocked_empty((height, width, 4), f32)

        if self.acc_mode == 'deferred':
            # Having a fixed, power-of-two log size makes things much easier
            log_size = 64 << 20
            d_log = cuda.mem_alloc(log_size * 4)
            d_log_sorted = cuda.mem_alloc(log_size * 4)
            sorter = sort.Sorter(log_size)
            # We need to cover each unique tag - address bits 20-23 - with one
            # write block per sort bin. Or somethinig like that.
            nwriteblocks = int(np.ceil(nbins / float(1<<20))) * 256

        # Calculate 'nslots', the number of simultaneous running threads that
        # can be active on the GPU during iteration (and thus the number of
        # slots for loading and storing RNG and point context that will be
        # prepared on the device), and derive 'rb_size', the number of blocks in
        # 'nslots'.
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
        # This statement may add extra seeds to simplify palette dithering.
        seeds = mwc.MWC.make_seeds(max(nslots, 256 * self.palette_height))
        d_seeds = cuda.to_device(seeds)

        # We used to auto-calculate this to a multiple of the number of SMs on
        # the device, but since we now use shorter launches and, to a certain
        # extent, allow simultaneous occupancy, that's not as important. The
        # 1024 is a magic constant to ensure reasonable and power-of-two log
        # size for deferred: 256MB / (4B * FUSE * NTHREADS). Enhancements to
        # the sort engine are needed to make this more flexible.
        ntemporal_samples = 1024
        genome_times, genome_knots = self._iter.packer.pack()
        d_genome_times = cuda.to_device(genome_times)
        d_genome_knots = cuda.to_device(genome_knots)
        info_size = 4 * len(self._iter.packer) * ntemporal_samples
        d_infos = cuda.mem_alloc(info_size)

        ptimes, pidxs = zip(*genome.palette_times)
        palint_times = np.empty(len(genome_times[0]), f32)
        palint_times.fill(1e10)
        palint_times[:len(ptimes)] = ptimes
        d_palint_times = cuda.to_device(palint_times)
        pvals = self.pal.prepare([genome.decoded_palettes[i] for i in pidxs])
        d_palint_vals = cuda.to_device(np.concatenate(pvals))

        if self.acc_mode in ('deferred', 'atomic'):
            palette_fun = self.mod.get_function("interp_palette_flat")
            dsc = util.argset(cuda.ArrayDescriptor3D(),
                    height=self.palette_height, width=256, depth=0,
                    format=cuda.array_format.SIGNED_INT32,
                    num_channels=2, flags=cuda.array3d_flags.SURFACE_LDST)
            palarray = cuda.Array(dsc)

            tref = self.mod.get_surfref('flatpal')
            tref.set_array(palarray, 0)
        else:
            palette_fun = self.mod.get_function("interp_palette")
            dsc = util.argset(cuda.ArrayDescriptor(),
                    height=self.palette_height, width=256,
                    format=cuda.array_format.UNSIGNED_INT8,
                    num_channels=4)
            d_palmem = cuda.mem_alloc(256 * self.palette_height * 4)

            tref = self.mod.get_texref('palTex')
            tref.set_address_2d(d_palmem, dsc, 1024)
            tref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
            tref.set_filter_mode(cuda.filter_mode.LINEAR)

        while next_frame is not None:
            # tc, td, ts, te: central, delta, start, end times
            idx, tc = next_frame
            td = genome.adj_frame_width(tc)
            ts, te = tc - 0.5 * td, tc + 0.5 * td

            if self.acc_mode in ('deferred', 'atomic'):
                # In this mode, the palette writes to a surface reference, but
                # requires dithering, so we pass it the seeds instead
                arg0 = d_seeds
            else:
                arg0 = d_palmem
            palette_fun(arg0, d_palint_times, d_palint_vals,
                        f32(ts), f32(td / self.palette_height),
                        block=(256,1,1), grid=(self.palette_height,1),
                        stream=write_stream)

            packer_fun(d_infos, d_genome_times, d_genome_knots,
                       f32(ts), f32(td / ntemporal_samples),
                       i32(ntemporal_samples), block=(256,1,1),
                       grid=(int(np.ceil(ntemporal_samples/256.)),1),
                       stream=iter_stream)

            # Reset points so that they will be FUSEd
            util.BaseCode.fill_dptr(self.mod, d_points, 4 * nslots,
                                    iter_stream, f32(np.nan))

            # Get interpolated control points for debugging
            #iter_stream.synchronize()
            #d_temp = cuda.from_device(d_infos,
                    #(ntemporal_samples, len(self._iter.packer)), f32)
            #for i, n in zip(d_temp[5], self._iter.packer.packed):
                #print '%60s %g' % ('_'.join(n), i)

            util.BaseCode.fill_dptr(self.mod, d_accum, 4 * nbins, write_stream)
            if self.acc_mode == 'atomic':
                util.BaseCode.fill_dptr(self.mod, d_atom, 2 * nbins, write_stream)
            nrounds = int( (genome.spp(tc) * width * height)
                         / (ntemporal_samples * 256 * 256) ) + 1
            if self.acc_mode == 'deferred':
                for i in range(nrounds):
                    iter_fun(np.uint64(d_log), d_seeds, d_points, d_infos,
                             block=(32, self._iter.NTHREADS/32, 1),
                             grid=(ntemporal_samples, 1), stream=iter_stream)
                    _sync_stream(write_stream, iter_stream)
                    sorter.sort(d_log_sorted, d_log, log_size, 3, True,
                                stream=write_stream)
                    _sync_stream(iter_stream, write_stream)
                    write_fun(d_accum, d_log_sorted, sorter.dglobal, i32(nbins),
                              block=(1024, 1, 1), grid=(nwriteblocks, 1),
                              stream=write_stream)
            else:
                args = [u64(d_accum), d_seeds, d_points, d_infos]
                if self.acc_mode == 'atomic':
                    args.append(u64(d_atom))
                iter_fun(*args, block=(32, self._iter.NTHREADS/32, 1),
                         grid=(ntemporal_samples, nrounds), stream=iter_stream)
                if self.acc_mode == 'atomic':
                    nblocks = int(np.ceil(np.sqrt(nbins/float(512))))
                    flush_fun(u64(d_accum), u64(d_atom), i32(nbins),
                              block=(512, 1, 1), grid=(nblocks, nblocks),
                              stream=iter_stream)

            util.BaseCode.fill_dptr(self.mod, d_out, 4 * nbins, filt_stream)
            _sync_stream(filt_stream, write_stream)
            filt.de(d_out, d_accum, genome, dim, tc, stream=filt_stream)
            _sync_stream(write_stream, filt_stream)
            filt.colorclip(d_out, genome, dim, tc, blend, stream=filt_stream)
            obuf_copy.set_dst_host(h_out_a)
            obuf_copy(filt_stream)

            if event_b:
                while not event_a.query():
                    timemod.sleep(0.01)
                gpu_time = event_a.time_since(event_b)
                result = RenderedImage(h_out_b, last_idx, gpu_time)
            else:
                result = None
            last_idx = idx

            event_a, event_b = cuda.Event().record(filt_stream), event_a
            h_out_a, h_out_b = h_out_b, h_out_a

            # TODO: add ability to flush a frame without breaking the pipe
            next_frame = yield result

        while not event_a.query():
            timemod.sleep(0.001)
        gpu_time = event_a.time_since(event_b)
        yield RenderedImage(h_out_b, last_idx, gpu_time)

