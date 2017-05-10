"""
Resources and tools to perform rendering.
"""

import os
import sys
import re
import time
import tempfile
from collections import namedtuple
import numpy as np
from numpy import float32 as f32, int32 as i32, uint32 as u32, uint64 as u64

import pycuda.driver as cuda
import pycuda.tools

import filters
import output
from code import util, mwc, iter, interp, sort
from code.util import ClsMod, devlib, filldptrlib, assemble_code, launch
from cuburn.genome.util import palette_decode

RenderedImage = namedtuple('RenderedImage', 'buf idx gpu_time')
Dimensions = namedtuple('Dimensions', 'w h aw ah astride')

class DurationEvent(cuda.Event):
    """
    A CUDA event which is implicitly aware of a prior event for time
    calculations.

    Note that instances retain a reference to their prior, so an unbroken
    chain of DurationEvents will leak. Use normal events as priors.
    """
    def __init__(self, prior):
        super(DurationEvent, self).__init__()
        self._prior = prior
    def time(self):
        return self.time_since(self._prior)

class Framebuffers(object):
    """
    The largest memory allocations, and a stream to serialize their use.

    ``d_front`` and ``d_back`` are separate buffers, each large enough to hold
    four float32 components per pixel (including any gutter pixels added for
    alignment or padding).

    Every user of this set of buffers may use and overwrite the buffers in
    any way, as long as the output for the next stage winds up in the front
    buffer. The front and back buffers can be exchanged by the ``flip()``
    method (which simply exchanges the pointers); while no similar method
    exists for the side buffer, you're free to do the same by taking local
    copies of the references and exchanging them yourself.

    ``d_left`` and ``d_right`` and ``d_uleft`` and ``d_uright`` are similar,
    but without strict dependencies. Each stage is free to stomp these buffers,
    but must be done with them by the next stage.

    There's one spot in the stream interleaving where the behavior is
    different: the ``Output.convert`` call must store its output to the back
    buffer, which will remain untouched until the dtoh copy of the converted
    buffer is finished. This happens while the ``iter`` kernel of the next
    frame writes to the front and side buffers, which means in practice that
    there's essentially no waiting on any buffers.

    If an output module decides to get all krazy and actually replaces the
    references to the buffers on this object - to, say, implement a temporally
    aware tone-mapping or denoising pass - that's probably okay, but just make
    sure it appears like it's expected to.
    """

    # Minimum extension of accumulation buffer beyond output region. Used to
    # alleviate edge effects during filtering. Actual gutter may be larger to
    # accomodate alignment requirements; when it is, that extension will be
    # applied to the lower-right corner of the buffer. This is asymmetrical,
    # but simplifies trimming logic when it's time for that.
    gutter = 12

    @classmethod
    def calc_dim(cls, width, height):
        """
        Given a width and height, return a valid set of dimensions which
        include at least enough gutter to exceed the minimum, and where
        (acc_width % 32) == 0 and (acc_height % 16) == 0.
        """
        awidth = width + 2 * cls.gutter
        aheight = 16 * int(np.ceil((height + 2 * cls.gutter) / 16.))
        astride = 32 * int(np.ceil(awidth / 32.))
        return Dimensions(width, height, awidth, aheight, astride)

    def __init__(self):
        self.stream = cuda.Stream()
        self.pool = pycuda.tools.PageLockedMemoryPool()
        self._clear()

        # These resources rely on the slots/ringbuffer mechanism for sharing,
        # and so can be shared across any number of launches, genomes, and
        # render kernels. Notably, seeds are self-synchronizing, so they're not
        # attached to either stream object.
        self.d_rb = cuda.to_device(np.array([0, 0], dtype=u32))
        seeds = mwc.make_seeds(util.DEFAULT_RB_SIZE * 256)
        self.d_seeds = cuda.to_device(seeds)
        self._len_d_points = util.DEFAULT_RB_SIZE * 256 * 16
        self.d_points = cuda.mem_alloc(self._len_d_points)

    def _clear(self):
        self.nbins = self.d_front = self.d_back = None
        self.d_left = self.d_right = self.d_uleft = self.d_uright = None

    def free(self, stream=None):
        if stream is not None:
            stream.synchronize()
        else:
            cuda.Context.synchronize()
        for p in (self.d_front, self.d_back, self.d_left, self.d_right,
                  self.d_uleft, self.d_uright):
            if p is not None:
                p.free()
        self._clear()

    def alloc(self, dim, stream=None):
        """
        Ensure that this object's framebuffers are large enough to handle the
        given dimensions, allocating new ones if not.

        If ``stream`` is not None and a reallocation is necessary, the stream
        will be synchronized before the old buffers are deallocated.
        """
        nbins = dim.ah * dim.astride
        if self.nbins >= nbins: return
        if self.nbins is not None: self.free()
        try:
            self.d_front  = cuda.mem_alloc(16 * nbins)
            self.d_back   = cuda.mem_alloc(16 * nbins)
            self.d_left   = cuda.mem_alloc(16 * nbins)
            self.d_right  = cuda.mem_alloc(16 * nbins)
            self.d_uleft  = cuda.mem_alloc(2  * nbins)
            self.d_uright = cuda.mem_alloc(2  * nbins)
            self.nbins = nbins
        except cuda.MemoryError, e:
            # If a frame that's too large sneaks by the task distributor, we
            # don't want to kill the server, but we also don't want to leave
            # it stuck without any free memory to complete the next alloc.
            # TODO: measure free mem and only take tasks that fit (but that
            # should be done elsewhere)
            self.free(stream)
            raise e

    def set_dim(self, width, height, stream=None):
        """
        Compute padded dimensions for given width and height, ensure that the
        buffers are large enough (and reallocate if not), and return the
        calculated dimensions.

        Note that the returned dimensions are always the same for a given
        width, height, and minimum gutter, even if the underlying buffers are
        larger due to a previous allocation.
        """
        dim = self.calc_dim(width, height)
        self.alloc(dim, stream)
        return dim

    def flip(self):
        """Flip the front and back buffers."""
        self.d_front, self.d_back = self.d_back, self.d_front

    def flip_side(self):
        """Flip the left and right buffers (float and uchar)."""
        self.d_left, self.d_right = self.d_right, self.d_left
        self.d_uleft, self.d_uright = self.d_uright, self.d_uleft

class DevSrc(object):
    """
    The buffers which represent a genome on-device, in the formats needed to
    serve as a source for interpolating temporal samples.
    """

    # Maximum number of knots per parameter. This also covers the maximum
    # number of palettes allowed.
    max_knots = 1 << util.DEFAULT_SEARCH_ROUNDS

    # Maximum number of parameters per genome. This number is exceedingly
    # high, and should never even come close to being hit.
    max_params = 1024

    def __init__(self):
        self.d_times = cuda.mem_alloc(4 * self.max_knots * self.max_params)
        self.d_knots = cuda.mem_alloc(4 * self.max_knots * self.max_params)
        self.d_ptimes = cuda.mem_alloc(4 * self.max_knots)
        self.d_pals = cuda.mem_alloc(4 * 4 * 256 * self.max_knots)

class DevInfo(object):
    """
    The buffers which hold temporal samples on-device, as used by iter.
    """

    # The palette texture/surface covers the color coordinate from [0,1] with
    # equidistant horizontal samples, and spans the temporal range of the
    # frame linearly with this many rows. Increasing these values increases the
    # number of uniquely-dithered samples when using pre-dithered surfaces, as
    # is done in 'atomic' accumulation.
    palette_width = 256 # TODO: make this setting be respected
    palette_height = 64

    # This used to be determined automagically, but simultaneous occupancy
    # and a much smaller block size simplifies this.
    ntemporal_samples = 1024

    # Number of iterations to iterate without write after generating a new
    # point. This number is currently fixed pretty deeply in the set of magic
    # constants which govern buffer sizes; changing the value here won't
    # actually change the code on the device to do something different.
    # It's here just for documentation purposes.
    fuse = 256

    def __init__(self):
        self.d_params = cuda.mem_alloc(
                self.ntemporal_samples * DevSrc.max_params * 4)
        self.palette_surf_dsc = util.argset(cuda.ArrayDescriptor3D(),
                height=self.palette_height, width=self.palette_width, depth=0,
                format=cuda.array_format.SIGNED_INT32,
                num_channels=2, flags=cuda.array3d_flags.SURFACE_LDST)
        self.d_pal_array = cuda.Array(self.palette_surf_dsc)

class Renderer(object):
    # Unloading a module triggers a context sync. To keep the renderer
    # asynchronous, and avoid expensive CPU polling, this hangs on to
    # a number of (relatively small) CUDA modules and flushes them together.
    MAX_MODREFS = 20
    _modrefs = {}

    @classmethod
    def compile(cls, gnm, arch=None, keep=False):
        packer, lib = iter.mkiterlib(gnm)
        cubin = util.compile('iter', assemble_code(lib), arch=arch, keep=keep)
        return packer, lib, cubin

    def load(self, cubin):
        if cubin in self._modrefs:
            return self._modrefs[cubin]
        mod = cuda.module_from_buffer(self.cubin)
        if len(self._modrefs) > self.MAX_MODREFS:
            self._modrefs.clear()
        self._modrefs[cubin] = mod
        return mod

    def __init__(self, gnm, gprof, keep=False, arch=None):
        self.packer, self.lib, self.cubin = self.compile(gnm, keep=keep, arch=arch)
        self.mod = self.load(self.cubin)
        self.filts = filters.create(gprof)
        self.out = output.get_output_for_profile(gprof)

class RenderManager(ClsMod):
    lib = devlib(deps=[interp.palintlib, filldptrlib])

    def __init__(self):
        super(RenderManager, self).__init__()
        self.fb = Framebuffers()
        self.src_a, self.src_b = DevSrc(), DevSrc()
        self.info_a, self.info_b = DevInfo(), DevInfo()
        self.stream_a, self.stream_b = cuda.Stream(), cuda.Stream()
        self.filt_evt = self.copy_evt = None

    def _copy(self, rdr, gnm):
        """
        Queue a copy of a host genome into a set of device interpolation source
        buffers.

        Note that for now, this is broken! It ignores ``gnm``, and only packs
        the genome that was used when creating the renderer.
        """
        times, knots = rdr.packer.pack(gnm, self.fb.pool)
        cuda.memcpy_htod_async(self.src_a.d_times, times, self.stream_a)
        cuda.memcpy_htod_async(self.src_a.d_knots, knots, self.stream_a)

        palsrc = dict([(v[0], palette_decode(v[1:])) for v in gnm['palette']])
        ptimes, pvals = zip(*sorted(palsrc.items()))
        palettes = self.fb.pool.allocate((len(palsrc), 256, 4), f32)
        palettes[:] = pvals
        palette_times = self.fb.pool.allocate((self.src_a.max_knots,), f32)
        palette_times.fill(1e9)
        palette_times[:len(ptimes)] = ptimes
        cuda.memcpy_htod_async(self.src_a.d_pals, palettes, self.stream_a)
        cuda.memcpy_htod_async(self.src_a.d_ptimes, palette_times,
                               self.stream_a)

        # TODO: use bilerp tex as src for palette interp

    def _interp(self, rdr, gnm, dim, ts, td):
        d_acc_size = rdr.mod.get_global('acc_size')[0]
        p_dim = self.fb.pool.allocate((len(dim),), u32)
        p_dim[:] = dim
        cuda.memcpy_htod_async(d_acc_size, p_dim, self.stream_a)

        tref = self.mod.get_surfref('flatpal')
        tref.set_array(self.info_a.d_pal_array, 0)
        launch('interp_palette_flat', self.mod, self.stream_a,
                256, self.info_a.palette_height,
                self.fb.d_rb, self.fb.d_seeds,
                self.src_a.d_ptimes, self.src_a.d_pals,
                f32(ts), f32(td / self.info_a.palette_height))

        nts = self.info_a.ntemporal_samples
        launch('interp_iter_params', rdr.mod, self.stream_a,
                256, np.ceil(nts / 256.),
                self.info_a.d_params, self.src_a.d_times, self.src_a.d_knots,
                f32(ts), f32(td / nts), i32(nts))
        #self._print_interp_knots(rdr)

    def _print_interp_knots(self, rdr, tsidx=5):
        infos = cuda.from_device(self.info_a.d_params,
                (tsidx + 1, len(rdr.packer)), f32)
        for i, n in zip(infos[-1], rdr.packer.packed):
            print '%60s %g' % ('_'.join(n), i)

    def _iter(self, rdr, gnm, gprof, dim, tc):
        tref = rdr.mod.get_surfref('flatpal')
        tref.set_array(self.info_a.d_pal_array, 0)

        nbins = dim.ah * dim.astride
        fill = lambda b, s, v=i32(0): util.fill_dptr(
                self.mod, b, s, stream=self.stream_a, value=v)
        fill(self.fb.d_front,  4 * nbins)
        fill(self.fb.d_left,   4 * nbins)
        fill(self.fb.d_right,  4 * nbins)
        fill(self.fb.d_points, self.fb._len_d_points / 4, f32(np.nan))
        fill(self.fb.d_uleft,  nbins / 2)
        fill(self.fb.d_uright, nbins / 2)

        nts = self.info_a.ntemporal_samples
        nsamps = (gprof.spp(tc) * dim.w * dim.h)
        nrounds = int(nsamps / (nts * 256. * 256)) + 1

        # Split the launch into multiple rounds, to prevent a system on older
        # GPUs from locking up and to give us a chance to flush some stuff.
        hidden_stream = cuda.Stream()
        iter_stream_left, iter_stream_right = self.stream_a, hidden_stream
        BLOCK_SIZE = 4

        while nrounds:
          n = min(nrounds, BLOCK_SIZE)
          launch('iter', rdr.mod, iter_stream_left, (32, 8, 1), (nts, n),
                 self.fb.d_front, self.fb.d_left,
                 self.fb.d_rb, self.fb.d_seeds, self.fb.d_points,
                 self.fb.d_uleft, self.info_a.d_params)

          # Make sure the other stream is done flushing before we start
          iter_stream_left.wait_for_event(cuda.Event().record(iter_stream_right))

          launch('flush_atom', rdr.mod, iter_stream_left,
                  (16, 16, 1), (dim.astride / 16, dim.ah / 16),
                  u64(self.fb.d_front), u64(self.fb.d_left),
                  u64(self.fb.d_uleft), i32(nbins))

          self.fb.flip_side()
          iter_stream_left, iter_stream_right = iter_stream_right, iter_stream_left
          nrounds -= n

        # Always wait on all events in the hidden stream before continuing on A
        self.stream_a.wait_for_event(cuda.Event().record(hidden_stream))

    def queue_frame(self, rdr, gnm, gprof, tc, copy=True):
        """
        Queue one frame for rendering.

        ``rdr`` is a compiled Renderer module. Caller must ensure that the
        module is compatible with the genome data provided.

        ``gnm`` is the genome to be rendered.

        ``tc`` is the center time at which to render.

        ``w``, ``h`` are the width and height of the desired output in px.

        If ``copy`` is False, the genome data will not be recopied for each
        new genome. This function must be called with ``copy=True`` the first
        time a new genome is used, and may be called in that manner
        subsequently without harm. I suspect the performance impact is low, so
        leave ``copy`` to True every time for now.

        The return value is a 2-tuple ``(evt, h_out)``, where ``evt`` is a
        DurationEvent and ``h_out`` is the return value of the output module's
        ``copy`` function. In the typical case, ``h_out`` will be a host
        allocation containing data in an appropriate format for the output
        module's file writer, and ``evt`` indicates when the asynchronous
        DMA copy which will populate ``h_out`` is complete. This can vary
        depending on the output module in use, though.

        This method is absolutely _not_ threadsafe, but it's fine to use it
        alongside non-threaded approaches to concurrency like coroutines.
        """
        timing_event = cuda.Event().record(self.stream_b)
        # Note: we synchronize on the previous stream if buffers need to be
        # reallocated, which implicitly also syncs the current stream.
        dim = self.fb.set_dim(gprof.width, gprof.height, self.stream_b)

        # TODO: calculate this externally somewhere?
        td = gprof.frame_width(tc) / round(gprof.fps * gprof.duration)
        ts, te = tc - 0.5 * td, tc + 0.5 * td

        # The stream interleaving here is nontrivial.
        # TODO: update diagram and link to it here
        if copy:
            self.src_a, self.src_b = self.src_b, self.src_a
            self._copy(rdr, gnm)
        self._interp(rdr, gnm, dim, ts, td)
        if self.filt_evt:
            self.stream_a.wait_for_event(self.filt_evt)
        self._iter(rdr, gnm, gprof, dim, tc)
        if self.copy_evt:
            self.stream_a.wait_for_event(self.copy_evt)
        for filt in rdr.filts:
            params = getattr(gprof.filters, filt.name)
            filt.apply(self.fb, gprof, params, dim, tc, self.stream_a)
        rdr.out.convert(self.fb, gprof, dim, self.stream_a)
        self.filt_evt = cuda.Event().record(self.stream_a)
        h_out = rdr.out.copy(self.fb, dim, self.fb.pool, self.stream_a)
        self.copy_evt = DurationEvent(timing_event).record(self.stream_a)

        self.info_a, self.info_b = self.info_b, self.info_a
        self.stream_a, self.stream_b = self.stream_b, self.stream_a
        return self.copy_evt, h_out
