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
import pycuda.tools
from pycuda.gpuarray import vec

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


        num_std_xf = len(dens)
        self.chaos_densities = np.zeros( (num_std_xf,num_std_xf) )
        for r in range(num_std_xf):
            chaos_row = np.array([ctypes_genome.chaos[r][c]
                                  for c in range(num_std_xf)])
            chaos_row = chaos_row * dens
            chaos_row /= np.sum(chaos_row)
            chaos_row = np.cumsum(chaos_row)
            self.chaos_densities[r,:] = chaos_row

        dens /= np.sum(dens)
        self.norm_density = np.cumsum(dens)

        # For performance reasons, defer this calculation
        self._camera_transform = None

    scale = property(lambda cp: 2.0 ** cp.zoom)
    adj_density = property(lambda cp: cp.sample_density * (cp.scale ** 2))
    ppu = property(lambda cp: cp.pixels_per_unit * cp.scale)

    @property
    def camera_transform(self):
        """
        An affine matrix which will transform IFS coordinates to image width
        and height. Assumes that width and height are constant.
        """
        cp = self
        if self._camera_transform is not None:
            return self._camera_transform
        g = Features.gutter
        if cp.estimator:
            # The filter shifts by this amount as a side effect of it being
            # written in a confusing and sloppy manner
            # TODO: this will be weird in an animation where one endpoint has
            # a radius of 0, and the other does not
            g -= Features.gutter / 2 - 1
        self._camera_transform = (
                 affine.translate(0.5 * cp.width + g, 0.5 * cp.height + g)
               * affine.scale(cp.ppu, cp.ppu)
               * affine.translate(-cp._center[0], -cp._center[1])
               * affine.rotate(cp.rotate * 2 * np.pi / 360,
                               cp.rot_center[0],
                               cp.rot_center[1]) )
        return self._camera_transform

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

    # Large launches lock the display for a considerable period and may be
    # killed due to a device timeout; small launches are harder to load-balance
    # on the GPU and incur overhead. This empirical value is multiplied by the
    # number of SMs on the device to determine how many blocks should be in
    # each launch. Extremely high quality, high resolution renders may still
    # encounter a device timeout, requiring the user to increase the split
    # amount. This factor is not used in async mode.
    SM_FACTOR = 8

    cmp_options = ('-use_fast_math', '-maxrregcount', '42')


    keep = False

    def __init__(self, ctypes_genome_array):
        self._g_arr = type(ctypes_genome_array)()
        libflam3.flam3_align(self._g_arr, ctypes_genome_array,
                             len(ctypes_genome_array))
        self.genomes = map(Genome, self._g_arr)
        self.features = Features(self.genomes)
        self._iter = self._de = self.src = self.cubin = self.mod = None

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

        self._iter = iter.IterCode(self.features)
        self._de = filtering.DensityEst(self.features, self.genomes[0])
        cclip = filtering.ColorClip(self.features)
        # TODO: make choice of filtering explicit
        self.src = util.assemble_code(util.BaseCode, mwc.MWC, self._iter.packer,
                                      self._iter, cclip, self._de)
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


    def render_frames(self, times=None, sync=False):
        """
        Render a flame for each genome in the iterable value 'genomes'.
        Returns a Python generator object which will yield a 2-tuple of
        ``(time, buf)``, where ``time`` is the central time of the frame and
        ``buf`` is a 3D (width, height, channel) NumPy array containing
        [0,1]-valued RGBA components.

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

        If ``sync`` is True, the CPU will sync with the GPU after every block
        of temporal samples and yield None until the frame is ready. This
        allows a single-card system to avoid having to go thirty seconds
        between window refreshes while rendering. Otherwise, tasks will be
        piled asynchronously on the card so that it is always under load.
        """
        if times == []:
            return

        f = self.features

        times = times if times is not None else [cp.time for cp in self.genomes]
        iter_stream = cuda.Stream()
        filt_stream = cuda.Stream()
        cen_cp = pyflam3.Genome()
        dst_cp = pyflam3.Genome()

        nbins = f.acc_height * f.acc_stride
        d_accum = cuda.mem_alloc(16 * nbins)
        d_out = cuda.mem_alloc(16 * nbins)

        num_sm = cuda.Context.get_device().multiprocessor_count
        if sync:
            cps_per_block = num_sm * self.SM_FACTOR
        else:
            cps_per_block = f.max_cps

        info_size = self._iter.packer.align * cps_per_block
        d_infos = cuda.mem_alloc(info_size)
        d_palmem = cuda.mem_alloc(256 * f.palette_height * 4)

        seeds = mwc.MWC.make_seeds(self._iter.NTHREADS * cps_per_block)
        d_seeds = cuda.to_device(seeds)

        h_infos = cuda.pagelocked_empty((info_size / 4,), np.float32)
        h_palmem = cuda.pagelocked_empty(
                (f.palette_height, 256, 4), np.uint8)
        h_out = cuda.pagelocked_empty((f.acc_height, f.acc_stride, 4), np.float32)

        filter_done_event = None

        packer = self._iter.packer
        iter_fun = self.mod.get_function("iter")
        #iter_fun.set_cache_config(cuda.func_cache.PREFER_L1)

        util.BaseCode.zero_dptr(self.mod, d_accum, 4 * nbins, filt_stream)

        last_time = times[0]

        for time in times:
            self._interp(cen_cp, time)

            h_palmem[:] = self._interp_colors(dst_cp, time,
                                              cen_cp.temporal_filter_width)
            cuda.memcpy_htod_async(d_palmem, h_palmem, iter_stream)
            tref = self.mod.get_texref('palTex')
            array_info = cuda.ArrayDescriptor()
            array_info.height = f.palette_height
            array_info.width = 256
            array_info.array_format = cuda.array_format.UNSIGNED_INT8
            array_info.num_channels = 4
            tref.set_address_2d(d_palmem, array_info, 1024)

            tref.set_format(cuda.array_format.UNSIGNED_INT8, 4)
            tref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
            tref.set_filter_mode(cuda.filter_mode.LINEAR)

            # Must be accumulated over all CPs
            gam, vib = 0, 0
            bkgd = np.zeros(3)

            mblur_times = enumerate( np.linspace(-0.5, 0.5, cen_cp.ntemporal_samples)
                                     * cen_cp.temporal_filter_width + time )

            for block_times in _chunk(list(mblur_times), cps_per_block):
                infos = []
                if len(self.genomes) > 1:
                    for n, t in block_times:
                        self._interp(dst_cp, t)
                        frac = float(n) / cen_cp.ntemporal_samples
                        info = packer.pack(cp=Genome(dst_cp), cp_step_frac=frac)
                        infos.append(info)
                        gam += dst_cp.gamma
                        vib += dst_cp.vibrancy
                        bkgd += np.array(dst_cp.background)
                else:
                    # Can't interpolate normally; just pack copies
                    packed = packer.pack(cp=self.genomes[0], cp_step_frac=0)
                    infos = [packed] * len(block_times)
                    gam += self.genomes[0].gamma * len(block_times)
                    vib += self.genomes[0].vibrancy * len(block_times)
                    bkgd += np.array(self.genomes[0].background) * len(block_times)

                infos = np.concatenate(infos)
                h_infos[:len(infos)] = infos
                cuda.memcpy_htod_async(d_infos, h_infos)

                if filter_done_event:
                    iter_stream.wait_for_event(filter_done_event)

                # TODO: replace with option to split long runs shorter ones
                # for interactivity
                for i in range(1):
                    iter_fun(d_seeds, d_infos, np.uint64(d_accum),
                             block=(32, self._iter.NTHREADS/32, 1),
                             grid=(len(block_times), 1),
                             texrefs=[tref], stream=iter_stream)

                    if sync:
                        iter_stream.synchronize()
                        yield None

            if filter_done_event and not sync:
                filt_stream.synchronize()
                yield last_time, self._trim(h_out)
                last_time = time

            util.BaseCode.zero_dptr(self.mod, d_out, 4 * nbins, filt_stream)
            self._de.invoke(self.mod, Genome(cen_cp), d_accum, d_out, filt_stream)
            util.BaseCode.zero_dptr(self.mod, d_accum, 4 * nbins, filt_stream)
            filter_done_event = cuda.Event().record(filt_stream)

            f32 = np.float32
            n = f32(cen_cp.ntemporal_samples)
            gam = f32(n / gam)
            vib = f32(vib / n)
            hipow = f32(cen_cp.highlight_power)
            lin = f32(cen_cp.gam_lin_thresh)
            lingam = f32(math.pow(cen_cp.gam_lin_thresh, gam-1.0) if lin > 0 else 0)
            bkgd = vec.make_float3(*(bkgd / n))

            color_fun = self.mod.get_function("colorclip")
            color_fun(d_out, gam, vib, hipow, lin, lingam, bkgd,
                      block=(256, 1, 1), grid=(nbins / 256, 1),
                      stream=filt_stream)
            cuda.memcpy_dtoh_async(h_out, d_out, filt_stream)

            if sync:
                filt_stream.synchronize()
                yield time, self._trim(h_out)

        if not sync:
            filt_stream.synchronize()
            yield time, self._trim(h_out)

    def _interp(self, cp, time):
        flam3_interpolate(self._g_arr, len(self._g_arr), time, 0, byref(cp))

    @staticmethod
    def _pal_to_np(cp):
        # Converting palettes by iteration has an enormous performance
        # overhead. We cheat massively and dangerously here.
        pal = cast(pointer(cp.palette), POINTER(c_double * (256 * 5)))
        val = np.frombuffer(buffer(pal.contents), count=256*5)
        return np.uint8(np.reshape(val, (256, 5))[:,1:] * 255.0)

    def _interp_colors(self, cp, time, twidth):
        # TODO: any visible difference between uint8 and richer formats?
        height = self.features.palette_height
        pal = np.empty((height, 256, 4), dtype=np.uint8)

        if len(self.genomes) > 1:
            # The typical case; applying real motion blur
            times = np.linspace(-0.5, 0.5, height) * twidth + time
            for n, t in enumerate(times):
                self._interp(cp, t)
                pal[n] = self._pal_to_np(cp)
        else:
            # Cannot call any interp functions on a single genome; rather than
            # have alternate code-paths, just copy the same colors everywhere
            pal[0] = self._pal_to_np(self.genomes[0])
            pal[1:] = pal[0]
        return pal

    def _trim(self, result):
        g = self.features.gutter
        return result[g:-g,g:-g].copy()


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

    # TODO: for now, we always throw away the alpha channel before writing.
    # All code is in place to not do this, we just need to find a way to expose
    # this preference via the API (or push alpha blending entirely on the client,
    # which I'm not opposed to)
    alpha_output_channel = False

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

        alphas = np.array([c.color[3] for g in genomes
                           for c in g.palette.entries])
        self.pal_has_alpha = np.any(alphas != 1.0)

        self.max_cps = max([cp.ntemporal_samples for cp in genomes])

        self.width = genomes[0].width
        self.height = genomes[0].height
        self.acc_width = genomes[0].width + 2 * self.gutter
        self.acc_height = genomes[0].height + 2 * self.gutter
        self.acc_stride = 32 * int(math.ceil(self.acc_width / 32.))
        self.std_xforms = filter(lambda v: v != self.final_xform_index,
                                 range(self.nxforms))
        self.chaos_used = False
        for cp in genomes:
            for r in range(len(self.std_xforms)):
                for c in range(len(self.std_xforms)):
                    if cp.chaos[r][c] != 1.0:
                        self.chaos_used = True



class XFormFeatures(object):
    def __init__(self, xforms, xform_id):
        self.id = xform_id
        any = lambda l: bool(filter(None, map(l, xforms)))

        self.has_post = any(lambda xf: not self.id_matrix(xf.post))
        self.vars = set()
        for x in xforms:
            self.vars = (
                self.vars.union(set([i for i, v in enumerate(x.var) if v])))

    @staticmethod
    def id_matrix(m):
        return (m[0][0] == 1 and m[1][0] == 0 and m[2][0] == 0 and
                m[0][1] == 0 and m[1][1] == 1 and m[2][1] == 0)

