import sys
import math
import re
from ctypes import *
from cStringIO import StringIO
import numpy as np

from fr0stlib import pyflam3
from fr0stlib.pyflam3._flam3 import *
from fr0stlib.pyflam3.constants import *

from cuburn.device_code import *
from cuburn.variations import Variations

Point = lambda x, y: np.array([x, y], dtype=np.double)

class Genome(pyflam3.Genome):
    pass

class XForm(object):
    """
    A Python structure (*not* a ctypes wrapper) storing an xform. There are
    a few differences between the meaning of properties on this object and
    those of the C version; they are noted below.
    """
    def __init__(self, **kwargs):
        read_affine = lambda c: [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]]]
        self.coefs = read_affine(map(float, kwargs.pop('coefs').split()))
        if 'post' in kwargs:
            self.post = read_affine(map(float, kwargs.pop('post').split()))
        # TODO: more verification, detection of unknown variables, etc
        for k, v in kwargs.items():
            setattr(self, k, float(v))

    # ctypes was being a pain. just parse the string.
    @classmethod
    def parse(cls, cp):
        flame_str = flam3_print_to_string(byref(cp))
        xforms = []
        for line in flame_str.split('\n'):
            if not line.strip().startswith('<xform'):
                continue
            props = dict(re.findall(r'(\w*)="([^"]*)"', line))
            xforms.append(cls(**props))
        # Set cumulative xform weight
        xforms[0].cweight = xforms[0].weight
        for i in range(1, len(xforms)):
            xforms[i].cweight = xforms[i].weight + xforms[i-1].cweight
        xforms[-1].cweight = 1.0
        return xforms

class _Frame(pyflam3.Frame):
    """
    ctypes flam3_frame object used for genome interpolation and
    spatial filter creation
    """
    def __init__(self, genomes, *args, **kwargs):
        pyflam3.Frame.__init__(self, *args, **kwargs)
        self.genomes = (BaseGenome * len(genomes))()
        for i in range(len(genomes)):
            memmove(byref(self.genomes[i]), byref(genomes[i]),
                    sizeof(BaseGenome))
        self.ngenomes = len(genomes)

        # TODO: allow user to override this
        self.pixel_aspect_ratio = 1.0

    def interpolate(self, time, stagger=0, cp=None):
        cp = cp or BaseGenome()
        flam3_interpolate(self.genomes, self.ngenomes, time,
                          stagger, byref(cp))
        return cp

class Frame(object):
    """
    Handler for a single frame of a rendered genome.
    """
    def __init__(self, _frame, time):
        self._frame = _frame
        self.center_cp = self._frame.interpolate(time)

    def upload_data(self, ctx, filters, time):
        """
        Prepare and upload the data needed to render this frame to the device.
        """
        center = self.center_cp
        ncps = center.nbatches * center.ntemporal_samples

        if ncps < ctx.nctas:
            raise NotImplementedError(
                "Distribution of a CP across multiple CTAs not yet done")

        # TODO: isn't this leaking xforms from C all over the place?
        stream = StringIO()
        cp_list = []

        for batch_idx in range(center.nbatches):
            for time_idx in range(center.ntemporal_samples):
                idx = time_idx + batch_idx * center.nbatches
                interp_time = time + filters.temporal_deltas[idx]
                cp = self._frame.interpolate(interp_time)
                cp_list.append(cp)

                cp.camera = Camera(self._frame, cp, filters)
                cp.nsamples = (cp.camera.sample_density *
                               center.width * center.height) / ncps
                cp.xforms = XForm.parse(cp)

        print "Expected writes:", (
                cp.camera.sample_density * center.width * center.height)
        min_time = min(filters.temporal_deltas)
        max_time = max(filters.temporal_deltas)
        for i, cp in enumerate(cp_list):
            cp.norm_time = (filters.temporal_deltas[i] - min_time) / (
                            max_time - min_time)
            CPDataStream.pack_into(ctx, stream, frame=self, cp=cp, cp_idx=idx)
        PaletteLookup.upload_palette(ctx, self, cp_list)
        stream.seek(0)
        IterThread.upload_cp_stream(ctx, stream.read(), ncps)

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
    def __init__(self, genomes):
        # _frame is the ctypes frame object used only for interpolation
        self._frame = _Frame(genomes)

        # Use the same set of filters throughout the anim, a la flam3
        self.filters = Filters(self._frame, genomes[0])
        self.features = Features(genomes, self.filters)

    def compile(self):
        """
        Create a PTX kernel optimized for this animation, compile it, and
        attach it to a LaunchContext with a thread distribution optimized for
        the active device.
        """
        # TODO: automatic optimization of block parameters
        entry = ptx.Entry("iterate", 512)
        iter = IterThread(entry, self.features)
        entry.finalize()
        iter.cp.finalize()
        srcmod = ptx.Module([entry])
        util.disass(srcmod)
        self.mod = run.Module([entry])

    def render_frame(self, time=0):
        # TODO: support more nuanced frame control than just 'time'
        # TODO: reuse more information between frames
        # TODO: allow animation-long override of certain parameters (size, etc)
        frame = Frame(self._frame, time)
        frame.upload_data(self.ctx, self.filters, time)
        IterThread.call(self.ctx)
        return HistScatter.get_bins(self.ctx, self.features)

class Filters(object):
    def __init__(self, frame, cp):
        # Use one oversample per filter set, even over multiple timesteps
        self.oversample = frame.genomes[0].spatial_oversample

        # Ugh. I'd really like to replace this mess
        spa_filt_ptr = POINTER(c_double)()
        spa_width = flam3_create_spatial_filter(byref(frame),
                                                flam3_field_both,
                                                byref(spa_filt_ptr))
        if spa_width < 0:
            raise EnvironmentError("flam3 call failed")
        self.spatial = np.asarray([[spa_filt_ptr[y*spa_width+x] for x in
            range(spa_width)] for y in range(spa_width)], dtype=np.double)
        self.spatial_width = spa_width
        flam3_free(spa_filt_ptr)

        tmp_filt_ptr = POINTER(c_double)()
        tmp_deltas_ptr = POINTER(c_double)()
        steps = cp.nbatches * cp.ntemporal_samples
        self.temporal_sum = flam3_create_temporal_filter(
                steps,
                cp.temporal_filter_type,
                cp.temporal_filter_exp,
                cp.temporal_filter_width,
                byref(tmp_filt_ptr),
                byref(tmp_deltas_ptr))
        self.temporal = np.asarray([tmp_filt_ptr[i] for i in range(steps)],
                                   dtype=np.double)
        flam3_free(tmp_filt_ptr)
        self.temporal_deltas = np.asarray(
                [tmp_deltas_ptr[i] for i in range(steps)], dtype=np.double)
        flam3_free(tmp_deltas_ptr)

        # TODO: density estimation
        self.gutter = (spa_width - self.oversample) / 2



class Features(object):
    """
    Determine features and constants required to render a particular set of
    genomes. The values of this class are fixed before compilation begins.
    """
    # Constant parameters which control handling of out-of-frame samples:
    # Number of iterations to iterate without write after new point
    fuse = 2
    # Maximum consecutive out-of-frame points before picking new point
    max_bad = 3

    # Height of the texture pallete which gets uploaded to the GPU (assuming
    # that palette-from-texture is enabled). For most genomes, this doesn't
    # need to be very large at all. However, since only an easily-cached
    # fraction of this will be accessed per SM, larger values shouldn't hurt
    # performance too much. Power-of-two, please.
    palette_height = 16

    def __init__(self, genomes, flt):
        any = lambda l: bool(filter(None, map(l, genomes)))
        self.max_ntemporal_samples = max(
                [cp.nbatches * cp.ntemporal_samples for cp in genomes])
        self.camera_rotation = any(lambda cp: cp.rotate)
        self.non_box_temporal_filter = genomes[0].temporal_filter_type
        self.palette_mode = genomes[0].palette_mode and "linear" or "nearest"

        xforms = [XForm.parse(cp) for cp in genomes]
        assert len(xforms[0]) == len(xforms[-1]), ("genomes must have same "
            "number of xforms! (try running through flam3-genome first)")
        self.xforms = [XFormFeatures([x[i] for x in xforms], i)
                       for i in range(len(xforms[0]))]
        if any(lambda cp: cp.final_xform_enable):
            raise NotImplementedError("Final xform")

        # Histogram (and log-density copy) width and height
        self.hist_width  = flt.oversample * genomes[0].width  + 2 * flt.gutter
        self.hist_height = flt.oversample * genomes[0].height + 2 * flt.gutter
        # Histogram stride, for better filtering. This code assumes the
        # 128-byte L1 cache line width of Fermi devices, and a 16-byte
        # histogram bucket size. TODO: detect these things programmatically,
        # particularly the histogram bucket size, which may be split soon
        self.hist_stride = 8 * int(math.ceil(self.hist_width / 8.0))

class XFormFeatures(object):
    def __init__(self, xforms, xform_id):
        self.id = xform_id
        any = lambda l: bool(filter(None, map(l, xforms)))
        self.has_post = any(lambda xf: getattr(xf, 'post', None))
        self.vars = set([n for x in xforms for n in Variations.names
                           if getattr(x, n, None)])

class Camera(object):
    """Viewport and exposure."""
    def __init__(self, frame, cp, filters):
        # Calculate the conversion matrix between the IFS space (xform
        # coordinates) and the sampling lattice (bucket addresses)
        # TODO: test this code (against compute_camera?)
        scale = 2.0 ** cp.zoom
        self.sample_density = cp.sample_density * scale * scale

        center = Point(cp._center[0], cp._center[1])
        size = Point(cp.width, cp.height)

        # pix per unit, where 'unit' is '1.0' in IFS space
        self.ppu = Point(
            cp.pixels_per_unit * scale / frame.pixel_aspect_ratio,
            cp.pixels_per_unit * scale)
        # extra shifts applied due to gutter
        gutter = filters.gutter / (cp.spatial_oversample * self.ppu)
        cornerLL = center - (size / (2 * self.ppu))
        self.lower_bounds = cornerLL - gutter
        self.upper_bounds = cornerLL + (size / self.ppu) + gutter
        self.norm_scale = 1.0 / (self.upper_bounds - self.lower_bounds)
        self.norm_offset = -self.norm_scale * self.lower_bounds
        self.idx_scale = size * self.norm_scale
        self.idx_offset = size * self.norm_offset
