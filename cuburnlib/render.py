
from ctypes import *
import numpy as np
from fr0stlib.pyflam3 import Genome, Frame
from fr0stlib.pyflam3._flam3 import *
from fr0stlib.pyflam3.constants import *

Point = lambda x, y: np.array([x, y], dtype=np.double)

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
        self.genomes = (Genome * len(genomes))()
        for i in range(len(genomes)):
            memmove(byref(self.genomes[i]), byref(genomes[i]),
                    sizeof(BaseGenome))

        self._frame = Frame()
        self._frame.genomes = cast(self.genomes, POINTER(BaseGenome))
        self._frame.ngenomes = len(genomes)

    def render_frame(self, time=0):
        # TODO: support more nuanced frame control than just 'time'
        # TODO: reuse more information between frames
        # TODO: allow animation-long override of certain parameters (size, etc)

        cp = BaseGenome()
        flam3_interpolate(self.frame.genomes, len(self.genomes), time, 0,
                          byref(cp))
        filt = Filters(self.frame, cp)
        rw = cp.spatial_oversample * cp.width  + 2 * filt.gutter
        rh = cp.spatial_oversample * cp.height + 2 * filt.gutter

        # Allocate buckets, accumulator
        # Loop over all batches:
        #   [density estimation]
        #   Loop over all temporal samples:
        #     Color scalar = temporal filter at index
        #     Interpolate and get control point
        #     Precalculate
        #     Prepare xforms
        #     Compute colormap
        #     Run iterations
        #     Accumulate vibrancy, gamma, background
        #   Calculate k1, k2
        #   If not DE, then do log filtering to accumulator
        #   Else, [density estimation]
        # Do final clip and filter

        # For now:
        # Loop over all batches:
        #   Loop over all temporal samples:
        #     Interpolate and get control point
        #     Read the
        #     Dump noise into buckets
        #   Do log filtering to accumulator
        # Do simplified final clip

class Filters(object):
    def __init__(self, frame, cp):
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
        self.gutter = (spa_width - cp.spatial_oversample) / 2

class Camera(object):
    """Viewport and exposure."""
    def __init__(self, frame, cp, filters):
        # Calculate the conversion matrix between the IFS space (xform
        # coordinates) and the sampling lattice (bucket addresses)
        # TODO: test this code (against compute_camera?)
        scale = 2.0 ** cp.zoom
        self.sample_density = cp.sample_density * scale * scale

        center = Point(cp.center[0], cp.center[1])
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
        self.ifs_space_size = 1.0 / (self.upper_bounds - self.lower_bounds)
        # TODO: coordinate transforms in concert with GPU (rotation, size)


