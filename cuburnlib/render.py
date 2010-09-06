from ctypes import *
from cStringIO import StringIO
import numpy as np

from fr0stlib import pyflam3
from fr0stlib.pyflam3._flam3 import *
from fr0stlib.pyflam3.constants import *

from cuburnlib.cuda import LaunchContext
from cuburnlib.device_code import IterThread, CPDataStream

Point = lambda x, y: np.array([x, y], dtype=np.double)

class Genome(pyflam3.Genome):
    pass

class Frame(pyflam3.Frame):
    def interpolate(self, time, cp):
        flam3_interpolate(self.genomes, self.ngenomes, time, 0, byref(cp))

    def pack_stream(self, ctx, time):
        """
        Pack and return the control point data stream to render this frame.
        """
        # Get the central control point, and calculate parameters that change
        # once per frame
        cp = BaseGenome()
        self.interpolate(time, cp)
        self.filt = Filters(self, cp)
        rw = cp.spatial_oversample * cp.width  + 2 * self.filt.gutter
        rh = cp.spatial_oversample * cp.height + 2 * self.filt.gutter

        # Interpolate each time step, calculate per-step variables, and pack
        # into the stream
        cp_streamer = ctx.ptx.instances[CPDataStream]
        stream = StringIO()
        print "Data stream contents:"
        cp_streamer.print_record()
        tcp = BaseGenome()
        for batch_idx in range(cp.nbatches):
            for time_idx in range(cp.ntemporal_samples):
                idx = time_idx + batch_idx * cp.nbatches
                cp_time = time + self.filt.temporal_deltas[idx]
                self.interpolate(time, tcp)
                tcp.camera = Camera(self, tcp, self.filt)

                # TODO: figure out which object to pack this into
                nsamples = ((tcp.camera.sample_density * cp.width * cp.height) /
                            (cp.nbatches * cp.ntemporal_samples))
                samples_per_thread = nsamples / ctx.threads + 15

                cp_streamer.pack_into(stream,
                        frame=self,
                        cp=tcp,
                        cp_idx=idx,
                        samples_per_thread=samples_per_thread)
        stream.seek(0)
        return (stream.read(), cp.nbatches * cp.ntemporal_samples)

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

        self.features = Features(genomes)
        self.frame = Frame()
        self.frame.genomes = cast(self.genomes, POINTER(BaseGenome))
        self.frame.ngenomes = len(genomes)

        self.ctx = None

    def compile(self):
        """
        Create a PTX kernel optimized for this animation, compile it, and
        attach it to a LaunchContext with a thread distribution optimized for
        the active device.
        """
        # TODO: user-configurable test control
        self.ctx = LaunchContext([IterThread], block=(256,1,1), grid=(54,1),
                                 tests=True)
        # TODO: user-configurable verbosity control
        self.ctx.compile(verbose=3, anim=self, features=self.features)
        # TODO: automatic optimization of block parameters

    def render_frame(self, time=0):
        # TODO: support more nuanced frame control than just 'time'
        # TODO: reuse more information between frames
        # TODO: allow animation-long override of certain parameters (size, etc)
        cp_stream, num_cps = self.frame.pack_stream(self.ctx, time)
        iter_thread = self.ctx.ptx.instances[IterThread]
        iter_thread.upload_cp_stream(self.ctx, cp_stream, num_cps)
        iter_thread.call(self.ctx)

class Features(object):
    """
    Determine features and constants required to render a particular set of
    genomes. The values of this class are fixed before compilation begins.
    """
    # Constant; number of rounds spent fusing points on first CP of a frame
    num_fuse_samples = 25

    def __init__(self, genomes):
        self.max_ntemporal_samples = max(
                [cp.nbatches * cp.ntemporal_samples for cp in genomes]) + 1

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
        self.ifs_space_size = 1.0 / (self.upper_bounds - self.lower_bounds)
        # TODO: coordinate transforms in concert with GPU (rotation, size)

