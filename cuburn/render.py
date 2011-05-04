import sys
import math
import re
from ctypes import *
from cStringIO import StringIO
import numpy as np

from fr0stlib import pyflam3
from fr0stlib.pyflam3._flam3 import *
from fr0stlib.pyflam3.constants import *

from cuburn import affine
from cuburn.variations import Variations

class Genome(pyflam3.Genome):
    @classmethod
    def from_string(cls, *args, **kwargs):
        gnms = super(Genome, cls).from_string(*args, **kwargs)
        for g in gnms: g._init()
        return gnms

    def _init(self):
        self.xforms = [self.xform[i] for i in range(self.num_xforms)]
        dens = np.array([x.density for i, x in enumerate(self.xforms)
                         if i != self.final_xform_index])
        dens /= np.sum(dens)
        self.norm_density = [np.sum(dens[:i+1]) for i in range(len(dens))]

    scale = property(lambda cp: 2.0 ** cp.zoom)
    adj_density = property(lambda cp: cp.sample_density * (cp.scale ** 2))
    ppu = property(lambda cp: cp.pixels_per_unit * cp.scale)

    @property
    def camera_transform(cp):
        """
        An affine matrix which will transform IFS coordinates to image width
        and height. Assumes that width and height are constant.
        """
        # TODO: when reading as a property during packing, this may be
        # calculated 6 times instead of 1
        return ( affine.translate(0.5 * cp.width, 0.5 * cp.height)
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
    def __init__(self, genomes, ngenomes = None):
        self.features = Features(genomes)

    def compile(self):
        pass
    def render_frame(self, time=0):
        pass

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
            if not reduce(lambda a, b: a == b,
                          [cp.final_xform_index for cp in genomes]):
                raise ValueError("Differing final xform indexes")
            self.final_xform_index = genomes[0].final_xform_index
        else:
            self.final_xform_index = None

        self.width = genomes[0].width
        self.height = genomes[0].height

class XFormFeatures(object):
    def __init__(self, xforms, xform_id):
        self.id = xform_id
        any = lambda l: bool(filter(None, map(l, xforms)))
        self.has_post = any(lambda xf: getattr(xf, 'post', None))
        self.vars = set()
        for x in xforms:
            self.vars = (
                self.vars.union(set([i for i, v in enumerate(x.var) if v])))


