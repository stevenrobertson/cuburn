import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.driver as cuda

from code.util import ClsMod, launch
from code.output import f32tou8lib

import scipy.misc

if not hasattr(scipy.misc, 'toimage'):
    raise ImportError("Could not find scipy.misc.toimage. "
                      "Are scipy and PIL installed?")

class Output(object):
    def convert(self, fb, gnm, dim, stream=None):
        """
        Convert a filtered buffer to whatever output format is needed by the
        writer.
        """
        raise NotImplementedError()

    def copy(self, fb, dim, pool, stream=None):
        """
        Schedule a copy from the device buffer to host memory, returning the
        target buffer.
        """
        raise NotImplementedError()

class PILOutput(Output, ClsMod):
    lib = f32tou8lib

    def convert(self, fb, gnm, dim, stream=None):
        launch('f32_to_u8', self.mod, stream,
                (32, 8, 1), (int(np.ceil(dim.w/32.)), int(np.ceil(dim.h/8.))),
                fb.d_rb, fb.d_seeds, fb.d_back, fb.d_front,
                i32(fb.gutter), i32(dim.w), i32(dim.astride), i32(dim.h))

    def copy(self, fb, dim, pool, stream=None):
        h_out = pool.allocate((dim.h, dim.w, 4), 'u1')
        cuda.memcpy_dtoh_async(h_out, fb.d_back, stream)
        return h_out

    @staticmethod
    def save(buf, name, type=None, quality=98):
        type = dict(jpg='jpeg', tif='tiff').get(type, type)
        if type == 'jpeg' or (type is None and name.endswith('.jpg')):
            buf = buf[:,:,:3]
        img = scipy.misc.toimage(buf, cmin=0, cmax=1)
        img.save(name, type, quality=quality)
