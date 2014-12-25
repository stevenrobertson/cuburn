import argparse
import unittest
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

from cuburn import render
from cuburn.output import launchC
from cuburn.code import output
from cuburn.code import util

class ProfileTest(unittest.TestCase, util.ClsMod):
    lib = util.devlib(deps=[output.pixfmtlib])

    def __init__(self, *args, **kwargs):
        super(ProfileTest, self).__init__(*args, **kwargs)
        self.load()
        self.fb = render.Framebuffers()
        self.dim = self.fb.calc_dim(640, 360)
        self.fb.alloc(self.dim)

    def test_clamping_below_0(self):
        ins = np.empty((self.dim.ah, self.dim.astride, 4), dtype='f4')
        ins[:] = -1
        cuda.memcpy_htod(self.fb.d_front, ins)

        launchC('f32_to_yuv444p', self.mod, None, self.dim, self.fb,
                self.fb.d_rb, self.fb.d_seeds)

        outs = np.empty((3, self.dim.h, self.dim.w), dtype='u1')
        cuda.memcpy_dtoh(outs, self.fb.d_back)
        self.assertTrue(np.all(outs[0] == 0))
        self.assertTrue(np.all(outs[1] >= 127))
        self.assertTrue(np.all(outs[1] <= 128))
        self.assertTrue(np.all(outs[2] >= 127))
        self.assertTrue(np.all(outs[2] <= 128))

    def test_clamping_above_1(self):
        ins = np.empty((self.dim.ah, self.dim.astride, 4), dtype='f4')
        ins[:] = 5
        cuda.memcpy_htod(self.fb.d_front, ins)

        launchC('f32_to_yuv444p', self.mod, None, self.dim, self.fb,
                self.fb.d_rb, self.fb.d_seeds)

        outs = np.empty((3, self.dim.h, self.dim.w), dtype='u1')
        cuda.memcpy_dtoh(outs, self.fb.d_back)
        self.assertTrue(np.all(outs[0] == 255))
        self.assertTrue(np.all(outs[1] >= 127))
        self.assertTrue(np.all(outs[1] <= 128))
        self.assertTrue(np.all(outs[2] >= 127))
        self.assertTrue(np.all(outs[2] <= 128))

    def test_yuv444p10_zero_passthru(self):
        ins = np.zeros((self.dim.ah, self.dim.astride, 4), dtype='f4')
        cuda.memcpy_htod(self.fb.d_front, ins)

        launchC('f32_to_yuv444p10', self.mod, None, self.dim, self.fb,
                self.fb.d_rb, self.fb.d_seeds)

        outs = np.empty((3, self.dim.h, self.dim.w), dtype='u2')
        cuda.memcpy_dtoh(outs, self.fb.d_back)
        self.assertTrue(np.all(outs[0] == 0))
        self.assertTrue(np.all(510 < outs[1]))
        self.assertTrue(np.all(outs[1] < 513))
        self.assertTrue(np.all(510 < outs[2]))
        self.assertTrue(np.all(outs[2] < 513))

    def test_yuv444p10_chroma_address_preservation(self):
        ins = np.empty((self.dim.ah, self.dim.astride, 4), dtype='f4')
        # Set everything to 0 except a few pixels
        ins[:] = 0
        ins[self.fb.gutter,self.fb.gutter,:] = [0, 1, 0, 1]
        ins[self.fb.gutter+1,self.fb.gutter+1,:] = [0, 1, 0, 1]
        cuda.memcpy_htod(self.fb.d_front, ins)

        launchC('f32_to_yuv444p10', self.mod, None, self.dim, self.fb,
                self.fb.d_rb, self.fb.d_seeds)

        outs = np.empty((3, self.dim.h, self.dim.w), dtype='u2')
        cuda.memcpy_dtoh(outs, self.fb.d_back)
        self.assertTrue(outs[0,0,0] > 0)
        self.assertTrue(outs[0,1,1] > 0)
        self.assertTrue(outs[1,0,0] < 500)
        self.assertTrue(outs[1,1,1] < 500)

    def test_yuv420p10_chroma_address_preservation(self):
        ins = np.empty((self.dim.ah, self.dim.astride, 4), dtype='f4')
        # Set everything to 0 except a few pixels
        ins[:] = 0

        # chroma loc (0,0): one pixel on
        ins[self.fb.gutter,self.fb.gutter,:] = [0, 1, 0, 1]

        # chroma loc (1,1): average of two strong pixels
        ins[self.fb.gutter+2,self.fb.gutter+2,:] = [0, 1, 0, 1]
        ins[self.fb.gutter+3,self.fb.gutter+3,:] = [1, 0, 0, 1]

        cuda.memcpy_htod(self.fb.d_front, ins)

        launchC('f32_to_yuv420p10', self.mod, None, self.dim, self.fb,
                self.fb.d_rb, self.fb.d_seeds)

        w, h = self.dim.w, self.dim.h
        outs = np.empty((3, h, w), dtype='u2')
        cuda.memcpy_dtoh(outs, self.fb.d_back)
        out_cr = outs[1,:h/4].reshape(h/2,w/2)
        out_cb = outs[1,h/4:h/2].reshape(h/2,w/2)

        # chroma blocking doesn't affect luma blocking
        self.assertTrue(outs[0,0,0] > 0)
        self.assertTrue(outs[0,1,0] == 0)
        self.assertTrue(outs[0,0,1] == 0)
        self.assertTrue(outs[0,1,1] == 0)

        # locations are preserved
        self.assertTrue(outs[0,2,2] > 0)
        self.assertTrue(outs[0,3,3] > 0)

        # chroma from first pixel makes it through, neighbor is fine
        self.assertTrue(172 <= out_cr[0,0] <= 174)
        self.assertTrue(511 <= out_cr[0,1] <= 512)
        self.assertTrue(511 <= out_cr[1,0] <= 512)
