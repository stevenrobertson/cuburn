#!/usr/bin/python
#
# flam3cuda, one of a surprisingly large number of ports of the fractal flame
# algorithm to NVIDIA GPUs.
#
# This one is copyright 2010 Steven Robertson <steven@strobe.cc>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 or later
# as published by the Free Software Foundation.

import os
import sys
import ctypes
import struct

import tempita

# These imports are order-sensitive!
import pyglet
import pyglet.gl as gl
gl.get_current_context()

import pycuda.driver as cuda
import pycuda.gl as cudagl
import pycuda.gl.autoinit
from pycuda.compiler import SourceModule

from multiprocessing import Process, Queue

import numpy as np

from fr0stlib import pyflam3

# PTX header and functions used for debugging.
prelude = """
.version 2.0
.target sm_20

.func (.reg .u32 $ret) get_gtid ()
{
    .reg .u16 tmp;
    .reg .u32 cta, ncta, tid, gtid;

    mov.u16         tmp,    %ctaid.x;
    cvt.u32.u16     cta,    tmp;
    mov.u16         tmp,    %ntid.x;
    cvt.u32.u16     ncta,   tmp;
    mul24.lo.u32    gtid,   cta,    ncta;

    mov.u16         tmp,    %tid.x;
    cvt.u32.u16     tid,    tmp;
    add.u32         gtid,   gtid,   tid;
    mov.b32         $ret,   gtid;
    ret;
}

.entry write_to_buffer ( .param .u32 bufbase )
{
    .reg .u32 base, gtid, off;

    ld.param.u32    base,       [bufbase];
    call.uni        (off),      get_gtid,   ();
    mad24.lo.u32    base,       off,        4,          base;
    st.volatile.global.b32      [base],     off;
}
"""

class CUGenome(pyflam3.Genome):
    def _render(self, frame, trans):
        obuf = (ctypes.c_ubyte * ((3+trans)*self.width*self.height))()
        stats = pyflam3.RenderStats()
        pyflam3.flam3_render(ctypes.byref(frame), obuf, pyflam3.flam3_field_both,
                     trans+3, trans, ctypes.byref(stats))
        return obuf, stats, frame

class LaunchContext(self):
    def __init__(self, seed=None):
        self.block, self.grid, self.threads = None, None, None
        self.stream = cuda.Stream()
        self.rand = mtrand.RandomState(seed)

    def set_size(self, block, grid):
        self.block, self.grid = block, grid
        self.threads = reduce(lambda a, b: a*b, self.block + self.grid)

class PTXFragment(object):
    """
    Wrapper for sections of template PTX.

    In order to provide the best optimization, and avoid a web of hard-coded
    parameters, the PTX module may be regenerated and recompiled several times
    with different or incomplete launch context parameters. To this end, avoid
    accessing the GPU in such functions, and do not depend on context values
    which are marked as "tuned" in the LaunchContext docstring being available.

    The final compilation pass is guaranteed to have all "tuned" values fixed
    in their final values for the stream.

    Template code will be processed recursively until all "{{" instances have
    been replaced, using the same namespace each time.
    """

    def deps(self, ctx):
        """
        Returns a list of PTXFragment objects on which this object depends
        for successful compilation. Circular dependencies are forbidden.
        """
        return []

    def subs(self, ctx):
        """
        Returns a dict of items to add to the template substitution namespace.
        The entire dict will be assembled, including all dependencies, before
        any templates are evaluated.
        """
        return {}

    def prelude(self, ctx):
        """
        Returns a template string containing any code (variable declarations,
        probably) that should be inserted at module scope. The prelude of
        all deps will be inserted above this prelude.
        """
        return ""

    def entryPrelude(self, ctx):
        """
        Returns a template string that should be inserted at the top of any
        entry point which depends on this method. The entry prelude of all
        deps will be inserted above this entry prelude.
        """
        return ""

    def setUp(self, ctx):
        """
        Do start-of-stream initialization, such as copying data to the device.
        """
        pass

    def test(self, ctx):
        """
        Perform device tests. Returns True on success, False on failure,
        or raises an exception.
        """
        return True

class PTXEntryPoint(PTXFragment):
    def entry(self, ctx):
        """
        Returns a template string corresponding to a PTX entry point.
        """
        pass

    def call(self, ctx):
        """
        Calls the entry point on the device. Haven't worked out the details
        of this one yet.
        """
        pass



class DeviceHelpers(PTXFragment):
    """This one's included by default, no need to depend on it"""
    def subs(self, ctx):
        return {
            'PTRT': ctypes.sizeof(ctypes.c_void_p) == 8 and '.u64' or '.u32',
            }

class MWCRandGen(PTXFragment):

    _prelude = """
    .const {{PTRT}} mwc_rng_mults_p;
    .const {{PTRT}} mwc_rng_seeds_p;
    """

    def __init__(self):
        if not os.path.isfile(os.path.join(os.path.dirname(__FILE__),
                                           'primes.bin')):
            raise EnvironmentError('primes.bin not found')

    def prelude(self):
        return self._prelude

    def setUp(self, ctx):
        # Load raw big-endian u32 multipliers from primes.bin.
        with open('primes.bin') as primefp:
            dt = np.dtype(np.uint32).newbyteorder('B')
            mults = np.frombuffer(primefp.read(), dtype=dt)
        # Randomness in choosing multipliers is good, but larger multipliers
        # have longer periods, which is also good. This is a compromise.
        ctx.rand.shuffle(mults[:ctx.threads*4])
        # Copy multipliers and seeds to the device
        devmp, devml = ctx.mod.get_global('mwc_rng_mults')
        cuda.memcpy_htod_async(devmp, mults.tostring()[:devml], ctx.stream)
        devsp, devsl = ctx.mod.get_global('mwc_rng_seeds')
        cuda.memcpy_htod_async(devsp, ctx.rand.bytes(devsl), ctx.stream)

    def _next_b32(self, dreg):
        return """
    mul.wide.u32    mwc_rng_
        mul.wide.u32



    def templates(self, ctx):
        return {'mwc_next_b32', self._next_b32}


    def test(self, ctx):








    def launch(self, ctx):
        if self.mults



def main(genome_path):




    #with open(genome_path) as fp:
        #genome = CUGenome.from_string(fp.read())[0]
    #genome.width, genome.height = 512, 512
    #genome.sample_density = 1000
    #obuf, stats, frame = genome.render(estimator=3)
    #gc.collect()

        ##q.put(str(obuf))
    ##p = Process(target=render, args=(q, genome_path))
    ##p.start()

    #window = pyglet.window.Window()
    #image = pyglet.image.ImageData(genome.width, genome.height, 'RGB', obuf)
    #tex = image.texture

    #@window.event
    #def on_draw():
        #window.clear()
        #tex.blit(0, 0)

    #pyglet.app.run()

if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print "First argument must be a path to a genome file"
        sys.exit(1)
    main(sys.argv[1])

