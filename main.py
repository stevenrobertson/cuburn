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
import time
import ctypes
import struct

# These imports are order-sensitive!
import pyglet
import pyglet.gl as gl
gl.get_current_context()

import numpy as np

from fr0stlib import pyflam3
from cuburnlib.device_code import MWCRNGTest
from cuburnlib.cuda import LaunchContext

class CUGenome(pyflam3.Genome):
    def _render(self, frame, trans):
        obuf = (ctypes.c_ubyte * ((3+trans)*self.width*self.height))()
        stats = pyflam3.RenderStats()
        pyflam3.flam3_render(ctypes.byref(frame), obuf,
                             pyflam3.flam3_field_both,
                             trans+3, trans, ctypes.byref(stats))
        return obuf, stats, frame


def main(genome_path):
    ctx = LaunchContext([MWCRNGTest], block=(256,1,1), grid=(64,1), tests=True)
    ctx.compile(True)
    ctx.run_tests()

    with open(genome_path) as fp:
        genome = CUGenome.from_string(fp.read())[0]


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

