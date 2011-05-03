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
from pprint import pprint
from ctypes import *

import numpy as np
np.set_printoptions(precision=5, edgeitems=20)
import scipy

from fr0stlib.pyflam3 import *
from fr0stlib.pyflam3._flam3 import *

import pyglet
window = pyglet.window.Window(1024, 1024)
import pycuda.gl.autoinit

from cuburn.render import *
from cuburn.code.mwc import MWCTest
from cuburn.code.iter import silly

def main(args):
    #MWCTest.test_mwc()
    with open(args[-1]) as fp:
        genomes = Genome.from_string(fp.read())
    anim = Animation(genomes)
    accum, den = silly(anim.features, genomes)

    if '-g' not in args:
        return

    noalpha = np.delete(accum, 3, axis=2)
    scipy.misc.imsave('rendered.png', noalpha)

    imgbuf = (np.minimum(accum * 255, 255)).astype(np.uint8)

    image = pyglet.image.ImageData(1024, 1024, 'RGBA', imgbuf.tostring(), -4096)
    tex = image.texture

    #pal = (anim.ctx.ptx.instances[PaletteLookup].pal * 255.).astype(np.uint8)
    #image2 = pyglet.image.ImageData(256, 16, 'RGBA', pal.tostring())

    @window.event
    def on_draw():
        window.clear()
        tex.blit(0, 0)
        #image2.blit(0, 0)

    @window.event
    def on_key_press(sym, mod):
        if sym == pyglet.window.key.Q:
            pyglet.app.exit()

    pyglet.app.run()

if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[-1]):
        print "Last argument must be a path to a genome file"
        sys.exit(1)
    main(sys.argv)

