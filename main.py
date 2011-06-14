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
from subprocess import Popen

from pprint import pprint
from ctypes import *

import numpy as np
np.set_printoptions(precision=5, edgeitems=20)
import scipy

import pyglet
import pycuda.autoinit

import cuburn._pyflam3_hacks
from fr0stlib import pyflam3
from cuburn.render import *
from cuburn.code.mwc import MWCTest

# Required on my system; CUDA doesn't yet work with GCC 4.5
os.environ['PATH'] = ('/usr/x86_64-pc-linux-gnu/gcc-bin/4.4.5:'
                     + os.environ['PATH'])

def main(args):
    if '-t' in args:
        MWCTest.test_mwc()

    with open(args[1]) as fp:
        genome_ptr, ngenomes = pyflam3.Genome.from_string(fp.read())
        genomes = cast(genome_ptr, POINTER(pyflam3.Genome*ngenomes)).contents
    anim = Animation(genomes)
    anim.compile()
    anim.load()
    for n, out in enumerate(anim.render_frames()):
        noalpha = np.delete(out, 3, axis=2)
        name = 'rendered_%03d' % n
        scipy.misc.imsave(name+'.png', noalpha)
        # Convert using imagemagick, to set custom quality
        Popen(['convert', name+'.png', '-quality', '90', name+'.jpg'])
    return

    #if '-g' not in args:
    #    return

    window = pyglet.window.Window(anim.features.width, anim.features.height)
    imgbuf = (np.minimum(accum * 255, 255)).astype(np.uint8)
    image = pyglet.image.ImageData(anim.features.width, anim.features.height,
                                   'RGBA', imgbuf.tostring(),
                                   -anim.features.width * 4)
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
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print "Last argument must be a path to a genome file"
        sys.exit(1)
    main(sys.argv)

