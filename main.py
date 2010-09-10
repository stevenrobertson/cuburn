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
from ctypes import *

import numpy as np

from cuburn.device_code import IterThread
from cuburn.cuda import LaunchContext
from fr0stlib.pyflam3 import *
from fr0stlib.pyflam3._flam3 import *
from cuburn.render import *

import pyglet

def dump_3d(nda):
    with open('/tmp/data.txt', 'w') as f:
        for row in nda:
            f.write('  |  '.join([' '.join(
                ['%4.1g\t' % x for x in pt]) for pt in row]) + '\n')

def main(args):
    verbose = 1
    if '-d' in args:
        verbose = 3

    with open(args[-1]) as fp:
        genomes = Genome.from_string(fp.read())
    anim = Animation(genomes)
    anim.compile()
    bins = anim.render_frame()
    #dump_3d(bins)
    bins /= ((np.mean(bins)+1e-9)/128.)
    bins.astype(np.uint8)


    if '-g' not in args:
        return

    print anim.features.hist_width
    print anim.features.hist_height
    print anim.features.hist_stride
    window = pyglet.window.Window(800, 600)
    image = pyglet.image.ImageData(anim.features.hist_width,
                                   anim.features.hist_height,
                                   'RGBA',
                                   bins.tostring(),
                                   anim.features.hist_stride*4)
    tex = image.texture

    @window.event
    def on_draw():
        window.clear()
        tex.blit(0, 0)

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

