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

from fr0stlib.pyflam3 import *
from fr0stlib.pyflam3._flam3 import *
import pyglet

from cuburn.render import *


def mwctest():
    mwcent = ptx.Entry("mwc_test", 512)
    mwctest = MWCRNGTest(mwcent)

    # Get the source for saving and disassembly before potentially crashing
    mod = ptx.Module([mwcent])
    print '\n'.join(['%4d %s' % t for t in enumerate(mod.source.split('\n'))])
    util.disass(mod)

    mod = run.Module([mwcent])
    mod.print_func_info()

    ctx = mod.get_context('mwc_test', 14)
    mwctest.run_test(ctx)

def main(args):
    #mwctest()
    with open(args[-1]) as fp:
        genomes = Genome.from_string(fp.read())
    anim = Animation(genomes)
    anim.compile()
    bins = anim.render_frame()
    w, h = anim.features.hist_width, anim.features.hist_height
    bins = bins[:,:w]
    alpha = bins[...,3]
    k2ish = (256./(np.mean(alpha)+1e-9))
    lses = 20 * np.log2(1.0 + alpha * k2ish) / (alpha+1e-6)
    bins *= lses.reshape(h,w,1).repeat(4,2)
    bins = np.minimum(bins, 255)
    bins = bins.astype(np.uint8)

    if '-g' not in args:
        return

    window = pyglet.window.Window(1600, 900)
    image = pyglet.image.ImageData(anim.features.hist_width,
                                   anim.features.hist_height,
                                   'RGBA',
                                   bins.tostring())
                                   #-anim.features.hist_stride*4)
    tex = image.texture

    pal = (anim.ctx.ptx.instances[PaletteLookup].pal * 255.).astype(np.uint8)
    image2 = pyglet.image.ImageData(256, 16, 'RGBA', pal.tostring())

    @window.event
    def on_draw():
        window.clear()
        tex.blit(0, 0)
        image2.blit(0, 0)

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

