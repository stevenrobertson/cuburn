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

from cuburnlib.device_code import IterThread
from cuburnlib.cuda import LaunchContext
from fr0stlib.pyflam3 import *
from fr0stlib.pyflam3._flam3 import *
from cuburnlib.render import *

def main(args):
    verbose = 1
    if '-d' in args:
        verbose = 3

    with open(args[-1]) as fp:
        genomes = Genome.from_string(fp.read())
    anim = Animation(genomes)
    anim.compile()
    anim.render_frame()


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
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[-1]):
        print "Last argument must be a path to a genome file"
        sys.exit(1)
    main(sys.argv)

