#!/usr/bin/python

# Temporary helper script while I update main.py

import os
import sys
import time
import argparse
import multiprocessing
from subprocess import Popen
from ctypes import *
from itertools import ifilter

import numpy as np
import Image
import scipy
import pycuda.autoinit

import cuburn.render
import cuburn.genome

import pycuda.compiler
import pycuda.driver as cuda
import cuburn.code.interp

np.set_printoptions(precision=5, edgeitems=20)

real_stdout = sys.stdout

def save(time, raw, pfx):
    noalpha = raw[:,:,:3]
    name = pfx + '%05d' % time
    img = scipy.misc.toimage(noalpha, cmin=0, cmax=1)
    img.save(name+'.png')
    print name

def main(jobfilepath, outprefix):
    # This includes the genomes and other cruft, a dedicated reader will be
    # built in time tho
    info = cuburn.genome.load_info(open(jobfilepath).read())

    times = np.linspace(0, 1, info.duration * info.fps + 1)

    # One still, one motion-blurred for testing
    rtimes = [(times[0], times[0]), (times[1], times[2])]

    renderer = cuburn.render.Renderer(info)
    renderer.compile()
    renderer.load()

    for idx, (ftime, out) in enumerate(renderer.render(rtimes)):
        save(idx, out, outprefix)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'out/')
