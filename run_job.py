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

def save(time, raw):
    noalpha = raw[:,:,:3]
    name = 'out/%05d' % time
    img = scipy.misc.toimage(noalpha, cmin=0, cmax=1)
    img.save(name+'.png')

def main(jobfilepath):
    contents = '\n'.join([l.split('//', 1)[0] for l in open(jobfilepath)])
    # This includes the genomes and other cruft, a dedicated reader will be
    # built in time tho
    info = cuburn.genome.load_info(contents)

    times = np.linspace(0, 1, info.duration * info.fps + 1)[:3]

    renderer = cuburn.render.Renderer(info)
    renderer.compile()
    renderer.load()

    for idx, (ftime, out) in enumerate(renderer.render(zip(times, times[1:]))):
        save(idx, out)

if __name__ == "__main__":
    main('04653_2.jsonc')
