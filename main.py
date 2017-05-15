#!/usr/bin/env python2
#
# cuburn, one of a surprisingly large number of ports of the fractal flame
# algorithm to NVIDIA GPUs.
#
# This one is copyright 2010-2012, Steven Robertson <steven@strobe.cc>
# and Erik Reckase <e.reckase@gmail.com>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 or later
# as published by the Free Software Foundation.

import os
import sys
import time
import json
import warnings
import argparse
from subprocess import Popen
from itertools import ifilter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cuburn import render, filters, output, profile
from cuburn.genome import convert, use, db

def main(args, prof):
    gdb = db.connect(args.genomedb)
    gnm, basename = gdb.get_anim(args.flame, args.half)
    if getattr(args, 'print'):
        print convert.to_json(gnm)
        return
    gprof = profile.wrap(prof, gnm)

    if args.name is not None:
        basename = args.name
    prefix = os.path.join(args.dir, basename)
    if args.subdir:
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        prefix_plus = prefix + '/'
    else:
        prefix_plus = prefix + '_'

    frames = [('%s%05d%s' % (prefix_plus, i, args.suffix), t)
              for i, t in profile.enumerate_times(gprof)]

    import pycuda.driver as cuda
    cuda.init()
    dev = cuda.Device(args.device or 0)
    cuctx = dev.make_context(flags=cuda.ctx_flags.SCHED_BLOCKING_SYNC)

    try:
      rmgr = render.RenderManager()
      arch = 'sm_{}{}'.format(
          dev.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR),
          dev.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR))
      rdr = render.Renderer(gnm, gprof, keep=args.keep, arch=arch)
      last_render_time_ms = 0

      m = os.path.getmtime(args.flame)
      for name, times in frames:
          if args.resume:
              fp = name + output.get_suffix_for_profile(gprof)
              if os.path.isfile(fp) and m < os.path.getmtime(fp):
                  continue

          def save(buf):
              out, log = rdr.out.encode(buf)
              for suffix, file_like in out.items():
                  with open(name + suffix, 'w') as fp:
                      fp.write(file_like.read())
                  if getattr(file_like, 'close', None):
                      file_like.close()
              for key, val in log:
                  print >> sys.stderr, '\n=== %s ===' % key
                  print >> sys.stderr, val

          evt = buf = next_evt = next_buf = None
          for idx, t in enumerate(list(times) + [None]):
              evt, buf = next_evt, next_buf
              if t is not None:
                  next_evt, next_buf = rmgr.queue_frame(rdr, gnm, gprof, t)
              if not evt: continue
              if last_render_time_ms > 2000:
                while not evt.query():
                  time.sleep(0.2)
              else:
                evt.synchronize()
              last_render_time_ms = evt.time()

              save(buf)

              if args.rawfn:
                  try:
                      buf.tofile(args.rawfn + '.tmp')
                      os.rename(args.rawfn + '.tmp', args.rawfn)
                  except:
                      import traceback
                      print >> sys.stderr, 'Failed to write %s: %s' % (
                          args.rawfn, traceback.format_exc())
              print >> sys.stderr, '%s%s (%3d/%3d), %dms' % (
                  ('%d: ' % args.device) if args.device >= 0 else '',
                  name, idx, len(times), last_render_time_ms)
              sys.stderr.flush()

          save(None)

    finally:
      cuda.Context.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render fractal flames.')

    parser.add_argument('flame', metavar='ID', type=str,
        help="Filename or flame ID of genome to render")
    parser.add_argument('-n', metavar='NAME', type=str, dest='name',
        help="Prefix to use when saving files (default is basename of input)")
    parser.add_argument('--suffix', metavar='NAME', type=str, dest='suffix',
        help="Suffix to use when saving files (default '')", default='')
    parser.add_argument('-o', metavar='DIR', type=str, dest='dir',
        help="Output directory", default='.')
    parser.add_argument('--resume', action='store_true', dest='resume',
        help="Don't overwrite output files that are newer than the input")
    parser.add_argument('--pause', action='store_true',
        help="Don't close the preview window after rendering is finished")
    parser.add_argument('-d', '--genomedb', metavar='PATH', type=str,
        help="Path to genome database (file or directory, default '.')",
        default='.')
    parser.add_argument('--subdir', action='store_true',
        help="Use basename as subdirectory of out dir, instead of prefix")
    parser.add_argument('--raw', metavar='PATH', type=str, dest='rawfn',
        help="Target file for raw buffer, to enable previews.")
    parser.add_argument('--half', action='store_true',
        help='Use half-loops when converting nodes to animations')
    parser.add_argument('--print', action='store_true',
        help="Print the blended animation and exit.")
    parser.add_argument('--device', metavar='NUM', type=int,
                        help="GPU device number to use (from nvidia-smi).")
    parser.add_argument('--keep', action='store_true',
                        help="Keep compiled kernels to help with profiling")
    profile.add_args(parser)

    args = parser.parse_args()
    pname, prof = profile.get_from_args(args)
    main(args, prof)
