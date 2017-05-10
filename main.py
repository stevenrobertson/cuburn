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

def save(output_module, name, rendered_frame):
    out, log = output_module.encode(rendered_frame)
    for suffix, file_like in out.items():
        with open(name + suffix, 'w') as fp:
            fp.write(file_like.read())
        if getattr(file_like, 'close', None):
            file_like.close()
    for key, val in log:
        print '\n=== %s ===' % key
        print val

def pyglet_preview(args, gprof, itr):
    import pyglet
    import pyglet.gl as gl
    w, h = gprof.width, gprof.height
    window = pyglet.window.Window(w, h, vsync=False)
    image = pyglet.image.CheckerImagePattern().create_image(w, h)
    tex = image.texture
    label = pyglet.text.Label('Rendering first frame', x=5, y=h-5,
                              width=w, anchor_y='top', font_size=16,
                              bold=True, multiline=True)

    @window.event
    def on_draw():
        window.clear()
        tex.blit(0, 0, 0)
        label.draw()

    @window.event
    def on_key_press(sym, mod):
        if sym == pyglet.window.key.Q:
            pyglet.app.exit()

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        pass

    last_time = [time.time()]

    def poll(dt):
        out = next(itr, False)
        if out is False:
            if args.pause:
                label.text = "Done. ('q' to quit)"
            else:
                pyglet.app.exit()
        elif out is not None:
            name, buf = out
            real_dt = time.time() - last_time[0]
            last_time[0] = time.time()
            if buf.dtype == np.uint8:
                fmt = gl.GL_UNSIGNED_BYTE
            elif buf.dtype == np.uint16:
                fmt = gl.GL_UNSIGNED_SHORT
            else:
                label.text = 'Unsupported format: ' + buf.dtype
                return

            h, w, ch = buf.shape
            gl.glEnable(tex.target)
            gl.glBindTexture(tex.target, tex.id)
            gl.glTexImage2D(tex.target, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGBA,
                            fmt, buf.tostring())
            gl.glDisable(tex.target)
            label.text = '%s (%g fps)' % (name, 1./real_dt)
        else:
            label.text += '.'

    pyglet.clock.set_fps_limit(20)
    pyglet.clock.schedule_interval(poll, 1/20.)
    pyglet.app.run()

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
    cuctx = dev.make_context(flags=cuda.ctx_flags.SCHED_YIELD)

    try:
      rmgr = render.RenderManager()
      arch = 'sm_{}{}'.format(
          dev.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR),
          dev.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR))
      rdr = render.Renderer(gnm, gprof, keep=args.keep, arch=arch)

      def render_iter():
          m = os.path.getmtime(args.flame)
          first = True
          for name, times in frames:
              if args.resume:
                  fp = name + rdr.out.suffix
                  if os.path.isfile(fp) and m < os.path.getmtime(fp):
                      continue

              for idx, t in enumerate(times):
                  evt, buf = rmgr.queue_frame(rdr, gnm, gprof, t, first)
                  first = False
                  while not evt.query():
                      time.sleep(0.01)
                      yield None
                  save(rdr.out, name, buf)
                  if args.rawfn:
                      try:
                          buf.tofile(args.rawfn + '.tmp')
                          os.rename(args.rawfn + '.tmp', args.rawfn)
                      except:
                          import traceback
                          print 'Failed to write %s: %s' % (args.rawfn,
                                                            traceback.format_exc())
                  print '%s (%3d/%3d), %dms' % (name, idx, len(times), evt.time())
                  sys.stdout.flush()
                  yield name, buf
              save(rdr.out, name, None)

      if args.gfx:
          pyglet_preview(args, gprof, render_iter())
      else:
          for i in render_iter(): pass
    finally:
      cuda.Context.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render fractal flames.')

    parser.add_argument('flame', metavar='ID', type=str,
        help="Filename or flame ID of genome to render")
    parser.add_argument('-g', action='store_true', dest='gfx',
        help="Show output in OpenGL window")
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
