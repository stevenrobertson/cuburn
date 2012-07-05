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
import pycuda.driver as cuda

sys.path.insert(0, os.path.dirname(__file__))
from cuburn import render, filters, output, profile
from cuburn.genome import convert, use, db

def save(out):
    # Temporary! TODO: fix this
    output.PILOutput.save(out.buf, out.idx)
    print out.idx, out.gpu_time

def main(args, prof):
    import pycuda.autoinit

    gdb = db.connect(args.genomedb)

    gnm, basename = gdb.get_anim(args.flame, args.half)
    gprof = profile.wrap(prof, gnm)

    if args.name is not None:
        basename = args.name
    prefix = os.path.join(args.dir, basename)
    if args.subdir:
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        prefix += '/'
    else:
        prefix += '_'
    frames = [('%s%05d%s.jpg' % (prefix, (i+1), args.suffix), t)
              for i, t in profile.enumerate_times(gprof)]
    if args.resume:
        m = os.path.getmtime(args.flame)
        frames = (f for f in frames
                  if not os.path.isfile(f[0]) or m > os.path.getmtime(f[0]))

    rmgr = render.RenderManager()
    gen = rmgr.render(gnm, gprof, frames)

    if not args.gfx:
        for out in gen:
            save(out)
        return

    import pyglet
    w, h = gprof.width, gprof.height
    window = pyglet.window.Window(w, h, vsync=False)
    image = pyglet.image.CheckerImagePattern().create_image(w, h)
    label = pyglet.text.Label('Rendering first frame', x=5, y=h-5,
                              width=w, anchor_y='top', font_size=16,
                              bold=True, multiline=True)

    @window.event
    def on_draw():
        window.clear()
        image.texture.blit(0, 0)
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
        out = next(gen, False)
        if out is False:
            if args.pause:
                label.text = "Done. ('q' to quit)"
                #pyglet.clock.unschedule(poll)
            else:
                pyglet.app.exit()
        elif out is not None:
            real_dt = time.time() - last_time[0]
            last_time[0] = time.time()
            save(out)
            imgbuf = np.uint8(out.buf.flatten() * 255)
            image.set_data('RGBA', -w*4, imgbuf.tostring())
            label.text = '%s (%g fps)' % (out.idx, 1./real_dt)
        else:
            label.text += '.'

    pyglet.clock.set_fps_limit(30)
    pyglet.clock.schedule_interval(poll, 1/30.)
    pyglet.app.run()


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
    parser.add_argument('--half', action='store_true',
        help='Use half-loops when converting nodes to animations')
    profile.add_args(parser)

    args = parser.parse_args()
    pname, prof = profile.get_from_args(args)
    main(args, prof)
