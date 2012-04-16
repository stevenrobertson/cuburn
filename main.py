#!/usr/bin/python
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
from cuburn import render, filters, output
from cuburn.genome import convert, use, db

profiles = {
    '1080p': dict(width=1920, height=1080),
    '720p': dict(width=1280, height=720),
    '540p': dict(width=960, height=540),
    'preview': dict(width=640, height=360, spp=1200, skip=1)
}

def save(out):
    # Temporary! TODO: fix this
    output.PILOutput.save(out.buf, out.idx)
    print out.idx, out.gpu_time

def main(args, prof):
    import pycuda.autoinit

    gdb = db.connect(args.genomedb)
    if os.path.isfile(args.flame) and (args.flame.endswith('.flam3') or
                                       args.flame.endswith('.flame')):
        with open(args.flame) as fp:
            gnm_str = fp.read()
        flames = convert.XMLGenomeParser.parse(gnm_str)
        if len(flames) != 1:
            warnings.warn('%d flames in file, only using one.' % len(flames))
        gnm = convert.flam3_to_node(flames[0])
    else:
        gnm = gdb.get(args.flame)

    if gnm['type'] == 'node':
        gnm = convert.node_to_anim(gnm, half=args.half)
    elif gnm['type'] == 'edge':
        gnm = convert.edge_to_anim(gdb, gnm)
    assert gnm['type'] == 'animation', 'Unrecognized genome type.'

    gprof, times = use.wrap_genome(prof, gnm)
    rmgr = render.RenderManager()

    basename = os.path.basename(args.flame).rsplit('.', 1)[0] + '_'
    if args.name is not None:
        basename = args.name
    prefix = os.path.join(args.dir, basename)
    frames = [('%s%05d%s.jpg' % (prefix, (i+1), args.suffix), t)
              for i, t in enumerate(times)]
    if args.end:
        frames = frames[:args.end]
    frames = frames[args.start::gprof.skip+1]
    if args.resume:
        m = os.path.getmtime(args.flame)
        frames = (f for f in frames
                  if not os.path.isfile(f[0]) or m > os.path.getmtime(f[0]))

    w, h = gprof.width, gprof.height
    gen = rmgr.render(gnm, gprof, frames)

    if not args.gfx:
        for out in gen:
            save(out)
        return

    import pyglet
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
        if args.sync:
            cuda.Context.synchronize()

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
    parser.add_argument('--genomedb', '-d', metavar='PATH', type=str,
        help="Path to genome database (file or directory, default '.')",
        default='.')

    parser.add_argument('--sync', action='store_true', dest='sync',
        help='Use synchronous launches whenever possible')

    parser.add_argument('--duration', type=float, metavar='TIME',
        help="Set base duration in seconds (30)", default=30)
    parser.add_argument('--start', metavar='FRAME_NO', type=int,
        default=0, help="First frame to render (inclusive)")
    parser.add_argument('--end', metavar='FRAME_NO', type=int,
        help="Last frame to render (exclusive, negative OK)")

    prof = parser.add_argument_group('Profile options')
    prof.add_argument('-p', dest='prof', choices=profiles.keys(),
        default='preview', help='Set profile, specifying defaults for all '
        'options below. (default: "preview")')
    prof.add_argument('--pfile', type=argparse.FileType(), metavar='PROFILE',
        help='Set profile using a JSON file, overriding -p.')
    prof.add_argument('--skip', dest='skip', metavar='N', type=int,
        help="Skip N frames between each rendered frame")
    prof.add_argument('--quality', type=int, metavar='SPP',
        help="Set base samples per pixel")
    prof.add_argument('--fps', type=float, dest='fps',
        help="Set frames per second (24)")
    prof.add_argument('--width', type=int, metavar='PX')
    prof.add_argument('--height', type=int, metavar='PX')

    node = parser.add_argument_group('Node options')
    node.add_argument('--half', action='store_true',
        help='Use a half-loop when rendering a node.')
    node.add_argument('--still', action='store_true',
        help='Override start, end, and temporal frame width to render one '
             'frame without motion blur. (Works on edges too)')

    args = parser.parse_args()
    prof = dict(profiles[args.prof])
    if args.pfile:
        prof = json.load(open(args.pfile))
    for k in ['duration', 'skip', 'quality', 'fps']:
        if getattr(args, k) is not None:
            prof[k] = getattr(args, k)
    if args.still:
        args.start = 0
        args.end = 1
        prof['frame_width'] = 0

    main(args, prof)
