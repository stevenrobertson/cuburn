#!/usr/bin/python
#
# cuburn, one of a surprisingly large number of ports of the fractal flame
# algorithm to NVIDIA GPUs.
#
# This one is copyright 2010-2011, Steven Robertson <steven@strobe.cc>
# and Eric Reckase <e.reckase@gmail.com>.
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
import Image
import scipy
import pycuda.driver as cuda

from cuburn import genome, render

profiles = {
    '1080p': dict(fps=24, width=1920, height=1080, quality=3000, skip=0),
    '720p': dict(fps=24, width=1280, height=720, quality=2500, skip=0),
    'preview': dict(fps=24, width=640, height=360, quality=800, skip=1)
}

def save(rframe):
    noalpha = rframe.buf[:,:,:3]
    img = scipy.misc.toimage(noalpha, cmin=0, cmax=1)
    img.save(rframe.idx, quality=95)
    print rframe.idx, rframe.gpu_time

def main(args, prof):
    import pycuda.autoinit

    gnm_str = args.flame.read()
    if '<' in gnm_str[:10]:
        flames = genome.XMLGenomeParser.parse(gnm_str)
        if len(flames) != 1:
            warnings.warn('%d flames in file, only using one.' % len(flames))
        gnm = genome.convert_flame(flames[0])
    else:
        gnm = json.loads(gnm_str)
    gnm = genome.Genome(gnm)
    err, times = gnm.set_profile(prof)

    anim = render.Renderer()
    anim.compile(gnm, keep=args.keep)
    anim.load(gnm)

    basename = os.path.basename(args.flame.name).rsplit('.', 1)[0] + '_'
    if args.flame.name == '-':
        basename = ''
    if args.name is not None:
        basename = args.name
    prefix = os.path.join(args.dir, basename)
    frames = [(prefix + '%05d.jpg' % (i+1), t) for i, t in enumerate(times)]
    if args.end:
        frames = frames[:args.end]
    frames = frames[args.start::prof['skip']+1]
    if args.resume:
        m = 0
        if args.flame.name != '-':
            m = os.path.getmtime(args.flame.name)
        frames = (f for f in frames
                  if not os.path.isfile(f[0]) or m > os.path.getmtime(f[0]))

    w, h = prof['width'], prof['height']
    gen = anim.render(gnm, frames, w, h)

    if args.gfx:
        import pyglet
        window = pyglet.window.Window(w, h)
        image = pyglet.image.CheckerImagePattern().create_image(w, h)
        label = pyglet.text.Label('Rendering first frame', x=5, y=5,
                font_size=16, bold=True)

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
                if args.nopause:
                    pyglet.app.exit()
                else:
                    label.text = "Done. ('q' to quit)"
                    pyglet.clock.unschedule(poll)
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
    else:
        for out in gen:
            save(out)
            if args.sync:
                cuda.Context.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render fractal flames.')

    parser.add_argument('flame', metavar='FILE', type=argparse.FileType(),
        help="Path to genome file ('-' for stdin)")
    parser.add_argument('-g', action='store_true', dest='gfx',
        help="Show output in OpenGL window")
    parser.add_argument('-n', metavar='NAME', type=str, dest='name',
        help="Prefix to use when saving files (default is basename of input)")
    parser.add_argument('-o', metavar='DIR', type=str, dest='dir',
        help="Output directory", default='.')
    parser.add_argument('--resume', action='store_true', dest='resume',
        help="Don't overwrite output files that are newer than the input")
    parser.add_argument('--nopause', action='store_true',
        help="Don't pause after rendering the last frame when previewing")

    parser.add_argument('--keep', action='store_true', dest='keep',
        help='Keep compilation directory (disables kernel caching)')
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
    prof.add_argument('--skip', dest='skip', metavar='N', type=int,
        help="Skip N frames between each rendered frame")
    prof.add_argument('--quality', type=int, metavar='SPP',
        help="Set base samples per pixel")
    prof.add_argument('--fps', type=float, dest='fps',
        help="Set frames per second (24)")
    prof.add_argument('--width', type=int, metavar='PX')
    prof.add_argument('--height', type=int, metavar='PX')

    args = parser.parse_args()
    prof = dict(profiles[args.prof])
    for k in prof:
        if getattr(args, k) is not None:
            prof[k] = getattr(args, k)
    prof['duration'] = args.duration

    main(args, prof)
