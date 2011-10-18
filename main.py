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

import cuburn._pyflam3_hacks
from fr0stlib import pyflam3
from cuburn import render
from cuburn.code.mwc import MWCTest

np.set_printoptions(precision=5, edgeitems=20)

real_stdout = sys.stdout

def fmt_time(time):
    # Format time in a lexically-ordered way that doesn't interfere with the
    # typical case of ascending natural numbers
    atime = abs(time)
    dcml = ('-' if time < 0 else '') + ('%05d' % np.floor(atime))
    frac = np.round((atime - np.floor(atime)) * 1000)
    if frac:
        return '%s_%03d' % (dcml, frac)
    return dcml

def fmt_filename(args, time):
    return os.path.join(args.dir, '%s_%s' % (args.name, fmt_time(time)))

def save(args, time, raw):
    noalpha = raw[:,:,:3]
    if args.raw:
        real_stdout.write(buffer(np.uint8(noalpha * 255.0)))
        sys.stderr.write('.')
        return

    name = fmt_filename(args, time)
    img = scipy.misc.toimage(noalpha, cmin=0, cmax=1)
    img.save(name+'.png')

    if args.jpg is not None:
        img.save(name+'.jpg', quality=args.jpg)
    print 'saved', name

def error(msg):
    print "Error:", msg
    sys.exit(1)

def main(args):
    if args.test:
        MWCTest.test_mwc()
        return

    if args.raw:
        sys.stdout = sys.stderr

    genome_ptr, ngenomes = pyflam3.Genome.from_string(args.flame.read())
    genomes = cast(genome_ptr, POINTER(pyflam3.Genome*ngenomes)).contents

    if args.qs:
        for g in genomes:
            g.sample_density *= args.qs
            g.ntemporal_samples = max(1, int(g.ntemporal_samples * args.qs))

    if args.width:
        if not args.scale:
            args.scale = float(args.width) / genomes[0].width
        for g in genomes:
            g.width = args.width
    elif not np.all([g.width == genomes[0].width for g in genomes]):
        error("Inconsistent width. Force with --width.")

    if args.height:
        for g in genomes:
            g.height = args.height
    elif not np.all([g.height == genomes[0].height for g in genomes]):
        error("Inconsistent height. Force with --height.")

    if args.scale:
        for g in genomes:
            g.pixels_per_unit *= args.scale

    if args.skip and not args.tempscale:
        args.tempscale = float(args.skip)
    if args.tempscale:
        for g in genomes:
            g.temporal_filter_width *= args.tempscale

    if not args.name:
        if args.flame.name == '<stdin>':
            args.name = genomes[0].name
        else:
            args.name = os.path.splitext(os.path.basename(args.flame.name))[0]

    cp_times = [cp.time for cp in genomes]
    if args.renumber is not None:
        for t, g in enumerate(genomes):
            g.time = t - args.renumber
        cp_times = [cp.time for cp in genomes]
    elif np.any(np.diff(cp_times) <= 0):
        error("Genome times are non-monotonic. Try using --renumber.")

    if len(cp_times) > 2:
        if args.start is None:
            # [1], not [0]; want to be inclusive on --start but exclude
            # the first genome when --start is not passed
            args.start = cp_times[1]
        if args.end is None:
            args.end = cp_times[-1]
        if args.skip:
            times = np.arange(args.start, args.end, args.skip)
        else:
            times = [t for t in cp_times if args.start <= t < args.end]
    else:
        times = cp_times

    if args.resume:
        times = [t for t in times
                 if not os.path.isfile(fmt_filename(args, t)+'.png')]

    if times == []:
        print 'No genomes to be rendered.'
        return

    anim = render.Animation(genomes)
    if args.debug:
        anim.cmp_options.append('-G')
    anim.keep = args.keep or args.debug
    anim.compile()
    anim.load()

    if args.gfx:
        import pyglet
        window = pyglet.window.Window(anim.features.width, anim.features.height)
        image = pyglet.image.CheckerImagePattern().create_image(
                anim.features.width, anim.features.height)
        label = pyglet.text.Label('Rendering first frame', x=5, y=5,
                font_size=24, bold=True)

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

        frames = anim.render_frames(times, sync=args.sync)
        def poll(dt):
            out = next(frames, False)
            if out is False:
                if args.nopause:
                    pyglet.app.exit()
                else:
                    label.text = "Done. ('q' to quit)"
                    pyglet.clock.unschedule(poll)
            elif out is not None:
                real_dt = time.time() - last_time[0]
                last_time[0] = time.time()
                ftime, buf = out
                save(args, ftime, buf)
                imgbuf = np.uint8(buf.flatten() * 255)
                image.set_data('RGBA', -anim.features.width*4, imgbuf.tostring())
                label.text = '%s %4g (%g fps)' % (args.name, ftime, 1./real_dt)
            else:
                label.text += '.'
            if args.sleep:
                time.sleep(args.sleep / 1000.)

        pyglet.clock.set_fps_limit(30)
        pyglet.clock.schedule_interval(poll, 1/30.)
        pyglet.app.run()

    else:
        for ftime, out in ifilter(None, anim.render_frames(times, sync=args.sync)):
            save(args, ftime, out)
            if args.sleep:
                time.sleep(args.sleep / 1000.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render fractal flames.')

    parser.add_argument('flame', metavar='FILE', type=argparse.FileType(),
        help="Path to genome file ('-' for stdin)")
    parser.add_argument('-g', action='store_true', dest='gfx',
        help="Show output in OpenGL window")
    parser.add_argument('-j', metavar='QUALITY', nargs='?',
        action='store', type=int, dest='jpg', const=90,
        help="Write .jpg in addition to .png (default quality 90)")
    parser.add_argument('-n', metavar='NAME', type=str, dest='name',
        help="Prefix to use when saving files (default is basename of input)")
    parser.add_argument('-o', metavar='DIR', type=str, dest='dir',
        help="Output directory", default='.')
    parser.add_argument('--resume', action='store_true', dest='resume',
        help="Do not render any frame for which a .png already exists.")
    parser.add_argument('--raw', action='store_true', dest='raw',
        help="Do not write files; instead, send raw RGBA data to stdout.")
    parser.add_argument('--nopause', action='store_true',
        help="Don't pause after rendering when preview is up")

    seq = parser.add_argument_group('Sequence options', description="""
        Control which frames are rendered from a genome sequence. If '-k' is
        not given, '-s' and '-e' act as limits, and any control point with a
        time in bounds is rendered at its central time. If '-k' is given,
        a list of times to render is given according to the semantics of
        Python's range operator, as in range(start, end, skip).

        If no options are given, all control points except the first and last
        are rendered. If only one or two control points are passed, everything
        gets rendered.""")
    seq.add_argument('-s', dest='start', metavar='TIME', type=float,
        help="Start time of image sequence (inclusive)")
    seq.add_argument('-e', dest='end', metavar='TIME', type=float,
        help="End time of image sequence (exclusive)")
    seq.add_argument('-k', dest='skip', metavar='TIME', type=float,
        help="Skip time between frames in image sequence. Auto-sets "
             "--tempscale, use '--tempscale 1' to override.")
    seq.add_argument('--renumber', metavar="TIME", type=float,
        dest='renumber', nargs='?', const=0,
        help="Renumber frame times, counting up from the supplied start time "
             "(default is 0).")

    genome = parser.add_argument_group('Genome options')
    genome.add_argument('--qs', type=float, metavar='SCALE',
        help="Scale quality and number of temporal samples")
    genome.add_argument('--scale', type=float, metavar='SCALE',
        help="Scale pixels per unit (camera zoom)")
    genome.add_argument('--tempscale', type=float, metavar='SCALE',
        help="Scale temporal filter width")
    genome.add_argument('--width', type=int, metavar='PIXELS',
        help="Use this width. Auto-sets scale, use '--scale 1' to override.")
    genome.add_argument('--height', type=int, metavar='PIXELS',
        help="Use this height (does *not* auto-set scale)")

    debug = parser.add_argument_group('Debug options')
    debug.add_argument('--test', action='store_true', dest='test',
        help='Run some internal tests')
    debug.add_argument('--keep', action='store_true', dest='keep',
        help='Keep compilation directory (disables kernel caching)')
    debug.add_argument('--debug', action='store_true', dest='debug',
        help='Compile kernel with debugging enabled (implies --keep)')
    debug.add_argument('--sync', action='store_true', dest='sync',
        help='Use synchronous launches whenever possible')
    debug.add_argument('--sleep', metavar='MSEC', type=int, dest='sleep',
            nargs='?', const='5',
            help='Sleep between invocations. Keeps a single-card system '
                 'usable. Implies --sync.')
    args = parser.parse_args()

    main(args)

