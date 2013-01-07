import os
import json
import argparse
import numpy as np

from genome.specs import toplevels
from genome.use import RefWrapper, SplineWrapper

BUILTIN={
    '1080p': dict(width=1920, height=1080),
    '720p': dict(width=1280, height=720),
    '540p': dict(width=960, height=540),
    'preview': dict(width=640, height=360, spp=1200, skip=1)
}

def add_args(parser=None):
    """
    Add profile argument groups to an ArgumentParser, for use with
    get_from_args. (If `parser` is None, a new one will be made.)
    """
    parser = argparse.ArgumentParser() if parser is None else parser
    prof = parser.add_argument_group('Profile options')
    prof.add_argument('-P', '--builtin-profile', choices=BUILTIN.keys(),
        help='Set parameters below from a builtin profile. (default: 720p)',
        default='720p')
    prof.add_argument('-p', '--profile', type=argparse.FileType(),
        metavar='PROFILE', help='Set profile from a JSON file.')

    tmp = parser.add_argument_group('Temporal options')
    tmp.add_argument('--duration', type=float, metavar='TIME',
        help="Override base duration in seconds")
    tmp.add_argument('--fps', type=float, dest='fps',
        help="Override frames per second")
    tmp.add_argument('--start', metavar='FRAME_NO', type=int,
        help="First frame to render (1-indexed, inclusive)")
    tmp.add_argument('--end', metavar='FRAME_NO', type=int,
        help="Last frame to render (1-indexed, exclusive, negative from end)")
    tmp.add_argument('--skip', dest='skip', metavar='N', type=int,
        help="Skip N frames between each rendered frame")
    # TODO: eliminate the 'silently overwritten' bit.
    tmp.add_argument('--shard', dest='shard', metavar='SECS', type=float,
        help="Write SECS of output into each file, instead of one frame per "
             "file. If set, causes 'start', 'end', and 'skip' to be ignored. "
             "If output codecs don't support multi-file writing, files will "
             "be silently overwritten.")
    tmp.add_argument('--frame_width', metavar='SCALE', type=float,
        help='Adjustment factor for temporal frame width.')

    tmp.add_argument('--still', action='store_true',
        help='Override start, end, and temporal frame width to render one '
             'frame without motion blur.')

    spa = parser.add_argument_group('Spatial options')
    spa.add_argument('--spp', type=int, metavar='SPP',
        help="Set base samples per pixel")
    spa.add_argument('--width', type=int, metavar='PX')
    spa.add_argument('--height', type=int, metavar='PX')

    out = parser.add_argument_group('Output options')
    out.add_argument('--codec', choices=['jpg', 'png', 'tiff', 'x264'])
    return parser

def get_from_args(args):
    """
    Get profile from an ArgumentParser result. Returns `(name, prof)`.
    """
    if args.profile:
        name = os.path.basename(args.profile.name).rsplit('.', 1)[0]
        base = json.load(args.profile)
    else:
        name = args.builtin_profile
        base = BUILTIN[args.builtin_profile]

    if args.still:
        base.update(frame_width=0, start=1, end=2)
    for arg in 'duration fps frame_width start end skip shard spp width height'.split():
        if getattr(args, arg, None) is not None:
            base[arg] = getattr(args, arg)
    if args.codec is not None:
        base.setdefault('output', {})['type'] = args.codec

    return name, base

def wrap(prof, gnm):
    """
    Create a wrapped profile from plain dicts `prof` and `gnm`. The wrapped
    profile follows the structure of the profile but returns genome-adjusted
    data for any RefScalar value in its spec.
    """
    scale = gnm.get('time', {}).get('duration', 1)
    return RefWrapper(prof, toplevels['profile'],
                      other=SplineWrapper(gnm, scale=scale))

def enumerate_times(gprof):
    """
    Given a profile, return a list of `(frame_no, center_times)` pairs. Note
    that the enumeration is applied before `start`, `end`, and `skip`, and so
    `frame_no` may be non-contiguous.
    """
    nframes = round(gprof.fps * gprof.duration)
    times = np.linspace(0, 1, nframes + 1)
    times = times[:-1] + 0.5 * (times[1] - times[0])
    if gprof.shard:
        s = max(1, int(round(gprof.fps * gprof.shard)))
        return [(i, times[t:t+s])
                for i, t in enumerate(range(0, len(times), s), 1)]
    else:
        times = [[t] for t in times]
    times = list(enumerate(times, 1))
    if gprof.end is not None:
        times = times[:gprof.end]
    if gprof.start is not None:
        times = times[gprof.start:]
    return times[::gprof.skip+1]
