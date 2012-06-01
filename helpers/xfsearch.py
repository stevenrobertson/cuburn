#!/usr/bin/python2

"""
Render multiple versions of a genome, disabling each xform and variation in
turn, for use in debugging.
"""

import os, sys, json, scipy, pycuda.autoinit
import numpy as np
from copy import deepcopy

sys.path.insert(0, '.')

from cuburn import genome, profile, render

from main import save

def main(gnm_path, time):
    basename = os.path.basename(gnm_path).rsplit('.', 1)[0]
    rmgr = render.RenderManager()
    def go(gj, name):
        gprof = profile.wrap(profile.BUILTIN['720p'], gj)
        rt = [('out/%s_%s_%04d.jpg' % (name, basename, time * 10000), time)]
        for out in rmgr.render(gnm, gprof, rt):
            save(out)

    gnm = json.load(open(gnm_path))

    for i in gnm['xforms']:
        xf = gnm['xforms'].pop(i)
        go(gnm, 'noxf_' + i)
        gnm['xforms'][i] = xf

    vars = set([v for g in gnm['xforms'].values() for v in g['variations']])
    for v in vars:
        g2 = deepcopy(gnm)
        for x in g2['xforms'].values():
            x['variations'].pop(v, None)
            if not x['variations']:
                x['variations']['linear'] = {'weight': 1}
        go(g2, 'novar_' + v)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('Usage: progname gnmpath render_time_01')
    main(sys.argv[1], float(sys.argv[2]))
