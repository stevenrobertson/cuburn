"""
The main iteration loop.
"""

import pycuda.driver as cuda
from pycuda.driver import In, Out, InOut
from pycuda.compiler import SourceModule
import numpy as np

from cuburn.code import mwc, variations
from cuburn.code.util import *

import tempita

class IterCode(HunkOCode):
    def __init__(self, features):
        self.features = features
        self.packer = DataPacker('iter_info')
        iterbody = self._iterbody()
        bodies = [self._xfbody(i,x) for i,x in enumerate(self.features.xforms)]
        bodies.append(iterbody)
        self.defs = '\n'.join(bodies)

    def _xfbody(self, xfid, xform):
        px = self.packer.view('info', 'xf%d_' % xfid)
        px.sub('xf', 'cp.xforms[%d]' % xfid)

        tmpl = tempita.Template("""
__device__
void apply_xf{{xfid}}(float *ix, float *iy, float *icolor,
                      const iter_info *info) {
    float tx, ty, ox = *ix, oy = *iy;
    {{apply_affine('ox', 'oy', 'tx', 'ty', px, 'xf.c', 'pre')}}

    ox = 0;
    oy = 0;

    {{for v in xform.vars}}
    if (1) {
        float w = {{px.get('xf.var[%d]' % v)}};
        {{variations.var_code[variations.var_nos[v]]()}}
    }
    {{endfor}}

    *ix = ox;
    *iy = oy;

    float csp = {{px.get('xf.color_speed')}};
    *icolor = *icolor * (1.0f - csp) + {{px.get('xf.color')}} * csp;
};
""")
        g = dict(globals())
        g.update(locals())
        return tmpl.substitute(g)

    def _iterbody(self):
        tmpl = tempita.Template("""
__global__
void iter(mwc_st *msts, const iter_info *infos, float *accbuf, float *denbuf) {
    mwc_st rctx = msts[gtid()];
    const iter_info *info = &(infos[blockIdx.x]);

    int consec_bad = -{{features.fuse}};
    int nsamps = 500;

    float x, y, color;
    x = mwc_next_11(&rctx);
    y = mwc_next_11(&rctx);
    color = mwc_next_01(&rctx);

    while (nsamps > 0) {
        float xfsel = mwc_next_01(&rctx);

        {{for xfid, xform in enumerate(features.xforms)}}
        if (xfsel <= {{packer.get('cp.norm_density[%d]' % xfid)}}) {
            apply_xf{{xfid}}(&x, &y, &color, info);
        } else
        {{endfor}}
        {
            denbuf[0] = xfsel;
            break; // TODO: fail here
        }

        if (consec_bad < 0) {
            consec_bad++;
            continue;
        }

        nsamps--;

        if (x <= -1.0f || x >= 1.0f || y <= -1.0f || y >= 1.0f) {
            consec_bad++;
            if (consec_bad > {{features.max_oob}}) {
                x = mwc_next_11(&rctx);
                y = mwc_next_11(&rctx);
                color = mwc_next_01(&rctx);
                consec_bad = -{{features.fuse}};
            }
            continue;
        }

        // TODO: dither?
        int i = ((int)((y + 1.0f) * 255.0f) * 512)
              +  (int)((x + 1.0f) * 255.0f);
        accbuf[i*4]     += color < 0.5f ? (1.0f - 2.0f * color) : 0.0f;
        accbuf[i*4+1]   += 1.0f - 2.0f * fabsf(0.5f - color);
        accbuf[i*4+2]   += color > 0.5f ? color * 2.0f - 1.0f : 0.0f;
        accbuf[i*4+3]   += 1.0f;

        denbuf[i] += 1.0f;

    }
}
""")
        return tmpl.substitute(
                features = self.features,
                packer = self.packer.view('info'))


def silly(features, cp):
    abuf = np.zeros((512, 512, 4), dtype=np.float32)
    dbuf = np.zeros((512, 512), dtype=np.float32)
    seeds = mwc.MWC.make_seeds(512 * 24)

    iter = IterCode(features)
    code = assemble_code(BaseCode, mwc.MWC, iter, iter.packer)
    print code
    mod = SourceModule(code, options=['-use_fast_math'], keep=True)

    info = iter.packer.pack(cp=cp)
    print info

    fun = mod.get_function("iter")
    fun(InOut(seeds), In(info), InOut(abuf), InOut(dbuf),
        block=(512,1,1), grid=(1,1), time_kernel=True)

    return abuf, dbuf

