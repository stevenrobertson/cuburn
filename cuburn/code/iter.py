"""
The main iteration loop.
"""

import pycuda.driver as cuda
from pycuda.driver import In, Out, InOut
from pycuda.compiler import SourceModule
import numpy as np

from cuburn import code
from cuburn.code import mwc

src = r"""
#define FUSE 20
#define MAXOOB 10

typedef struct {
    // Number of iterations to perform, *per thread*.
    uint32_t    niters;

    // Number of accumulators per row and column in the accum buffer
    uint32_t    accwidth, accheight;
} iter_info;

__global__
void silly(mwc_st *msts, iter_info *infos, float *accbuf, float *denbuf) {
    mwc_st rctx = msts[gtid()];
    iter_info *info = &(infos[blockIdx.x]);

    float consec_bad = -FUSE;
    float nsamps = info->niters;

    float x, y, color;
    x = mwc_next_11(&rctx);
    y = mwc_next_11(&rctx);
    color = mwc_next_01(&rctx);

    while (nsamps > 0.0f) {
        float xfsel = mwc_next_01(&rctx);

        x *= 0.5f;
        y *= 0.5f;
        color *= 0.5f;
        if (xfsel < 0.33f) {
            color += 0.25f;
            x += 0.5f;
        } else if (xfsel < 0.66f) {
            color += 0.5f;
            y += 0.5f;
        }

        if (consec_bad < 0.0f) {
            consec_bad++;
            continue;
        }

        if (x <= -1.0f || x >= 1.0f || y <= -1.0f || y >= 1.0f
            || consec_bad < 0.0f) {

            consec_bad++;
            if (consec_bad > MAXOOB) {
                x = mwc_next_11(&rctx);
                y = mwc_next_11(&rctx);
                color = mwc_next_01(&rctx);
                consec_bad = -FUSE;
            }
            continue;
        }

        // TODO: dither?
        int i = ((int)((y + 1.0f) * 256.0f) * 512)
              +  (int)((x + 1.0f) * 256.0f);
        accbuf[i*4]     += color < 0.5f ? (1.0f - 2.0f * color) : 0.0f;
        accbuf[i*4+1]   += 1.0f - 2.0f * fabsf(0.5f - color);
        accbuf[i*4+2]   += color > 0.5f ? color * 2.0f - 1.0f : 0.0f;
        accbuf[i*4+3]   += 1.0f;

        denbuf[i] += 1.0f;

        nsamps--;
    }
}
"""

def silly():
    mod = SourceModule(code.base + mwc.src + src)
    abuf = np.zeros((512, 512, 4), dtype=np.float32)
    dbuf = np.zeros((512, 512), dtype=np.float32)
    seeds = mwc.build_mwc_seeds(512 * 24, seed=5)

    info = np.zeros(3, dtype=np.uint32)
    info[0] = 5000
    info[1] = 512
    info[2] = 512
    info = np.repeat([info], 24, axis=0)

    fun = mod.get_function("silly")
    fun(InOut(seeds), In(info), InOut(abuf), InOut(dbuf),
        block=(512,1,1), grid=(24,1), time_kernel=True)

    print abuf
    print dbuf
    print sum(dbuf)
    return abuf, dbuf

