"""
The main iteration loop.
"""

from ctypes import byref, memset, sizeof

import pycuda.driver as cuda
from pycuda.driver import In, Out, InOut
from pycuda.compiler import SourceModule
import numpy as np

from fr0stlib.pyflam3 import flam3_interpolate
from cuburn.code import mwc, variations, filter
from cuburn.code.util import *
from cuburn.render import Genome

class IterCode(HunkOCode):
    def __init__(self, features):
        self.features = features
        self.packer = DataPacker('iter_info')
        iterbody = self._iterbody()
        bodies = [self._xfbody(i,x) for i,x in enumerate(self.features.xforms)]
        bodies.append(iterbody)
        self.defs = '\n'.join(bodies)

    decls = """
// Note: for normalized lookups, uchar4 actually returns floats
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> palTex;
__shared__ iter_info info;
"""

    def _xfbody(self, xfid, xform):
        px = self.packer.view('info', 'xf%d_' % xfid)
        px.sub('xf', 'cp.xforms[%d]' % xfid)

        tmpl = Template("""
__device__
void apply_xf{{xfid}}(float *ix, float *iy, float *icolor, mwc_st *rctx) {
    float tx, ty, ox = *ix, oy = *iy;
    {{apply_affine_flam3('ox', 'oy', 'tx', 'ty', px, 'xf.c', 'pre')}}

    ox = 0;
    oy = 0;

    {{for v in xform.vars}}
    if (1) {
        float w = {{px.get('xf.var[%d]' % v)}};
        {{variations.var_code[variations.var_nos[v]].substitute(locals())}}
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
        tmpl = Template(r"""
__global__
void iter(mwc_st *msts, iter_info *infos, float4 *accbuf, float *denbuf) {
    mwc_st rctx = msts[gtid()];
    iter_info *info_glob = &(infos[blockIdx.x]);

    // load info to shared memory cooperatively
    for (int i = threadIdx.y * 32 + threadIdx.x;
         i * 4 < sizeof(iter_info); i += blockDim.x * blockDim.y)
        reinterpret_cast<float*>(&info)[i] =
            reinterpret_cast<float*>(info_glob)[i];

    int consec_bad = -{{features.fuse}};
    // TODO: make nsteps adjustable via genome
    int nsamps = {{packer.get('cp.width * cp.height / 512000. * cp.adj_density')}};

    float x, y, color;
    x = mwc_next_11(&rctx);
    y = mwc_next_11(&rctx);
    color = mwc_next_01(&rctx);

    while (nsamps > 0) {
        float xfsel = mwc_next_01(&rctx);

        {{for xfid, xform in enumerate(features.xforms)}}
        {{if xfid != features.final_xform_index}}
        if (xfsel <= {{packer.get('cp.norm_density[%d]' % xfid)}}) {
            apply_xf{{xfid}}(&x, &y, &color, &rctx);
        } else
        {{endif}}
        {{endfor}}
        {
            denbuf[0] = xfsel;
            break; // TODO: fail here
        }
        {{if features.final_xform_index}}
        float fx = x, fy = y, fcolor;
        apply_xf{{features.final_xform_index}}(&fx, &fy, &fcolor, &rctx);
        {{endif}}

        if (consec_bad < 0) {
            consec_bad++;
            continue;
        }

        nsamps--;

        // TODO: this may not optimize well, verify.

        float cx, cy;
        {{if features.final_xform_index}}
        {{apply_affine('fx', 'fy', 'cx', 'cy', packer,
                       'cp.camera_transform', 'cam')}}
        {{else}}
        {{apply_affine('x', 'y', 'cx', 'cy', packer,
                       'cp.camera_transform', 'cam')}}
        {{endif}}

        float ditherwidth = {{packer.get('0.5 * cp.spatial_filter_radius')}};
        float ditherx = mwc_next_11(&rctx) * ditherwidth;
        float dithery = mwc_next_11(&rctx) * ditherwidth;

        int ix = trunca(cx+ditherx), iy = trunca(cy+dithery);

        if (ix < 0 || ix >= {{features.width}} ||
            iy < 0 || iy >= {{features.height}} ) {
            consec_bad++;
            if (consec_bad > {{features.max_oob}}) {
                x = mwc_next_11(&rctx);
                y = mwc_next_11(&rctx);
                color = mwc_next_01(&rctx);
                consec_bad = -{{features.fuse}};
            }
            continue;
        }

        int i = iy * {{features.width}} + ix;

        float4 outcol = tex2D(palTex, color, {{packer.get('cp_step_frac')}});
        float4 pix = accbuf[i];
        pix.x += outcol.x;
        pix.y += outcol.y;
        pix.z += outcol.z;
        pix.w += outcol.w;
        accbuf[i] = pix;    // TODO: atomic operations (or better)
        denbuf[i] += 1.0f;
    }
}
""")
        return tmpl.substitute(
                features = self.features,
                packer = self.packer.view('info'),
                **globals())

def render(features, cps):
    # TODO: make this adjustable via genome
    nsteps = 1000
    abuf = np.zeros((features.height, features.width, 4), dtype=np.float32)
    dbuf = np.zeros((features.height, features.width), dtype=np.float32)
    seeds = mwc.MWC.make_seeds(512 * nsteps)

    iter = IterCode(features)
    de = filter.DensityEst(features, cps[0])
    code = assemble_code(BaseCode, mwc.MWC, iter.packer, iter,
                         filter.ColorClip, de)

    for lno, line in enumerate(code.split('\n')):
        print '%3d %s' % (lno, line)
    mod = SourceModule(code,
            options=['-use_fast_math', '-maxrregcount', '32'])

    cps_as_array = (Genome * len(cps))()
    for i, cp in enumerate(cps):
        cps_as_array[i] = cp

    infos = []
    pal = np.empty((16, 256, 4), dtype=np.uint8)

    # TODO: move this into a common function
    if len(cps) > 1:
        cp = Genome()
        memset(byref(cp), 0, sizeof(cp))

        sampAt = [int(i/15.*(nsteps-1)) for i in range(16)]
        for n in range(nsteps):
            flam3_interpolate(cps_as_array, 2, float(n)/nsteps - 0.5,
                              0, byref(cp))
            cp._init()
            if n in sampAt:
                pidx = sampAt.index(n)
                for i, e in enumerate(cp.palette.entries):
                    pal[pidx][i] = np.uint8(np.array(e.color) * 255.0)
            infos.append(iter.packer.pack(cp=cp, cp_step_frac=float(n)/nsteps))
    else:
        for i, e in enumerate(cps[0].palette.entries):
            pal[0][i] = np.uint8(np.array(e.color) * 255.0)
        pal[1:] = pal[0]
        infos.append(iter.packer.pack(cp=cps[0], cp_step_frac=0))
        infos *= nsteps

    infos = np.concatenate(infos)

    dpal = cuda.make_multichannel_2d_array(pal, 'C')
    tref = mod.get_texref('palTex')
    tref.set_array(dpal)
    tref.set_format(cuda.array_format.UNSIGNED_INT8, 4)
    tref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    tref.set_filter_mode(cuda.filter_mode.LINEAR)

    abufd = cuda.to_device(abuf)
    dbufd = cuda.to_device(dbuf)

    fun = mod.get_function("iter")
    fun.set_cache_config(cuda.func_cache.PREFER_L1)
    t = fun(InOut(seeds), InOut(infos), abufd, dbufd,
        block=(32,16,1), grid=(nsteps,1), time_kernel=True)
    print "Completed render in %g seconds" % t

    f = np.float32

    npix = features.width * features.height

    obufd = cuda.to_device(abuf)
    de.invoke(mod, abufd, obufd, dbufd)

    fun = mod.get_function("colorclip")
    t = fun(obufd, f(1 / cp.gamma), f(cp.vibrancy), f(cp.highlight_power),
        block=(256,1,1), grid=(npix/256,1), time_kernel=True)
    print "Completed color filtering in %g seconds" % t

    abuf = cuda.from_device_like(obufd, abuf)
    return abuf, dbuf



# TODO: find a better place to stick this code
class MemBench(HunkOCode):
    decls = """
__shared__ uint32_t coord[512];
"""

    defs_tmpl = Template("""
__global__
void iter{{W}}(mwc_st *mwcs, uint32_t *buf) {
    mwc_st rctx = mwcs[gtid()];

    int mask = (1 << {{W}}) - 1;
    int smoff = threadIdx.x >> {{W}};
    int writer = (threadIdx.x & mask) == 0;

    for (int i = 0; i < 1024 * 32; i++) {
        if (writer)
            coord[smoff] = mwc_next(&rctx) & 0x7ffffff; // 512MB / 4 bytes
        __syncthreads();
        uint32_t *dst = buf + (coord[smoff] + (threadIdx.x & mask));
        uint32_t val = mwc_next(&rctx);
        asm("st.global.u32  [%0],   %1;" :: "l"(dst), "r"(val));
    }
}
""")

    @property
    def defs(self):
        return '\n'.join([self.defs_tmpl.substitute(W=w) for w in range(8)])

def membench():
    code = assemble_code(BaseCode, mwc.MWC, MemBench())
    mod = SourceModule(code)

    buf = cuda.mem_alloc(512 << 20)
    seeds = mwc.MWC.make_seeds(512 * 21)

    for w in range(8):
        fun = mod.get_function('iter%d' % w)
        print 'Launching with W=%d' % w
        t = fun(cuda.In(seeds), buf,
                block=(512, 1, 1), grid=(21, 1), time_kernel=True)
        print 'Completed in %g' % t

