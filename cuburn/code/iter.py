"""
The main iteration loop.
"""

from cuburn.code import mwc, variations
from cuburn.code.util import *

class IterCode(HunkOCode):
    # The number of threads per block
    NTHREADS = 512

    def __init__(self, features):
        self.features = features
        self.packer = DataPacker('iter_info')
        iterbody = self._iterbody()
        bodies = [self._xfbody(i,x) for i,x in enumerate(self.features.xforms)]
        bodies.append(iterbody)
        self.defs = '\n'.join(bodies)
        self.decls += self.pix_helpers.substitute(features=features)

    decls = """
// Note: for normalized lookups, uchar4 actually returns floats
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> palTex;
__shared__ iter_info info;

"""

    pix_helpers = Template("""
__device__
void read_pix(float4 &pix, float &den) {
    den = pix.w;
    {{if features.pal_has_alpha}}
    read_half(pix.z, pix.w, pix.z, den);
    {{endif}}
}

__device__
void write_pix(float4 &pix, float den) {
    {{if features.pal_has_alpha}}
    write_half(pix.z, pix.z, pix.w, den);
    {{endif}}
    pix.w = den;
}
""")

    def _xfbody(self, xfid, xform):
        px = self.packer.view('info', 'xf%d_' % xfid)
        px.sub('xf', 'cp.xforms[%d]' % xfid)

        tmpl = Template("""
__device__
void apply_xf{{xfid}}(float &ox, float &oy, float &color, mwc_st &rctx) {
    float tx, ty;

    {{apply_affine_flam3('ox', 'oy', 'tx', 'ty', px, 'xf.c', 'pre')}}

    ox = 0;
    oy = 0;

    {{for v in xform.vars}}
    {{if variations.var_nos[v].startswith('pre_')}}
    if (1) {
        float w = {{px.get('xf.var[%d]' % v)}};
        {{variations.var_code[variations.var_nos[v]].substitute(locals())}}
    }
    {{endif}}
    {{endfor}}

    {{for v in xform.vars}}
    {{if not variations.var_nos[v].startswith('pre_')}}
    if (1) {
        float w = {{px.get('xf.var[%d]' % v)}};
        {{variations.var_code[variations.var_nos[v]].substitute(locals())}}
    }
    {{endif}}
    {{endfor}}

    float csp = {{px.get('xf.color_speed')}};
    color = color * (1.0f - csp) + {{px.get('xf.color')}} * csp;
};
""")
        g = dict(globals())
        g.update(locals())
        return tmpl.substitute(g)

    def _iterbody(self):
        tmpl = Template(r'''
__global__
void iter(mwc_st *msts, iter_info *infos, float4 *accbuf, float *denbuf) {
    __shared__ int nsamps;
    mwc_st rctx = msts[gtid()];
    iter_info *info_glob = &(infos[blockIdx.x]);

    // load info to shared memory cooperatively
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i * 4 < sizeof(iter_info); i += blockDim.x * blockDim.y)
        reinterpret_cast<float*>(&info)[i] =
            reinterpret_cast<float*>(info_glob)[i];

    if (threadIdx.y == 0 && threadIdx.x == 0)
        nsamps = {{packer.get("cp.width * cp.height / cp.ntemporal_samples * cp.adj_density")}};

    {{if features.chaos_used}}
    int last_xf_used = 0;
    {{else}}
    // Size can be reduced by a factor of four using a slower 4-stage reduce
    __shared__ float swap[2048];
    __shared__ float cosel[16];
    if (threadIdx.y == 0 && threadIdx.x < 16)
        cosel[threadIdx.x] = mwc_next_01(rctx);
    {{endif}}

    if (threadIdx.y == 1 && threadIdx.x == 0) {
        float ditherwidth = {{packer.get("0.33 * cp.spatial_filter_radius")}};
        float u0 = mwc_next_01(rctx);
        float r = ditherwidth * sqrt(-2.0f * log2f(u0) / M_LOG2E);

        float u1 = 2.0f * M_PI * mwc_next_01(rctx);
        info.cam_xo += r * cos(u1);
        info.cam_yo += r * sin(u1);
    }

    __syncthreads();
    int consec_bad = -{{features.fuse}};

    float x, y, color;
    x = mwc_next_11(rctx);
    y = mwc_next_11(rctx);
    color = mwc_next_01(rctx);

    while (1) {
        {{if features.chaos_used}}
        // For now, we can't use the swap buffer with chaos enabled
        float xfsel = mwc_next_01(rctx);
        {{else}}
        float xfsel = cosel[threadIdx.y];
        {{endif}}

        {{if features.chaos_used}}
        {{for density_row_idx, prior_xform_idx in enumerate(features.std_xforms)}}
        {{for density_col_idx,  this_xform_idx in enumerate(features.std_xforms)}}
        if (last_xf_used == {{prior_xform_idx}} &&
                xfsel <= {{packer.get("cp.chaos_densities[%d][%d]" % (density_row_idx, density_col_idx))}}) {
            apply_xf{{this_xform_idx}}(x, y, color, rctx);
            last_xf_used = {{this_xform_idx}};
        } else
        {{endfor}}
        {{endfor}}
        {{else}}
        {{for density_col_idx, this_xform_idx in enumerate(features.std_xforms)}}
        if (xfsel <= {{packer.get("cp.norm_density[%d]" % (density_col_idx))}}) {
            apply_xf{{this_xform_idx}}(x, y, color, rctx);
        } else
        {{endfor}}
        {{endif}}
        {
            printf("Reached trap, aborting execution! %g (%d,%d,%d)\n",
                   xfsel, blockIdx.x, threadIdx.y, threadIdx.x);
            asm volatile ("trap;");
        }

        {{if not features.chaos_used}}
        // Swap thread states here so that writeback skipping logic doesn't die
        int sw = (threadIdx.y * 32 + threadIdx.x * 33) & 0x1ff;
        int sr = threadIdx.y * 32 + threadIdx.x;

        swap[sw] = consec_bad;
        swap[sw+512] = x;
        swap[sw+1024] = y;
        swap[sw+1536] = color;
        __syncthreads();
        // This is in the middle of the function so that only one sync is
        // required per loop.
        if (nsamps < 0) break;

        {{if not features.chaos_used}}
        // Similarly, we select the next xforms here.
        if (threadIdx.y == 0 && threadIdx.x < 16)
            cosel[threadIdx.x] = mwc_next_01(rctx);
        {{endif}}

        consec_bad = swap[sr];
        x = swap[sr+512];
        y = swap[sr+1024];
        color = swap[sr+1536];
        {{endif}}

        if (consec_bad < 0) {
            consec_bad++;
            continue;
        }

        int remain = __popc(__ballot(1));
        if (threadIdx.x == 0) atomicSub(&nsamps, remain);

        {{if features.final_xform_index}}
        float fx = x, fy = y, fcolor;
        apply_xf{{features.final_xform_index}}(fx, fy, fcolor, rctx);
        {{endif}}

        // TODO: this may not optimize well, verify.
        float cx, cy;
        {{if features.final_xform_index}}
        {{apply_affine('fx', 'fy', 'cx', 'cy', packer,
                       'cp.camera_transform', 'cam')}}
        {{else}}
        {{apply_affine('x', 'y', 'cx', 'cy', packer,
                       'cp.camera_transform', 'cam')}}
        {{endif}}
        uint32_t ix = trunca(cx), iy = trunca(cy);

        if (ix >= {{features.acc_width}} || iy >= {{features.acc_height}} ) {
            consec_bad++;
            if (consec_bad > {{features.max_oob}}) {
                x = mwc_next_11(rctx);
                y = mwc_next_11(rctx);
                color = mwc_next_01(rctx);
                consec_bad = -{{features.fuse}};
            }
            continue;
        }

        uint32_t i = iy * {{features.acc_stride}} + ix;

        float4 outcol = tex2D(palTex, color, {{packer.get("cp_step_frac")}});
        float4 pix = accbuf[i];
        float den;
        // TODO: unify read/write_pix cycle when alpha is needed
        read_pix(pix, den);
        pix.x += outcol.x;
        pix.y += outcol.y;
        pix.z += outcol.z;
        pix.w += outcol.w;
        den += 1.0f;

        write_pix(pix, den);
        accbuf[i] = pix;
    }
    msts[gtid()] = rctx;
}
''')
        return tmpl.substitute(
                features = self.features,
                packer = self.packer.view('info'),
                **globals())

