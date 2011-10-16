"""
The main iteration loop.
"""

from cuburn.code import mwc, variations
from cuburn.code.util import *

class IterCode(HunkOCode):
    # The number of threads per block
    NTHREADS = 256

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

__device__
void update_pix(uint64_t ptr, uint32_t i, float4 c) {
    {{if features.pal_has_alpha}}
    asm volatile ({{crep('''
    {
        .reg .u16       sz, sw;
        .reg .u64       base, off;
        .reg .f32       x, y, z, w, den, rc, tz, tw;

        // TODO: this limits the accumulation buffer to <4GB
        shl.b32         %0,     %0,     4;
        cvt.u64.u32     off,    %0;
        add.u64         base,   %1,     off;
        ld.cg.v4.f32    {x, y, z, den},         [base];
        add.f32         x,      x,      %2;
        add.f32         y,      y,      %3;
        mov.b32         {sz, sw},       z;
        cvt.rn.f32.u16  tz,     sz;
        cvt.rn.f32.u16  tw,     sw;
        mul.f32         tz,     tz,     den;
        mul.f32         tw,     tz,     den;
        fma.f32         tz,     %4,     65535.0,    tz;
        fma.f32         tw,     %5,     65535.0,    tw;
        add.f32         den,    1.0;
        rcp.approx.f32  rc,     den;
        mul.f32         tz,     tz,     rc;
        mul.f32         tw,     tw,     rc;
        cvt.rni.u16.f32 sz,     tz;
        cvt.rni.u16.f32 sw,     tw;
        mov.b32         z,      {sz, sw};
        st.cs.v4.f32    [base], {x, y, z, den};
    }
    ''')}} : "+r"(i) : "l"(ptr), "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
    {{else}}
    asm volatile ({{crep('''
    {
        .reg .u64       base, off;
        .reg .f32       x, y, z, den;

        // TODO: this limits the accumulation buffer to <4GB
        shl.b32         %0,     %0,     4;
        cvt.u64.u32     off,    %0;
        add.u64         base,   %1,     off;
        ld.cg.v4.f32    {x, y, z, den},         [base];
        add.f32         x,      x,      %2;
        add.f32         y,      y,      %3;
        add.f32         z,      z,      %4;
        add.f32         den,    den,    1.0;
        st.cs.v4.f32    [base], {x, y, z, den};
    }
    ''')}} : "+r"(i) : "l"(ptr), "f"(c.x), "f"(c.y), "f"(c.z));
    {{endif}}
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
    if (1) {
        float w = {{px.get('xf.var[%d]' % v)}};
        {{variations.var_code[variations.var_nos[v]].substitute(locals())}}
    }
    {{endfor}}

    {{endif}}

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
void iter(mwc_st *msts, iter_info *infos, uint64_t accbuf_ptr) {
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
    __shared__ float swap[{{4*NTHREADS}}];
    __shared__ float cosel[{{NWARPS}}];
    if (threadIdx.y == 0 && threadIdx.x < {{NWARPS}})
        cosel[threadIdx.x] = mwc_next_01(rctx);
    {{endif}}

    __syncthreads();
    int consec_bad = -{{features.fuse}};

    if (threadIdx.y == 1 && threadIdx.x == 0) {
        float ditherwidth = {{packer.get("0.33 * cp.spatial_filter_radius")}};
        float u0 = mwc_next_01(rctx);
        float r = ditherwidth * sqrtf(-2.0f * log2f(u0) / M_LOG2E);

        float u1 = 2.0f * M_PI * mwc_next_01(rctx);
        info.cam_xo += r * cos(u1);
        info.cam_yo += r * sin(u1);
    }

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
        int sw = (threadIdx.y * 32 + threadIdx.x * 33) & {{NTHREADS-1}};
        int sr = threadIdx.y * 32 + threadIdx.x;

        swap[sw] = consec_bad;
        swap[sw+{{NTHREADS}}] = x;
        swap[sw+{{2*NTHREADS}}] = y;
        swap[sw+{{3*NTHREADS}}] = color;
        __syncthreads();
        // This is in the middle of the function so that only one sync is
        // required per loop.
        if (nsamps < 0) break;

        {{if not features.chaos_used}}
        // Similarly, we select the next xforms here.
        if (threadIdx.y == 0 && threadIdx.x < {{NWARPS}})
            cosel[threadIdx.x] = mwc_next_01(rctx);
        {{endif}}

        consec_bad = swap[sr];
        x = swap[sr+{{NTHREADS}}];
        y = swap[sr+{{2*NTHREADS}}];
        color = swap[sr+{{3*NTHREADS}}];
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
        update_pix(accbuf_ptr, i, outcol);

    }
    msts[gtid()] = rctx;
}
''')
        return tmpl.substitute(
                features = self.features,
                packer = self.packer.view('info'),
                NTHREADS = self.NTHREADS,
                NWARPS = self.NTHREADS / 32,
                **globals())

