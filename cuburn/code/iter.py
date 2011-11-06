"""
The main iteration loop.
"""

from cuburn.code import mwc, variations, interp
from cuburn.code.util import *

def precalc_densities(pcp, std_xforms):
    # This pattern recurs a few times for precalc segments. Unfortunately,
    # namespace stuff means it's not easy to functionalize this boilerplate
    pre_cp = pcp._precalc()
    pre_cp._code(Template(r"""

        float sum = 0.0f;

        {{for n in std_xforms}}
        float den_{{n}} = {{pre_cp.xforms[n].density}};
        sum += den_{{n}};
        {{endfor}}

        float rsum = 1.0f / sum;
        sum = 0.0f;

        {{for n in std_xforms[:-1]}}
        sum += den_{{n}} * rsum;
        {{pre_cp._set('den_' + n)}} = sum;
        {{endfor}}
    """).substitute(locals()))

def precalc_chaos(pcp, std_xforms):
    pre_cp = pcp._precalc()
    pre_cp._code(Template("""

        float sum, rsum;

        {{for p in std_xforms}}
        sum = 0.0f;

        {{for n in std_xforms}}
        float den_{{p}}_{{n}} = {{pre_cp.xforms[p].chaos[n]}};
        sum += den_{{p}}_{{n}};
        {{endfor}}

        rsum = 1.0f / sum;
        sum = 0.0f;

        {{for n in std_xforms[:-1]}}
        sum += den_{{p}}_{{n}} * rsum;
        {{pre_cp._set('chaos_%s_%s' % (p, n))}} = sum;
        {{endfor}}

        {{endfor}}

    """).substitute(locals()))

def precalc_camera(info, pcam):
    pre_cam = pcam._precalc()

    # Maxima code to check my logic:
    #   matrix([1,0,0.5*width + g],[0,1,0.5*height+g],[0,0,1])
    # . matrix([width * scale,0,0], [0,width * scale,0], [0,0,1])
    # . matrix([cosr,-sinr,0], [sinr,cosr,0], [0,0,1])
    # . matrix([1,0,-cenx],[0,1,-ceny],[0,0,1])
    # . matrix([X],[Y],[1]);

    pre_cam._code(Template(r"""

        float rot = {{pre_cam.rotation}} * M_PI / 180.0f;
        float rotsin = sin(rot), rotcos = cos(rot);
        float cenx = {{pre_cam.center.x}}, ceny = {{pre_cam.center.y}};
        float scale = {{pre_cam.scale}} * {{info.width}};

        float ditherwidth = {{pre_cam.dither_width}} * 0.33f;
        float u0 = mwc_next_01(rctx);
        float r = ditherwidth * sqrtf(-2.0f * log2f(u0) / M_LOG2E);

        float u1 = 2.0f * M_PI * mwc_next_01(rctx);
        float ditherx = r * cos(u1);
        float dithery = r * sin(u1);

        {{pre_cam._set('xx')}} = scale * rotcos;
        {{pre_cam._set('xy')}} = scale * -rotsin;
        {{pre_cam._set('xo')}} = scale * (rotsin * ceny - rotcos * cenx)
                              + {{0.5 * (info.width + info.gutter + 1)}} + ditherx;

        {{pre_cam._set('yx')}} = scale * rotsin;
        {{pre_cam._set('yy')}} = scale * rotcos;
        {{pre_cam._set('yo')}} = scale * -(rotsin * cenx + rotcos * ceny)
                              + {{0.5 * (info.height + info.gutter + 1)}} + dithery;

    """).substitute(locals()))

def precalc_xf_affine(px):
    pre = px._precalc()
    pre._code(Template(r"""

        float pri = {{pre.angle}} * M_PI / 180.0f;
        float spr = {{pre.spread}} * M_PI / 180.0f;

        float magx = {{pre.magnitude.x}};
        float magy = {{pre.magnitude.y}};

        {{pre._set('xx')}} = magx * cos(pri-spr);
        {{pre._set('xy')}} = magx * sin(pri-spr);
        {{pre._set('yx')}} = magy * cos(pri+spr);
        {{pre._set('yy')}} = magy * sin(pri+spr);
        {{pre._set('xo')}} = {{pre.offset.x}};
        {{pre._set('yo')}} = {{pre.offset.y}};

    """).substitute(locals()))

class IterCode(HunkOCode):
    # The number of threads per block
    NTHREADS = 256

    def __init__(self, info):
        self.info = info
        self.packer = interp.GenomePacker('iter_params')
        self.pcp = self.packer.view('params', self.info.genome, 'cp')

        iterbody = self._iterbody()
        bodies = [self._xfbody(i,x) for i,x in sorted(info.genome.xforms.items())]
        bodies.append(iterbody)
        self.defs = '\n'.join(bodies)
        self.decls += self.pix_helpers.substitute(info=info)

    decls = """
// Note: for normalized lookups, uchar4 actually returns floats
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> palTex;
__shared__ iter_params params;
__device__ int rb_head, rb_tail, rb_size;

"""

    pix_helpers = Template("""
__device__
void read_pix(float4 &pix, float &den) {
    den = pix.w;
    {{if info.pal_has_alpha}}
    read_half(pix.z, pix.w, pix.z, den);
    {{endif}}
}

__device__
void write_pix(float4 &pix, float den) {
    {{if info.pal_has_alpha}}
    write_half(pix.z, pix.z, pix.w, den);
    {{endif}}
    pix.w = den;
}

__device__
void update_pix(uint64_t ptr, uint32_t i, float4 c) {
    {{if info.pal_has_alpha}}
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
        px = self.pcp.xforms[xfid]
        tmpl = Template(r"""
__device__
void apply_xf_{{xfid}}(float &ox, float &oy, float &color, mwc_st &rctx) {
    float tx, ty;

    {{precalc_xf_affine(px.affine)}}
    {{apply_affine('ox', 'oy', 'tx', 'ty', px.affine)}}

    ox = 0;
    oy = 0;

    {{for name in xform.variations}}
    if (1) {
    {{py:pv = px.variations[name]}}
    float w = {{pv.weight}};
    {{variations.var_code[name].substitute(locals())}}
    }
    {{endfor}}

    {{if 'post' in xform}}
    tx = ox;
    ty = oy;
    {{precalc_xf_affine(px.post)}}
    {{apply_affine('tx', 'ty', 'ox', 'oy', px.post)}}
    {{endif}}

    float csp = {{px.color_speed}};
    color = color * (1.0f - csp) + {{px.color}} * csp;
};
""")
        g = dict(globals())
        g.update(locals())
        return tmpl.substitute(g)

    def _iterbody(self):
        tmpl = Template(r'''

__global__ void reset_rb(int size) {
    rb_head = rb_tail = 0;
    rb_size = size;
}

__global__
void iter(uint64_t accbuf_ptr, mwc_st *msts, float4 *points,
          const iter_params *all_params, int nsamps_to_generate) {
    const iter_params *global_params = &(all_params[blockIdx.x]);

    __shared__ int nsamps;
    nsamps = nsamps_to_generate;

    __shared__ float time_frac;
    time_frac = blockIdx.x / (float) gridDim.x;

    // load params to shared memory cooperatively
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i * 4 < sizeof(iter_params); i += blockDim.x * blockDim.y)
        reinterpret_cast<float*>(&params)[i] =
            reinterpret_cast<const float*>(global_params)[i];

    __shared__ int rb_idx;
    if (threadIdx.x == 1 && threadIdx.y == 1)
        rb_idx = 32 * blockDim.y * (atomicAdd(&rb_head, 1) % rb_size);

    __syncthreads();
    int this_rb_idx = rb_idx + threadIdx.x + 32 * threadIdx.y;
    mwc_st rctx = msts[this_rb_idx];
    float4 old_point = points[this_rb_idx];
    float x = old_point.x, y = old_point.y,
          color = old_point.z, fuse_rounds = old_point.w;

{{if info.chaos_used}}
    int last_xf_used = 0;
{{else}}
    // Shared memory size can be reduced by a factor of four using a slower
    // 4-stage reduce, but on Fermi hardware shmem use isn't a bottleneck
    __shared__ float swap[{{4*NTHREADS}}];
    __shared__ float cosel[{{NWARPS}}];

    // This is normally done after the swap-sync in the main loop
    if (threadIdx.y == 0 && threadIdx.x < {{NWARPS}})
        cosel[threadIdx.x] = mwc_next_01(rctx);
    __syncthreads();
{{endif}}


    while (1) {
        // This condition checks for large numbers, Infs, and NaNs.
        if (!(-(fabsf(x) + fabsf(y) > -1.0e6f))) {
            x = mwc_next_11(rctx);
            y = mwc_next_11(rctx);
            color = mwc_next_01(rctx);
            fuse_rounds = {{info.fuse / 32}};
        }

        // 32 rounds is somewhat arbitrary, but it has a pleasing 32-ness
        for (int i = 0; i < 32; i++) {

{{if info.chaos_used}}

            {{precalc_chaos(pcp, std_xforms)}}

            // For now, we don't attempt to use the swap buffer when chaos is used
            float xfsel = mwc_next_01(rctx);

            {{for prior_xform_idx, prior_xform_name in enumerate(std_xforms)}}
            if (last_xf_used == {{prior_xform_idx}}) {
                {{for xform_idx, xform_name in enumerate(std_xforms[:-1])}}
                if (xfsel <= {{pcp['chaos_'+prior_xform_name+'_'+xform_name]}}) {
                    apply_xf_{{xform_name}}(x, y, color, rctx);
                    last_xf_used = {{xform_idx}};
                } else
                {{endfor}}
                {
                    apply_xf_{{std_xforms[-1]}}(x, y, color, rctx);
                    last_xf_used = {{len(std_xforms)-1}};
                }
            } else
            {{endfor}}
            {
                printf("Something went *very* wrong.\n");
                asm("trap;");
            }

{{else}}
            {{precalc_densities(pcp, std_xforms)}}
            float xfsel = cosel[threadIdx.y];

            {{for xform_name in std_xforms[:-1]}}
            if (xfsel <= {{pcp['den_'+xform_name]}}) {
                apply_xf_{{xform_name}}(x, y, color, rctx);
            } else
            {{endfor}}
                apply_xf_{{std_xforms[-1]}}(x, y, color, rctx);

            int sw = (threadIdx.y * 32 + threadIdx.x * 33) & {{NTHREADS-1}};
            int sr = threadIdx.y * 32 + threadIdx.x;

            swap[sw] = fuse_rounds;
            swap[sw+{{NTHREADS}}] = x;
            swap[sw+{{2*NTHREADS}}] = y;
            swap[sw+{{3*NTHREADS}}] = color;
            __syncthreads();

            // We select the next xforms here, since we've just synced.
            if (threadIdx.y == 0 && threadIdx.x < {{NWARPS}})
                cosel[threadIdx.x] = mwc_next_01(rctx);

            fuse_rounds = swap[sr];
            x = swap[sr+{{NTHREADS}}];
            y = swap[sr+{{2*NTHREADS}}];
            color = swap[sr+{{3*NTHREADS}}];

{{endif}}

            if (fuse_rounds > 0.0f) continue;

{{if 'final' in cp.xforms}}
            float fx = x, fy = y, fcolor = color;
            apply_xf_final(fx, fy, fcolor, rctx);
{{endif}}

            float cx, cy, cc;

            {{precalc_camera(info, pcp.camera)}}

{{if 'final' in cp.xforms}}
            {{apply_affine('fx', 'fy', 'cx', 'cy', pcp.camera)}}
            cc = fcolor;
{{else}}
            {{apply_affine('x', 'y', 'cx', 'cy', pcp.camera)}}
            cc = color;
{{endif}}

            uint32_t ix = trunca(cx), iy = trunca(cy);

            if (ix >= {{info.acc_width}} || iy >= {{info.acc_height}})
                continue;

            uint32_t i = iy * {{info.acc_stride}} + ix;

            float4 outcol = tex2D(palTex, cc, time_frac);
            update_pix(accbuf_ptr, i, outcol);
        }

        int num_okay = __popc(__ballot(fuse_rounds == 0.0f));
        // Some xforms give so many badvals that a thread is almost guaranteed
        // to hit another badval before the fuse is over, causing the card to
        // spin forever. To avoid this, we count a fuse round as 1/4 of a
        // sample below.
        if (threadIdx.x == 0) atomicSub(&nsamps, 256 + num_okay * 24);
        fuse_rounds = fmaxf(0.0f, fuse_rounds - 1.0f);

        __syncthreads();
        if (nsamps <= 0) break;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0)
        rb_idx = 32 * blockDim.y * (atomicAdd(&rb_tail, 1) % rb_size);
    __syncthreads();
    this_rb_idx = rb_idx + threadIdx.x + 32 * threadIdx.y;

    points[this_rb_idx] = make_float4(x, y, color, fuse_rounds);
    msts[this_rb_idx] = rctx;
    return;
}
''')
        return tmpl.substitute(
                info = self.info,
                cp = self.info.genome,
                pcp = self.pcp,
                NTHREADS = self.NTHREADS,
                NWARPS = self.NTHREADS / 32,
                std_xforms = [n for n in sorted(self.info.genome.xforms)
                              if n != 'final'],
                **globals())

