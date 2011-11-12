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

    decls = """
// Note: for normalized lookups, uchar4 actually returns floats
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> palTex;
__shared__ iter_params params;
__device__ int rb_head, rb_tail, rb_size;

"""

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
void iter(
        uint64_t out_ptr,
        mwc_st *msts,
        float4 *points,
        const iter_params *all_params,
        int nsamps_to_generate
) {
    const iter_params *global_params = &(all_params[blockIdx.x]);

{{if info.acc_mode != 'deferred'}}
    __shared__ float time_frac;
    time_frac = blockIdx.x / (float) gridDim.x;
{{endif}}

    // load params to shared memory cooperatively
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < (sizeof(iter_params) / 4); i += blockDim.x * blockDim.y)
        reinterpret_cast<float*>(&params)[i] =
            reinterpret_cast<const float*>(global_params)[i];

    __shared__ int rb_idx;
    if (threadIdx.x == 1 && threadIdx.y == 1)
        rb_idx = 32 * blockDim.y * (atomicAdd(&rb_head, 1) % rb_size);

    __syncthreads();
    int this_rb_idx = rb_idx + threadIdx.x + 32 * threadIdx.y;
    mwc_st rctx = msts[this_rb_idx];

    // TODO: 4th channel unused. Kill or use for something helpful
    float4 old_point = points[this_rb_idx];
    float x = old_point.x, y = old_point.y, color = old_point.z;

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

    bool fuse = false;

    // This condition checks for large numbers, Infs, and NaNs.
    if (!(-(fabsf(x) + fabsf(y)) > -1.0e6f)) {
        x = mwc_next_11(rctx);
        y = mwc_next_11(rctx);
        color = mwc_next_01(rctx);
        fuse = true;
    }

    // TODO: link up with FUSE, etc
    for (int round = 0; round < 256; round++) {

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

        swap[sw] = fuse ? 1.0f : 0.0f;
        swap[sw+{{NTHREADS}}] = x;
        swap[sw+{{2*NTHREADS}}] = y;
        swap[sw+{{3*NTHREADS}}] = color;
        __syncthreads();

        // We select the next xforms here, since we've just synced.
        if (threadIdx.y == 0 && threadIdx.x < {{NWARPS}})
            cosel[threadIdx.x] = mwc_next_01(rctx);

        fuse = swap[sr];
        x = swap[sr+{{NTHREADS}}];
        y = swap[sr+{{2*NTHREADS}}];
        color = swap[sr+{{3*NTHREADS}}];

{{endif}}

{{if info.acc_mode == 'deferred'}}
        int tid = threadIdx.y * 32 + threadIdx.x;
        int offset = 4 * (256 * (256 * blockIdx.x + round) + tid);
        int *log = reinterpret_cast<int*>(out_ptr + offset);
{{endif}}

        if (fuse) {
{{if info.acc_mode == 'deferred'}}
            *log = 0xffffffff;
{{endif}}
            continue;
        }

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

        if (ix >= {{info.acc_width}} || iy >= {{info.acc_height}}) {
{{if info.acc_mode == 'deferred'}}
            *log = 0xffffffff;
{{endif}}
            continue;
        }

        uint32_t i = iy * {{info.acc_stride}} + ix;

{{if info.acc_mode == 'atomic'}}
        float4 outcol = tex2D(palTex, cc, time_frac);
        float *accbuf_f = reinterpret_cast<float*>(out_ptr + (16*i));
        atomicAdd(accbuf_f,   outcol.x);
        atomicAdd(accbuf_f+1, outcol.y);
        atomicAdd(accbuf_f+2, outcol.z);
        atomicAdd(accbuf_f+3, 1.0f);
{{elif info.acc_mode == 'global'}}
        float4 outcol = tex2D(palTex, cc, time_frac);
        float4 *accbuf = reinterpret_cast<float4*>(out_ptr + (16*i));
        float4 pix = *accbuf;
        pix.x += outcol.x;
        pix.y += outcol.y;
        pix.z += outcol.z;
        pix.w += 1.0f;
        *accbuf = pix;
{{elif info.acc_mode == 'deferred'}}
        // 'color' gets the top 9 bits. TODO: add dithering via precalc.
        uint32_t icolor = fminf(1.0f, cc) * 511.0f;
        asm("bfi.b32    %0, %1, %0, 23, 9;" : "+r"(i) : "r"(icolor));
        *log = i;
{{endif}}
    }

    if (threadIdx.x == 0 && threadIdx.y == 0)
        rb_idx = 32 * blockDim.y * (atomicAdd(&rb_tail, 1) % rb_size);
    __syncthreads();
    this_rb_idx = rb_idx + threadIdx.x + 32 * threadIdx.y;

    points[this_rb_idx] = make_float4(x, y, color, 0.0f);
    msts[this_rb_idx] = rctx;
    return;
}

// Block size, shared accumulation bits, shared accumulation width.
#define BS 1024
#define SHAB 12
#define SHAW (1<<SHAB)

// Read from the shm accumulators and write to the global ones.
__device__
void write_shmem_helper(
        float4 *acc,
        const int glo_idx,
        const uint32_t dr,
        const uint32_t gb
) {
    float4 pix = acc[glo_idx];
    pix.x += (dr & 0xffff) / 255.0f;
    pix.w += dr >> 16;
    pix.y += (gb & 0xffff) / 255.0f;
    pix.z += (gb >> 16) / 255.0f;
    acc[glo_idx] = pix;
}

// Read the point log, accumulate in shared memory, and write the results.
// This kernel is to be launched with one block for every 4,096 addresses to
// be processed, and will handle those addresses.
//
// log_bounds is an array mapping radix values to the first index in the log
// with that radix position. For performance reasons in other parts of the
// code, the radix may actually include bits within the lower SHAB part of the
// address, or it might not cover the first few bits after the SHAB part;
// log_bounds_shift covers that. glob_addr_bits specifies the number of bits
// above SHAB which are address bits.

__global__ void
__launch_bounds__(BS, 1)
write_shmem(
        float4 *acc,
        const uint32_t *log,
        const uint32_t *log_bounds,
        const int log_bounds_shift
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // These two accumulators, used in write_shmem, hold {density, red} and
    // {green, blue} values as packed u16 pairs. The fixed size represents
    // 4,096 pixels in the accumulator.
    __shared__ uint32_t s_acc_dr[SHAW];
    __shared__ uint32_t s_acc_gb[SHAW];

    // TODO: doesn't respect SHAW/BS
    // TODO: compare generated code with unrolled for-loop
    s_acc_dr[tid] = 0;
    s_acc_gb[tid] = 0;
    s_acc_dr[tid+BS] = 0;
    s_acc_gb[tid+BS] = 0;
    s_acc_dr[tid+2*BS] = 0;
    s_acc_gb[tid+2*BS] = 0;
    s_acc_dr[tid+3*BS] = 0;
    s_acc_gb[tid+3*BS] = 0;
    __syncthreads();

    // TODO: share across threads - discernable performance impact?
    int lb_idx_lo, lb_idx_hi;
    if (log_bounds_shift > 0) {
        lb_idx_hi = ((bid + 1) << log_bounds_shift) - 1;
        lb_idx_lo = (bid << log_bounds_shift) - 1;
    } else {
        lb_idx_hi = bid >> (-log_bounds_shift);
        lb_idx_lo = lb_idx_hi - 1;
    }

    int idx_lo, idx_hi;
    if (lb_idx_lo < 0) idx_lo = 0;
    else idx_lo = log_bounds[lb_idx_lo] & ~(BS-1);
    idx_hi = (log_bounds[lb_idx_hi] & ~(BS - 1)) + BS;

    float rnrounds = 1.0f / (idx_hi - idx_lo);
    float time = tid * rnrounds;
    float time_step = BS * rnrounds;

    int glo_base = bid << SHAB;

    for (int i = idx_lo + tid; i < idx_hi; i += BS) {
        int entry = log[i];


        // TODO: constant '11' is really just 32 - 9 - SHAB, where 9 is the
        // number of bits assigned to color. This ignores opacity.
        bfe_decl(glob_addr, entry, SHAB, 11);
        if (glob_addr != bid) continue;

        bfe_decl(shr_addr, entry, 0, SHAB);
        bfe_decl(color, entry, 23, 9);

        float colorf = color / 511.0f;
        float4 outcol = tex2D(palTex, colorf, time);

        // TODO: change texture sampler to return shorts and avoid this
        uint32_t r = 255.0f * outcol.x;
        uint32_t g = 255.0f * outcol.y;
        uint32_t b = 255.0f * outcol.z;

        uint32_t dr = atomicAdd(s_acc_dr + shr_addr, r + 0x10000);
        uint32_t gb = atomicAdd(s_acc_gb + shr_addr, g + (b << 16));
        uint32_t d = dr >> 16;

        // Neat trick: if overflow is about to happen, write the accumulator,
        // and subtract the last known values from the accumulator again.
        // Even if the ints end up wrapping around once before the subtraction
        // can occur, the results after the subtraction will be correct.
        // (Wrapping twice will mess up the intermediate write, but is pretty
        // unlikely.)
        if (d == 250) {
            atomicSub(s_acc_dr + shr_addr, dr);
            atomicSub(s_acc_gb + shr_addr, gb);
            write_shmem_helper(acc, glo_base + shr_addr, dr, gb);
        }
        time += time_step;
    }

    __syncthreads();
    int idx = tid;
    for (int i = 0; i < (SHAW / BS); i++) {
        write_shmem_helper(acc, glo_base + idx, s_acc_dr[idx], s_acc_gb[idx]);
        idx += BS;
    }
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

