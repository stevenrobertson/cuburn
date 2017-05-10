"""
The main iteration loop.
"""

import variations
import interp
from util import Template, devlib, ringbuflib
from mwc import mwclib

import cuburn.genome.specs

def precalc_densities(cp):
    # This pattern recurs a few times for precalc segments. Unfortunately,
    # namespace stuff means it's not easy to functionalize this boilerplate
    cp._code(Template(r"""
        float sum = 0.0f;

        {{for n in cp.xforms}}
        float den_{{n}} = {{cp.xforms[n].weight}};
        sum += den_{{n}};
        {{endfor}}

        float rsum = 1.0f / sum;
        sum = 0.0f;

        {{for n in cp.xforms.keys()[:-1]}}
        sum += den_{{n}} * rsum;
        {{cp._set('den_' + n)}} = sum;
        {{endfor}}
    """, name='precalc_densities').substitute(cp=cp))

def precalc_chaos(cp):
    cp._code(Template("""
        float sum, rsum;

        {{for p in cp.xforms}}
        sum = 0.0f;

        {{for n in cp.xforms}}
        float den_{{p}}_{{n}} = {{cp.xforms[n].weight}}
                              * {{cp.xforms[p].chaos[n]}};
        sum += den_{{p}}_{{n}};
        {{endfor}}

        rsum = 1.0f / sum;
        sum = 0.0f;

        {{for n in cp.xforms.keys()[:-1]}}
        sum += den_{{p}}_{{n}} * rsum;
        {{cp._set('chaos_%s_%s' % (p, n))}} = sum;
        {{endfor}}

        {{endfor}}
    """, name='precalc_chaos').substitute(cp=cp))

def precalc_camera(cam):
    # Maxima code to check my logic:
    #   matrix([1,0,0.5*width + g],[0,1,0.5*height+g],[0,0,1])
    # . matrix([width * scale,0,0], [0,width * scale,0], [0,0,1])
    # . matrix([cosr,-sinr,0], [sinr,cosr,0], [0,0,1])
    # . matrix([1,0,-cenx],[0,1,-ceny],[0,0,1])
    # . matrix([X],[Y],[1]);

    cam._code(Template(r"""
        float rot = {{cam.rotation}} * M_PI / 180.0f;
        float rotsin = sin(rot), rotcos = cos(rot);
        float cenx = {{cam.center.x}}, ceny = {{cam.center.y}};
        float scale = {{cam.scale}} * acc_size.width;

        {{cam._set('xx')}} = scale * rotcos;
        {{cam._set('xy')}} = scale * -rotsin;
        {{cam._set('xo')}} = scale * (rotsin * ceny - rotcos * cenx)
                           + 0.5f * acc_size.awidth;

        {{cam._set('yx')}} = scale * rotsin;
        {{cam._set('yy')}} = scale * rotcos;
        {{cam._set('yo')}} = scale * -(rotsin * cenx + rotcos * ceny)
                           + 0.5f * acc_size.aheight;
    """, 'precalc_camera').substitute(cam=cam))

def precalc_xf_affine(px):
    px._code(Template(r"""
        float pri = {{px.angle}} * M_PI / 180.0f;
        float spr = {{px.spread}} * M_PI / 180.0f;

        float magx = {{px.magnitude.x}};
        float magy = {{px.magnitude.y}};

        {{px._set('xx')}} = magx * cos(pri-spr);
        {{px._set('yx')}} = -magx * sin(pri-spr);
        {{px._set('xy')}} = -magy * cos(pri+spr);
        {{px._set('yy')}} = magy * sin(pri+spr);
        {{px._set('xo')}} = {{px.offset.x}};
        {{px._set('yo')}} = -{{px.offset.y}};
    """, 'precalc_xf_affine').substitute(px=px))

def apply_affine(names, packer):
    x, y, xo, yo = names.split()
    return Template("""
    {{xo}} = {{packer.xx}} * {{x}} + {{packer.xy}} * {{y}} + {{packer.xo}};
    {{yo}} = {{packer.yx}} * {{x}} + {{packer.yy}} * {{y}} + {{packer.yo}};
    """, 'apply_affine').substitute(locals())

# The number of threads per block used in the iteration function. Don't change
# it lightly; the code may depend on it in unusual ways.
NTHREADS = 256

iter_decls = """
__shared__ iter_params params;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t awidth;
    uint32_t aheight;
    uint32_t astride;
} acc_size_t;
__constant__ acc_size_t acc_size;
"""

iter_xf_body_code = r"""
__device__
void apply_xf_{{xfid}}(float &ox, float &oy, float &color, mwc_st &rctx) {
    float tx, ty;

    {{precalc_xf_affine(px.pre_affine._precalc())}}
    {{apply_affine('ox oy tx ty', px.pre_affine)}}

    ox = 0;
    oy = 0;

    {{for name, pv in px.variations.items()}}
  {
    float w = {{pv.weight}};
    {{variations.var_code[name].substitute(locals())}}
  }
    {{endfor}}

    {{if 'post_affine' in px}}
    tx = ox;
    ty = oy;
    {{precalc_xf_affine(px.post_affine._precalc())}}
    {{apply_affine('tx ty ox oy', px.post_affine)}}
    {{endif}}

    float csp = {{px.color_speed}};
    color = color * (1.0f - csp) + {{px.color}} * csp;
};
"""

def iter_xf_body(cp, xfid, px):
    tmpl = Template(iter_xf_body_code, 'apply_xf_'+xfid)
    g = dict(globals())
    g.update(locals())
    return tmpl.substitute(g)

iter_body_code = r'''
__global__ void
iter(uint64_t out_ptr, uint64_t atom_ptr,
     ringbuf *rb, mwc_st *msts, float4 *points,
     const __restrict__ uint32_t *hotspots,
     const __restrict__ iter_params *all_params)
{
    // load params to shared memory cooperatively
    const iter_params *global_params = &(all_params[blockIdx.x]);
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < (sizeof(iter_params) / 4); i += blockDim.x * blockDim.y)
        reinterpret_cast<float*>(&params)[i] =
            reinterpret_cast<const float*>(global_params)[i];

    int this_rb_idx = rb_incr(rb->head, blockDim.x * threadIdx.y + threadIdx.x);
    mwc_st rctx = msts[this_rb_idx];

    {{precalc_camera(cp.camera._precalc())}}
    if (threadIdx.y == 5 && threadIdx.x == 4) {
        float ditherwidth = {{cp.camera.dither_width}} * 0.5f;
        {{cp.camera.xo}} += ditherwidth * mwc_next_11(rctx);
        {{cp.camera.yo}} += ditherwidth * mwc_next_11(rctx);
    }

    int time = blockIdx.x >> 4;
    float color_dither = 0.49f * mwc_next_11(rctx);

    float4 old_point = points[this_rb_idx];
    float x = old_point.x, y = old_point.y, color = old_point.z;

{{if not chaos_used}}
    // Shared memory size can be reduced by a factor of four using a slower
    // 4-stage reduce, but on Fermi hardware shmem use isn't a bottleneck.
    __shared__ float swap[{{4*NTHREADS}}];

    // Cooperative branch selection, used for deciding when all threads in a
    // warp should share a branch.
    __shared__ float cosel[{{2*NWARPS}}];

    // This is normally done after the swap-sync in the main loop
    if (threadIdx.y == 0 && threadIdx.x < {{NWARPS*2}})
        cosel[threadIdx.x] = mwc_next_01(rctx);
    __syncthreads();
{{endif}}

    bool fuse = false;
    int last_xf_used = 0;

    // If there is a NaN at the start of this set of iterations, it's usually
    // the signal that this is the first iter to use this point data, so reset
    // and run without writeback to stabilize the trajectory.
    if (!isfinite(fabsf(x) + fabsf(y))) {
        x = mwc_next_11(rctx);
        y = mwc_next_11(rctx);
        color = mwc_next_01(rctx);
        fuse = true;
    }

    for (int round = 0; round < 256; round++) {
        // If we're still getting NaNs, the flame is probably divergent. The
        // correct action would be to allow the NaNs to be filtered out.
        // However, this deviates from flam3 behavior, and makes it difficult
        // to correct a flame (manually or programmatically) by editing
        // splines, since incremental improvements won't be visible until the
        // system is sufficiently convergent. We reset but do *not* set fuse.
        if (!isfinite(fabsf(x) + fabsf(y))) {
            x = mwc_next_11(rctx);
            y = mwc_next_11(rctx);
            color = mwc_next_01(rctx);
        }


{{py:xk = cp.xforms.keys()}}
{{if chaos_used}}

        {{precalc_chaos(cp)}}

        // For now, we don't attempt to use the swap buffer when chaos is used
        float xfsel = mwc_next_01(rctx);

        {{for prior_xform_idx, prior_xform_name in enumerate(xk)}}
        if (last_xf_used == {{prior_xform_idx}}) {
            {{for xform_idx, xform_name in enumerate(xk[:-1])}}
            if (xfsel <= {{cp['chaos_'+prior_xform_name+'_'+xform_name]}}) {
                apply_xf_{{xform_name}}(x, y, color, rctx);
                last_xf_used = {{xform_idx}};
            } else
            {{endfor}}
            {
                apply_xf_{{xk[-1]}}(x, y, color, rctx);
                last_xf_used = {{len(xk)-1}};
            }
        } else
        {{endfor}}
        {
            //printf("Something went *very* wrong.\n");
            asm("trap;");
        }

{{else}}
        {{precalc_densities(cp._precalc())}}
        float xfsel = cosel[threadIdx.y];

        {{for xform_idx, xform_name in enumerate(xk[:-1])}}
        if (xfsel <= {{cp['den_'+xform_name]}}) {
            apply_xf_{{xform_name}}(x, y, color, rctx);
            last_xf_used = {{xform_idx}};
        } else
        {{endfor}}
        {
            apply_xf_{{xk[-1]}}(x, y, color, rctx);
            last_xf_used = {{len(xk)-1}};
        }

        // Rotate points between threads.
        int swr = threadIdx.y + threadIdx.x
                + (round & 1) * (threadIdx.x / {{NTHREADS / 32}});
        int sw = (swr * 32 + threadIdx.x) & {{NTHREADS-1}};
        int sr = threadIdx.y * 32 + threadIdx.x;

        swap[sw] = fuse ? -1.0f : last_xf_used;
        swap[sw+{{NTHREADS}}] = x;
        swap[sw+{{2*NTHREADS}}] = y;
        swap[sw+{{3*NTHREADS}}] = color;
        __syncthreads();

        // We select the next xforms here, since we've just synced.
        if (threadIdx.y == 0 && threadIdx.x < {{NWARPS*2}})
            cosel[threadIdx.x] = mwc_next_01(rctx);

        last_xf_used = swap[sr];
        fuse = (last_xf_used < 0);
        x = swap[sr+{{NTHREADS}}];
        y = swap[sr+{{2*NTHREADS}}];
        color = swap[sr+{{3*NTHREADS}}];

{{endif}}

        if (fuse) {
            continue;
        }

        float cx, cy, cc;
{{if 'final_xform' in cp}}
        float fx = x, fy = y, fcolor = color;
        apply_xf_final(fx, fy, fcolor, rctx);
        {{apply_affine('fx fy cx cy', cp.camera)}}
        cc = fcolor;
{{else}}
        {{apply_affine('x y cx cy', cp.camera)}}
        cc = color;
{{endif}}

        uint32_t ix = trunca(cx), iy = trunca(cy);

        if (ix >= acc_size.astride || iy >= acc_size.aheight) {
            continue;
        }

        uint32_t hotspot_i = (
            ((iy >> 4) * acc_size.astride) +
            (ix & 0xfffffff0) + (iy & 0xf));
        uint32_t hotspot_rshift = (ix & 0xf) << 1;
        uint32_t hotspot_value = (hotspots[hotspot_i] >> hotspot_rshift) & 0x3;
        float hotspot_mult = 1.0f;
        if (hotspot_value > 0) {
          hotspot_mult = ((1 << (hotspot_value << 1)) >> 1);
          float prob = __frcp_rn(hotspot_mult);
          if (mwc_next_01(rctx) > prob) continue;
        }

        uint32_t i = iy * acc_size.astride + ix;
        asm volatile ({{crep("""
{
    // To prevent overflow, we need to flush each pixel before the density
    // wraps at 1024 points. This atomic segment performs writes to the
    // integer buffer, occasionally checking the results. If they exceed a
    // threshold, it zeros that cell in the integer buffer, converts the
    // former contents to floats, and adds them to the float4 buffer.

    .reg .pred  p, h;
    .reg .u32   off, hotmap, color, hi, lo, d, y, u, v;
    .reg .f32   colorf, yf, uf, vf, df, mult, prob;
    .reg .u64   ptr, val;

    // TODO: coord dithering better, or pre-supersampled palette?
    fma.rn.ftz.f32      colorf, %0,     255.0,  %1;
    cvt.rni.u32.f32     color,  colorf;
    shl.b32             color,  color,  3;

    // Load the pre-packed 64-bit uint from the palette surf
    suld.b.2d.v2.b32.clamp      {lo, hi},   [flatpal, {color, %2}];
    mov.b64             val,    {lo, hi};

    // Calculate the output address in the atomic integer accumulator
    shl.b32             off,    %3,     3;
    cvt.u64.u32         ptr,    off;
    add.u64             ptr,    ptr,    %4;

    // 97% of the time, do an atomic add, then jump to the end without
    // stalling the thread waiting for the data value
    setp.le.f32         p,      %5,     0.97;
@p  red.global.add.u64  [ptr],  val;
@p  bra                 oflow_end;

    // 3% of the time, do the atomic add, and wait for the results
    atom.global.add.u64 val,    [ptr],  val;
    mov.b64             {lo, hi},       val;

    // If the density is less than 64, jump to the end
    setp.lo.u32         p,      hi,     (256 << 23);
@p  bra                 oflow_end;

    // Atomically swap the integer cell with 0 and read its current value
    atom.global.exch.b64 val,   [ptr],  0;
    mov.b64             {lo, hi},       val;

    // If the integer cell is zero, another thread has captured the full value
    // in between the first atomic read and the second, so we can skip to the
    // end again.
    setp.eq.u32         p,      hi,     0;
@p  bra                 oflow_end;

    // Extract the values from the packed integer, convert to floats, and add
    // them to the floating-point buffer.
    shr.u32             d,      hi,     22;
    bfe.u32             y,      hi,     4,      18;
    bfe.u32             u,      lo,     18,     14;
    bfi.b32             u,      hi,     u,      14,     4;
    and.b32             v,      lo,     ((1<<18)-1);
    cvt.rn.f32.u32      yf,     y;
    cvt.rn.f32.u32      uf,     u;
    cvt.rn.f32.u32      vf,     v;
    cvt.rn.f32.u32      df,     d;

    mul.rn.ftz.f32      df,     df,     %7;
    mul.rn.ftz.f32      mult,   %7,     (1.0/255.0);
    mul.rn.ftz.f32      yf,     yf,     mult;
    mul.rn.ftz.f32      uf,     uf,     mult;
    mul.rn.ftz.f32      vf,     vf,     mult;
    shl.b32             off,    %3,     4;
    cvt.u64.u32         ptr,    off;
    add.u64             ptr,    ptr,    %6;
    red.global.add.f32  [ptr],          yf;
    red.global.add.f32  [ptr+4],        uf;
    red.global.add.f32  [ptr+8],        vf;
    red.global.add.f32  [ptr+12],       df;
oflow_end:
}
        """)}}  ::  "f"(cc), "f"(color_dither), "r"(time), "r"(i),
                    "l"(atom_ptr), "f"(cosel[threadIdx.y + {{NWARPS}}]),
                    "l"(out_ptr), "f"(hotspot_mult));
    }

    this_rb_idx = rb_incr(rb->tail, blockDim.x * threadIdx.y + threadIdx.x);
    points[this_rb_idx] = make_float4(x, y, color, 0.0f);
    msts[this_rb_idx] = rctx;
    return;
}

__global__ void flush_atom(uint64_t out_ptr, uint64_t atom_ptr,
                           __restrict__ uint32_t *hotspot_ptr, int nbins) {
    const int yi = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int xi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int gi = yi * acc_size.astride + xi;

    const int hoti = (
        ((yi >> 4) * acc_size.astride) +
        (xi & 0xfffffff0) + (yi & 0xf));
    asm volatile ({{crep("""
{
    .reg .u32   off, hi, lo, d, y, u, v, hotflag, tidx;
    .reg .u64   val, ptr;
    .reg .f32   yf, uf, vf, df, yg, ug, vg, dg, mult;
    .reg .pred  p, q;

    // Load the hotflag and compute the scale factor.
    shl.b32             off,    %1,   2;
    cvt.u64.u32         ptr,    off;
    add.u64             ptr,    ptr,    %4;
    ld.global.b32       hotflag, [ptr];

    mov.b32             tidx,   %tid.x;
    shl.b32             tidx,   tidx,       1;
    shr.b32             hotflag,  hotflag,  tidx;
    and.b32             hotflag,  hotflag,  0x3;

    shl.b32             hotflag,  hotflag,  1;
    shl.b32             hotflag,  1,        hotflag;
    shr.b32             hotflag,  hotflag,  1;
    cvt.rn.f32.u32      mult,     hotflag;
    setp.eq.f32         p,        0.0,      mult;
@p  mov.f32             mult,     1.0;

    shl.b32             off,    %0,     3;
    cvt.u64.u32         ptr,    off;
    add.u64             ptr,    ptr,    %2;
    ld.cs.global.v2.u32 {lo, hi},   [ptr];
    st.cs.global.b64    [ptr],  0;

    shl.b32             off,    %0,     4;
    cvt.u64.u32         ptr,    off;
    add.u64             ptr,    ptr,    %3;
    ld.cs.global.v4.f32 {yg,ug,vg,dg},  [ptr];
    shr.u32             d,      hi,     22;
    bfe.u32             y,      hi,     4,      18;
    bfe.u32             u,      lo,     18,     14;
    bfi.b32             u,      hi,     u,      14,     4;
    and.b32             v,      lo,     ((1<<18)-1);
    cvt.rn.f32.u32      yf,     y;
    cvt.rn.f32.u32      uf,     u;
    cvt.rn.f32.u32      vf,     v;
    cvt.rn.f32.u32      df,     d;
    fma.rn.ftz.f32      dg,     df,     mult,    dg;
    mul.rn.ftz.f32      mult,   mult,   (1.0/255.0);
    fma.rn.ftz.f32      yg,     yf,     mult,    yg;
    fma.rn.ftz.f32      ug,     uf,     mult,    ug;
    fma.rn.ftz.f32      vg,     vf,     mult,    vg;

    st.cs.global.v4.f32 [ptr],  {yg,ug,vg,dg};

    .reg .u32   bal1, bal2, bal3, tidy, mask, tbit, tbit2, tbit3;
    .reg .u32   outlo, outhi;

    // Vote on whether this thread's point passes the (for now, predetermined)
    // thresholds for each mask step
    setp.gt.f32         p,      dg,     128.0;
    vote.ballot.b32     bal1,   p;
    setp.gt.f32         p,      dg,     512.0;
    vote.ballot.b32     bal2,   p;
    setp.gt.f32         p,      dg,     2048.0;
    vote.ballot.b32     bal3,   p;

    // Set 'q' with true if we're a low bit
    mov.b32             tidx,   %laneid;
    and.b32             mask,   tidx,   1;
    setp.eq.u32         q,      mask,   0;

    // Set 'mask' with the bit that we're considering in the ballots:
    // 1 << (laneid / 2)
    shr.b32             mask,   tidx,   1;
    shl.b32             mask,   1,      mask;

    // If this is a low bit, it's set if either the hotflag should be 1 or 3
    //   (reduces to 1 & (2 = 3), since 2 will never be unset if 3 is set)
    // If this is a high bit, it's set if it should be 2 or 3
    // Set lo bit on all threads, then overwrite it on just hi ones
    // Vote to get the value we should write
    and.b32             tbit,   mask,   bal1;
    setp.gt.u32         p,      tbit,   0;
    and.b32             tbit,   mask,   bal2;
    and.b32             tbit2,  mask,   bal3;
    setp.and.eq.u32     p,      tbit,   tbit2,  p;
@q  setp.gt.u32         p,      tbit,   0;

    vote.ballot.b32     outlo,  p;

    // Repeat this process for the high half of the warp
    shl.b32             mask,   mask,   16;
    and.b32             tbit,   mask,   bal1;
    setp.gt.u32         p,      tbit,   0;
    and.b32             tbit,   mask,   bal2;
    and.b32             tbit2,  mask,   bal3;
    setp.and.eq.u32     p,      tbit,   tbit2,  p;
@q  setp.gt.u32         p,      tbit,   0;

    vote.ballot.b32     outhi,  p;

    // Set 'p' with whether we're the low or high bit of this warp
    mov.b32             tidy,   %tid.y;
    and.b32             tidy,   tidy,   1;
    setp.gt.u32         p,      tidy,   0;
@p  mov.b32             outlo,  outhi;

    shl.b32             off,    %1,     2;
    cvt.u64.u32         ptr,    off;
    add.u64             ptr,    ptr,    %4;
    mov.b32             tidx,   %tid.x;
    // Set 'p' with whether we're thread 0 of tid.x
    setp.eq.u32         p,      tidx,   0;
@p  st.cs.global.b32    [ptr],  outlo;
}
""")}}  ::  "r"(gi), "r"(hoti), "l"(atom_ptr), "l"(out_ptr), "l"(hotspot_ptr),
"r"(xi), "r"(yi));
}
'''

def iter_body(cp):
    tmpl = Template(iter_body_code, 'iter_body')
    NWARPS = NTHREADS / 32

    # TODO: detect this properly and use it
    chaos_used = False

    vars = globals()
    vars.update(locals())
    return tmpl.substitute(vars)

def mkiterlib(gnm):
    packer = interp.GenomePacker('iter_params', 'params',
                                 cuburn.genome.specs.anim)
    cp = packer.view(gnm)

    iterbody = iter_body(cp)
    bodies = [iter_xf_body(cp, i, x) for i, x in sorted(cp.xforms.items())]
    if 'final_xform' in cp:
        bodies.append(iter_xf_body(cp, 'final', cp.final_xform))
    bodies.append(iterbody)
    packer_lib = packer.finalize()

    lib = devlib(deps=[packer_lib, mwclib, ringbuflib],
                 # We grab the surf decl from palintlib as well
                 decls=iter_decls + interp.palintlib.decls,
                 defs='\n'.join(bodies))
    return packer, lib
