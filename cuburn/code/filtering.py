
import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.driver as cuda
import pycuda.compiler
from pycuda.gpuarray import vec

from cuburn.code.util import *

_CODE = r'''
#include<math_constants.h>
#include<stdio.h>

__global__
void colorclip(float4 *pixbuf, float gamma, float vibrance, float highpow,
               float linrange, float lingam, float3 bkgd,
               int fbsize, int blend_background_color) {
    int i = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
    if (i >= fbsize) return;

    float4 pix = pixbuf[i];

    if (pix.w <= 0) {
        pixbuf[i] = make_float4(bkgd.x, bkgd.y, bkgd.z, 0.0f);
        return;
    }

    pix.x = fmaxf(0.0f, pix.x);
    pix.y = fmaxf(0.0f, pix.y);
    pix.z = fmaxf(0.0f, pix.z);

    float4 opix = pix;

    float alpha = powf(pix.w, gamma);
    if (pix.w < linrange) {
        float frac = pix.w / linrange;
        alpha = (1.0f - frac) * pix.w * lingam + frac * alpha;
    }

    if (!blend_background_color) {
        float ls = alpha / pix.w;
        pix.x *= ls;
        pix.y *= ls;
        pix.z *= ls;
        pix.w = alpha;
        pixbuf[i] = pix;
        return;
    }

    float ls = vibrance * alpha / pix.w;
    alpha = fminf(1.0f, fmaxf(0.0f, alpha));

    float maxc = fmaxf(pix.x, fmaxf(pix.y, pix.z));
    float maxa = maxc * ls;
    float newls = 1.0f / maxc;

    if (maxa > 1.0f && highpow >= 0.0f) {
        float lsratio = powf(newls / ls, highpow);
        pix.x *= newls;
        pix.y *= newls;
        pix.z *= newls;

        // Reduce saturation (according to the HSV model) by proportionally
        // increasing the values of the other colors.
        pix.x = maxc - (maxc - pix.x) * lsratio;
        pix.y = maxc - (maxc - pix.y) * lsratio;
        pix.z = maxc - (maxc - pix.z) * lsratio;
    } else {
        float adjhlp = -highpow;
        if (adjhlp > 1.0f || maxa <= 1.0f) adjhlp = 1.0f;
        if (maxc > 0.0f) {
            float adj = ((1.0f - adjhlp) * newls + adjhlp * ls);
            pix.x *= adj;
            pix.y *= adj;
            pix.z *= adj;
        }
    }

    pix.x += (1.0f - vibrance) * powf(opix.x, gamma);
    pix.y += (1.0f - vibrance) * powf(opix.y, gamma);
    pix.z += (1.0f - vibrance) * powf(opix.z, gamma);

    pix.x += (1.0f - alpha) * bkgd.x;
    pix.y += (1.0f - alpha) * bkgd.y;
    pix.z += (1.0f - alpha) * bkgd.z;

    pix.x = fminf(1.0f, pix.x);
    pix.y = fminf(1.0f, pix.y);
    pix.z = fminf(1.0f, pix.z);
    pix.w = alpha;

    pixbuf[i] = pix;
}

__global__
void logscale(float4 *pixbuf, float4 *outbuf, float k1, float k2) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float4 pix = pixbuf[i];

    float ls = fmaxf(0, k1 * logf(1.0f + pix.w * k2) / pix.w);
    pix.x *= ls;
    pix.y *= ls;
    pix.z *= ls;
    pix.w *= ls;

    outbuf[i] = pix;
}

__device__ void fmav(float4 &dst, float4 src, float scale) {
    dst.x += src.x * scale;
    dst.y += src.y * scale;
    dst.z += src.z * scale;
    dst.w += src.w * scale;
}

/*
// The 7-tap filters are missing the leading zero to compensate for delay. I
// have no frigging clue what the theory behind that is, but it seems to work.
__constant__ float daub97_lo[9] = {
     0.03782845550726404f, -0.023849465019556843f, -0.11062440441843718f,
     0.37740285561283066f,  0.8526986790088938f,    0.37740285561283066f,
    -0.11062440441843718f, -0.023849465019556843f,  0.03782845550726404f
};
__constant__ float daub97_hi[9] = {
                           -0.06453888262869706f,   0.04068941760916406f,
     0.41809227322161724f, -0.7884856164055829f,    0.41809227322161724f,
     0.04068941760916406f, -0.06453888262869706f,   0.0f,
};
__constant__ float daub97_ilo[9] = {
                           -0.06453888262869706f,  -0.04068941760916406f,
     0.41809227322161724f,  0.7884856164055829f,    0.41809227322161724f,
    -0.04068941760916406f, -0.06453888262869706f,   0.0f
};
__constant__ float daub97_ihi[9] = {
    -0.03782845550726404f, -0.023849465019556843f,  0.11062440441843718f,
     0.37740285561283066f, -0.8526986790088938f,    0.37740285561283066f,
     0.11062440441843718f, -0.023849465019556843f, -0.03782845550726404f
};
*/


// The 7-tap filters are missing the leading zero to compensate for delay. I
// have no frigging clue what the theory behind that is, but it seems to work.
__constant__ float daub97_lo[9] = {
     0.03782845550726404f, -0.023849465019556843f, -0.11062440441843718f,
     0.37740285561283066f,  0.8526986790088938f,    0.37740285561283066f,
    -0.11062440441843718f, -0.023849465019556843f,  0.03782845550726404f
};
__constant__ float daub97_hi[9] = {
                           -0.06453888262869706f,   0.04068941760916406f,
     0.41809227322161724f, -0.7884856164055829f,    0.41809227322161724f,
     0.04068941760916406f, -0.06453888262869706f,   0.0f,
};
__constant__ float daub97_ilo[9] = {
                           -0.06453888262869706f,  -0.04068941760916406f,
     0.41809227322161724f,  0.7884856164055829f,    0.41809227322161724f,
    -0.04068941760916406f, -0.06453888262869706f,   0.0f
};
__constant__ float daub97_ihi[9] = {
    -0.03782845550726404f, -0.023849465019556843f,  0.11062440441843718f,
     0.37740285561283066f, -0.8526986790088938f,    0.37740285561283066f,
     0.11062440441843718f, -0.023849465019556843f, -0.03782845550726404f
};


/*
#define S 0.7071067811f
__constant__ float daub97_lo[9] = { S, S };
__constant__ float daub97_hi[9] = { -S, S };
__constant__ float daub97_ilo[9] = { 0, 0, 0, 0, 0, 0, S, S };
__constant__ float daub97_ihi[9] = { 0, 0, 0, 0, 0, 0, S, -S };
*/

texture<float4, cudaTextureType2D> conv_down_src;
__global__
void conv_down(float4 *dst, int astride, int as_eff, int ah_eff,
               int vert, float xo, float yo) {
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y;

    float4 lo = make_float4(0, 0, 0, 0);
    float4 hi = make_float4(0, 0, 0, 0);

    if (vert) {
        if (xi >= as_eff) return;
        float x = xi, y = yi * 2 - yo;

        #pragma unroll
        for (int i = 0; i < 9; i++) {
            float4 src = tex2D(conv_down_src, x, y);
            fmav(lo, src, daub97_lo[i]);
            fmav(hi, src, daub97_hi[i]);
            y -= 1.0f;
            if (y < 0) y += ah_eff;
        }
        dst[(yi + ah_eff / 2) * astride + xi] = hi;
    } else {
        if (xi >= as_eff / 2) return;
        float x = xi * 2 - xo, y = yi;

        #pragma unroll
        for (int i = 0; i < 9; i++) {
            float4 src = tex2D(conv_down_src, x, y);
            fmav(lo, src, daub97_lo[i]);
            fmav(hi, src, daub97_hi[i]);
            x -= 1.0f;
            if (x < 0) x += as_eff;
        }
        dst[yi * astride + xi + as_eff / 2] = hi;
    }
    dst[yi * astride + xi] = lo;
}

texture<float4, cudaTextureType2D> conv_up_src_lo;
texture<float4, cudaTextureType2D> conv_up_src_hi;
__global__
void conv_up(float4 *dst, int astride, int as_eff,
             int vert, int xo, int yo) {
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y;
    if (xi >= as_eff) return;

    float4 out = make_float4(0, 0, 0, 0);
    if (vert) {
        float x = xi, y = yi / 2;
        for (int i = ~yi & 1; i < 9; i+=2) {
            fmav(out, tex2D(conv_up_src_lo, x, y), daub97_ilo[8-i]);
            fmav(out, tex2D(conv_up_src_hi, x, y), daub97_ihi[8-i]);
            y += 1.0f;
            if (y >= gridDim.y / 2) y = 0.0f;
        }
        yi -= yo;
    } else {
        float x = xi / 2, y = yi;
        for (int i = ~xi & 1; i < 9; i+=2) {
            fmav(out, tex2D(conv_up_src_lo, x, y), daub97_ilo[8-i]);
            fmav(out, tex2D(conv_up_src_hi, x, y), daub97_ihi[8-i]);
            x += 1.0f;
            if (x >= as_eff / 2) x = 0.0f;
        }
        xi -= xo;
    }
    if (xi < 0) xi += as_eff;
    if (yi < 0) yi += gridDim.y;
    dst[yi * astride + xi] = out;
}

__global__
void simple_thresh(float4 *buf, int astride, float thr, int min_x, int min_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= astride || (x < min_x && y < min_y)) return;

    float4 val = buf[y * astride + x];
    float fact = expf(val.w * val.w * thr) < 0.2f ? 1.0f : 0.0f;
    val.x *= fact;
    val.y *= fact;
    val.z *= fact;
    val.w *= fact;
    buf[y * astride + x] = val;
}

__global__
void buf_abs(float4 *buf, int astride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= astride) return;

    float4 val = buf[y * astride + x];
    val.x = fabsf(val.x);
    val.y = fabsf(val.y);
    val.z = fabsf(val.z);
    val.w = fabsf(val.w);
    buf[y * astride + x] = val;
}

__global__
void fma_buf(float4 *dst, const float4 *sub, int astride, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * astride + x;
    if (x >= astride) return;
    float4 d = dst[i], s = sub[i];
    d.x += s.x * scale;
    d.y += s.y * scale;
    d.z += s.z * scale;
    d.w += s.w * scale;
    dst[i] = d;
}

#define W 21        // Filter width (regardless of standard deviation chosen)
#define W2 10       // Half of filter width, rounded down
#define FW 53       // Width of local result storage (NW+W2+W2)
#define FW2 (FW*FW)

__global__
void density_est(float4 *outbuf, const float4 *pixbuf,
                 float scale_coeff, float est_curve, float edge_clamp,
                 float k1, float k2, int height, int stride) {
    __shared__ float de_r[FW2], de_g[FW2], de_b[FW2], de_a[FW2];

    // The max supported radius is really 7, but the extra units simplify the
    // logic in the bottlenecked section below.
    __shared__ int minradius[32];

    for (int i = threadIdx.x + 32*threadIdx.y; i < FW2; i += 1024)
        de_r[i] = de_g[i] = de_b[i] = de_a[i] = 0.0f;
    __syncthreads();

    for (int imrow = threadIdx.y; imrow < height; imrow += 32) {
        // Prepare the shared voting buffer. Syncing afterward is
        // almost unnecessary but we do it to be safe
        if (threadIdx.y == 0)
            minradius[threadIdx.x] = 0.0f;
        __syncthreads();

        int col = blockIdx.x * 32 + threadIdx.x;
        int in_idx = stride * imrow + col;
        float4 in = pixbuf[in_idx];
        float den = in.w;

        float ls = k1 * logf(1.0f + in.w * k2) / in.w;

        // The synchronization model used now prevents us from
        // cutting early in the case of a zero point, so we just carry
        // it along with us here
        if (den <= 0) ls = 0.0f;

        in.x *= ls;
        in.y *= ls;
        in.z *= ls;
        in.w *= ls;

        // Base index of destination for writes
        int si = (threadIdx.y + W2) * FW + threadIdx.x + W2;

        // Calculate scaling coefficient for the Gaussian kernel. This
        // does not match with a normal Gaussian; it just fits with
        // flam3's implementation.
        float scale = powf(den, est_curve) * scale_coeff;

        // Force a minimum blur radius. This works out to be a
        // standard deviation of about 0.35px. Also force a maximum,
        // which limits spherical error to about 2 quanta at 10 bit
        // precision.
        scale = max(0.30f, min(scale, 2.0f));

        // Determine a minimum radius for this image section.
        int radius = (int) min(ceilf(2.12132f / scale), 10.0f);
        if (den <= 0) radius = 0;
        minradius[radius] = radius;

        // Go bottlenecked to compute the maximum radius in this block
        __syncthreads();
        if (threadIdx.y == 0) {
            int blt = __ballot(minradius[threadIdx.x]);
            minradius[0] = 31 - __clz(blt);
        }
        __syncthreads();

        radius = minradius[0];

        for (int jj = -radius; jj <= radius; jj++) {
            float jjf = (jj - 0.5f) * scale;
            float jdiff = erff(jjf + scale) - erff(jjf);

            for (int ii = -radius; ii <= radius; ii++) {
                float iif = (ii - 0.5f) * scale;
                float coeff = 0.25f * jdiff * (erff(iif + scale) - erff(iif));

                int idx = si + FW * ii + jj;
                de_r[idx] += in.x * coeff;
                de_g[idx] += in.y * coeff;
                de_b[idx] += in.z * coeff;
                de_a[idx] += in.w * coeff;
                __syncthreads();
            }
        }
        __syncthreads();
        // TODO: could coalesce this, but what a pain
        for (int i = threadIdx.x; i < FW; i += 32) {
            int out_idx = stride * imrow + blockIdx.x * 32 + i;
            int si = threadIdx.y * FW + i;
            float *out = reinterpret_cast<float*>(&outbuf[out_idx]);
            atomicAdd(out,   de_r[si]);
            atomicAdd(out+1, de_g[si]);
            atomicAdd(out+2, de_b[si]);
            atomicAdd(out+3, de_a[si]);
        }

        __syncthreads();
        int tid = threadIdx.y * 32 + threadIdx.x;
        for (int i = tid; i < FW*(W2+W2); i += 1024) {
            de_r[i] = de_r[i+FW*32];
            de_g[i] = de_g[i+FW*32];
            de_b[i] = de_b[i+FW*32];
            de_a[i] = de_a[i+FW*32];
        }

        __syncthreads();
        for (int i = tid + FW*(W2+W2); i < FW2; i += 1024) {
            de_r[i] = 0.0f;
            de_g[i] = 0.0f;
            de_b[i] = 0.0f;
            de_a[i] = 0.0f;
        }
        __syncthreads();
    }
}
'''

class Filtering(object):

    mod = None

    @classmethod
    def init_mod(cls):
        if cls.mod is None:
            cls.mod = pycuda.compiler.SourceModule(_CODE,
                    options=['-use_fast_math', '-maxrregcount', '32'])

    def __init__(self):
        self.init_mod()
        self.scratch = None

    def blur_density(self, ddst, dscratch, dim, stream=None):
        blur_h = self.mod.get_function("blur_density_h")
        blur_h(dscratch, ddst, i32(dim.astride), block=(192, 1, 1),
               grid=(1 + dim.astride / 192, dim.ah), stream=stream)
        blur_v = self.mod.get_function("blur_density_v")
        blur_v(ddst, dscratch, i32(dim.astride), i32(dim.ah), block=(192, 1, 1),
               grid=(1 + dim.astride / 192, dim.ah), stream=stream)

    def de(self, ddst, dsrc, gnm, dim, tc, stream=None):
        from cuburn.render import argset
        np.set_printoptions(linewidth=160, precision=4)

        k1 = f32(gnm.color.brightness(tc) * 268 / 256)
        # Old definition of area is (w*h/(s*s)). Since new scale 'ns' is now
        # s/w, new definition is (w*h/(s*s*w*w)) = (h/(s*s*w))
        area = dim.h / (gnm.camera.scale(tc) ** 2 * dim.w)
        k2 = f32(1.0 / (area * gnm.spp(tc)))

        # Stride in bytes, buffer size
        sb = 16 * dim.astride
        bs = sb * dim.ah

        if self.scratch is None:
            self.scratch = cuda.mem_alloc(bs)
            self.hi = cuda.mem_alloc(bs)
            self.aux = cuda.mem_alloc(bs)
        q = np.zeros((dim.ah, dim.astride * 4), dtype=np.float32)
        q[100:102,128:1024] = 1
        #cuda.memcpy_htod(dsrc, q)

        dsc = argset(cuda.ArrayDescriptor(), height=dim.ah,
                width=dim.astride, format=cuda.array_format.FLOAT,
                num_channels=4)
        bl = (192, 1, 1)
        gr = lambda x, y: (1 + dim.astride / (192 << x), dim.ah >> y)

        def get_tref(name):
            tref = self.mod.get_texref(name)
            tref.set_filter_mode(cuda.filter_mode.POINT)
            tref.set_address_mode(0, cuda.address_mode.WRAP)
            tref.set_address_mode(1, cuda.address_mode.WRAP)
            return tref
        conv_down_src, conv_up_src_lo, conv_up_src_hi = map(get_tref,
                ['conv_down_src', 'conv_up_src_lo', 'conv_up_src_hi'])

        conv_down = self.mod.get_function('conv_down')
        conv_up = self.mod.get_function('conv_up')
        fma_buf = self.mod.get_function('fma_buf')
        thresh = self.mod.get_function('simple_thresh')
        buf_abs = self.mod.get_function('buf_abs')

        memcpy = cuda.Memcpy2D()
        memcpy.src_pitch = sb
        memcpy.src_height = memcpy.dst_height = memcpy.height = dim.ah
        memcpy.set_src_device(self.scratch)
        memcpy.set_dst_device(self.hi)

        STEPS=4
        SHIFTS = [(0, 0), (0, 1), (1, 1), (1, 0),
                  (3, 0), (5, 0), (7, 0), (15, 0),
                  (0, 3), (0, 5), (0, 7), (0, 15),
                  (3, 3), (5, 5), (7, 7), (15, 15)]

        #SHIFTS = [(0, 0)]
        #SHIFTS = [(0, 0), (3, 0)]

        def th(x):
            x = np.int64(x*1e6)
            v = np.nonzero(x)[0]
            print np.array((v, x[v]))

        stream.synchronize()
        cuda.memset_d32(ddst, int(0), bs / 4)

        cuda.memcpy_dtod_async(self.aux, dsrc, bs, stream)

        for xo, yo in SHIFTS:
            for i in range(STEPS):
                xon, yon = (xo, yo) if i == 0 else (0, 0)
                as_eff, ah_eff = dim.astride >> i, dim.ah >> i
                dsc.width, dsc.height = as_eff, ah_eff
                if i == 0:
                    conv_down_src.set_address_2d(self.aux, dsc, sb)
                else:
                    conv_down_src.set_address_2d(dsrc, dsc, sb)
                conv_down(self.scratch, i32(dim.astride),
                          i32(as_eff), i32(ah_eff), i32(1), f32(xon), f32(yon),
                          block=bl, grid=gr(i, i+1),
                          texrefs=[conv_down_src], stream=stream)
                #cuda.memcpy_dtod_async(self.scratch, dsrc, bs, stream)
                conv_down_src.set_address_2d(self.scratch, dsc, sb)
                conv_down(dsrc, i32(dim.astride),
                          i32(as_eff), i32(ah_eff), i32(0), f32(xon), f32(yon),
                          block=bl, grid=gr(i+1, i),
                          texrefs=[conv_down_src], stream=stream)
                #cuda.memcpy_dtod_async(dsrc, self.scratch, bs, stream)
                #th(cuda.from_device_like(self.scratch, q).T[128])

            for i, t in enumerate([-0.05, -0.1, -0.3, -0.5]):
                thresh(dsrc, i32(dim.astride), f32(t),
                       i32(dim.astride >> (i+1)), i32(dim.ah >> (i+1)),
                       block=bl, grid=gr(i, i), stream=stream)
            #buf_abs(dsrc, i32(dim.astride),
                   #block=bl, grid=gr(0, 0), stream=stream)

            for i in reversed(range(STEPS)):
                xon, yon = (xo, yo) if i == 0 else (0, 0)
                dsc.width, dsc.height = dim.astride >> i, dim.ah >> (i+1)
                conv_up_src_lo.set_address_2d(dsrc, dsc, sb)
                hi_addr = int(dsrc) + sb * (dim.ah >> (i+1))
                conv_up_src_hi.set_address_2d(hi_addr, dsc, sb)
                conv_up(self.scratch, i32(dim.astride), i32(dim.astride >> i),
                        i32(1), i32(xon), i32(yon),
                        block=bl, grid=gr(i, i), stream=stream,
                        texrefs=[conv_up_src_lo, conv_up_src_hi])
                sb_2 = sb >> (i+1)
                memcpy.dst_pitch = sb_2
                memcpy.width_in_bytes = sb_2
                memcpy.src_x_in_bytes = sb_2
                memcpy(stream)
                dsc.width, dsc.height = dim.astride >> (i+1), dim.ah >> i
                conv_up_src_lo.set_address_2d(self.scratch, dsc, sb)
                conv_up_src_hi.set_address_2d(self.hi, dsc, sb>>(i+1))
                conv_up(dsrc, i32(dim.astride), i32(dim.astride >> i),
                        i32(0), i32(xon), i32(yon),
                        block=bl, grid=gr(i, i), stream=stream,
                        texrefs=[conv_up_src_lo, conv_up_src_hi])
                #cuda.memcpy_dtod_async(dsrc, self.scratch, bs, stream)

            #th(cuda.from_device_like(self.scratch, q).T[128])
            fma_buf(ddst, dsrc, i32(dim.astride), f32(1.0 / len(SHIFTS)),
                    block=bl, grid=gr(0, 0), stream=stream)

        cuda.memcpy_dtod_async(dsrc, ddst, bs, stream)

        #y = cuda.from_device_like(self.scratch, q)
        #print y[93:110,128].T

        #print x[93:110,128].T + y[93:110,128].T

        #fun = self.mod.get_function('fma_buf')
        #fun(self.scratch, self.lo, i32(dim.astride), f32(1),
            #block=bl, grid=gr, stream=stream)
        #cuda.memcpy_dtod_async(ddst, self.scratch, bs, stream=stream)
        #return

        if gnm.de.radius(tc) == 0 or True:
            nbins = dim.ah * dim.astride
            fun = self.mod.get_function("logscale")
            t = fun(dsrc, ddst, k1, k2,
                    block=(512, 1, 1), grid=(nbins/512, 1), stream=stream)
        else:
            scale_coeff = f32(1.0 / gnm.de.radius(tc))
            est_curve = f32(gnm.de.curve(tc))
            # TODO: experiment with this
            edge_clamp = f32(1.2)
            fun = self.mod.get_function("density_est")
            fun(ddst, dsrc, scale_coeff, est_curve, edge_clamp, k1, k2,
                i32(dim.ah), i32(dim.astride),
                block=(32, 32, 1), grid=(dim.aw/32, 1), stream=stream)

    def colorclip(self, dbuf, gnm, dim, tc, blend, stream=None):
        nbins = dim.ah * dim.astride

        # TODO: implement integration over cubic splines?
        gam = f32(1 / gnm.color.gamma(tc))
        vib = f32(gnm.color.vibrance(tc))
        hipow = f32(gnm.color.highlight_power(tc))
        lin = f32(gnm.color.gamma_threshold(tc))
        lingam = f32(lin ** (gam-1.0) if lin > 0 else 0)
        bkgd = vec.make_float3(
                gnm.color.background.r(tc),
                gnm.color.background.g(tc),
                gnm.color.background.b(tc))

        color_fun = self.mod.get_function("colorclip")
        blocks = int(np.ceil(np.sqrt(nbins / 256)))
        color_fun(dbuf, gam, vib, hipow, lin, lingam, bkgd, i32(nbins),
                  i32(blend), block=(256, 1, 1), grid=(blocks, blocks),
                  stream=stream)
