
import numpy as np
from numpy import float32 as f32, int32 as i32

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

// SS must be no greater than 4x
__global__
void blur_density_h(float *scratch, const float4 *pixbuf, int astride) {
    // TODO: respect supersampling? configurable blur width?
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > astride) return;
    int y = blockIdx.y;
    const float *dsrc =
        reinterpret_cast<const float*>(&pixbuf[astride * y]) + 3;

    float den = 0.0f;

    den += dsrc[4*max(0, x-2)] * 0.00135f;
    den += dsrc[4*max(0, x-1)] * 0.1573f;
    den += dsrc[4*x] * 0.6827f;
    den += dsrc[4*min(astride-1, x+1)] * 0.1573f;
    den += dsrc[4*min(astride-1, x+2)] * 0.00135f;
    scratch[astride * y + x] = den;
}

__global__
void blur_density_v(float4 *pixbuf, const float *scratch,
                    int astride, int aheight) {
    // TODO: respect supersampling? configurable blur width?
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > astride) return;
    int y = blockIdx.y;
    const float *dsrc = scratch + x;

    float den = 0.0f;

    den += dsrc[astride*max(0, y-2)] * 0.00135f;
    den += dsrc[astride*max(0, y-1)] * 0.1573f;
    den += dsrc[astride*y] * 0.6827f;
    den += dsrc[astride*min(astride-1, y+1)] * 0.1573f;
    den += dsrc[astride*min(astride-1, y+2)] * 0.00135f;

    float *ddst = reinterpret_cast<float*>(&pixbuf[astride * y + x]) + 3;
    *ddst = min(*ddst, den);
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

    def blur_density(self, ddst, dscratch, dim, stream=None):
        blur_h = self.mod.get_function("blur_density_h")
        blur_h(dscratch, ddst, i32(dim.astride), block=(192, 1, 1),
               grid=(dim.astride / 192, dim.ah), stream=stream)
        blur_v = self.mod.get_function("blur_density_v")
        blur_v(ddst, dscratch, i32(dim.astride), i32(dim.ah), block=(192, 1, 1),
               grid=(dim.astride / 192, dim.ah), stream=stream)

    def de(self, ddst, dsrc, gnm, dim, tc, stream=None):
        k1 = f32(gnm.color.brightness(tc) * 268 / 256)
        # Old definition of area is (w*h/(s*s)). Since new scale 'ns' is now
        # s/w, new definition is (w*h/(s*s*w*w)) = (h/(s*s*w))
        area = dim.h / (gnm.camera.scale(tc) ** 2 * dim.w)
        k2 = f32(1.0 / (area * gnm.spp(tc)))

        if gnm.de.radius == 0:
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
