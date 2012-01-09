
import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.compiler
from pycuda.gpuarray import vec

from cuburn.code.util import *

_CODE = '''
#include<math_constants.h>

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


#define W 15        // Filter width (regardless of standard deviation chosen)
#define W2 7        // Half of filter width, rounded down
#define FW 46       // Width of local result storage (NW+W2+W2)
#define FW2 (FW*FW)

__shared__ float de_r[FW2], de_g[FW2], de_b[FW2], de_a[FW2];

__device__ void de_add(int ibase, int ii, int jj, float4 scaled) {
    int idx = ibase + FW * ii + jj;
    atomicAdd(de_r+idx, scaled.x);
    atomicAdd(de_g+idx, scaled.y);
    atomicAdd(de_b+idx, scaled.z);
    atomicAdd(de_a+idx, scaled.w);
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


// See helpers/filt_err.py for source of these values.
#define MAX_SCALE -0.12f
#define MIN_SCALE -9.2103404f

__global__
void density_est(float4 *pixbuf, float4 *outbuf,
                 float scale_coeff, float est_curve, float edge_clamp,
                 float k1, float k2, int height, int stride) {
    for (int i = threadIdx.x + 32*threadIdx.y; i < FW2; i += 32)
        de_r[i] = de_g[i] = de_b[i] = de_a[i] = 0.0f;
    __syncthreads();

    for (int imrow = threadIdx.y + W2; imrow < (height - W2); imrow += 32)
    {
        int idx = stride * imrow + blockIdx.x * 32 + threadIdx.x + W2;

        float4 in = pixbuf[idx];
        float den = in.w;

        if (den > 0) {

            // Compute a fast and dirty approximation of a "gradient" using
            // a [[-1 0 0][0 0 0][0 0 1]]/4 matrix (and its reflection)
            // for angled edge detection, and limit blurring in those regions
            // to both provide a bit of smoothing and prevent irregular
            // bleed-out along gradients close to the image grid.
            //
            // For such a simple operator - particularly one whose entire
            // justification is "it feels right" - it gives very good results
            // over a wide range of images without any per-flame
            // parameter tuning. In rare cases, color clamping and extreme
            // palette changes can cause aliasing to reappear after the DE
            // step; the only way to fix that is through a color-buffer AA
            // like MLAA.
            float *dens = reinterpret_cast<float*>(pixbuf);
            int didx = idx * 4 + 3;
            float x = dens[didx+stride*4+4] - dens[didx-stride*4-4];
            float y = dens[didx+stride*4-4] - dens[didx-stride*4+4];
            float diag_mag = sqrtf(x*x + y*y) * 0.3333333f;

            float ls = k1 * logf(1.0f + in.w * k2) / in.w;
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

            // If the gradient scale is smaller than the minimum scale, we're
            // probably on a strong edge; blur slightly.
            if (diag_mag > den * edge_clamp) {
                scale = -2.0f;
                // Uncomment to see which pixels are being clamped
                // de_g[si] = 1.0f;
            }

            // Below a certain threshold, only one coeffecient would be
            // retained anyway; we hop right to it.
            if (scale <= MIN_SCALE) {
                de_add(si, 0,  0, in);
            } else {
                // These polynomials approximates the reciprocal of the sum of
                // all retained filter coefficients. See helpers/filt_err.py.
                float filtsum;
                if (scale < -1.1f) {
                    filtsum = 5.20066078e-06f;
                    filtsum = filtsum * scale +   2.14025771e-04f;
                    filtsum = filtsum * scale +   3.62761668e-03f;
                    filtsum = filtsum * scale +   3.21970172e-02f;
                    filtsum = filtsum * scale +   1.54297248e-01f;
                    filtsum = filtsum * scale +   3.42210710e-01f;
                    filtsum = filtsum * scale +   3.06015890e-02f;
                    filtsum = filtsum * scale +   1.33724615e-01f;
                } else {
                    filtsum = -1.23516649e-01f;
                    filtsum = filtsum * scale +  -5.14862895e-01f;
                    filtsum = filtsum * scale +  -8.61198902e-01f;
                    filtsum = filtsum * scale +  -7.41916001e-01f;
                    filtsum = filtsum * scale +  -3.51667106e-01f;
                    filtsum = filtsum * scale +  -9.07439440e-02f;
                    filtsum = filtsum * scale +  -3.30008656e-01f;
                    filtsum = filtsum * scale +  -4.78249392e-04f;
                }

                for (int jj = 0; jj <= W2; jj++) {
                    float jj2f = jj;
                    jj2f *= jj2f;

                    float iif = 0;
                    for (int ii = 0; ii <= jj; ii++) {
                        float coeff = expf((jj2f + iif * iif) * scale)
                                    * filtsum;
                        if (coeff < 0.0001f) break;
                        iif += 1;

                        float4 scaled;
                        scaled.x = in.x * coeff;
                        scaled.y = in.y * coeff;
                        scaled.z = in.z * coeff;
                        scaled.w = in.w * coeff;

                        de_add(si,  ii,  jj, scaled);
                        if (jj == 0) continue;
                        de_add(si,  ii, -jj, scaled);
                        if (ii != 0) {
                            de_add(si, -ii,  jj, scaled);
                            de_add(si, -ii, -jj, scaled);
                            if (ii == jj) continue;
                        }
                        de_add(si,  jj,  ii, scaled);
                        de_add(si, -jj,  ii, scaled);

                        if (ii == 0) continue;
                        de_add(si, -jj, -ii, scaled);
                        de_add(si,  jj, -ii, scaled);

                    }
                }
            }
        }

        __syncthreads();
        // TODO: could coalesce this, but what a pain
        for (int i = threadIdx.x; i < FW; i += 32) {
            idx = stride * imrow + blockIdx.x * 32 + i + W2;
            int si = threadIdx.y * FW + i;
            float *out = reinterpret_cast<float*>(&outbuf[idx]);
            atomicAdd(out,   de_r[si]);
            atomicAdd(out+1, de_g[si]);
            atomicAdd(out+2, de_b[si]);
            atomicAdd(out+3, de_a[si]);
        }

        __syncthreads();
        int tid = threadIdx.y * 32 + threadIdx.x;
        for (int i = tid; i < FW*(W2+W2); i += 512) {
            de_r[i] = de_r[i+FW*32];
            de_g[i] = de_g[i+FW*32];
            de_b[i] = de_b[i+FW*32];
            de_a[i] = de_a[i+FW*32];
        }

        __syncthreads();
        for (int i = tid + FW*(W2+W2); i < FW2; i += 512) {
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

    def de(self, ddst, dsrc, gnm, dim, tc, stream=None):
        k1 = f32(gnm.color.brightness(tc) * 268 / 256)
        # Old definition of area is (w*h/(s*s)). Since new scale 'ns' is now
        # s/w, new definition is (w*h/(s*s*w*w)) = (h/(s*s*w))
        area = dim.h / (gnm.camera.scale(tc) ** 2 * dim.w)
        k2 = f32(1 / (area * gnm.spp(tc)))

        if gnm.de.radius == 0:
            nbins = dim.ah * dim.astride
            fun = self.mod.get_function("logscale")
            t = fun(dsrc, ddst, k1, k2,
                    block=(512, 1, 1), grid=(nbins/512, 1), stream=stream)
        else:
            scale_coeff = f32(-(1 + gnm.de.radius(tc)) ** -2.0)
            est_curve = f32(2 * gnm.de.curve(tc))
            # TODO: experiment with this
            edge_clamp = f32(1.2)
            fun = self.mod.get_function("density_est")
            fun(dsrc, ddst, scale_coeff, est_curve, edge_clamp, k1, k2,
                i32(dim.ah), i32(dim.astride), block=(32, 32, 1),
                grid=(dim.aw/32, 1), stream=stream)

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

