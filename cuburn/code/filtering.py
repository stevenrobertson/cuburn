
import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.driver as cuda
import pycuda.compiler
from pycuda.gpuarray import vec

from cuburn.code.util import *

_CODE = r'''
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
    pix.y -= 0.5f * pix.w;
    pix.z -= 0.5f * pix.w;
    float3 tmp = yuv2rgb(make_float3(pix.x, pix.y, pix.z));
    pix.x = tmp.x;
    pix.y = tmp.y;
    pix.z = tmp.z;

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
void logscale(float4 *outbuf, const float4 *pixbuf, float k1, float k2) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float4 pix = pixbuf[i];

    // float ls = fmaxf(0, k1 * logf(1.0f + pix.w * pix.w * k2 / (1 + pix.w)) / pix.w);
    float ls = fmaxf(0, k1 * logf(1.0f + pix.w * k2) / pix.w);
    pix.x *= ls;
    pix.y *= ls;
    pix.z *= ls;
    pix.w *= ls;

    outbuf[i] = pix;
}

// Element-wise computation of ``dst[i]=dst[i]+src[i]*scale``.
__global__
void fma_buf(float4 *dst, const float4 *src, int astride, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * astride + x;
    float4 d = dst[i], s = src[i];
    d.x += s.x * scale;
    d.y += s.y * scale;
    d.z += s.z * scale;
    d.w += s.w * scale;
    dst[i] = d;
}

texture<float4, cudaTextureType2D> bilateral_src;
texture<float,  cudaTextureType2D> blur_src;

// Apply a Gaussian-esque blur to the density channel of the texture in
// ``bilateral_src`` in the horizontal direction, and write it to ``dst``, a
// one-channel buffer
__global__ void blur_h(float *dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float den = 0.0f;
    den += tex2D(bilateral_src, x - 2, y).w * 0.00135f;
    den += tex2D(bilateral_src, x - 1, y).w * 0.1573f;
    den += tex2D(bilateral_src, x,     y).w * 0.6827f;
    den += tex2D(bilateral_src, x + 1, y).w * 0.1573f;
    den += tex2D(bilateral_src, x + 2, y).w * 0.00135f;
    dst[y * (blockDim.x * gridDim.x) + x] = den;
}

// As above, but with a one-channel texture as source
__global__ void blur_h_1cp(float *dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float den = 0.0f;
    den += tex2D(blur_src, x - 2, y) * 0.00135f;
    den += tex2D(blur_src, x - 1, y) * 0.1573f;
    den += tex2D(blur_src, x,     y) * 0.6827f;
    den += tex2D(blur_src, x + 1, y) * 0.1573f;
    den += tex2D(blur_src, x + 2, y) * 0.00135f;
    dst[y * (blockDim.x * gridDim.x) + x] = den;
}

// As above, but in the vertical direction
__global__ void blur_v(float *dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float den = 0.0f;
    den += tex2D(blur_src, x, y - 2) * 0.00135f;
    den += tex2D(blur_src, x, y - 1) * 0.1573f;
    den += tex2D(blur_src, x, y    ) * 0.6827f;
    den += tex2D(blur_src, x, y + 1) * 0.1573f;
    den += tex2D(blur_src, x, y + 2) * 0.00135f;
    dst[y * (blockDim.x * gridDim.x) + x] = den;
}

/* sstd: spatial standard deviation (Gaussian filter)
 * cstd: color standard deviation (Gaussian on the range [0, 1], where 1
 *       represents an "opposite" color).
 *
 * Density is controlled by a power-of-two Gompertz distribution:
 *  v = 1 - 2^(-sum^dpow * 2^((dhalfpt - x) * dspeed))
 *
 * dhalfpt: The difference in density values between two points at which the
 *          filter admits 50% of the spatial and color kernels, when dpow
 *          is 0. `3` seems to be a good fit for most images at decent
 *          sampling levels.
 * dspeed:  The sharpness of the filter's cutoff around dhalfpt. At `1`, the
 *          filter admits 75% of a point that differs by one fewer than
 *          `dhalfpt` density steps from the current point (when dpow is 0);
 *          at `2`, it admits 93.75% of the same. `0.5` works pretty well.
 * dpow:    The change of filter intensity as density scales. This should be
 *          set automatically in response to changes in expected density per
 *          cell.
 */
__global__
void bilateral(float4 *dst, float sstd, float cstd,
               float dhalfpt, float dspeed, float dpow) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int r = 7;

    // Precalculate the spatial coeffecients.
    __shared__ float spa_coefs[32];
    if (threadIdx.y == 0) {
        float df = threadIdx.x;
        spa_coefs[threadIdx.x] = expf(df * df / (-M_SQRT2 * sstd));
    }

    // 3.0f compensates for [0,3] range of `cdiff`
    float cscale = 1.0f / (-M_SQRT2 * 3.0f * cstd);

    // Gather the center point, and pre-average the color values for easier
    // comparison.
    float4 cen = tex2D(bilateral_src, x, y);
    if (cen.w > 0.0f) {
        float cdrcp = 1.0f / cen.w;
        cen.x *= cdrcp;
        cen.y *= cdrcp;
        cen.z *= cdrcp;
    }

    // Compute the Sobel directional derivative of a pre-blurred version of
    // the density channel at the center point.
    float nw = tex2D(blur_src, x - 1, y - 1);
    float ne = tex2D(blur_src, x + 1, y - 1);
    float sw = tex2D(blur_src, x - 1, y + 1);
    float se = tex2D(blur_src, x + 1, y + 1);
    float h = ne + se + 2 * tex2D(blur_src, x + 1, y)
            -(nw + sw + 2 * tex2D(blur_src, x - 1, y));
    float v = se + sw + 2 * tex2D(blur_src, x, y + 1)
            -(ne + nw + 2 * tex2D(blur_src, x, y - 1));

    // TODO: figure out how to work `mag` in to scaling the degree to which
    // the directionality filter clamps
    float mag = sqrtf(h * h + v * v);

    float4 out = make_float4(0, 0, 0, 0);
    float weightsum = 0.0f;

    // Be extra-sure spatial coeffecients have been written
    __syncthreads();

    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            float4 pix = tex2D(bilateral_src, x + i, y + j);
            float cdiff = 0.5f;

            if (pix.w > 0.0f && cen.w > 0.0f) {
                float pdrcp = 1.0f / pix.w;
                float yd = pix.x * pdrcp - cen.x;
                float ud = pix.y * pdrcp - cen.y;
                float vd = pix.z * pdrcp - cen.z;
                cdiff = yd * yd + ud * ud + vd * vd;
            }
            float ddiff = dspeed * (dhalfpt - fabsf(pix.w - cen.w) - 1.0f);
            float dsum = -powf(0.5f * (pix.w + cen.w + 1.0f), dpow);
            float dfact = 1.0f - exp2f(dsum * exp2f(ddiff));
            float angfact = (h * i + v * j) / (sqrtf(i*i + j*j) * mag + 1.0e-10f);

            // Oh, this is ridiculous. But it works!
            float factor = spa_coefs[abs(i)]  * spa_coefs[abs(j)]
                         * expf(cscale * cdiff) * dfact
                         * exp2f(2.0f * (angfact - 1.0f));

            weightsum += factor;
            out.x += factor * pix.x;
            out.y += factor * pix.y;
            out.z += factor * pix.z;
            out.w += factor * pix.w;
        }
    }

    float weightrcp = 1.0f / (weightsum + 1e-10f);
    out.x *= weightrcp;
    out.y *= weightrcp;
    out.z *= weightrcp;
    out.w *= weightrcp;

    // Uncomment to write directional gradient using YUV colors
    /*
    out.x = mag;
    out.y = h;
    out.z = v;
    out.w = mag;
    */

    const int astride = blockDim.x * gridDim.x;
    dst[y * astride + x] = out;
}
'''

class Filtering(HunkOCode):
    mod = None
    defs = _CODE

    @classmethod
    def init_mod(cls):
        if cls.mod is None:
            cls.mod = pycuda.compiler.SourceModule(assemble_code(BaseCode, cls),
                    options=['-use_fast_math', '-maxrregcount', '32'])

    def __init__(self):
        self.init_mod()

    def de(self, ddst, dsrc, dscratch, gnm, dim, tc, nxf, stream=None):
        # Helper variables and functions to keep it clean
        sb = 16 * dim.astride
        bs = sb * dim.ah
        bl, gr = (32, 8, 1), (dim.astride / 32, dim.ah / 8)

        def launch(f, *args, **kwargs):
            f(*args, block=bl, grid=gr, stream=stream, **kwargs)
        mkdsc = lambda c: argset(cuda.ArrayDescriptor(), height=dim.ah,
                                 width=dim.astride, num_channels=c,
                                 format=cuda.array_format.FLOAT)
        def mktref(n):
            tref = self.mod.get_texref(n)
            tref.set_filter_mode(cuda.filter_mode.POINT)
            tref.set_address_mode(0, cuda.address_mode.WRAP)
            tref.set_address_mode(1, cuda.address_mode.WRAP)
            return tref

        dsc = mkdsc(4)
        tref = mktref('bilateral_src')
        grad_dsc = mkdsc(1)
        grad_tref = mktref('blur_src')

        bilateral, blur_h, blur_h_1cp, blur_v, fma_buf = map(
                self.mod.get_function,
                'bilateral blur_h blur_h_1cp blur_v fma_buf'.split())
        ROUNDS = 2 # TODO: user customizable?

        def do_bilateral(bsrc, bdst):
            tref.set_address_2d(bsrc, dsc, sb)
            launch(blur_h, np.intp(bdst), texrefs=[tref])
            grad_tref.set_address_2d(bdst, grad_dsc, sb / 4)
            launch(blur_v, dscratch, texrefs=[grad_tref])
            grad_tref.set_address_2d(dscratch, grad_dsc, sb / 4)
            launch(blur_h_1cp, np.intp(bdst), texrefs=[tref])
            grad_tref.set_address_2d(bdst, grad_dsc, sb / 4)
            launch(blur_v, dscratch, texrefs=[grad_tref])
            grad_tref.set_address_2d(dscratch, grad_dsc, sb / 4)
            launch(bilateral, np.intp(bdst), f32(4), f32(0.1),
                   f32(5), f32(0.4), f32(0.6), texrefs=[tref, grad_tref])
            return bdst, bsrc

        # Filter the first xform, using `ddst` as an intermediate buffer.
        # End result is the filtered copy in `dsrc`.
        a, b = dsrc, ddst
        for i in range(ROUNDS):
            a, b = do_bilateral(a, b)
        if ROUNDS % 2:
            cuda.memcpy_dtod_async(b, a, bs, stream)

        # Filter the remaining xforms, using `ddst` as an intermediate
        # buffer, then add the result to `dsrc` (at the zero'th xform).
        for x in range(1, nxf):
            a, b = int(dsrc) + x * bs, ddst
            for i in range(ROUNDS):
                a, b = do_bilateral(a, b)
            launch(fma_buf, dsrc, np.intp(a), i32(dim.astride), f32(1))

        # Log-scale the accumulated buffer in `dsrc`.
        k1 = f32(gnm.color.brightness(tc) * 268 / 256)
        # Old definition of area is (w*h/(s*s)). Since new scale 'ns' is now
        # s/w, new definition is (w*h/(s*s*w*w)) = (h/(s*s*w))
        area = dim.h / (gnm.camera.scale(tc) ** 2 * dim.w)
        k2 = f32(1.0 / (area * gnm.spp(tc)))
        nbins = dim.ah * dim.astride
        logscale = self.mod.get_function("logscale")
        t = logscale(ddst, dsrc, k1, k2,
                block=(512, 1, 1), grid=(nbins/512, 1), stream=stream)

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
