from util import devlib
from color import yuvlib

texshearlib = devlib(defs=r'''
// Filter directions specified in degrees, using image/texture addressing
// [(0,0) is upper left corner, 90 degrees is down].

__constant__ float2 addressing_patterns[16] = {
    { 1.0f,  0.0f},        { 0.0f,       1.0f}, //  0,  1:   0,    90
    { 1.0f,  1.0f},        {-1.0f,       1.0f}, //  2,  3:  45,   135
    { 1.0f,  0.5f},        {-0.5f,       1.0f}, //  4,  5:  22.5, 112.5
    { 1.0f, -0.5f},        { 0.5f,       1.0f}, //  6,  7: -22.5,  67.5
    { 1.0f,  0.666667f},   {-0.666667f,  1.0f}, //  8,  9:  30,   120
    { 1.0f, -0.666667f},   { 0.666667f,  1.0f}, // 10, 11: -30,    60
    { 1.0f,  0.333333f},   {-0.333333f,  1.0f}, // 12, 13:  15,   105
    { 1.0f, -0.333333f},   { 0.333333f,  1.0f}, // 14, 15: -15,    75
};

// Mon dieu! A C++ feature? Gotta to close the "extern C" added by the compiler.
}

template <typename T> __device__ T
tex_shear(texture<T, cudaTextureType2D> ref, int pattern,
          float x, float y, float radius) {
    float2 scale = addressing_patterns[pattern];
    float i = scale.x * radius, j = scale.y * radius;
    // Round i and j to the nearest integer, choosing the nearest even when
    // equidistant. It's critical that this be done before adding 'x' and 'y',
    // so that addressing patterns remain consistent across the grid.
    asm("{\n\t"
        "cvt.rni.ftz.f32.f32    %0, %0;\n\t"
        "cvt.rni.ftz.f32.f32    %1, %1;\n\t"
        "}\n" : "+f"(i), "+f"(j));
    return tex2D(ref, x + i, y + j);
}

extern "C" {
''')

logscalelib = devlib(defs=r'''
__global__ void
logscale(float4 *outbuf, const float4 *pixbuf, float k1, float k2) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float4 pix = pixbuf[i];

    float ls = fmaxf(0, k1 * logf(1.0f + pix.w * k2) / pix.w);
    pix.x *= ls;
    pix.y *= ls;
    pix.z *= ls;
    pix.w *= ls;

    outbuf[i] = pix;
}
''')

fmabuflib = devlib(defs=r'''
// Element-wise computation of ``dst[i]=dst[i]+src[i]*scale``.
__global__ void
fma_buf(float4 *dst, const float4 *src, int astride, float scale) {
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
''')

denblurlib = devlib(decls='''
texture<float,  cudaTextureType2D> blur_src;

__constant__ float gauss_coefs[7] = {
    0.00443305f,  0.05400558f,  0.24203623f,  0.39905028f,
    0.24203623f,  0.05400558f,  0.00443305f
};
''', defs=r'''
// Apply a Gaussian-esque blur to the density channel of the texture in
// ``bilateral_src`` in the horizontal direction, and write it to ``dst``, a
// one-channel buffer.
__global__ void den_blur(float *dst, int pattern, int upsample) {
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    float x = xi, y = yi;

    float den = 0.0f;

    #pragma unroll
    for (int i = 0; i < 7; i++)
        den += tex_shear(bilateral_src, pattern, x, y, (i - 3) << upsample).w
             * gauss_coefs[i];
    dst[yi * (blockDim.x * gridDim.x) + xi] = den;
}

// As above, but with the one-channel texture as source
__global__ void den_blur_1c(float *dst, int pattern, int upsample) {
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    float x = xi, y = yi;

    float den = 0.0f;

    #pragma unroll
    for (int i = 0; i < 7; i++)
        den += tex_shear(blur_src, pattern, x, y, (i - 3) << upsample)
             * gauss_coefs[i];
    dst[yi * (blockDim.x * gridDim.x) + xi] = den;
}
''')

bilaterallib = devlib(deps=[logscalelib, texshearlib, denblurlib], decls='''
texture<float4, cudaTextureType2D> bilateral_src;
''', defs=r'''
/* sstd:    spatial standard deviation (Gaussian filter)
 * cstd:    color standard deviation (Gaussian on the range [0, 1], where 1
 *          represents an "opposite" color).
 * dstd:    Standard deviation (exp2f) of density filter at density = 1.0.
 * dpow:    Exponent applied to density values before taking difference.
 *          At dpow=0.8, difference between 1000 and 1001 is about 0.2.
 *          Use bigger dstd and bigger dpow to blur low-density areas more
 *          without clobbering high-density areas.
 * gspeed:  Speed of (exp2f) Gompertz distribution governing how much to
 *          tighten gradients. Zero and negative values OK.
 */
__global__ void
bilateral(float4 *dst, int pattern, int radius,
          float sstd, float cstd, float dstd, float dpow, float gspeed)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    float x = xi, y = yi;

    // Precalculate the spatial coeffecients.
    __shared__ float spa_coefs[32];

    if (threadIdx.y == 0) {
        float df = threadIdx.x;
        spa_coefs[threadIdx.x] = expf(df * df / (-M_SQRT2 * sstd));
    }

    // 3.0f compensates for [0,3] range of `cdiff`
    float cscale = 1.0f / (-M_SQRT2 * 3.0f * cstd);
    float dscale = -0.5f / dstd;

    // Gather the center point, and pre-average the color values for faster
    // comparison.
    float4 cen = tex2D(bilateral_src, x, y);
    float cdrcp = 1.0f / (cen.w + 1.0e-6f);
    cen.x *= cdrcp;
    cen.y *= cdrcp;
    cen.z *= cdrcp;

    float cpowden = powf(cen.w, dpow);

    float4 out = make_float4(0, 0, 0, 0);
    float weightsum = 0.0f;

    // Be extra-sure spatial coeffecients have been written
    __syncthreads();

    float4 pix = tex_shear(bilateral_src, pattern, x, y, -radius - 1.0f);
    float4 next = tex_shear(bilateral_src, pattern, x, y, -radius);

    for (float r = -radius; r <= radius; r++) {
        float prev = pix.w;
        pix = next;
        next = tex_shear(bilateral_src, pattern, x, y, r + 1.0f);

        // This initial factor is arbitrary, but seems to do a decent job at
        // preventing excessive bleed-out from points inside an empty region.
        // (It's used when either the center or the current point has no
        // sample energy at all.)
        float cdiff = 0.5f;

        if (pix.w > 0.0f && cen.w > 0.0f) {
            // Compute the color difference as the simple magnitude difference
            // between the YUV colors at the sampling location, unweighted by
            // density. Essentially, this just identifies regions whose average
            // color coordinates are similar.
            float pdrcp = 1.0f / pix.w;
            float yd = pix.x * pdrcp - cen.x;
            float ud = pix.y * pdrcp - cen.y;
            float vd = pix.z * pdrcp - cen.z;
            cdiff = yd * yd + ud * ud + vd * vd;
        }

        // Density factor
        float powden = powf(pix.w, dpow);
        float dfact = exp2f(dscale * fabsf(cpowden - powden));

        // Gradient energy factor. This favors points whose local energy
        // gradient points towards the current point - in essence, it draws
        // sampling energy "uphill" into denser regions rather than allowing
        // it to be smeared in all directions. The effect is modulated by the
        // average energy in the region (as determined from a blurred copy of
        // the density map); weak gradients in dense image regions aren't
        // affected as strongly. This is all very experimental, with little
        // theoretical justification, but it seems to work very well.
        //
        // Note that both the gradient and the blurred weight are calculated
        // in one dimension, along the current sampling vector.
        float avg = tex_shear(blur_src, pattern, x, y, r);
        float gradfact = (next.w - prev) / (avg + 1.0e-6f);
        if (r < 0) gradfact = -gradfact;
        gradfact = exp2f(-exp2f(gspeed * gradfact));

        float factor = spa_coefs[(int) fabsf(r)] * expf(cscale * cdiff) * dfact;
        if (r != 0) factor *= gradfact;

        weightsum += factor;
        out.x += factor * pix.x;
        out.y += factor * pix.y;
        out.z += factor * pix.z;
        out.w += factor * pix.w;
    }

    float weightrcp = 1.0f / (weightsum + 1e-10f);
    out.x *= weightrcp;
    out.y *= weightrcp;
    out.z *= weightrcp;
    out.w *= weightrcp;

    const int astride = blockDim.x * gridDim.x;
    dst[yi * astride + xi] = out;
}
''')

colorcliplib = devlib(deps=[yuvlib], defs=r'''
__global__ void
colorclip(float4 *pixbuf, float gamma, float vibrance, float highpow,
          float linrange, float lingam, float3 bkgd, int fbsize)
{
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
''')
