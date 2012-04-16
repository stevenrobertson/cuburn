import numpy as np

from util import devlib

# The JPEG YUV full-range matrix, without bias into the positve regime.
# This assumes input color space is CIERGB D65, encoded with gamma 2.2.
# Note that some interpolated colors may exceed the sRGB and YUV gamuts.
YUV_MATRIX = np.matrix([[ 0.299,        0.587,      0.114],
                        [-0.168736,    -0.331264,   0.5],
                        [ 0.5,         -0.418688,  -0.081312]])

yuvlib = devlib(defs='''
__device__ float3 rgb2yuv(float3 rgb);
__device__ float3 yuv2rgb(float3 yuv);
''', decls=r'''
/* This conversion uses the JPEG full-range standard. Note that UV have range
 * [-0.5, 0.5], so consider biasing the results. */
__device__ float3 rgb2yuv(float3 rgb) {
    return make_float3(
        0.299f      * rgb.x + 0.587f    * rgb.y + 0.114f    * rgb.z,
        -0.168736f  * rgb.x - 0.331264f * rgb.y + 0.5f      * rgb.z,
        0.5f        * rgb.x - 0.418688f * rgb.y - 0.081312f * rgb.z);
}

__device__ float3 yuv2rgb(float3 yuv) {
    return make_float3(
        yuv.x                    + 1.402f   * yuv.z,
        yuv.x - 0.34414f * yuv.y - 0.71414f * yuv.z,
        yuv.x + 1.772f   * yuv.y);
}

// As used in the various cliplibs.
__device__ void yuvo2rgb(float4& pix) {
    pix.y -= 0.5f * pix.w;
    pix.z -= 0.5f * pix.w;
    float3 tmp = yuv2rgb(make_float3(pix.x, pix.y, pix.z));
    pix.x = fmaxf(0.0f, tmp.x);
    pix.y = fmaxf(0.0f, tmp.y);
    pix.z = fmaxf(0.0f, tmp.z);
}

''')

hsvlib = devlib(decls='''
__device__ float3 rgb2hsv(float3 rgb);
__device__ float3 hsv2rgb(float3 hsv);
''', defs=r'''
__device__ float3 rgb2hsv(float3 rgb) {
    float M = fmaxf(fmaxf(rgb.x, rgb.y), rgb.z);
    float m = fminf(fminf(rgb.x, rgb.y), rgb.z);
    float C = M - m;

    float s = M > 0.0f ? C / M : 0.0f;

    float h = 0.0f;
    if (s != 0.0f) {
        C = 1.0f / C;
        float rc = (M - rgb.x) * C;
        float gc = (M - rgb.y) * C;
        float bc = (M - rgb.z) * C;

        if      (rgb.x == M)  h = bc - gc;
        else if (rgb.y == M)  h = 2 + rc - bc;
        else                  h = 4 + gc - rc;

        if (h < 0) h += 6.0f;
    }
    return make_float3(h, s, M);
}

__device__ float3 hsv2rgb(float3 hsv) {
    float whole = floorf(hsv.x);
    float frac = hsv.x - whole;
    float val = hsv.z;
    float min = val * (1 - hsv.y);
    float mid = val * (1 - (hsv.y * frac));
    float alt = val * (1 - (hsv.y * (1 - frac)));

    float3 out;
         if (whole == 0.0f) { out.x = val; out.y = alt; out.z = min; }
    else if (whole == 1.0f) { out.x = mid; out.y = val; out.z = min; }
    else if (whole == 2.0f) { out.x = min; out.y = val; out.z = alt; }
    else if (whole == 3.0f) { out.x = min; out.y = mid; out.z = val; }
    else if (whole == 4.0f) { out.x = alt; out.y = min; out.z = val; }
    else                    { out.x = val; out.y = min; out.z = mid; }
    return out;
}
''')
