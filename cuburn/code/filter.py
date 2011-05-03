
from cuburn.code.util import *

class ColorClip(HunkOCode):
    defs = """
__global__
void logfilt(float4 *pixbuf, float k1, float k2,
             float gamma, float vibrancy, float highpow) {
    // TODO: test if over an edge of the framebuffer
    int i = 1024 * blockIdx.x + threadIdx.x;
    float4 pix = pixbuf[i];

    if (pix.w <= 0) return;

    float ls = k1 * logf(1.0 + pix.w * k2) / pix.w;
    pix.x *= ls;
    pix.y *= ls;
    pix.z *= ls;
    pix.w *= ls;

    float4 opix = pix;

    // TODO: linearized bottom range
    float alpha = powf(pix.w, gamma);
    ls = vibrancy * alpha / pix.w;

    float maxc = fmaxf(pix.x, fmaxf(pix.y, pix.z));
    float newls = 1 / maxc;

    // TODO: detect if highlight power is globally disabled and drop
    // this branch

    if (maxc * ls > 1 && highpow >= 0) {
        // TODO: does CUDA autopromote the int here to a float before GPU?
        float lsratio = powf(newls / ls, highpow);

        pix.x *= newls;
        pix.y *= newls;
        pix.z *= newls;
        maxc  *= newls;

        // Reduce saturation (according to the HSV model) by proportionally
        // increasing the values of the other colors.

        pix.x = maxc - (maxc - pix.x) * lsratio;
        pix.y = maxc - (maxc - pix.y) * lsratio;
        pix.z = maxc - (maxc - pix.z) * lsratio;

    } else {
        highpow = -highpow;
        if (highpow > 1 || maxc * ls <= 1) highpow = 1;
        float adj = ((1.0 - highpow) * newls + highpow * ls);
        pix.x *= adj;
        pix.y *= adj;
        pix.z *= adj;
    }

    pix.x = fminf(1.0, pix.x + (1.0 - vibrancy) * powf(opix.x, gamma));
    pix.y = fminf(1.0, pix.y + (1.0 - vibrancy) * powf(opix.y, gamma));
    pix.z = fminf(1.0, pix.z + (1.0 - vibrancy) * powf(opix.z, gamma));

    pixbuf[i] = pix;
}
"""


