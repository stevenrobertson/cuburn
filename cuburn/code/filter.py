
from cuburn.code.util import *

class ColorClip(HunkOCode):
    defs = """
__global__
void logfilt(float4 *pixbuf, float k1, float k2,
             float gamma, float vibrancy, float highpow) {
    // TODO: test if over an edge of the framebuffer
    int i = blockDim.x * blockIdx.x + threadIdx.x;
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

class DensityEst(HunkOCode):
    """
    NOTE: for now, this *must* be invoked with a block size of (32,16,1), and
    a grid size of (W/32) for vertical filtering or (H/32) for horizontal.

    It will probably fail for images that are not multiples of 32.
    """

    def __init__(self, features, cp):
        self.features, self.cp = features, cp

    @property
    def defs(self):
        return self.defs_tmpl.substitute(features=self.features, cp=self.cp)

    defs_tmpl = Template("""
#define W 15        // Filter width (regardless of standard deviation chosen)
#define W2 7        // Half of filter width, rounded down
#define NW 16       // Number of warps in each set of points
#define FW 30       // Width of local result storage per-lane (NW+W2+W2)
#define BX 32       // The size of a block's X dimension (== 1 warp)

__global__
void density_est(float4 *pixbuf, float *denbuf, int vertical) {
    __shared__ float r[BX*FW], g[BX*FW], b[BX*FW], a[BX*FW];

    int ibase;    // The index of the first element within this lane's strip
    int imax;     // The maximum offset from the first element in the strip
    int istride;  // Number of indices until next point to filter

    if (vertical) {
        ibase = threadIdx.x + blockIdx.x * BX;
        imax = {{features.acc_height}};
        istride = {{features.acc_stride}};
    } else {
        ibase = (blockIdx.x * BX + threadIdx.x) * {{features.acc_stride}};
        imax = {{features.acc_width}};
        istride = 1;
    }

    for (int i = threadIdx.x + BX*threadIdx.y; i < BX*FW; i += NW * BX)
        r[i] = g[i] = b[i] = a[i] = 0.0f;

    for (int i = threadIdx.y; i < imax; i += NW) {
        int idx = ibase+i*istride;
        float4 in = pixbuf[idx];
        float den = denbuf[idx];

        int j = (threadIdx.y + W2) * 32 + threadIdx.x;

        float sd = {{0.35 * cp.estimator}} / powf(den+1.0f, {{cp.estimator_curve}});
        {{if cp.estimator_minimum > 1}}
        sd = fmaxf(sd, {{cp.estimator_minimum}});
        {{endif}}
        sd *= sd;

        // TODO: log scaling here? matches flam3, but, ick
        // TODO: investigate harm caused by varying standard deviation in a
        // separable environment
        float coeff = rsqrtf(2.0f*M_PI*sd*sd);
        atomicAdd(r+j, in.x * coeff);
        atomicAdd(g+j, in.y * coeff);
        atomicAdd(b+j, in.z * coeff);
        atomicAdd(a+j, in.w * coeff);
        sd = -0.5/sd;

        // #pragma unroll
        for (int k = 1; k <= W2; k++) {
            float scale = exp(sd*k*k)*coeff;
            idx = j+k*32;
            atomicAdd(r+idx, in.x * scale);
            atomicAdd(g+idx, in.y * scale);
            atomicAdd(b+idx, in.z * scale);
            atomicAdd(a+idx, in.w * scale);
            idx = j-k*32;
            atomicAdd(r+idx, in.x * scale);
            atomicAdd(g+idx, in.y * scale);
            atomicAdd(b+idx, in.z * scale);
            atomicAdd(a+idx, in.w * scale);
        }

        __syncthreads();
        float4 out;
        j = threadIdx.y * BX + threadIdx.x;
        out.x = r[j];
        out.y = g[j];
        out.z = b[j];
        out.w = a[j];
        idx = ibase+(i-W2)*istride;
        if (idx > 0)
            pixbuf[idx] = out;
        __syncthreads();

        // TODO: shift instead of copying
        idx = threadIdx.x + BX * threadIdx.y;
        if (threadIdx.y < NW-2) {
            r[idx] = r[idx + BX*NW];
            g[idx] = g[idx + BX*NW];
            b[idx] = b[idx + BX*NW];
            a[idx] = a[idx + BX*NW];
        }
        __syncthreads();

        r[idx + BX*(NW-2)] = 0.0f;
        g[idx + BX*(NW-2)] = 0.0f;
        b[idx + BX*(NW-2)] = 0.0f;
        a[idx + BX*(NW-2)] = 0.0f;
        __syncthreads();
    }
}

""")

    def invoke(self, mod, abufd, dbufd):
        fun = mod.get_function("density_est")
        t = fun(abufd, dbufd, np.int32(0),
                block=(32, 16, 1), grid=(self.features.acc_height/32,1),
                time_kernel=True)
        print "Horizontal density estimation: %g" % t

        t = fun(abufd, dbufd, np.int32(1),
                block=(32, 16, 1), grid=(self.features.acc_width/32,1),
                time_kernel=True)
        print "Vertical density estimation: %g" % t
