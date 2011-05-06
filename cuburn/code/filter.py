
from cuburn.code.util import *

class ColorClip(HunkOCode):
    defs = """
__global__
void colorclip(float4 *pixbuf, float gamma, float vibrancy, float highpow) {
    // TODO: test if over an edge of the framebuffer
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float4 pix = pixbuf[i];

    if (pix.w <= 0) return;

    float4 opix = pix;

    // TODO: linearized bottom range
    float alpha = powf(pix.w, gamma);
    float ls = vibrancy * alpha / pix.w;

    float maxc = fmaxf(pix.x, fmaxf(pix.y, pix.z));
    float newls = 1 / maxc;

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
    NOTE: for now, this *must* be invoked with a block size of (32,32,1), and
    a grid size of (W/32,1). At least 7 pixel gutters are required, and the
    stride and height probably need to be multiples of 32.
    """

    # Note, changing this does not yet have any effect, it's just informational
    MAX_WIDTH=15

    def __init__(self, features, cp):
        self.features, self.cp = features, cp

    headers = "#include<math_constants.h>\n"
    @property
    def defs(self):
        return self.defs_tmpl.substitute(features=self.features, cp=self.cp)

    defs_tmpl = Template("""
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

    float ls = fmaxf(0, k1 * logf(1.0 + pix.w * k2) / pix.w);
    pix.x *= ls;
    pix.y *= ls;
    pix.z *= ls;
    pix.w *= ls;

    outbuf[i] = pix;
}

__global__
void density_est(float4 *pixbuf, float4 *outbuf, float *denbuf,
                 float k1, float k2) {
    for (int i = threadIdx.x + 32*threadIdx.y; i < FW2; i += 32)
        de_r[i] = de_g[i] = de_b[i] = de_a[i] = 0.0f;
    __syncthreads();

    for (int imrow = threadIdx.y + W2; imrow < {{features.acc_height}}; imrow += 32)
    {
        int idx = {{features.acc_stride}} * imrow +
                + blockIdx.x * 32 + threadIdx.x + W2;

        float4 in = pixbuf[idx];
        float den = denbuf[idx];

        if (in.w > 0 && den > 0) {
            float ls = k1 * 12 * logf(1.0 + in.w * k2) / in.w;
            in.x *= ls;
            in.y *= ls;
            in.z *= ls;
            in.w *= ls;

            // Calculate standard deviation of Gaussian kernel.
            // flam3_gaussian_filter() uses an implicit standard deviation of 0.5,
            // but the DE filters scale filter distance by the default spatial
            // support factor of 1.5, so the effective base SD is (0.5/1.5)=1/3.
            float sd = {{cp.estimator / 3.}};

            // The base SD is then scaled in inverse proportion to the density of
            // the point being scaled.
            sd *= powf(den+1.0f, {{-cp.estimator_curve}});

            // Clamp the final standard deviation. Things will go badly if the
            // minimum is undershot.
            sd = fmaxf(sd, {{max(cp.estimator_minimum / 3., 0.3)}} );

            // This five-term polynomial approximates the sum of the filters with
            // the clamping logic used here. See helpers/filt_err.py.
            float filtsum;
            filtsum = -0.20885075f  * sd +  0.90557721f;
            filtsum = filtsum       * sd +  5.28363054f;
            filtsum = filtsum       * sd + -0.11733533f;
            filtsum = filtsum       * sd +  0.35670333f;
            float filtscale = 1 / filtsum;

            // The reciprocal SD scaling coeffecient in the Gaussian exponent.
            // exp(-x^2/(2*sd^2)) = exp2f(x^2*rsd)
            float rsd = -0.5f * CUDART_L2E_F / (sd * sd);

            int si = (threadIdx.y + W2) * FW + threadIdx.x + W2;
            for (int jj = 0; jj <= W2; jj++) {
                float jj2f = jj;
                jj2f *= jj2f;

                float iif = 0;
                for (int ii = 0; ii <= jj; ii++) {
                    iif += 1;

                    float coeff = exp2f((jj2f + iif * iif) * rsd) * filtscale;
                    if (coeff < 0.0001f) break;

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

                    // TODO: validate that the above avoids bank conflicts
                }
            }
        }

        __syncthreads();
        // TODO: could coalesce this, but what a pain
        for (int i = threadIdx.x; i < FW; i += 32) {
            idx = {{features.acc_stride}} * imrow + blockIdx.x * 32 + i + W2;
            int si = threadIdx.y * FW + i;
            float *out = reinterpret_cast<float*>(&outbuf[idx]);
            atomicAdd(out,   de_r[si]);
            atomicAdd(out+1, de_g[si]);
            atomicAdd(out+2, de_b[si]);
            atomicAdd(out+3, de_a[si]);
        }

        if (threadIdx.y == 5000) {
            for (int i = threadIdx.x; i < FW; i += 32) {
                idx = {{features.acc_stride}} * (imrow + 32)
                    + blockIdx.x * 32 + i + W2;
                int si = 32 * FW + i;
                float *out = reinterpret_cast<float*>(&outbuf[idx]);
                atomicAdd(out,   0.2 + de_r[si]);
                atomicAdd(out+1, de_g[si]);
                atomicAdd(out+2, de_b[si]);
                atomicAdd(out+3, de_a[si]);
            }
        }

        __syncthreads();
        // TODO: shift instead of copying
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

""")

    def invoke(self, mod, abufd, obufd, dbufd):
        # TODO: add no-est version
        # TODO: come up with a general way to average these parameters
        k1 = self.cp.brightness * 268 / 256
        area = self.features.width * self.features.height / self.cp.ppu ** 2
        k2 = 1 / (area * self.cp.adj_density)

        if self.cp.estimator == 0:
            fun = mod.get_function("logscale")
            t = fun(abufd, obufd, np.float32(k1), np.float32(k2),
                    block=(self.features.acc_width, 1, 1),
                    grid=(self.features.acc_height, 1), time_kernel=True)
        else:
            fun = mod.get_function("density_est")
            t = fun(abufd, obufd, dbufd, np.float32(k1), np.float32(k2),
                    block=(32, 32, 1), grid=(self.features.acc_stride/32 - 1, 1),
                    time_kernel=True)
            print "Density estimation: %g" % t

