from util import devlib, ringbuflib
from mwc import mwclib

pixfmtlib = devlib(deps=[ringbuflib, mwclib], defs=r'''
// Clamp an input between 0 and a given peak (inclusive), dithering its output,
// with full clamping for pixels that are true-black for compressibility.
__device__ float dclampf(mwc_st &rctx, float peak, float in) {
  float ret = 0.0f;
  if (in > 0.0f) {
    ret = fminf(peak, in * peak + 0.99f * mwc_next_01(rctx));
  }
  return ret;
}

// Perform a conversion from float32 values to uint8 ones, applying
// pixel- and channel-independent dithering to reduce suprathreshold banding
// artifacts. Clamps values larger than 1.0f.
// TODO: move to a separate module?
// TODO: less ineffecient mwc_st handling?
__global__ void f32_to_rgba_u8(
    uchar4 *dst, const float4 *src,
    int gutter, int dstride, int sstride, int height,
    ringbuf *rb, mwc_st *rctxs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > dstride || y > height) return;
    int isrc = sstride * (y + gutter) + x + gutter;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    mwc_st rctx = rctxs[rb_incr(rb->head, tid)];

    float4 in = src[isrc];
    uchar4 out = make_uchar4(
        dclampf(rctx, 255.0f, in.x),
        dclampf(rctx, 255.0f, in.y),
        dclampf(rctx, 255.0f, in.z),
        dclampf(rctx, 255.0f, in.w)
    );

    int idst = dstride * y + x;
    dst[idst] = out;
    rctxs[rb_incr(rb->tail, tid)] = rctx;
}

// Perform a conversion from float32 values to uint16 ones, as above.
__global__ void f32_to_rgba_u16(
    ushort4 *dst, const float4 *src,
    int gutter, int dstride, int sstride, int height,
    ringbuf *rb, mwc_st *rctxs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > dstride || y > height) return;
    int isrc = sstride * (y + gutter) + x + gutter;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    mwc_st rctx = rctxs[rb_incr(rb->head, tid)];

    float4 in = src[isrc];
    ushort4 out = make_ushort4(
        dclampf(rctx, 65535.0f, in.x),
        dclampf(rctx, 65535.0f, in.y),
        dclampf(rctx, 65535.0f, in.z),
        dclampf(rctx, 65535.0f, in.w)
    );

    int idst = dstride * y + x;
    dst[idst] = out;
    rctxs[rb_incr(rb->tail, tid)] = rctx;
}

// Convert from rgb444 to planar YUV with no chroma subsampling.
// Uses JPEG full-range color primaries.
__global__ void f32_to_yuv444p(
    char *dst, const float4 *src,
    int gutter, int dstride, int sstride, int height,
    ringbuf *rb, mwc_st *rctxs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > dstride || y > height) return;
    int isrc = sstride * (y + gutter) + x + gutter;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    mwc_st rctx = rctxs[rb_incr(rb->head, tid)];

    float4 in = src[isrc];
    uchar3 out = make_uchar3(
        dclampf(rctx, 255.0f, 0.299f      * in.x + 0.587f     * in.y + 0.114f     * in.z),
        dclampf(rctx, 255.0f, -0.168736f  * in.x - 0.331264f  * in.y + 0.5f       * in.z + 0.5f),
        dclampf(rctx, 255.0f, 0.5f        * in.x - 0.418688f  * in.y - 0.081312f  * in.z + 0.5f)
    );

    int idst = dstride * y + x;
    dst[idst] = out.x;
    idst += dstride * height;
    dst[idst] = out.y;
    idst += dstride * height;
    dst[idst] = out.z;
    rctxs[rb_incr(rb->tail, tid)] = rctx;
}

// Convert from rgb444 to planar YUV 10-bit, using JPEG full-range primaries.
// TODO(strobe): Decide how YouTube will handle Rec. 2020, and then do that here.
__global__ void f32_to_yuv444p10(
    uint16_t *dst, const float4 *src,
    int gutter, int dstride, int sstride, int height,
    ringbuf *rb, mwc_st *rctxs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > dstride || y > height) return;
    int isrc = sstride * (y + gutter) + x + gutter;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    mwc_st rctx = rctxs[rb_incr(rb->head, tid)];

    float4 in = src[isrc];
    ushort3 out = make_ushort3(
        dclampf(rctx, 1023.0f, 0.299f      * in.x + 0.587f     * in.y + 0.114f     * in.z),
        dclampf(rctx, 1023.0f, -0.168736f  * in.x - 0.331264f  * in.y + 0.5f       * in.z + 0.5f),
        dclampf(rctx, 1023.0f, 0.5f        * in.x - 0.418688f  * in.y - 0.081312f  * in.z + 0.5f)
    );

    int idst = dstride * y + x;
    dst[idst] = out.x;
    idst += dstride * height;
    dst[idst] = 1023.0f * (-0.168736f  * in.x - 0.331264f  * in.y + 0.5f       * in.z + 0.5f);
    idst += dstride * height;
    dst[idst] = out.z;

    rctxs[rb_incr(rb->tail, tid)] = rctx;
}

// Convert from rgb444 to planar YUV 10-bit, using JPEG full-range primaries.
// Perform subsampling of chroma using weighted averages.
__global__ void f32_to_yuv420p10(
    uint16_t *dst, const float4 *src,
    int gutter, int dstride, int sstride, int height,
    ringbuf *rb, mwc_st *rctxs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > dstride || y > height) return;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    mwc_st rctx = rctxs[rb_incr(rb->head, tid)];

    // Perform luma using real addressing
    int isrc = sstride * (y + gutter) + x + gutter;
    int idst = dstride * y + x;
    float4 in = src[isrc];
    dst[idst] = dclampf(rctx, 1023.0f, 0.299f      * in.x + 0.587f     * in.y + 0.114f     * in.z);

    // Drop into subsampling mode for chroma components
    if (x * 2 > dstride || y * 2 > height) return;

    // Recompute addressing and collect weighted averages
    // TODO(strobe): characterize overflow here
    isrc = sstride * (y * 2 + gutter) + x * 2 + gutter;
    in = src[isrc];
    float sum = in.w + 1e-12;
    float cb = in.w * (-0.168736f  * in.x - 0.331264f  * in.y + 0.5f       * in.z);
    float cr = in.w * (0.5f        * in.x - 0.418688f  * in.y - 0.081312f  * in.z);

    in = src[isrc + 1];
    sum += in.w;
    cb += in.w * (-0.168736f  * in.x - 0.331264f  * in.y + 0.5f       * in.z);
    cr += in.w * (0.5f        * in.x - 0.418688f  * in.y - 0.081312f  * in.z);

    isrc += sstride;
    in = src[isrc];
    sum += in.w;
    cb += in.w * (-0.168736f  * in.x - 0.331264f  * in.y + 0.5f       * in.z);
    cr += in.w * (0.5f        * in.x - 0.418688f  * in.y - 0.081312f  * in.z);

    in = src[isrc + 1];
    sum += in.w;
    cb += in.w * (-0.168736f  * in.x - 0.331264f  * in.y + 0.5f       * in.z);
    cr += in.w * (0.5f        * in.x - 0.418688f  * in.y - 0.081312f  * in.z);

    // For this to work, dstride must equal the output frame width
    // and be a multiple of four.
    idst = dstride * height + dstride / 2 * y + x;
    dst[idst] = dclampf(rctx, 1023.0f, cb / sum + 0.5f);
    idst += dstride * height / 4;
    dst[idst] = dclampf(rctx, 1023.0f, cr / sum + 0.5f);

    rctxs[rb_incr(rb->tail, tid)] = rctx;
}

// Convert from rgb444 to planar YUV 12-bit studio swing,
// using the Rec. 709 matrix.
__global__ void f32_to_yuv444p12(
    uint16_t *dst, const float4 *src,
    int gutter, int dstride, int sstride, int height,
    ringbuf *rb, mwc_st *rctxs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > dstride || y > height) return;
    int isrc = sstride * (y + gutter) + x + gutter;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    mwc_st rctx = rctxs[rb_incr(rb->head, tid)];

    float4 in = src[isrc];
    in.x = fminf(1.0f, fmaxf(0.0f, in.x));
    in.y = fminf(1.0f, fmaxf(0.0f, in.y));
    in.z = fminf(1.0f, fmaxf(0.0f, in.z));
    ushort3 out = make_ushort3(
        dclampf(rctx, 3504.0f, 0.2126f   * in.x + 0.7152f  * in.y + 0.0722f   * in.z) + 256.0f,
        dclampf(rctx, 3584.0f, -0.11457f * in.x - 0.38543f * in.y + 0.5f      * in.z + 0.5f) + 256.0f,
        dclampf(rctx, 3584.0f, 0.5f      * in.x - 0.45416f * in.y - 0.04585f  * in.z + 0.5f) + 256.0f
    );

    int idst = dstride * y + x;
    dst[idst] = out.x;
    idst += dstride * height;
    dst[idst] = out.y;
    idst += dstride * height;
    dst[idst] = out.z;

    rctxs[rb_incr(rb->tail, tid)] = rctx;
}
''')

