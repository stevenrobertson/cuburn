from util import devlib, ringbuflib
from mwc import mwclib

ditherlib = devlib(deps=[mwclib], defs=r'''
// Clamp an input between 0 and a given peak (inclusive), dithering its output,
// with full clamping for pixels that are true-black for compressibility.
__device__ float dclampf(mwc_st &rctx, float peak, float in) {
  float ret = 0.0f;
  if (in > 0.0f) {
    ret = fminf(peak, fmaxf(0.0f, in * peak + 0.49f * mwc_next_11(rctx)));
  }
  return ret;
}
''')

rgba8lib = devlib(deps=[ringbuflib, mwclib, ditherlib], defs=r'''
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
''')

rgba16lib = devlib(deps=[ringbuflib, mwclib, ditherlib], defs=r'''
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
''')

pixfmtlib = devlib(deps=[rgba8lib, rgba16lib])
