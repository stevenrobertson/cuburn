from util import devlib, ringbuflib
from mwc import mwclib

f32tou8lib = devlib(deps=[ringbuflib, mwclib], defs=r'''
// Perform a conversion from float32 values to uint8 ones, applying
// pixel- and channel-independent dithering to reduce suprathreshold banding
// artifacts. Clamps values larger than 1.0f.
// TODO: move to a separate module?
// TODO: less ineffecient mwc_st handling?
__global__ void f32_to_u8(
    ringbuf *rb, mwc_st *rctxs, uchar4 *dst, const float4 *src,
    int gutter, int dstride, int sstride, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > dstride || y > height) return;
    int isrc = sstride * (y + gutter) + x + gutter;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    mwc_st rctx = rctxs[rb_incr(rb->head, tid)];

    float4 in = src[isrc];
    uchar4 out = make_uchar4(
        fminf(1.0f, in.x) * 255.0f + 0.49f * mwc_next_11(rctx),
        fminf(1.0f, in.y) * 255.0f + 0.49f * mwc_next_11(rctx),
        fminf(1.0f, in.z) * 255.0f + 0.49f * mwc_next_11(rctx),
        fminf(1.0f, in.w) * 255.0f + 0.49f * mwc_next_11(rctx)
    );

    int idst = dstride * y + x;
    dst[idst] = out;
    rctxs[rb_incr(rb->head, tid)] = rctx;
}
''')
