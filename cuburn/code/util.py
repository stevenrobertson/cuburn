"""
Provides tools and miscellaneous functions for building device code.
"""

import numpy as np
import tempita

def crep(s):
    """Escape for PTX assembly"""
    return '"%s"' % s.encode("string_escape")

class Template(tempita.Template):
    default_namespace = tempita.Template.default_namespace.copy()
Template.default_namespace.update({'np': np, 'crep': crep})

class HunkOCode(object):
    """An apparently passive container for device code."""
    # Use property objects to make these dynamic
    headers = ''
    decls = ''
    defs = ''

def assemble_code(*sections):
    return ''.join([''.join([getattr(sect, kind) for sect in sections])
                    for kind in ['headers', 'decls', 'defs']])

def apply_affine(x, y, xo, yo, packer):
    return Template("""
    {{xo}} = {{packer.xx}} * {{x}} + {{packer.xy}} * {{y}} + {{packer.xo}};
    {{yo}} = {{packer.yx}} * {{x}} + {{packer.yy}} * {{y}} + {{packer.yo}};
    """).substitute(locals())

class BaseCode(HunkOCode):
    headers = """
#include<cuda.h>
#include<stdint.h>
#include<stdio.h>
"""

    decls = """
float3 rgb2hsv(float3 rgb);
float3 hsv2rgb(float3 hsv);
"""

    defs = Template(r"""
#undef M_E
#undef M_LOG2E
#undef M_LOG10E
#undef M_LN2
#undef M_LN10
#undef M_PI
#undef M_PI_2
#undef M_PI_4
#undef M_1_PI
#undef M_2_PI
#undef M_2_SQRTPI
#undef M_SQRT2
#undef M_SQRT1_2

#define  M_E          2.71828174591064f
#define  M_LOG2E      1.44269502162933f
#define  M_LOG10E     0.43429449200630f
#define  M_LN2        0.69314718246460f
#define  M_LN10       2.30258512496948f
#define  M_PI         3.14159274101257f
#define  M_PI_2       1.57079637050629f
#define  M_PI_4       0.78539818525314f
#define  M_1_PI       0.31830987334251f
#define  M_2_PI       0.63661974668503f
#define  M_2_SQRTPI   1.12837922573090f
#define  M_SQRT2      1.41421353816986f
#define  M_SQRT1_2    0.70710676908493f

// TODO: use launch parameter preconfig to eliminate unnecessary parts
__device__
uint32_t gtid() {
    return threadIdx.x + blockDim.x *
            (threadIdx.y + blockDim.y *
                (threadIdx.z + blockDim.z *
                    (blockIdx.x + (gridDim.x * blockIdx.y))));
}


/* Returns the ID of this thread on the device. Note that this counter is
 * volatile according to the PTX ISA. It should be used for loading and saving
 * state that must be unique across running threads, not for accessing things
 * in a known order. */
__device__
int devtid() {
    int result;
    asm({{crep('''
    {
        .reg .u32   tmp1, tmp2;
        mov.u32     %0,     %smid;
        mov.u32     tmp1,   %nsmid;
        mov.u32     tmp2,   %warpid;
        mad.lo.u32  %0,     %0,     tmp1,   tmp2;
        mov.u32     tmp1,   %nwarpid;
        mov.u32     tmp2,   %laneid;
        mad.lo.u32  %0,     %0,     tmp1,   tmp2;
    }''')}} : "=r"(result) );
    return result;
}

__device__
uint32_t trunca(float f) {
    // truncate as used in address calculations. note the use of a signed
    // conversion is intentional here (simplifies image bounds checking).
    uint32_t ret;
    asm("cvt.rni.s32.f32    %0,     %1;" : "=r"(ret) : "f"(f));
    return ret;
}

__global__
void fill_dptr(uint32_t* dptr, int size, uint32_t value) {
    int i = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        dptr[i] = value;
    }
}

/* read_half and write_half decode and encode, respectively, two
 * floating-point values from a 32-bit value (typed as a 'float' for
 * convenience but not really). The values are packed into u16s as a
 * proportion of a third value, as in 'ux = u16( x / d * (2^16-1) )'.
 * This is used during accumulation.
 *
 * TODO: also write a function that will efficiently add a value to the packed
 * values while incrementing the density, to improve the speed of this
 * approach when the alpha channel is present.
 */

__device__
void read_half(float &x, float &y, float xy, float den) {
    asm("\n\t{"
        "\n\t   .reg .u16       x, y;"
        "\n\t   .reg .f32       rc;"
        "\n\t   mov.b32         {x, y},     %2;"
        "\n\t   mul.f32         rc,         %3,     0f37800080;" // 1/65535.
        "\n\t   cvt.rn.f32.u16     %0,         x;"
        "\n\t   cvt.rn.f32.u16     %1,         y;"
        "\n\t   mul.f32         %0,         %0,     rc;"
        "\n\t   mul.f32         %1,         %1,     rc;"
        "\n\t}"
        : "=f"(x), "=f"(y) : "f"(xy), "f"(den));
}

__device__
void write_half(float &xy, float x, float y, float den) {
    asm("\n\t{"
        "\n\t   .reg .u16       x, y;"
        "\n\t   .reg .f32       rc, xf, yf;"
        "\n\t   rcp.approx.f32  rc,         %3;"
        "\n\t   mul.f32         rc,         rc,     65535.0;"
        "\n\t   mul.f32         xf,         %1,     rc;"
        "\n\t   mul.f32         yf,         %2,     rc;"
        "\n\t   cvt.rni.u16.f32 x,  xf;"
        "\n\t   cvt.rni.u16.f32 y,  yf;"
        "\n\t   mov.b32         %0,         {x, y};"
        "\n\t}"
        : "=f"(xy) : "f"(x), "f"(y), "f"(den));
}

__device__
float3 rgb2hsv(float3 rgb) {
    float M = fmaxf(fmaxf(rgb.x, rgb.y), rgb.z);
    float m = fminf(fminf(rgb.x, rgb.y), rgb.z);
    float C = M - m;

    float s = M > 0.0f ? C / M : 0.0f;

    float h;
    if (s != 0.0f) {
        C = 1.0f / C;
        float rc = (M - rgb.x) * C;
        float gc = (M - rgb.y) * C;
        float bc = (M - rgb.z) * C;

        if      (rgb.x == M)  h = bc - gc;
        else if (rgb.y == M)  h = 2 + rc - bc;
        else                  h = 4 + gc - rc;

        if (h < 0) h += 6;
    }
    return make_float3(h, s, M);
}

__device__
float3 hsv2rgb(float3 hsv) {

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
""").substitute()

    @staticmethod
    def fill_dptr(mod, dptr, size, stream=None, value=np.uint32(0)):
        """
        A memory zeroer which can be embedded in a stream, unlike the various
        memset routines. Size is the number of 4-byte words in the pointer;
        value is the word to fill it with. If value is not an np.uint32, it
        will be coerced to a buffer and the first four bytes taken.
        """
        fill = mod.get_function("fill_dptr")
        if not isinstance(value, np.uint32):
            if isinstance(value, int):
                value = np.uint32(value)
            else:
                value = np.frombuffer(buffer(value), np.uint32)[0]
        blocks = int(np.ceil(np.sqrt(size / 1024 + 1)))
        fill(dptr, np.int32(size), value, stream=stream,
             block=(1024, 1, 1), grid=(blocks, blocks, 1))


