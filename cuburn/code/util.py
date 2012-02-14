"""
Provides tools and miscellaneous functions for building device code.
"""
import os
import tempfile
from collections import namedtuple

import pycuda.driver as cuda
import pycuda.compiler
import numpy as np
import tempita

def argset(obj, **kwargs):
    """
    Allow an object with many properties to be set using one call.

    >>> x = argset(X(), a=1, b=2)
    >>> x.a
    1
    """
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj

def launch(name, mod, stream, block, grid, *args, **kwargs):
    """
    Oft-used kernel launch helper. Provides a nice boost in readability of
    densely-packed asynchronous code launches.
    """
    fun = mod.get_function(name)
    if isinstance(block, (int, np.number)):
        block = (int(block), 1, 1)
    if isinstance(grid, (int, np.number)):
        grid = (int(grid), 1)
    fun(*args, block=block, grid=grid, stream=stream, **kwargs)

def crep(s):
    """Multiline literal escape for inline PTX assembly."""
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    return '"%s"' % s.encode("string_escape")

class Template(tempita.Template):
    """
    A Tempita template with extra stuff in the default namespace.
    """
    default_namespace = tempita.Template.default_namespace.copy()
Template.default_namespace.update({'np': np, 'crep': crep})

# Passive container for device code.
DevLib = namedtuple('DevLib', 'deps headers decls defs')

def devlib(deps=(), headers='', decls='', defs=''):
    """Create a library of device code."""
    # This exists because namedtuple doesn't support default args
    return DevLib(deps, headers, decls, defs)

def assemble_code(*libs):
    seen = set()
    out = []
    def go(lib):
        map(go, lib.deps)
        code = lib[1:]
        if code not in seen:
            seen.add(code)
            out.append(code)
    go(stdlib)
    map(go, libs)
    return ''.join(sum(zip(*out), ()))

DEFAULT_CMP_OPTIONS = ('-use_fast_math', '-maxrregcount', '42')
DEFAULT_SAVE_KERNEL = True
def compile(name, src, opts=DEFAULT_CMP_OPTIONS, save=DEFAULT_SAVE_KERNEL):
    """
    Compile a module. Returns a copy of the source (for inspection or
    display) and the compiled cubin.
    """
    dir = tempfile.gettempdir()
    if save:
        with open(os.path.join(dir, name + '_kern.cu'), 'w') as fp:
            fp.write(src)
    cubin = pycuda.compiler.compile(src, options=list(opts))
    if save:
        with open(os.path.join(dir, name + '_kern.cubin'), 'w') as fp:
            fp.write(cubin)
    return cubin

class ClsMod(object):
    """
    Convenience class or mixin that automatically compiles and loads a module
    once per class, saving potentially expensive code regeneration. Only use
    if your class does not employ run-time code generation.
    """
    mod = None
    # Supply your own DevLib on this property
    lib = None

    def __init__(self):
        super(ClsMod, self).__init__()
        self.load()

    @classmethod
    def load(cls, name=None):
        if cls.mod is None:
            if name is None:
                name = cls.__name__.lower()
            cubin = compile(name, assemble_code(cls.lib))
            cls.mod = cuda.module_from_buffer(cubin)

# This lib is included with *every* assembled module. It contains no global
# functions, so it shouldn't slow down compilation time too much.
stdlib = devlib(headers="""
#include<cuda.h>
#include<stdint.h>
#include<stdio.h>
""", decls=r"""
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

#define bfe(d, s, o, w) \
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(s), "r"(o), "r"(w))

#define bfe_decl(d, s, o, w) \
        int d; \
        bfe(d, s, o, w)
""", defs=r'''
__device__ uint32_t gtid() {
    return threadIdx.x + blockDim.x *
            (threadIdx.y + blockDim.y *
                (threadIdx.z + blockDim.z *
                    (blockIdx.x + (gridDim.x * blockIdx.y))));
}

__device__ uint32_t trunca(float f) {
    // truncate as used in address calculations. note the use of a signed
    // conversion is intentional here (simplifies image bounds checking).
    uint32_t ret;
    asm("cvt.rni.s32.f32    %0,     %1;" : "=r"(ret) : "f"(f));
    return ret;
}
''')

def mkbinsearchlib(rounds):
    """
    Search through the fixed-size list 'hay' to find the rightmost index which
    contains a value strictly smaller than the input 'needle'. The list must
    be exactly '2^rounds' long, although padding at the top with very large
    numbers or even +inf effectively shortens it.
    """
    # TODO: this doesn't optimize well on a 64-bit arch, not that it's a
    # performance-critical chunk of code or anything
    src = Template(r'''
__device__ int bitwise_binsearch(const float *hay, float needle) {
    int lo = 0;

    // TODO: improve efficiency on 64-bit arches
    {{for i in range(search_rounds-1, -1, -1)}}
    if (needle > hay[lo + {{1 << i}}])
        lo += {{1 << i}};
    {{endfor}}
    return lo;
}
''', 'bitwise_binsearch')
    return devlib(defs=src.substitute(search_rounds=rounds))

# 2^search_rounds is the maximum number of knots allowed in a single spline.
# This includes the four required knots, so a 5 round search supports 28
# interior knots in the domain (0, 1). 2^5 fits nicely on a single cache line.
DEFAULT_SEARCH_ROUNDS = 5
binsearchlib = mkbinsearchlib(DEFAULT_SEARCH_ROUNDS)


filldptrlib = devlib(defs=r'''
__global__ void
fill_dptr(uint32_t* dptr, int size, uint32_t value) {
    int i = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        dptr[i] = value;
    }
}
''')
def fill_dptr(mod, dptr, size, stream=None, value=np.uint32(0)):
    """
    A memory zeroer which can be embedded in a stream, unlike the various
    memset routines. Size is the number of 4-byte words in the pointer;
    value is the word to fill it with. If value is not an np.uint32, it
    will be coerced to a buffer and the first four bytes taken.
    """
    if not isinstance(value, np.uint32):
        if isinstance(value, int):
            value = np.uint32(value)
        else:
            value = np.frombuffer(buffer(value), np.uint32)[0]
    blocks = int(np.ceil(np.sqrt(size / 1024.)))
    launch('fill_dptr', mod, stream, (1024, 1, 1), (blocks, blocks),
            dptr, np.int32(size), value)

writehalflib = devlib(defs=r'''
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

__device__ void
read_half(float &x, float &y, float xy, float den) {
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

__device__ void
write_half(float &xy, float x, float y, float den) {
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
''')


def mkringbuflib(rb_size):
    """
    A ringbuffer for access to shared resources.

    Some components, such as the MWC contexts, are expensive to generate, and
    have no affinity to a particular block. Rather than maintain a separate
    copy of each of these objects for every thread block in a launch, we want
    to only keep enough copies of these resources around to service every
    thread block that could possibly be active simultaneously on one card,
    which is often considerably smaller. This class provides a simple
    ringbuffer type and an increment function, used in a couple places to
    implement this kind of sharing.
    """
    return devlib(headers="#define RB_SIZE_MASK %d" % (rb_size - 1), decls='''
typedef struct {
    int head;
    int tail;
} ringbuf;
''', defs=r'''
__shared__ int rb_idx;
__device__ int rb_incr(int &rb_base, int tidx) {
    if (threadIdx.y == 1 && threadIdx.x == 1)
        rb_idx = 256 * (atomicAdd(&rb_base, 1) & RB_SIZE_MASK);
    __syncthreads();
    return rb_idx + tidx;
}
''')

# For now, the number of entries is fixed to a value known to work on all
# Fermi cards. Autodetection, or perhaps just a global limit increase, will be
# done when I get my hands on a Kepler device. The fixed size assumes blocks
# of 256 threads, although even at that size there are pathological cases that
# could break the assumption.
DEFAULT_RB_SIZE = 64
ringbuflib = mkringbuflib(DEFAULT_RB_SIZE)
