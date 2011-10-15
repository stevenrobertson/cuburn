"""
Provides tools and miscellaneous functions for building device code.
"""

import numpy as np
import tempita

class Template(tempita.Template):
    default_namespace = tempita.Template.default_namespace.copy()
Template.default_namespace.update({'np': np})

class HunkOCode(object):
    """An apparently passive container for device code."""
    # Use property objects to make these dynamic
    headers = ''
    decls = ''
    defs = ''

def assemble_code(*sections):
    return ''.join([''.join([getattr(sect, kind) for sect in sections])
                    for kind in ['headers', 'decls', 'defs']])

def apply_affine(x, y, xo, yo, packer, base_accessor, base_name):
    return Template("""
    {{xo}} = {{packer.get(ba + '[0,0]', bn + '_xx')}} * {{x}}
           + {{packer.get(ba + '[0,1]', bn + '_xy')}} * {{y}}
           + {{packer.get(ba + '[0,2]', bn + '_xo')}};
    {{yo}} = {{packer.get(ba + '[1,0]', bn + '_yx')}} * {{x}}
           + {{packer.get(ba + '[1,1]', bn + '_yy')}} * {{y}}
           + {{packer.get(ba + '[1,2]', bn + '_yo')}};
    """).substitute(x=x, y=y, xo=xo, yo=yo, packer=packer,
                    ba=base_accessor, bn=base_name)

def apply_affine_flam3(x, y, xo, yo, packer, base_accessor, base_name):
    """Read an affine transformation in *flam3 order* and apply it."""
    return tempita.Template("""
    {{xo}} = {{packer.get(ba + '[0][0]', bn + '_xx')}} * {{x}}
           + {{packer.get(ba + '[1][0]', bn + '_xy')}} * {{y}}
           + {{packer.get(ba + '[2][0]', bn + '_xo')}};
    {{yo}} = {{packer.get(ba + '[0][1]', bn + '_yx')}} * {{x}}
           + {{packer.get(ba + '[1][1]', bn + '_yy')}} * {{y}}
           + {{packer.get(ba + '[2][1]', bn + '_yo')}};
    """).substitute(x=x, y=y, xo=xo, yo=yo, packer=packer,
                    ba=base_accessor, bn=base_name)

class BaseCode(HunkOCode):
    headers = """
#include<cuda.h>
#include<stdint.h>
#include<stdio.h>
"""

    defs = r"""
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

__device__
uint32_t trunca(float f) {
    // truncate as used in address calculations. note the use of a signed
    // conversion is intentional here (simplifies image bounds checking).
    uint32_t ret;
    asm("cvt.rni.s32.f32    %0,     %1;" : "=r"(ret) : "f"(f));
    return ret;
}

__global__
void zero_dptr(float* dptr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dptr[i] = 0.0f;
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


"""

    @staticmethod
    def zero_dptr(mod, dptr, size, stream=None):
        """
        A memory zeroer which can be embedded in a stream. Size is the
        number of 4-byte words in the pointer.
        """
        zero = mod.get_function("zero_dptr")
        zero(dptr, np.int32(size), stream=stream,
             block=(1024, 1, 1), grid=(size/1024+1, 1))

class DataPackerView(object):
    """
    View of a data packer. Intended to be initialized using DataPacker.view().

    All views of a data packer share the same stream parameters, such as
    position and total size, but do not share other parameters, such as the
    pointer name used in emitted code lookups or the lookup context.
    """
    def __init__(self, packer, ptr, prefix, ns):
        self.packer, self.ptr, self.prefix, self.ns = packer, ptr, prefix, ns

    def get(self, accessor, name=None):
        """
        Add an access to the stream, returning the formatted load expression
        for device use. If 'name' is missing, the name components after the
        final dot in the accessor will be used. Little effort is made to
        ensure that this is valid C.
        """
        if name is None:
            name = accessor.rsplit('.', 1)[-1]
            name = name.replace('[', '_').replace(']', '')
        name = self.prefix + name
        self.packer._access(self, accessor, name)
        return '%s.%s' % (self.ptr, name)

    def sub(self, dst, src):
        """Add a substitution to the namespace."""
        self.ns.append((src, dst))

    def view(self, ptr_name, prefix=''):
        """
        As DataPacker.view(), but preserving the current set of namespace
        substitutions.
        """
        return DataPackerView(self.packer, ptr_name, prefix, list(self.ns))

    def _apply_subs(self, ns):
        for s, d in self.ns:
            ns[d] = eval(s, ns)
        return ns

class DataPacker(HunkOCode):
    """
    Packs 32-bit float values into a dynamic data structure, and emits
    accessors to those data values from device code. Might get fancier in the
    future, but for now it's incredibly barebones.
    """

    default_namespace = {'np': np}

    def __init__(self, tname, clsize=4):
        """
        Create a new DataPacker.

        ``tname`` is the name of the structure typedef that will be emitted
        via this object's ``decls`` property.

        ``clsize`` is the size of a cache line, in bytes. The resulting
        data structure will be padded to that size.
        """
        self.tname = tname
        self.clsize = clsize
        self.packed = {}
        self.packed_order = []

    def view(self, ptr_name, prefix=''):
        """Create a DataPacker view. See DataPackerView class for details."""
        return DataPackerView(self, ptr_name, prefix, list())

    def _access(self, view, accessor, name):
        if name in self.packed:
            pview, paccessor, pcomp = self.packed[name]
            if pview == view and (accessor is None or paccessor == accessor):
                return
            raise ValueError("Same name, different accessor or view: %s" % name)
        comp_accessor = compile(accessor, '{{template}}', 'eval')
        self.packed[name] = (view, accessor, comp_accessor)
        self.packed_order.append(name)

    def __len__(self):
        return len(self.packed_order)

    @property
    def align(self):
        return (4 * len(self) + self.clsize - 1) / self.clsize * self.clsize

    def pack(self, **kwargs):
        base_ns = self.default_namespace.copy()
        base_ns.update(kwargs)
        out = np.zeros(self.align/4, dtype=np.float32)
        subbed_nses = {}

        for i, name in enumerate(self.packed_order):
            view, accessor, comp = self.packed[name]
            if view not in subbed_nses:
                subbed_nses[view] = view._apply_subs(dict(base_ns))
            try:
                val = eval(comp, subbed_nses[view])
            except Exception, e:
                print 'Error while evaluating accessor "%s"' % accessor
                raise e
            out[i] = val
        return out

    @property
    def decls(self):
        tmpl = Template("""
typedef struct {
{{for name, accessor in values}}
    float   {{'%-20s' % name}}; // {{accessor}}
{{endfor}}
{{if padding > 0}}
    // Align to fill whole cache lines
    float   padding[{{padding}}];
{{endif}}
} {{tname}};
""")
        return tmpl.substitute(
                values  = [(n, self.packed[n][1]) for n in self.packed_order],
                padding = self.align / 4 - len(self),
                tname   = self.tname
            )

