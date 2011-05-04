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
"""

    defs = """
// TODO: use launch parameter preconfig to eliminate unnecessary parts
__device__
uint32_t gtid() {
    return threadIdx.x + blockDim.x *
            (threadIdx.y + blockDim.y *
                (threadIdx.z + blockDim.z *
                    (blockIdx.x + (gridDim.x * blockIdx.y))));
}

__device__
int trunca(float f) {
    // truncate as used in address calculations
    int ret;
    asm("cvt.rni.s32.f32    %0,     %1;" : "=r"(ret) : "f"(f));
    return ret;
}
"""

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
        return '%s->%s' % (self.ptr, name)

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

    def __init__(self, tname, clsize=128):
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
            pview, paccessor = self.packed[name]
            if pview == view and (accessor is None or paccessor == accessor):
                return
            raise ValueError("Same name, different accessor or view: %s" % name)
        self.packed[name] = (view, accessor)
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
            view, accessor = self.packed[name]
            if view not in subbed_nses:
                subbed_nses[view] = view._apply_subs(dict(base_ns))
            try:
                val = eval(accessor, subbed_nses[view])
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

