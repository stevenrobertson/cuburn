from collections import OrderedDict
from itertools import cycle
import numpy as np

from cuburn.genome import specs
from cuburn.genome.util import resolve_spec
from cuburn.genome.use import Wrapper, SplineEval

import util
from util import Template, assemble_code, devlib, binsearchlib, ringbuflib
from color import yuvlib
from mwc import mwclib

class PackerWrapper(Wrapper):
    """
    Obtain accessors in generated code.

    Call ``GenomePacker.view(ptr_name, wrapped_obj, prefix)`` to generate a
    view, where ``ptr_name`` is the name of the CUDA pointer which holds the
    interpolated values, ``wrapped_obj`` is the base Genome instance which
    will be uploaded to the device for interpolating from, and ``prefix`` is
    the prefix that will be assigned to property accessors from this object.

    Accessing a property on the object synthesizes an accessor for use in your
    code and an interpolator for use in generating that code. This conversion
    is done when the property is coerced into a string by the templating
    mechanism, so you can easily nest objects by saying, for instance,
    {{pcp.camera.rotation}} from within templated code.
    """
    packer = property(lambda s: s._params['packer'])

    def wrap_spline(self, path, spec, val):
        return PackerSpline(self.packer, path, spec)

    def __getattr__(self, name):
        path = self.path + (name,)
        if path in self.packer.packed_precalc:
            return self.packer.devname(path)
        return super(PackerWrapper, self).__getattr__(name)

    def _precalc(self):
        """Create a GenomePackerPrecalc object. See that class for details."""
        return PrecalcWrapper(self._val, self.spec, self.path,
                              packer=self.packer)

class PackerSpline(object):
    def __init__(self, packer, path, spec):
        self.packer, self.path, self.spec = packer, path, spec

    def __str__(self):
        """
        Returns the packed name in a format suitable for embedding directly
        into device code.
        """
        # When the template calls __str__ to format one of these splines, this
        # allocates the corresponding spline.
        return self.packer._require(self.spec, self.path)

class PrecalcSpline(PackerSpline):
    def __str__(self):
        return self.packer._require_pre(self.spec, self.path)

class PrecalcWrapper(PackerWrapper):
    """
    Insert precalculated values into the packed genome.

    Create a Precalc object by calling a view object's _precalc() method. The
    returned object shares the same referent dict as the view object, but
    instead of returning view accessors for use in the body of your code,
    accessing a property synthesizes a call to an interpolation function for
    use within the precalculating function. Use this in your precalculated
    code to obtain values from a genome.

    Once you've obtained the needed parameters and performed the
    precalculation, write them to the genome object with a call to the '_set'
    method in your precalc template. This method generates a new accessor to
    which you can assign a value in your precalculation function. It also
    makes this accessor available for use on the original packer object from
    within your main function.

    Finally, call the '_code' method on the precalc object with your generated
    precalculation code to add it to the precalculation function. The code
    will be wrapped in a C block to prevent namespace leakage, so that
    multiple invocations of the precalculation code on different code blocks
    can be performed.

    Example:

        def do_precalc(pcam):
            pcam._code(Template('''
                {{pcam._set('prop_sin')}} = sin({{pcam.prop}});
                ''').substitute(pcam=pcam))

        def gen_code(px):
            return Template('''
                {{do_precalc(px._precalc())}}
                printf("The sin of %g is %g.", {{px.prop}}, {{px.prop_sin}});
                ''').substitute(px=px)
    """
    def wrap_spline(self, path, spec, val):
        return PrecalcSpline(self.packer, path, spec)

    def _set(self, name):
        path = self.path + (name,)
        return self.packer._pre_alloc(path)

    def _code(self, code):
        self.packer.precalc_code.append(code)

class GenomePacker(object):
    """
    Packs a genome for use in iteration.
    """
    def __init__(self, tname, ptr_name, spec):
        """
        Create a new DataPacker.

        ``tname`` is the name of the structure typedef that will be emitted
        via this object's ``decls`` property.
        """
        self.tname, self.ptr_name, self.spec = tname, ptr_name, spec
        # We could do this in the order that things are requested, but we want
        # to be able to treat the direct stuff as a list so this function
        # doesn't unroll any more than it has to. So we separate things into
        # direct requests, and those that need precalculation.
        # Values of OrderedDict are unused; basically, it's just OrderedSet.
        self.packed_direct = OrderedDict()
        # Feel kind of bad about this, but it's just under the threshold of
        # being worth refactoring to be agnostic to interpolation types
        self.packed_direct_mag = OrderedDict()
        self.genome_precalc = OrderedDict()
        self.packed_precalc = OrderedDict()
        self.precalc_code = []

        self._len = None
        self.decls = None
        self.defs = None

        self.packed = None
        self.genome = None
        self.search_rounds = util.DEFAULT_SEARCH_ROUNDS

    def __len__(self):
        """Length in elements. (*4 for length in bytes.)"""
        assert self._len is not None, 'len() called before finalize()'
        return self._len

    def view(self, val={}):
        """Create a DataPacker view. See DataPackerView class for details."""
        return PackerWrapper(val, self.spec, packer=self)

    def _require(self, spec, path):
        """
        Called to indicate that the named parameter from the original genome
        must be available during interpolation.
        """
        if spec.interp == 'mag':
            self.packed_direct_mag[path] = None
        else:
            self.packed_direct[path] = None
        return self.devname(path)

    def _require_pre(self, spec, path):
        i = len(self.genome_precalc) << self.search_rounds
        self.genome_precalc[path] = None
        func = 'catmull_rom_mag' if spec.interp == 'mag' else 'catmull_rom'
        return '%s(&times[%d], &knots[%d], time)' % (func, i, i)

    def _pre_alloc(self, path):
        self.packed_precalc[path] = None
        return '%s->%s' % (self.ptr_name, '_'.join(path))

    def devname(self, path):
        return '%s.%s' % (self.ptr_name, '_'.join(path))

    def finalize(self):
        """
        Create the code to render this genome.
        """
        # At the risk of packing a few things more than once, we don't
        # uniquify the overall precalc order, sparing us the need to implement
        # recursive code generation
        direct = self.packed_direct.keys() + self.packed_direct_mag.keys()
        self.packed = direct + self.packed_precalc.keys()
        self.genome = direct + self.genome_precalc.keys()

        self._len = len(self.packed)

        decls = self._decls.substitute(**self.__dict__)
        defs = self._defs.substitute(**self.__dict__)

        return devlib(deps=[catmullromlib], decls=decls, defs=defs)

    def pack(self, gnm, pool=None):
        """
        Return a packed copy of the genome ready for uploading to the GPU,
        as two float32 NDArrays for the knot times and values.
        """
        width = 1 << self.search_rounds
        if pool:
            times = pool.allocate((len(self.genome), width), 'f4')
            knots = pool.allocate((len(self.genome), width), 'f4')
        else:
            times, knots = np.empty((2, len(self.genome), width), 'f4')
        times.fill(1e9)

        for idx, path in enumerate(self.genome):
            attr = gnm
            for name in path:
                if name not in attr:
                    attr = resolve_spec(specs.anim, path).default
                    break
                attr = attr[name]
            attr = SplineEval.normalize(attr)
            times[idx,:len(attr[0])] = attr[0]
            knots[idx,:len(attr[1])] = attr[1]
        return times, knots

    _defs = Template(r"""
__global__ void interp_{{tname}}(
        {{tname}}* {{ptr_name}},
        const float *times, const float *knots,
        float tstart, float tstep, int maxid)
{
    int id = gtid();
    if (id >= maxid) return;
    {{ptr_name}} = &{{ptr_name}}[id];
    float time = tstart + id * tstep;

    float *outf = reinterpret_cast<float*>({{ptr_name}});

    {{py:lpd = len(packed_direct)}}
    {{py:lpdm = len(packed_direct_mag)}}

    // TODO: unroll pragma?
    for (int i = 0; i < {{lpd}}; i++) {
        int j = i << {{search_rounds}};
        outf[i] = catmull_rom(&times[j], &knots[j], time);
    }

    for (int i = {{lpd}}; i < {{lpd+lpdm}}; i++) {
        int j = i << {{search_rounds}};
        outf[i] = catmull_rom_mag(&times[j], &knots[j], time);
    }

    // Advance 'times' and 'knots' to the purely generated sections, so that
    // the pregenerated statements emitted by _require_pre are correct.
    times = &times[{{(lpd+lpdm)<<search_rounds}}];
    knots = &knots[{{(lpd+lpdm)<<search_rounds}}];

    {{for hunk in precalc_code}}
{
    {{hunk}}
}
    {{endfor}}
}
""")

    _decls = Template(r"""
typedef struct {
{{for path in packed}}
    float   {{'_'.join(path)}};
{{endfor}}
} {{tname}};


""")

catmullromlib = devlib(deps=[binsearchlib], decls=r'''
__device__ __noinline__
float catmull_rom(const float *times, const float *knots, float t);

__device__ __noinline__
float catmull_rom_mag(const float *times, const float *knots, float t);
''', defs=r'''

// ELBOW is the linearization threhsold; above this magnitude, a value scales
// logarithmically, and below it, linearly. ELOG1 is a constant used to make
// this happen. See helpers/spline_mag_domain_interp.wxm for nice graphs.
#define ELBOW 0.0625f   // 2^(-4)
#define ELOG1 5.0f      // 1 - log2(elbow)

// Transform from linear to magnitude domain
__device__ float linlog(float x) {
    if (x > ELBOW)  return   log2f(x)  + ELOG1;
    if (x < -ELBOW) return -(log2f(-x) + ELOG1);
    return x / ELBOW;
}

// Reverse of above
__device__ float linexp(float v) {
    if (v >= 1.0)   return  exp2f( v - ELOG1);
    if (v <= -1.0)  return -exp2f(-v - ELOG1);
    return v * ELBOW;
}

__device__ float linslope(float x, float m) {
    if (x >=  ELBOW) return m /  x;
    if (x <= -ELBOW) return m / -x;
    return m / ELBOW;
}

__device__ float
catmull_rom_base(const float *times, const float *knots, float t, bool mag) {
    int idx = bitwise_binsearch(times, t);

    // The left bias of the search means that we never have to worry about
    // overshooting unless the genome is corrupted
    idx = max(idx, 1);

    float t1 = times[idx], t2 = times[idx+1] - t1;
    float rt2 = 1.0f / t2;
    float t0 = (times[idx-1] - t1) * rt2, t3 = (times[idx+2] - t1) * rt2;
    t = (t - t1) * rt2;

    // Now t1 is effectively 0 and t2 is 1

    float k0 = knots[idx-1], k1 = knots[idx],
          k2 = knots[idx+1], k3 = knots[idx+2];

    float m1 = (k2 - k0) / (1.0f - t0),
          m2 = (k3 - k1) / (t3);

    if (mag) {
        m1 = linslope(k1, m1);
        m2 = linslope(k2, m2);
        k1 = linlog(k1);
        k2 = linlog(k2);
    }

    float tt = t * t, ttt = tt * t;

    float r = m1 * (      ttt - 2.0f*tt + t)
            + k1 * ( 2.0f*ttt - 3.0f*tt + 1)
            + m2 * (      ttt -      tt)
            + k2 * (-2.0f*ttt + 3.0f*tt);

    if (mag) r = linexp(r);
    return r;
}

// Variants with scaling domain logic inlined
__device__ __noinline__
float catmull_rom(const float *times, const float *knots, float t) {
    return catmull_rom_base(times, knots, t, false);
}

__device__ __noinline__
float catmull_rom_mag(const float *times, const float *knots, float t) {
    return catmull_rom_base(times, knots, t, true);
}
''')

palintlib = devlib(deps=[binsearchlib, ringbuflib, yuvlib, mwclib], decls='''
surface<void, cudaSurfaceType2D> flatpal;
''', defs=r'''
__device__ float4
interp_color(const float *times, const float4 *sources, float time)
{
    int idx = fmaxf(bitwise_binsearch(times, time) + 1, 1);
    float tr = times[idx];

    float lf = (tr - time) / (tr - times[idx-1]);
    float rf = 1.0f - lf;

    float4 left  = sources[blockDim.x * (idx - 1) + threadIdx.x];
    float4 right = sources[blockDim.x * (idx)     + threadIdx.x];
    float3 yuv;

    if (tr > 1.0f) {
        // The right-side time is larger than 1.0. This only normally occurs
        // in a single-palette genome. In any case, we don't want to consider
        // the rightmost palette, since it's out of bounds now.
        right = left;
        lf = 1.0f;  // Correct for possibility of inf/nan
        rf = 0.0f;
    }

    float3 l3 = make_float3(left.x, left.y, left.z);
    float3 r3 = make_float3(right.x, right.y, right.z);

    float3 lyuv = rgb2yuv(l3);
    float3 ryuv = rgb2yuv(r3);
    yuv.x = lyuv.x * lf + ryuv.x * rf;
    yuv.y = lyuv.y * lf + ryuv.y * rf;
    yuv.z = lyuv.z * lf + ryuv.z * rf;

    yuv.y += 0.5f;
    yuv.z += 0.5f;

    return make_float4(yuv.x, yuv.y, yuv.z, left.w * lf + right.w * rf);
}

__global__ void interp_palette_flat(
        ringbuf *rb, mwc_st *rctxs,
        const float *times, const float4 *sources,
        float tstart, float tstep)
{
    mwc_st rctx = rctxs[rb_incr(rb->head, threadIdx.x)];

    float time = tstart + blockIdx.x * tstep;
    float4 yuva = interp_color(times, sources, time);

    // TODO: pack Y at full precision, UV at quarter
    uint2 out;

    uint32_t y = yuva.x * 255.0f + 0.49f * mwc_next_11(rctx);
    uint32_t u = yuva.y * 255.0f + 0.49f * mwc_next_11(rctx);
    uint32_t v = yuva.z * 255.0f + 0.49f * mwc_next_11(rctx);
    y = min(255, y);
    u = min(255, u);
    v = min(255, v);
    out.y = (1 << 22) | (y << 4);
    out.x = (u << 18) | v;

    surf2Dwrite(out, flatpal, 8 * threadIdx.x, blockIdx.x);
    rctxs[rb_incr(rb->tail, threadIdx.x)] = rctx;
}
''')

testcrlib = devlib(defs=r'''
__global__ void
test_cr(const float *times, const float *knots, const float *t, float *r) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    r[i] = catmull_rom(times, knots, t[i]);
}
''')

if __name__ == "__main__":
    # Test spline evaluation. This code will probably drift pretty often.
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.autoinit
    from cuburn.genome import SplEval

    gp = GenomePacker("unused")
    gp.finalize()
    mod = SourceModule(assemble_code(BaseCode, gp))
    times = np.sort(np.concatenate(([-2.0, 0.0, 1.0, 3.0], np.random.rand(12))))
    knots = np.random.randn(16)

    print times
    print knots

    evaltimes = np.float32(np.linspace(0, 1, 1024))
    sp = SplEval([x for k in zip(times, knots) for x in k])
    vals = np.array([sp(t) for t in evaltimes], dtype=np.float32)

    dtimes = np.empty((32,), dtype=np.float32)
    dtimes.fill(1e9)
    dtimes[:16] = times
    dknots = np.zeros_like(dtimes)
    dknots[:16] = knots

    dvals = np.empty_like(vals)
    mod.get_function("test_cr")(cuda.In(dtimes), cuda.In(dknots),
            cuda.In(evaltimes), cuda.Out(dvals), block=(1024, 1, 1))
    for t, v, d in zip(evaltimes, vals, dvals):
        print '%6f %8g %8g' % (t, v, d)
