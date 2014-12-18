"""
The multiply-with-carry random number generator.
"""

import os
import warnings
import numpy as np

from util import devlib, assemble_code

# Keeping this live in the module isn't necessary, but loading the mults
# can be surprisingly slow.
mults = None

def load_mults():
    pfpath = os.path.join(os.path.dirname(__file__), 'primes.bin')
    if os.path.isfile(pfpath):
        with open(pfpath) as fp:
            return np.frombuffer(fp.read(), dtype='<u4')

    warnings.warn('primes.bin not found, trying to download it')
    import bz2, urllib2
    ufp = urllib2.urlopen('http://aduro.strobe.cc/primes.diff.bin.bz2')
    diffs = np.frombuffer(bz2.decompress(ufp.read()), dtype='<u2')
    mults = np.cumsum(-np.array(diffs, dtype='<u4'), dtype='<u4')
    with open(pfpath, 'wb') as fp:
        fp.write(mults)
    return mults

def make_seeds(nthreads, host_seed=None):
    global mults
    if mults is None:
        mults = load_mults()
    if host_seed:
        rand = np.random.RandomState(host_seed)
    else:
        rand = np.random

    # Create the seed structures. TODO: check that struct is 4-byte aligned
    seeds = np.empty((nthreads, 3), dtype=np.uint32)
    seeds[:,0] = mults[:nthreads]

    # Excludes 0xffffffff for 32-bit compatibility with laziness
    seeds[:,1] = rand.randint(1, 0x7fffffff, size=nthreads)
    seeds[:,2] = rand.randint(1, 0x7fffffff, size=nthreads)

    return seeds

mwclib = devlib(decls=r'''
typedef struct {
    uint32_t    mul;
    uint32_t    state;
    uint32_t    carry;
} mwc_st;
''', defs=r'''
__device__ uint32_t mwc_next(mwc_st &st) {
    asm("{\n\t"
        ".reg .u32 tmp;\n\t"
        "mad.lo.cc.u32   tmp,    %2,     %1,     %0;\n\t"
        "madc.hi.u32     %0,     %2,     %1,     0;\n\t"
        "mov.u32         %1,     tmp;\n\t"
    "}" : "+r"(st.carry), "+r"(st.state) : "r"(st.mul));
    return st.state;
}

__device__ float mwc_next_01(mwc_st &st) {
    return mwc_next(st) * (1.0f / 4294967296.0f);
}

__device__ float mwc_next_11(mwc_st &st) {
    uint32_t val = mwc_next(st);
    float ret;
    asm("cvt.rn.f32.s32 %0,     %1;\n\t"
        "mul.f32        %0,     %0,     (1.0 / 2147483648.0);"
        : "=f"(ret) : "r"(val));
    return ret;
}
''')

mwctestlib = devlib(deps=[mwclib], defs="""
__global__ void test_mwc(mwc_st *msts, uint64_t *sums, float nrounds) {
    mwc_st rctx = msts[gtid()];
    uint64_t sum = 0;
    for (float i = 0; i < nrounds; i++) sum += mwc_next(rctx);
    sums[gtid()] = sum;
    msts[gtid()] = rctx;
}
""")

def test_mwc(rounds=5000, nblocks=64, blockwidth=512):
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import time

    nthreads = blockwidth * nblocks
    seeds = make_seeds(nthreads, host_seed=42)
    dseeds = cuda.to_device(seeds)

    mod = SourceModule(assemble_code(mwctestlib))

    for trial in range(2):
        print "Trial %d, on CPU: " % trial,
        sums = np.zeros(nthreads, dtype=np.uint64)
        ctime = time.time()
        mults = seeds[0].astype(np.uint64)
        states = seeds[1]
        carries = seeds[2]
        for i in range(rounds):
            step = np.frombuffer((mults * states + carries).data,
                       dtype=np.uint32).reshape((2, nthreads), order='F')
            states[:] = step[0]
            carries[:] = step[1]
            sums += states

        ctime = time.time() - ctime
        print "Took %g seconds." % ctime

        print "Trial %d, on device: " % trial,
        dsums = cuda.mem_alloc(8*nthreads)
        fun = mod.get_function("test_mwc")
        dtime = fun(dseeds, dsums, np.float32(rounds),
                    block=(blockwidth,1,1), grid=(nblocks,1),
                    time_kernel=True)
        print "Took %g seconds." % dtime
        dsums = cuda.from_device(dsums, nthreads, np.uint64)
        if not np.all(np.equal(sums, dsums)):
            print "Sum discrepancy!"
            print sums
            print dsums

if __name__ == "__main__":
    import pycuda.autoinit
    test_mwc()
