import os
import time

import pycuda.driver as cuda
import numpy as np

from cuburnlib.ptx import PTXFragment, PTXEntryPoint, PTXTest

class MWCRNG(PTXFragment):
    def __init__(self):
        if not os.path.isfile('primes.bin'):
            raise EnvironmentError('primes.bin not found')

    prelude = (".global .u32 mwc_rng_mults[{{ctx.threads}}];\n"
               ".global .u64 mwc_rng_state[{{ctx.threads}}];")

    def _next_b32(self, dreg):
        # TODO: make sure PTX optimizes away superfluous move instrs
        return """
        {
        // MWC next b32
        .reg .u64       mwc_out;
        cvt.u64.u32     mwc_out,    mwc_car;
        mad.wide.u32    mwc_out,    mwc_st,     mwc_mult,   mwc_out;
        mov.b64         {mwc_st,    mwc_car},   mwc_out;
        mov.u32         %s,         mwc_st;
        }
        """ % dreg

    def subs(self, ctx):
        return {'mwc_next_b32': self._next_b32}

    entry_start = """
    .reg .u32 mwc_st, mwc_mult, mwc_car;
    {
        // MWC load multipliers and RNG states
        .reg .u32       mwc_off, mwc_addr;
        {{ get_gtid('mwc_off') }}
        mov.u32         mwc_addr,   mwc_rng_mults;
        mad.lo.u32      mwc_addr,   mwc_off,    4,  mwc_addr;
        ld.global.u32   mwc_mult,   [mwc_addr];
        mov.u32         mwc_addr,   mwc_rng_state;
        mad.lo.u32      mwc_addr,   mwc_off,    8,  mwc_addr;
        ld.global.v2.u32 {mwc_st, mwc_car}, [mwc_addr];
    }
    """

    entry_end = """
    {
        // MWC save states
        .reg .u32       mwc_addr, mwc_off;
        {{ get_gtid('mwc_off') }}
        mov.u32         mwc_addr,   mwc_rng_state;
        mad.lo.u32      mwc_addr,   mwc_off,    8,      mwc_addr;
        st.global.v2.u32    [mwc_addr],     {mwc_st, mwc_car};
    }
    """

    def set_up(self, ctx):
        # Load raw big-endian u32 multipliers from primes.bin.
        with open('primes.bin') as primefp:
            dt = np.dtype(np.uint32).newbyteorder('B')
            mults = np.frombuffer(primefp.read(), dtype=dt)
        # Randomness in choosing multipliers is good, but larger multipliers
        # have longer periods, which is also good. This is a compromise.
        # TODO: fix mutability, enable shuffle here
        # TODO: prevent period-1 random generators
        #ctx.rand.shuffle(mults[:ctx.threads*4])
        # Copy multipliers and seeds to the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        # TODO: get async to work
        #cuda.memcpy_htod_async(multdp, mults.tostring()[:multl], ctx.stream)
        cuda.memcpy_htod(multdp, mults.tostring()[:multl])
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        #cuda.memcpy_htod_async(statedp, ctx.rand.bytes(statel), ctx.stream)

        cuda.memcpy_htod(statedp, ctx.rand.bytes(statel))

    def tests(self, ctx):
        return [MWCRNGTest]

class MWCRNGTest(PTXTest):
    name = "MWC RNG sum-of-threads"
    deps = [MWCRNG]
    rounds = 10000

    prelude = ".global .u64 mwc_rng_test_sums[{{ctx.threads}}];"

    def entry(self, ctx):
        return ('MWC_RNG_test', '', """
            .reg .u64   sum, addl;
            .reg .u32   addend;
            mov.u64     sum,    0;
            {
                .reg .u32   loopct;
                .reg .pred  p;
                mov.u32     loopct, %s;
loopstart:
                {{ mwc_next_b32('addend') }}
                cvt.u64.u32 addl,   addend;
                add.u64     sum,    sum,    addl;
                sub.u32     loopct, loopct, 1;
                setp.gt.u32 p,      loopct, 0;
            @p  bra.uni     loopstart;
            }
            {
                .reg .u32       addr, offset;
                {{ get_gtid('offset') }}
                mov.u32         addr,   mwc_rng_test_sums;
                mad.lo.u32      addr,   offset,     8,      addr;
                st.global.u64   [addr], sum;
            }
            """ % self.rounds)

    def call(self, ctx):
        # Get current multipliers and seeds from the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        mults = cuda.from_device(multdp, ctx.threads, np.uint32)
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        fullstates = cuda.from_device(statedp, ctx.threads, np.uint64)
        sums = np.zeros(ctx.threads, np.uint64)

        print "Running states forward %d rounds" % self.rounds
        ctime = time.time()
        for i in range(self.rounds):
            states = fullstates & 0xffffffff
            carries = fullstates >> 32
            fullstates = mults * states + carries
            sums = sums + (fullstates & 0xffffffff)
        ctime = time.time() - ctime
        print "Done on host, took %g seconds" % ctime

        func = ctx.mod.get_function('MWC_RNG_test')
        dtime = func(block=ctx.block, grid=ctx.grid, time_kernel=True)
        print "Done on device, took %g seconds (%gx)" % (dtime, ctime/dtime)
        dfullstates = cuda.from_device(statedp, ctx.threads, np.uint64)
        print dfullstates, fullstates
        if not (dfullstates == fullstates).all():
            print "State discrepancy"
            print dfullstates
            print fullstates
            return False

        sumdp, suml = ctx.mod.get_global('mwc_rng_test_sums')
        dsums = cuda.from_device(sumdp, ctx.threads, np.uint64)
        print dsums, sums
        if not (dsums == sums).all():
            print "Sum discrepancy"
            print dsums
            print sums
            return False
        return True


