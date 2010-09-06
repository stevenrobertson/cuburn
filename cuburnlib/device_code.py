"""
Contains the PTX fragments which will drive the device.
"""

import os
import time
import struct

import pycuda.driver as cuda
import numpy as np

from cuburnlib.ptx import *

class IterThread(PTXTest):
    entry_name = 'iter_thread'
    entry_params = []

    def __init__(self):
        self.cps_uploaded = False

    def deps(self):
        return [MWCRNG, CPDataStream]

    @ptx_func
    def module_setup(self):
        mem.global_.u32('g_cp_array',
                        cp_stream_size*features.max_ntemporal_samples)
        mem.global_.u32('g_num_cps')
        # TODO move into debug statement
        mem.global_.u32('g_num_rounds', ctx.threads)
        mem.global_.u32('g_num_writes', ctx.threads)

    @ptx_func
    def entry(self):
        reg.f32('x_coord y_coord color_coord alpha_coord')

        # TODO: temporary, for testing
        reg.u32('num_rounds num_writes')
        op.mov.u32(num_rounds, 0)
        op.mov.u32(num_writes, 0)

        # TODO: MWC float output types
        #mwc_next_f32_01(x_coord)
        #mwc_next_f32_01(y_coord)
        #mwc_next_f32_01(color_coord)
        #mwc_next_f32_01(alpha_coord)

        # Registers are hard to come by. To avoid having to track both the count
        # of samples processed and the number of samples to generate,
        # 'num_samples' counts *down* from the CP's desired sample count.
        # When it hits 0, we move on to the next CP.
        #
        # FUSE complicates things. To track it, we store the *negative* number
        # of points we have left to fuse before we start to store the results.
        # When it hits -1, we're done fusing, and can move on to the real
        # thread. The execution flow between 'cp_loop', 'fuse_start', and
        # 'iter_loop_start' is therefore tricky, and bears close inspection.
        #
        # In summary:
        #   num_samples == 0: Load next CP, set num_samples from that
        #   num_samples >  0: Iterate, store the result, decrement num_samples
        #   num_samples < -1: Iterate, don't store, increment num_samples
        #   num_samples == -1: Done fusing, enter normal flow
        # TODO: move this to qlocal storage
        reg.s32('num_samples')
        op.mov.s32(num_samples, -(features.num_fuse_samples+1))

        # TODO: Move cp_num to qlocal storage (or spill it, rarely accessed)
        reg.u32('cp_idx cpA')
        op.mov.u32(cp_idx, 0)

        label('cp_loop_start')
        op.bar.sync(0)

        with block('Check to see if this is the last CP'):
            reg.u32('num_cps')
            reg.pred('p_last_cp')
            op.ldu.u32(num_cps, addr(g_num_cps))
            op.setp.ge.u32(p_last_cp, cp_idx, num_cps)
            op.bra.uni('all_cps_done', ifp=p_last_cp)

        with block('Load CP address'):
            op.mov.u32(cpA, g_cp_array)
            op.mad.lo.u32(cpA, cp_idx, cp_stream_size, cpA)

        with block('Increment CP index, load num_samples (unless in fuse)'):
            reg.pred('p_not_in_fuse')
            op.setp.ge.s32(p_not_in_fuse, num_samples, 0)
            op.add.u32(cp_idx, cp_idx, 1, ifp=p_not_in_fuse)
            cp_stream_get(cpA, num_samples, 'samples_per_thread',
                          ifp=p_not_in_fuse)

        label('fuse_loop_start')
        with block('FUSE-specific stuff'):
            reg.pred('p_fuse')
            comment('If num_samples == -1, set it to 0 and jump back up')
            comment('This will start the normal CP loading machinery')
            op.setp.eq.s32(p_fuse, num_samples, -1)
            op.mov.s32(num_samples, 0, ifp=p_fuse)
            op.bra.uni(cp_loop_start, ifp=p_fuse)

            comment('If num_samples < -1, still fusing, so increment')
            op.setp.lt.s32(p_fuse, num_samples, -1)
            op.add.s32(num_samples, num_samples, 1, ifp=p_fuse)

        label('iter_loop_start')

        comment('Do... well, most of everything')

        op.add.u32(num_rounds, num_rounds, 1)

        with block("Test if we're still in FUSE"):
            reg.pred('p_in_fuse')
            op.setp.lt.s32(p_in_fuse, num_samples, 0)
            op.bra.uni(fuse_loop_start, ifp=p_in_fuse)

        with block("Ordinarily, we'd write the result here"):
            op.add.u32(num_writes, num_writes, 1)

        with block("Check to see if we're done with this CP"):
            reg.pred('p_cp_done')
            op.add.s32(num_samples, num_samples, -1)
            op.setp.eq.s32(p_cp_done, num_samples, 0)
            op.bra.uni(cp_loop_start, ifp=p_cp_done)

        op.bra.uni(iter_loop_start)

        label('all_cps_done')
        # TODO this is for testing, move it to a debug statement
        store_per_thread(g_num_rounds, num_rounds)
        store_per_thread(g_num_writes, num_writes)

    def upload_cp_stream(self, ctx, cp_stream, num_cps):
        cp_array_dp, cp_array_l = ctx.mod.get_global('g_cp_array')
        assert len(cp_stream) <= cp_array_l, "Stream too big!"
        cuda.memcpy_htod_async(cp_array_dp, cp_stream)
        num_cps_dp, num_cps_l = ctx.mod.get_global('g_num_cps')
        cuda.memcpy_htod_async(num_cps_dp, struct.pack('i', num_cps))
        self.cps_uploaded = True

    def call(self, ctx):
        if not self.cps_uploaded:
            raise Error("Cannot call IterThread before uploading CPs")
        func = ctx.mod.get_function('iter_thread')
        dtime = func(block=ctx.block, grid=ctx.grid, time_kernel=True)

        num_rounds_dp, num_rounds_l = ctx.mod.get_global('g_num_rounds')
        num_writes_dp, num_writes_l = ctx.mod.get_global('g_num_writes')
        rounds = cuda.from_device(num_rounds_dp, ctx.threads, np.uint32)
        writes = cuda.from_device(num_writes_dp, ctx.threads, np.uint32)
        print "Rounds:", rounds
        print "Writes:", writes

class MWCRNG(PTXFragment):
    def __init__(self):
        self.rand = np.random
        self.threads_ready = 0
        if not os.path.isfile('primes.bin'):
            raise EnvironmentError('primes.bin not found')

    def set_seed(self, seed):
        self.rand = np.random.mtrand.RandomState(seed)

    @ptx_func
    def module_setup(self):
        mem.global_.u32('mwc_rng_mults', ctx.threads)
        mem.global_.u64('mwc_rng_state', ctx.threads)

    @ptx_func
    def entry_setup(self):
        reg.u32('mwc_st mwc_mult mwc_car')
        with block('Load MWC multipliers and states'):
            reg.u32('mwc_off mwc_addr')
            get_gtid(mwc_off)
            op.mov.u32(mwc_addr, mwc_rng_mults)
            op.mad.lo.u32(mwc_addr, mwc_off, 4, mwc_addr)
            op.ld.global_.u32(mwc_mult, addr(mwc_addr))

            op.mov.u32(mwc_addr, mwc_rng_state)
            op.mad.lo.u32(mwc_addr, mwc_off, 8, mwc_addr)
            op.ld.global_.v2.u32(vec(mwc_st, mwc_car), addr(mwc_addr))

    @ptx_func
    def entry_teardown(self):
        with block('Save MWC states'):
            reg.u32('mwc_off mwc_addr')
            get_gtid(mwc_off)
            op.mov.u32(mwc_addr, mwc_rng_state)
            op.mad.lo.u32(mwc_addr, mwc_off, 8, mwc_addr)
            op.st.global_.v2.u32(addr(mwc_addr), vec(mwc_st, mwc_car))

    @ptx_func
    def next_b32(self, dst_reg):
        with block('Load next random into ' + dst_reg.name):
            reg.u64('mwc_out')
            op.cvt.u64.u32(mwc_out, mwc_car)
            op.mad.wide.u32(mwc_out, mwc_st, mwc_mult, mwc_out)
            op.mov.b64(vec(mwc_st, mwc_car), mwc_out)
            op.mov.u32(dst_reg, mwc_st)

    def to_inject(self):
        return dict(mwc_next_b32=self.next_b32)

    def device_init(self, ctx):
        if self.threads_ready >= ctx.threads:
            # Already set up enough random states, don't push again
            return

        # Load raw big-endian u32 multipliers from primes.bin.
        with open('primes.bin') as primefp:
            dt = np.dtype(np.uint32).newbyteorder('B')
            mults = np.frombuffer(primefp.read(), dtype=dt)
        stream = cuda.Stream()
        # Randomness in choosing multipliers is good, but larger multipliers
        # have longer periods, which is also good. This is a compromise.
        mults = np.array(mults[:ctx.threads*4])
        self.rand.shuffle(mults)
        # Copy multipliers and seeds to the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        cuda.memcpy_htod_async(multdp, mults.tostring()[:multl])
        # Intentionally excludes both 0 and (2^32-1), as they can lead to
        # degenerate sequences of period 0
        states = np.array(self.rand.randint(1, 0xffffffff, size=2*ctx.threads),
                          dtype=np.uint32)
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        cuda.memcpy_htod_async(statedp, states.tostring())
        self.threads_ready = ctx.threads

    def tests(self):
        return [MWCRNGTest]

class MWCRNGTest(PTXTest):
    name = "MWC RNG sum-of-threads"
    rounds = 5000
    entry_name = 'MWC_RNG_test'
    entry_params = ''

    def deps(self):
        return [MWCRNG]

    @ptx_func
    def module_setup(self):
        mem.global_.u64('mwc_rng_test_sums', ctx.threads)

    @ptx_func
    def entry(self):
        reg.u64('sum addl')
        reg.u32('addend')
        op.mov.u64(sum, 0)
        with block('Sum next %d random numbers' % self.rounds):
            reg.u32('loopct')
            reg.pred('p')
            op.mov.u32(loopct, self.rounds)
            label('loopstart')
            mwc_next_b32(addend)
            op.cvt.u64.u32(addl, addend)
            op.add.u64(sum, sum, addl)
            op.sub.u32(loopct, loopct, 1)
            op.setp.gt.u32(p, loopct, 0)
            op.bra.uni(loopstart, ifp=p)

        with block('Store sum and state'):
            reg.u32('adr offset')
            get_gtid(offset)
            op.mov.u32(adr, mwc_rng_test_sums)
            op.mad.lo.u32(adr, offset, 8, adr)
            op.st.global_.u64(addr(adr), sum)

    def call(self, ctx):
        # Get current multipliers and seeds from the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        mults = cuda.from_device(multdp, ctx.threads, np.uint32)
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        fullstates = cuda.from_device(statedp, ctx.threads, np.uint64)
        sums = np.zeros(ctx.threads, np.uint64)

        print "Running %d states forward %d rounds" % (len(mults), self.rounds)
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
        if not (dfullstates == fullstates).all():
            print "State discrepancy"
            print dfullstates
            print fullstates
            return False

        sumdp, suml = ctx.mod.get_global('mwc_rng_test_sums')
        dsums = cuda.from_device(sumdp, ctx.threads, np.uint64)
        if not (dsums == sums).all():
            print "Sum discrepancy"
            print dsums
            print sums
            return False
        return True

class CameraCoordTransform(PTXFragment):
    pass

class CPDataStream(DataStream):
    """DataStream which stores the control points."""
    prefix = 'cp'

