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
                        cp.stream_size*features.max_ntemporal_samples)
        mem.global_.u32('g_num_cps')
        mem.global_.u32('g_num_cps_started')
        # TODO move into debug statement
        mem.global_.u32('g_num_rounds', ctx.threads)
        mem.global_.u32('g_num_writes', ctx.threads)

    @ptx_func
    def entry(self):
        # For now, we indulge in the luxury of shared memory.

        # Index number of current CP, shared across CTA
        mem.shared.u32('s_cp_idx')

        # Number of samples that have been generated so far in this CTA
        # If this number is negative, we're still fusing points, so this
        # behaves slightly differently (see ``fuse_loop_start``)
        mem.shared.u32('s_num_samples')
        op.st.shared.u32(addr(s_num_samples), -(features.num_fuse_samples+1))

        # TODO: temporary, for testing
        reg.u32('num_rounds num_writes')
        op.mov.u32(num_rounds, 0)
        op.mov.u32(num_writes, 0)

        reg.f32('x_coord y_coord color_coord')
        mwc.next_f32_11(x_coord)
        mwc.next_f32_11(y_coord)
        mwc.next_f32_01(color_coord)

        comment("Ensure all init is done")
        op.bar.sync(0)

        label('cp_loop_start')
        reg.u32('cp_idx cpA')
        with block("Claim a CP"):
            std.set_is_first_thread(reg.pred('p_is_first'))
            op.atom.inc.u32(cp_idx, addr(g_num_cps_started), 1, ifp=p_is_first)
            op.st.shared.u32(addr(s_cp_idx), cp_idx, ifp=p_is_first)

        comment("Load the CP index in all threads")
        op.bar.sync(0)
        op.ld.shared.u32(cp_idx, addr(s_cp_idx))

        with block("Check to see if this CP is valid (if not, we're done"):
            reg.u32('num_cps')
            reg.pred('p_last_cp')
            op.ldu.u32(num_cps, addr(g_num_cps))
            op.setp.ge.u32(p_last_cp, cp_idx, 1)
            op.bra.uni('all_cps_done', ifp=p_last_cp)

        with block('Load CP address'):
            op.mov.u32(cpA, g_cp_array)
            op.mad.lo.u32(cpA, cp_idx, cp.stream_size, cpA)

        label('fuse_loop_start')
        # When fusing, num_samples holds the (negative) number of iterations
        # left across the CP, rather than the number of samples in total.
        with block("If still fusing, increment count unconditionally"):
            std.set_is_first_thread(reg.pred('p_is_first'))
            op.red.shared.add.s32(addr(s_num_samples), 1, ifp=p_is_first)
            op.bar.sync(0)

        label('iter_loop_start')

        comment('Do... well, most of everything')

        op.add.u32(num_rounds, num_rounds, 1)

        with block("Test if we're still in FUSE"):
            reg.s32('num_samples')
            reg.pred('p_in_fuse')
            op.ld.shared.u32(num_samples, addr(s_num_samples))
            op.setp.lt.s32(p_in_fuse, num_samples, 0)
            op.bra.uni(fuse_loop_start, ifp=p_in_fuse)

        with block("Ordinarily, we'd write the result here"):
            op.add.u32(num_writes, num_writes, 1)

        # For testing, declare and clear p_badval
        reg.pred('p_goodval')
        op.setp.eq.u32(p_goodval, 1, 1)

        with block("Increment number of samples by number of good values"):
            reg.b32('good_samples')
            op.vote.ballot.b32(good_samples, p_goodval)
            op.popc.b32(good_samples, good_samples)
            std.set_is_first_thread(reg.pred('p_is_first'))
            op.red.shared.add.s32(addr(s_num_samples), good_samples,
                                  ifp=p_is_first)

        with block("Check to see if we're done with this CP"):
            reg.pred('p_cp_done')
            reg.s32('num_samples num_samples_needed')
            op.ld.shared.s32(num_samples, addr(s_num_samples))
            cp.get(cpA, num_samples_needed, 'cp.nsamples')
            op.setp.ge.s32(p_cp_done, num_samples, num_samples_needed)
            op.bra.uni(cp_loop_start, ifp=p_cp_done)

        op.bra.uni(iter_loop_start)

        label('all_cps_done')
        # TODO this is for testing, move it to a debug statement
        std.store_per_thread(g_num_rounds, num_rounds)
        std.store_per_thread(g_num_writes, num_writes)

    def upload_cp_stream(self, ctx, cp_stream, num_cps):
        cp_array_dp, cp_array_l = ctx.mod.get_global('g_cp_array')
        assert len(cp_stream) <= cp_array_l, "Stream too big!"
        cuda.memcpy_htod_async(cp_array_dp, cp_stream)

        num_cps_dp, num_cps_l = ctx.mod.get_global('g_num_cps')
        cuda.memset_d32(num_cps_dp, num_cps, 1)
        self.cps_uploaded = True

    def call(self, ctx):
        if not self.cps_uploaded:
            raise Error("Cannot call IterThread before uploading CPs")
        num_cps_st_dp, num_cps_st_l = ctx.mod.get_global('g_num_cps_started')
        cuda.memset_d32(num_cps_st_dp, 0, 1)

        func = ctx.mod.get_function('iter_thread')
        dtime = func(block=ctx.block, grid=ctx.grid, time_kernel=True)

        num_rounds_dp, num_rounds_l = ctx.mod.get_global('g_num_rounds')
        num_writes_dp, num_writes_l = ctx.mod.get_global('g_num_writes')
        rounds = cuda.from_device(num_rounds_dp, ctx.threads, np.uint32)
        writes = cuda.from_device(num_writes_dp, ctx.threads, np.uint32)
        print "Rounds:", rounds
        print "Writes:", writes

class MWCRNG(PTXFragment):
    shortname = "mwc"

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
            std.get_gtid(mwc_off)
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
            std.get_gtid(mwc_off)
            op.mov.u32(mwc_addr, mwc_rng_state)
            op.mad.lo.u32(mwc_addr, mwc_off, 8, mwc_addr)
            op.st.global_.v2.u32(addr(mwc_addr), vec(mwc_st, mwc_car))

    @ptx_func
    def _next(self):
        # Call from inside a block!
        reg.u64('mwc_out')
        op.cvt.u64.u32(mwc_out, mwc_car)
        op.mad.wide.u32(mwc_out, mwc_st, mwc_mult, mwc_out)
        op.mov.b64(vec(mwc_st, mwc_car), mwc_out)

    @ptx_func
    def next_b32(self, dst_reg):
        with block('Load next random u32 into ' + dst_reg.name):
            self._next()
            op.mov.u32(dst_reg, mwc_st)

    @ptx_func
    def next_f32_01(self, dst_reg):
        # TODO: verify that this is the fastest-performance method
        # TODO: verify that this actually does what I think it does
        with block('Load random float [0,1] into ' + dst_reg.name):
            self._next()
            op.cvt.rn.f32.u32(dst_reg, mwc_st)
            op.mul.f32(dst_reg, dst_reg, '0f0000802F') # 1./(1<<32)

    @ptx_func
    def next_f32_11(self, dst_reg):
        with block('Load random float [-1,1) into ' + dst_reg.name):
            self._next()
            op.cvt.rn.f32.s32(dst_reg, mwc_st)
            op.mul.f32(dst_reg, dst_reg, '0f00000030') # 1./(1<<31)

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
            mwc.next_b32(addend)
            op.cvt.u64.u32(addl, addend)
            op.add.u64(sum, sum, addl)
            op.sub.u32(loopct, loopct, 1)
            op.setp.gt.u32(p, loopct, 0)
            op.bra.uni(loopstart, ifp=p)

        with block('Store sum and state'):
            reg.u32('adr offset')
            std.get_gtid(offset)
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
    shortname = 'cp'

