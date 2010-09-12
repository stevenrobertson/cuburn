#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
Various micro-benchmarks and other experiments.
"""
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from cuburn.ptx import PTXFragment, PTXTest, ptx_func, instmethod
from cuburn.cuda import LaunchContext
from cuburn.device_code import MWCRNG, MWCRNGTest

class L2WriteCombining(PTXTest):
    """
    Test of L2 write combining.
    """
    entry_name = 'l2_write_combining'
    entry_params = [('u64', 'a_report_addr'), ('u64', 'a_scratch_addr')]

    block_size = 2**20 # 1MB/CTA.
    rounds = int(1e6)

    @ptx_func
    def entry(self):
        mem.shared.u32('s_offset')
        reg.u32('bytes_written offset write_size laneid ctaid rounds x')
        reg.u64('scratch_addr scratch_offset clka clkb bytes')
        reg.pred('p_write p_loop_wrsz p_is_first p_done p_coalesced')

        op.mov.u32(laneid, '%laneid')
        op.setp.eq.u32(p_is_first, laneid, 0)

        op.ld.param.u32(scratch_addr, addr(a_scratch_addr))
        op.mov.u32(ctaid, '%ctaid.x')
        op.cvt.u64.u32(scratch_offset, ctaid)
        op.mad.lo.u64(scratch_addr, scratch_offset, self.block_size,
                      scratch_addr)

        op.mov.u32(x, 0)

        label('l2_restart')
        comment("If CTA is even, do coalesced first")
        op.and_.b32(ctaid, ctaid, 1)
        op.setp.eq.u32(p_coalesced, ctaid, 0)
        op.bra.uni('l2_loop_start')

        label('l2_loop_start')
        op.st.shared.u32(addr(s_offset), 0, ifp=p_is_first)
        op.mov.u32(rounds, 0)
        op.mov.u32(write_size, 16)
        op.mov.u64(clka, '%clock64')
        op.mov.u64(bytes, 0)

        label('l2_loop')
        comment("Increment offset across the CTA")
        op.atom.shared.add.u32(offset, addr(s_offset), write_size,
                               ifp=p_is_first)

        comment("Find write address from current offset and lane")
        op.ld.shared.u32(offset, addr(s_offset))
        op.add.u32(offset, offset, laneid)
        op.mul.lo.u32(offset, offset, 8)
        op.and_.b32(offset, offset, self.block_size-1)

        op.cvt.u64.u32(scratch_offset, offset)
        op.add.u64(scratch_offset, scratch_offset, scratch_addr)

        comment("If lane < write_size, write to address")
        op.setp.lt.u32(p_write, laneid, write_size)
        op.st.u64(addr(scratch_offset), scratch_offset, ifp=p_write)

        comment("Add to number of bytes written")
        op.add.u64(bytes, bytes, 8, ifp=p_write)

        comment("If uncoalesced, store new write size")
        op.add.u32(write_size, write_size, 1, ifnotp=p_coalesced)
        op.setp.gt.u32(p_loop_wrsz, write_size, 32)
        op.mov.u32(write_size, 2, ifp=p_loop_wrsz)

        comment("Loop!")
        op.add.u32(rounds, rounds, 1)
        op.setp.ge.u32(p_done, rounds, self.rounds)
        op.bra.uni(l2_loop, ifnotp=p_done)

        label('l2_loop_end')
        op.mov.u64(clkb, '%clock64')
        op.sub.u64(clka, clkb, clka)
        with block("Store the time l2_loop took"):
            reg.u64('report_addr report_offset')
            reg.u32('gtid')
            std.get_gtid(gtid)
            op.mul.lo.u32(gtid, gtid, 32)
            op.add.u32(gtid, gtid, 16, ifnotp=p_coalesced)
            op.cvt.u64.u32(report_offset, gtid)
            op.ld.param.u64(report_addr, addr(a_report_addr))
            op.add.u64(report_addr, report_addr, report_offset)
            op.st.u64(addr(report_addr), clka)
            op.st.u64(addr(report_addr,8), bytes)

        comment("If we did coalesced, go back and do uncoalesced")
        op.add.u32(ctaid, ctaid, 1)
        op.add.u32(x, x, 1)
        op.setp.ge.u32(p_done, x, 2)
        op.bra.uni(l2_restart, ifnotp=p_done)

    def call_setup(self, ctx):
        self.scratch = np.zeros(self.block_size*ctx.nctas/4, np.uint64)
        self.times_bytes = np.zeros((4, ctx.nthreads), np.uint64, 'F')

    def _call(self, ctx, func):
        super(L2WriteCombining, self)._call(ctx, func,
                cuda.InOut(self.times_bytes), cuda.InOut(self.scratch))

    def call_teardown(self, ctx):
        pm = lambda a: (np.mean(a), np.std(a) / np.sqrt(len(a)))
        print "Clks for coa was %g ± %g" % pm(self.times_bytes[0])
        print "Bytes for coa was %g ± %g" % pm(self.times_bytes[1])
        print "Clks for uncoa was %g ± %g" % pm(self.times_bytes[2])
        print "Bytes for uncoa was %g ± %g" % pm(self.times_bytes[3])
        print

class SimulOccupancy(PTXTest):
    """
    Test to discover whether Fermi GPUs will launch multiple entry points
    in the same kernel on the same CTA simultaneously.
    """
    entry_name = 'simul1'
    # Only has to be big enough to hold the kernel on the device for a while
    rounds = 1000000

    def deps(self):
        return [MWCRNG]

    @ptx_func
    def module_setup(self):
        n = self.entry_name + '_'
        mem.global_.u64(n+'start', ctx.nthreads)
        mem.global_.u64(n+'end', ctx.nthreads)
        mem.global_.u32(n+'smid', ctx.nthreads)
        mem.global_.u32(n+'warpid_start', ctx.nthreads)
        mem.global_.u32(n+'warpid_end', ctx.nthreads)

    @ptx_func
    def entry(self):
        n = self.entry_name + '_'
        reg.u64('now')
        reg.u32('warpid')
        op.mov.u64(now, '%clock64')
        op.mov.u32(warpid, '%warpid')
        std.store_per_thread(n+'start', now,
                             n+'warpid_start', warpid)

        reg.u32('loopct rnd')
        reg.pred('p_done')
        op.mov.u32(loopct, self.rounds)
        label('loopstart')
        mwc.next_b32(rnd)
        std.store_per_thread(n+'smid', rnd)
        op.sub.u32(loopct, loopct, 1)
        op.setp.eq.u32(p_done, loopct, 0)
        op.bra.uni(loopstart, ifnotp=p_done)

        reg.u32('smid')
        op.mov.u32(smid, '%smid')
        op.mov.u32(warpid, '%warpid')
        op.mov.u64(now, '%clock64')
        std.store_per_thread(n+'end', now,
                             n+'smid', smid,
                             n+'warpid_end', warpid)

    def _call(self, ctx, func):
        stream1, stream2 = cuda.Stream(), cuda.Stream()
        self._call2(ctx, stream1)
        _SimulOccupancy._call2(ctx, stream2)
        stream2.synchronize()
        stream1.synchronize()

    @instmethod
    def _call2(self, ctx, stream):
        func = ctx.mod.get_function(self.entry_name)
        func.prepare([], ctx.block)
        # TODO: load number of SMs from ctx
        func.launch_grid_async(7, 1, stream)

    def call_teardown(self, ctx):
        sm_log = [[] for i in range(7)]
        self._teardown(ctx, sm_log)
        _SimulOccupancy._teardown(ctx, sm_log)
        for sm in range(len(sm_log)):
            print "\nPrinting log for SM %d" % sm
            for t, ev in sorted(sm_log[sm]):
                print '%6d %s' % (t/1000, ev)

    @instmethod
    def _teardown(self, ctx, sm_log):
        # For this method, the GPU is intentionally underloaded; trim results
        th = 7 * ctx.threads_per_cta
        n = self.entry_name + '_'
        start = ctx.get_per_thread(n+'start', np.uint64)[:th]
        end = ctx.get_per_thread(n+'end', np.uint64)[:th]
        smid = ctx.get_per_thread(n+'smid', np.uint32)[:th]
        warpid_start = ctx.get_per_thread(n+'warpid_start', np.uint32)[:th]
        warpid_end = ctx.get_per_thread(n+'warpid_end', np.uint32)[:th]
        for i in range(0, th, 32):
            sm_log[smid[i]].append((start[i], "%s%4d entered SM" % (n, i/32)))
            sm_log[smid[i]].append((end[i],   "%s%4d left SM" % (n, i/32)))
        if not np.alltrue(np.equal(warpid_start, warpid_end)):
            print "Warp IDs changed. Do further research."

class _SimulOccupancy(SimulOccupancy):
    # Don't call this one
    entry_name = 'simul2'
    def call(self, ctx):
        pass
    def call_teardown(self, ctx):
        pass

def printover(a, r, s=1):
    for i in range(0, len(a), r*s):
        for j in range(i, i+r*s, s):
            if j < len(a): print a[j],
        print

def main():
    # TODO: block/grid auto-optimization
    ctx = LaunchContext([L2WriteCombining, SimulOccupancy, _SimulOccupancy],
                        block=(128,1,1), grid=(7*8,1), tests=True)
    ctx.compile(verbose=3)
    ctx.run_tests()
    SimulOccupancy.call(ctx)
    L2WriteCombining.call(ctx)

if __name__ == "__main__":
    main()


