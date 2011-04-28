"""
Contains the PTX fragments which will drive the device.
"""

import os
import time
import struct

import pycuda.driver as cuda
import numpy as np

from pyptx import ptx, run, util
from cuburn.variations import Variations

class IterThread(object):
    def __init__(self, entry, features):
        self.features = features
        self.mwc = MWCRNG(entry)
        self.cp = util.DataStream(entry)
        self.vars = Variations(features)
        self.camera = CameraTransform(features)
        self.hist = HistScatter(entry, features)
        self.shuf = ShufflePoints(entry)

        entry.add_param('u32', 'num_cps')
        entry.add_ptr_param('u32', 'cp_started_count')
        entry.add_ptr_param('u8', 'cp_data')

        with entry.body():
            self.entry_body(entry)

    def entry_body(self, entry):
        e, r, o, m, p, s = entry.locals
        # Index of this CTA's current CP
        e.declare_mem('shared', 'u32', 'cp_idx')

        # Number of samples that have been generated so far in this CTA
        # If this number is negative, we're still fusing points, so this
        # behaves slightly differently (see ``fuse_loop_start``)
        # TODO: replace (or at least simplify) this logic
        e.declare_mem('shared', 'u32', 'num_samples')

        # The per-warp transform selection indices
        e.declare_mem('shared', 'f32', 'xf_sel', e.nwarps_cta)

        # TODO: re-add this logic using the printf formatter.
        #mem.local.u32('l_num_rounds')
        #mem.local.u32('l_num_writes')
        #op.st.local.u32(addr(l_num_rounds), 0)
        #op.st.local.u32(addr(l_num_writes), 0)

        # Declare IFS-space coordinates for doing iterations
        r.x, r.y, r.color = r.f32(), r.f32(), r.f32()
        r.x, r.y = self.mwc.next_f32_11(), self.mwc.next_f32_11()
        r.color = self.mwc.next_f32_01()

        # This thread's sample's good/bad/fusing state
        r.consec_bad = r.f32(-self.features.fuse)

        e.comment("The main loop entry point")
        cp_loop_start = e.label()
        with s.tid_x == 0:
            o.st(m.cp_idx.addr, o.atom.add(p.cp_started_count[0], 1))
            o.st(m.num_samples.addr, 0)

        e.comment("Load the CP index in all threads")
        o.bar.sync(0)
        cp_idx = o.ld.volatile(m.cp_idx.addr)

        e.comment("Check to see if this CP is valid (if not, we're done)")
        all_cps_done = e.forward_label()
        with cp_idx < p.num_cps.val:
            o.bra.uni(all_cps_done)
        self.cp.addr = p.cp_data[cp_idx * self.cp.stream_size]

        loop_start = e.forward_label()
        with s.tid_x < e.nwarps_cta:
            o.bra(loop_start)

        e.comment("Choose the xform for each warp")
        choose_xform = e.label()
        o.st.volatile(m.xf_sel[s.tid_x], self.mwc.next_f32_01())

        e.declare_label(loop_start)
        e.comment("Execute the xform given by xf_sel")
        xf_labels = [e.forward_label() for xf in self.features.xforms]
        xf_sel = o.ld.volatile(m.xf_sel[s.tid_x >> 5])
        for i, xf in enumerate(self.features.xforms):
            xf_density = self.cp.get.f32('cp.xforms[%d].cweight'%xf.id)
            with xf_density <= xf_sel:
                o.bra.uni(xf_labels[i])

        e.comment("This code should be unreachable")
        o.trap()

        xforms_done = e.forward_label()
        for i, xf in enumerate(self.features.xforms):
            e.declare_label(xf_labels[i])
            r.x, r.y, r.color = self.vars.apply_xform(
                    e, self.cp, r.x, r.y, r.color, xf.id)
            o.bra.uni(xforms_done)

        e.comment("Determine write location, and whether point is valid")
        e.declare_label(xforms_done)
        histidx, is_valid = self.camera.get_index(e, self.cp, r.x, r.y)
        is_valid &= (r.consec_bad >= 0)

        e.comment("Scatter point to pointbuffer")
        self.hist.scatter(self.cp, histidx, r.color, 0, is_valid)

        done_picking_new_point = e.forward_label()
        with ~is_valid:
            r.consec_bad += 1
        with r.consec_bad < self.features.max_bad:
            o.bra(done_picking_new_point)

        e.comment("If too many consecutive bad values, pick a new point")
        r.x, r.y = self.mwc.next_f32_11(), self.mwc.next_f32_11()
        r.color = self.mwc.next_f32_01()
        r.consec_bad = -self.features.fuse

        e.declare_label(done_picking_new_point)

        e.comment("Determine number of good samples, and whether we're done")
        num_samples = o.ld(m.num_samples.addr)
        num_samples += o.bar.red.popc(0, is_valid)
        with s.tid_x == 0:
            o.st(m.num_samples, num_samples)
        with num_samples >= self.cp.get.u32('nsamples'):
            o.bra.uni(cp_loop_start)

        self.shuf.shuffle(e, r.x, r.y, r.color, r.consec_bad)

        with s.tid_x < e.nwarps_cta:
            o.bra(choose_xform)
        o.bra(loop_start)

        e.declare_label(all_cps_done)

    def upload_cp_stream(self, ctx, cp_stream, num_cps):
        cp_array_dp, cp_array_l = ctx.mod.get_global('g_cp_array')
        assert len(cp_stream) <= cp_array_l, "Stream too big!"
        cuda.memcpy_htod(cp_array_dp, cp_stream)

        num_cps_dp, num_cps_l = ctx.mod.get_global('g_num_cps')
        cuda.memset_d32(num_cps_dp, num_cps, 1)
        # TODO: "if debug >= 3"
        print "Uploaded stream to card:"
        CPDataStream.print_record(ctx, cp_stream, 5)
        self.cps_uploaded = True

    def call_setup(self, ctx):
        if not self.cps_uploaded:
            raise Error("Cannot call IterThread before uploading CPs")
        num_cps_st_dp, num_cps_st_l = ctx.mod.get_global('g_num_cps_started')
        cuda.memset_d32(num_cps_st_dp, 0, 1)

    def _call(self, ctx, func):
        # Get texture reference from the Palette
        # TODO: more elegant method than reaching into ctx.ptx?
        tr = ctx.ptx.instances[PaletteLookup].texref
        super(IterThread, self)._call(ctx, func, texrefs=[tr])

    def call_teardown(self, ctx):
        def print_thing(s, a):
            print '%s:' % s
            for i, r in enumerate(a):
                for j in range(0,len(r),ctx.warps_per_cta):
                    print '%2d' % i,
                    for k in range(j,j+ctx.warps_per_cta,8):
                        print '\t' + ' '.join(
                            ['%8g'%np.mean(r[l]) for l in range(k,k+8)])

        rounds = ctx.get_per_thread('g_num_rounds', np.int32, shaped=True)
        writes = ctx.get_per_thread('g_num_writes', np.int32, shaped=True)
        print_thing("Rounds", rounds)
        print_thing("Writes", writes)
        print "Total number of rounds:", np.sum(rounds)

        dp, l = ctx.mod.get_global('g_num_cps_started')
        cps_started = cuda.from_device(dp, 1, np.uint32)
        print "CPs started:", cps_started

class CameraTransform(object):
    def __init__(self, features):
        self.features = features

    def rotate(self, cp, x, y):
        """
        Rotate an IFS-space coordinate as defined by the camera.
        """
        if not self.features.camera_rotation:
            return x, y
        rot_center_x, rot_center_y = cp.get.v2.f32('cp.rot_center[0]',
                                                   'cp.rot_center[1]')
        tx, ty = x - rot_center_x, y - rot_center_y
        rot_cos_t, rot_sin_t = cp.get.v2.f32('cos(cp.rotate * 2 * pi / 360.)',
                                            '-sin(cp.rotate * 2 * pi / 360.)')
        rx = tx * rot_cos_t    + ty * rot_sin_t + rot_center_x
        ry = tx * (-rot_sin_t) + ty * rot_cos_t + rot_center_y
        return rx, ry

    def get_norm(self, cp, x, y):
        """
        Find the [0,1]-normalized floating-point histogram coordinates
        ``norm_x, norm_y`` from the given IFS-space coordinates ``x, y``.
        """
        rx, ry = self.rotate(cp, x, y)
        cam_scale, cam_offset = cp.get.v2.f32(
                'cp.camera.norm_scale[0]', 'cp.camera.norm_offset[0]')
        norm_x = rx * cam_scale + cam_offset
        cam_scale, cam_offset = cp.get.v2.f32(
                'cp.camera.norm_scale[1]', 'cp.camera.norm_offset[1]')
        norm_y = ry * cam_scale + cam_offset
        return norm_x, norm_y

    def get_index(self, entry, cp, x, y):
        """
        Find the histogram index (as a u32) from the IFS spatial coordinate in
        ``x, y``. Returns ``index, oob``, where ``oob`` is a predicate value
        that is set if the result is out of bounds.
        """
        o = entry.ops
        rx, ry = self.rotate(cp, x, y)
        cam_scale, cam_offset = cp.get.v2.f32(
                'cp.camera.idx_scale[0]', 'cp.camera.idx_offset[0]')
        idx_x = rx * cam_scale + cam_offset
        cam_scale, cam_offset = cp.get.v2.f32(
                'cp.camera.idx_scale[1]', 'cp.camera.idx_offset[1]')
        idx_y = ry * cam_scale + cam_offset

        idx_x_u32 = o.cvt.rzi.s32(idx_x)
        idx_y_u32 = o.cvt.rzi.s32(idx_y)
        oob = o.setp.lt.u32(idx_x_u32, self.features.hist_width)
        oob |= o.setp.lt.u32(idx_y_u32, self.features.hist_height)

        idx = idx_y_u32 * self.features.hist_stride + idx_x_u32
        return idx, oob

class PaletteLookup(object):
    def __init__(self, entry, features):
        self.entry, self.features = entry, features
        #entry.declare_mem('global', 'texref', 'palette')

    def look_up(self, color, norm_time):
        """
        Look up the values of ``r, g, b, a`` corresponding to ``color_coord``
        at the CP indexed in ``timestamp_idx``. Note that both ``color_coord``
        and ``timestamp_idx`` should be [0,1]-normalized floats.
        """
        n = self.entry.ops.mov.f32
        return n(1), n(1), n(1), n(1)
        #o.tex._2d.v4.f32.f32(m.palette, norm_time, color)
        if features.non_box_temporal_filter:
            raise NotImplementedError("Non-box temporal filters not supported")

    def upload_palette(self, ctx, frame, cp_list):
        """
        Extract the palette from the given list of interpolated CPs, and upload
        it to the device as a texture.
        """
        # TODO: figure out if storing the full list is an actual drag on
        # performance/memory
        if frame.center_cp.temporal_filter_type != 0:
            # TODO: make texture sample based on time, not on CP index
            raise NotImplementedError("Use box temporal filters for now")
        pal = np.ndarray((self.texheight, 256, 4), dtype=np.float32)
        inv = float(len(cp_list) - 1) / (self.texheight - 1)
        for y in range(self.texheight):
            for x in range(256):
                for c in range(4):
                    # TODO: interpolate here?
                    cy = int(round(y * inv))
                    pal[y][x][c] = cp_list[cy].palette.entries[x].color[c]
        dev_array = cuda.make_multichannel_2d_array(pal, "C")
        self.texref = ctx.mod.get_texref('t_palette')
        # TODO: float16? or can we still use interp with int storage?
        self.texref.set_format(cuda.array_format.FLOAT, 4)
        self.texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        self.texref.set_filter_mode(cuda.filter_mode.LINEAR)
        self.texref.set_address_mode(0, cuda.address_mode.CLAMP)
        self.texref.set_address_mode(1, cuda.address_mode.CLAMP)
        self.texref.set_array(dev_array)
        self.pal = pal

    def call_setup(self, ctx):
        assert self.texref, "Must upload palette texture before launch!"

class HistScatter(object):
    def __init__(self, entry, features):
        self.entry, self.features = entry, features
        self.palette = PaletteLookup(entry, features)
        entry.add_ptr_param('f32', 'hist_bins')

    def scatter(self, cp, hist_index, color, xf_idx, p_valid):
        """
        Scatter the given point directly to the histogram bins.
        """
        e, r, o, m, p, s = self.entry.locals
        norm_time = cp.get.f32('cp.norm_time')
        base = p.hist_bins[4*hist_index]
        colors = self.palette.look_up(color, norm_time)
        g_colors = o.ld.v4(base)
        for col, gcol in zip(colors, g_colors):
            gcol += col
        o.st.v4(base, *g_colors)


class ShufflePoints(object):
    """
    Shuffle points in shared memory. See helpers/shuf.py for details.
    """
    def __init__(self, entry):
        entry.declare_mem('shared', 'b32', 'shuf_data', entry.nthreads_cta)

    def shuffle(self, entry, *args, **kwargs):
        """
        Shuffle the data from each register in args across threads. Keyword
        argument ``bar`` specifies which barrier to use (default is 0). Each
        register is overwritten in place.
        """
        e, r, o, m, p, s = entry.locals
        bar = kwargs.pop('bar', 0)
        assert not kwargs, "Unrecognized keyword arguments."

        e.comment("Calculate read and write offsets for shuffle")
        # See helpers/shuf.py for details
        shuf_write = m.shuf_data[s.tid_x]
        shuf_read = m.shuf_data[(s.tid_x + (32 * s.laneid)) &
                                (e.nthreads_cta - 1)]
        for var in args:
            o.bar.sync(bar)
            o.st.volatile(shuf_write, var)
            o.bar.sync(bar)
            var.val = o.ld.volatile(shuf_read)

class MWCRNG(object):
    """
    Marsaglia multiply-with-carry random number generator. Produces very long
    periods with sufficient statistical properties using only three 32-bit
    state registers. Since each thread uses a separate multiplier, no two
    threads will ever be on the same sequence, but beyond this the independence
    of each thread's sequence was not explicitly tested.

    The RNG must be seeded at least once per entry point using the ``seed``
    method.
    """
    def __init__(self, entry):
        # TODO: install this in data directory or something
        if not os.path.isfile('primes.bin'):
            raise EnvironmentError('primes.bin not found')
        self.nthreads_ready = 0
        self.mults, self.state = None, None
        self.entry = entry

        entry.add_ptr_param('u32', 'mwc_mults')
        entry.add_ptr_param('u32', 'mwc_states')

        with entry.head():
            self.entry_head()
        entry.tail_callback(self.entry_tail)

    def entry_head(self):
        e, r, o, m, p, s = self.entry.locals
        gtid = s.ctaid_x * s.ntid_x + s.tid_x
        r.mwc_mult, r.mwc_state, r.mwc_carry = r.u32(), r.u32(), r.u32()
        r.mwc_mult = o.ld(p.mwc_mults[gtid])
        r.mwc_state, r.mwc_carry = o.ld.v2(p.mwc_states[2*gtid])

    def entry_tail(self):
        e, r, o, m, p, s = self.entry.locals
        gtid = s.ctaid_x * s.ntid_x + s.tid_x
        o.st.v2.u32(p.mwc_states[2*gtid], r.mwc_state, r.mwc_carry)

    def next_b32(self):
        e, r, o, m, p, s = self.entry.locals
        carry = o.cvt.u64(r.mwc_carry)
        mwc_out = o.mad.wide(r.mwc_mult, r.mwc_state, carry)
        r.mwc_state, r.mwc_carry = o.split.v2(mwc_out)
        return r.mwc_state

    def next_f32_01(self):
        e, r, o, m, p, s = self.entry.locals
        mwc_float = o.cvt.rn.f32.u32(self.next_b32())
        return o.mul.f32(mwc_float, 1./(1<<32))

    def next_f32_11(self):
        e, r, o, m, p, s = self.entry.locals
        mwc_float = o.cvt.rn.f32.s32(self.next_b32())
        return o.mul.f32(mwc_float, 1./(1<<31))

    def seed(self, ctx, seed=None, force=False):
        """
        Seed the random number generators with values taken from a
        ``np.random`` instance.
        """
        if force or self.nthreads_ready < ctx.nthreads:
            if seed:
                rand = np.random.RandomState(seed)
            else:
                rand = np.random
            # Load raw big-endian u32 multipliers from primes.bin.
            with open('primes.bin') as primefp:
                dt = np.dtype(np.uint32).newbyteorder('B')
                mults = np.frombuffer(primefp.read(), dtype=dt)
            # Randomness in choosing multipliers is good, but larger multipliers
            # have longer periods, which is also good. This is a compromise.
            mults = np.array(mults[:ctx.nthreads*4])
            rand.shuffle(mults)
            #locked_mults = ctx.hostpool.allocate(ctx.nthreads, np.uint32)
            #locked_mults[:] = mults[ctx.nthreads]
            #self.mults = ctx.pool.allocate(4*ctx.nthreads)
            #cuda.memcpy_htod_async(self.mults, locked_mults.base, ctx.stream)
            self.mults = cuda.mem_alloc(4*ctx.nthreads)
            cuda.memcpy_htod(self.mults, mults[:ctx.nthreads].tostring())
            # Intentionally excludes both 0 and (2^32-1), as they can lead to
            # degenerate sequences of period 0
            states = np.array(rand.randint(1, 0xffffffff, size=2*ctx.nthreads),
                              dtype=np.uint32)
            #locked_states = ctx.hostpool.allocate(2*ctx.nthreads, np.uint32)
            #locked_states[:] = states
            #self.states = ctx.pool.allocate(8*ctx.nthreads)
            #cuda.memcpy_htod_async(self.states, locked_states, ctx.stream)
            self.states = cuda.mem_alloc(8*ctx.nthreads)
            cuda.memcpy_htod(self.states, states.tostring())
            self.nthreads_ready = ctx.nthreads
        ctx.set_param('mwc_mults', self.mults)
        ctx.set_param('mwc_states', self.states)

class MWCRNGTest(object):
    """
    Test the ``MWCRNG`` class. This is not a test of the generator's
    statistical properties, but merely a test that the generator is implemented
    correctly on the GPU.
    """
    rounds = 5000

    def __init__(self, entry):
        self.mwc = MWCRNG(entry)
        entry.add_ptr_param('u64', 'mwc_test_sums')

        with entry.body():
            self.entry_body(entry)

    def entry_body(self, entry):
        e, r, o, m, p, s = entry.locals
        r.sum = r.u64(0)
        r.count = r.f32(self.rounds)
        start = e.label()
        r.sum = r.sum + o.cvt.u64.u32(self.mwc.next_b32())
        r.count = r.count - 1
        with r.count > 0:
            o.bra.uni(start)
        e.comment('yay')
        gtid = s.ctaid_x * s.ntid_x + s.tid_x
        o.st(p.mwc_test_sums[gtid], r.sum)

    def run_test(self, ctx):
        self.mwc.seed(ctx)
        mults = cuda.from_device(self.mwc.mults, ctx.nthreads, np.uint32)
        states = cuda.from_device(self.mwc.states, ctx.nthreads, np.uint64)

        for trial in range(2):
            print "Trial %d, on CPU: " % trial,
            sums = np.zeros_like(states)
            ctime = time.time()
            for i in range(self.rounds):
                vals = states & 0xffffffff
                carries = states >> 32
                states = mults * vals + carries
                sums += states & 0xffffffff
            ctime = time.time() - ctime
            print "Took %g seconds." % ctime

            print "Trial %d, on device: " % trial,
            dsums = cuda.mem_alloc(8*ctx.nthreads)
            ctx.set_param('mwc_test_sums', dsums)
            print "Took %g seconds." % ctx.call_timed()
            dsums = cuda.from_device(dsums, ctx.nthreads, np.uint64)
            if not np.all(np.equal(sums, dsums)):
                print "Sum discrepancy!"
                print sums
                print dsums

class MWCRNGFloatsTest(object):
    """
    Note this only tests that the distributions are in the correct range, *not*
    that they have good random properties. MWC is a suitable algorithm, but
    implementation bugs may still lead to poor performance.
    """
    rounds = 1024
    entry_name = 'MWC_RNG_floats_test'

    def deps(self):
        return [MWCRNG]

    def module_setup(self):
        mem.global_.f32('mwc_rng_float_01_test_sums', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_01_test_mins', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_01_test_maxs', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_11_test_sums', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_11_test_mins', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_11_test_maxs', ctx.nthreads)

    def loop(self, kind):
        with block('Sum %d floats in %s' % (self.rounds, kind)):
            reg.f32('loopct val rsum rmin rmax')
            reg.pred('p_done')
            op.mov.f32(loopct, 0.)
            op.mov.f32(rsum, 0.)
            op.mov.f32(rmin, 2.)
            op.mov.f32(rmax, -2.)
            label('loopstart' + kind)
            getattr(mwc, 'next_f32_' + kind)(val)
            op.add.f32(rsum, rsum, val)
            op.min.f32(rmin, rmin, val)
            op.max.f32(rmax, rmax, val)
            op.add.f32(loopct, loopct, 1.)
            op.setp.ge.f32(p_done, loopct, float(self.rounds))
            op.bra('loopstart' + kind, ifnotp=p_done)
            op.mul.f32(rsum, rsum, 1./self.rounds)
            std.store_per_thread('mwc_rng_float_%s_test_sums' % kind, rsum,
                                 'mwc_rng_float_%s_test_mins' % kind, rmin,
                                 'mwc_rng_float_%s_test_maxs' % kind, rmax)

    def entry(self):
        self.loop('01')
        self.loop('11')

    def call_teardown(self, ctx):
        # Tolerance of all-threads averages
        tol = 0.05
        # float distribution kind, test kind, expected value, limit func
        tests = [
                    ('01', 'sums',  0.5, None),
                    ('01', 'mins',  0.0, np.min),
                    ('01', 'maxs',  1.0, np.max),
                    ('11', 'sums',  0.0, None),
                    ('11', 'mins', -1.0, np.min),
                    ('11', 'maxs',  1.0, np.max)
                ]

        for fkind, rkind, exp, lim in tests:
            name = 'mwc_rng_float_%s_test_%s' % (fkind, rkind)
            vals = ctx.get_per_thread(name, np.float32)
            avg = np.mean(vals)
            if np.abs(avg - exp) > tol:
                raise PTXTestFailure("%s %s %g too far from %g" %
                                     (fkind, rkind, avg, exp))
            if lim is None: continue
            if lim([lim(vals), exp]) != exp:
                raise PTXTestFailure("%s %s %g violates hard limit %g" %
                                     (fkind, rkind, lim(vals), exp))

class CPDataStream(object):
    """DataStream which stores the control points."""
    shortname = 'cp'

class Timeouter(object):
    """Time-out infinite loops so that data can still be retrieved."""
    shortname = 'timeout'

    def entry_setup(self):
        mem.shared.u64('s_timeouter_start_time')
        with block("Load start time for this block"):
            reg.u64('now')
            op.mov.u64(now, '%clock64')
            op.st.shared.u64(addr(s_timeouter_start_time), now)

    def check_time(self, secs):
        """
        Drop this into your mainloop somewhere.
        """
        # TODO: if debug.device_timeout_loops or whatever
        with block("Check current time for this loop"):
            d = cuda.Context.get_device()
            clks = int(secs * d.clock_rate * 1000)
            reg.u64('now then')
            op.mov.u64(now, '%clock64')
            op.ld.shared.u64(then, addr(s_timeouter_start_time))
            op.sub.u64(now, now, then)
            std.asrt("Loop timed out", 'lt.u64', now, clks)


