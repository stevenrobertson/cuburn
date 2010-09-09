"""
Contains the PTX fragments which will drive the device.
"""

import os
import time
import struct

import pycuda.driver as cuda
import numpy as np

from cuburnlib.ptx import *

class IterThread(PTXEntryPoint):
    entry_name = 'iter_thread'
    entry_params = []

    def __init__(self):
        self.cps_uploaded = False

    def deps(self):
        return [MWCRNG, CPDataStream, HistScatter]

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
            op.atom.add.u32(cp_idx, addr(g_num_cps_started), 1, ifp=p_is_first)
            op.st.shared.u32(addr(s_cp_idx), cp_idx, ifp=p_is_first)
            op.st.shared.u32(addr(s_num_samples), 0, ifp=p_is_first)

        comment("Load the CP index in all threads")
        op.bar.sync(1)
        op.ld.shared.u32(cp_idx, addr(s_cp_idx))

        with block("Check to see if this CP is valid (if not, we're done)"):
            reg.u32('num_cps')
            reg.pred('p_last_cp')
            op.ldu.u32(num_cps, addr(g_num_cps))
            op.setp.ge.u32(p_last_cp, cp_idx, num_cps)
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
            op.bar.sync(2)

        label('iter_loop_start')

        comment('Do... well, most of everything')

        mwc.next_f32_11(x_coord)
        mwc.next_f32_11(y_coord)
        mwc.next_f32_01(color_coord)

        op.add.u32(num_rounds, num_rounds, 1)

        with block("Test if we're still in FUSE"):
            reg.s32('num_samples')
            reg.pred('p_in_fuse')
            op.ld.shared.s32(num_samples, addr(s_num_samples))
            op.setp.lt.s32(p_in_fuse, num_samples, 0)
            op.bra.uni(fuse_loop_start, ifp=p_in_fuse)

        reg.pred('p_point_is_valid')
        with block("Write the result"):
            hist.scatter(x_coord, y_coord, color_coord, 0, p_point_is_valid)
            op.add.u32(num_writes, num_writes, 1, ifp=p_point_is_valid)

        with block("Increment number of samples by number of good values"):
            reg.b32('good_samples laneid')
            reg.pred('p_is_first')
            op.vote.ballot.b32(good_samples, p_point_is_valid)
            op.popc.b32(good_samples, good_samples)
            op.mov.u32(laneid, '%laneid')
            op.setp.eq.u32(p_is_first, laneid, 0)
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

    @instmethod
    def upload_cp_stream(self, ctx, cp_stream, num_cps):
        cp_array_dp, cp_array_l = ctx.mod.get_global('g_cp_array')
        assert len(cp_stream) <= cp_array_l, "Stream too big!"
        cuda.memcpy_htod_async(cp_array_dp, cp_stream)

        num_cps_dp, num_cps_l = ctx.mod.get_global('g_num_cps')
        cuda.memset_d32(num_cps_dp, num_cps, 1)
        # TODO: "if debug >= 3"
        print "Uploaded stream to card:"
        CPDataStream.print_record(ctx, cp_stream, 5)
        self.cps_uploaded = True

    @instmethod
    def call(self, ctx):
        if not self.cps_uploaded:
            raise Error("Cannot call IterThread before uploading CPs")
        num_cps_st_dp, num_cps_st_l = ctx.mod.get_global('g_num_cps_started')
        cuda.memset_d32(num_cps_st_dp, 0, 1)

        func = ctx.mod.get_function('iter_thread')
        tr = ctx.ptx.instances[PaletteLookup].texref
        dtime = func(block=ctx.block, grid=ctx.grid, time_kernel=True,
                     texrefs=[tr])

        shape = (ctx.grid[0], ctx.block[0]/32, 32)
        num_rounds_dp, num_rounds_l = ctx.mod.get_global('g_num_rounds')
        num_writes_dp, num_writes_l = ctx.mod.get_global('g_num_writes')
        rounds = cuda.from_device(num_rounds_dp, shape, np.int32)
        writes = cuda.from_device(num_writes_dp, shape, np.int32)
        print "Rounds:", sum(rounds)
        print "Writes:", sum(writes)
        print rounds
        print writes

class CameraTransform(PTXFragment):
    shortname = 'camera'
    def deps(self):
        return [CPDataStream]

    @ptx_func
    def rotate(self, rotated_x, rotated_y, x, y):
        """
        Rotate an IFS-space coordinate as defined by the camera.
        """
        if features.camera_rotation:
            assert rotated_x.name != x.name and rotated_y.name != y.name
            with block("Rotate %s, %s to camera alignment" % (x, y)):
                reg.f32('rot_center_x rot_center_y')
                cp.get_v2(cpA, rot_center_x, 'cp.rot_center[0]',
                                      rot_center_y, 'cp.rot_center[1]')
                op.sub.f32(x, x, rot_center_x)
                op.sub.f32(y, y, rot_center_y)

                reg.f32('rot_sin_t rot_cos_t rot_old_x rot_old_y')
                cp.get_v2(cpA, rot_cos_t,  'cos(cp.rotate * 2 * pi / 360.)',
                               rot_sin_t, '-sin(cp.rotate * 2 * pi / 360.)')

                comment('rotated_x = x * cos(t) - y * sin(t) + rot_center_x')
                op.fma.rn.f32(rotated_x, x, rot_cos_t, rot_center_x)
                op.fma.rn.f32(rotated_x, y, rot_sin_t, rotated_x)

                op.neg.f32(rot_sin_t, rot_sin_t)
                comment('rotated_y = x * sin(t) + y * cos(t) + rot_center_y')
                op.fma.rn.f32(rotated_y, x, rot_sin_t, rot_center_y)
                op.fma.rn.f32(rotated_y, y, rot_cos_t, rotated_y)

                # TODO: if this is a register-critical section, reloading
                # rot_center_[xy] here should save two regs. OTOH, if this is
                # *not* reg-crit, moving the subtraction above to new variables
                # may save a few clocks
                op.add.f32(x, x, rot_center_x)
                op.add.f32(y, y, rot_center_y)
        else:
            comment("No camera rotation in this kernel")
            op.mov.f32(rotated_x, x)
            op.mov.f32(rotated_y, y)

    @ptx_func
    def get_norm(self, norm_x, norm_y, x, y):
        """
        Find the [0,1]-normalized floating-point histogram coordinates
        ``norm_x, norm_y`` from the given IFS-space coordinates ``x, y``.
        """
        self.rotate(norm_x, norm_y, x, y)
        with block("Scale rotated points to [0,1]-normalized coordinates"):
            reg.f32('cam_scale cam_offset')
            cp.get_v2(cpA, cam_scale,  'cp.camera.norm_scale[0]',
                           cam_offset, 'cp.camera.norm_offset[0]')
            op.fma.f32(norm_x, norm_x, cam_scale, cam_offset)
            cp.get_v2(cpA, cam_scale,  'cp.camera.norm_scale[1]',
                           cam_offset, 'cp.camera.norm_offset[1]')
            op.fma.f32(norm_y, norm_y, cam_scale, cam_offset)

    @ptx_func
    def get_index(self, index, x, y, pred=None):
        """
        Find the histogram index (as a u32) from the IFS spatial coordinate in
        ``x, y``.

        If the coordinates are out of bounds, 0xffffffff will be stored to
        ``index``. If ``pred`` is given, it will be set if the point is valid,
        and cleared if not.
        """
        # A few instructions could probably be shaved off of this one
        with block("Find histogram index"):
            reg.f32('norm_x norm_y')
            self.rotate(norm_x, norm_y, x, y)
            comment('Scale and offset from IFS to index coordinates')
            reg.f32('cam_scale cam_offset')
            cp.get_v2(cpA, cam_scale,  'cp.camera.idx_scale[0]',
                           cam_offset, 'cp.camera.idx_offset[0]')
            op.fma.rn.f32(norm_x, norm_x, cam_scale, cam_offset)

            cp.get_v2(cpA, cam_scale,  'cp.camera.idx_scale[1]',
                           cam_offset, 'cp.camera.idx_offset[1]')
            op.fma.rn.f32(norm_y, norm_y, cam_scale, cam_offset)

            comment('Check for bad value')
            reg.u32('index_x index_y')
            if not pred:
                pred = reg.pred('p_valid')

            op.cvt.rzi.s32.f32(index_x, norm_x)
            op.setp.ge.s32(pred, index_x, 0)
            op.setp.lt.and_.s32(pred, index_x, features.hist_width, pred)

            op.cvt.rzi.s32.f32(index_y, norm_y)
            op.setp.ge.and_.s32(pred, index_y, 0, pred)
            op.setp.lt.and_.s32(pred, index_y, features.hist_height, pred)

            op.mad.lo.u32(index, index_y, features.hist_stride, index_x)
            op.mov.u32(index, 0xffffffff, ifnotp=pred)

class PaletteLookup(PTXFragment):
    shortname = "palette"
    # Resolution of texture on device. Bigger = more palette rez, maybe slower
    texheight = 16

    def __init__(self):
        self.texref = None

    def deps(self):
        return [CPDataStream]

    @ptx_func
    def module_setup(self):
        mem.global_.texref('t_palette')

    @ptx_func
    def look_up(self, r, g, b, a, color, norm_time):
        """
        Look up the values of ``r, g, b, a`` corresponding to ``color_coord``
        at the CP indexed in ``timestamp_idx``. Note that both ``color_coord``
        and ``timestamp_idx`` should be [0,1]-normalized floats.
        """
        op.tex._2d.v4.f32.f32(vec(r, g, b, a),
                addr([t_palette, ', ',  vec(norm_time, color)]))
        if features.non_box_temporal_filter:
            raise NotImplementedError("Non-box temporal filters not supported")

    @instmethod
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

    def device_init(self, ctx):
        assert self.texref, "Must upload palette texture before launch!"

class HistScatter(PTXFragment):
    shortname = "hist"
    def deps(self):
        return [CPDataStream, CameraTransform, PaletteLookup]

    @ptx_func
    def module_setup(self):
        mem.global_.f32('g_hist_bins',
                        features.hist_height * features.hist_stride * 4)

    @ptx_func
    def entry_setup(self):
        comment("For now, assume histogram bins have been cleared by host")

    @ptx_func
    def scatter(self, x, y, color, xf_idx, p_valid=None):
        """
        Scatter the given point directly to the histogram bins. I think this
        technique has the worst performance of all of 'em. Accesses ``cpA``
        directly.
        """
        with block("Scatter directly to buffer"):
            if p_valid is None:
                p_valid = reg.pred('p_valid')
            reg.u32('hist_index')
            camera.get_index(hist_index, x, y, p_valid)
            reg.u32('hist_bin_addr')
            op.mov.u32(hist_bin_addr, g_hist_bins)
            op.mad.lo.u32(hist_bin_addr, hist_index, 16, hist_bin_addr)

            reg.f32('r g b a norm_time')
            cp.get(cpA, norm_time, 'cp.norm_time')
            palette.look_up(r, g, b, a, color, norm_time)
            # TODO: look up, scale by xform visibility
            op.red.add.f32(addr(hist_bin_addr), r)
            op.red.add.f32(addr(hist_bin_addr,4), g)
            op.red.add.f32(addr(hist_bin_addr,8), b)
            op.red.add.f32(addr(hist_bin_addr,12), a)


    def device_init(self, ctx):
        hist_bins_dp, hist_bins_l = ctx.mod.get_global('g_hist_bins')
        cuda.memset_d32(hist_bins_dp, 0, hist_bins_l/4)

    @instmethod
    def get_bins(self, ctx, features):
        hist_bins_dp, hist_bins_l = ctx.mod.get_global('g_hist_bins')
        return cuda.from_device(hist_bins_dp,
                (features.hist_height, features.hist_stride, 4),
                dtype=np.float32)

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
            op.mul.f32(dst_reg, dst_reg, '0f2F800000') # 1./(1<<32)

    @ptx_func
    def next_f32_11(self, dst_reg):
        with block('Load random float [-1,1) into ' + dst_reg.name):
            reg.u32('mwc_to_float')
            self._next()
            op.cvt.rn.f32.s32(dst_reg, mwc_st)
            op.mul.f32(dst_reg, dst_reg, '0f30000000') # 1./(1<<31)

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

