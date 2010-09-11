"""
Contains the PTX fragments which will drive the device.
"""

import os
import time
import struct

import pycuda.driver as cuda
import numpy as np

from cuburn.ptx import *
from cuburn.variations import Variations

class IterThread(PTXEntryPoint):
    entry_name = 'iter_thread'
    entry_params = []
    maxnreg = 16

    def __init__(self):
        self.cps_uploaded = False

    def deps(self):
        return [MWCRNG, CPDataStream, HistScatter, Variations, Timeouter]

    @ptx_func
    def module_setup(self):
        mem.global_.u32('g_cp_array',
                        cp.stream_size*features.max_ntemporal_samples)
        mem.global_.u32('g_num_cps')
        mem.global_.u32('g_num_cps_started')
        # TODO move into debug statement
        mem.global_.u32('g_num_rounds', ctx.threads)
        mem.global_.u32('g_num_writes', ctx.threads)
        mem.global_.b32('g_whatever', ctx.threads)

    @ptx_func
    def entry(self):
        # For now, we indulge in the luxury of shared memory.
        # Index number of current CP, shared across CTA
        mem.shared.u32('s_cp_idx')

        # Number of samples that have been generated so far in this CTA
        # If this number is negative, we're still fusing points, so this
        # behaves slightly differently (see ``fuse_loop_start``)
        mem.shared.s32('s_num_samples')
        op.st.shared.s32(addr(s_num_samples), -(features.num_fuse_samples+1))

        mem.shared.f32('s_xf_sel', ctx.warps_per_cta)

        std.store_per_thread(g_whatever, 1234)

        # TODO: temporary, for testing
        mem.local.u32('l_num_rounds')
        mem.local.u32('l_num_writes')
        op.st.local.u32(addr(l_num_rounds), 0)
        op.st.local.u32(addr(l_num_writes), 0)

        mem.local.f32('l_consec')
        op.st.local.f32(addr(l_consec), 0.)

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

        with block("If done fusing, reset the sample count now"):
            reg.pred("p_done_fusing")
            reg.s32('num_samples')
            op.ld.shared.s32(num_samples, addr(s_num_samples))
            op.setp.gt.s32(p_done_fusing, num_samples, 0)
            op.st.shared.s32(addr(s_num_samples), 0, ifp=p_done_fusing)

        comment("Load the CP index in all threads")
        op.bar.sync(0)
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

        label('iter_loop_choose_xform')
        with block("Choose the xform for each warp"):
            comment("On subsequent runs, only warp 0 will hit this code")
            reg.u32('x_addr x_offset')
            reg.f32('xf_sel')
            op.mov.u32(x_addr, s_xf_sel)
            op.mov.u32(x_offset, '%tid.x')
            op.and_.b32(x_offset, x_offset, ctx.warps_per_cta-1)
            op.mad.lo.u32(x_addr, x_offset, 4, x_addr)
            mwc.next_f32_01(xf_sel)
            op.st.volatile.shared.f32(addr(x_addr), xf_sel)

        label('iter_loop_start')

        timeout.check_time(10)

        # TODO: diagram and fix syncing (can this be automated?)
        #op.bar.sync(1)

        with block():
            reg.u32('num_rounds')
            reg.pred('overload')
            op.ld.local.u32(num_rounds, addr(l_num_rounds))
            op.add.u32(num_rounds, num_rounds, 1)
            op.st.local.u32(addr(l_num_rounds), num_rounds)



        with block("Select an xform"):
            reg.f32('xf_sel')
            reg.u32('warp_offset xf_sel_addr')
            op.mov.u32(warp_offset, '%tid.x')
            op.mov.u32(xf_sel_addr, s_xf_sel)
            op.shr.u32(warp_offset, warp_offset, 5)
            op.mad.lo.u32(xf_sel_addr, warp_offset, 4, xf_sel_addr)
            op.ld.volatile.shared.f32(xf_sel, addr(xf_sel_addr))

            reg.f32('xf_density')
            reg.pred('xf_jump')
            for xf in features.xforms:
                cp.get(cpA, xf_density, 'cp.xforms[%d].cweight' % xf.id)
                op.setp.le.f32(xf_jump, xf_sel, xf_density)
                op.bra('XFORM_%d' % xf.id, ifp=xf_jump)
            std.asrt("Reached end of xforms without choosing one")

            for xf in features.xforms:
                label('XFORM_%d' % xf.id)
                variations.apply_xform(x_coord, y_coord, color_coord,
                                       x_coord, y_coord, color_coord, xf.id)
                op.bra.uni("xform_done")



        label("xform_done")
        with block("Test if we're still in FUSE"):
            reg.s32('num_samples')
            reg.pred('p_in_fuse')
            op.ld.shared.s32(num_samples, addr(s_num_samples))
            op.setp.lt.s32(p_in_fuse, num_samples, 0)
            op.bra.uni(fuse_loop_start, ifp=p_in_fuse)

        reg.pred('p_point_is_valid')
        with block("Write the result"):
            hist.scatter(x_coord, y_coord, color_coord, 0, p_point_is_valid)
            with block():
                reg.u32('num_writes')
                op.ld.local.u32(num_writes, addr(l_num_writes))
                op.add.u32(num_writes, num_writes, 1, ifp=p_point_is_valid)
                op.st.local.u32(addr(l_num_writes), num_writes)

        with block("If the result was invalid, handle badvals"):
            reg.f32('consec')
            reg.pred('need_new_point')
            op.ld.local.f32(consec, addr(l_consec))
            op.mov.f32(consec, 0., ifp=p_point_is_valid)
            op.add.f32(consec, consec, 1., ifnotp=p_point_is_valid)
            op.setp.ge.f32(need_new_point, consec, 5.)
            op.bra('badval_done', ifnotp=need_new_point)
            mwc.next_f32_11(x_coord)
            mwc.next_f32_11(y_coord)
            mwc.next_f32_01(color_coord)
            op.mov.f32(consec, 0.)
            label('badval_done')
            op.st.local.f32(addr(l_consec), consec)

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

        with block("If first warp, pick new thread offset"):
            reg.u32('warpid')
            reg.pred('first_warp')
            op.mov.u32(warpid, '%tid.x')
            op.shr.b32(warpid, warpid, 5)
            op.setp.eq.u32(first_warp, warpid, 0)
            #std.asrt("Looks like we're not the first warp", notp=first_warp,
                    #ret=True)
            op.bra.uni(iter_loop_choose_xform, ifp=first_warp)
        op.bra.uni(iter_loop_start)

        label('all_cps_done')
        # TODO this is for testing, move it to a debug statement
        with block():
            reg.u32('num_rounds num_writes')
            op.ld.local.u32(num_rounds, addr(l_num_rounds))
            op.ld.local.u32(num_writes, addr(l_num_writes))
            std.store_per_thread(g_num_rounds, num_rounds,
                                 g_num_writes, num_writes)

    @instmethod
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
        shape = (ctx.grid[0], ctx.block[0]/32, 32)

        def print_thing(s, a):
            print '%s:' % s
            for i, r in enumerate(a):
                for j in range(0,len(r),8):
                    print '%2d\t%s' % (i,
                        '\t'.join(['%g '%np.mean(r[k]) for k in range(j,j+8)]))

        num_rounds_dp, num_rounds_l = ctx.mod.get_global('g_num_rounds')
        num_writes_dp, num_writes_l = ctx.mod.get_global('g_num_writes')
        whatever_dp, whatever_l = ctx.mod.get_global('g_whatever')
        rounds = cuda.from_device(num_rounds_dp, shape, np.int32)
        writes = cuda.from_device(num_writes_dp, shape, np.int32)
        whatever = cuda.from_device(whatever_dp, shape, np.int32)
        print_thing("Rounds", rounds)
        print_thing("Writes", writes)
        print_thing("Whatever", whatever)

        print np.sum(rounds)

        dp, l = ctx.mod.get_global('g_num_cps_started')
        cps_started = cuda.from_device(dp, 1, np.uint32)
        print "CPs started:", cps_started

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

    def call_setup(self, ctx):
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
            # TODO: Make this more performant
            reg.f32('gval')
            for i, val in enumerate([r, g, b, a]):
                #op.red.add.f32(addr(hist_bin_addr,4*i), val)
                op.ld.f32(gval,addr(hist_bin_addr,4*i))
                op.add.f32(gval, gval, val)
                op.st.f32(addr(hist_bin_addr,4*i),gval)


    def call_setup(self, ctx):
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
        self.threads_ready = 0
        if not os.path.isfile('primes.bin'):
            raise EnvironmentError('primes.bin not found')

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

    @instmethod
    def seed(self, ctx, rand=np.random):
        """
        Seed the random number generators with values taken from a
        ``np.random`` instance.
        """
        # Load raw big-endian u32 multipliers from primes.bin.
        with open('primes.bin') as primefp:
            dt = np.dtype(np.uint32).newbyteorder('B')
            mults = np.frombuffer(primefp.read(), dtype=dt)
        stream = cuda.Stream()
        # Randomness in choosing multipliers is good, but larger multipliers
        # have longer periods, which is also good. This is a compromise.
        mults = np.array(mults[:ctx.threads*4])
        rand.shuffle(mults)
        # Copy multipliers and seeds to the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        cuda.memcpy_htod_async(multdp, mults.tostring()[:multl])
        # Intentionally excludes both 0 and (2^32-1), as they can lead to
        # degenerate sequences of period 0
        states = np.array(rand.randint(1, 0xffffffff, size=2*ctx.threads),
                          dtype=np.uint32)
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        cuda.memcpy_htod_async(statedp, states.tostring())
        self.threads_ready = ctx.threads

    def call_setup(self, ctx):
        if self.threads_ready < ctx.threads:
            self.seed(ctx)

    def tests(self):
        return [MWCRNGTest, MWCRNGFloatsTest]

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

    def call_setup(self, ctx):
        # Get current multipliers and seeds from the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        self.mults = cuda.from_device(multdp, ctx.threads, np.uint32)
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        self.fullstates = cuda.from_device(statedp, ctx.threads, np.uint64)
        self.sums = np.zeros(ctx.threads, np.uint64)

        print "Running %d states forward %d rounds" % \
              (len(self.mults), self.rounds)
        ctime = time.time()
        for i in range(self.rounds):
            states = self.fullstates & 0xffffffff
            carries = self.fullstates >> 32
            self.fullstates = self.mults * states + carries
            self.sums += self.fullstates & 0xffffffff
        ctime = time.time() - ctime
        print "Done on host, took %g seconds" % ctime

    def call_teardown(self, ctx):
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        statedp, statel = ctx.mod.get_global('mwc_rng_state')

        dfullstates = cuda.from_device(statedp, ctx.threads, np.uint64)
        if not (dfullstates == self.fullstates).all():
            print "State discrepancy"
            print dfullstates
            print self.fullstates
            raise PTXTestFailure("MWC RNG state discrepancy")

        sumdp, suml = ctx.mod.get_global('mwc_rng_test_sums')
        dsums = cuda.from_device(sumdp, ctx.threads, np.uint64)
        if not (dsums == self.sums).all():
            print "Sum discrepancy"
            print dsums
            print self.sums
            raise PTXTestFailure("MWC RNG sum discrepancy")

class MWCRNGFloatsTest(PTXTest):
    """
    Note this only tests that the distributions are in the correct range, *not*
    that they have good random properties. MWC is a suitable algorithm, but
    implementation bugs may still lead to poor performance.
    """
    rounds = 1024
    entry_name = 'MWC_RNG_floats_test'

    def deps(self):
        return [MWCRNG]

    @ptx_func
    def module_setup(self):
        mem.global_.f32('mwc_rng_float_01_test_sums', ctx.threads)
        mem.global_.f32('mwc_rng_float_01_test_mins', ctx.threads)
        mem.global_.f32('mwc_rng_float_01_test_maxs', ctx.threads)
        mem.global_.f32('mwc_rng_float_11_test_sums', ctx.threads)
        mem.global_.f32('mwc_rng_float_11_test_mins', ctx.threads)
        mem.global_.f32('mwc_rng_float_11_test_maxs', ctx.threads)

    @ptx_func
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

    @ptx_func
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
            dp, l = ctx.mod.get_global(
                    'mwc_rng_float_%s_test_%s' % (fkind, rkind))
            vals = cuda.from_device(dp, ctx.threads, np.float32)
            avg = np.mean(vals)
            if np.abs(avg - exp) > tol:
                raise PTXTestFailure("%s %s %g too far from %g" %
                                     (fkind, rkind, avg, exp))
            if lim is None: continue
            if lim([lim(vals), exp]) != exp:
                raise PTXTestFailure("%s %s %g violates hard limit %g" %
                                     (fkind, rkind, lim(vals), exp))

class CPDataStream(DataStream):
    """DataStream which stores the control points."""
    shortname = 'cp'

class Timeouter(PTXFragment):
    """Time-out infinite loops so that data can still be retrieved."""
    shortname = 'timeout'

    @ptx_func
    def entry_setup(self):
        mem.shared.u64('s_timeouter_start_time')
        with block("Load start time for this block"):
            reg.u64('now')
            op.mov.u64(now, '%clock64')
            op.st.shared.u64(addr(s_timeouter_start_time), now)

    @ptx_func
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


