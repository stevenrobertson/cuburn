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

    def __init__(self):
        self.cps_uploaded = False

    def deps(self):
        return [MWCRNG, CPDataStream, HistScatter, Variations, ShufflePoints,
                Timeouter]

    @ptx_func
    def module_setup(self):
        mem.global_.u32('g_cp_array',
                        cp.stream_size*features.max_ntemporal_samples)
        mem.global_.u32('g_num_cps')
        mem.global_.u32('g_num_cps_started')
        # TODO move into debug statement
        mem.global_.u32('g_num_rounds', ctx.nthreads)
        mem.global_.u32('g_num_writes', ctx.nthreads)
        mem.global_.b32('g_whatever', ctx.nthreads)

    @ptx_func
    def entry(self):
        # Index number of current CP, shared across CTA
        mem.shared.u32('s_cp_idx')

        # Number of samples that have been generated so far in this CTA
        # If this number is negative, we're still fusing points, so this
        # behaves slightly differently (see ``fuse_loop_start``)
        # TODO: replace (or at least simplify) this logic
        mem.shared.s32('s_num_samples')
        mem.shared.f32('s_xf_sel', ctx.warps_per_cta)

        # TODO: temporary, for testing
        mem.local.u32('l_num_rounds')
        mem.local.u32('l_num_writes')
        op.st.local.u32(addr(l_num_rounds), 0)
        op.st.local.u32(addr(l_num_writes), 0)

        reg.f32('x y color consec_bad')
        mwc.next_f32_11(x)
        mwc.next_f32_11(y)
        mwc.next_f32_01(color)
        op.mov.f32(consec_bad, float(-features.fuse))

        comment("Ensure all init is done")
        op.bar.sync(0)



        label('cp_loop_start')
        reg.u32('cp_idx cpA')
        with block("Claim a CP"):
            std.set_is_first_thread(reg.pred('p_is_first'))
            op.atom.add.u32(cp_idx, addr(g_num_cps_started), 1, ifp=p_is_first)
            op.st.shared.u32(addr(s_cp_idx), cp_idx, ifp=p_is_first)
            op.st.shared.s32(addr(s_num_samples), 0)

        comment("Load the CP index in all threads")
        op.bar.sync(0)
        op.ld.shared.u32(cp_idx, addr(s_cp_idx))

        with block("Check to see if this CP is valid (if not, we're done)"):
            reg.u32('num_cps')
            reg.pred('p_last_cp')
            op.ldu.u32(num_cps, addr(g_num_cps))
            op.setp.ge.u32(p_last_cp, cp_idx, num_cps)
            op.bra('all_cps_done', ifp=p_last_cp)

        with block('Load CP address'):
            op.mov.u32(cpA, g_cp_array)
            op.mad.lo.u32(cpA, cp_idx, cp.stream_size, cpA)



        label('iter_loop_choose_xform')
        with block("Choose the xform for each warp"):
            timeout.check_time(5)
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

        #timeout.check_time(10)

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
                variations.apply_xform(x, y, color, x, y, color, xf.id)
                op.bra("xform_done")

        label("xform_done")

        reg.pred('p_valid_pt')
        with block("Write the result"):
            reg.u32('hist_index')
            camera.get_index(hist_index, x, y, p_valid_pt)
            comment('if consec_bad < 0, point is fusing; treat as invalid')
            op.setp.and_.ge.f32(p_valid_pt, consec_bad, 0., p_valid_pt)
            # TODO: save and pass correct color value here
            hist.scatter(hist_index, color, 0, p_valid_pt, 'ldst')
            with block():
                reg.u32('num_writes')
                op.ld.local.u32(num_writes, addr(l_num_writes))
                op.add.u32(num_writes, num_writes, 1, ifp=p_valid_pt)
                op.st.local.u32(addr(l_num_writes), num_writes)

        with block("If the result was invalid, handle badvals"):
            reg.pred('need_new_point')
            op.add.f32(consec_bad, consec_bad, 1., ifnotp=p_valid_pt)
            op.setp.ge.f32(need_new_point, consec_bad, float(features.max_bad))
            op.bra('badval_done', ifnotp=need_new_point)

            comment('If consec_bad > 5, pick a new random point')
            mwc.next_f32_11(x)
            mwc.next_f32_11(y)
            mwc.next_f32_01(color)
            op.mov.f32(consec_bad, float(-features.fuse))
            label('badval_done')

        with block("Increment number of samples by number of good values"):
            reg.b32('good_samples laneid')
            reg.pred('p_is_first')
            op.vote.ballot.b32(good_samples, p_valid_pt)
            op.popc.b32(good_samples, good_samples)
            op.mov.u32(laneid, '%laneid')
            op.setp.eq.u32(p_is_first, laneid, 0)
            op.red.shared.add.s32(addr(s_num_samples), good_samples,
                                  ifp=p_is_first)

        with block("Check to see if we're done with this CP"):
            reg.pred('p_cp_done')
            reg.s32('num_samples num_samples_needed')
            comment('Sync before making decision to prevent divergence')
            op.bar.sync(3)
            op.ld.shared.s32(num_samples, addr(s_num_samples))
            cp.get(cpA, num_samples_needed, 'cp.nsamples')
            op.setp.ge.s32(p_cp_done, num_samples, num_samples_needed)
            op.bra.uni(cp_loop_start, ifp=p_cp_done)

        comment('Shuffle points between threads')
        shuf.shuffle(x, y, color, consec_bad)

        with block("If in first warp, pick new offset"):
            reg.u32('tid')
            reg.pred('first_warp')
            op.mov.u32(tid, '%tid.x')
            assert ctx.warps_per_cta <= 32, \
                   "Special-case for CTAs with >1024 threads not implemented"
            op.setp.lo.u32(first_warp, tid, 32)
            op.bra(iter_loop_choose_xform, ifp=first_warp)
        op.bra(iter_loop_start)

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
    def look_up(self, r, g, b, a, color, norm_time, ifp):
        """
        Look up the values of ``r, g, b, a`` corresponding to ``color_coord``
        at the CP indexed in ``timestamp_idx``. Note that both ``color_coord``
        and ``timestamp_idx`` should be [0,1]-normalized floats.
        """
        op.tex._2d.v4.f32.f32(vec(r, g, b, a),
                addr([t_palette, ', ',  vec(norm_time, color)]), ifp=ifp)
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
        self.pal = pal

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
        comment("Target to ensure fake local values get written")
        mem.global_.f32('g_hist_dummy')

    @ptx_func
    def entry_setup(self):
        comment("Fake bins for fake scatter")
        mem.local.f32('l_scatter_fake_adr')
        mem.local.f32('l_scatter_fake_alpha')

    @ptx_func
    def entry_teardown(self):
        with block("Store fake histogram bins to dummy global"):
            reg.b32('hist_dummy')
            op.ld.local.b32(hist_dummy, addr(l_scatter_fake_adr))
            op.st.volatile.b32(addr(g_hist_dummy), hist_dummy)
            op.ld.local.b32(hist_dummy, addr(l_scatter_fake_alpha))
            op.st.volatile.b32(addr(g_hist_dummy), hist_dummy)

    @ptx_func
    def scatter(self, hist_index, color, xf_idx, p_valid, type='ldst'):
        """
        Scatter the given point directly to the histogram bins. I think this
        technique has the worst performance of all of 'em. Accesses ``cpA``
        directly.
        """
        with block("Scatter directly to buffer"):
            reg.u32('hist_bin_addr')
            op.mov.u32(hist_bin_addr, g_hist_bins)
            op.mad.lo.u32(hist_bin_addr, hist_index, 16, hist_bin_addr)

            if type == 'fake_notex':
                op.st.local.u32(addr(l_scatter_fake_adr), hist_bin_addr)
                op.st.local.f32(addr(l_scatter_fake_alpha), color)
                return

            reg.f32('r g b a norm_time')
            cp.get(cpA, norm_time, 'cp.norm_time')
            palette.look_up(r, g, b, a, color, norm_time, ifp=p_valid)
            # TODO: look up, scale by xform visibility
            # TODO: Make this more performant
            if type == 'ldst':
                reg.f32('gr gg gb ga')
                op.ld.v4.f32(vec(gr, gg, gb, ga), addr(hist_bin_addr),
                             ifp=p_valid)
                op.add.f32(gr, gr, r)
                op.add.f32(gg, gg, g)
                op.add.f32(gb, gb, b)
                op.add.f32(ga, ga, a)
                op.st.v4.f32(addr(hist_bin_addr), vec(gr, gg, gb, ga),
                             ifp=p_valid)
            elif type == 'red':
                for i, val in enumerate([r, g, b, a]):
                    op.red.add.f32(addr(hist_bin_addr,4*i), val, ifp=p_valid)
            elif type == 'fake':
                op.st.local.u32(addr(l_scatter_fake_adr), hist_bin_addr)
                op.st.local.f32(addr(l_scatter_fake_alpha), a)

    def call_setup(self, ctx):
        hist_bins_dp, hist_bins_l = ctx.mod.get_global('g_hist_bins')
        cuda.memset_d32(hist_bins_dp, 0, hist_bins_l/4)

    @instmethod
    def get_bins(self, ctx, features):
        hist_bins_dp, hist_bins_l = ctx.mod.get_global('g_hist_bins')
        return cuda.from_device(hist_bins_dp,
                (features.hist_height, features.hist_stride, 4),
                dtype=np.float32)


class ShufflePoints(PTXFragment):
    """
    Shuffle points in shared memory. See helpers/shuf.py for details.
    """
    shortname = "shuf"

    @ptx_func
    def module_setup(self):
        # TODO: if needed, merge this shared memory block with others
        mem.shared.f32('s_shuf_data', ctx.threads_per_cta)

    @ptx_func
    def shuffle(self, *args, **kwargs):
        """
        Shuffle the data from each register in args across threads. Keyword
        argument ``bar`` specifies which barrier to use (default is 2).
        """
        bar = kwargs.pop('bar', 2)
        with block("Shuffle across threads"):
            reg.u32('shuf_read shuf_write')
            with block("Calculate read and write offsets"):
                reg.u32('shuf_off shuf_laneid')
                op.mov.u32(shuf_off, '%tid.x')
                op.mov.u32(shuf_write, s_shuf_data)
                op.mad.lo.u32(shuf_write, shuf_off, 4, shuf_write)
                op.mov.u32(shuf_laneid, '%laneid')
                op.mad.lo.u32(shuf_off, shuf_laneid, 32, shuf_off)
                op.and_.b32(shuf_off, shuf_off, ctx.threads_per_cta - 1)
                op.mov.u32(shuf_read, s_shuf_data)
                op.mad.lo.u32(shuf_read, shuf_off, 4, shuf_read)
            for var in args:
                op.bar.sync(bar)
                op.st.shared.b32(addr(shuf_write), var)
                op.bar.sync(bar)
                op.ld.shared.b32(var, addr(shuf_read))

class MWCRNG(PTXFragment):
    shortname = "mwc"

    def __init__(self):
        self.threads_ready = 0
        if not os.path.isfile('primes.bin'):
            raise EnvironmentError('primes.bin not found')

    @ptx_func
    def module_setup(self):
        mem.global_.u32('mwc_rng_mults', ctx.nthreads)
        mem.global_.u64('mwc_rng_state', ctx.nthreads)

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
        mults = np.array(mults[:ctx.nthreads*4])
        rand.shuffle(mults)
        # Copy multipliers and seeds to the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        cuda.memcpy_htod(multdp, mults.tostring()[:multl])
        # Intentionally excludes both 0 and (2^32-1), as they can lead to
        # degenerate sequences of period 0
        states = np.array(rand.randint(1, 0xffffffff, size=2*ctx.nthreads),
                          dtype=np.uint32)
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        cuda.memcpy_htod(statedp, states.tostring())
        self.threads_ready = ctx.nthreads

    def call_setup(self, ctx):
        if self.threads_ready < ctx.nthreads:
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
        mem.global_.u64('mwc_rng_test_sums', ctx.nthreads)

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
        self.mults = ctx.get_per_thread('mwc_rng_mults', np.uint32)
        self.fullstates = ctx.get_per_thread('mwc_rng_state', np.uint64)
        self.sums = np.zeros(ctx.nthreads, np.uint64)

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
        dfullstates = ctx.get_per_thread('mwc_rng_state', np.uint64)
        if not (dfullstates == self.fullstates).all():
            print "State discrepancy"
            print dfullstates
            print self.fullstates
            raise PTXTestFailure("MWC RNG state discrepancy")


        dsums = ctx.get_per_thread('mwc_rng_test_sums', np.uint64)
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
        mem.global_.f32('mwc_rng_float_01_test_sums', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_01_test_mins', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_01_test_maxs', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_11_test_sums', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_11_test_mins', ctx.nthreads)
        mem.global_.f32('mwc_rng_float_11_test_maxs', ctx.nthreads)

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


