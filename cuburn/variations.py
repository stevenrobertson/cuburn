from cuburn.ptx import PTXFragment, ptx_func

class Variations(PTXFragment):
    """
    You know it.
    """
    # TODO: precalc

    shortname = "variations"

    def __init__(self):
        self.xform_idx = None

    names = [ "linear", "sinusoidal", "spherical", "swirl", "horseshoe",
        "polar", "handkerchief", "heart", "disc", "spiral", "hyperbolic",
        "diamond", "ex", "julia", "bent", "waves", "fisheye", "popcorn",
        "exponential", "power", "cosine", "rings", "fan", "blob", "pdj",
        "fan2", "rings2", "eyefish", "bubble", "cylinder", "perspective",
        "noise", "julian", "juliascope", "blur", "gaussian_blur",
        "radial_blur", "pie", "ngon", "curl", "rectangles", "arch", "tangent",
        "square", "rays", "blade", "secant2", "twintrian", "cross", "disc2",
        "super_shape", "flower", "conic", "parabola", "bent2", "bipolar",
        "boarders", "butterfly", "cell", "cpow", "curve", "edisc", "elliptic",
        "escher", "foci", "lazysusan", "loonie", "pre_blur", "modulus",
        "oscilloscope", "polar2", "popcorn2", "scry", "separation", "split",
        "splits", "stripes", "wedge", "wedge_julia", "wedge_sph", "whorl",
        "waves2", "exp", "log", "sin", "cos", "tan", "sec", "csc", "cot",
        "sinh", "cosh", "tanh", "sech", "csch", "coth", "auger", "flux", ]

    @ptx_func
    def xfg(self, dst, expr):
        """
        Convenience wrapper around cp.get which loads the given property from
        the current CP and XF.
        """
        # xform_idx is set by apply_xform on the current instance, but the
        # expression will be evaluated using each CP in stream packing.
        cp.get(cpA, dst, 'cp.xforms[%d].%s' % (self.xform_idx, expr))

    @ptx_func
    def xfg_v2(self, dst1, expr1, dst2, expr2):
        cp.get_v2(cpA, dst1, 'cp.xforms[%d].%s' % (self.xform_idx, expr1),
                       dst2, 'cp.xforms[%d].%s' % (self.xform_idx, expr2))

    @ptx_func
    def xfg_v4(self, d1, e1, d2, e2, d3, e3, d4, e4):
        cp.get_v4(cpA, d1, 'cp.xforms[%d].%s' % (self.xform_idx, e1),
                       d2, 'cp.xforms[%d].%s' % (self.xform_idx, e2),
                       d3, 'cp.xforms[%d].%s' % (self.xform_idx, e3),
                       d4, 'cp.xforms[%d].%s' % (self.xform_idx, e4))

    @ptx_func
    def apply_xform(self, xo, yo, co, xi, yi, ci, xform_idx):
        """
        Apply a transform.

        This function makes a copy of the input variables, so it's safe to use
        the same registers for input and output.
        """
        with block("Apply xform %d" % xform_idx):
            self.xform_idx = xform_idx

            with block('Modify color'):
                reg.f32('c_speed c_new')
                cp.get_v2(cpA,
                    c_speed, '(1.0 - cp.xforms[%d].color_speed)' % xform_idx,
                    c_new,   'cp.xforms[%d].color * cp.xforms[%d].color_speed' %
                             (xform_idx, xform_idx))
                op.fma.rn.ftz.f32(co, ci, c_speed, c_new)

            reg.f32('xt yt')
            with block("Do affine transformation"):
                # TODO: verify that this is the best performance (register
                # usage vs number of loads)
                reg.f32('c00 c10 c20 c01 c11 c21')
                self.xfg_v4(c00, 'coefs[0][0]', c01, 'coefs[0][1]',
                            c20, 'coefs[2][0]', c21, 'coefs[2][1]')
                op.fma.rn.ftz.f32(xt, c00, xi, c20)
                op.fma.rn.ftz.f32(yt, c01, xi, c21)
                self.xfg_v2(c10, 'coefs[1][0]', c11, 'coefs[1][1]')
                op.fma.rn.ftz.f32(xt, c10, yi, xt)
                op.fma.rn.ftz.f32(yt, c11, yi, yt)

            op.mov.f32(xo, '0.0')
            op.mov.f32(yo, '0.0')

            for var_name in sorted(features.xforms[xform_idx].vars):
                func = getattr(self, var_name, None)
                if not func:
                    raise NotImplementedError(
                            "Haven't implemented %s yet" % var_name)
                with block('%s variation' % var_name):
                    reg.f32('wgt')
                    self.xfg(wgt, var_name)
                    func(xo, yo, xt, yt, wgt)

            if features.xforms[xform_idx].has_post:
                with block("Affine post-transformation"):
                    op.mov.f32(xt, xo)
                    op.mov.f32(yt, yo)
                    reg.f32('c00 c10 c20 c01 c11 c21')
                    self.xfg_v4(c00, 'post[0][0]', c01, 'post[0][1]',
                                c20, 'post[2][0]', c21, 'post[2][1]')
                    op.fma.rn.ftz.f32(xo, c00, xt, c20)
                    op.fma.rn.ftz.f32(yo, c01, xt, c21)
                    self.xfg_v2(c10, 'post[1][0]', c11, 'post[1][1]')
                    op.fma.rn.ftz.f32(xo, c10, yt, xo)
                    op.fma.rn.ftz.f32(yo, c11, yt, yo)

    @ptx_func
    def linear(self, xo, yo, xi, yi, wgt):
        op.fma.rn.ftz.f32(xo, xi, wgt, xo)
        op.fma.rn.ftz.f32(yo, yi, wgt, xo)

    @ptx_func
    def sinusoidal(self, xo, yo, xi, yi, wgt):
        reg.f32('sinval')
        op.sin.approx.ftz.f32(sinval, xi)
        op.fma.rn.ftz.f32(xo, sinval, wgt, xo)
        op.sin.approx.ftz.f32(sinval, yi)
        op.fma.rn.ftz.f32(yo, sinval, wgt, yo)

    @ptx_func
    def spherical(self, xo, yo, xi, yi, wgt):
        reg.f32('r2')
        op.fma.rn.ftz.f32(r2, xi, xi, '1e-9')
        op.fma.rn.ftz.f32(r2, yi, yi, r2)
        op.rcp.approx.f32(r2, r2)
        op.mul.rn.ftz.f32(r2, r2, wgt)
        op.fma.rn.ftz.f32(xo, xi, r2, xo)
        op.fma.rn.ftz.f32(yo, yi, r2, yo)


