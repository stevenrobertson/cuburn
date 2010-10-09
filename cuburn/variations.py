from pyptx import ptx

class Variations(object):
    """
    You know it.
    """
    # TODO: precalc

    shortname = "variations"

    def __init__(self, features):
        self.features = features

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

    def apply_xform(self, entry, cp, x, y, color, xform_idx):
        """
        Apply a transform.

        This function necessarily makes a copy of the input variables, so it's
        safe to use the same registers for input and output.
        """
        e, r, o, m, p, s = entry.locals

        # For use in retrieving properties from the control point datastream
        xfs = lambda stval: 'cp.xforms[%d].%s' % (xform_idx, stval)

        e.comment('Color transformation')
        c_speed, c_val = cp.get.v2.f32('1.0 - %s' % xfs('color_speed'),
                '%s * %s' % (xfs('color'), xfs('color_speed')))
        color = color * c_speed + c_val

        e.comment('Affine transformation')
        c00, c20 = cp.get.v2.f32(xfs('coefs[0][0]'), xfs('coefs[2][0]'))
        xt = x * c00 + c20
        c01, c21 = cp.get.v2.f32(xfs('coefs[0][1]'), xfs('coefs[2][1]'))
        yt = x * c01 + c21
        c10, c11 = cp.get.v2.f32(xfs('coefs[1][0]'), xfs('coefs[1][1]'))
        xt += y * c10
        yt += y * c11

        xo, yo = o.mov.f32(0), o.mov.f32(0)
        for var_name in sorted(self.features.xforms[xform_idx].vars):
            func = getattr(self, var_name, None)
            if not func:
                raise NotImplementedError(
                        "Haven't implemented %s yet" % var_name)
            e.comment('%s variation' % var_name)
            xtemp, ytemp = func(o, xt, yt, cp.get.f32(xfs(var_name)))
            xo += xtemp
            yo += ytemp

        if self.features.xforms[xform_idx].has_post:
            e.comment('Affine post-transformation')
            c00, c20 = cp.get.v2.f32(xfs('post[0][0]'), xfs('post[2][0]'))
            xt = xo * c00 + c20
            c01, c21 = cp.get.v2.f32(xfs('post[0][1]'), xfs('post[2][1]'))
            yt = xo * c01 + c21
            c10, c11 = cp.get.v2.f32(xfs('post[1][0]'), xfs('post[1][1]'))
            xt += yo * c10
            yt += yo * c11
            xo, yo = xt, yt

        self.xform_idx = None
        return xo, yo, color

    def linear(self, o, x, y, wgt):
        return x * wgt, y * wgt

    def sinusoidal(self, o, x, y, wgt):
        return o.sin(x) * wgt, o.sin(y) * wgt

    def spherical(self, o, x, y, wgt):
        rsquared = x * x + y * y
        rrcp = o.rcp(rsquared) * wgt
        return x * wgt, y * wgt

