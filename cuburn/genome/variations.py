from spectypes import spline, scalespline
import numpy as np
# Pre-instantiated default splines. Used a *lot*.
s, ss, sz = spline(), scalespline(), scalespline(min=0)

__all__ = ["var_names", "var_params"]

# A map from flam3 variation numbers to variation names. Some variations may
# not be included in this list if they don't yet exist in flam3.
var_names = {}

# A map from variation names to a dict of parameter types, suitable for
# inclusion in the genome schema.
var_params = {}

def var(num, name, **params):
    if num is not None:
        var_names[num] = name
    # Mark as a variation parameter spline. This can be handled in various
    # ways by interpolation - usually by copying a value when missing, instead
    # of reading the default value.
    for k, v in params.items():
        params[k] = v._replace(var=True)
    params['weight'] = scalespline(0)
    var_params[name] = params

# TODO: review all parameter splines, possibly programmatically
var(0, 'linear')
var(1, 'sinusoidal')
var(2, 'spherical')
var(3, 'swirl')
var(4, 'horseshoe')
var(5, 'polar')
var(6, 'handkerchief')
var(7, 'heart')
var(8, 'disc')
var(9, 'spiral')
var(10, 'hyperbolic')
var(11, 'diamond')
var(12, 'ex')
var(13, 'julia')
var(14, 'bent')
var(15, 'waves')
var(16, 'fisheye')
var(17, 'popcorn')
var(18, 'exponential')
var(19, 'power')
var(20, 'cosine')
var(21, 'rings')
var(22, 'fan')
var(23, 'blob', low=ss, high=ss, waves=ss)
var(24, 'pdj', a=s, b=s, c=s, d=s)
var(25, 'fan2', x=s, y=s)
var(26, 'rings2', val=s)
var(27, 'eyefish')
var(28, 'bubble')
var(29, 'cylinder')
var(30, 'perspective', angle=s, dist=s) # TODO: period
var(31, 'noise')
var(32, 'julian', power=ss, dist=ss)
var(33, 'juliascope', power=ss, dist=ss)
var(34, 'blur')
var(35, 'gaussian_blur')
var(36, 'radial_blur', angle=spline(period=4))
var(37, 'pie', slices=spline(6, 1), rotation=s, thickness=spline(0.5, 0, 1))
var(38, 'ngon', sides=spline(5), power=spline(3),
                circle=spline(1), corners=spline(2))
var(39, 'curl', c1=spline(1), c2=s) # TODO: not identity?

var(40, 'rectangles', x=s, y=s)
var(41, 'arch')
var(42, 'tangent')
var(43, 'square')
var(44, 'rays')
var(45, 'blade')
var(46, 'secant2')
var(48, 'cross')
var(49, 'disc2', rot=s, twist=s)
var(50, 'super_shape', rnd=s, m=s, n1=ss, n2=spline(1), n3=spline(1), holes=s)
var(51, 'flower', holes=s, petals=s)
var(52, 'conic', holes=s, eccentricity=spline(1))
var(53, 'parabola', height=ss, width=ss)
var(54, 'bent2', x=ss, y=ss)
var(55, 'bipolar', shift=s)
var(56, 'boarders')
var(57, 'butterfly')
var(58, 'cell', size=ss)
var(59, 'cpow', r=ss, i=s, power=ss)
var(60, 'curve', xamp=s, yamp=s, xlength=ss, ylength=ss)
var(61, 'edisc')
var(62, 'elliptic')
var(63, 'escher', beta=spline(period=2*np.pi))
var(64, 'foci')
var(65, 'lazysusan', x=s, y=s, twist=s, space=s, spin=s)
var(66, 'loonie')
var(67, 'pre_blur')
var(68, 'modulus', x=s, y=s)
var(69, 'oscope', separation=spline(1), frequency=scalespline(np.pi),
                  amplitude=ss, damping=s)
var(70, 'polar2')
var(71, 'popcorn2', x=s, y=s, c=s)
var(72, 'scry')
var(73, 'separation', x=s, xinside=s, y=s, yinside=s)
var(74, 'split', xsize=s, ysize=s)
var(75, 'splits', x=s, y=s)
var(76, 'stripes', space=s, warp=s)
var(77, 'wedge', angle=s, hole=s, count=ss, swirl=s)
var(80, 'whorl', inside=s, outside=s)
var(81, 'waves2', scalex=ss, scaley=ss,
                  freqx=scalespline(np.pi), freqy=scalespline(np.pi))
var(82, 'exp')
var(83, 'log')
var(84, 'sin')
var(85, 'cos')
var(86, 'tan')
var(87, 'sec')
var(88, 'csc')
var(89, 'cot')
var(90, 'sinh')
var(91, 'cosh')
var(92, 'tanh')
var(93, 'sech')
var(94, 'csch')
var(95, 'coth')
var(97, 'flux', spread=s)
var(98, 'mobius', re_a=s, im_a=s, re_b=s, im_b=s,
                  re_c=s, im_c=s, re_d=s, im_d=s)
