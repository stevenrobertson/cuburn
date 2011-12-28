#!/usr/bin/env python2

import base64
import warnings
import xml.parsers.expat
import numpy as np

from code.variations import var_code, var_params
from code.util import crep

class SplEval(object):
    _mat = np.matrix([[1.,-2, 1, 0], [2,-3, 0, 1],
                      [1,-1, 0, 0], [-2, 3, 0, 0]])
    _deriv = np.matrix(np.diag([3,2,1], 1))

    def __init__(self, knots, v0=None, v1=None):
        self.knots = self.normalize(knots, v0, v1)

    @staticmethod
    def normalize(knots, v0=None, v1=None):
        if isinstance(knots, (int, float)):
            knots = [0.0, knots, 1.0, knots]
        elif not np.all(np.diff(np.float32(np.asarray(knots))[::2]) > 0):
            raise ValueError("Spline times are non-monotonic. (Use "
                    "nextafterf()-spaced times to anchor tangents.)")

        # If stabilizing knots are missing before or after the edges of the
        # [0,1] interval, add them.
        if knots[0] >= 0:
            if v0 is None:
                v0 = (knots[3] - knots[1]) / float(knots[2] - knots[0])
            knots = [-2, knots[3] - (knots[2] + 2) * v0] + knots
        if knots[-2] <= 1:
            if v1 is None:
                v1 = (knots[-1] - knots[-3]) / float(knots[-2] - knots[-4])
            knots.extend([3, knots[-3] + (3 - knots[-4]) * v1])

        knotarray = np.zeros((2, len(knots)/2))
        knotarray.T.flat[:] = knots
        return knotarray

    def find_knots(self, itime):
        idx = np.searchsorted(self.knots[0], itime) - 2
        idx = max(0, min(idx, len(self.knots[0]) - 4))

        times = self.knots[0][idx:idx+4]
        vals = self.knots[1][idx:idx+4]
        # Normalize to [0,1]
        t = itime - times[1]
        times = times - times[1]
        scale = 1 / times[2]
        t = t * scale
        times = times * scale
        return times, vals, t, scale

    def __call__(self, itime, deriv=0):
        times, vals, t, scale = self.find_knots(itime)

        m1 = (vals[2] - vals[0]) / (1.0 - times[0])
        m2 = (vals[3] - vals[1]) / times[3]

        mat = self._mat
        if deriv:
            mat = mat * (scale * self._deriv) ** deriv
        val = [m1, vals[1], m2, vals[2]] * mat * np.array([[t**3, t**2, t, 1]]).T
        return val[0,0]

    def _plt(self, name='SplEval', fig=111, show=True):
        import matplotlib.pyplot as plt
        x = np.linspace(-0.0, 1.0, 500)
        r = x[1] - x[0]
        plt.figure(fig)
        plt.title(name)
        plt.plot(x,map(self,x),x,[self(i,1) for i in x],'--',
                 self.knots[0],self.knots[1],'x')
        plt.xlim(0.0, 1.0)
        if show:
            plt.show()

    def __str__(self):
        return '[%g:%g]' % (self(0), self(1))
    def __repr__(self):
        return '<interp [%g:%g]>' % (self(0), self(1))

    @property
    def knotlist(self):
        # TODO: scale error constants proportional to RMS?
        # If everything is constant, return a constant
        if np.std(self.knots[1]) < 1e-6:
            return self.knots[1][0]
        # If constant slope, omit the end knots
        slopes = np.diff(self.knots[1]) / np.diff(self.knots[0])
        if np.std(slopes) < 1e-6:
            return list(self.knots.T.flat)[2:-2]
        return list(self.knots.T.flat)

    def insert_knot(self, t, v):
        knots = list(sum(sorted(zip(*self.knots) + [(t,v)]), ()))
        self.knots = self.normalize(knots, self(0, 1), self(1, 1))

def palette_decode(datastrs):
    """
    Decode a palette (stored as a list suitable for JSON packing) into a
    palette. Internal palette format is simply as a (256,4) array of [0,1]
    RGBA floats.
    """
    if datastrs[0] != 'rgb8':
        raise NotImplementedError
    raw = base64.b64decode(''.join(datastrs[1:]))
    pal = np.reshape(np.fromstring(raw, np.uint8), (256, 3))
    data = np.ones((256, 4), np.float32)
    data[:,:3] = pal / 255.0
    return data

def palette_encode(data, format='rgb8'):
    """
    Encode an internal-format palette to an external representation.
    """
    if format != 'rgb8':
        raise NotImplementedError
    clamp = np.maximum(0, np.minimum(255, np.round(data[:,:3]*255.0)))
    enc = base64.b64encode(np.uint8(clamp))
    return ['rgb8'] + [enc[i:i+64] for i in range(0, len(enc), 64)]

class _AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError('%s not a dict key' % name)

    @classmethod
    def _wrap(cls, dct):
        for k, v in dct.items():
            if (isinstance(v, (float, int)) or
                    (isinstance(v, list) and isinstance(v[1], (float, int)))):
                dct[k] = SplEval(v)
            elif isinstance(v, dict):
                dct[k] = cls._wrap(cls(v))
        return dct

class Genome(_AttrDict):
    """
    Load a genome description, wrapping all data structures in _AttrDicts,
    converting lists of numbers to splines, and deriving some values. Derived
    values are stored as instance properties, rather than replacing the
    original values, such that JSON-encoding this structure should always
    print a valid genome.
    """
    # For now, we base the Genome class on an _AttrDict, letting its structure
    # be defined implicitly by the way it is used in device code, except for
    # these derived properties.
    def __init__(self, gnm):
        super(Genome, self).__init__(gnm)
        for k, v in self.items():
            if not isinstance(v, dict):
                continue
            v = _AttrDict(v)
            # These two properties must be handled separately
            if k not in ('info', 'time'):
                _AttrDict._wrap(v)
            self[k] = v

        self.decoded_palettes = map(palette_decode, self.palettes)
        pal = self.color.palette_times
        if isinstance(pal, basestring):
            self.palette_times = [(0.0, int(pal)), (1.0, int(pal))]
        else:
            self.palette_times = zip(pal[::2], map(int, pal[1::2]))

        self.adj_frame_width, self.spp = None, None
        self.canonical_right = not (self.get('link') and
                (self.link == 'self' or self.link.get('right')))

    def set_profile(self, prof, offset=0.0, err_spread=True):
        """
        Sets the instance props which are dependent on a profile. Also
        calculates timing information, which is returned instead of being
        attached to the genome. May be called multiple times to set different
        options.

        ``prof`` is a profile dictionary. ``offset`` is the time in seconds
        that the first frame's effective presentation time should be offset
        from the natural presentation time. ``err_spread`` will spread the
        rounding error in this frame across all frames, such that PTS+(1/FPS)
        is exactly equal to the requested duration.

        Returns ``(err, times)``, where ``err`` is the rounding error in
        seconds (taking ``offset`` into account), and ``times`` is a list of
        the central time of each frame in the animation in relative-time
        coordinates. Also sets the ``spp`` and ``adj_frame_width`` properties.
        """
        self.spp = SplEval(self.camera.density.knotlist)
        self.spp.knots[1] *= prof['quality']
        fps, base_dur = prof['fps'], prof['duration']

        # TODO: test!
        dur = self.time.duration
        if isinstance(dur, basestring):
            clock = float(dur[:-1]) + offset
        else:
            clock = dur * base_dur + offset
        if self.canonical_right:
            nframes = int(np.floor(clock * fps))
        else:
            nframes = int(np.ceil(clock * fps))
        err = (clock - nframes / fps) / clock

        fw = self.time.frame_width
        if not isinstance(fw, list):
            fw = [0, fw, 1, fw]
        fw = [float(f[:-1]) * fps if isinstance(f, basestring)
              else float(f) / (clock * fps) for f in fw]
        self.adj_frame_width = SplEval(fw)

        times = np.linspace(offset, 1 - err, nframes + 1)
        # Move each time to a center time, and discard the last value
        times = times[:-1] + 0.5 * (times[1] - times[0])
        if err_spread:
            epts = np.linspace(-2*np.pi, 2*np.pi, nframes)
            times = times + 0.5 * err * (np.tanh(epts) + 1)
        return err, times

def json_encode_genome(obj):
    """
    Encode an object into JSON notation. This serializer only works on the
    subset of JSON used in genomes.
    """
    result = _js_enc_obj(obj).lstrip()
    result = '\n'.join(l.rstrip() for l in result.split('\n'))
    return result + '\n'

def _js_enc_obj(obj, indent=0):
    # TODO: test, like so many other things
    isnum = lambda v: isinstance(v, (float, int, np.number))

    def wrap(pairs, delims):
        do, dc = delims
        i = ' ' * indent
        out = ''.join([do, ', '.join(pairs), dc])
        if '\n' not in out and len(out) + indent < 70:
            return out
        return ''.join(['\n', i, do, ' ', ('\n'+i+', ').join(pairs),
                        '\n', i, dc])

    if isinstance(obj, dict):
        if not obj:
            return '{}'
        digsort = lambda kv: (int(kv[0]), kv[1]) if kv[0].isdigit() else kv
        ks, vs = zip(*sorted(obj.items(), key=digsort))
        if ks == ('b', 'g', 'r'):
            ks, vs = reversed(ks), reversed(vs)
        ks = [crep('%.6g' % k if isnum(k) else str(k)) for k in ks]
        vs = [_js_enc_obj(v, indent+2) for v in vs]
        return wrap(['%s: %s' % p for p in zip(ks, vs)], '{}')
    elif isinstance(obj, list):
        vs = [_js_enc_obj(v, indent+2) for v in obj]
        if vs and len(vs) % 2 == 0 and isnum(obj[0]):
            vs = map(', '.join, zip(vs[::2], vs[1::2]))
        return wrap(vs, '[]')
    elif isinstance(obj, SplEval):
        return _js_enc_obj(obj.knotlist, indent)
    elif isinstance(obj, basestring):
        return crep(obj)
    elif isnum(obj):
        return '%.6g' % obj
    raise TypeError("Don't know how to serialize %s of type %s" %
                    (obj, type(obj)))

class XMLGenomeParser(object):
    """
    Parse an XML genome into a list of dictionaries.
    """
    def __init__(self):
        self.flames = []
        self._flame = None
        self.parser = xml.parsers.expat.ParserCreate()
        self.parser.StartElementHandler = self.start_element
        self.parser.EndElementHandler = self.end_element

    def start_element(self, name, attrs):
        if name == 'flame':
            assert self._flame is None
            self._flame = dict(attrs)
            self._flame['xforms'] = []
            self._flame['palette'] = np.ones((256, 4), dtype=np.float32)
        elif name == 'xform':
            self._flame['xforms'].append(dict(attrs))
        elif name == 'finalxform':
            self._flame['finalxform'] = dict(attrs)
        elif name == 'color':
            idx = int(attrs['index'])
            self._flame['palette'][idx][:3] = [float(v) / 255.0
                                               for v in attrs['rgb'].split()]
        elif name == 'symmetry':
            self._flame['symmetry'] = int(attrs['kind'])
    def end_element(self, name):
        if name == 'flame':
            self.flames.append(self._flame)
            self._flame = None

    @classmethod
    def parse(cls, src):
        parser = cls()
        parser.parser.Parse(src, True)
        return parser.flames

def convert_flame(flame):
    """
    Convert an XML flame (as returned by XMLGenomeParser) into a plain dict
    in cuburn's JSON genome format representing a loop edge.
    """
    cvt = lambda ks: dict((k, float(flame[k])) for k in ks)
    camera = {
        'center': dict(zip('xy', map(float, flame['center'].split()))),
        'scale': float(flame['scale']) / float(flame['size'].split()[0]),
        'dither_width': float(flame['filter']),
        'rotation': float(flame['rotate']),
        'density': 1.0
    }

    info = {}
    if 'name' in flame:
        info['name'] = flame['name']
    if 'nick' in flame:
        info['authors'] = [flame['nick']]
        if flame.get('url'):
            info['authors'][0] = info['authors'][0] + ', http://' + flame['url']

    time = dict(frame_width=float(flame.get('temporal_filter_width', 1)),
                duration=1)

    color = cvt(['brightness', 'gamma'])
    color.update((k, float(flame.get(k, d))) for k, d in
                 [('highlight_power', -1), ('gamma_threshold', 0.01)])
    color['vibrance'] = float(flame['vibrancy'])
    color['background'] = dict(zip('rgb',
                               map(float, flame['background'].split())))
    color['palette_times'] = "0"
    pal = palette_encode(flame['palette'])

    de = dict((k, float(flame.get(f, d))) for f, k, d in
                [('estimator', 'radius', 11),
                 ('estimator_minimum', 'minimum', 0),
                 ('estimator_curve', 'curve', 0.6)])

    num_xf = len(flame['xforms'])
    xfs = dict([(str(k), convert_xform(v, num_xf))
                for k, v in enumerate(flame['xforms'])])
    if 'symmetry' in flame:
        xfs.update(make_symm_xforms(flame['symmetry'], len(xfs)))
    if 'finalxform' in flame:
        xfs['final'] = convert_xform(flame['finalxform'], num_xf, True)

    return dict(camera=camera, color=color, de=de, xforms=xfs,
                info=info, time=time, palettes=[pal], link='self')

def convert_xform(xf, num_xf, isfinal=False):
    # TODO: chaos
    xf = dict(xf)
    symm = float(xf.pop('symmetry', 0))
    anim = xf.pop('animate', symm <= 0)
    out = dict((k, float(xf.pop(k, v))) for k, v in
               dict(color=0, color_speed=(1-symm)/2, opacity=1).items())
    if not isfinal:
        out['density'] = float(xf.pop('weight'))
    out['affine'] = convert_affine(xf.pop('coefs'), anim)
    if 'post' in xf and map(float, xf['post'].split()) != [1, 0, 0, 1, 0, 0]:
        out['post'] = convert_affine(xf.pop('post'))
    if 'chaos' in xf:
        chaos = map(float, xf.pop('chaos').split())
        out['chaos'] = dict()
        for i in range(num_xf):
            if i < len(chaos):
                out['chaos'][str(i)] = chaos[i]
            else:
                out['chaos'][str(i)] = 1.0
        
    out['variations'] = {}
    for k in var_code:
        if k in xf:
            var = dict(weight=float(xf.pop(k)))
            for param, default in var_params.get(k, {}).items():
                var[param] = float(xf.pop('%s_%s' % (k, param), default))
            out['variations'][k] = var
    assert not xf, 'Unrecognized parameters remain: ' + str(xf)
    return out

def convert_affine(aff, animate=False):
    xx, yx, xy, yy, xo, yo = map(float, aff.split())
    # Invert all instances of y (yy is inverted twice)
    yx, xy, yo = -yx, -xy, -yo

    xa = np.degrees(np.arctan2(yx, xx))
    ya = np.degrees(np.arctan2(yy, xy))
    xm = np.hypot(xx, yx)
    ym = np.hypot(xy, yy)

    angle_between = ya - xa
    if angle_between < 0:
        angle_between += 360

    if angle_between < 180:
        spread = angle_between / 2.0
    else:
        spread = -(360-angle_between) / 2.0

    angle = xa + spread
    if angle < 0:
        angle += 360.0

    if animate:
        angle = [0, angle, 1, angle - 360]

    return dict(spread=spread, magnitude={'x': xm, 'y': ym},
                angle=angle, offset={'x': xo, 'y': yo})

def make_symm_xforms(kind, offset):
    assert kind != 0, 'Pick your own damn symmetry.'
    out = []
    boring_xf = dict(color=1, color_speed=0, density=1, opacity=1,
                     variations={'linear': {'weight': 1}})
    if kind < 0:
        out.append(boring_xf.copy())
        out[-1]['affine'] = dict(angle=135, magnitude={'x': 1, 'y': 1},
                                 spread=-45, offset={'x': 0, 'y': 0})
        kind = -kind
    for i in range(1, kind):
        out.append(boring_xf.copy())
        if kind >= 3:
            out[-1]['color'] = (i - 1) / (kind - 2.0)
        ang = (45 + 360 * i / float(kind)) % 360
        out[-1]['affine'] = dict(angle=ang, magnitude={'x': 1, 'y': 1},
                                 spread=-45, offset={'x': 0, 'y': 0})
    return dict((str(i+offset), v) for i, v in enumerate(out))

def convert_file(path):
    """Quick one-shot conversion for an XML genome."""
    flames = XMLGenomeParser.parse(open(path).read())
    if len(flames) > 10:
        warnings.warn("Lot of flames in this file. Sure it's not a "
                      "frame-based animation?")
    for flame in flames:
        yield convert_flame(flame)

if __name__ == "__main__":
    import sys
    print '\n\n'.join(map(json_encode_genome, convert_file(sys.argv[1])))
