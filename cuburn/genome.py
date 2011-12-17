import base64
import numpy as np

from code.util import crep

class SplEval(object):
    _mat = np.matrix([[1.,-2, 1, 0], [2,-3, 0, 1],
                      [1,-1, 0, 0], [-2, 3, 0, 0]])
    _deriv = np.matrix(np.diag([3,2,1], 1))

    def __init__(self, knots):
        # If a single value is passed, normalize to a constant set of knots
        if isinstance(knots, (int, float)):
            knots = [-0.1, knots, 0.0, knots, 1.0, knots, 1.1, knots]
        elif not np.all(np.diff(np.float32(np.asarray(knots))[::2]) > 0):
            raise ValueError("Spline times are non-monotonic. (Use "
                    "nextafterf()-spaced times to anchor tangents.)")

        # If stabilizing knots are missing before or after the edges of the
        # [0,1] interval, add them. We choose knots that preserve the first
        # differential, which is probably what is expected when two knots are
        # present but may be less accurate for three. Of course, one can just
        # add stabilizing knots to the genome to change this.
        if knots[0] >= 0:
            m = (knots[3] - knots[1]) / (knots[2] - knots[0])
            knots = [-0.1, knots[1] + -0.1 * m] + knots
        if knots[-2] <= 1:
            m = (knots[-1] - knots[-3]) / (knots[-2] - knots[-4])
            knots.extend([1.1, knots[-1] + 0.1 * m])

        self.knots = np.zeros((2, len(knots)/2))
        self.knots.T.flat[:] = knots

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
        if np.std(self.knots[1]) < 1e-6:
            return self.knots[1][0]
        return list(self.knots.T.flat)

class Palette(object):
    def __init__(self, datastrs):
        if datastrs[0] != 'rgb8':
            raise NotImplementedError
        self.width = 256
        raw = base64.b64decode(''.join(datastrs[1:]))
        pal = np.reshape(np.fromstring(raw, np.uint8), (256, 3))
        self.data = np.ones((256, 4), np.float32)
        self.data[:,:3] = pal / 255.0

class _AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

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

        self.decoded_palettes = map(Palette, self.palettes)
        print self.color
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

def json_encode_genome(obj, indent=0):
    """
    Encode an object into JSON notation. This serializer only works on the
    subset of JSON used in genomes.
    """
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
        ks, vs = zip(*sorted(obj.items()))
        if ks == ('b', 'g', 'r'):
            ks, vs = reversed(ks), reversed(vs)
        ks = [crep('%.8g' % k if isnum(k) else str(k)) for k in ks]
        vs = [json_encode_genome(v, indent+2) for v in vs]
        return wrap(['%s: %s' % p for p in zip(ks, vs)], '{}')
    elif isinstance(obj, list):
        vs = [json_encode_genome(v, indent+2) for v in obj]
        if vs and len(vs) % 2 == 0 and isnum(obj[0]):
            vs = map(', '.join, zip(vs[::2], vs[1::2]))
        return wrap(vs, '[]')
    elif isinstance(obj, SplEval):
        return json_encode_genome(obj.knotlist, indent)
    elif isinstance(obj, basestring):
        return crep(obj)
    elif isnum(obj):
        return '%.8g' % obj
    raise TypeError("Don't know how to serialize %s of type %s" %
                    (obj, type(obj)))
