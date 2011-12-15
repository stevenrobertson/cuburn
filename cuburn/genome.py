import json
import base64
import numpy as np

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
    """Wafer-thin wrapper around palettes. For the future!"""
    def __init__(self, datastr, fmt='rgb8'):
        if fmt != 'rgb8':
            raise NotImplementedError
        if len(datastr) != 768:
            raise ValueError("Unsupported palette width")
        self.width = 256
        pal = np.reshape(np.fromstring(datastr, np.uint8), (256, 3))
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
    # For now, we base the Genome class on an _AttrDict, letting its structure
    # be defined implicitly by the way it is used in device code. More formal
    # unpacking will happen soon.
    def __init__(self, gnm, base_den):
        super(Genome, self).__init__(gnm)
        for k, v in self.items():
            v = _AttrDict(v)
            if k not in ('info', 'time'):
                _AttrDict._wrap(v)
            self[k] = v
        # TODO: this is a hack, figure out how to solve it more elegantly
        self.spp = SplEval(self.camera.density.knotlist)
        self.spp.knots[1] *= base_den
        # TODO: decide how to handle palettes. For now, it's the caller's
        # responsibility to replace this list with actual palettes.
        pal = self.color.palette
        if isinstance(pal, basestring):
            self.color.palette = [(0.0, pal), (1.0, pal)]
        elif isinstance(pal, list):
            self.color.palette = zip(pal[::2], pal[1::2])

        # TODO: caller also needs to call set_timing()
        self.adj_frame_width = None
        self.canonical_right = (not self.get('link') or not self.link == 'self'
                                or not self.link.get('right'))

    def set_timing(self, base_dur, fps, offset=0.0, err_spread=True):
        """
        Set frame timing. Must be called at least once prior to rendering.
        """
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
