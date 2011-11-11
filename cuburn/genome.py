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

    @classmethod
    def wrap(cls, obj):
        """
        Given a dict 'obj' representing, for instance, a Genome object, walk
        through the object recursively and in-place, turning any number or
        list of numbers into an SplEval.
        """
        for k, v in obj.items():
            if (isinstance(v, (float, int)) or
                    (isinstance(v, list) and isinstance(v[1], (float, int)))):
                obj[k] = cls(v)
            elif isinstance(v, dict):
                cls.wrap(v)

class RenderInfo(object):
    """
    Determine features and constants required to render a particular set of
    genomes. The values of this class are fixed before compilation begins.
    """
    # Number of iterations to iterate without write after generating a new
    # point. This number is currently fixed pretty deeply in the set of magic
    # constants which govern buffer sizes; changing the value here won't
    # actually change the code on the device to do something different.
    fuse = 256

    # Height of the texture pallete which gets uploaded to the GPU (assuming
    # that palette-from-texture is enabled). For most genomes, this doesn't
    # need to be very large at all. However, since only an easily-cached
    # fraction of this will be accessed per SM, larger values shouldn't hurt
    # performance too much. Power-of-two, please.
    palette_height = 16

    # Maximum width of DE and other spatial filters, and thus in turn the
    # amount of padding applied. Note that, for now, this must not be changed!
    # The filtering code makes deep assumptions about this value.
    gutter = 22

    # TODO: for now, we always throw away the alpha channel before writing.
    # All code is in place to not do this, we just need to find a way to expose
    # this preference via the API (or push alpha blending entirely on the client,
    # which I'm not opposed to)
    alpha_output_channel = False

    # There are three settings for this somewhat ersatz paramater. 'global'
    # uses unsynchronized global writes to accumulate sample points, 'atomic'
    # uses atomic global writes, and 'deferred' stores color and position in a
    # sample log, sorts the log by position, and uses shared memory to
    # perform the accumulation. Deferred has the accuracy of 'atomic' and
    # the speed of 'global' (it's actually faster!), but packs color and
    # position into a single 32-bit int for now, which limits resolution to
    # 1080p when xform opacity is respected, so the other two modes will hang
    # around until that can be extended to be memory-limited again.
    acc_mode = 'deferred'

    # TODO: fix this
    chaos_used = False

    def __init__(self, db, **kwargs):
        self.db = db
        # Copy all args into this object's namespace
        self.__dict__.update(kwargs)

        self.acc_width = self.width + 2 * self.gutter
        self.acc_height = self.height + 2 * self.gutter
        self.acc_stride = 32 * int(np.ceil(self.acc_width / 32.))
        self.density = self.quality

        # Deref genome
        self.genome = self.db.genomes[self.genome]

        for k, v in self.db.palettes.items():
            pal = np.fromstring(base64.b64decode(v), np.uint8)
            pal = np.reshape(pal, (256, 3))
            pal_a = np.ones((256, 4), np.float32)
            pal_a[:,:3] = pal / 255.0
            self.db.palettes[k] = pal_a

class _AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

def load_info(contents):
    result = json.loads(contents, object_hook=_AttrDict)
    SplEval.wrap(result.genomes)

    # A Job object will have more details or something
    result = RenderInfo(result, **result.renders.values()[0])
    return result

class HacketyGenome(object):
    """
    Holdover class to postpone a very deep refactoring as long as possible.
    Converts property accesses into interpolations over predetermined times.
    """
    def __init__(self, referent, times):
        # Times can be singular
        self.referent, self.times = referent, times
    def __getattr__(self, name):
        r = getattr(self.referent, str(name))
        if isinstance(r, _AttrDict):
            return HacketyGenome(r, self.times)
        elif isinstance(r, SplEval):
            return r(self.times)
        return r
    __getitem__ = __getattr__

if __name__ == "__main__":
    import sys
    import pprint
    pprint.pprint(read_genome(sys.stdin))
