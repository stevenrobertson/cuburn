import numpy as np

from spectypes import Enum, Spline, Scalar, RefScalar, Map, List
from specs import toplevels

class Wrapper(object):
    """
    Weird reverse visitor. Traversals of the tree are normally done externally
    (via property accessors, in a lot of cases). This class alters the
    returned representation of the underlying genome according to the provided
    spec without imposing flow control.
    """
    def __init__(self, val, spec=None, path=()):
        if spec is None:
            assert val.get('type') in toplevels, 'Unrecognized dict type'
            spec = toplevels[val['type']]
        # plain 'val' would conflict with some variation property names
        self._val, self.spec, self.path = val, spec, path

    def wrap(self, name, spec, val):
        path = self.path + (name,)
        if isinstance(spec, Enum):
            return self.wrap_enum(path, spec, val)
        if isinstance(spec, Spline):
            return self.wrap_spline(path, spec, val)
        elif isinstance(spec, Scalar):
            return self.wrap_scalar(path, spec, val)
        elif isinstance(spec, RefScalar):
            return self.wrap_refscalar(path, spec, val)
        elif isinstance(spec, dict):
            return self.wrap_dict(path, spec, val)
        elif isinstance(spec, Map):
            return self.wrap_Map(path, spec, val)
        elif isinstance(spec, List):
            return self.wrap_List(path, spec, val)
        return self.wrap_default(path, spec, val)

    def wrap_default(self, path, spec, val):
        return val

    def wrap_enum(self, path, spec, val):
        return val or spec.default

    def wrap_spline(self, path, spec, val):
        return val

    def wrap_scalar(self, path, spec, val):
        return val if val is not None else spec.default

    def wrap_dict(self, path, spec, val):
        return type(self)(val or {}, spec, path)

    def wrap_Map(self, path, spec, val):
        return self.wrap_dict(path, spec, val)

    def wrap_List(self, path, spec, val):
        val = val if val is not None else spec.default
        return [self.wrap(path, spec.type, v) for v in val]

    def get_spec(self, name):
        if isinstance(self.spec, Map):
            return self.spec.type
        return self.spec[name]

    @classmethod
    def visit(cls, obj):
        """
        Visit every node. Note that for simplicity, this function will be
        called on all elements (i.e. pivoting to a new Wrapper type inside the
        wrapping function and overriding visit() won't do anything).
        """
        if isinstance(obj, (Wrapper, dict)):
            return dict((k, cls.visit(obj[k])) for k in obj)
        elif isinstance(obj, list):
            return [cls.visit(o) for o in obj]
        return obj

    def __getattr__(self, name):
        return self.wrap(name, self.get_spec(name), self._val.get(name))

    # Container emulation
    def keys(self):
        return sorted(self._val.keys())
    def items(self):
        return sorted((k, self[k]) for k in self)
    def __contains__(self, name):
        self.get_spec(name) # raise IndexError if name is not typed
        return name in self._val
    def __iter__(self):
        return iter(sorted(self._val))
    def __getitem__(self, name):
        return getattr(self, str(name))

class RefWrapper(Wrapper):
    """
    Wrapper that handles RefScalars, as with profile objects.
    """
    # Turns out (somewhat intentionally) that every spline parameter used on
    # the host has a matching parameter in the profile, so this
    def __init__(self, val, other, spec=None, path=()):
        super(RefWrapper, self).__init__(val, spec, path)
        self.other = other

    def wrap_dict(self, path, spec, val):
        return type(self)(val or {}, self.other, spec, path)

    def wrap_refscalar(self, path, spec, val):
        spev = self.other
        for part in spec.ref.split('.'):
            spev = spev[part]
        spev *= val if val is not None else spec.default
        return spev

class SplineWrapper(Wrapper):
    def wrap_spline(self, path, spec, val):
        return SplineEval(val if val is not None else spec.default,
                          spec.interp)

class SplineEval(object):
    _mat = np.matrix([[1.,-2, 1, 0], [2,-3, 0, 1],
                      [1,-1, 0, 0], [-2, 3, 0, 0]])
    _deriv = np.matrix(np.diag([3,2,1], 1))

    def __init__(self, knots, interp='linear'):
        self.knots, self.interp = self.normalize(knots), interp

    @staticmethod
    def normalize(knots):
        if isinstance(knots, (int, float)):
            v0, v1 = 0, 0
            knots = [(0, knots), (1, knots)]
        elif len(knots) % 2 != 0:
            raise ValueError("List with odd number of elements given")
        elif len(knots) == 2:
            v0, v1 = 0, 0
            knots = [(0, knots[0]), (1, knots[1])]
        else:
            p0, v0, p1, v1 = knots[:4]
            knots = [(0, p0), (1, p1)] + zip(knots[4::2], knots[5::2])

        knots = sorted(knots)

        # If stabilizing knots are missing before or after the edges of the
        # [0,1] interval, add them. In almost all cases, the precise timing of
        # the end knots has little affect on the shape of the curve.
        td = 2
        if knots[0][0] >= 0:
            knots = [(-td, knots[1][1] - (knots[1][0] - (-td)) * v0)] + knots
        if knots[-1][0] <= 1:
            knots.extend([(1+td, knots[-2][1] + (1+td - knots[-2][0]) * v1)])

        knotarray = np.zeros((2, len(knots)))
        knotarray.T[:] = knots
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
        # TODO: respect 'interp' THIS IS IMPORTANT.
        times, vals, t, scale = self.find_knots(itime)

        m1 = (vals[2] - vals[0]) / (1.0 - times[0])
        m2 = (vals[3] - vals[1]) / times[3]

        mat = self._mat
        if deriv:
            mat = mat * (scale * self._deriv) ** deriv
        val = [m1, vals[1], m2, vals[2]] * mat * np.array([[t**3, t**2, t, 1]]).T
        return val[0,0]

    def __imul__(self, other):
        self.knots[1] *= other
        return self

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

def wrap_genome(prof, gnm):
    # It's not obvious that this is what needs to happen, so we wrap. The
    # timing is simplistic, and may get expanded or moved later.
    gprof = RefWrapper(prof, SplineWrapper(gnm), toplevels['profile'])
    nframes = round(gprof.fps * gprof.duration)
    times = np.linspace(0, 1, nframes + 1)
    times = times[:-1] + 0.5 * (times[1] - times[0])
    return gprof, times
