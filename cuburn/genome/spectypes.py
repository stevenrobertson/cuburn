from collections import namedtuple

Map = namedtuple('Map', 'type doc')
map_ = lambda type, d=None: Map(type, d)

List = namedtuple('List', 'type default doc')
list_ = lambda type, default=(), d=None: List(type, default, d)

Spline = namedtuple('Spline', 'default min max interp period doc var')
def spline(default=0, min=None, max=None, interp='linear', period=None, d=None):
    return Spline(default, min, max, interp, period, d, False)
def scalespline(default=1, min=0, max=None, d=None):
    """Spline helper, with defaults appropriate for a scaling parameter."""
    return Spline(default, min, None, 'mag', None, d, False)

class XYPair(dict):
    """
    Specialization of spline over two dimensions. Separate type is a hint to
    UIs and mutator, but this may be treated just like a normal dict.
    """
    def __init__(self, type):
        self['x'] = self['y'] = self.type = type

Scalar = namedtuple('Scalar', 'default doc')
scalar = lambda default, d=None: Scalar(default, d)

# These are scalars, as used in profiles, but which are scaled by some other
# parameter (in the genome) given by name as ``ref``.
RefScalar = namedtuple('RefScalar', 'default ref doc')
refscalar = lambda default, ref, d=None: RefScalar(default, ref, d)

String = namedtuple('String', 'doc')
def string_(d=None):
    return String(d)
Enum = namedtuple('Enum', 'choices default doc')
def enum(choices, default=None, d=None):
    """Enum helper. 'choices' is a list or a space-separated string."""
    if isinstance(choices, basestring):
        choices = choices.split()
    return Enum(choices, default, d)

Palette = namedtuple('Palette', '')
