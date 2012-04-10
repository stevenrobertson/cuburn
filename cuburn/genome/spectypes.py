from collections import namedtuple

Map = namedtuple('Map', 'type doc')
map_ = lambda type, d=None: Map(type, d)

List = namedtuple('List', 'type doc')
list_ = lambda type, d=None: List(type, d)

# A list as above, but where each element is a dict with a 'type' parameter
# corresponding to one of the specs listed in the 'types' dict on this spec.
TypedList = namedtuple('TypedList', 'types defaults doc')
typedlist = lambda types, defaults=[], d=None: TypedList(types, defaults, d)

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
Enum = namedtuple('Enum', 'choices doc')
def enum(choices, d=None):
    """Enum helper. 'choices' is a space-separated string."""
    return Enum(choices.split(), d)

Palette = namedtuple('Palette', '')
