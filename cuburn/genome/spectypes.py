# -*- encoding: utf-8 -*-

u"""
These types are used to define the document schemas in the `specs` module. The
specs are native-Python representations of the schemas, but are designed to be
easily transformed into, say, JSON, for use in other languages and
applications.

Each spectype describes the method used by cuburn to encode its value into a
plain JSON entity. With one exception (splines), the encoding is
straightforward and context-free.

Specs may change without notice, and are not currently versioned.
Implementations should tolerate ambiguity, missing or unknown properties, and
improper encodings in whatever way is best for the user.

# Dict

The most common spectype is a plain Python dict, and accordingly doesn't have
a type defined in this module. Each key must be a string which would be a
valid Python identifier; values may be any type below, or a nested dict.
There are two keys that hold special significance:

- "type", if present, shall be an ordinary string, used for identifying the
  type of a top-level object in situations where that information may not be
  recoverable from context. If a spec dict has a "type" param, any instances
  of the spec should have an identical "type" property.

- "doc", if present, shall be an ordinary string used to describe the dict
  (in Markdown syntax). This shall be ignored by any instances, and is mostly
  there for self-describing UIs.

Instances of spec dicts should be encoded to native JSON dicts.

Missing values have semantic significance[^missing]. Encoders must omit any
key from an encoded dict which a user did not include. Encoders must also
recursively omit any key whose child is an empty dict.

[^missing]: Although direct property access in cuburn through a `use.Wrapper`
type will often provide a sensible default value, various container methods —
`'x' in y`, `y.keys()` — only include keys that were present on the decoded
object. In the case of rendering, the output will be the same, but the process
will take longer. In other cases, such as mutation, having default, dummy
values present *will* change the result. Plus it just makes things into an
unreadable soup to have too many default values in there.

# Map

A Map is a dict where possible keys are not known in advance. The spec of each
value in the Map is the same, and is given by `Map.type`. This is used, for
instance, to implement the `"xforms"` entity.

Maps should be encoded exactly as plain dicts are above.

# List

A List indicates a list of values of a single known type. Encode as a plain
JSON list.

An empty List is semantically distinct from a missing List. (For example, it
might be overriding a List with a non-empty default value.) If a user has
specified an empty list, it must be present in the encoded entity.

It's recommended to avoid lists of Scalars or RefScalars, as these can be
confused for splines by simple-minded parsers/formatters.

# Spline

A cubic spline. The heart of the genome specification, and easily the most
complex type, with a lot of exceptions and some context sensitivity.

## Spectype parameters

- `default` specifies a default scalar value. This value will be used if no
  value has been specified. It may also be used during blending (see
  "Blending" below).

- `min` and `max` document the minimum and maximum values that a parameter
  should ever hit during interpolation. These values may be used by
  higher-level tools, such as UIs, but should not be enforced at the JSON
  encoding level in any way. May be 'None' to indicate no limit.

- `interp` indicates the kind of interpolation scaling that should be
  performed. See "Interpolation scaling" below for details.

- `period` is the size of the period, such as `360` for a parameter that
  specifies angles in degrees. For non-periodic parameters, this must be
  `None`. See "Periodic extension" below.

- `var` is a boolean property. If True, the value should be interpreted as a
  variation property. See "Blending" below.

## JSON encoding and semantics

Splines make up the overwhelming majority of a given genome's description. I
felt it appropriate to optimize for readability and hand-editability at the
expense of simplicity of implementation. As a result, splines have a few
different ways of being encoded. I know it's icky to overload JSON lists in so
many ways, but the parsing complexity this introduces should be very easily
contained and the readability benefits are entirely worth it.

It is recommended that all numbers be encoded using six significant values
(i.e. the '%6g' format specifier). This will in most cases eliminate
inconsistencies due to single-precision floating-point rounding error.

### Nodes

Nodes specify a single point in time across an entire file. Each Spline value
in a node specifies the position and velocity of the parameter at that time.
(Acceleration is not a free parameter in cubic spline interpolation, and is
not specified.)

If velocity is nonzero at a given point, the parameter value must be expressed
as a list of precisely two numbers, with the first being position and the
second velocity: `[135.4, -360]`. If velocity is zero, the parameter may be
expressed as a single number: `1.3`.

Implementations must not produce a single-element or empty list; it's either a
pair of values in a list, a plain number, or the key should be absent.

### Edges

Edges specify parameter values over a continuous period of time, starting from
one node and ending at another. The position and velocity of each Spline at
the beginning and end of this time are therefore specified by the source and
destination nodes, and save for periodic extension (see "Blending" below),
these endpoints cannot be altered by the edge.

Edges can, however, insert knots at any point in time strictly inside the
`[0,1]` normalized time over which an edge is defined. Each knot is a `time,
position` pair (velocity may not be specified directly). Each pair of values
should be concatenated into a single list, such as `[0.5, 1]` or `[0.25, -1,
0.75, 3]`.

As with nodes, empty lists and lists with odd numbers of elements must not be
produced during encoding.

### Animations

Animations are the blended results produced by merging an edge and its two
nodes (each of which in turn may be the result of merging multiple edits).
Animations are self-contained and ready to render, so animation Splines
contain all the information above.

There are three representations of a Spline in an animation:

- Plain number: `1.1`. This indicates the value at both `t=0` and `t=1`, and
  implies that this value remains constant across the entire animation.

- Two-element list: `[3.2, 0]`. The first element indicates the value at
  `t=0`; the second at `t=1`. The velocity at both endpoints is inferred to be
  zero.

- Four-or-more-element list: `[45, -360, -135, 0, 0.3, 150]`. This is a list
  of pairs, and must always contain an even number of elements. The first pair
  indicates the position and velocity of the parameter at `t=0`; the second,
  at `t=1`. Subsequent pairs indicate `[time, position]` knots, just as in
  edges.

### Merging edits

Nodes and edges support an "edit stack" (to be documented elsewhere). When
merging edits to form a single node, each new Spline value replaces
both the position and velocity of any previous Spline value for a given
parameter, even if the velocity is an implied 0.

When merging edge edits, the composed Spline should consist of a list of
pairs with no more than one unique value of `time` in the list. If multiple
edits include a value for the same `time`, the `position` value for that time
should be the one specified on the highest (most recent) edit. To enable
removal of lower entries by higher ones, `position` values may also be
encoded as JSON `nil` values; these values must be removed from the combined
list before passing to the application.

A simple conceptual procedure for doing this is:

- Concatenate all Spline lists in all edits, such that the lowest entries are
  first;
- Reverse the list;
- For each `time, position` pair starting with the first, remove any subsequent
  pairs which share the same value of `time`;
- For each `time, position` pair starting with the first, remove that pair if
  the value of `position` is `nil`.

### Temporal scaling

When you change the duration of a file, you shouldn't have to go around
changing every other value in the file to compensate. And you don't! But the
way time is encoded is a bit subtle as a result.

Cuburn implicitly defines an abstract unit of time (call it U for this
section). When rendering, the profile contains a parameter,
`profile.duration`, that specifies the duration of 1U in seconds. Combined
with `profile.fps`, this defines how Cuburn maps U to individual frames.

Nodes can define velocities. We always want edges to match up perfectly with
the nodes on each side, so that concatenating an edge that has a certain
destination node with another edge that has that node as its source will
produce a perfectly seamless transition. This means that the velocities in a
node must have the same meaning no matter what context. As a result,
velocities are defined as the change in position per 1U time.

Edges, on the other hand, can specify times. If you want to specify that, say,
a big spike in a parameter value should happen half-way through the edge, you
can specify it by adding a knot at `t=0.5`. Here's the catch: an edge can have
an arbitrary duration. The `time.duration` property on an edge specifies how
long the edge should be against units U, so `time.duration=3.2` says that this
edge should last for 3.2U.

If we specified knot times directly in U, that means changing the duration
from 1 to 3.2 would suddenly put that big spike at `t=0.5` about a sixth of
the way through the video, and would leave nothing interesting happening in
the latter two-thirds of the video. So instead, knot times are specified
relative to the edge duration. (Whenever we use `t`, like in `t=0.5`, we're
referring to this file-local unit of time.)

Ultimately, cuburn (and any other implementations) will take care of this for
users, automatically rescaling the velocities to file-local time before it
hits the GPU or the UI. The only visible consequence is that scaling duration
isn't perfectly linear; the paths taken by splines when you stretch out the
video won't be quite the same, because the endcap knots (another internal
detail) will change to match the velocity.

## Blending

When blending two nodes, the possibility arises that a parameter may be
present on one node but not the other. In this case, the missing position will
in most circumstances be determined by the value of the `default`
parameter of the Spline.

However, for variations, this is undesirable. Variations in general have no
guarantees on parameter stability under interpolation, and interpolating to or
from the default value may cause the animation to react violently. Since
a variation has no effect as its weight drops to zero, we minimize the chances
of these wild interpolations by instead copying a variation parameter's missing
position from the opposite node. This behavior is indicated by the `var`
property of the Spline; when it is True, the copy-from-other behavior is used
during blending.

There is still the possibility, regardless of `var`, that position information
is missing from *both* nodes being blended (because, say, the edge specifies
knots for a node missing from both). In this case both sides use the default
position value.

## Periodic extension

For periodic parameters, adjusting the parameter's position by an integer
multiple of the `period` will still result in a signal that perfectly aligns
with the node specification. The blend algorithm takes advantage of this to
minimize the deviation between the average of the scaled velocities of both
nodes and the average velocity of the parameter's path for periodic signals.

> An example: two nodes both define the `camera.rotation` property as having a
> position of 0 and a velocity of -360 (meaning one full counterclockwise
> rotation per unit of time U). An edge between these two nodes has a
> `blend.duration` value of 3. The average velocity of the two nodes is -360/U,
> and the duration is 3U, so the resulting animation has a spline that starts
> at 0 and ends at -1080.

An edge can define additional knots to shape the interior of any Spline path,
but in most circumstances may not define any behavior outside of the closed
interval `t=(0,1)`; such knots should simply be ignored. However, for Splines
that define a `period`, this restriction is expanded to the open interval
`t=[0,1]`, to enable per-spline control over the behavior described above. A
knot value for a boundary time will be rounded to the nearest value such that
the difference between that value and the position specified by the node is an
integer multiple of the period size.

> Continuing the example: let's add a knot for `camera.rotation` at `[1, 10]`
> to the edge. Now, the blend will read the knot, round it to the nearest
> equivalent value in mod-360 — in this case, just back to 0 — and use that in
> place of the -1080 calculated by velocity. The camera will then appear to
> rotate counterclockwise for a time, slow to a stop, begin traveling
> clockwise for a time, stop again, and finally travel counterclockwise again.
"""

from collections import namedtuple

Map = namedtuple('Map', 'type doc')
List = namedtuple('List', 'type default doc')
Spline = namedtuple('Spline', 'default min max interp period doc var')
Scalar = namedtuple('Scalar', 'default doc')
RefScalar = namedtuple('RefScalar', 'default ref doc')
String = namedtuple('String', 'doc')
Enum = namedtuple('Enum', 'choices default doc')
Palette = namedtuple('Palette', '')

# Plain helper constructors
map_ = lambda type, d=None: Map(type, d)
list_ = lambda type, default=(), d=None: List(type, default, d)
scalar = lambda default, d=None: Scalar(default, d)
refscalar = lambda default, ref, d=None: RefScalar(default, ref, d)

# Specialized helper constructors
def spline(default=0, min=None, max=None, interp='linear', period=None, d=None):
    return Spline(default, min, max, interp, period, d, False)
def scalespline(default=1, min=0, max=None, d=None):
    """Spline helper, with defaults appropriate for a scaling parameter."""
    return Spline(default, min, None, 'mag', None, d, False)
def enum(choices, default=None, d=None):
    """Enum helper. 'choices' is a list or a space-separated string."""
    if isinstance(choices, basestring):
        choices = choices.split()
    return Enum(choices, default, d)

class XYPair(dict):
    """
    Specialization of spline over two dimensions. Separate type is a hint to
    UIs and mutator, but this may be treated just like a normal dict.
    """
    def __init__(self, type):
        self['x'] = self['y'] = self.type = type

def export_spec(spec):
    """
    Return a JSON-serializable representation of a spec for use in non-Python
    applications.
    """
    if isinstance(spec, dict):
        return dict((k, export_spec(v)) for k, v in spec.items())
    elif isinstance(spec, basestring):
        return spec
    else:
        r = spec._asdict()
        r.update(type=type(spec).__name__)
        return r
