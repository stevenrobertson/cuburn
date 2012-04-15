from spectypes import *
from variations import var_params

affine = (
  { 'angle': spline(45, period=360)
  , 'spread': spline(45, period=180)
  , 'magnitude': XYPair(scalespline())
  , 'offset': XYPair(spline())
  })

xform = (
  { 'pre_affine': affine
  , 'post_affine': affine
  , 'color': spline(0, 0, 1)
  , 'color_speed': spline(0.5, 0, 1)
  , 'weight': spline()
  , 'opacity': scalespline(max=1)
  , 'variations': var_params
  })

# Since the structure of the info element differs between anims, nodes and
# edges, we pull out some of the common elements here
author = String('Attribution in the form: "Name [<email>][, url]"')
name = String('A human-readable name for this entity')
src = String('The identifier of the source node')
dst = String('The identifier of the destination node')

filters = (
  { 'bilateral':
    { 'spatial_std': scalespline(6,
        d='Spatial filter radius, normalized to 1080p pixels')
    , 'color_std': scalespline(0.05,
        d='Color filter radius, in YUV space, normalized to [0,1]')
    , 'density_std': scalespline(1.5, d='Density standard deviation')
    , 'density_pow': scalespline(0.8, d='Density pre-filter power')
    , 'gradient': scalespline(4.0, min=None,
        d='Intensity of gradient amplification (can be negative)')
    }
  , 'colorclip':
    { 'gamma': scalespline(4)
    , 'gamma_threshold': spline(0.01, 0, 1)
    , 'highlight_power': spline(-1, -1, 1)
    , 'vibrance': scalespline()
    }
  , 'de':
    { 'radius': scalespline(11, d='Spatial filter radius in flam3 units')
    , 'minimum': scalespline(0, max=1, d='Proportional min radius')
    , 'curve': scalespline(0.6, d='Power of filter radius with density')
    }
  , 'haloclip': {'gamma': scalespline(4)}
  , 'logscale': {'brightness': scalespline(4, d='Log-scale brightness')}
  })

camera = (
  { 'center': XYPair(spline())
  , 'spp': scalespline(d='Samples per pixel multiplier')
  , 'dither_width': scalespline()
  , 'rotation': spline(period=360)
  , 'scale': scalespline()
  })

time = (
  { 'duration': scalar(1)
  , 'frame_width': scalespline(d='Scale of profile temporal width per frame.')
  })

base = (
  { 'camera': camera
  , 'filters': filters
  , 'palette': list_(Palette())
  , 'xforms': map_(xform)
  , 'final_xform': xform
  })

anim = dict(base)
anim.update(type='animation', time=time,
            info=dict(authors=list_(author), name=name, src=src, dst=dst,
                      origin=string_()))

# TODO
node = dict(base)
node.update(type='node', info=dict(author=author, author_url=string_(),
                                   id=string_(), name=name))

# TODO
edge = dict(anim)
edge.update(type='edge',
            info=dict(author=author, id=string_(), src=src, dst=dst),
            xforms=dict(src=map_(xform), dst=map_(xform)))

# Yeah, now I'm just messing around.
prof_filters = dict([(fk, dict([(k, refscalar(1, '.'.join(['filters', fk, k])))
                           for k in fv])) for fk, fv in filters.items()])
# And here's a completely stupid hack to drag scale into the logscale filter
prof_filters['logscale']['scale'] = refscalar(1, 'camera.scale')

default_filters = [{'type': k} for k in ['bilateral', 'logscale', 'colorclip']]

profile = (
  { 'duration': RefScalar(30, 'time.duration', 'Base duration in seconds')
  , 'fps': Scalar(24, 'Frames per second')
  , 'height': Scalar(1920, 'Output height in pixels')
  , 'width': Scalar(1080, 'Output width in pixels')
  , 'frame_width': refscalar(1, 'time.frame_width')
  , 'spp': RefScalar(2000, 'camera.spp', 'Base samples per pixel')
  , 'skip': Scalar(0, 'Skip this many frames between each rendered frame')
  , 'filters': TypedList(prof_filters, default_filters,
                         'Ordered list of filters to apply')
  })

# Types recognized as independent units with a 'type' key
toplevels = dict(animation=anim, node=node, edge=edge, profile=profile)
