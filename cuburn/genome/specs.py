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

author = (
  { 'name': String("Human-readable name of author")
  , 'user': String("Email or other unique identifier")
  , 'url': String("Website or other link provided by author")
  })

link = (
  { 'src': String("Origin node ID and temporal offset")
  , 'dst': String("Destination node ID and temporal offset")
  })

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
    , 'highlight_power': spline(-1, -1)
    , 'vibrance': scalespline()
    }
  , 'de':
    { 'radius': scalespline(11, d='Spatial filter radius in flam3 units')
    , 'minimum': scalespline(0, max=1, d='Proportional min radius')
    , 'curve': scalespline(0.6, d='Power of filter radius with density')
    }
  , 'haloclip': {}
  , 'smearclip': {'width': scalespline(0.7, d='Spatial stdev of filter')}
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

blend = (
  { 'duration': scalar(2, d='The base duration of the animation. This value '
        'affects the periodic extension of angular parameters (values are '
        'extended to minimize deviation from average velocity), and also '
        'sets the default value for `time.duration` in an animation.')
  , 'xform_sort': enum('weightflip weight natural color', 'weightflip')
  , 'xform_map': list_(list_(String('xfid'), d='A pair of src, dst IDs'))
  })

base = (
  { 'name': String("Human-readable name of this work")
  , 'base': String("DB path to base from which this doc inherits")
  , 'camera': camera
  , 'filters': filters
  , 'palette': list_(Palette())
  , 'xforms': map_(xform)
  , 'final_xform': xform
  , 'time': time
  })

# TODO
node = dict(base)
node.update(type='node', blend=blend, author=author)

# TODO
edge = dict(base)
edge.update(type='edge', author=author, blend=blend, link=link,
            xforms=dict(src=map_(xform), dst=map_(xform)))

anim = dict(base)
anim.update(type='animation', authors=list_(author), link=link)

default_filters = ['bilateral', 'logscale', 'smearclip']
# Yeah, now I'm just messing around.
prof_filters = dict([(fk, dict([(k, refscalar(1, '.'.join(['filters', fk, k])))
                           for k in fv])) for fk, fv in filters.items()])
# And here's a completely stupid hack to drag scale into the logscale filter
prof_filters['logscale']['scale'] = refscalar(1, 'camera.scale')

profile = (
  { 'duration': RefScalar(30, 'time.duration', 'Base duration in seconds')
  , 'fps': Scalar(24, 'Frames per second')
  , 'frame_width': refscalar(1, 'time.frame_width')

  , 'start': Scalar(None, 'First frame to render (1-indexed, inclusive)')
  , 'end': Scalar(None, 'Last frame to render (1-indexed, exclusive; '
                  'negative indexes from the end)')
  , 'skip': Scalar(0, 'Skip this many frames between each rendered frame')
  , 'shard': Scalar(0, 'Pack this many frames in each output file '
                    '(causing start, end, and skip to be ignored)')

  , 'height': Scalar(1920, 'Output height in pixels')
  , 'width': Scalar(1080, 'Output width in pixels')
  , 'spp': RefScalar(2000, 'camera.spp', 'Base samples per pixel')

  , 'filter_order': list_(enum(filters.keys()), default_filters)
  , 'filters': prof_filters

  # The other keys in the 'output' dictionary are format-specific and not
  # documented here.
  , 'output': {'type': enum('jpeg png tiff x264', 'jpeg')}
  })

# Types recognized as independent units with a 'type' key
toplevels = dict(animation=anim, node=node, edge=edge, profile=profile)
