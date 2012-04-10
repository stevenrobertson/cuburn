from schematypes import *
from variations import var_

affine = (
  { angle: spline(45, period=360)
  , spread: spline(45, period=180)
  # TODO: should these scale relative to magnitude?
  , off_x: spline()
  , off_y: spline()
  # TODO: this is probably an inappropriate scaling domain? Should one be
  # constructed specifically for magnitudes?
  , mag_x: spline(1)
  , mag_y: spline(1)
  })

xform = (
  { affine: affine
  , post: affine
  , color: spline(0, 0, 1)
  , color_speed: spline(0.5, 0, 1)
  , density: spline()
  , opacity: scalespline(max=1)
  , variations: cuburn.code.variations.params
  })

# Since the structure of the info element differs between anims, nodes and
# edges, we pull out some of the common elements here
author = String('Attribution in the form: "Name [<email>][, url]"')
name = String('A human-readable name for this entity')
src = String('The identifier of the source node')
dst = String('The identifier of the destination node')

filters = (
  { bilateral:
    { spatial_std: scalespline(d='Scale of profile spatial standard deviation')
    , color_std: scalespline(d='Scale of profile color standard deviation')
    , density_std: scalespline(d='Scale of profile density standard deviation')
    , density_pow: scalespline(d='Scale of profile density pre-blur exponent')
    , gradient: spline(1, d='Scale of profile gradient filter intensity '
                            '(can be negative)')
    }
  , colorclip:
    { bg_r: spline(0, 0, 1)
    , bg_g: spline(0, 0, 1)
    , bg_b: spline(0, 0, 1)
    , gamma: scalespline()
    , gamma_threshold: spline(0.01, 0, 1)
    , highlight_power: spline(-1, -1, 1)
    , vibrance: spline(1, 0, 1)
    }
  , de:
    { radius: scalespline(d='Scale of profile filter radius')
    , minimum: scalespline(0, d='Scale against adjusted DE radius of '
                                'minimum radius')
    , curve: scalespline(0.6, d='Absolute (unscaled) value of DE curve')
    }
               # TODO: absolute or relative?
  , logscale: {brightness: scalespline(4, d='Absolute log brightness')}
  })

anim = (
  { type: 'animation'
  , info: dict(authors=List(author), name=name, src=src, dst=dst)
  , camera:
    # Should center_{xy} be scaled relative to the 'scale' parameter, or is
    # that just too complicated for this representation?
    { center_x: spline()
    , center_y: spline()
    , density: scalespline()
    , dither_width: scalespline()
    , rotation: spline(period=360)
    , scale: scalespline()
    }
  , filters: filters
  , time:
    { duration:
    , frame_width: scalespline(d='Scale of profile temporal width per frame.')
    }
  , palettes: list_(Palette())
  , xforms: map(xform)
  , final_xform: xform
  })
