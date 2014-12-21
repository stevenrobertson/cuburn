# Cuburn

This project is a fractal flame renderer, similar to [flam3](http://flam3.com),
built for CUDA GPUs. It also includes a new blending system, the ability to
perform spline-based animation of individual parameters over time, and new,
HDR-aware filtering mechanisms. The overall result is a distinctive, dreamlike
atmosphere — rendered up to one hundred times faster than CPU.

This project is licensed under the GPL version 3.

## Getting started

To use cuburn, you'll need the following dependencies:

  - A CUDA-compatible graphics card with SM 2.0 (GTX 400 series) or newer
  - A recent CUDA toolkit (at least v4.1) and drivers
  - [pycuda](http://mathema.tician.de/software/pycuda/)
  - [numpy](http://numpy.scipy.org/)
  - [scipy](http://scipy.org/)
  - [tempita](http://pythonpaste.org/tempita/)
  - Maybe some other stuff, I'll come back to check later

Perform a git checkout of cuburn, `cd` to cuburn's directory, and run

    python main.py -g input.flam3

... and your GPU will start rendering your fractal flame, saving the results to
the current home directory. `main.py` is the primary interface to cuburn, and
it includes built-in help; run `python main.py --help` to find out what it
supports.

Once the program runs, check out [the sample flock](https://github.com/stevenrobertson/cuburn-sample-flock)
for more details on the design of the JSON flame format and how to use it to
manage and render a collection of flames.

## Differences between flam3 and cuburn

### Gorgeous, fluid interpolation

Erik Reckase and Vitor Bosshard came up with a phenomenal new approach to
applying the typical rotational parameter interpolation used by flam3's
animation system, which served as a template for the current spline-based
system used by cuburn (which Erik also worked on). The result can literally
stop you in your tracks when you see it, and I'm thrilled to have been able to
implement their ideas.

### Everything is an animation

Cuburn is built for beautiful, fluid animations, and the interface is built
with that in mind. If you want a single image — no motion blur, no velocity
interpolation — you can pass the `--still` argument to any cuburn command that
accepts profile arguments.

### Temporally-aware, graphical, JSON-encoded flame representation

Cuburn can read flam3-style XML flame descriptions, but it really shines when
used with its native JSON dialect. This method of representing flames has
temporality in every expression, and was designed from the ground up for
nondestructive composition and blending, so that everything from individual
frames to composed sequences to entire flocks can be lovingly tended by a
community of artists and editors.

The format is also amenable to composition with less directed input, such as
that produced by a genetic algorithm, which can add variety and spark
inspiration without destroying the original intent or severing the "anchor
points" which tie a flock together.

It still needs a frontend to realize this humanely, so right now this might as
well be vaporware.

### Output profiles

Cuburn separates the description of the underlying mathematical system from the
particulars of rendering. Instead of containing information such as output
resolution within the flame XML, cuburn allows flames to specify the camera in
terms of IFS coordinates, and then applies an output profile to convert camera
coordinates to image-space coordinates for rendering.

Filtering parameters and sampling patterns are done the same way, and in
almost every case, parameters are continuously-valued. This means that the
same splines which allow animators complete control over the movement of the
frame can be used to describe its filtering relative to the other sheep in a
flock.

### Totally revised filtering

I sat down with a notebook and a laptop running Maxima, and came out with some
weird, inefficient shear-sampled hybrid directional-bilateral-gradient
nonsense filter and a new tone-mapping algorithm that intentionally adds
spatial distortion around clipped regions. Looks great, tho.
