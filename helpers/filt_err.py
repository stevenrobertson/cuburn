import numpy as np

# The maximum number of coeffecients that will ever be retained on the device
FWIDTH = 15

# The number of points on either side of the center in one dimension
F2 = int(FWIDTH/2)

# The maximum size of any one coeffecient to be retained
COEFF_EPS = 0.0001

dists2d = np.fromfunction(lambda i, j: np.hypot(i-F2, j-F2), (FWIDTH, FWIDTH))
dists = dists2d.flatten()


# This translates to a cap on DE filter radius of 50. Even this fits very
# comfortably within the chosen COEFF_EPS.
MAX_SCALE = -3/25.

# When the scale is above this value, we'd be directly clamping to one bin
MIN_SCALE = np.log(0.0001)

# Everything above this scale uses one approximation; below, another.
SPLIT_SCALE = -1.1

# The upper endpoints are overshot by this proportion to reduce error
OVERSHOOT = 1.01

# No longer 'scale'-related, but we call it that anyway
scales_hi = np.linspace(SPLIT_SCALE, MAX_SCALE * OVERSHOOT, num=1000)
scales_lo = np.linspace(MIN_SCALE, SPLIT_SCALE * OVERSHOOT, num=1000)

def eval_scales(scales, name, nterms):
    # Calculate the filter sums at each coordinate
    sums = []
    for scale in scales:
        coeffs = np.exp(dists**2 * scale)
        # Note that this sum is the sum of all coordinates, though it should
        # actually be the result of the polynomial approximation. We could do
        # a feedback loop to improve accuracy, but I don't think the difference
        # is worth worrying about.
        sum = np.sum(coeffs)
        sums.append(1./np.sum(filter(lambda v: v / sum > COEFF_EPS, coeffs)))
    print 'Evaluating %s:' % name
    poly, resid, rank, sing, rcond = np.polyfit(scales, sums, nterms, full=True)
    print 'Fit for %s:' % name, poly, resid, rank, sing, rcond
    return sums, poly

import matplotlib.pyplot as plt

sums_hi, poly_hi = eval_scales(scales_hi, 'hi', 7)
sums_lo, poly_lo = eval_scales(scales_lo, 'lo', 7)

num_overshoots = len(filter(lambda v: v > MAX_SCALE, scales_hi))
scales_hi = scales_hi[num_overshoots:]
sums_hi = sums_hi[num_overshoots:]

num_overshoots = len(filter(lambda v: v > SPLIT_SCALE, scales_lo))
scales_lo = scales_lo[num_overshoots:]
sums_lo = sums_lo[num_overshoots:]

polyf_hi = np.float32(poly_hi)
vals_hi = np.polyval(polyf_hi, scales_hi)
polyf_lo = np.float32(poly_lo)
vals_lo = np.polyval(polyf_lo, scales_lo)

def print_filt(filts):
    print '    filtsum = %4.8ef;' % filts[0]
    for f in filts[1:]:
        print '    filtsum = filtsum * scale + % 16.8ef;' % f

print '\n\nFor your convenience:'
print '#define MIN_SCALE %.8gf' % MIN_SCALE
print '#define MAX_SCALE %.8gf' % MAX_SCALE
print 'if (scale < %gf) {' % SPLIT_SCALE
print_filt(polyf_lo)
print '} else {'
print_filt(polyf_hi)
print '}'

scales = np.concatenate([scales_lo, scales_hi])
sums = np.concatenate([sums_lo, sums_hi])
vals = np.concatenate([vals_lo, vals_hi])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(scales, sums)
ax.plot(scales, vals)
ax.set_xlabel('stdev')
ax.set_ylabel('filter sum')

ax = ax.twinx()
ax.plot(scales, [abs((s-v)/v) for s, v in zip(sums, vals)])
ax.set_ylabel('rel err')

plt.show()

