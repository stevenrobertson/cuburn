import numpy as np

# The maximum number of coeffecients that will ever be retained on the device
FWIDTH = 15

# The number of points on either side of the center in one dimension
F2 = int(FWIDTH/2)

# The maximum size of any one coeffecient to be retained
COEFF_EPS = 0.0001

dists2d = np.fromfunction(lambda i, j: np.hypot(i-F2, j-F2), (FWIDTH, FWIDTH))
dists = dists2d.flatten()

# A flam3 estimator radius corresponds to a Gaussian filter with a standard
# deviation of 1/3 the radius. We choose 13 as an arbitrary upper bound for the
# max filter radius. The filter should reject larger radii.
MAX_SD = 13 / 3.

# The minimum estimator radius can be set as low as 0, but below a certain
# radius only one coeffecient is retained. Since things get unstable near 0,
# we explicitly set a minimum threshold below which no coeffecients are
# retained.
MIN_SD = np.sqrt(-1 / (2 * np.log(COEFF_EPS)))

# Using two predicated three-term approximations is much more accurate than
# using a very large number of terms, due to nonlinear behavior at low SD.
# Everything above this SD uses one approximation; below, another.
SPLIT_SD = 0.75

# The lower endpoints are undershot by this proportion to reduce error
UNDERSHOOT = 0.98

sds_hi = np.linspace(SPLIT_SD * UNDERSHOOT, MAX_SD, num=1000)
sds_lo = np.linspace(MIN_SD * UNDERSHOOT, SPLIT_SD, num=1000)

print 'At MIN_SD = %g, these are the coeffs:' % MIN_SD
print np.exp(dists2d**2 / (-2 * MIN_SD ** 2))

def eval_sds(sds, name, nterms):
    # Calculate the filter sums at each coordinate
    sums = []
    for sd in sds:
        coeffs = np.exp(dists**2 / (-2 * sd ** 2))
        # Note that this sum is the sum of all coordinates, though it should
        # actually be the result of the polynomial approximation. We could do
        # a feedback loop to improve accuracy, but I don't think the difference
        # is worth worrying about.
        sum = np.sum(coeffs)
        sums.append(np.sum(filter(lambda v: v / sum > COEFF_EPS, coeffs)))
    print 'Evaluating %s:' % name
    poly, resid, rank, sing, rcond = np.polyfit(sds, sums, nterms, full=True)
    print 'Fit for %s:' % name, poly, resid, rank, sing, rcond
    return sums, poly

import matplotlib.pyplot as plt

sums_hi, poly_hi = eval_sds(sds_hi, 'hi', 8)
sums_lo, poly_lo = eval_sds(sds_lo, 'lo', 7)

num_undershoots = len(filter(lambda v: v < SPLIT_SD, sds_hi))
sds_hi = sds_hi[num_undershoots:]
sums_hi = sums_hi[num_undershoots:]

num_undershoots = len(filter(lambda v: v < MIN_SD, sds_lo))
sds_lo = sds_lo[num_undershoots:]
sums_lo = sums_lo[num_undershoots:]

polyf_hi = np.float32(poly_hi)
vals_hi = np.polyval(polyf_hi, sds_hi)
polyf_lo = np.float32(poly_lo)
vals_lo = np.polyval(polyf_lo, sds_lo)

def print_filt(filts):
    print '    filtsum = %4.8ff;' % filts[0]
    for f in filts[1:]:
        print '    filtsum = filtsum * sd + % 16.8ff;' % f

print '\n\nFor your convenience:'
print '#define MIN_SD %.8f' % MIN_SD
print '#define MAX_SD %.8f' % MAX_SD
print 'if (sd < %g) {' % SPLIT_SD
print_filt(polyf_lo)
print '} else {'
print_filt(polyf_hi)
print '}'

sds = np.concatenate([sds_lo, sds_hi])
sums = np.concatenate([sums_lo, sums_hi])
vals = np.concatenate([vals_lo, vals_hi])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(sds, sums)
ax.plot(sds, vals)
ax.set_xlabel('stdev')
ax.set_ylabel('filter sum')

ax = ax.twinx()
ax.plot(sds, [abs((s-v)/v) for s, v in zip(sums, vals)])
ax.set_ylabel('rel err')

plt.show()

