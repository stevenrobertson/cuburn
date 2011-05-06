import numpy as np

# The maximum number of coeffecients that will ever be retained on the device
FWIDTH = 15

# The number of points on either side of the center in one dimension
F2 = int(FWIDTH/2)

# The maximum size of any one coeffecient to be retained
COEFF_EPS = 0.0001

dists = np.fromfunction(lambda i, j: np.hypot(i-F2, j-F2), (FWIDTH, FWIDTH))
dists = dists.flatten()

# A flam3 estimator radius corresponds to a Gaussian filter with a standard
# deviation of 1/3 the radius. We choose 13 as an arbitrary upper bound for the
# max filter radius. Larger radii will work without
MAX_SD = 13 / 3.

# The minimum estimator radius is 1. In flam3, this is effectively no
# filtering, but since the cutoff structure is defined by COEFF_EPS in cuburn,
# we undershoot it a bit to make the polyfit behave better at high densities.
MIN_SD = 0.3

sds = np.logspace(np.log10(MIN_SD), np.log10(MAX_SD), num=100)

# Calculate the filter sums at each coordinate
sums = []
for sd in sds:
    coeffs = np.exp(dists**2 / (-2 * sd ** 2))
    sums.append(np.sum(filter(lambda v: v / np.sum(coeffs) > COEFF_EPS, coeffs)))
print sums

import matplotlib.pyplot as plt

poly, resid, rank, sing, rcond = np.polyfit(sds, sums, 4, full=True)
print poly, resid, rank, sing, rcond

polyf = np.float32(poly)
plt.plot(sds, sums)
plt.plot(sds, np.polyval(polyf, sds))
plt.show()

print np.polyval(poly, 1.1)

# TODO: calculate error more fully, verify all this logic
