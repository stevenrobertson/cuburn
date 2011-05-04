"""
Some simple operations on 2D affine matrices. These matrices are all stored
in row-major order, like C, instead of Fortran-style column-major storage.
"""

import numpy as np

def from_flam3(a):
    """Convert from flam3-format [3][2] arrays to an affine matrix."""
    return np.matrix([ [a[0][0], a[1][0], a[2][0]]
                     , [a[0][1], a[1][1], a[2][1]]
                     , [0, 0, 1]])

def scale(x, y):
    return np.matrix([[x,0,0], [0,y,0], [0,0,1]])

def translate(x, y):
    return np.matrix([[1,0,x], [0,1,y], [0,0,1]])

def rotOrigin(rad):
    c = np.cos(rad)
    s = np.sin(rad)
    return np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotate(rad, x, y):
    """Rotates around the given point (x, y)."""
    return translate(x, y) * rotOrigin(rad) * translate(-x, -y)

def apply(m, x, y):
    """Apply matrix to point, returning new point as a tuple. Extends point
    to homogeneous coordinates before applying. Mostly here as an example."""
    r = m * np.matrix([x, y, 1]).T
    return r[0], r[1]

