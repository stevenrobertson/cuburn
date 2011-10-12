"""
Some simple operations on 2D affine matrices. These matrices are all stored
in row-major order, like C, instead of Fortran-style column-major storage.
"""

import numpy as np

_ident = np.matrix([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
_point = np.matrix([0,0,1], dtype=np.float64).T

def from_flam3(a):
    """Convert from flam3-format [3][2] arrays to an affine matrix."""
    return np.matrix([ [a[0][0], a[1][0], a[2][0]]
                     , [a[0][1], a[1][1], a[2][1]]
                     , [0, 0, 1]])

def scale(x, y):
    r = _ident.copy()
    r[0,0] = x
    r[1,1] = y
    return r

def translate(x, y):
    r = _ident.copy()
    r[0,2] = x
    r[1,2] = y
    return r

def rotOrigin(rad):
    r = _ident.copy()
    r[0,0] = r[1,1] = np.cos(rad)
    s = np.sin(rad)
    r[0,1] = -s
    r[1,0] = s
    return r

def rotate(rad, x, y):
    """Rotates around the given point (x, y)."""
    return translate(x, y) * rotOrigin(rad) * translate(-x, -y)

def apply(m, x, y):
    """Apply matrix to point, returning new point as a tuple. Extends point
    to homogeneous coordinates before applying. Mostly here as an example."""
    p = _point.copy()
    p[0,0] = x
    p[1,0] = y
    r = m * p
    return r[0], r[1]

