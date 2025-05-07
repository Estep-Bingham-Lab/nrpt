# TODO: FINISH EVALUATOR
from collections import namedtuple

import jax
from jax import random
from jax import lax
from jax import numpy as jnp
import scipy.interpolate 

# define a namedtuple container for a  cubic piece-wise interpolator
PiecewiseCubicInterpolator = namedtuple(
    'PiecewiseCubicInterpolator',
    [
        'x', 'dx', 'y', 'dy_dx', 'smoothed_slope', 'coeffs'
    ]
)

# constructor
def build_piecewiese_cubic_poly(x, dx, y, dy_dx, smoothed_slope):
    slope_diff = (smoothed_slope[:-1] + smoothed_slope[1:] - 2 * dy_dx) / dx
    coeffs = jnp.array(
        [
            slope_diff / dx,
            (dy_dx - smoothed_slope[:-1]) / dx - slope_diff,
            smoothed_slope[:-1],
            y[:-1]
        ]
    ).swapaxes(1,0)
    assert coeffs.shape == (len(dx), 4)
    return PiecewiseCubicInterpolator(x, dx, y, dy_dx, smoothed_slope, coeffs)

# Akima recipe for a monotone piecewise cubic interpolator
# Simplified (only 1D args) adaptation from SciPy implementation (BSD 3):
# https://github.com/scipy/scipy/blob/c41054dfd5f3779bc350d9896191df10efe6bd91/scipy/interpolate/_cubic.py#L378
def build_akima_interpolator(x, y):
    # Original implementation in MATLAB by N. Shamsundar (BSD licensed), see
    # https://www.mathworks.com/matlabcentral/fileexchange/1814-akima-interpolation
    assert jnp.ndim(x) == 1 and jnp.ndim(y) == 1 and len(x) == len(y)
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    dy_dx = dy / dx

    if y.shape[0] == 2:
        # edge case: only have two points, use linear interpolation
        smoothed_slope = dy_dx
    else:
        # add two additional points on the left and two more on the right
        # NB: modify implementation to avoid in-place computation using a single 
        # jnp.insert statement. In the original scipy
        #   m = [m[0], m[1], dy_dx, m[-2], m[-1]]
        # => for all i>1,
        #    m[ i] = dy_dx[ i-2]
        #    m[-i] = dy_dx[-i+2] 
        m_one  = 2. * dy_dx[0] - dy_dx[1]   # 2. * m[2] - m[3]
        m_zero = 2. * m_one - dy_dx[0]      # 2. * m[1] - m[2]
        m_pen  = 2. * dy_dx[-1] - dy_dx[-2] # 2. * m[-3] - m[-4]
        m_end  = 2. * m_pen - dy_dx[-1]     # 2. * m[-2] - m[-3]
        m = jnp.insert(
            dy_dx,
            jnp.array([0, 0, len(dy_dx), len(dy_dx)+1]),
            jnp.array([m_zero, m_one, m_pen, m_end])
        )
        assert len(m) == len(dy_dx) + 4
        
        # get the denominator of the slope
        dm = jnp.abs(jnp.diff(m))
        
        # NB: Use default method
        # pm = jnp.abs(m[1:] + m[:-1])
        # f1 = dm[2:] + 0.5 * pm[2:]
        # f2 = dm[:-2] + 0.5 * pm[:-2]
        f1 = dm[2:]
        f2 = dm[:-2]
        f12 = f1 + f2

        # NB: there may be undefined terms here. To avoid the in-place fix in the
        # scipy implementation, just compute original and fixed and then choose 
        # using `jnp.where`
        smoothed_slope_ok = (f1 * m[1:-2] + f2 * m[2:-1]) / f12
        
        # if m1 == m2 != m3 == m4, the slope at the breakpoint is not
        # defined. This is the fill value:
        smoothed_slope_fill = .5 * (m[3:] + m[:-3])

        # These are the mask of where the slope at breakpoint is defined:
        ind = f12 > 1e-9 * jnp.max(f12, initial=-jnp.inf)
        smoothed_slope = jnp.where(ind, smoothed_slope_ok, smoothed_slope_fill)

    return build_piecewiese_cubic_poly(x, dx, y, dy_dx, smoothed_slope)

x_key, y_key = random.split(random.key(0))
n_dx = 9
dx = random.uniform(x_key, n_dx)
x = jnp.insert(dx.cumsum(), 0, 0)
y = -lax.expm1(-x)
interpolator = build_akima_interpolator(x,y)

# test set
x_new = jnp.linspace(0,x[-1],5)
y_new = -lax.expm1(-x_new)

intervals = jnp.searchsorted(interpolator.x, x_new,'right')-1
x_inds = lax.min(intervals, len(dx)-1)
jax.vmap(jnp.polyval)(interpolator.coeffs[intervals], x_new-x[x_inds])

pend = interpolator.coeffs[2]
xend = x_new[1]-x[intervals[1]]

jnp.dot(pend, jnp.array([xend**3, xend**2, xend, 1]))
import scipy
import numpy as np

scipy_int = scipy.interpolate.Akima1DInterpolator(np.array(x), np.array(y))
jnp.abs(interpolator.coeffs - np.swapaxes(scipy_int.c, 1, 0)).max() < 1e-5
scipy_int(np.array(x_new))


# determine slopes between breakpoints
m = np.empty((x.size + 3, ) + y.shape[1:])
dx = dx[(slice(None), ) + (None, ) * (y.ndim - 1)]
m[2:-2] = np.diff(y, axis=0) / dx

# add two additional points on the left ...
m[1] = 2. * m[2] - m[3]
m[0] = 2. * m[1] - m[2]
# ... and on the right
m[-2] = 2. * m[-3] - m[-4]
m[-1] = 2. * m[-2] - m[-3]

# if m1 == m2 != m3 == m4, the slope at the breakpoint is not
# defined. This is the fill value:
t = .5 * (m[3:] + m[:-3])
# get the denominator of the slope t
dm = np.abs(np.diff(m, axis=0))
pm = np.abs(m[1:] + m[:-1])
f1 = dm[2:] + 0.5 * pm[2:]
f2 = dm[:-2] + 0.5 * pm[:-2]
f12 = f1 + f2
# These are the mask of where the slope at breakpoint is defined:
ind = np.nonzero(f12 > 1e-9 * np.max(f12, initial=-np.inf))
x_ind, y_ind = ind[0], ind[1:]
# Set the slope at breakpoint
t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] +
        f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]
len(m)
interpolator.smoothed_slope
m[2:-2] == interpolator.dy_dx