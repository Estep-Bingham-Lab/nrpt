from collections import namedtuple

import jax
from jax import random
from jax import lax
from jax import numpy as jnp

# define a namedtuple container for a  cubic piece-wise interpolator
PiecewiseCubicInterpolator = namedtuple(
    'PiecewiseCubicInterpolator', ['x', 'coeffs']
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
    return PiecewiseCubicInterpolator(x, coeffs)

# Akima recipe for a piecewise cubic interpolator
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

def interpolate(interpolator, x_new):
    """
    Evaluate a piece-wise polynomial interpolator on a set of points `x_new`. 
    """
    assert jnp.ndim(x_new) == 1
    x, coeffs = interpolator
    intervals = jnp.searchsorted(x, x_new, 'right') - 1
    x_inds = lax.min(intervals, len(x) - 2)
    return jax.vmap(jnp.polyval)(coeffs[intervals], x_new - x[x_inds])
