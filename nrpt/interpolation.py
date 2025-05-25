from collections import namedtuple

import jax
from jax import lax
from jax import numpy as jnp

# define a namedtuple container for a  cubic piece-wise interpolator
# note: the original response `y` is not strictly needed for interpolation
# but it's useful for other purposes.
PiecewiseCubicInterpolator = namedtuple(
    'PiecewiseCubicInterpolator', ['x', 'y', 'coeffs']
)

# dummy constructor for initialization
def empty_interpolator(n):
    return PiecewiseCubicInterpolator(
        jnp.zeros(n),
        jnp.zeros(n),
        jnp.zeros((n-1,4))
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
    return PiecewiseCubicInterpolator(x, y, coeffs)

def interpolate(interpolator, x_new):
    """
    Evaluate a piece-wise polynomial interpolator on a set of points `x_new`.
    IMPORTANT: fails silently when `x_new` is outside the range of the data.
    """
    assert jnp.ndim(x_new) == 1
    x, _, coeffs = interpolator
    intervals = jnp.searchsorted(x, x_new, 'right') - 1
    x_inds = lax.max(0, lax.min(intervals, len(x) - 2))
    return jax.vmap(jnp.polyval)(coeffs[intervals], x_new - x[x_inds])

###############################################################################
# Akima recipe for a piecewise cubic interpolator
# Simplified (only 1D args) adaptation from SciPy implementation (BSD 3):
# https://github.com/scipy/scipy/blob/c41054dfd5f3779bc350d9896191df10efe6bd91/scipy/interpolate/_cubic.py#L378
###############################################################################

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
        smoothed_slope_ok = (f1 * m[1:-2] + f2 * m[2:-1]) / f12
        
        # if m1 == m2 != m3 == m4, the slope at the breakpoint is not
        # defined. This is the fill value:
        smoothed_slope_fill = .5 * (m[3:] + m[:-3])

        # These are the masks of where the slope at breakpoint is defined:
        ind = f12 > 1e-9 * jnp.max(f12, initial=-jnp.inf)
        smoothed_slope = lax.select(ind, smoothed_slope_ok, smoothed_slope_fill)

    return build_piecewiese_cubic_poly(x, dx, y, dy_dx, smoothed_slope)

###############################################################################
# PCHIP monotonic piecewise cubic interpolator
# Simplified (only 1D args) adaptation from SciPy implementation (BSD 3):
# https://github.com/scipy/scipy/blob/c41054dfd5f3779bc350d9896191df10efe6bd91/scipy/interpolate/_cubic.py#L160
###############################################################################

def pchip_edge_case(h0, h1, m0, m1):
    # one-sided three-point estimate for the derivative
    d = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)
    mask = jnp.sign(d) != jnp.sign(m0)
    mask2 = (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > 3.*jnp.abs(m0))
    return lax.select(
        mask,
        jnp.zeros_like(d),
        lax.select(mask2, 3.*m0, d)
    )

def pchip_find_derivatives(x, y):
    # Let m_k be the slope of the kth segment (between k and k+1)
    # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
    # else use weighted harmonic mean:
    #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
    #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
    #   where h_k is the spacing between x_k and x_{k+1}
    assert jnp.shape(x) == jnp.shape(y)
    assert jnp.ndim(x) == 1

    # these are length n-1
    hk = jnp.diff(x)
    mk = jnp.diff(y) / hk

    if y.shape[0] == 2:
        # edge case: only have two points, use linear interpolation
        dk = jnp.full_like(y, mk)
        return dk

    smk = jnp.sign(mk)
    condition = jnp.logical_or(  # this is length n-2
        smk[1:] != smk[:-1],
        jnp.logical_or(mk[1:] == 0, mk[:-1] == 0)
    )

    # these are length n-2
    w1 = 2*hk[1:] + hk[:-1]
    w2 = hk[1:] + 2*hk[:-1]

    # values where division by zero occurs will be excluded
    # by 'condition'
    whmean = (w1/mk[:-1] + w2/mk[1:]) / (w1 + w2)
    dk_interior = lax.select(
        condition, 
        jnp.zeros_like(y, shape=condition.shape), 
        jnp.reciprocal(whmean)
    )

    # special case endpoints, as suggested in
    # Cleve Moler, Numerical Computing with MATLAB, Chap 3.6 (pchiptx.m)
    dk_start = pchip_edge_case(hk[0], hk[1], mk[0], mk[1])
    dk_end   = pchip_edge_case(hk[-1], hk[-2], mk[-1], mk[-2])
    dk = jnp.insert(
        dk_interior,
        jnp.array([0, len(dk_interior)+1]),
        jnp.array([dk_start, dk_end])
    )
    assert dk.shape == jnp.shape(y)
    return hk, mk, dk

def build_pchip_interpolator(x, y):
    dx, dy_dx, smoothed_slope = pchip_find_derivatives(x, y)
    return build_piecewiese_cubic_poly(x, dx, y, dy_dx, smoothed_slope)
