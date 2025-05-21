import unittest

import numpy as np
from scipy import interpolate

from jax import numpy as jnp

from nrpt import interpolation

class TestInterpolation(unittest.TestCase):

    def test_interpolation(self):
        # Fritsch-Carlson RPN 15A data
        # this breaks Akima, as it is not monotonicity preserving
        # note: use np arrays to be able to compare with scipy's answer
        x = np.array([7.99, 8.09, 8.19, 8.7, 9.2, 10., 12., 15., 20.])
        y = np.array([
            0.,
            2.76429E-5,
            4.37498E-2,
            0.169183,
            0.469428,
            0.943740,
            0.998636,
            0.999919,
            0.999994
        ])
        x_new = jnp.linspace(x[0],x[-1],500)
        interpolator = interpolation.build_pchip_interpolator(x, y)
        y_int = interpolation.interpolate(interpolator, x_new)
        self.assertTrue(jnp.all(jnp.diff(y_int) >= 0)) # check non-decreasing
        
        # compare with scipy
        P = interpolate.PchipInterpolator(x, y)
        self.assertTrue(
            jnp.abs(P.c - interpolator.coeffs.swapaxes(0,1)).max() < 1e-2
        )
        self.assertTrue(jnp.abs(P(x_new) - y_int).max() < 1e-6)

        # check Akima breaks monotonicity but is consistent with scipy
        interpolator = interpolation.build_akima_interpolator(x, y)
        y_int = interpolation.interpolate(interpolator, x_new)
        self.assertFalse(jnp.all(jnp.diff(y_int) >= 0))
        P = interpolate.Akima1DInterpolator(x, y)
        self.assertTrue(
            jnp.abs(P.c - interpolator.coeffs.swapaxes(0,1)).max() < 1e-2
        )
        self.assertTrue(
            jnp.abs(P(x_new)[1:] - y_int[1:]).max() < 1e-6 # avoid a nan created in scipy's answer
        )

if __name__ == '__main__':
    unittest.main()
    