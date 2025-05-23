import unittest

import numpy as np
from scipy import interpolate

import jax
from jax import numpy as jnp
from jax import random

from nrpt import logZ

class TestLogZ(unittest.TestCase):

    def test_online_dlogZ(self):
        # test the function `logZ.update_estimates`
        # simulate a PT run with 1-step path with ref N(0,1) and target N(0,mu)
        # use exact logpdfs so true logZ = dlogZ = 0.
        lo_key, hi_key = random.split(random.key(254))
        mu = 1.
        N = 2**16
        x_lo = random.normal(lo_key, N)
        x_hi = mu + random.normal(hi_key, N)
        loglik = lambda x: jax.scipy.stats.norm.logpdf(x,loc=mu)-jax.scipy.stats.norm.logpdf(x)
        log_liks = jnp.array([loglik(x_lo), loglik(x_hi)]).swapaxes(0,1)
        current_round_dlogZ_estimates = logZ.init_estimates(2)
        dbeta = jnp.array([1.])
        out = jax.lax.scan(
            lambda carry, l: (logZ.update_estimates(carry, dbeta, l), None),
            current_round_dlogZ_estimates,
            log_liks
        )[0]
        self.assertTrue(
            jnp.isclose(
                out[0,0],
                jax.scipy.special.logsumexp(dbeta*log_liks[:,0])
            )
        )
        self.assertTrue(
            jnp.isclose(
                out[0,1],
                jax.scipy.special.logsumexp(-dbeta*log_liks[:,1])
            )
        )
        self.assertTrue(
            jnp.isclose(0., 0.5*(out[0,0]-out[0,1]), atol=1e-2)
        )

if __name__ == '__main__':
    unittest.main()
    