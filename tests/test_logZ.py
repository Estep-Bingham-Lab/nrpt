import unittest

import jax
from jax import numpy as jnp
from jax import random

from nrpt import logZ

def online_update_estimates(current_round_dlogZ_estimates,dbeta,log_liks):
    return jax.lax.scan(
        lambda carry, l: (logZ.update_estimates(carry, dbeta, l), None),
        current_round_dlogZ_estimates,
        log_liks
    )[0]

class TestLogZ(unittest.TestCase):

    def test_online_dlogZ(self):
        # test the function `logZ.update_estimates`
        # simulate a PT run with 1-step path with ref N(0,1) and target N(0,mu)
        # use exact logpdfs so true logZ = dlogZ = 0.
        lo_key, hi_key = random.split(random.key(254))
        mu = 1.
        tru_logZ = 0.
        N = 2**16
        x_lo = random.normal(lo_key, N)
        x_hi = mu + random.normal(hi_key, N)
        loglik = lambda x: jax.scipy.stats.norm.logpdf(x,loc=mu)-jax.scipy.stats.norm.logpdf(x)
        log_liks = jnp.array([loglik(x_lo), loglik(x_hi)]).swapaxes(0,1)
        current_round_dlogZ_estimates = logZ.init_estimates(2)
        dbeta = jnp.array([1.])
        out = online_update_estimates(current_round_dlogZ_estimates,dbeta,log_liks)
        self.assertAlmostEqual(
            out[0,0], jax.scipy.special.logsumexp( log_liks[:,0]), delta=1e-4 # dbeta == 1
        )
        self.assertAlmostEqual(
            out[0,1], jax.scipy.special.logsumexp(-log_liks[:,1]), delta=1e-4 # dbeta == 1
        )
        self.assertAlmostEqual(
            tru_logZ, out[0,0] - jnp.log(N), delta=0.02
        )
        self.assertAlmostEqual(
            tru_logZ, jnp.log(N) - out[0,1], delta=0.02
        )
        self.assertAlmostEqual(
            tru_logZ, 0.5*(out[0,0]-out[0,1]), delta=0.01
        )

        # same exercise but with an unnormalized loglik
        # x^2 + y = (x-u)^2 => y = (x-u)^2 - x^2 = (x-u-x)(x-u+x) = -u(2x-u) = u^2 - 2xu
        # therefore
        # exp(-0.5(x-u)^2) = exp(-0.5(x^2 + y)) = exp(-0.5x^2)exp(ux)exp(-u^2/2)
        # <=>
        # (2pi)^{-1/2}exp(-0.5(x-u)^2) = [(2pi)^{-1/2}exp(-0.5x^2)][exp(ux) / exp(u^2/2)]
        #            ^ normalized                   ^ normalized    ^[unnormalized / normalization]
        # So Z = exp(u^2/2) => logZ = u^2/2.
        mu = 3/4
        tru_logZ = 9/32
        x_hi = mu + random.normal(random.key(4), N)
        loglik = lambda x: mu*x
        log_liks = jnp.array([loglik(x_lo), loglik(x_hi)]).swapaxes(0,1)
        current_round_dlogZ_estimates = logZ.init_estimates(2)
        out = online_update_estimates(current_round_dlogZ_estimates,dbeta,log_liks)
        self.assertAlmostEqual(
            tru_logZ, out[0,0] - jnp.log(N), delta=0.01
        )
        self.assertAlmostEqual(
            tru_logZ, jnp.log(N) - out[0,1], delta=0.01
        )
        self.assertAlmostEqual(
            tru_logZ, 0.5*(out[0,0]-out[0,1]), delta=0.005
        )
        
if __name__ == '__main__':
    unittest.main()
    