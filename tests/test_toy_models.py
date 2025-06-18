import unittest

from functools import partial

import math

import jax
from jax import random
from jax import numpy as jnp

from autostep import autohmc

from nrpt import initialization
from nrpt import sampling
from nrpt import statistics
from nrpt import toy_examples

from tests import utils as testutils

class TestToyExamples(unittest.TestCase):

    def test_toy_examples(self):
        rng_key = random.key(123)

        ## Unidentifiable
        n_heads, n_flips = 50000,100000
        model, model_args, model_kwargs = toy_examples.toy_unid_example(n_heads, n_flips)
        true_barrier = 3.25 # long run
        kernel = autohmc.AutoMALA(model)
        pt_sampler = initialization.PT(
            kernel, 
            rng_key,
            n_replicas=math.ceil(2*true_barrier),
            n_rounds = 10,
            model_args=model_args,
            model_kwargs=model_kwargs
        )
        pt_sampler = sampling.run(pt_sampler)
        pt_state = pt_sampler.pt_state
        
        # check loglik ac1 are in the correct range 
        # (only true when estimator has stabilized)
        ll_acs = statistics.loglik_autocors(pt_state)
        self.assertTrue(jnp.all(jnp.logical_and(ll_acs >= -1, ll_acs <= 1)))

        # check logZ and barrier estimates
        inv_temp_schedule = pt_state.replica_states.inv_temp[
            pt_state.chain_to_replica_idx
        ]
        vmapped_fn = partial(toy_examples.toy_unid_exact_logZ, n_flips, n_heads)
        true_logZs = jax.vmap(vmapped_fn)(inv_temp_schedule)
        total_barrier = sampling.total_barrier(pt_state.stats.barrier_fit)
        self.assertTrue(
            jnp.allclose(pt_state.stats.logZ_fit.y, true_logZs, atol=0.15, rtol=0.15)
        )
        self.assertTrue(jnp.isclose(total_barrier, true_barrier, rtol=0.15))

        # check base step size decreases with inv_temp
        self.assertTrue(testutils.is_increasing(
            -pt_state.replica_states.base_step_size[pt_state.chain_to_replica_idx[1:]]
        ))

        # check samples
        samples = pt_state.samples
        total_samples = sampling.n_scans_in_round(pt_sampler.n_rounds)
        self.assertTrue(
            jax.tree.all(
                jax.tree.map(lambda x: x.shape[0] == total_samples, samples)
            )
        )
        self.assertTrue(jnp.all(samples['p'] == samples['p1']*samples['p2']))
        self.assertTrue(jnp.allclose(0.5, samples['p'], rtol=0.1))


if __name__ == '__main__':
    unittest.main()
    