import unittest

from functools import partial

import math

import jax
from jax import random
from jax import numpy as jnp

from autostep import autohmc, tempering

from nrpt import initialization, sampling, statistics, toy_examples

from tests import utils as testutils

class TestToyExamples(unittest.TestCase):

    def test_toy_examples(self):
        rng_key = random.key(123)

        ## Unidentifiable
        n_heads, n_flips = 50000,100000
        model, model_args, model_kwargs = toy_examples.toy_unid_example(n_heads, n_flips)
        true_barrier = 3.25 # long run
        init_params = None
        kernel_model = autohmc.AutoMALA(model)
        logprior_and_loglik = partial(
            tempering.model_logprior_and_loglik, 
            model, 
            model_args, 
            model_kwargs
        )
        kernel_no_model = autohmc.AutoMALA(
            model, logprior_and_loglik=logprior_and_loglik
        )
        for kernel in (kernel_model, kernel_no_model):
            pt_sampler = initialization.PT(
                kernel, 
                rng_key,
                n_replicas=math.ceil(2*true_barrier),
                init_params=init_params,
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
            print(true_logZs)
            print(pt_state.stats.logZ_fit.y)
            total_barrier = sampling.total_barrier(pt_state.stats.barrier_fit)
            self.assertTrue(
                jnp.allclose(pt_state.stats.logZ_fit.y, true_logZs, atol=1., rtol=0.4)
            )
            self.assertTrue(jnp.isclose(total_barrier, true_barrier, rtol=0.2))

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
            if kernel.model is None:
                # samples['p'] is available only when model because of non-trivial
                # kernel.postprocess_fn to re-constrain samples and compute deterministics
                self.assertTrue(jnp.all(samples['p'] == samples['p1']*samples['p2']))
                self.assertTrue(jnp.allclose(0.5, samples['p'], rtol=0.1))

            # update init_params
            replica_at_target_idx = pt_state.chain_to_replica_idx[-1]
            init_params = jax.tree.map(
                lambda u: u[replica_at_target_idx],
                pt_state.replica_states.x
            )

if __name__ == '__main__':
    unittest.main()
    