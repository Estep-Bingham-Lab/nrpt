import unittest

from functools import partial
import jax
from jax import lax
from jax import random
from jax import numpy as jnp

from autostep import autohmc

from nrpt import initialization
from nrpt import sampling
from nrpt import toy_examples

class TestToyExamples(unittest.TestCase):

    def test_toy_examples(self):
        rng_key = random.key(123)

        ## Unidentifiable
        model, model_args, model_kwargs = toy_examples.toy_unid_example()
        kernel = autohmc.AutoMALA(model)
        pt_sampler = initialization.PT(
            kernel, 
            rng_key,
            n_replicas=4,
            model_args=model_args, 
            model_kwargs=model_kwargs
        )
        pt_sampler = sampling.run(pt_sampler)
        pt_state = pt_sampler.pt_state
        inv_temp_schedule = pt_state.replica_states.inv_temp[
            pt_state.chain_to_replica_idx
        ]
        true_logZs = jax.vmap(partial(toy_examples.toy_unid_exact_logZ, 100,50))(
            inv_temp_schedule
        )
        total_barrier = sampling.total_barrier(pt_state.stats.barrier_fit)
        self.assertTrue(
            jnp.allclose(pt_state.stats.logZ_fit.y, true_logZs, atol=0.4)
        )
        self.assertTrue(jnp.isclose(total_barrier, 1.38, rtol=0.05)) # long Pigeons run


if __name__ == '__main__':
    unittest.main()
    