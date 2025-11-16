import unittest

from jax import random
from jax import numpy as jnp

from automcmc import autohmc

from nrpt import initialization, toy_examples


class TestInitialization(unittest.TestCase):

    def test_invalid_init_params_crash(self):
        model, model_args, model_kwargs = toy_examples.toy_unid_example()
        init_params = {'p1': jnp.float32(-1), 'p2': jnp.float32(jnp.nan)}
        kernel = autohmc.AutoMALA(model)
        with self.assertRaises(RuntimeError):
            initialization.PT(
                kernel, 
                rng_key = random.key(123),
                init_params = init_params,
                model_args=model_args, 
                model_kwargs=model_kwargs
            )
    
    def test_initial_schedule(self):
        model, model_args, model_kwargs = toy_examples.toy_unid_example()

        # try passing broken initial schedule 
        kernel = autohmc.AutoMALA(model)
        with self.assertRaises(AssertionError):
            initialization.PT(
                kernel, 
                rng_key = random.key(123),
                model_args=model_args, 
                model_kwargs=model_kwargs,
                initial_schedule=jnp.arange(75)
            )

        # check log-linear property
        kernel = autohmc.AutoMALA(model)
        pt_sampler = initialization.PT(
            kernel, 
            rng_key = random.key(123),
            model_args=model_args, 
            model_kwargs=model_kwargs,
            initial_schedule="log"
        )
        inv_temp_schedule = pt_sampler.pt_state.replica_states.inv_temp
        self.assertAlmostEqual(
            jnp.diff(jnp.log2(inv_temp_schedule[1:])).var(),
            0
        )
        

if __name__ == '__main__':
    unittest.main()
    