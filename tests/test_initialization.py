import unittest

from jax import random
from jax import numpy as jnp

from autostep import autohmc

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
        

if __name__ == '__main__':
    unittest.main()
    