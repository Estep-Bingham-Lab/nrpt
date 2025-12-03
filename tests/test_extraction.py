import unittest

from jax import random

from automcmc import autohmc

from nrpt import initialization, sampling, toy_examples

class TestExtraction(unittest.TestCase):

    def test_excluded_latent_vars(self):
        rng_key = random.key(123)
        model, model_args, model_kwargs = toy_examples.toy_unid_example()
        kernel = autohmc.AutoMALA(model)
        pt_sampler = initialization.PT(
            kernel, 
            rng_key = rng_key,
            model_args=model_args, 
            model_kwargs=model_kwargs,
            excluded_latent_vars={"p1", "p2"}
        )
        pt_sampler = sampling.run(pt_sampler)
        samples = pt_sampler.pt_state.samples
        self.assertNotIn("p1", samples)
        self.assertNotIn("p2", samples)

if __name__ == '__main__':
    unittest.main()
    