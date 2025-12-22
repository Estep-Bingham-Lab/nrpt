import unittest

from jax import random

from automcmc import autohmc, autorwmh

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
    
    def test_thinning(self):
        rng_key = random.key(1)
        model, model_args, model_kwargs = toy_examples.eight_schools_example()
        n_rounds = 7

        # make a schedule of thinning values such that the biggest value is 
        # the maximum allowed 2**n_round.
        n_thinning_vals = 5
        min_thinning=2
        max_thinning=2**n_rounds
        d_thinning = (max_thinning-min_thinning)//(n_thinning_vals-1)
        all_thinnings = [
            -x for x in reversed(range(-max_thinning, -min_thinning, d_thinning))
        ]

        # loop thinnings
        for thinning in all_thinnings:
            print(f"thinning={thinning}")
            kernel = autorwmh.AutoRWMH(model)
            rng_key, mcmc_key = random.split(rng_key)
            pt_sampler = initialization.PT(
                kernel, 
                rng_key = mcmc_key,
                n_rounds = n_rounds,
                n_replicas = 3,
                n_refresh = 4,
                model_args=model_args,
                model_kwargs=model_kwargs,
                collect_samples=thinning
            )
            self.assertEqual(pt_sampler.thinning, thinning)
            self.assertEqual(
                pt_sampler.pt_state.samples['log_lik'].shape[0],
                sampling.n_scans_in_round(pt_sampler.n_rounds) // pt_sampler.thinning
            )
            pt_sampler = sampling.run(pt_sampler)
            if divmod(max_thinning, thinning)[-1] == 0:
                # when thinning divides the total number of scans in the last 
                # round, the samples should capture the last state
                pt_state = pt_sampler.pt_state
                self.assertEqual(
                    pt_state.samples['log_joint'][-1],
                    pt_state.replica_states.log_joint[pt_state.chain_to_replica_idx[-1]]
                )

if __name__ == '__main__':
    unittest.main()
    