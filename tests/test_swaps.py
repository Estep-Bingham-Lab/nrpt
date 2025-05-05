import unittest

from jax import random
from jax import numpy as jnp

from autostep import autohmc

from nrpt import initialization
from nrpt import sampling
from nrpt import toy_examples

from numpyro import infer

from tests import utils as testutils

class TestSwaps(unittest.TestCase):

    def test_swaps(self):
        rng_key = random.key(1)
        kernel = testutils.NoneExplorer()
        
        # check 2 scans give even(odd(initial state))
        pt_sampler = initialization.PT(kernel, rng_key)
        pt_sampler = sampling.pt_round(pt_sampler)
        self.assertTrue(jnp.all(
            pt_sampler.pt_state.replica_to_chain_idx == 
            pt_sampler.swap_group_actions[1][pt_sampler.swap_group_actions[0]]
        ))

        # check 2*n_replicas scans give the initial permutation
        pt_sampler = initialization.PT(kernel, rng_key)
        init_replica_to_chain_idx = pt_sampler.pt_state.replica_to_chain_idx
        pt_sampler = sampling.pt_round(pt_sampler, 2*len(init_replica_to_chain_idx))
        self.assertTrue(jnp.all(
            pt_sampler.pt_state.replica_to_chain_idx == 
            init_replica_to_chain_idx
        ))


if __name__ == '__main__':
    unittest.main()
    