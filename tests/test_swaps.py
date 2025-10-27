import unittest

from jax import random
from jax import numpy as jnp

from nrpt import initialization
from nrpt import sampling
from nrpt import swaps
from nrpt import utils

from tests import utils as testutils

def check_maps_inv_temps(pt_state):
    a = replica_chain_maps_agree(
        pt_state.replica_to_chain_idx, pt_state.chain_to_replica_idx
    )
    b = chain_to_replica_map_sorts_inv_temps(
        pt_state.replica_states.inv_temp, pt_state.chain_to_replica_idx
    )
    return a and b

def replica_chain_maps_agree(replica_to_chain_idx, chain_to_replica_idx):
    return jnp.all(
        replica_to_chain_idx[chain_to_replica_idx] ==
        jnp.arange(len(replica_to_chain_idx))
    )

def chain_to_replica_map_sorts_inv_temps(inv_temps, chain_to_replica_idx):
    return utils.is_increasing(inv_temps[chain_to_replica_idx])

class TestSwaps(unittest.TestCase):

    def test_swaps(self):
        rng_key = random.key(1)
        
        #######################################################################
        ## check predictable patterns for swaps that are always accepted
        #######################################################################

        # trivial kernel simulating targeting a joint distribution with 
        # log_lik= constant, so that swaps are always accepted
        kernel = testutils.NoneExplorer()
        
        # check 1 round gives even(odd(initial state))
        rng_key, run_key = random.split(rng_key)
        pt_sampler = initialization.PT(
            kernel, run_key, n_rounds=1, collect_samples=False
        )
        init_chain_to_replica_idx = pt_sampler.pt_state.chain_to_replica_idx
        pt_sampler = sampling.run(pt_sampler)
        pt_state = pt_sampler.pt_state
        self.assertEqual(pt_state.stats.round_idx, 2)
        self.assertEqual(pt_state.stats.scan_idx, 1)
        self.assertTrue(check_maps_inv_temps(pt_state))
        self.assertTrue(jnp.all(
            pt_state.chain_to_replica_idx == 
            init_chain_to_replica_idx[
                pt_sampler.swap_group_actions[1][pt_sampler.swap_group_actions[0]]
            ]
        ))

        # check `n_replicas` scans reverse the initial permutation
        rng_key, run_key = random.split(rng_key)
        pt_sampler = initialization.PT(
            kernel, run_key, n_replicas=6, n_rounds=2, collect_samples=False
        )
        init_chain_to_replica_idx = pt_sampler.pt_state.chain_to_replica_idx
        pt_sampler = sampling.run(pt_sampler) # 2 rounds == 2+4=6 scans ==> reverse init array
        pt_state = pt_sampler.pt_state
        self.assertEqual(pt_state.stats.round_idx, 3)
        self.assertEqual(pt_state.stats.scan_idx, 1)
        self.assertTrue(check_maps_inv_temps(pt_state))
        self.assertTrue(jnp.all(
            pt_state.chain_to_replica_idx == jnp.flip(init_chain_to_replica_idx)
        ))

        # check 2*n_replicas scans return to the initial permutation
        rng_key, run_key = random.split(rng_key)
        pt_sampler = initialization.PT(
            kernel, run_key, n_replicas=7, n_rounds=3, collect_samples=False
        )
        init_chain_to_replica_idx = pt_sampler.pt_state.chain_to_replica_idx
        pt_sampler = sampling.run(pt_sampler) # 3 rounds == 2+4+8=14 scans ==> return to init array
        pt_state = pt_sampler.pt_state
        self.assertEqual(pt_state.stats.round_idx, 4)
        self.assertEqual(pt_state.stats.scan_idx, 1)
        self.assertTrue(check_maps_inv_temps(pt_state))
        self.assertTrue(jnp.all(
            pt_state.chain_to_replica_idx == init_chain_to_replica_idx
        ))

        #######################################################################
        ## check resolvers with non-trivial swap decisions, using only low
        ## level functions and a random initial permutation
        #######################################################################

        n_replicas = 10
        id_perm = jnp.arange(n_replicas)
        swap_group_actions = initialization.init_swap_group_actions(n_replicas)
        rng_key, run_key = random.split(rng_key)
        chain_to_replica_idx = random.permutation(run_key, jnp.arange(n_replicas))
        replica_to_chain_idx = chain_to_replica_idx.argsort()

        for is_odd_scan in (0,1):
            rng_key, run_key = random.split(rng_key)
            proto_swap_decisions = random.bernoulli(
                run_key, shape=(n_replicas-1,)
            )
            new_chain_to_replica_idx = swaps.resolve_chain_to_replica_idx(
                is_odd_scan,
                proto_swap_decisions,
                chain_to_replica_idx,
                swap_group_actions
            )
            (
                replica_swap_partner, 
                new_replica_to_chain_idx
            ) = swaps.resolve_replica_maps(
                replica_to_chain_idx, new_chain_to_replica_idx
            )
            # replica stays put if and only if it is a fixed point of 
            # replica_swap_partner
            self.assertTrue(jnp.all(
                (new_replica_to_chain_idx == replica_to_chain_idx) ==
                (replica_swap_partner == id_perm)
            ))
            # replica_swap_partner is nilpotent
            self.assertTrue(
                jnp.all(
                    replica_swap_partner[replica_swap_partner] == id_perm
                )
            )
            # maps agree
            self.assertTrue(
                replica_chain_maps_agree(
                    new_replica_to_chain_idx, new_chain_to_replica_idx
                )
            )


if __name__ == '__main__':
    unittest.main()
    