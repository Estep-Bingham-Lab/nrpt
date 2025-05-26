from collections import namedtuple

import jax
from jax import numpy as jnp
from jax import lax
from jax import random

from numpyro.util import is_prng_key

from nrpt import sampling
from nrpt import statistics

###############################################################################
# type definitions
###############################################################################

PTSampler = namedtuple(
    "PTSampler",
    [
        "kernel",
        "pt_state",
        "n_rounds",
        "n_refresh",
        "model_args",
        "model_kwargs",
        "swap_group_actions",
    ],
)
"""
A :func:`~collections.namedtuple` defining a Parallel Tempering sampler. It 
consists of the fields:

 - **kernel** - jhfg.
 - **pt_state** - jhfg.
 - **n_rounds** - jhfg.
 - **n_refresh** - jhfg.
 - **model_args** - jhfg.
 - **model_kwargs** - jhfg.
 - **swap_group_actions** - jhfg.
"""

PTState = namedtuple(
    "PTState",
    [
        "replica_states",
        "replica_to_chain_idx",
        "chain_to_replica_idx",
        "rng_key",
        "stats",
        "samples"
    ],
)
"""
A :func:`~collections.namedtuple` defining the state of a Parallel Tempering
ensemble. It consists of the fields:

 - **replica_states** - jhfg.
 - **replica_to_chain_idx** - jhfg.
 - **chain_to_replica_idx** - jhfg.
 - **rng_key** - jhfg.
 - **stats** - jhfg.
 - **samples** - jhfg.
"""

###############################################################################
# constructors
###############################################################################

# define DEO swap actions
# maximal proposed change if all swaps are accepted
# examples:
# n_replicas=5:
# E: [0,1,2,3,4] -> identity perm
#    [1,0,3,2,4] -> all swaps acc perm == even group action
# O: [0,1,2,3,4] -> identity perm
#    [0,2,1,4,3] -> all swaps acc perm == odd group action 
# n_replicas=6:
# E: [0,1,2,3,4,5] -> identity perm
#    [1,0,3,2,5,4] -> all swaps acc perm == even group action
# O: [0,1,2,3,4,5] -> identity perm
#    [0,2,1,4,3,5] -> all swaps acc perm == odd group action
# n_replicas=7:
# E: [0,1,2,3,4,5,6] -> identity perm
#    [1,0,3,2,5,4,6] -> all swaps acc perm == even group action
# O: [0,1,2,3,4,5,6] -> identity perm
#    [0,2,1,4,3,6,5] -> all swaps acc perm == odd group action
def init_swap_group_actions(n_replicas):
    idx_even_group_action = [
        (
            i - 1 if i%2 else
            i + 1 if i<n_replicas-1 else
            i
        ) 
        for i in range(n_replicas)
    ]
    idx_odd_group_action = [
        (
            min(i + 1, n_replicas-1) if i%2 else
            i - 1 if i > 0 else
            i      
        ) 
        for i in range(n_replicas)
    ]
    return jnp.array([idx_even_group_action, idx_odd_group_action])

def init_replica_states(kernel, rng_key, n_replicas, model_args, model_kwargs):
    # use the kernel initialization to get a prototypical state
    prototype_init_state = kernel.init(rng_key, 0, None, model_args, model_kwargs)

    # extend the prototypical state to all replicas
    return jax.tree.map(
        lambda xs: (
            random.split(xs, n_replicas) 
            if is_prng_key(xs)
            else lax.broadcast(xs, (n_replicas,))
        ),
        prototype_init_state
    )

def init_schedule(replica_states, n_replicas):
    chain_to_replica_idx = jnp.arange(n_replicas)    # init to identity permutation
    replica_to_chain_idx = jnp.arange(n_replicas)    # id = argsort(id)
    inv_temp_schedule = jnp.linspace(0,1,n_replicas) # init to uniform grid
    replica_states = replica_states._replace(inv_temp=inv_temp_schedule)
    return replica_states, replica_to_chain_idx, chain_to_replica_idx

def init_samples_container(
        kernel, 
        n_rounds, 
        model_args, 
        model_kwargs, 
        replica_states
    ):
    # grab a generic replica state
    constrained_sample_with_extras = sampling.extract_sample(
        kernel, 
        model_args, 
        model_kwargs, 
        replica_states, 
        0,
    )

    # use the above as template to create container
    n_stored_samples = sampling.n_scans_in_round(n_rounds)
    return jax.tree.map(
        lambda x: jnp.empty_like(
            x, 
            shape = (n_stored_samples, *x.shape),
            # device = jax.devices('cpu')[0] # TODO: should we add option to do this?
        ),
        constrained_sample_with_extras
    )

def init_pt_state(
        kernel, 
        rng_key, 
        n_replicas, 
        n_rounds,
        model_args, 
        model_kwargs, 
        collect_samples
    ):
    rng_key, init_key = random.split(rng_key)
    replica_states = init_replica_states(
        kernel, init_key, n_replicas, model_args, model_kwargs
    )
    replica_states, replica_to_chain_idx, chain_to_replica_idx = init_schedule(
        replica_states, n_replicas
    )
    stats = statistics.init_stats(n_replicas)
    if collect_samples:
        samples = init_samples_container(
            kernel, 
            n_rounds, 
            model_args, 
            model_kwargs,
            replica_states
        )
    else: 
        samples = None

    return PTState(
        replica_states, 
        replica_to_chain_idx, 
        chain_to_replica_idx, 
        rng_key, 
        stats,
        samples
    )

def PT(
        kernel, 
        rng_key, 
        n_replicas = 10, 
        n_rounds = 10, 
        n_refresh = 3, 
        model_args = (), 
        model_kwargs = {},
        collect_samples = True
    ):
    swap_group_actions = init_swap_group_actions(n_replicas)
    pt_state = init_pt_state(
        kernel, 
        rng_key, 
        n_replicas, 
        n_rounds,
        model_args, 
        model_kwargs, 
        collect_samples
    )
    return PTSampler(
        kernel, 
        pt_state, 
        n_rounds,
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions
    )
