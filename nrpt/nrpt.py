from collections import namedtuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import lax
from jax import random

from numpygeons import toy_examples
from numpygeons.bridge import loop_sample
from numpyro.util import is_prng_key

from autostep import autohmc

PTState = namedtuple(
    "PTState",
    [
        "replica_states",
        "replica_to_chain_idx",
        "swap_group_actions",
        "rng_key"
    ],
)
"""
A :func:`~collections.namedtuple` defining the state of a Parallel Tempering
ensemble. It consists of the fields:

 - **replica_states** - jhfg.
 - **replica_to_chain_idx** - jhfg.
 - **swap_group_actions** - jhfg.
 - **rng_key** - jhfg.
"""

def init_pt_state(kernel, rng_key, n_replicas):
    swap_group_actions = init_swap_group_actions(n_replicas)
    rng_key, init_key = random.split(rng_key)
    replica_states = init_replica_states(
        kernel, init_key, n_replicas, model_args, model_kwargs
    )
    replica_states, replica_to_chain_idx = init_schedule(replica_states)
    return PTState(
        replica_states, replica_to_chain_idx, swap_group_actions, rng_key
    )

# core swap mechanism
# Note: `swap_idx` and `swap_decision` must always refer to the same swap.
def swap_scan_fn(carry, x):
    replica_idx, swap_idx, swap_decision = carry
    prop_chain_idx, chain_idx, maybe_irrelevant_swap_decision = x

    # check swap membership
    is_swap_bottom = swap_idx   == replica_idx
    is_swap_top    = swap_idx+1 == replica_idx

    # update chain index
    new_chain_idx = lax.select(
        jnp.logical_and(
            swap_decision, 
            jnp.logical_or(is_swap_bottom, is_swap_top)
        ),
        prop_chain_idx,
        chain_idx
    )

    # update swap index and decision when we hit the top partner of a swap
    new_swap_idx = lax.select(is_swap_top, swap_idx+1, swap_idx)
    new_swap_decision = lax.select(
        is_swap_top, maybe_irrelevant_swap_decision, swap_decision
    )
    new_replica_idx = replica_idx+1
    return (new_replica_idx, new_swap_idx, new_swap_decision), new_chain_idx


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

def init_schedule(replica_states):
    n_replicas = len(replica_states.inv_temp)
    replica_to_chain_idx = jnp.arange(n_replicas)    # init to identity permutation
    inv_temp_schedule = jnp.linspace(0,1,n_replicas) # init to uniform grid
    replica_states = replica_states._replace(inv_temp=inv_temp_schedule)
    return replica_states, replica_to_chain_idx

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
    idx_even_group_action = jnp.array([
        (
            i - 1 if i%2 else
            i + 1 if i<n_replicas-1 else
            i
        ) 
        for i in range(n_replicas)
    ])
    idx_odd_group_action = jnp.array([
        (
            min(i + 1, n_replicas-1) if i%2 else
            i - 1 if i > 0 else
            i      
        ) 
        for i in range(n_replicas)
    ])
    return {0: idx_even_group_action, 1: idx_odd_group_action}

# test
n_replicas = 4
n_refresh = 32
rng_key = random.key(1)
model, model_args, model_kwargs = toy_examples.eight_schools_example()
kernel = autohmc.AutoMALA(model, init_inv_temp=1.0)
pt_state = init_pt_state(kernel, rng_key, n_replicas)
pt_state

# scan
scan_idx = 1

def scan(scan_idx, pt_state):
    pass

# TODO: do iid sampling at inv_temp=0
def exploration_step(kernel, replica_states, n_refresh, model_args, model_kwargs):
    p_loop_sample = partial(
        loop_sample, 
        kernel, 
        n_refresh=n_refresh, 
        model_args=model_args, 
        model_kwargs=model_kwargs
    )
    return jax.vmap(p_loop_sample)(replica_states)

# Communication step
#  
# acc prob of swapping 2 states is proportional to ratio
#   acc ratio = Pi(with-swap)/Pi(no-swap)
# with
#   Pi(no-swap) = exp(Tempered-Log-Joint(xi, pi, beta_i) + Tempered-Log-Joint(xi+1, pi+1, beta_i+1)])  
#     \propto exp(-[V0(xi)+beta_iV(xi)+K(pi)] -[V0(xi+1)+beta_i+1V(xi+1)+K(pi+1)])
#   Pi(with-swap) = exp(Tempered-Log-Joint(xi, pi, beta_i+1) + Tempered-Log-Joint(xi+1, pi+1, beta_i)])
#     \propto exp(-[V0(xi)+beta_i+1V(xi)+K(pi)] -[V0(xi+1)+beta_iV(xi+1)+K(pi+1)])
# So
#   acc ratio = exp(-[beta_i+1V(xi)] -[beta_iV(xi+1)] + [beta_iV(xi)] +[beta_i+1V(xi+1)])
#    = exp([beta_i+1 - beta_i][V(xi+1)-V(xi)])
#    = exp(-[beta_i+1 - beta_i][LL(xi+1)-LL(xi)])
# where LL=-V is the loglik.
# Intuition: distributions with higher beta favor higher LL values. Therefore,
# a swap happens w.p. 1 if the lower-beta replica has a higher-LL sample. This
# is a more "satisfactory" arrangement for the ensemble.
# Also: if U~Unif[0,1], then
#   U < ratio <=> E := -logU = > -log(ratio) =: nlaccr
# where E ~ Exp(1). Finally, the rejection probability can be computed as
#   R = 1-A = 1-min{1,exp(-nlaccr)}=1-exp(-max{0,nlaccr})
def communication_step(replica_states, swap_group_actions):
    # compute relevant quantities
    # note: these contains all the swaps: [0<->1], [1<->2], [2<->3] => even the 
    # irrelevants for this scan. The reason is that it is faster to compute these
    # in a vectorized fashion. Iow, the savings are not asymptotically relevant;
    # we'll still do O(n_replicas) [n_replicas/2] ops but less efficiently.
    # Furthermore, we can still use the rejection prob of inactive swaps for
    # adaptation purposes
    inv_temp_schedule = replica_states.inv_temp
    n_replicas = len(inv_temp_schedule) 
    delta_inv_temp = inv_temp_schedule[1:] - inv_temp_schedule[:-1]
    delta_LL = replica_states.log_lik[1:] - replica_states.log_lik[:-1]
    accept_thresholds = delta_inv_temp * delta_LL
    swap_reject_probs = -lax.expm1(-lax.max(0., accept_thresholds))
    rng_key, exp_key = random.split(rng_key)
    randexp_vars = random.exponential(exp_key, n_replicas-1)
    swap_decisions = randexp_vars > accept_thresholds

    # iterate replicas
    # determine the maximal swap group action (i.e., in case all are accepted)
    is_odd_scan = scan_idx % 2
    idx_group_action = swap_group_actions[is_odd_scan]
    proposed_replica_to_chain_idx = replica_to_chain_idx[idx_group_action]

    
    # resolve new chain indices
    _, new_replica_to_chain_idx = lax.scan(
        swap_scan_fn,
        xs=(
            proposed_replica_to_chain_idx,
            replica_to_chain_idx, 
            jnp.insert(swap_decisions, 0, False) # all `xs` need same shape. harmless addition as this value will never be used neither in even nor odd swaps
        ),
        init=(0, is_odd_scan, swap_decisions[is_odd_scan]) # need to skip the [0<->1] swap in odd scans
    )
