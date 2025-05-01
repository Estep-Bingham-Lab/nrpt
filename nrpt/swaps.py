from jax import numpy as jnp
from jax import lax
from jax import random


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
def communication_step(is_odd_scan, replica_states, swap_group_actions):
    # compute relevant quantities
    # note: these contains all the swaps: [0<->1], [1<->2], [2<->3] => even the 
    # irrelevants for this scan. The reason is that it is faster to compute these
    # in a vectorized fashion. Iow, the savings are not asymptotically relevant;
    # we'll still do O(n_replicas) ops but less efficiently.
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
    replica_to_chain_idx = replica_states.replica_to_chain_idx
    idx_group_action = swap_group_actions[is_odd_scan]
    proposed_replica_to_chain_idx = replica_to_chain_idx[idx_group_action]
    
    # resolve new chain indices and update replica states
    _, new_replica_to_chain_idx = lax.scan(
        swap_scan_fn,
        xs=(
            proposed_replica_to_chain_idx,
            replica_to_chain_idx, 
            jnp.insert(swap_decisions, 0, False) # all `xs` need same shape. harmless addition as this value will never be used neither in even nor odd swaps
        ),
        init=(0, is_odd_scan, swap_decisions[is_odd_scan]) # need to skip the [0<->1] swap in odd scans
    )
    replica_states = replica_states._replace(
        replica_to_chain_idx = new_replica_to_chain_idx
    )

    return (replica_states, swap_reject_probs)

