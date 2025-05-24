import jax
from jax import numpy as jnp
from jax import lax
from jax import random

# core swap mechanism
# Note: `swap_idx` is the idx of the bottom replica of a swap
# Note: `swap_idx` and `swap_decision` must always refer to the same swap.
def swap_scan_fn(carry, x):
    # replica_idx, swap_idx = 0, is_odd_scan
    # prop_chain_idx, chain_idx = proposed_replica_to_chain_idx[replica_idx], replica_to_chain_idx[replica_idx]
    replica_idx, swap_idx, swap_decisions = carry
    prop_chain_idx, chain_idx = x

    # check swap membership
    # both False if and only if 
    #   - replica_idx==0 and this is an odd step
    #   - replica_idx==n_replicas-1 and it is not part of a swap (i.e., when n_replicas even + is odd swap, or viceversa)
    is_swap_bottom = jnp.logical_and(
        swap_idx == replica_idx, 
        prop_chain_idx != chain_idx # only may be false for excluded endpoints
    )
    is_swap_top = swap_idx+1 == replica_idx

    # note: clunky `jnp.logical_*` are needed because they can handle traced
    # values during compilation, whereas `and`, `or`, etc try to eval them
    new_chain_idx = lax.select(
        jnp.logical_and(
            # note: this can be out of bounds for excluded right endpoint, but 
            # jax does not error; the actual behavior does not matter since
            # in this situation the replica is neither top nor bottom
            swap_decisions[swap_idx],
            jnp.logical_or(is_swap_bottom, is_swap_top)
        ),
        prop_chain_idx,
        chain_idx
    )

    # update swap index when we hit the top partner of a swap
    new_swap_idx = lax.select(is_swap_top, swap_idx+2, swap_idx)
    new_replica_idx = replica_idx+1
    return (new_replica_idx, new_swap_idx, swap_decisions), new_chain_idx

# scan over replica indices to resolve swaps given proposals and decisions
def resolve_replica_to_chain_idx(
        replica_to_chain_idx,
        proposed_replica_to_chain_idx,
        is_odd_scan,
        swap_decisions
    ):
    _, new_replica_to_chain_idx = lax.scan(
        swap_scan_fn,
        xs=(proposed_replica_to_chain_idx, replica_to_chain_idx),
        init=(0, is_odd_scan, swap_decisions)
    )
    return new_replica_to_chain_idx

# to determine the c2r map compatible with the new r2c map, use
#   chain -> new_replica == chain -> old_replica -> new_chain -> old_replica
# I.e., to get the new replica in charge of c, fetch the replica that was in 
# charge of the new chain c' that the old replica in charge of c is in charge
# of now.
def resolve_chain_to_replica_idx(
        old_chain_to_replica_idx, 
        new_replica_to_chain_idx
    ):
    return old_chain_to_replica_idx[
        new_replica_to_chain_idx[old_chain_to_replica_idx]
    ]

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
def communication_step(pt_state, is_odd_scan, swap_group_actions):
    # compute swap probabilities
    # note: these contains all the swaps: [0<->1], [1<->2], [2<->3] => even the 
    # irrelevants for this scan. The reason is that it is faster to compute these
    # in a vectorized fashion. Iow, the savings are not asymptotically relevant;
    # we'll still do O(n_replicas) ops but less efficiently.
    # Furthermore, we can still use the rejection prob of inactive swaps for
    # adaptation purposes
    replica_to_chain_idx = pt_state.replica_to_chain_idx
    chain_to_replica_idx = pt_state.chain_to_replica_idx
    replica_states = pt_state.replica_states

    # We need the map
    #   Chain -> (inv_temp, loglik) == Chain -> Replica -> (inv_temp, loglik)
    inv_temp_schedule = replica_states.inv_temp[chain_to_replica_idx]
    chain_log_liks = replica_states.log_lik[chain_to_replica_idx]
    delta_inv_temp = jnp.diff(inv_temp_schedule)
    delta_LL = jnp.diff(chain_log_liks)
    neg_log_acc_ratio = delta_inv_temp * delta_LL # == -log(accept ratio)
    swap_reject_probs = -lax.expm1(-lax.max(0., neg_log_acc_ratio))

    # sample swap decisions
    # note: these are length n_replicas-1, as the last replica can only by a
    # top in a swap---and therefore its decision is the same as the one for the
    # penultimate replica---or skipped altogether.
    n_replicas = len(inv_temp_schedule) 
    rng_key, exp_key = random.split(pt_state.rng_key)
    randexp_vars = random.exponential(exp_key, n_replicas-1)
    swap_decisions = randexp_vars > neg_log_acc_ratio

    # determine the maximal swap group action (i.e., in case all were accepted)
    idx_group_action = swap_group_actions[is_odd_scan]
    proposed_replica_to_chain_idx = replica_to_chain_idx[idx_group_action]
    
    # resolve index map updates
    new_replica_to_chain_idx = resolve_replica_to_chain_idx(
        replica_to_chain_idx,
        proposed_replica_to_chain_idx,
        is_odd_scan,
        swap_decisions
    )
    new_chain_to_replica_idx = resolve_chain_to_replica_idx(
        chain_to_replica_idx, 
        new_replica_to_chain_idx
    )
    
    # update state: prng key, chain-replica maps, and replica inv_temps
    replica_states = replica_states._replace(
        # Replica -> new temp == Replica -> new chain -> temp
        inv_temp = inv_temp_schedule[new_replica_to_chain_idx]
    )
    pt_state = pt_state._replace(
        rng_key = rng_key,
        replica_states = replica_states,
        replica_to_chain_idx = new_replica_to_chain_idx,
        chain_to_replica_idx = new_chain_to_replica_idx
    )

    return (pt_state, swap_reject_probs, delta_inv_temp, chain_log_liks)

