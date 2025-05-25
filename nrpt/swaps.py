from jax import numpy as jnp
from jax import lax
from jax import random

# core (deterministic) swap mechanism
def resolve_chain_to_replica_idx(
        is_odd_scan,
        proto_accept_decisions,
        old_chain_to_replica_idx,
        swap_group_actions
    ):
    # determine the maximal swap group action (i.e., in case all were accepted)
    idx_group_action = swap_group_actions[is_odd_scan]
    n_replicas = len(idx_group_action)
    proposed_chain_to_replica_idx = old_chain_to_replica_idx[idx_group_action]

    # determine the participating chains, and their status in the swap
    is_in_swap = proposed_chain_to_replica_idx != old_chain_to_replica_idx
    is_swap_bottom = (jnp.arange(n_replicas) + is_odd_scan + 1) % 2

    # determine the actual swap decision
    accept_decisions = jnp.logical_or(
        jnp.logical_and(
            jnp.logical_and(is_in_swap, is_swap_bottom),
            # insert any value at the end to make it len == n_replicas
            # harmless because the last chain is never swap bottom
            jnp.insert(proto_accept_decisions, n_replicas, True)
        ),
        jnp.logical_and(
            jnp.logical_and(is_in_swap, jnp.logical_not(is_swap_bottom)),
            # insert any value at the beginning to make it len == n_replicas
            # harmless because first chain is never in swap when not a bottom
            jnp.insert(proto_accept_decisions, 0, True)
        )
    )
    return lax.select(
        accept_decisions, 
        proposed_chain_to_replica_idx, 
        old_chain_to_replica_idx
    )

# to determine the new r2c map compatible with the new c2r map, use
#   replica -> new_chain == replica -> old_chain -> new_replica -> old_chain
# I.e., to get the new chain owned by r, fetch the chain that was owned by 
# the replica r' that now owns the chain previously held by r
def resolve_replica_to_chain_idx(
        old_replica_to_chain_idx, 
        new_chain_to_replica_idx
    ):
    return old_replica_to_chain_idx[
        new_chain_to_replica_idx[old_replica_to_chain_idx]
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
    proto_accept_decisions = randexp_vars > neg_log_acc_ratio

    # resolve index map updates
    new_chain_to_replica_idx = resolve_chain_to_replica_idx(
        is_odd_scan,
        proto_accept_decisions,
        chain_to_replica_idx,
        swap_group_actions
    )
    new_replica_to_chain_idx = resolve_replica_to_chain_idx(
        replica_to_chain_idx, 
        new_chain_to_replica_idx
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

