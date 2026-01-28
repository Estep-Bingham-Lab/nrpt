from operator import itemgetter

import jax
from jax import numpy as jnp
from jax import lax
from jax import random

from numpyro import util

from automcmc import automcmc

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

# resolve replica swap partners and update replica to chain map. 
# use the identities
#   replica -> new chain == replica -> swap partner => replica -> old chain
#   replica -> swap partner == replica -> old chain => chain -> new_replica
# notes about the swap partner permutation:
#   - it is nilpotent ("the partner of my partner is me")
#   - its fixed points are the replicas which didn't participate or
#     whose swaps failed
def resolve_replica_maps(old_replica_to_chain_idx, new_chain_to_replica_idx):
    replica_swap_partner = new_chain_to_replica_idx[old_replica_to_chain_idx]
    new_replica_to_chain_idx = old_replica_to_chain_idx[replica_swap_partner]
    return replica_swap_partner, new_replica_to_chain_idx

# swap replica states: need to exchange between replica swap partners the 
# fields that are chain-specific. At the very least this includes inv_temp.
# AutoMCMC kernels also have
#   - stats
#   - base_step_size
#   - base_precond_state
# use the identity
#   Replica -> new chain val == Replica -> swap partner => Replica -> val
def swap_replica_states(kernel, replica_states, replica_swap_partner):
    new_inv_temp = replica_states.inv_temp[replica_swap_partner] # equivalent to inv_temp_schedule[new_replica_to_chain_idx]

    if not isinstance(kernel, automcmc.AutoMCMC):
        return replica_states._replace(inv_temp = new_inv_temp)

    gettr = itemgetter(replica_swap_partner)
    return replica_states._replace(
        stats = jax.tree.map(gettr, replica_states.stats),
        base_step_size = replica_states.base_step_size[replica_swap_partner],
        base_precond_state = jax.tree.map(
            gettr, replica_states.base_precond_state
        ),
        inv_temp = new_inv_temp
    )

def sanitize_log_liks(kernel, chain_log_liks):
    """
    In most interesting models, the likelihood has a support strictly included
    in the support of the prior (or reference). So if iid sampling is used at 
    this chain, the sample obtained might show non-finite log likelihood value.
    This utility fixes this.
    """
    # don't do anything if iid sampling is not available; we don't want to mask
    # other types of errors
    if kernel.model is None or not jnp.issubdtype(chain_log_liks,jnp.floating):
        return chain_log_liks
    
    # set the log lik at the reference to a very negative val if it isn't finite
    return lax.cond(
        jnp.isfinite(chain_log_liks[0]),
        util.identity,
        lambda x: x.at[0].set(jnp.finfo(x.dtype).min),
        chain_log_liks
    )

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
def communication_step(kernel, pt_state, is_odd_scan, swap_group_actions):
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

    # Obtain the inverse temperatures and log likelihood values sorted by chain.
    # We need the map
    #   Chain -> (inv_temp, loglik) == Chain -> Replica -> (inv_temp, loglik)
    inv_temp_schedule = replica_states.inv_temp[chain_to_replica_idx]
    chain_log_liks = replica_states.log_lik[chain_to_replica_idx]
    
    # compute acceptance ratios and rejection probs
    chain_log_liks = sanitize_log_liks(kernel, chain_log_liks) 
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

    # resolve chain to replica map
    new_chain_to_replica_idx = resolve_chain_to_replica_idx(
        is_odd_scan,
        proto_accept_decisions,
        chain_to_replica_idx,
        swap_group_actions
    )

    # resolve replica swap partners and new chains
    replica_swap_partner, new_replica_to_chain_idx = resolve_replica_maps(
        replica_to_chain_idx, new_chain_to_replica_idx
    )
    
    # swap replica states
    replica_states = swap_replica_states(
        kernel, replica_states, replica_swap_partner
    )
    
    # update pt_state
    pt_state = pt_state._replace(
        rng_key = rng_key,
        replica_states = replica_states,
        replica_to_chain_idx = new_replica_to_chain_idx,
        chain_to_replica_idx = new_chain_to_replica_idx
    )

    return (pt_state, swap_reject_probs, delta_inv_temp, chain_log_liks)


