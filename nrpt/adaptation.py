from functools import partial
from operator import itemgetter

import jax
from jax import numpy as jnp
from jax import lax

from autostep import autostep

from nrpt import interpolation

def adapt_schedule(pt_state):
    """
    Update inverse temperature schedule targeting equi-rejection.
    """
    # We need the map
    #   Chain -> inv_temp == Chain -> Replica -> inv_temp
    chain_to_replica_idx = pt_state.chain_to_replica_idx
    inv_temp_schedule = pt_state.replica_states.inv_temp[chain_to_replica_idx]
    n_replicas = len(inv_temp_schedule)
    current_round_rej_probs = pt_state.stats.current_round_rej_probs
    
    # compute the barrier estimates
    # note: add eps to estimated rej probs to enforce strict monotonicity when
    # no rejections are observed
    result_dtype = current_round_rej_probs.dtype
    eps = jnp.finfo(result_dtype).eps
    barrier_estimate = (current_round_rej_probs+eps).cumsum()
    total_barrier_estimate = barrier_estimate[-1]
    normalized_estimated_barrier = jnp.insert(
        barrier_estimate[:-1] / total_barrier_estimate,
        jnp.array([0, n_replicas-1]),
        jnp.arange(2, dtype=result_dtype) # force endpoints to be exactly (0,1) to avoid interpolator issues
    )

    # find the equi-rejection schedule via interpolation
    # 1) fit: P(norm-cumulative_barrier) = schedule (with P monotonic)
    # 2) update: new_schedule = P(linspace in [0,1])
    norm_barrier_to_inv_temp_interp = interpolation.build_pchip_interpolator(
        normalized_estimated_barrier, inv_temp_schedule
    )
    new_inv_temp_schedule = interpolation.interpolate(
        norm_barrier_to_inv_temp_interp, 
        jnp.linspace(0, 1, n_replicas, dtype=result_dtype)
    )
    new_inv_temp_schedule = new_inv_temp_schedule.at[-1].set(1.) # force it to be exactly 1 (without it, it differs by ~1e-7)

    # update inv_temps in replicas: need the map
    # replica -> new inv temp == replica -> chain -> new_inv_temp
    pt_state = pt_state._replace(
        replica_states = pt_state.replica_states._replace(
            inv_temp = new_inv_temp_schedule[pt_state.replica_to_chain_idx]
        )
    )

    # fit P(inv_temp) = barrier for stats purpose
    # note: schedule is len n_replicas but barrier_estimate is n_replicas-1
    inv_temp_to_barrier_interpolator = interpolation.build_pchip_interpolator(
        inv_temp_schedule, jnp.insert(barrier_estimate, 0, 0.)
    )

    # return updated state with barrier fit
    return pt_state, inv_temp_to_barrier_interpolator


def adapt_explorers(kernel, pt_state):
    """
    Adapt the exploration kernels.
    """
    if not isinstance(kernel, autostep.AutoStep):
        return pt_state
    
    # trigger `adapt` on all replicas
    replica_states = jax.vmap(partial(kernel.adapt, force=True))(
        pt_state.replica_states
    )

    return pt_state._replace(replica_states = replica_states)
