from functools import partial

import numpy

from scipy.interpolate import PchipInterpolator

import jax
from jax import numpy as jnp

def adapt_schedule(pt_state):
    """
    Update inverse temperature schedule targeting equi-rejection.
    """
    # compute the barrier estimates
    # note: add eps to estimated rej probs to enforce strict monotonicity when
    # no rejections are observed
    inv_temp_schedule = pt_state.replica_states.inv_temp
    current_round_rej_probs = pt_state.stats.current_round_rej_probs
    eps = jnp.finfo(current_round_rej_probs.dtype).eps
    cum_current_round_rej_probs = (current_round_rej_probs+eps).cumsum()
    barrier_estimate = cum_current_round_rej_probs[-1]
    normalized_estimated_barrier = jnp.insert(
        cum_current_round_rej_probs / cum_current_round_rej_probs[-1],
        0,
        jnp.zeros_like(cum_current_round_rej_probs, shape=())
    )

    # find the equi-rejection schedule via interpolation
    # note: need to use scipy as this is not implemented in JAX
    interpolator = PchipInterpolator(
        normalized_estimated_barrier, inv_temp_schedule
    )
    new_inv_temp_schedule = jnp.array(
        interpolator(numpy.linspace(0, 1, len(inv_temp_schedule)))
    )

    # update schedule and return with barrier estimate
    pt_state = pt_state._replace(
        replica_states = pt_state.replica_states._replace(
            inv_temp = new_inv_temp_schedule
        )
    )
    return pt_state, barrier_estimate


def adapt_explorers(kernel, pt_state):
    """
    Adapt the exploration kernels.
    """
    new_replica_states = jax.vmap(partial(kernel.adapt, force=True))(
        pt_state.replica_states
    )
    return pt_state._replace(replica_states = new_replica_states)
