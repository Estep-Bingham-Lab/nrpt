from collections import namedtuple

from jax import numpy as jnp

###############################################################################
# statistics used for adaptation
###############################################################################

PTStats = namedtuple(
    "PTStats",
    [
        "scan_idx",
        "round_idx",
        "current_round_rej_probs",
        "last_round_rej_probs",
        "barrier_estimate"
    ],
)
"""
A :func:`~collections.namedtuple` containing statistics gathered during a PT 
run. It consists of the fields:

 - **scan_idx** - jhfg.
 - **round_idx** - jhfg.
 - **current_round_rej_probs** - jhfg.
 - **last_round_rej_probs** - jhfg.
 - **barrier_estimate** - jhfg.
"""

def init_state(n_replicas):
    return PTStats(
        1, 1, jnp.zeros(n_replicas-1), jnp.zeros(n_replicas-1), jnp.array(0.)
    )

# TODO: logZ + loglik autocorrelation
def end_of_scan_stats_update(pt_state, swap_reject_probs):
    stats = pt_state.stats

    # update swap rejection probs
    # note: scan_idx starts at 1, so the number of elements in the online 
    # estimator is scan_idx-1
    new_current_round_rej_probs = stats.current_round_rej_probs + (
        swap_reject_probs - stats.current_round_rej_probs
    ) / stats.scan_idx

    return pt_state._replace(
        stats = stats._replace(
            scan_idx = stats.scan_idx + 1,
            current_round_rej_probs = new_current_round_rej_probs
        )
    )

def end_of_round_stats_update(pt_state, barrier_estimate):
    stats = pt_state.stats
    return pt_state._replace(
        stats = stats._replace(
            scan_idx = 1,
            round_idx = stats.round_idx + 1,
            current_round_rej_probs = jnp.zeros_like(stats.current_round_rej_probs),
            last_round_rej_probs = stats.current_round_rej_probs,
            barrier_estimate = barrier_estimate
        )
    )

