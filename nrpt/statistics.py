from collections import namedtuple

from jax import numpy as jnp

from nrpt import interpolation
from nrpt import logZ

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
        "barrier_fit",
        "current_round_dlogZ_estimates",
        "logZ_fit"
    ],
)
"""
A :func:`~collections.namedtuple` containing statistics gathered during a PT 
run. It consists of the fields:

 - **scan_idx** - jhfg.
 - **round_idx** - jhfg.
 - **current_round_rej_probs** - jhfg.
 - **last_round_rej_probs** - jhfg.
 - **barrier_fit** - jhfg.
 - **current_round_dlogZ_estimates** - jhfg.
 - **logZ_fit** - jhfg.
"""

def init_stats(n_replicas):
    return PTStats(
        1, 
        1, 
        jnp.zeros(n_replicas-1), 
        jnp.zeros(n_replicas-1),
        interpolation.empty_interpolator(n_replicas),
        logZ.init_estimates(n_replicas),
        interpolation.empty_interpolator(n_replicas)
    )

def post_scan_stats_update(
        pt_state, 
        swap_reject_probs,
        delta_inv_temp, 
        chain_log_liks
    ):
    stats = pt_state.stats

    # update swap rejection probs
    # note: scan_idx starts at 1, so the number of elements in the online 
    # estimator is scan_idx-1
    new_current_round_rej_probs = stats.current_round_rej_probs + (
        swap_reject_probs - stats.current_round_rej_probs
    ) / stats.scan_idx

    # update logZ estimates
    new_current_round_dlogZ_estimates = logZ.update_estimates(
        stats.current_round_dlogZ_estimates, delta_inv_temp, chain_log_liks
    )

    return pt_state._replace(
        stats = stats._replace(
            scan_idx = stats.scan_idx + 1,
            current_round_rej_probs = new_current_round_rej_probs,
            current_round_dlogZ_estimates = new_current_round_dlogZ_estimates
        )
    )

def end_of_round_stats_update(pt_state, barrier_fit):
    stats = pt_state.stats
    return pt_state._replace(
        stats = stats._replace(
            scan_idx = 1,
            round_idx = stats.round_idx + 1,
            current_round_rej_probs = jnp.zeros_like(
                stats.current_round_rej_probs
            ),
            last_round_rej_probs = stats.current_round_rej_probs,
            barrier_fit = barrier_fit,
            current_round_dlogZ_estimates = logZ.empty_estimates(
                stats.current_round_dlogZ_estimates
            ),
            logZ_fit = logZ.fit_interpolator(
                barrier_fit.x, stats.current_round_dlogZ_estimates
            )
        )
    )

