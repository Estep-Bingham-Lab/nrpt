from collections import namedtuple
import time

import jax
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
        "logZ_fit",
        "current_round_loglik_stats",
        "last_round_loglik_stats",
        "last_round_start_time"
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
 - **current_round_loglik_stats** - jhfg.
 - **last_round_loglik_stats** - jhfg.
 - **last_round_start_time** - jhfg.
"""

def init_stats(n_replicas):
    return PTStats(
        1, 
        1, 
        jnp.zeros(n_replicas-1), 
        jnp.zeros(n_replicas-1),
        interpolation.empty_interpolator(n_replicas),
        logZ.init_estimates(n_replicas),
        interpolation.empty_interpolator(n_replicas),
        jnp.zeros((3, n_replicas)),
        jnp.zeros((3, n_replicas)),
        time.perf_counter()
    )

def update_loglik_stats(
        scan_idx, 
        current_round_loglik_stats, 
        chain_log_liks,
        pre_explore_chain_log_liks
    ):
    assert jnp.ndim(chain_log_liks) == 1
    n_replicas = len(chain_log_liks)
    assert jnp.shape(current_round_loglik_stats) == (3, n_replicas)

    # unpack
    chain_log_liks_means    = current_round_loglik_stats[0]
    chain_log_liks_vars     = current_round_loglik_stats[1]
    chain_log_liks_autocovs = current_round_loglik_stats[2]

    # update online estimators
    new_chain_log_liks_means = chain_log_liks_means + (
        chain_log_liks - chain_log_liks_means
    ) / scan_idx
    centered_log_liks = chain_log_liks - new_chain_log_liks_means
    new_chain_log_liks_vars = (
        (scan_idx-1)*chain_log_liks_vars + 
        centered_log_liks*(chain_log_liks - chain_log_liks_means)
    ) / scan_idx
    new_chain_log_liks_autocovs = (
        (scan_idx-1)*chain_log_liks_autocovs + 
        centered_log_liks*(pre_explore_chain_log_liks - chain_log_liks_means)
    ) / scan_idx

    return jnp.array([
        new_chain_log_liks_means,
        new_chain_log_liks_vars,
        new_chain_log_liks_autocovs
    ])


def post_scan_stats_update(
        pt_state, 
        swap_reject_probs,
        delta_inv_temp, 
        chain_log_liks,
        pre_explore_chain_log_liks
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

    # update logZ estimates
    new_current_round_loglik_stats = update_loglik_stats(
        stats.scan_idx, 
        stats.current_round_loglik_stats,
        chain_log_liks,
        pre_explore_chain_log_liks
    )

    return pt_state._replace(
        stats = stats._replace(
            scan_idx = stats.scan_idx + 1,
            current_round_rej_probs = new_current_round_rej_probs,
            current_round_dlogZ_estimates = new_current_round_dlogZ_estimates,
            current_round_loglik_stats = new_current_round_loglik_stats
        )
    )

def end_of_round_stats_update(pt_state, barrier_fit):
    stats = pt_state.stats
    round_ending_time = jax.experimental.io_callback(
        time.perf_counter, 
        jnp.array(1, stats.current_round_rej_probs.dtype),
        ordered=True
    )
    round_duration = round_ending_time - stats.last_round_start_time
    pt_state = pt_state._replace(
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
            ),
            current_round_loglik_stats = jnp.zeros_like(
                stats.current_round_loglik_stats
            ),
            last_round_loglik_stats = stats.current_round_loglik_stats,
            last_round_start_time = round_ending_time
        )
    )
    return pt_state, round_duration

def loglik_autocors(pt_state):
    last_round_loglik_stats = pt_state.stats.last_round_loglik_stats
    return last_round_loglik_stats[-1]/last_round_loglik_stats[-2]