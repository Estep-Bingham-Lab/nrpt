from functools import partial

import jax
from jax import lax

from nrpt import adaptation
from nrpt import exploration
from nrpt import statistics
from nrpt import swaps

def n_scans_in_round(round_idx):
    return 2 ** round_idx

def total_scans(n_rounds):
    return 2 ** (n_rounds+1) - 2

def total_barrier(barrier_fit):
    return barrier_fit.y[-1]

def end_of_round_adaptation(kernel, pt_state):
    ending_round_idx = pt_state.stats.round_idx

    # adapt explorers
    pt_state = adaptation.adapt_explorers(kernel, pt_state)
    
    # adapt schedule
    pt_state, barrier_fit = adaptation.adapt_schedule(pt_state)

    # collect statistics
    pt_state = statistics.end_of_round_stats_update(pt_state, barrier_fit)

    # print info
    # TODO: print a header before the fist call to this (in `run`?)
    jax.debug.print(
        "Round {i} \t Λ = {b:.2f} \t RejProbs (mean/max) = {rm:.1f}/{rM:.1f}",
        ordered=True,
        i=ending_round_idx,
        b=total_barrier(barrier_fit),
        rm=pt_state.stats.last_round_rej_probs.mean(),
        rM=pt_state.stats.last_round_rej_probs.max()
    )

    return pt_state

def pt_scan(
        kernel, 
        pt_state, 
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions
    ):
    """
    Run a full NRPT scan -- exploration + DEO communication -- and collect
    statistics. If it is the last scan in a round, perform adaptation.
    """
    # exploration
    pt_state = exploration.exploration_step(
        kernel, pt_state, n_refresh, model_args, model_kwargs
    )

    # communication
    is_odd_scan = pt_state.stats.scan_idx % 2
    pt_state, swap_reject_probs = swaps.communication_step(
        pt_state, is_odd_scan, swap_group_actions
    )

    # store scan statistics
    pt_state = statistics.end_of_scan_stats_update(pt_state, swap_reject_probs)

    # if end of run, do adaptation
    # note: scan_idx was just updated in prev line, so we need to substract 1
    # to get the index of the scan that just finished
    pt_state = lax.cond(
        pt_state.stats.scan_idx-1 == n_scans_in_round(pt_state.stats.round_idx),
        partial(end_of_round_adaptation, kernel),
        lambda s: s,
        pt_state
    )

    return pt_state

def run(pt_sampler):
    """
    Run NRPT in (implicit) round-based mode.
    """
    (
        kernel, 
        pt_state, 
        n_rounds,
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions
    ) = pt_sampler

    # perform scans sequentially in a `lax.scan` loop
    n_scans = total_scans(n_rounds)
    pt_state = lax.scan(
        lambda pt_state, _: (
            pt_scan(
                kernel, 
                pt_state, 
                n_refresh, 
                model_args, 
                model_kwargs,
                swap_group_actions
            ),
            None
        ), 
        pt_state,
        length = n_scans
    )[0]

    # update sampler object and return
    return pt_sampler._replace(pt_state = pt_state)
