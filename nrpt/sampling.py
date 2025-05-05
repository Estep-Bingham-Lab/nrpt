from jax import lax

from nrpt import exploration
from nrpt import statistics
from nrpt import swaps

def n_scans_in_round(round_idx):
    return 2 ** round_idx

def pt_scan(
        kernel, 
        pt_state, 
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions
    ):
    is_odd_scan = pt_state.stats.scan_idx % 2
    pt_state = exploration.exploration_step(
        kernel, pt_state, n_refresh, model_args, model_kwargs
    )
    pt_state, swap_reject_probs = swaps.communication_step(
        pt_state, is_odd_scan, swap_group_actions
    )
    pt_state = statistics.end_of_scan_stats_update(pt_state, swap_reject_probs)
    return pt_state

def pt_round(pt_sampler, n_scans = None):
    (
        kernel, 
        pt_state, 
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions
    ) = pt_sampler
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
        length = (
            n_scans_in_round(pt_state.stats.round_idx) 
            if n_scans is None else n_scans
        )
    )[0]
    pt_state = statistics.end_of_round_stats_update(pt_state)
    return pt_sampler._replace(pt_state = pt_state)
