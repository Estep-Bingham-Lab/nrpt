from jax import lax

from nrpt import adaptation
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

def pt_round(pt_sampler):
    """
    Perform a full round of NRPT. The index and length of the round are 
    dictated by the iterators in `pt_sampler.pt_state.stats`.

    Note: the `jax.lax.scan` function is used to carry out the sampling loop.
    Because this function assumes a static loop length -- while each round has
    an increasing number of scans -- JAX will trigger a recompilation each time 
    a round begins. This can take several seconds even in moderately complex
    models. Fortunately, the relative importance of this cost decays as the
    number of rounds increases.
    """
    (
        kernel, 
        pt_state, 
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions
    ) = pt_sampler

    # perform scans sequentially in a `lax.scan` loop
    n_scans = n_scans_in_round(pt_state.stats.round_idx)
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

    # adapt explorers
    pt_state = adaptation.adapt_explorers(kernel, pt_state)
    
    # adapt schedule
    pt_state, barrier_estimate = adaptation.adapt_schedule(pt_state)

    # collect statistics
    pt_state = statistics.end_of_round_stats_update(pt_state, barrier_estimate)

    # update sampler object and return
    return pt_sampler._replace(pt_state = pt_state)
