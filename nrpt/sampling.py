from functools import partial
from operator import itemgetter
import time

import jax
from jax import lax
from jax import numpy as jnp

from numpyro import util

from automcmc import automcmc

from nrpt import adaptation
from nrpt import exploration
from nrpt import statistics
from nrpt import swaps

def n_scans_in_round(round_idx):
    return 2 ** round_idx

def total_scans(n_rounds):
    return 2 ** (n_rounds+1) - 2

def get_explorer_mean_acc_prob(kernel, pt_state):
    if not isinstance(kernel, automcmc.AutoMCMC):
        return jnp.array([1.])
    # sort by chain and exclude the first (no exploration, only iid sampling)
    replica_acc_probs = pt_state.replica_states.stats.adapt_stats.mean_acc_prob
    return replica_acc_probs[pt_state.chain_to_replica_idx[1:]]

def total_barrier(barrier_fit):
    return barrier_fit.y[-1]

def logZ_at_target(logZ_fit):
    return logZ_fit.y[-1]

def extract_sample(
        kernel, 
        model_args, 
        model_kwargs, 
        replica_states, 
        replica_idx,
        extra_fields = ('log_prior', 'log_lik', 'log_joint')
    ):
    # grab the state of the requested replica
    target_replica_state = jax.tree.map(itemgetter(replica_idx), replica_states)

    # extract the kernel's sample field, then constrain the sample
    unconstrained_sample = getattr(target_replica_state, kernel.sample_field)
    constrained_sample = kernel.postprocess_fn(model_args, model_kwargs)(
        unconstrained_sample
    )

    # add extra fields and return
    constrained_sample_with_extras = constrained_sample
    for f in extra_fields:
        constrained_sample[f] = getattr(target_replica_state, f)
    return constrained_sample_with_extras

def store_sample(kernel, model_args, model_kwargs, pt_state):
    constrained_sample_with_extras = extract_sample(
        kernel, 
        model_args, 
        model_kwargs, 
        pt_state.replica_states, 
        pt_state.chain_to_replica_idx[-1] # get the state of the replica in charge of the target chain
    )
    scan_idx = pt_state.stats.scan_idx
    samples = jax.tree.map(
        lambda x,y: x.at[scan_idx-1].set(y), # NB: scan_idx is 1-based 
        pt_state.samples, 
        constrained_sample_with_extras
    )
    return pt_state._replace(samples = samples)

def maybe_store_sample(kernel, model_args, model_kwargs, pt_state, n_rounds):
    # skip if we are not yet at the last round
    return lax.cond(
        n_rounds == pt_state.stats.round_idx,
        partial(store_sample, kernel, model_args, model_kwargs),
        util.identity,
        pt_state
    )

def print_summary_header():
    jax.debug.print(
        "  R |      Δt / ETA (s) |    Λ |      logZ | ρ (mean/max/amax) |    β₁ | α (min/mean) | AC (mean/max) \n" \
        "------------------------------------------------------------------------------------------------------",
        ordered=True
    )
    return

def print_round_summary(
        ending_round_idx, 
        explorer_mean_acc_prob, 
        pt_state,
        round_duration,
        n_rounds
    ):
    # print a header in first round
    lax.cond(ending_round_idx == 1, print_summary_header, lambda: None)

    # print row
    ETA = round_duration*total_scans(n_rounds-ending_round_idx) # using this because we need the same type of sum 2+4+8+...
    ll_ac1s = statistics.loglik_autocors(pt_state)
    replica_beta_1 = pt_state.chain_to_replica_idx[1]
    beta_1 = pt_state.replica_states.inv_temp[replica_beta_1]
    arg_max_rej = pt_state.stats.last_round_rej_probs.argmax()
    jax.debug.print(
        " {i:>2}   {tt:.1e} / {et:.1e}   {b:4.1f}   {lZ: .2e}    {rm:.2f} / {rM:.2f} / {iM:>2}   {b1:.0e}    {am:.2f} / {aM:.2f}    {cm: .2f} /{cM: .2f}",
        ordered=True,
        i=ending_round_idx,
        tt=round_duration,
        et=ETA,
        b=total_barrier(pt_state.stats.barrier_fit),
        lZ=logZ_at_target(pt_state.stats.logZ_fit),
        rm=pt_state.stats.last_round_rej_probs.mean(),
        rM=pt_state.stats.last_round_rej_probs[arg_max_rej],
        iM=arg_max_rej,
        b1=beta_1,
        am=explorer_mean_acc_prob.min(),
        aM=explorer_mean_acc_prob.mean(),
        # exclude ref as its loglik is usually non-integrable
        cm=ll_ac1s[1:].mean(),
        cM=ll_ac1s[1:].max()
    )

def postprocess_round(kernel, n_rounds, pt_state):
    ending_round_idx = pt_state.stats.round_idx

    # adapt schedule
    pt_state, barrier_fit = adaptation.adapt_schedule(pt_state)
    old_inv_temp_schedule = barrier_fit.x

    # adapt explorers
    # capture acc probs before they are deleted
    explorer_mean_acc_prob = get_explorer_mean_acc_prob(kernel, pt_state)
    pt_state = adaptation.adapt_explorers(
        kernel, pt_state, old_inv_temp_schedule
    )

    # collect statistics
    pt_state, round_duration = statistics.end_of_round_stats_update(
        pt_state, barrier_fit
    )

    # print info
    print_round_summary(
        ending_round_idx, 
        explorer_mean_acc_prob, 
        pt_state, 
        round_duration,
        n_rounds
    )

    return pt_state

def pt_scan(
        kernel, 
        pt_state, 
        n_rounds,
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions
    ):
    """
    Run a full NRPT scan -- exploration + DEO communication -- and collect
    statistics. If it is the last scan in a round, perform adaptation.
    """
    # capture logliks before exploration to estimate autocorrelation
    pre_explore_chain_log_liks = pt_state.replica_states.log_lik[
        pt_state.chain_to_replica_idx
    ]

    # exploration
    pt_state = exploration.exploration_step(
        kernel, pt_state, n_refresh, model_args, model_kwargs
    )

    # communication
    is_odd_scan = pt_state.stats.scan_idx % 2
    (
        pt_state, 
        swap_reject_probs, 
        delta_inv_temp, 
        chain_log_liks
    ) = swaps.communication_step(
        kernel, pt_state, is_odd_scan, swap_group_actions
    )

    # store sample at target chain if requested
    # no-op if this is not the last round
    if pt_state.samples is not None:
        pt_state = maybe_store_sample(
            kernel, 
            model_args, 
            model_kwargs, 
            pt_state, 
            n_rounds
        )
    
    # stats update (in particular, the iterators) 
    pt_state = statistics.post_scan_stats_update(
        pt_state, 
        swap_reject_probs, 
        delta_inv_temp, 
        chain_log_liks,
        pre_explore_chain_log_liks
    )

    # if end of run, do adaptation
    # note: scan_idx was just updated in prev line, so we need to substract 1
    # to get the index of the scan that just finished
    pt_state = lax.cond(
        pt_state.stats.scan_idx-1 == n_scans_in_round(pt_state.stats.round_idx),
        partial(postprocess_round, kernel, n_rounds),
        util.identity,
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
    # capture the starting time of the first round
    # note: `run` itself is not jitted so we can just directly call the timer
    pt_state._replace(
        stats= pt_state.stats._replace(
            last_round_start_time = time.perf_counter()
        )
    )

    # perform scans sequentially in a `lax.scan` loop
    n_scans = total_scans(n_rounds)
    pt_state = lax.scan(
        lambda pt_state, _: (
            pt_scan(
                kernel, 
                pt_state, 
                n_rounds,
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
