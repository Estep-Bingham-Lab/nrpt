from functools import partial
from operator import itemgetter

import jax
from jax import lax
from jax import numpy as jnp

from numpyro.handlers import seed, trace
from numpyro.infer import util

from automcmc import automcmc

# sample from prior using NumPyro handlers
def sample_from_prior(model, model_args, model_kwargs, rng_key):
    traced_model = trace(seed(model, rng_seed=rng_key))
    unconstrain = partial(util.unconstrain_fn, model, model_args, model_kwargs)
    exec_trace = traced_model.get_trace(*model_args, **model_kwargs)
    params = {
        name: site["value"] for name, site in exec_trace.items() 
        if site["type"] == "sample" and not site["is_observed"]
    }
    return unconstrain(params)

# make a new kernel state by refreshing the sample field from the prior
def sample_iid_kernel_state(
        kernel,
        model_args, 
        model_kwargs, 
        kernel_state
    ):
    # make PRNG keys
    new_rng_key, x_key, precond_key, aux_key = jax.random.split(
        kernel_state.rng_key, 4
    )

    # sample x from the prior in unconstrained space, then store along with
    # the updated master PRNG key
    unconstrained_sample = sample_from_prior(
        kernel.model, model_args, model_kwargs, x_key
    )
    kernel_state = kernel_state._replace(
        rng_key=new_rng_key, 
        **{kernel.sample_field: unconstrained_sample}
    )

    # To match the behavior of MCMC exploration, we must also refresh the
    # auxiliary variable (and thus also possibly the preconditioner)
    precond_state = kernel.preconditioner.maybe_alter_precond_state(
        kernel_state.base_precond_state, precond_key
    )
    kernel_state = kernel.refresh_aux_vars(
        aux_key, kernel_state, precond_state
    )

    # update logprobs and return
    return kernel.update_log_joint(kernel_state, precond_state)

# sequentially sample multiple times, discard intermediate states
def loop_sample(kernel, n_refresh, model_args, model_kwargs, kernel_state):
    """
    Performs `n_refresh` Markov steps with the provided kernel, returning
    only the last state of the chain.
    
    :param kernel: An instance of `numpyro.infer.MCMC`.
    :param n_refresh: Number of Markov steps to take with the sampler.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param kernel_state: Starting point of the sampler.
    :return: The last state of the chain.
    """
    return lax.scan(
        lambda state, _: (kernel.sample(state,model_args,model_kwargs), None),
        kernel_state,
        length=n_refresh
    )[0]

# Note on implementation:
# An alternative to the current approach would be to have a `cond`-based 
# dispatcher at the chain level that sends the ref to iid sampling and all
# others to loop sampling. The problem is that when that dispatcher is vmapped,
# the `cond` is replaced with a `jnp.where` so both branches are executed for
# all replicas. Whereas with the current approach, we only dispatch the loop
# to all replicas, but the iid sampling is guaranteed to be carried out only
# once, which cuts down on (expensive) target evaluations 
def exploration_step(kernel, pt_state, n_refresh, model_args, model_kwargs):
    """
    Exploration step.
    """
    # start by vmapping `loop_sample` over all replicas    
    vmap_explore = jax.vmap(
        partial(loop_sample, kernel, n_refresh, model_args, model_kwargs)
    )
    replica_states = vmap_explore(pt_state.replica_states)

    # if there is a model available, draw an iid sample from reference and
    # update the state of the replica handling it
    if kernel.model is not None:
        # grab the state of the replica handling the reference
        ref_replica_idx = pt_state.chain_to_replica_idx[0]
        ref_replica_state = jax.tree.map(
            itemgetter(ref_replica_idx), replica_states
        )

        # draw the iid sample and update the corresponding replica state
        ref_replica_state = sample_iid_kernel_state(
            kernel,
            model_args, 
            model_kwargs, 
            ref_replica_state
        )
        replica_states = jax.tree.map(
            lambda col, x: col.at[ref_replica_idx].set(x),
            replica_states,
            ref_replica_state
        )
    
    # update the pt state and return
    return pt_state._replace(replica_states=replica_states) 
