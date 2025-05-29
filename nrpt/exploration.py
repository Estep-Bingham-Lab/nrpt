from functools import partial

import jax
from jax import lax
from jax import numpy as jnp

from numpyro.handlers import seed, trace
from numpyro.infer import util

from autostep import autostep

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
    if not isinstance(kernel, autostep.AutoStep):
        return kernel_state

    # sample from the prior in unconstraiend space
    new_rng_key, iid_key = jax.random.split(kernel_state.rng_key)
    unconstrained_sample = sample_from_prior(
        kernel.model, model_args, model_kwargs, iid_key
    )

    # store sample
    kernel_state = kernel_state._replace(
        rng_key=new_rng_key, **{kernel.sample_field: unconstrained_sample}
    )

    # update logprobs and return
    return kernel.update_log_joint(
        kernel_state, kernel_state.base_precond_state
    )

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

def explore(
        kernel, 
        n_refresh, 
        model_args, 
        model_kwargs, 
        kernel_state, 
        chain_idx
    ):
    """
    Sample iid from prior if replica is targeting the reference chain; 
    otherwise, take `n_refresh` MCMC steps.
    """
    return lax.cond(
        jnp.logical_and(kernel.model is not None, chain_idx == 0),
        partial(sample_iid_kernel_state, kernel, model_args, model_kwargs),
        partial(loop_sample, kernel, n_refresh, model_args, model_kwargs),
        kernel_state
    )

def exploration_step(kernel, pt_state, n_refresh, model_args, model_kwargs):
    """
    Exploration step vectorized over replicas.
    """
    vmap_explore = jax.vmap(
        partial(explore, kernel, n_refresh, model_args, model_kwargs)
    )
    return pt_state._replace(
        replica_states = vmap_explore(
            pt_state.replica_states, pt_state.replica_to_chain_idx
        )
    )
