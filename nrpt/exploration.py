from functools import partial

import jax
from jax import lax

# sequentially sample multiple times, discard intermediate states
def loop_sample(kernel, init_state, n_refresh, model_args, model_kwargs):
    """
    Performs `n_refresh` Markov steps with the provided kernel, returning
    only the last state of the chain.
    
    :param kernel: An instance of `numpyro.infer.MCMC`.
    :param init_state: Starting point of the sampler.
    :param n_refresh: Number of Markov steps to take with the sampler.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :return: The last state of the chain.
    """
    return lax.scan(
        lambda state, _: (kernel.sample(state,model_args,model_kwargs), None), 
        init_state,
        length=n_refresh
    )[0]

# TODO: do iid sampling at inv_temp=0
def exploration_step(kernel, pt_state, model_args, model_kwargs):
    p_loop_sample = partial(
        loop_sample, 
        kernel, 
        n_refresh=n_refresh, 
        model_args=model_args, 
        model_kwargs=model_kwargs
    )
    return jax.vmap(p_loop_sample)(replica_states)
