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

# TODO: do iid sampling at chain_idx=0
def exploration_step(kernel, pt_state, n_refresh, model_args, model_kwargs):
    vmap_loop_sample = jax.vmap(
        partial(
            loop_sample, 
            kernel, 
            n_refresh=n_refresh, 
            model_args=model_args, 
            model_kwargs=model_kwargs
        )
    )
    return pt_state._replace(
        replica_states = vmap_loop_sample(pt_state.replica_states)
    )
