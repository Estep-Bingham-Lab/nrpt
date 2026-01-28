from collections import namedtuple
from collections.abc import Iterable
from typing import Optional
import warnings

import jax
from jax import numpy as jnp
from jax import lax
from jax import random
from jax.typing import ArrayLike

from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import is_prng_key

from nrpt import sampling
from nrpt import statistics
from nrpt import utils

###############################################################################
# type definitions
###############################################################################

PTSampler = namedtuple(
    "PTSampler",
    [
        "kernel",
        "pt_state",
        "n_rounds",
        "n_refresh",
        "model_args",
        "model_kwargs",
        "swap_group_actions",
        "excluded_latent_vars",
        "thinning"
    ],
)
"""
A :func:`~collections.namedtuple` defining a Parallel Tempering sampler. It 
consists of the fields:

 - **kernel** - the exploration kernel.
 - **pt_state** - an instance of :class:`PTState`.
 - **n_rounds** - number of rounds to run.
 - **n_refresh** - number of refreshments per exploration phase.
 - **model_args** - optional model arguments.
 - **model_kwargs** - optional model keyword arguments.
 - **swap_group_actions** - permutations of the replica indexes proposed at
     odd and even communication steps.
 - **excluded_latent_vars** - a `frozenset` of latent variables to exclude when
     storing samples.
 - **thinning** - number of scans to skip when storing samples.
"""

PTState = namedtuple(
    "PTState",
    [
        "replica_states",
        "replica_to_chain_idx",
        "chain_to_replica_idx",
        "rng_key",
        "stats",
        "samples"
    ],
)
"""
A :func:`~collections.namedtuple` defining the state of a Parallel Tempering
ensemble. It consists of the fields:

 - **replica_states** - pytree of vectorized kernel states for each replica.
 - **replica_to_chain_idx** - chain index assigned to a given replica.
 - **chain_to_replica_idx** - replica index in charge of a given chain.
 - **rng_key** - a JAX PRNG key.
 - **stats** - an instance of :class:`statistics.PTStats`.
 - **samples** - a pythree for holding latent variable samples if collection
     is activated.
"""

###############################################################################
# constructors
###############################################################################

# define DEO swap actions
# maximal proposed change if all swaps are accepted
# examples:
# n_replicas=5:
# E: [0,1,2,3,4] -> identity perm
#    [1,0,3,2,4] -> all swaps acc perm == even group action
# O: [0,1,2,3,4] -> identity perm
#    [0,2,1,4,3] -> all swaps acc perm == odd group action 
# n_replicas=6:
# E: [0,1,2,3,4,5] -> identity perm
#    [1,0,3,2,5,4] -> all swaps acc perm == even group action
# O: [0,1,2,3,4,5] -> identity perm
#    [0,2,1,4,3,5] -> all swaps acc perm == odd group action
# n_replicas=7:
# E: [0,1,2,3,4,5,6] -> identity perm
#    [1,0,3,2,5,4,6] -> all swaps acc perm == even group action
# O: [0,1,2,3,4,5,6] -> identity perm
#    [0,2,1,4,3,6,5] -> all swaps acc perm == odd group action
def init_swap_group_actions(n_replicas):
    idx_even_group_action = [
        (
            i - 1 if i%2 else
            i + 1 if i<n_replicas-1 else
            i
        ) 
        for i in range(n_replicas)
    ]
    idx_odd_group_action = [
        (
            min(i + 1, n_replicas-1) if i%2 else
            i - 1 if i > 0 else
            i      
        ) 
        for i in range(n_replicas)
    ]
    return jnp.array([idx_even_group_action, idx_odd_group_action])

def validate_initial_replica_states(kernel, replica_states):
    if not hasattr(kernel, "update_log_joint"):
        return replica_states

    # compute log joint at the initial point
    replica_states = jax.vmap(kernel.update_log_joint)(
        replica_states, replica_states.base_precond_state
    )

    # check everything remains finite
    valid_initial_states = jax.tree.all(
        jax.tree.map(
            lambda x: jnp.all(jnp.isfinite(x)), 
            replica_states
        )
    )

    # abort if not
    if not valid_initial_states:
        raise RuntimeError(
            f"Found invalid initial replica states; dumping below\n{replica_states}"
        )
    
    # handle the case where the log likelihood term is int32(0), which occurs 
    # when a numpyro model has no `obs` sample statements
    if not jnp.issubdtype(replica_states.log_lik.dtype, jnp.floating):
        warnings.warn(
            "The model may not have a likelihood component so running Parallel " \
            "Tempering may be pointless (there is nothing to temper)."
        )

    return replica_states

def init_replica_states(
        kernel, 
        rng_key, 
        n_replicas, 
        init_params, 
        model_args, 
        model_kwargs
    ):
    # use the kernel initialization to get a prototypical state
    prototype_init_state = kernel.init(
        rng_key, 0, init_params, model_args, model_kwargs
    )

    # extend the prototypical state to all replicas
    replica_states = jax.tree.map(
        lambda xs: (
            random.split(xs, n_replicas) 
            if is_prng_key(xs)
            else lax.broadcast(xs, (n_replicas,))
        ),
        prototype_init_state
    )

    # validate initial replica state and return if all good
    return validate_initial_replica_states(kernel, replica_states)

def init_schedule_log(replica_states, n_replicas):
    """
    Find a log2-linear schedule with the property that
    `beta[1]*initial_log_lik` is close to 0.
    """
    log2_beta_1 = jnp.clip(
        # aim for beta1 ~ [max_i(|loglik_i|)]^{-1}
        -jnp.log2(jnp.abs(replica_states.log_lik).max()),
        # don't go under the precision
        jnp.finfo(replica_states.log_prior.dtype).minexp,
        # don't go over the exponent we would get with linear schedule
        -jnp.log2(n_replicas-1).astype(replica_states.log_prior.dtype)
    )
    return jnp.insert(
        2**jnp.linspace(log2_beta_1, 0, n_replicas-2,False), 
        jnp.array([0, n_replicas]), 
        jnp.array([0, 1])
    )

def init_schedule(replica_states, n_replicas, initial_schedule):
    if isinstance(initial_schedule, ArrayLike):
        assert (
            initial_schedule.shape == (n_replicas,) and
            initial_schedule[ 0] == 0 and 
            initial_schedule[-1] == 1 and 
            utils.is_increasing(initial_schedule)
        )
        inv_temp_schedule = initial_schedule
    elif initial_schedule == "linear":
        inv_temp_schedule = jnp.linspace(0,1,n_replicas)
    elif initial_schedule == "log":
        inv_temp_schedule = init_schedule_log(replica_states, n_replicas)
    else:
        raise ValueError(
            f"Don't know how to handle `initial_schedule`={initial_schedule}"
        )
    
    # update the replica states and return
    # note: this assume that the initial permutation is the identity
    return replica_states._replace(inv_temp=inv_temp_schedule)


def init_samples_container(
        kernel, 
        n_rounds, 
        model_args, 
        model_kwargs, 
        replica_states,
        excluded_latent_vars,
        thinning
    ):
    # grab a generic replica state
    constrained_sample_with_extras = sampling.extract_sample(
        kernel, 
        model_args, 
        model_kwargs, 
        replica_states, 
        0,
        excluded_latent_vars
    )

    # use the above as template to create container
    n_stored_samples = sampling.n_scans_in_round(n_rounds) // thinning
    return jax.tree.map(
        lambda x: jnp.empty_like(
            x, 
            shape = (n_stored_samples, *x.shape),
        ),
        constrained_sample_with_extras
    )

def init_pt_state(
        kernel, 
        rng_key, 
        n_replicas, 
        n_rounds,
        init_params,
        model_args, 
        model_kwargs, 
        collect_samples,
        initial_schedule,
        excluded_latent_vars
    ):
    rng_key, init_key = random.split(rng_key)

    # init_replica states
    replica_states = init_replica_states(
        kernel, init_key, n_replicas, init_params, model_args, model_kwargs
    )

    # init permutations
    chain_to_replica_idx = jnp.arange(n_replicas) # init to identity permutation
    replica_to_chain_idx = jnp.arange(n_replicas) # id = argsort(id)

    # init schedule
    replica_states = init_schedule(replica_states, n_replicas, initial_schedule)

    # init statistics
    stats = statistics.init_stats(n_replicas)

    # maybe build samples container
    thinning = int(collect_samples)
    if collect_samples:
        samples = init_samples_container(
            kernel, 
            n_rounds, 
            model_args, 
            model_kwargs,
            replica_states,
            excluded_latent_vars,
            thinning
        )
    else: 
        samples = None

    # build the state object
    pt_state = PTState(
        replica_states, 
        replica_to_chain_idx, 
        chain_to_replica_idx, 
        rng_key, 
        stats,
        samples
    )

    return pt_state, thinning


def PT(
        kernel: MCMCKernel, 
        rng_key: ArrayLike, 
        n_replicas: int = 10, 
        n_rounds: int = 10, 
        n_refresh: int = 3,
        init_params: Optional[dict] = None,
        model_args: tuple = (), 
        model_kwargs: dict = {},
        collect_samples: bool | int = True,
        initial_schedule: str | ArrayLike = "log",
        excluded_latent_vars: Iterable = {}
    ) -> PTSampler:
    """
    Initialize a :class:`PTSampler`.
    
    :param kernel: An MCMC sampler used as explorer.
    :param rng_key: A JAX PRNG key.
    :param n_replicas: Number of replicas to use.
    :param n_rounds: Number of rounds to run.
    :param n_refresh: Number of :func:`kernel.sample` calls to execute during
        each exploration phase.
    :param init_params: Optional initial parameters to initialize the replicas.
        Must be given in unconstrained form.
    :param model_args: Optional model arguments.
    :param model_kwargs: Optional model keyword arguments
    :param collect_samples: Whether to collect samples. If `int` and non-zero,
        it is assumed to request thinning of the last round samples.  
    :param initial_schedule: Settings for building the initial inverse 
        temperature schedule. Can be a `str` equal to `'linear'` for linearly
        spaced schedule or `'log'` for logarithmically spaced (default). It can
        also be a vector of length `n_replicas` corresponding to a valid 
        inverse temperature schedule.
    :param excluded_latent_vars: Optional iterable of latent variables to
        exclude when storing a sample. This is useful to keep GPU memory in
        check by discarding nuisance variables that can be nonetheless massive
        in terms of memory usage.
    :return: An initialized :class:`PTSampler`.
    """
    swap_group_actions = init_swap_group_actions(n_replicas)
    pt_state, thinning = init_pt_state(
        kernel, 
        rng_key, 
        n_replicas, 
        n_rounds,
        init_params,
        model_args, 
        model_kwargs, 
        collect_samples,
        initial_schedule,
        excluded_latent_vars
    )
    return PTSampler(
        kernel, 
        pt_state, 
        n_rounds,
        n_refresh, 
        model_args, 
        model_kwargs,
        swap_group_actions,
        frozenset(excluded_latent_vars),
        thinning
    )
