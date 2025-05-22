from collections import namedtuple

from jax import numpy as jnp

from numpyro import infer

def is_sorted(x):
    assert jnp.ndim(x) == 1
    return jnp.all(x[1:] >= x[:-1])

#######################################
# Dummy target+explorer to test swaps
#######################################

NoneState = namedtuple('NoneState', ['log_lik', 'inv_temp'])
class NoneExplorer(infer.mcmc.MCMCKernel):
    def __init__(self, *args, **kwargs):
        self._potential_fn = lambda _: jnp.float32(0)

    def init(self, *args, **kwargs):
        return NoneState(log_lik=jnp.float32(0), inv_temp=jnp.float32(1))
    
    def sample(self, state, *args):
        return state
    
    def adapt(self, state, *args, **kwargs):
        return state
