from jax import numpy as jnp

def is_increasing(x):
    assert jnp.ndim(x) == 1
    return jnp.all(x[1:] >= x[:-1])