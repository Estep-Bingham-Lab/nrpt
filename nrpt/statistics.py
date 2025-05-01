from collections import namedtuple

import jax
from jax import numpy as jnp
from jax import lax
from jax import random

from numpyro.util import is_prng_key

###############################################################################
# statistics used for adaptation
###############################################################################

PTStats = namedtuple(
    "PTStats",
    [
        "scan_idx",
        "round_idx",
        "mean_round_rej_probs",
    ],
)
"""
A :func:`~collections.namedtuple` defining a Parallel Tempering sampler. It 
consists of the fields:

 - **scan_idx** - jhfg.
 - **round_idx** - jhfg.
 - **mean_round_rej_probs** - jhfg.
"""
