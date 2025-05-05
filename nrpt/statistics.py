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

def end_of_scan_stats_update(pt_state, swap_reject_probs):
    stats = pt_state.stats
    new_scan_idx = stats.scan_idx + 1
    new_mean_round_rej_probs = stats.mean_round_rej_probs + (
        swap_reject_probs - stats.mean_round_rej_probs
    ) / new_scan_idx
    return pt_state._replace(
        stats = stats._replace(
            scan_idx = new_scan_idx,
            mean_round_rej_probs = new_mean_round_rej_probs
        )
    )

# TODO: schedule adaptation
def end_of_round_stats_update(pt_state):
    stats = pt_state.stats
    return pt_state._replace(
        stats = stats._replace(
            scan_idx = 1,
            round_idx = stats.round_idx + 1
        )
    )
