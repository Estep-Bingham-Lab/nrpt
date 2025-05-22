from collections import namedtuple

from jax import numpy as jnp

###############################################################################
# statistics used for adaptation
###############################################################################

PTStats = namedtuple(
    "PTStats",
    [
        "scan_idx",
        "round_idx",
        "current_round_rej_probs",
        "last_round_rej_probs",
        "barrier_estimate"
    ],
)
"""
A :func:`~collections.namedtuple` containing statistics gathered during a PT 
run. It consists of the fields:

 - **scan_idx** - jhfg.
 - **round_idx** - jhfg.
 - **current_round_rej_probs** - jhfg.
 - **last_round_rej_probs** - jhfg.
 - **barrier_estimate** - jhfg.
"""

def init_state(n_replicas):
    return PTStats(
        1, 1, jnp.zeros(n_replicas-1), jnp.zeros(n_replicas-1), jnp.array(0.)
    )

# TODO: logZ + loglik autocorrelation
def end_of_scan_stats_update(pt_state, swap_reject_probs):
    stats = pt_state.stats

    # update swap rejection probs
    # note: scan_idx starts at 1, so the number of elements in the online 
    # estimator is scan_idx-1
    new_current_round_rej_probs = stats.current_round_rej_probs + (
        swap_reject_probs - stats.current_round_rej_probs
    ) / stats.scan_idx

    return pt_state._replace(
        stats = stats._replace(
            scan_idx = stats.scan_idx + 1,
            current_round_rej_probs = new_current_round_rej_probs
        )
    )

def end_of_round_stats_update(pt_state, barrier_estimate):
    stats = pt_state.stats
    return pt_state._replace(
        stats = stats._replace(
            scan_idx = 1,
            round_idx = stats.round_idx + 1,
            current_round_rej_probs = jnp.zeros_like(stats.current_round_rej_probs),
            last_round_rej_probs = stats.current_round_rej_probs,
            barrier_estimate = barrier_estimate
        )
    )

# TODO: logZ
# For b \neq b',
# Z(b') = E_0[L(x)^b'] = E_0[exp(b'*l(x))]
# = E_0[exp(b*l(x))exp((b'-b)*l(x))]
# = Z(b)E_b[exp((b'-b)*l(x))]
# <=>
# Z(b')/Z(b) = E_b[exp((b'-b)*l(x))]
# <=> (by relabeling b' <-> b)
# Z(b')/Z(b) = E_b'[exp((b-b')*l(x))]^{-1}
# Furthermore, since Z(0)=1, by telescoping prop
# Z(1) = prod_i Z(b_{i})/Z(b_{i-1})
# = exp[sum_i logZ(b_{i})- logZ(b_{i-1})]
# so
# logZ = logZ(1) = sum_i DlogZ_i
# where
# DlogZ_i := logZ(b_{i+1})- logZ(b_{i})]
# = log(E_b[exp((b'-b)*l(x))]) \approx -logN + logsumexp[(b'-b)*l(x_n)],   x_n~pi(b),   n=1..N
# = -log(E_b'[exp((b-b')*l(x))]) \approx logN - logsumexp[(b-b')*l(x_n)],  x_n~pi(b'),  n=1..N
# note:
# logsumexp(x[1:N]) = log(sum_i^N exp(xi)) 
# = log(exp(xN) + exp(log[sum_i^{N-1} exp(xi)])) 
# = logaddexp(xN, logsumexp(x[1:N-1]))  ---> online estimator
# and logsumexp(x[1:1]) = x[1:1]
