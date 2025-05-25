import jax
from jax import numpy as jnp

from nrpt import interpolation

# TODO: logZ
# For b \neq b',
#   Z(b') = E_0[L(x)^b'] = E_0[exp(b'*l(x))]
#    = E_0[exp(b*l(x))exp((b'-b)*l(x))]
#    = Z(b)E_b[exp((b'-b)*l(x))]
# <=>
#   Z(b')/Z(b) = E_b[exp((b'-b)*l(x))]
# <=> (by inverting and relabeling b' <-> b)
#   Z(b')/Z(b) = E_b'[exp((b-b')*l(x))]^{-1} 
#              = E_b'[exp(-(b'-b)*l(x))]^{-1}
# Furthermore, since Z(0)=1, by telescoping prop
#   Z(1) = prod_i Z(b_{i})/Z(b_{i-1})
#    = exp[sum_i logZ(b_{i})- logZ(b_{i-1})]
# Hence
#   logZ(1) = sum_i DlogZ_i
# where
#   DlogZ_i := logZ(b_{i+1})- logZ(b_{i})]
#    = log(E_b[exp((b'-b)*l(x))]) \approx -logN + logsumexp_n[(b'-b)*l(x_n^{b})],   x_n^{b}  ~pi(b),   n=1..N
#    = -log(E_b'[exp((b-b')*l(x))]) \approx logN - logsumexp_n[(b-b')*l(x_n^{b'})], x_n^{b'} ~pi(b'),  n=1..N
# note:
#   logsumexp(x[1:N]) = log(sum_i^N exp(xi)) 
#    = log(exp(xN) + sum_i^{N-1} exp(xi) ) 
#    = log(exp(xN) + exp(log[sum_i^{N-1} exp(xi)])) 
#    = logaddexp(xN, logsumexp(x[1:N-1]))
# with logsumexp(x[1:-1]) = -inf. Thus it can be estimated online.
# Note that the average cancels the logN values
#   A
#    := (-logN + logsumexp[(b'-b)*l(x_n^{b})] + logN - logsumexp[(b-b')*l(x_n^{b'})])/2
#     = (logsumexp[(b'-b)*l(x_n^{b})] - logsumexp[(b-b')*l(x_n^{b'})])/2
# Let
#   L_i := logsumexp_n[ (b_i-b_{i-1})*l_n^{(b_{i-1})}]
#   R_i := logsumexp_n[-(b_i-b_{i-1})*l_n^{(b_i)}]
# and l_n^{(b)} := l(x_n)^{(b)}. Then
#   A_i = 0.5(L_i - R_i)
#   logZ(b_i) = cumsum(A)_i
# Idea: keep array of size (n_replicas-1) x 2, with columns (L,R) such
# that the n-th entry has the log(Z(b_n)/Z(b_{n-1})) estimates
def update_estimates_vmap_fn(dlogZ_est, delta_b, log_lik):
    return jnp.logaddexp(delta_b*log_lik, dlogZ_est)

# online update of dlogZ estimates
def update_estimates(current_round_dlogZ_estimates, delta_inv_temp, chain_log_liks):
    assert jnp.ndim(chain_log_liks) == 1
    n_replicas = len(chain_log_liks)
    assert jnp.shape(delta_inv_temp) == (n_replicas-1,)
    assert jnp.shape(current_round_dlogZ_estimates) == (n_replicas-1, 2)
    
    fwd_new = jax.vmap(update_estimates_vmap_fn)(
        current_round_dlogZ_estimates[:,0], delta_inv_temp, chain_log_liks[:-1]
    )
    bwd_new = jax.vmap(update_estimates_vmap_fn)(
        current_round_dlogZ_estimates[:,1], -delta_inv_temp, chain_log_liks[1:]
    )
    return jnp.array([fwd_new, bwd_new]).swapaxes(0,1)

def init_estimates(n_replicas):
    return jnp.full((n_replicas-1, 2), -jnp.inf)

def empty_estimates(current_round_dlogZ_estimates):
    return jnp.full_like(current_round_dlogZ_estimates, -jnp.inf)

def fit_interpolator(inv_temp_schedule, current_round_dlogZ_estimates):
    assert jnp.ndim(inv_temp_schedule) == 1
    n_replicas = len(inv_temp_schedule)
    assert jnp.shape(current_round_dlogZ_estimates) == (n_replicas-1, 2)
    logZ_estimates = 0.5*(
        current_round_dlogZ_estimates[:,0] - current_round_dlogZ_estimates[:,1]
    ).cumsum()
    return interpolation.build_pchip_interpolator(
        inv_temp_schedule, 
        jnp.insert(logZ_estimates, 0, 0.)
    )
