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
# <=> (by relabeling b' <-> b)
#   Z(b')/Z(b) = E_b'[exp((b-b')*l(x))]^{-1}
# Furthermore, since Z(0)=1, by telescoping prop
#   Z(1) = prod_i Z(b_{i})/Z(b_{i-1})
#    = exp[sum_i logZ(b_{i})- logZ(b_{i-1})]
# so
#   logZ = logZ(1) = sum_i DlogZ_i
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
#   F :=  logsumexp[(b'-b)*l_n^{b}]
#   B := -logsumexp[(b-b')*l_n^{b'}] = -logsumexp[-(b'-b)*l_n^{b'}]
# and l_n^{(b)} := l(x_n)^{(b)}
# Idea: keep array of size (n_replicas-1) \times 3, with columns (F,B,A) such
# that the n-th entry has the log(Z(b_n)/Z(b_{n-1})) estimates
def update_estimates_vmap_fn_fwd(dlogZ_est_fwd, delta_b, log_lik_at_lower):
    return jnp.logaddexp(delta_b*log_lik_at_lower, dlogZ_est_fwd)

# note: the outer sign is handled in the average not here
def update_estimates_vmap_fn_bwd(dlogZ_est_bwd, delta_b, log_lik_at_upper):
    return jnp.logaddexp(-delta_b*log_lik_at_upper, dlogZ_est_bwd)

# online update of dlogZ estimates
def update_estimates(current_round_dlogZ_estimates, delta_inv_temp, chain_log_liks):
    assert jnp.ndim(chain_log_liks) == 1
    n_replicas = len(chain_log_liks)
    assert jnp.shape(delta_inv_temp) == (n_replicas-1,)
    assert jnp.shape(current_round_dlogZ_estimates) == (n_replicas-1, 3)

    # jax.debug.print("old dlogZ: {}", current_round_dlogZ_estimates, ordered=True)
    # jax.debug.print("dbeta: {}", delta_inv_temp, ordered=True)
    # jax.debug.print("LL: {}", chain_log_liks, ordered=True)
    
    fwd_new = jax.vmap(update_estimates_vmap_fn_fwd)(
        current_round_dlogZ_estimates[:,0], delta_inv_temp, chain_log_liks[:-1]
    )
    bwd_new = jax.vmap(update_estimates_vmap_fn_bwd)(
        current_round_dlogZ_estimates[:,1], delta_inv_temp, chain_log_liks[1:]
    )
    avg_new = 0.5*(fwd_new - bwd_new) # outer sign of the bwd is handled here 
    new_current_round_dlogZ_estimates = jnp.array(
        [fwd_new, bwd_new, avg_new]
    ).swapaxes(0,1)

    # jax.debug.print("new dlogZ: {}", new_current_round_dlogZ_estimates, ordered=True)
    return new_current_round_dlogZ_estimates

def init_estimates(n_replicas):
    return jnp.full((n_replicas-1, 3), -jnp.inf)

def empty_estimates(current_round_dlogZ_estimates):
    return jnp.full_like(current_round_dlogZ_estimates, -jnp.inf)

def fit_interpolator(inv_temp_schedule, current_round_dlogZ_estimates):
    assert jnp.ndim(inv_temp_schedule) == 1
    n_replicas = len(inv_temp_schedule)
    assert jnp.shape(current_round_dlogZ_estimates) == (n_replicas-1, 3)
    logZ_estimates = current_round_dlogZ_estimates[:,-1].cumsum()
    return interpolation.build_pchip_interpolator(
        inv_temp_schedule, 
        jnp.insert(logZ_estimates, 0, 0.)
    )
