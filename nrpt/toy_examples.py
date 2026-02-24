from jax import lax
from jax.scipy.special import betaln, digamma

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

##############################################################################
# Conjugate normal model, true posterior is Normal(mu, v) with
#   mu := m v
#   v  := (1 + sigma0^{-2})^{-1}
##############################################################################

def toy_conjugate_normal(
        d = jnp.int32(3), 
        m = jnp.float32(2.), 
        sigma0 = jnp.float32(2.)
    ):
    def model(sigma0, y):
        with numpyro.plate('dim', len(y)):
            x = numpyro.sample('x', dist.Normal(scale=sigma0))
            numpyro.sample('obs', dist.Normal(x), obs=y)

    # inputs
    y = jnp.full((d,), m)
    model_args = (sigma0, y)
    model_kwargs = {}
    return model, model_args, model_kwargs

##############################################################################
# classic Pigeons toy unidentifiable example
##############################################################################

def __toy_unid(n_flips, n_heads=None):
    p1 = numpyro.sample('p1', dist.Uniform())
    p2 = numpyro.sample('p2', dist.Uniform())
    p = numpyro.deterministic('p', p1*p2)
    numpyro.sample('n_heads', dist.Binomial(n_flips, p), obs=n_heads)

def toy_unid_example(n_heads=50, n_flips=100):
    model = __toy_unid
    model_args = (n_flips,)
    model_kwargs={"n_heads": n_heads}
    return (model, model_args, model_kwargs)

def binomln(n, k):
    # https://stackoverflow.com/a/21775412/5443023
    # Assumes binom(n, k) >= 0
    return -betaln(1 + n - k, 1 + k) - jnp.log(n + 1)

# https://github.com/Julia-Tempering/Pigeons.jl/blob/main/test/supporting/analytic_solutions.jl
def toy_unid_exact_logZ(n,y,inv_temp):
    A = inv_temp*binomln(n,y)
    B = betaln(inv_temp*y+1,inv_temp*(n-y)+1)
    C = jnp.log(digamma(inv_temp*n+2) - digamma(inv_temp*y+1))
    return A+B+C

##############################################################################
# eight schools example
##############################################################################
 
def __eight_schools(sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('school_plate', len(sigma)):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

def eight_schools_example():
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    model = __eight_schools
    model_args = (sigma,)
    model_kwargs={'y': y}
    return (model, model_args, model_kwargs)

###############################################################################
# mRNA transfection example from Ballnus et al. (2017, dataset M1a)
# https://doi.org/10.1186/s12918-017-0433-1
###############################################################################

# Expected value of the concentration at a time after the start of reaction
#   m(t) = (km0/(delta-beta))[exp(-beta(t-t0)) - exp(-delta(t-t0))]
# for km0,delta,beta,dt>0 and dt=t-t0. Note that this expression is
#   - always non-negative
#   - invariant to switching delta <-> beta, which is the reason for the 
#     unidentifiability of the model
# To avoid loss of precision from the `exp` difference, we can rewrite as
#   (e^{-betaT}-e^{-deltaT}) 
#   = e^{-beta T}[1-exp{-(delta-beta)T}] 
#   =-e^{-beta T}expm1{-(delta-beta)T}
# or as
#   = e^{-deltaT}[exp{ (delta-beta)T}-1] 
#   = e^{-deltaT}expm1{ (delta-beta)T}
# Both expressions are valid. We use the first one when delta>beta, and the
# second one otherwise. Since beta,delta>0, this approach ensures that both
# exponentials have negative arguments and thus never blow up. Finally, we
# can write m(t) using a single expression
#   m(t) = -(km0/|delta-beta|) e^{-min(delta,beta)T}expm1{ -|delta-beta|T}
# This expression is also non-negative (the expm1 term is <=0) and invariant to
# switching delta <-> beta
def _mrna_mean_fn(km0, delta, beta, dt):
    abs_diff = lax.abs(delta-beta)
    exp_prod = lax.exp(-lax.min(delta,beta)*dt)*lax.expm1(-abs_diff*dt)
    return -(km0/abs_diff)*exp_prod

def _mrna(ts, ys=None):
    # priors
    lt0 = numpyro.sample('lt0', dist.Uniform(-2,1))
    lkm0 = numpyro.sample('lkm0', dist.Uniform(-5,5))
    lbeta = numpyro.sample('lbeta', dist.Uniform(-5,5))
    ldelta = numpyro.sample('ldelta', dist.Uniform(-5,5))
    lsigma = numpyro.sample('lsigma', dist.Uniform(-2,2))

    # transformed vars
    t0 = 10. ** lt0
    km0 = 10. ** lkm0
    beta = 10. ** lbeta
    delta = 10. ** ldelta
    sigma = 10. ** lsigma

    # likelihood: conditionally indep normals with varying means
    rel_ts = lax.max(jnp.zeros_like(ts), ts-t0)
    ms = _mrna_mean_fn(km0, delta, beta, rel_ts)
    with numpyro.plate('dim', len(ts)):
        numpyro.sample('obs', dist.Normal(ms, scale=sigma), obs=ys)

def mrna():
    """
    mRNA transfection time series model (dataset M1a) described in 
    Ballnus et al. (2017).
    """
    import pandas as pd
    
    # Load the data
    dta = pd.read_csv(
        'https://raw.githubusercontent.com/Julia-Tempering/Pigeons.jl/refs/heads/main/examples/data/Ballnus_et_al_2017_M1a.csv',
        header=None
    )
    model = _mrna
    model_args = (jnp.array(dta[0]),)
    model_kwargs={'ys': jnp.array(dta[1])}
    return (model, model_args, model_kwargs)
