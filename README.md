# `nrpt`

[![Build Status](https://github.com/Estep-Bingham-Lab/nrpt/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Estep-Bingham-Lab/nrpt/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Estep-Bingham-Lab/nrpt/branch/main/graph/badge.svg)](https://codecov.io/gh/Estep-Bingham-Lab/nrpt)

*A JAX-based, NumPyro-compatible implementation of Non-Reversible Parallel Tempering (NRPT)*

**Warning:** `nrpt` is under active development.


## Installation

**Optional**: if you want to run your NumPyro models on an accelerator (GPU/TPU),
make sure to 
[install the correct version of JAX](https://jax.readthedocs.io/en/latest/installation.html)
before proceeding. Otherwise, the following will install the default, CPU-only 
version of JAX.

Using pip
```bash
pip install autostep @ git+https://github.com/UBC-Stat-ML/autostep.git
pip install nrpt @ git+https://github.com/Estep-Bingham-Lab/nrpt.git
```

## Example usage

**Note:** In the following we will require the additional packages `pandas` and 
`corner`, which can be installed from common repositories.

To showcase the power of `nrpt`, we will analyze a challenging benchmark problem
described in [Ballnus et al. 2017](https://doi.org/10.1186/s12918-017-0433-1). 
The objective is to estimate the parameters of an Ordinary Differential Equation
(ODE) given noisy observations of its solution. The ODE itself was described in 
[Leonhardt et al. 2014](https://doi.org/10.1016/j.nano.2013.11.008), while the
Bayesian formulation of the inference problem is from 
[Ballnus et al. 2017](https://doi.org/10.1186/s12918-017-0433-1). The latter
shows an empirical comparison of several MCMC samplers on the ODE problem,
indicating that schemes that used Parallel Tempering were the only ones able
to accurately describe the posterior distribution. Indeed, its density is 
bimodal and features narrow ridges.

Here we will show that `nrpt` can leverage a simple yet fast gradient-free 
sampler---autoRWMH, described in 
[Liu et al. (2025)](https://arxiv.org/abs/2410.18929)---to tackle this inference task.
For brevity, we won't go into the details of the model here; be sure to check 
the references if you are curious. We also assume that you are familiar with 
NRPT. Beyond the [original paper](https://doi.org/10.1111/rssb.12464), a good 
reference is the documentation of the Julia package 
[Pigeons.jl](https://pigeons.run/stable/); `nrpt` is heavily inspired by it.

We will aim to reproduce Figure 6 in 
[Ballnus et al. 2017](https://doi.org/10.1186/s12918-017-0433-1), which shows
a corner plot of the posterior samples of the unknown parameters of the ODE.
The model has been written in NumPyro and included in `nrpt`. 
We can load it along all the required dependencies using
```python
from jax import random
from jax import numpy as jnp

from numpyro.diagnostics import print_summary

from autostep import autohmc

from nrpt import initialization
from nrpt import sampling
from nrpt import toy_examples

import numpy as np

import corner

import time

model, model_args, model_kwargs = toy_examples.mrna()
```
`model` is a python function written using NumPyro primitives. This function
takes as input the observation times -- contained in the tuple `model_args` -- 
and the noisy observations inside the `model_kwargs` dictionary.

Following the NumPyro convention, we enclose the model in an MCMC sampler. In
`nrpt`, this sampler will be used as the *explorer* in the NRPT terminology.
Currently, `nrpt` only works with the MCMC samplers of the 
[`autostep` package](https://github.com/UBC-Stat-ML/autostep). For this 
example, we will use the AutoHMC sampler with the default 32 leapfrog steps.
```python
kernel = autohmc.AutoHMC(model)
```

With the explorer in place, we can proceed to instantiate a `PTSampler` object
```python
pt_sampler = initialization.PT(
    kernel, 
    rng_key = random.key(123),
    n_rounds = 14,
    n_replicas = 15,
    n_refresh = 2,
    model_args=model_args, 
    model_kwargs=model_kwargs
)
```
Note that the model arguments are passed to the constructor. There are 
several other settings being provided:

- A JAX PRNG key, used to draw (pseudo-)random variates.
- The number of NRPT rounds is set to 14, so that a total of 
$2^{14}$=16384 samples are returned.
- The number of replicas is set to 12, which is roughly 2 times the global
barrier $\Lambda$ of the problem.
- The number of explorer refreshments within each exploration step is set to 2.
This allows us to achieve a worse-case autocorrelation of the log-likelihood 
(across replicas) of less than 0.95.

We can run NRPT typing
```python
pt_sampler = sampling.run(pt_sampler)
```
The above will produce an output similar to this
```
  R |  Δt (s) |    Λ |      logZ | ρ (mean/max/amax) |    β₁ | α (min/mean) | llAC (mean/max) 
----------------------------------------------------------------------------------------------
  1   1.2e+01    1.1   -3.62e+07    0.10 / 0.50 /  0   6e-10    0.00 / 0.30       nan / nan
  2   1.3e-01    3.4   -1.63e+03    0.31 / 1.00 / 10   5e-09    0.00 / 0.46       8.04 / 80.02
  3   2.2e-01    3.9   -1.34e+03    0.36 / 0.88 / 10   4e-07    0.43 / 0.65       2.27 / 23.02
  4   4.1e-01    4.7   -7.40e+02    0.42 / 0.94 / 10   4e-07    0.12 / 0.63       0.46 / 1.00
  5   8.7e-01    5.9   -5.76e+02    0.54 / 0.92 /  5   4e-06    0.40 / 0.69       0.70 / 1.00
  6   1.6e+00    6.3   -4.10e+02    0.57 / 0.96 /  3   3e-06    0.64 / 0.78       0.71 / 0.97
  7   2.3e+00    5.6   -3.69e+02    0.50 / 0.98 /  3   3e-06    0.72 / 0.85       0.58 / 0.93
  8   4.4e+00    5.5   -3.67e+02    0.50 / 0.82 /  4   2e-06    0.64 / 0.84       0.59 / 0.91
  9   1.0e+01    5.6   -3.71e+02    0.51 / 0.68 /  5   3e-06    0.59 / 0.81       0.68 / 0.96
 10   2.3e+01    5.8   -3.71e+02    0.53 / 0.65 /  3   2e-06    0.70 / 0.81       0.70 / 0.93
 11   3.8e+01    5.9   -3.70e+02    0.53 / 0.61 /  4   2e-06    0.67 / 0.83       0.68 / 0.93
 12   7.9e+01    5.9   -3.70e+02    0.54 / 0.59 / 10   2e-06    0.69 / 0.84       0.68 / 0.93
 13   1.6e+02    5.9   -3.71e+02    0.53 / 0.56 /  6   3e-06    0.67 / 0.83       0.69 / 0.93
 14   3.2e+02    5.9   -3.70e+02    0.54 / 0.58 /  3   3e-06    0.70 / 0.83       0.69 / 0.92
```
The figures shown here correspond, from left to right, to

- The round index
- The duration of each round (in seconds)
- Estimates of the global barrier, which at the last round is 
$\Lambda \approx 5.9$. 
- Estimates of the log-normalization constant, which in the last round gives
$\log(\mathcal{Z})\approx -370$.
- Average and worst-case swap rejection probabilities. When the average is
close to the maximum -- as in the last 6 rounds -- the ideal *equi-rejection*
condition has been approximately attained. Note that the mean rejection rate
is ~50%. This is intentional, as we chose `n_replicas=12` roughly equal to twice
the value of $\Lambda$ (which we knew from previous runs). This is the optimal
recommended number of chains in Syed et al. (2022).
- The `amax` field in the previous column indicates the chain index that 
shows the highest rejection probability. That is, when
`amax=i`, it means that the swap between chains `i` and `i+1` shows the highest
rejection rate. The next column shows the value of the first non-zero inverse 
temperature. This helps with diagnosing high rejection rates for `amax=0`, 
which is the most common index having the highest rejection probability.
- Average and worst-case explorer acceptance probabilities. If the explorer is
working correctly along the path of distributions, we expect both values
to be away from 0 and 1.
- Average and worst-case autocorrelation of the log-likelihood before and 
after the exploration steps. As described above, the number of refreshments
was set so that the maximum was below 0.95. **Note**: the estimator does not
behave well in small samples, which is why we can see autocorrelation values
larger than one in earlier rounds.

We can now extract the samples and use the `print_summary` function from
NumPyro to show a brief description of the latent values of the model
```python
>>> samples = pt_sampler.pt_state.samples
>>> print_summary(samples, group_by_chain=False)

                     mean       std    median      5.0%     95.0%     n_eff     r_hat
          lbeta     -1.85      0.93     -2.12     -2.80     -0.66    100.94      1.00
         ldelta     -1.41      0.95     -0.79     -2.58     -0.67    128.63      1.00
           lkm0      1.08      0.02      1.08      1.05      1.12    173.47      1.00
      log_joint   -322.16     68.58   -358.25   -363.05   -194.15   6742.01      1.00
        log_lik   -349.57      1.78   -349.23   -352.12   -347.01    445.06      1.00
  log_posterior   -357.20      1.98   -356.74   -360.09   -354.49    381.83      1.00
         lsigma      0.39      0.03      0.39      0.35      0.43    236.62      1.00
            lt0      0.18      0.03      0.18      0.12      0.23    275.55      1.00
```
The summary includes the model parameters as well as the log likelihood (`log_lik`),
log posterior, and log joint---corresponding to log posterior plus the log density
of the momentum. For all these quantities, we see an effective sample size (`n_eff`)
of at least 100, together with $\hat R$ diagnostics close to 1. This indicates
successfull exploration of the posterior distribution.

Finally, we can recreate the corner plot in Ballnus et al. (2017) using
```python
transformed_samples = np.array(jnp.vstack(
    [
        samples['lt0'],
        samples['lkm0'],
        samples['lbeta'],
        samples['ldelta'],
        samples['lsigma']
    ]
).swapaxes(0,1))
figure = corner.corner(
    transformed_samples,
    labels=[
        r"$\log_{10}(t_0)$",
        r"$\log_{10}(\kappa)$",
        r"$\log_{10}(\beta)$",
        r"$\log_{10}(\delta)$",
        r"$\log_{10}(\sigma)$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    plot_contours=False,
    smooth=False,
    plot_density=False,
    data_kwargs={'color': (0.0,0.6056031611752245,0.9786801175696073)}
)
figure.savefig('mrna_corner.png', bbox_inches='tight')
```
![Corner plot](./docs/_static/img/mrna_corner.png)

Note that the posterior is clearly bimodal. Not only that, the shapes
of these two modes are completely different. Moreover, there is a clear
ridge visible in the $(\log_{10}(\beta),\log_{10}(\delta))$ plot. These 
features make this problem extremely hard to tackle using traditional 
MCMC algorithms.


## References

Syed, S., Bouchard-Côté, A., Deligiannidis, G., & Doucet, A. (2022). 
[Non-reversible parallel tempering: a scalable highly parallel MCMC scheme](https://doi.org/10.1111/rssb.12464). 
*Journal of the Royal Statistical Society Series B: Statistical Methodology*, 84(2), 321-350.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024).
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). 
*arXiv preprint arXiv:2410.18929*. Accepted to ICML 2025.


## TODO

- Documentation
