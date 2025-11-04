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
    n_replicas = 12,
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

We can run NRPT typing (takes about 6 minutes on an Nvidia RTX 2000 Ada 
generation laptop GPU)
```python
pt_sampler = sampling.run(pt_sampler)
```
The above will produce an output similar to this
```
  R |      Δt / ETA (s) |    Λ |      logZ | ρ (mean/max/amax) |    β₁ | α (min/mean) | AC (mean/max) 
------------------------------------------------------------------------------------------------------
  1   1.2e+01 / 2.0e+05    1.1   -3.62e+07    0.10 / 0.50 /  0   6e-10    0.00 / 0.30     nan / nan
  2   1.2e-01 / 9.4e+02    2.9   -1.24e+03    0.26 / 1.00 / 10   1e-08    0.12 / 0.42     11.39 / 80.42
  3   1.6e-01 / 6.5e+02    5.1   -8.90e+05    0.47 / 1.00 / 10   9e-09    0.00 / 0.47    -0.04 / 1.00
  4   3.2e-01 / 6.5e+02    5.0   -7.27e+02    0.45 / 0.81 /  9   6e-08    0.59 / 0.74     0.44 / 1.00
  5   4.7e-01 / 4.8e+02    6.6   -5.48e+02    0.60 / 1.00 /  9   3e-07    0.60 / 0.77     0.62 / 1.00
  6   9.9e-01 / 5.1e+02    6.2   -4.42e+02    0.56 / 0.98 /  4   7e-06    0.60 / 0.79     0.67 / 1.07
  7   1.6e+00 / 4.2e+02    6.2   -3.84e+02    0.56 / 0.96 /  3   6e-06    0.69 / 0.82     0.73 / 0.96
  8   2.6e+00 / 3.3e+02    5.7   -3.67e+02    0.52 / 0.82 /  3   5e-06    0.71 / 0.85     0.64 / 0.88
  9   5.0e+00 / 3.1e+02    5.6   -3.68e+02    0.51 / 0.67 /  5   4e-06    0.69 / 0.84     0.65 / 0.94
 10   1.2e+01 / 3.5e+02    5.8   -3.70e+02    0.53 / 0.60 /  5   3e-06    0.69 / 0.82     0.66 / 0.93
 11   2.2e+01 / 3.1e+02    5.9   -3.70e+02    0.54 / 0.61 /  8   3e-06    0.68 / 0.80     0.67 / 0.91
 12   4.2e+01 / 2.5e+02    5.9   -3.70e+02    0.54 / 0.63 /  3   3e-06    0.68 / 0.83     0.69 / 0.92
 13   8.4e+01 / 1.7e+02    5.9   -3.70e+02    0.54 / 0.55 /  1   3e-06    0.68 / 0.82     0.68 / 0.92
 14   1.7e+02 / 0.0e+00    5.9   -3.70e+02    0.53 / 0.56 /  4   3e-06    0.69 / 0.80     0.68 / 0.92
```
The figures shown here correspond, from left to right, to

- The round index
- The duration of the round and the estimated time until sampling is completed
(both in seconds).
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
- Average and worst-case autocorrelation (AC) of the log-likelihood before and 
after the exploration steps. As described above, the number of refreshments
was set so that the maximum was below 0.95. **Note**: the estimator does not
behave well in small samples, which is why we can see autocorrelation values
larger than one in earlier rounds.

We can now extract the samples and use the `print_summary` function from
NumPyro to show a brief description of the latent values of the model
```python
samples = pt_sampler.pt_state.samples
print_summary(samples, group_by_chain=False)
```
```
                 mean       std    median      5.0%     95.0%     n_eff     r_hat
      lbeta     -1.56      0.99     -0.81     -2.87     -0.67    161.73      1.00
     ldelta     -1.74      0.97     -2.05     -2.82     -0.67    156.57      1.00
       lkm0      1.08      0.02      1.08      1.05      1.12    305.61      1.00
  log_joint   -320.25     69.73   -358.33   -362.76   -194.51   9912.67      1.00
    log_lik   -349.62      1.70   -349.36   -352.04   -347.11    488.56      1.00
  log_prior     -7.63      0.36     -7.53     -7.89     -7.36    453.69      1.00
     lsigma      0.40      0.02      0.40      0.36      0.44    335.41      1.00
        lt0      0.18      0.03      0.18      0.14      0.23    308.53      1.00
```
The summary includes the model parameters as well as the log prior, log likelihood,
and log joint---corresponding to log posterior plus the log density
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
