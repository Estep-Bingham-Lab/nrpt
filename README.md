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
pip install automcmc @ git+https://github.com/UBC-Stat-ML/automcmc.git
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

Here we will show that `nrpt` can leverage an automatically tuned 
sampler described in [Liu et al. (2025)](https://arxiv.org/abs/2410.18929)
to tackle this inference task.
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

from automcmc import autohmc

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
[`automcmc` package](https://github.com/UBC-Stat-ML/automcmc). For this 
example, we will use the AutoHMC sampler with the default 32 leapfrog steps.
```python
kernel = autohmc.AutoHMC(model)
```

With the explorer in place, we can proceed to instantiate a `PTSampler` object
```python
pt_sampler = initialization.PT(
    kernel, 
    rng_key = random.key(1),
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
$2^{15}$=16384 samples -- corresponding to the last round -- are returned.
- The number of replicas is set to 15, which is roughly 2.5 times the global
barrier $\Lambda$ of the problem. This is the ideal minimum number of chains
needed for NRPT to correctly bootstrap itself via adaptation. The number of
replicas can be increased past this point until either device memory is 
exhausted or until significant speed deterioration is observed. Of course, 
since $\Lambda$ is a priori unknown, setting `n_replicas` requires some iteration.
- The number of explorer refreshments within each exploration step is set to 2.
This allows us to achieve a worse-case autocorrelation of the log-likelihood 
(across replicas) of less than 0.95.

We can run NRPT typing (takes less than 5 minutes on an Nvidia RTX 2000 Ada
generation laptop GPU)
```python
pt_sampler = sampling.run(pt_sampler)
```
The above will produce an output similar to this
```
  R |        Δt |       ETA |    Λ |      logZ | ρ (mean/max/amax) | newβ₁ | α (min/mean) | AC (mean/max) 
----------------------------------------------------------------------------------------------------------
  1     0:00:12    54:56:08    3.9   -8.56e+02    0.28 / 0.92 / 12   1e-19    0.00 / 0.45    -0.30 / 1.33
  2     0:00:00     0:06:56    4.4   -7.83e+02    0.32 / 0.95 /  2   1e-05    0.32 / 0.56     0.54 / 1.00
  3     0:00:00     0:08:40    6.3   -7.48e+02    0.45 / 0.79 /  8   6e-08    0.37 / 0.61     0.42 / 1.01
  4     0:00:00     0:05:44    8.4   -4.73e+02    0.60 / 0.91 /  8   7e-06    0.47 / 0.67     0.62 / 1.06
  5     0:00:00     0:05:31    6.2   -4.08e+02    0.44 / 0.77 /  3   4e-09    0.56 / 0.75     0.78 / 0.98
  6     0:00:01     0:04:55    5.1   -3.68e+02    0.36 / 0.99 /  4   1e-06    0.77 / 0.88     0.47 / 0.87
  7     0:00:01     0:03:02    5.2   -3.66e+02    0.37 / 0.93 /  4   5e-08    0.79 / 0.92     0.58 / 0.94
  8     0:00:02     0:04:44    5.6   -3.66e+02    0.40 / 0.56 /  5   5e-07    0.70 / 0.90     0.65 / 0.97
  9     0:00:04     0:03:39    6.2   -3.71e+02    0.44 / 0.65 /  6   7e-07    0.72 / 0.88     0.68 / 0.93
 10     0:00:08     0:03:59    6.3   -3.70e+02    0.45 / 0.52 / 12   3e-07    0.70 / 0.87     0.67 / 0.93
 11     0:00:13     0:02:59    6.2   -3.71e+02    0.44 / 0.50 /  6   3e-07    0.72 / 0.88     0.69 / 0.92
 12     0:00:27     0:02:41    6.3   -3.71e+02    0.45 / 0.50 /  9   5e-07    0.72 / 0.87     0.70 / 0.92
 13     0:00:52     0:01:44    6.1   -3.70e+02    0.44 / 0.48 / 11   4e-07    0.70 / 0.87     0.69 / 0.92
 14     0:01:43     0:00:00    6.2   -3.70e+02    0.44 / 0.46 / 10   4e-07    0.72 / 0.89     0.70 / 0.92
```
From left to right, the figures shown here correspond to:

- The round index
- The duration of the round
- The estimated time until sampling is completed. Note that this is very 
inaccurate in the earlier rounds. It begins to stabilize roughly after round 7,
depending on the complexity of the target.
- Estimate of the global barrier, which at the last round is 
$\Lambda \approx 6.2$. 
- Estimate of the log-normalization constant, which in the last round gives
$\log(\mathcal{Z})\approx -370$.
- Average and worst-case swap rejection probabilities. When the average is
close to the maximum -- as in the last 3 rounds -- the ideal *equi-rejection*
condition has been approximately attained.
- The `amax` field in the previous column indicates the chain index that 
shows the highest rejection probability. That is, when
`amax=i`, it means that the swap between chains `i` and `i+1` shows the highest
rejection rate. The next column shows the updated value of the first non-zero 
inverse temperature. This helps with diagnosing high rejection rates for `amax=0`.
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
      lbeta     -1.51      0.92     -0.80     -2.60     -0.66    185.15      1.00
     ldelta     -1.81      1.06     -2.09     -3.07     -0.66    148.99      1.00
       lkm0      1.08      0.02      1.08      1.05      1.12    347.15      1.00
    log_lik   -349.62      1.80   -349.29   -352.20   -347.06    466.17      1.00
  log_prior     -7.65      0.42     -7.52     -7.94     -7.37    382.82      1.00
     lsigma      0.40      0.02      0.39      0.36      0.44    241.64      1.01
        lt0      0.18      0.03      0.18      0.13      0.22    347.33      1.01
```
The summary includes the model parameters as well as the log prior, log likelihood,
and log joint---corresponding to log posterior plus the log density
of the momentum. For all these quantities, we see effective sample sizes (`n_eff`)
of over 100, together with $\hat R$ diagnostics close to 1. This indicates
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
