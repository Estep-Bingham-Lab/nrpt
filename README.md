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

from autostep import autorwmh

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
[`autostep` package](https://github.com/UBC-Stat-ML/autostep), which is
automatically installed along `nrpt`. For this example, we will use the
AutoRWMH (auto random-walk Metropolis-Hastings) sampler.
```python
kernel = autorwmh.AutoRWMH(
    model, 
    initialization_settings={'strategy': "L-BFGS", 'params': {'n_iter': 128}}
)
```
The additional argument is requesting an initial optimization phase where the
starting point of the Markov chain is chosen to be close to the mode of the
posterior -- i.e., close to the maximum *a posteriori* (MAP). For complicated
distributions such as the present one, gradient-free samplers can take a long
time to find the bulk of the mass. Thus, they can benefit from an optimized
initialization. Specifically, we request `128` iterations of the L-BFGS 
optimization algorithm---provided by 
[`optax`](https://optax.readthedocs.io/en/latest/))---to improve the starting point.

With the explorer in place, we can proceed to instantiate a `PTSampler` object
```python
pt_sampler = initialization.PT(
    kernel, 
    rng_key = random.key(123),
    n_rounds = 14,
    n_replicas = 15,
    n_refresh = 32,
    model_args=model_args, 
    model_kwargs=model_kwargs
)
```
Note that the model arguments are passed to the constructor. There are 
several other settings being provided:

- A JAX PRNG key, used to draw (pseudo-)random variates.
- The number of NRPT rounds is set to 14, so that a total of 
$2^{14}$=16384 samples are returned.
- The number of replicas is set to 15, which is roughly 2.5 times the global
barrier of the problem.
- The number of explorer refreshments within each exploration step is set to 32.
This allows us to achieve a worse-case autocorrelation of the log-likelihood 
(across replicas) of less than 0.95.

After executing the above, we see the output
```
Using L-BFGS to improve initial state.
Initial energy: 3.6e+07
Final energy: 3.5e+02
```
In this context, tempered potential simply corresponds to the negative log
posterior density. Note that the optimization phase reduced the potential
by 5 orders of magnitude.

Now we are in place to run NRPT, which we do via
```python
start_time = time.time()
pt_sampler = sampling.run(pt_sampler)
print(f"--- {time.time() - start_time} seconds ---")
```
The above will produce an output similar to this
```
 Round |     Λ |      logZ | ρ (mean/max) | α (min/mean) | llAC (mean/max) 
---------------------------------------------------------------------------
     1     1.0   -3.63e+02    0.07 / 1.00    0.00 / 0.00       nan / nan
     2     3.7   -4.89e+02    0.27 / 1.00    0.21 / 0.46       0.07 / 1.17
     3     5.1   -3.64e+02    0.36 / 0.79    0.28 / 0.43      -0.08 / 0.96
     4     5.9   -4.04e+02    0.42 / 1.00    0.31 / 0.35       0.43 / 0.94
     5     6.5   -3.82e+02    0.46 / 0.72    0.32 / 0.36       0.57 / 0.98
     6     5.9   -3.70e+02    0.42 / 0.79    0.30 / 0.37       0.54 / 0.96
     7     6.1   -3.70e+02    0.43 / 0.80    0.34 / 0.45       0.57 / 0.91
     8     6.2   -3.69e+02    0.44 / 0.59    0.32 / 0.40       0.61 / 0.94
     9     6.2   -3.70e+02    0.44 / 0.62    0.33 / 0.41       0.58 / 0.93
    10     6.1   -3.72e+02    0.44 / 0.66    0.35 / 0.42       0.62 / 0.94
    11     6.0   -3.70e+02    0.43 / 0.52    0.35 / 0.41       0.60 / 0.93
    12     6.1   -3.70e+02    0.44 / 0.48    0.37 / 0.42       0.61 / 0.93
    13     6.1   -3.70e+02    0.44 / 0.46    0.38 / 0.42       0.61 / 0.93
    14     6.2   -3.70e+02    0.44 / 0.48    0.36 / 0.41       0.62 / 0.93
--- 439.30020546913147 seconds ---
```
The figures shown here correspond to

- Estimates of the global barrier, which at the last round is 
$\Lambda \approx 6.2$. 
- Estimates of the log-normalization constant, which in the last round gives
$\log(\mathcal{Z})\approx -370$.
- Average and worst-case swap rejection probabilities. When the average is
close to the maximum -- as in the last 5 rounds -- the ideal *equi-rejection*
condition has been approximately attained.
- Average and worst-case explorer acceptance probabilities. If the explorer is
working correctly along the path of distributions, we expect both values
to be away from 0 and 1.
- Average and worst-case autocorrelation of the log-likelihood before and 
after the exploration steps. As described above, the number of refreshments
was set so that the maximum was below 0.95. **Note**: the estimator does not
behave well in small samples, which is why we can see autocorrelation values
larger than one in earlier rounds.

We can now extract the samples and create the corner plot using
```python
samples = pt_sampler.pt_state.samples
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
of these two modes are completely different. These features make this
problem extremely hard to tackle using traditional MCMC algorithms.


## References

Syed, S., Bouchard-Côté, A., Deligiannidis, G., & Doucet, A. (2022). 
[Non-reversible parallel tempering: a scalable highly parallel MCMC scheme](https://doi.org/10.1111/rssb.12464). 
*Journal of the Royal Statistical Society Series B: Statistical Methodology*, 84(2), 321-350.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024).
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). 
*arXiv preprint arXiv:2410.18929*. Accepted to ICML 2025.


## TODO

- Measure time elapsed in round 
- Documentation
