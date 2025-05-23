# `nrpt`

[![Build Status](https://github.com/Estep-Bingham-Lab/nrpt/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Estep-Bingham-Lab/nrpt/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Estep-Bingham-Lab/nrpt/branch/main/graph/badge.svg)](https://codecov.io/gh/Estep-Bingham-Lab/nrpt)

***A NumPyro implementation of Non-Reversible Parallel Tempering (NRPT)***

## Installation

**Optional**: if you want to run your NumPyro models on an accelerator (GPU/TPU),
make sure to 
[install the correct version of JAX](https://jax.readthedocs.io/en/latest/installation.html).
Otherwise, the following will install the default, CPU-only version of JAX.

Using pip
```bash
pip install nrpt @ git+https://github.com/Estep-Bingham-Lab/nrpt.git
```

## TODO

- Capture samples
- loglik autocorrelation before v. after exploration
- Measure time elapsed in round 
- README example with mRNA and pairplot
- (perhaps related to the above) Rename the `PTStats` to `PTRecorders` and
organize its entries into sub-recorders depending on their purpose.
