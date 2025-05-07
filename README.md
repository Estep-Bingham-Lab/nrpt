# `nrpt`

***A NumPyro implementation of Non-Reversible Parallel Tempering (NRPT)***

## TODO

- Capture samples
- LogZ estimate
- (perhaps related to the above) Rename the `PTStats` to `PTRecorders` and
organize its entries into sub-recorders depending on their purpose.
- Remove `lax.scan` compilation time at beginning of round. See next subsection.


### Single `lax.scan` approach

**Background**: `lax.scan` requires a static `length` parameter. Using multiple
calls to this operator with increasing length -- one call for each round --
results in JAX recompiling the scan loop at the beginning of each round. This
can take several seconds even in moderately complex models.

**Idea**: Instead of executing `n_rounds` `lax.scan` loops of exponentially 
increasing length, run one big loop of `2+4+8+... = 2**(n_rounds+1)-2` 
scans. This way we only compile a single `lax.scan` loop. The rounds 
then correspond to exponentially spaced "special" scans, where we do adaptation.
Samples would be collected only from the scans corresponding to the last round.

This is the same approach taken by usual MCMC implementations -- NumPyro in
particular -- where there's a single loop of `n_warmup+n_keep` steps, which
are treated differently depending on state. Stan even further subdivides the
warmup steps to do different kinds of adaptation. 
