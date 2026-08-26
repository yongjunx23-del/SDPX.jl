# Gibbs relative-entropy exponential-cone benchmark

This directory contains deterministic, build-only exponential-cone programs
for a finite-level Gibbs thermal state.  For the benchmark-generated energy
grid `E_i` and fixed inverse temperature `beta=2`, it constructs

```text
q_i = exp(-beta E_i) / Z,       Z = sum_j exp(-beta E_j),
```

and minimizes the relative entropy of a candidate diagonal state `p` from
`q`:

```text
minimize    sum_i t_i
subject to  (-t_i, p_i, q_i) in K_exp,
            sum_i p_i = 1.
```

SDPX uses
`K_exp = {(x,y,z): y exp(x/y) <= z, y > 0}` together with its closure.
Consequently each cone block is exactly

```text
(-t_i,p_i,q_i) in K_exp  <=>  t_i >= p_i log(p_i/q_i)
```

for positive `p_i`.  Gibbs' inequality gives the analytic boundary optimum
`p=q`, `t=0`, and objective `D(q||q)=0`.  The artifact also stores the strict
interior point `p=q`, `t=q`; every block then has `x/y=-1` and positive margin
`q_i(1-exp(-1))`.

| scale | energy levels | exponential cones | canonical rows |
|---|---:|---:|---:|
| tiny | 8 | 8 | 25 |
| small | 32 | 32 | 97 |
| medium | 128 | 128 | 385 |
| stress | 512 | 512 | 1537 |

The physics provenance is the canonical Gibbs construction from Jaynes and
the finite-distribution relative information of Kullback and Leibler.  The
linear spectrum, `beta=2`, scale ladder, and conic transcription are all
benchmark-derived.  This is not a numerical reproduction of either paper,
so every artifact and catalog entry has `reference_status=:build_only` and
`paper_equivalent=false`.

The semantic fingerprint covers the exact source/claim boundary, spectrum,
Boltzmann weights, partition function, arithmetic-normalized probabilities,
both analytic witnesses, cone counts, and expected objective.  Validation
rebuilds these fields deterministically.  The catalog builder's problem
payload is an `SDPX.CanonicalConicProgram`; it never calls a solver or chooses
a provider.
