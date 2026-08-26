# Hellerman modular fixed-point LP benchmark

This directory provides deterministic, build-only linear programs derived
from Simeon Hellerman, *A Universal Inequality for CFT and Quantum Gravity*,
[arXiv:0902.2790v2](https://arxiv.org/abs/0902.2790).

For a symmetric benchmark point `c_L = c_R = 2`, the artifact discretizes the
non-vacuum primary dimensions on a finite positive grid. A nonnegative
variable `n_j` is assigned to every grid point `Delta_j`. Modular invariance at
the fixed point `tau=i`, or `beta=2pi`, gives the odd-derivative equations

```text
sum_j n_j f_p(Delta_j + Ehat0) exp(-2pi Delta_j) = -b_p(Ehat0),
Ehat0 = (2-c_L-c_R)/24,
```

where `f_p` and `b_p` follow Eqs. (3.22) and (3.28). The public LP lowering is
`min 0`, `G=I`, `h=0`, so SDPX's `G*x >= h` convention enforces `n_j >= 0`.

| scale | maximum odd derivative | dimension points | dimension range |
|---|---:|---:|---:|
| tiny | 1 | 16 | 0.25–4 |
| small | 3 | 32 | 0.25–6 |
| medium | 5 | 64 | 0.25–8 |
| stress | 7 | 128 | 0.25–10 |

This is a finite-grid feasibility model, not a rigorous continuum modular
functional bound and not a reproduction of Hellerman's analytic bound on the
lightest primary. A finite grid can miss continuum spectra and can itself be
infeasible. The artifact therefore fixes `reference_status=:build_only` and
`paper_equivalent=false`; the benchmark tests construct but never solve it.

The semantic fingerprint includes the source/version/status, grid, derivative
orders, every coefficient and right-hand side, numerical eta truncation,
counts, and excluded claims. Validation rebuilds all of these from the spec.
The injected catalog uses the runner's construction-only path, so selecting
this benchmark cannot accidentally launch a solve while its reference remains
`:build_only`.
