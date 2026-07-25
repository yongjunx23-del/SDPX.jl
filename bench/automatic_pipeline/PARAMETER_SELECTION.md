# Automatic Initial-Parameter Selection

Date: 2026-07-25

## Decision

The dedicated LP path uses a guarded fast-start profile in automatic mode:

- `beta = 1/50`;
- `gamma = 99/100`;
- every arithmetic type uses this profile only when the row-scale-invariant
  initial-distance indicator is at most `1000`;
- a larger or non-finite indicator keeps the configured conservative values
  (`0.1/0.9` by default);
- `parameter_policy=:fixed` remains an exact override;
- forcing an LP through `algorithm=:sdp` retains the general-conic
  parameters.

The indicator is

```text
max(max_i |h_i| / ||G_i||_inf, max_j |b_j| / ||B_j||_inf).
```

It estimates the distance from the zero start to the constraint hyperplanes
and is invariant to positive rescaling of an individual row.

No SDP parameter default changed. In particular, adaptive beta/gamma control
remains opt-in.

## End-to-end benchmark

Apple Silicon, Julia 1.12.6, four Julia threads available, one BLAS thread.
Times are medians of five warmed solves. The old rows use
`parameter_policy=:fixed, beta=0.1, gamma=0.9`; the new rows use the automatic
profile.

| Arithmetic and problem | Profile | Iterations | Runtime | Speedup | Relative gap | Primal residual | Dual residual |
|---|---|---:|---:|---:|---:|---:|---:|
| Float64, 80 variables / 400 inequalities | old | 14 | 15.500 ms | 1.00x | 5.03e-9 | 2.22e-16 | 1.62e-15 |
| Float64, 80 variables / 400 inequalities | automatic | 10 | 14.670 ms | 1.06x | 8.97e-10 | 2.22e-16 | 1.08e-15 |
| Float64, 256 variables / 4,000 inequalities | old | 18 | 714.534 ms | 1.00x | 4.37e-9 | 2.22e-16 | 4.78e-15 |
| Float64, 256 variables / 4,000 inequalities | automatic | 14 | 642.803 ms | 1.11x | 2.52e-9 | 2.22e-16 | 5.10e-15 |
| BigFloat 256-bit, 32 variables / 128 inequalities | old | 25 | 178.918 ms | 1.00x | 1.06e-21 | 8.64e-78 | 6.34e-34 |
| BigFloat 256-bit, 32 variables / 128 inequalities | automatic | 15 | 103.351 ms | 1.73x | 9.37e-21 | 8.64e-78 | 3.05e-32 |

The automatic scale scan added about 27 KB to total allocation in both
Float64 benchmark sizes, less than 0.2% for the medium case and less than
0.01% for the large case. An earlier independent five-run comparison measured
`1.17x` and `1.15x`; the same-run table above is retained as the conservative
result under concurrent machine load.

## Robustness sweep

The deterministic Float64 validation used 32 variables, 160 inequalities,
12 random seeds, and both zero and five equality constraints: 24 cases per
scale.

| Solution scale | Old median iterations | Fast median iterations | Old failures | Fast failures | Automatic decision |
|---:|---:|---:|---:|---:|---|
| 1e-4 | 14 | 10 | 0 | 0 | fast |
| 1 | 13 | 9 | 0 | 0 | fast |
| 1e4 | 32 | 32 | 1 | 2 | conservative |

A second 20-seed sweep had no failures for either profile through solution
scale `100`. At scale `1000` the median iteration count was equal, and the
distance indicator was already above the `1000` crossover on representative
instances. This provides a conservative buffer before the observed Float64
failure regime.

At 256-bit BigFloat, the fast profile reduced iterations from `23-25` to
`15-16` at solution scale `1`. A larger 16-case sweep with equality and
inequality models reached indicators of `5.78e4-1.07e5` at solution scale
`1e4`. Both parameter profiles had eight non-converged equality cases there,
and the fast profile's worst relative gap was `0.943` versus `2.10e-16` for
the conservative profile. At solution scales `1e6` and `1e8`, the fast
profile likewise provided no consistent robustness improvement. BigFloat
therefore uses the same `1000` guard: exponent range and significand precision
do not solve globalization from a distant infeasible start.

## SDP controls retained

The representative CSDR sparse SDP still favors the fixed strategy:

| Strategy | Iterations | Median runtime | Relative gap | Primal residual | Dual residual |
|---|---:|---:|---:|---:|---:|
| fixed | 19 | 13.70 ms | 8.07e-9 | 4.51e-14 | 3.62e-11 |
| adaptive | 33 | 22.79 ms | 5.55e-8 | 4.09e-14 | 2.28e-9 |

Changing the sparse-SDP predictor from `:classic` to `:sdpb` reduced one
iteration in a separate ten-run check but did not improve median runtime.
The predictor default is therefore unchanged.

The validated dense lattice initialization also remains unchanged:
`omega_p=100`, `omega_d=0.001`, `predictor=:sdpb`. Its final Float64
Task_Low08 run was Optimal in 27 iterations with objective
`0.653291393898`, relative gap `4.55e-7`, primal residual `2.06e-10`, and
dual residual `3.68e-9`.

## Validation

- focused automatic-pipeline tests: 180 passed;
- LP and numeric-genericity regressions: 82 passed;
- forced LP-through-SDP behavior remains valid;
- explicit fixed parameters remain unchanged;
- automatic profile and conservative crossover are visible in execution-plan
  diagnostics;
- adaptive history remains bounded and fallback-safe.
