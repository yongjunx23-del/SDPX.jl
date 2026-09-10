# P0-03 — platform-dependent `direction_breakdown` on the bordered LP fixture

Status: **VERIFIED** (root cause found, fixed, re-checked on three ISAs).
Category: **R — correctness fix** (bound propagation in a robustness guard).

## Symptom

`test/native_structure_diagnostics.jl`, testset *"bordered LP above threshold
records compact plan"* (2 free variables, 12 nonnegative rows, `kkt_route =
:bordered`, compact Schur dimension 3):

- aarch64 (Apple M1/M4) and x86 znver4 / icelake-server: `optimal`, 12
  iterations, `certificate.valid == true`.
- x86 znver3: `numerical_breakdown`, `reason=direction_breakdown`,
  `iters=6`, `tau=0.104475755246474`, `kappa=0.00018430599715242167`,
  `mu=9.564345445233547e-6`.

The same result was produced at `--threads=1` and `--threads=4` on the same
machine, so threading was excluded before any further work.

## How the first divergence was exported

A read-only probe (`SDPX_DEBUG_DIRECTION=1`, optional
`SDPX_DEBUG_DIRECTION_ITER=<n>`) prints, at every direction stage:

```
SDpxGate stage=... epoch=... iters=... mu=... tau=... kappa=... threshold=...
         route=... primal_ok=... primal_ratio=... dual_ok=... dual_ratio=...
         cone_ok=... cone_ratio=... gap_ratio=... last_reason=... solves=...
         refinements=... groups_passing=n/4
```

and, when the triangular factor-solve certificate rejects, a second line with
the offending component:

```
SDpxTriCert n=3 operations=24 gamma=5.32907051820078e-15 worst_kind=backward
            worst_index=3 worst_ratio=2.7721741244162823 max_f=0.0
            max_u=1.1487884347136448e-20
```

The probe is env-gated, is only reached on a failure path or after a gate has
already been evaluated, only reads state, and never participates in a
numerical decision.

## Evidence

`znver3` (failing), epoch 7:

```
SDpxTriCert  n=3 operations=24 gamma=5.32907051820078e-15 worst_kind=backward
             worst_index=3 worst_ratio=2.7721741244162823 max_f=0.0
             max_u=1.1487884347136448e-20
SDpxGate     stage=solve_shift_raw epoch=7 iters=6 mu=9.564345445233547e-6
             tau=0.104475755246474 kappa=0.00018430599715242167
             threshold=7.62939453125e-6 route=bordered primal_ok=true
             primal_ratio=0.0 dual_ok=true dual_ratio=1.520500276910364e-14
             cone_ok=false cone_ratio=2.0271588964584716e-5
             cone_res=7.622926043704403e-5 gap_ratio=5.782402008421477e-17
             last_reason=SYMMETRIC_BORDERED_SOLVE_FAILED solves=1 refinements=0
```

`aarch64` (passing), the same epoch:

```
SDpxGate stage=direction_ok epoch=7 iters=6 mu=9.564345445233677e-6
         tau=0.10447575524647393 kappa=0.0001843059971524258
         primal_ok=true primal_ratio=5.337909158626507e-18 dual_ok=true
         dual_ratio=1.1090240106105159e-13 cone_ok=true
         cone_ratio=1.180962338914981e-16 gap_ratio=6.938882410105678e-17
         last_reason=SYMMETRIC_BORDERED_READY solves=2 refinements=0
```

Two facts settle the classification:

1. **The iterates agree.** `mu`, `tau` and `kappa` at the failing epoch match
   the passing platforms to ~1e-15 relative.  This is not a divergent solver
   trajectory; it is the same iterate.
2. **The rejected solve is numerically excellent.** `max_f = 0.0` (the
   replayed forward substitution is exact) and the largest backward residual
   is `1.15e-20`, i.e. thirteen orders of magnitude below every solver
   tolerance.  The rejection is `worst_ratio = 2.77` against a bound of
   `gamma(8n)·work` with `gamma(24) = 5.33e-15` — a factor 2.77 miss on a
   quantity at the 1e-20 level.

The observed `cone_ok=false / cone_ratio=2.03e-5` in the failure line is an
artefact of the probe reading a state whose corrector RHS was never accepted;
it is not the cause.

## Root cause

`_product_bordered_triangular_solution_ok!` certified `U·x = y` with an
allowance of

```
gamma(operations) * ((|y_i|) + Σ_{j≥i} |F_ij·x_j|)
```

but `y` is produced by the forward substitution that the same routine replays
immediately above (`L·y = P·rhs`).  `y` is therefore only known to within
`gamma(operations) * forward_work`, and that propagated term was dropped from
the allowance.  The check asserted a tighter statement than the arithmetic
supports, so it accepted or rejected depending on how the LAPACK
`getrs`/`trsv` kernel rounds: fused on znver4/icelake/aarch64 (accepted),
and a different accumulation on znver3 (rejected).

This is a **bound-propagation error** in the guard, one of the categories the
plan authorises fixing.  It is not a stopping tolerance, not one of the five
equation gates, and not the original-coordinate certificate standard.

## Fix

`src/hsd/product_cone_hsd.jl`, `_product_bordered_triangular_solution_ok!`:
the backward allowance becomes the composed bound

```
gamma(operations) * ( backward_work_i + forward_work_i )
```

with the same `gamma`, the same operation count, the same work definition and
the same arithmetic.  Only the propagated forward term is no longer
discarded.  Because the new allowance is strictly larger, every direction
that was accepted before is still accepted: on the passing platforms the
trajectory is unchanged, and on znver3 the direction that those platforms
already accept is now accepted too.

Result (GitHub Actions diagnostic run 34438051521, workflow
`.github/workflows/diag-border.yml`):

| machine | before | after |
|---------|--------|-------|
| x86 `znver3` | `numerical_breakdown`, 6 iterations | `optimal`, 12 iterations |
| x86 `znver4` | `optimal`, 12 iterations | `optimal`, 12 iterations |
| x86 `icelake-server` | `optimal`, 12 iterations | `optimal`, 12 iterations |
| aarch64 `apple-m1` | `optimal`, 12 iterations | `optimal`, 12 iterations |
| aarch64 `apple-m4` (local) | `optimal`, 12 iterations | `optimal`, 12 iterations |

The fix does not make the guard vacuous: it still rejects gross factor-solve
corruption, and the authoritative five-equation direction gate and the
original-coordinate certificate are untouched.

## Residual risk

- The guard is now slightly more permissive in the last few ulps of the
  triangular solve.  Any genuine factor-solve failure that previously showed
  up only as a 2–3× miss of this bound would now be caught by the
  five-equation gate instead.  No such case was observed.
- The underlying sensitivity (a replay-based certificate that must reproduce
  the LAPACK kernel's rounding model) remains.  A kernel-independent
  backward-error test of the original bordered system would be the durable
  fix; it is a larger numerical change and is not attempted here.
