# P3-00 / P3-01 — iteration telemetry and the β single-factor experiment

Status: **P3-00 measured; P3-01 complete with a negative result.**
The legacy `beta = 0.9` default is retained; no new policy is enabled.

## P3-00 — acceptance telemetry

`SDPX_DEBUG_ITER=1` (read-only, env-gated) prints one line per accepted
iteration from `_product_hsd_line_search!`:

```
iter, beta, alpha, backtracking, mu, mu_aff_ratio, sigma_requested, tau, kappa
```

`beta` reports `default(0.9)` when the literal is in use, so a run is always
self-describing about which damping was applied.  The probe only reads state
and is never consulted by a solve.  It complements the existing
`SDPX_DEBUG_DIRECTION` (five-equation gate ratios, per iteration) and
`SDPX_DEBUG_LINE_SEARCH` (trial neighbourhood/merit decision) probes.

## P3-01 — β single factor on CSDR α3 Float64x4

Warm second solve, `--threads=4`, `iteration_knobs=(; beta=...)`:

| beta | status | iterations | certificate | seconds |
|------|--------|-----------|-------------|---------|
| default (0.9) | optimal | **105** | valid | **15.04** |
| 0.95 | optimal | 107 | valid | 15.14 |
| 0.98 | optimal | 118 | valid | 16.65 |

**The plan's hypothesis is disconfirmed on this workload.** Raising the
fraction-to-boundary damping increases the iteration count monotonically:
0.95 costs +2 iterations, 0.98 costs +13.  Per the plan's retain rule
("只减少迭代但时间增加，不保留"), nothing is retained and the historical
literal stays on the default path.

Objective values were `-31.6721557349…` (default), `-31.6721557400…` (0.95)
and `-31.6721557582…` (0.98) against the analytic reference
`-31.672155970636578`; all three are certified, and the differences are in the
2.1–2.4e-8 band, i.e. they reflect the different terminal iterates, not a
correctness difference.  No tolerance, gate or certificate rule changed.

## Attribution: what limits the 105 iterations

From the `SDPX_DEBUG_ITER` trace of the default run:

- `backtracking == 0` in **every one of the 105 iterations**.  No iteration is
  limited by a rejected trial, a merit failure or a line-search contraction.
- The accepted step is well inside the β cap: `alpha` runs 0.15–0.51 with a
  median near 0.4, so the cap `beta = 0.9` is never the binding constraint.
- `mu_aff/mu` runs 0.64–0.87 (never close to 0), which is what drives
  `sigma = min(1, (mu_aff/mu)^3)` to a small value and produces the observed
  slow μ reduction.

**Conclusion.** The iteration count is set by the predictor/corrector
centering quality (σ from `mu_aff/mu`), not by the fraction-to-boundary
damping and not by line-search failure.  A β-only controller (P3-02) is
therefore unlikely to help this workload, and the plan's P3-02 should be
justified against σ (P3-03) rather than β.  Because increasing β makes
`mu_aff/mu` worse, the two are not independent; the plan's instruction to
validate β and σ one factor at a time is the right discipline, and β
is now measured as a dead end here.

## Scope caveat

This experiment is CSDR-only.  The plan explicitly forbids adopting a global
default from a single workload's optimum ("不用单个 CSDR 的最优 β 设置全局
默认"), which this document also does not do: the default is unchanged
because 0.9 is the legacy literal, not because CSDR prefers it.  A
multi-family β sweep (LP/SOC/SDP/Exp/Power) has not been run and is required
before any β conclusion is claimed for the general matrix.
