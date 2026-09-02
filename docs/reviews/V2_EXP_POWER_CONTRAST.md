# V2 EXP/Power contrast diagnosis

**Branch:** `diag/v2-exp-power-contrast` from `da55fdb`
**Scope:** diagnosis only; no solver-source changes.
**Probe:** `benchmark/general/v2/exp_power_contrast_probe.jl`

## Executive classification

There is no blanket V1-to-V2 regression in the EXP/Power families.

* The only actual V1 EXP optimum with a valid certificate,
  `exp_unit_small`, is reproduced by the V2 `unit_epigraph` construction.
* The V1 `exp_entropy_small` and `exp_logsumexp_small` rows are **not** passing
  optima on this checkout: they return `numerical_failure`/`numerical_breakdown`
  with invalid certificates.  V2 entropy/log-sum-exp return the same native
  failure class.  Their open status is therefore native EXP support, not a
  V2-only lowering regression.
* The V1 passing Power row, `power_epigraph_small`, is not the same mathematical
  instance as the V2 candidate tranche.  V1 has three random interior targets,
  all alpha `0.5`; V2 uses fixed `x_i=1`, five heterogeneous alphas, and its
  weighted candidate introduces fixed nonnegative variables.  V2's failures are
  therefore a boundary/heterogeneous-parameter stress case, not evidence that
  the V1 instance was lowered incorrectly.

At the historical diagnosis commit `24a78b6` (based on `da55fdb`), the V2
`power_tranche_catalog()` intentionally had zero registered instances; its
three `PowerArtifact` values were typed source candidates only. That sentence
is retained as historical probe context, not the current catalog status. At
reviewed HEAD `e14df2a`, the catalog has one separately constructed certified
interior row, `v2_power_interior_epigraph_small` (alpha=1/2, fixed x=1/2,
optimum t=1/4), with an exact source oracle and Float64 certificate. The
historical five-alpha boundary/heterogeneous, weighted-mean, and alpha-sweep
candidates remain unregistered and open.

## Exact V1 baselines and V2 counterparts

All rows below were solved through the public `Model`/`optimize!` API with
Float64, one solver thread, certification enabled, Julia 1.12,
`--gcthreads=1`.

| V1 row | V1 result | V2 construction/result | Relationship |
|---|---|---|---|
| `exp_unit_small` | `:optimal`, cert=true, objective `1.0`, 14 iterations | `unit_epigraph`: `:optimal`, cert=true, objective `1.0`, 14 iterations | same `(0,1,x)` cone; no regression |
| `exp_entropy_small` | `:numerical_failure`, cert=false, objective `-1.0986122912092953`, 26 iterations | n=2 typed entropy: `:numerical_breakdown`, cert=false, objective `0.0`, 7 iterations | both native EXP failure; not a V1 passing counterpart |
| `exp_logsumexp_small` | `:numerical_breakdown`, cert=false, objective `0.0`, 0 iterations | n=2 equal-coefficient typed LSE: `:numerical_breakdown`, cert=false, objective `0.0`, 0 iterations | same native failure class |
| `power_epigraph_small` | `:optimal`, cert=true, objective `1.1242390809627705`, 52 iterations | V2 five-alpha `separable_p_power`, fixed `x_i=1`, fails (`:numerical_breakdown`, cert=false, 57 iterations in the exact candidate probe) | different instance; V2 is boundary + heterogeneous alpha |
| `power_geomean_small` | `:numerical_breakdown`, cert=false, 6 iterations | V2 `weighted_mean`, variables fixed to `(1,1)` then one power row: `:numerical_failure`, cert=false, 18 iterations | both fail; no V1 passing counterpart |

V1 source construction is at `benchmark/general/exp.jl:15-44` and
`benchmark/general/power.jl:16-40`.  V1's runner uses ordinary public settings
with automatic tolerances (`GenericConicBenchmark.jl:98-108`), and does not
apply a family-specific presolve or hidden override.

## End-to-end construction diff

### EXP

V1 and V2 use the same public cone orientation:

* unit: `(0, 1, x) in K_exp`, minimize `x`;
* entropy: `(-r_i, p_i, 1) in K_exp`, `sum(p)=1`, minimize `sum(r)`;
* log-sum-exp: `(a_i-t, 1, z_i) in K_exp`, `sum(z)<=1`, minimize `t`.

V1 entropy registers the normalization row **before** the EXP rows
(`exp.jl:25-33`); V2 appends it **after** the EXP rows
(`v2/exp_power_catalog.jl:105-118`).  The LSE row order and orientation are
the same (`exp.jl:34-44`, `v2/exp_power_catalog.jl:119-131`).  The entropy
row-order change is a genuine model-layout difference, but it does not explain
the observed result: V1 entropy n=3 already fails, and a V1-style n=2 entropy
model also returns `:numerical_breakdown`.

V2 explicitly supplies the reviewed solver tolerance in `run_instance`
(`v2/GeneralBenchmarkV2.jl:939-947`), while V1 leaves tolerances automatic.
The direct probe also tested V2 with the same automatic-tolerance settings as
V1; the EXP failures remained.  Thus neither tolerance policy nor row order
currently provides evidence for a V2-only defect.  The minimal root cause for
the open entropy/LSE rows is **native exponential-cone numerical support**.

### Power

V1 passing `power_epigraph_small` (`power.jl:19-30`) has:

* `n=3`, alpha `0.5` for every cone;
* random signed targets from `0.25` to `1.0`, hence generally interior rather
  than the exact cone boundary;
* variables `x` and nonnegative `t`, with alternating fix and power rows.

The V2 separable candidate (`v2/exp_power_catalog.jl:179-192` and
`:275-287`) has:

* five alpha values `[1/2, 1/3, 2/3, 2/5, 7/10]`;
* exact fixed `x_i=1`, a cone-boundary witness for every row;
* five `x` fix equalities and five heterogeneous PowerCone blocks;
* objective 5 (one unit epigraph per alpha).

The V2 weighted candidate (`:193-201`) adds `left` and `right` nonnegative
variables plus two fix equalities before the power row.  V1 geomean instead
uses constants directly in one cone row (`power.jl:31-36`).  This is a
real variable-offset/block-layout difference, but V1's corresponding
`power_geomean_small` also fails, so it is not a regression from a passing V1
case.

## Minimal isolation probe

Run from this checkout with a temporary developed SDPX environment:

```bash
export JULIA_DEPOT_PATH="/tmp/sdpx-autoresearch/.auto/depot:$HOME/Desktop/project/SDPX/.julia-depot:$HOME/.julia"
JULIA_NUM_THREADS=4 julia --gcthreads=1 --startup-file=no --project=. -e \
  'using Pkg; Pkg.activate(; temp=true); Pkg.develop(; path=pwd()); \
   include("benchmark/general/v2/exp_power_contrast_probe.jl")'
```

Important controls from the probe:

* V2 one-cone, alpha `1/2`, fixed `x=1/2`: **optimal, cert=true**;
* V2 four-cone heterogeneous prefix
  `[1/2,1/3,2/3,2/5]`, fixed `x_i=1/2`: **optimal, cert=true**;
* adding alpha `7/10` to the five-cone heterogeneous set, fixed `x_i=1/2`:
  **iteration_limit**, cert=false (the exact `x_i=1` candidate returns
  numerical breakdown);
* each one-cone alpha in the reviewed set solves optimally for interior `x`
  controls, while the aggregate heterogeneous candidate does not.

This isolates the Power issue to the **multi-cone heterogeneous-alpha / boundary
stress geometry**.  The artifact's cone orientation and fixed-variable
mathematics are sound for the passing controls; the native route does not yet
provide a certified receipt for the proposed aggregate candidate.

## Recommended minimal fix and expected impact

### Fix placement

**Immediate policy: V2-side catalog, not solver source.** The historical
five-alpha boundary/heterogeneous and weighted candidates remain typed but
unregistered because their native receipts fail. The new interior
`v2_power_interior_epigraph_small` is a distinct artifact, not a relabeling of
those failures: its exact alpha=1/2, x=1/2 witness reaches t=1/4 and its
Float64 status/certificate/original-coordinate gates pass. Do not change
tolerances, alter certificate rules, or promote the harder candidates from
probe outcomes. Any future Power row needs a new exact source contract,
independent oracle, and certificate receipt.

**Potential future solver work:** if the product-HSD route is required to
support the five-cone heterogeneous or exact-boundary cases, investigate
PowerCone initialization/centrality and multi-block alpha scaling in the
solver robustness queue.  That is a solver-side enhancement requiring its own
frozen-trajectory and certificate gates; this diagnosis does not authorize it.

### Impact on open rows

* **EXP entropy and log-sum-exp:** no V1 passing baseline was found.  Keep
  `known_solver_finding`/unregistered status until native EXP support produces
  an independently certified receipt.  The V2 row-order difference is worth a
  controlled follow-up, but changing order alone is not justified by the probe.
* **Power boundary/heterogeneous separable, alpha-sweep, and weighted-mean
  candidates:** remain open and are not promoted. The certified interior row is
  separate; a future redesign of the harder candidates must still be a new
  exact artifact with a new independent oracle and certificate receipt, never a
  relabeling of a failed probe.
* **V1 `exp_unit_small` and `power_epigraph_small`:** preserve as passing V1
  compatibility cases; the unit EXP V2 construction is already reproduced,
  while the Power row is not mathematically equivalent to the V2 candidates.

## Residual risk

Solver outcomes for multi-cone Power controls show some sensitivity across
sizes/alpha sets (e.g. iteration limit versus numerical breakdown).  The
classification above relies on the exact probe receipts and does not claim a
universal threshold in cone count.  No solver source was modified in this
branch.
