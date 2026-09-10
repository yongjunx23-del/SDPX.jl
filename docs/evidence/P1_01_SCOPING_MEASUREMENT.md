# P1-01 scoping measurement — repeated cone-operator materialization

Status: **measured; the plan's P1-01 target does not exist on the primary
workload.**  P1-01 remains valid for the generic (non-fixed-trace) route.

## What P1-01 assumed

> 现有通用块物化会为某个块的一列构造全局 basis 并调用整个 runtime 的
> `apply_Theta!`；corrector 又进入物化函数。 … CSDR 争取约 1.08–1.20×

i.e. that the corrector re-materializes the whole cone operator, and that
removing that repeated work is worth 8–20 % on CSDR.

## What is actually in the code

Two different linearization paths exist.

**(a) Fixed-trace Q3 — the CSDR/MultiFloat production route.**
`_product_hsd_fixed_trace_hkm_linearization!(state, target,
include_affine_product, refresh_metric)` already takes a `refresh_metric`
flag.  The predictor calls it with `refresh_metric=true`; the corrector calls
it with `refresh_metric=false`, and in that case it

- zeroes only `state.h` and `cone.corrector_rhs`, never `cone.operators`;
- refuses to run at all if `core.linearization_epoch != base.epoch`, so a
  stale operator can never be silently reused;
- shares the single factor and homogeneous solve owned by the numeric epoch.

So the predictor/corrector split that P1-01 asks for **is already
implemented** on this route.

**(b) Generic `_product_hsd_symmetric_core_linearization!` — the non-fixed-trace
route.**  Both the predictor and the corrector call the same function
(`predictor_corrector.jl` ~line 943 and ~1039), which loops over every block,
builds a full-`m` basis vector per local column and calls
`apply_Theta!` on the whole runtime.  This is the pattern P1-01 describes,
and it is the route the refactor should target.

## Measurement on CSDR α3 Float64x4 (105 iterations)

Warm second solve, `--threads=4`, phase timings from `SDPX.diagnostics`:

| phase | seconds | share of core |
|---|---|---|
| `core` (total) | 13.95 | 100 % |
| `direction_seconds` | 9.02 | 65 % |
| `schur_assembly_seconds` | 0.66 | 4.7 % |
| `corrector_rhs_seconds` | 0.87 | 6.2 % |
| `kkt_factorization_seconds` | 3.19 | 23 % |
| `q3_metric_seconds` | 2.59 | 19 % (of which `q3_gram_seconds` 1.86) |
| `certification_seconds` | 1.61 | 12 % |
| `line_search_seconds` | 1.26 | 9 % |
| `residual_seconds` | 1.01 | 7 % |

`schur_assembly_seconds + corrector_rhs_seconds` is 11 % of the core, and on
this route most of it is the cone-RHS refresh and system assembly that
P1-01 explicitly wants to keep.  There is no 8–20 % of *repeated operator
materialization* available here.

## Consequence for the plan

- P1-01 must be **re-scoped** to the generic non-fixed-trace symmetric-core
  path, and its acceptance evidence must come from a generic multi-block
  SOC/PSD/mixed benchmark, not from CSDR.
- The CSDR performance lever is therefore unchanged from the earlier
  autoresearch finding: `direction_seconds` (65 %) is dominated by the
  predictor/corrector solves and `q3_metric_seconds` by the Gram SYRK, which
  is already at the hardware limit.  The remaining lever on this workload is
  algorithmic (iteration count), i.e. P3/P4 territory.
- No code change is claimed by this document.

## Required next step before implementing P1-01

Instrument the generic route (count `apply_Theta!` calls and total
`m × blocks` work per solve for a generic SOC/PSD/mixed benchmark) and
confirm the repeated share is material there.  Only then refactor.
