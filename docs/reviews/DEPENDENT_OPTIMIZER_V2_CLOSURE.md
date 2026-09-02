# Dependent-optimizer V2 closure receipt

Date: 2026-09-02

## Scope

The V2 branch is consumed as an explicit external source checkout; it is not
merged into `feat/dependent-benchmark-optimization-loop`. Set
`SDPX_V2_ROOT=/tmp/sdpx-general-v2-fix` to the pinned V2 checkout. The bridge
requires an exact checked-out commit and tree SHA and loads the real V2
`GeneralBenchmarkV2.jl` and `v2_schema9_adapter.jl` from that root.

## First target

`v2_lp_box_small` is emitted by the V2 adapter, not synthesized by the
optimizer branch. The V2 adapter performs one excluded warmup and exactly
three rebuilt same-process measured solves. Every sample must be optimal,
certificate-valid, independently referenced, iteration/objective
 deterministic, and route-receipt deterministic. The resulting row is then
passed through this branch's `ProfileCatalog.validate_profile_row(row;
live=true)` before schema-v9 TSV/TOML emission.

The local profile cycle receipt is produced by:

```bash
JULIA_NUM_THREADS=4 julia --gcthreads=1 --startup-file=no --project=. \
  benchmark/optimization/test_v2_target_bridge.jl
```

with `SDPX_V2_ROOT` set as above. The test removes its temporary output
folder after parsing and checking the emitted schema-v9 document.

## Readiness state

The local prerequisite is **met**: one complete solve-eligible schema-v9
V2 target exists and passes this branch's live profile validator. The
repository variable `SDPX_ENABLE_DEPENDENT_OPTIMIZATION` is intentionally
**not mutated locally**. It may be set to `true` only after the catalog
workflow publishes an exact-head catalog artifact on the target repository's
main branch. The following remain disabled/open:

- fresh-process profile samples (current adapter is explicitly
  `same_process_three_sample`);
- full catalog breadth/holdout completion;
- provider-backed precision qualification for every target;
- Stage-B performance, RSS, allocation, and repeated-solve gates;
- automatic GitHub workflow activation until the exact catalog artifact and
  source/tree identities are published by the catalog workflow.

A local target receipt does not unlock the six-round optimization loop by
itself. No fixture mode is used by the bridge, and no fields are filled from
missing API data.
