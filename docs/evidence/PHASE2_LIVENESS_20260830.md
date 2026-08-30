# Phase 2 production liveness and ownership map (2026-08-30)

**Source branch:** `agent/plan-1-8`

This report is rooted at the live public surfaces required by the review plan:
`src/SDPX.jl` exports, Model `optimize!`, MOI, CLI/bridge, benchmark/general,
validation, and downstream CSDR qualified usage.  Static absence from exports is
not sufficient evidence; qualified references were searched across `src/`,
`ext/`, `validation/`, `test/`, and `benchmark/`.

## Production path confirmed live

```text
Model / MOI / CLI
 -> compile_product_cone_model
 -> NativeConeProgram
 -> _public_optimize_native_hsd
 -> ProductConeHSDState
 -> frozen NewtonSystem
 -> bordered / expanded / sparse_schur
 -> original-coordinate certificate
```

`_optimize_impl` directly compiles the model and calls
`_public_optimize_native_hsd`; it does not call the older family-lowering path.

## First completed move

| Candidate | Static callers outside owner | Classification | Action |
|---|---:|---|---|
| `src/cones/nonsymmetric/full_newton_reference.jl` | 0 | validation/reference oracle | moved to `validation/oracles/`; production include removed; independent 256-bit five-equation smoke 6/6 |

## Old family lowering cluster

| Owner/file | Root entry | External production caller | Classification |
|---|---|---:|---|
| `src/ir/lower_lp.jl` | `lower_lp_native` | `_public_lower_native` only | unreachable implementation candidate |
| `src/ir/lower_soc.jl` | `lower_soc_native` | `_public_lower_native` only | unreachable implementation candidate |
| `src/ir/lower_sdp.jl` | `lower_sdp_native` | `_public_lower_native` only | unreachable implementation candidate |
| `src/ir/lift_psd.jl` | `lower_mixed_psd_native` | `_public_lower_native` only | unreachable implementation candidate |
| `src/ir/lift_psd.jl` | `lift_to_psd` | 0 | unreachable implementation candidate |
| `src/public/optimize.jl` | `_public_lower_native` | 0 | unreachable implementation candidate |
| `src/public/optimize.jl` | `_public_result_from_lowering` | 0 | unreachable implementation candidate |

The old result reconstruction helpers and their lowering types are one coupled
cluster.  They must be removed in one bounded batch only after recording the
exact source slice and rerunning public LP/SOC/PSD/Exp/Power E2E plus MOI/CLI
smokes.  The native canonicalization/reconstruction owner remains live and is
not part of this deletion.

## Retained compatibility/provider owners

| Candidate | Evidence | Classification/action |
|---|---|---|
| `src/la_backends/legacy.jl` | symbols referenced by `src/la_backend.jl`, `src/kkt.jl`, provider validations | live compatibility backend; retain until FactorCache/LA protocol consolidation |
| AppleAccelerate extension | `src/blas_backend.jl`, extension thread control | live optional capability; do not prune without capability/API migration |
| GenericLinearAlgebra extension | provider reference-role validation | live validation/reference provider; retain for now |
| JLD2 extension | checkpoint and spectrum public extension hooks | live optional public capability; retain unless public API is deliberately retired |

## Downstream compatibility roots

CSDR qualified surfaces remain live: ingest/compatibility model construction,
prepared/solve adapters where present, `solve_socp`, and fixed-trace Q3 helpers.
No downstream-qualified surface is deleted in the first Phase 2 batch.

## Required gate for old-lowering deletion

1. package load/precompile;
2. `test/runtests.jl` 21/21;
3. `validation/newton_system_reference.jl`;
4. `validation/symmetric_core_reference.jl`;
5. C1/C2 validations;
6. CLI C5/C6 suite;
7. fixed-trace Q3 references and CSDR alpha3 bit-identical certificate;
8. direct proof that `_public_lower_native` and `_public_result_from_lowering`
   have no caller after the edit.

## Execution update

- **Completed (`c383ccc`):** old family lowering/result stack removed.  Four
  lower/lift files and 1,049 lines of public dispatch/reconstruction were
  deleted (3,462 production lines total). All required gates passed and CSDR
  remained bit-identical.
- **Retained after compile probe:** `src/kkt.jl`. Core public loading does not
  execute it, but `SDPXMultiFloatsExt` still extends reduced-arrow and
  intermediate-panel methods/types defined by that compatibility stack;
  provider smoke fails if the include is removed.  It may be deleted only
  after splitting validation/legacy arrow methods out of the production MFLA
  extension and rerunning provider compatibility evidence.
