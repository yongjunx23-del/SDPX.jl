# SDPX support matrix (R6-A)

Status: **2026-09-09**, dev branch `development/scientific-core-20260907`, HEAD `f74a465`, SDPX 0.6.1.
This matrix records the *actual* qualification state. A route listed as
`experimental` is opt-in and fail-closed; `unavailable` means the request is
refused or returns a non-optimal result with an invalid certificate. No entry
claims more than its evidence supports.

Authority documents: `docs/design/SCIENTIFIC_CORE_ROADMAP.md` (R0–R6 plan),
`docs/design/R0P4_FACTOR_PAIR_BOUNDARY.md` (opt-in half-Power design),
`validation/scientific_core/exp_runtime/COMPENSATED_EXP_DESIGN.md` (R0-E
research design).

## Default public route (`nonsymmetric_backend = NativeNonsymmetricBackend`)

| Area | Status | Evidence / caveat |
|---|---|---|
| LP / orthant (Float64) | supported | full suite; bordered + expanded routes; `test/runtests.jl` |
| SOC / RSOC (Float64) | supported | full suite; route guard tests |
| PSD (Float64, `PSDTriangle`) | supported | full suite; PSD finite gate + spectral route |
| Q3 / fixed-trace compact kernels | supported (route-internal) | full suite; `fixed_trace_q3` tests |
| BigFloat 256/512 | supported where the provider matrix passed | precision ladder records; `BigFloatLinearAlgebra` provider tests |
| MultiFloats x2/x3/x4 | supported where MFLA providers loaded | `validation/providers/*`; provider qualification records |
| Float64 Power products | **unavailable (known failure)** | `numerical_breakdown`, invalid certificate; frozen in `validation/scientific_core/power_runtime/`; E2E known-issue control `test/runtests.jl` (flip back when R0-P4 default dispatch is qualified) |
| Float64 Exp products | **unavailable (known failure)** | `:line_search_breakdown`; frozen in `validation/scientific_core/exp_runtime/`; E2E known-issue control |
| Mixed Exp/Power/PSD products | unavailable (known failure follows the component) | same frozen controls |
| Sparse KKT memory admission | unavailable (fail-closed) | R3-D: no complete peak bound; sparse routes remain research-only |
| Cross-solve symbolic reuse | **not implemented** | R2-A counter: 100 same-structure updates → 100 CHOLMOD analyses (`validation/scientific_core/test_r2a_symbolic_numeric_separation.jl`) |
| Iteration history / trace | unavailable by design | `optimize: engine=:native_hsd does not publish iteration history` |

## Experimental opt-in routes (never default, fail-closed)

| Route | Selector | Scope | Status |
|---|---|---|---|
| Half-Power factor-pair HSD | `nonsymmetric_backend=ExperimentalHalfPowerFactorPairBackend` | Float64, orthant + exactly half-Power blocks (>=1), contiguous layout, `kkt_route=:bordered`, `provider=:auto`, `formulation=:auto`, `sparse=:auto`, no equilibration, threads=1, classic knobs | **experimental; qualification incomplete**: public `optimize!` → `:optimal`, `certificate.valid=true`, obj err `1.48e-9` on `power_epigraph_small`; tested excluded shapes refuse typed; a 64 MiB estimate threshold is enforced, but the estimate is incomplete and does not bound actual peak memory; ordinary original-coordinate certificate authority |
| PSD relative-2 spectral route | `PSDNTScaling(; eigen_route=:experimental_relative2)` | Float64, n=2, explicit only | research; dyadic residual `7.9e-31` vs production `0.707`; no fallback |
| Experimental sparse core | route-specific opt-in | research only | memory bound incomplete (R3-D); not a public sparse claim |
| Compensated Exp evaluator | none (validation-only) | frozen records | research: 6/6 records, log radii ~1e-33; **no production dispatch change** |

## Known failures and tracking pointers

| Known failure | Tracking | Flip-back requirement |
|---|---|---|
| Default Float64 Power breakdown | R0-P4 | E2E + public-qualification controls assert non-optimal; flip only after default-dispatch qualification |
| Default Float64 Exp breakdown | R0-E | same controls; flip only after an Exp repair is admitted |
| `power_epigraph_small` / `mixed_orthant_exp_small` E2E | R0-P4 / R0-E | `test/runtests.jl` known-issue control with explicit flip-back note |
| Expanded-route mixed-Exp ownership tests | R0-E | `test/native_structure_diagnostics.jl` known-issue solve control |
| R2-A symbolic reuse gate | R2 | test encodes the truthful 100-update linear law; update when reuse lands |
| Sparse KKT peak memory | R3-D | admission stays unavailable until a complete bound exists |
| N14 BigFloat512 certification | R6-C | open; finite-N14 correctness does not qualify all bootstrap families |
| New-standard CSDR baseline | R6-C | open; old 101-iteration qualification/fingerprint must not be reused |
| SDPX/SDPB 8/16/32-core comparison | R6-C | open; requires real PBS allocations (no oversubscription claims) |

## Arithmetic/precision identity

- MultiFloats effective bits: `53N - (N-1)` (x2 105, x3 157, x4 209); BigFloat
  precision is the requested bit count; Float64 is 53. Do not conflate x4,
  209/208 bits and BigFloat256 as one configuration.
- The experimental half-Power route requires the verified arithmetic context
  (Julia 1.12.6, aarch64 Darwin, default math, nearest rounding, FTZ off, FMA
  available). Other contexts refuse typed; the tests assert truthful refusal.

## Reproduction

```
julia --startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G \
  --project=/tmp/sdpx-scientific-core-env-20260907 \
  -e 'using SDPX, Test; include("test/runtests.jl")'
```

Last full run: **6825/6825 assertions, 0 failures** at `f74a465`.
