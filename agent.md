# SDPX.jl agent guide

## Mission

Build one arbitrary-precision Julia product-cone optimizer for numerical
bootstrap workloads:

- LP, SOC, RSOC, SDP, exponential, and power cones;
- Float64, MultiFloat, and BigFloat;
- one product-cone HSD state machine;
- one five-equation `NewtonSystem`;
- replaceable KKT/linear-algebra providers; and
- original-coordinate optimality and infeasibility certificates.

Current work is tracked in `docs/PLAN.md`. Mathematical contracts live in
`docs/design/`.

## Non-negotiable rules

1. Original-coordinate certificates are the only authority for terminal status.
2. Provider/factor success is not a mathematical certificate.
3. Do not delete or weaken an active test merely to obtain green output.
4. Do not loosen primal, dual, gap, cone, ray, or refinement tolerances to hide
   a regression.
5. Do not silently retry through a legacy engine, PSD lift, or Float64.
6. MFLA/BFLA own their kernels; SDPX uses provider interfaces rather than
   copying implementations.
7. `:bordered` remains the default KKT route until a complete evidence matrix
   justifies another default.
8. Generated benchmark results, review packages, and historical archives stay
   outside the source tree.

## Repository boundaries

- `src/` — solver implementation;
- `ext/` — optional provider extensions;
- `benchmark/general/` — general solver cases and public-data readers;
- `benchmark/bootstrap/physics/` — provenance-backed physical catalogs;
- `test/` — the sole black-box modeling-to-certified-result E2E;
- `validation/` — manual provider and independent mathematical checks;
- `docs/src/` — current user manual;
- `docs/design/` — frozen mathematical/design contracts.

## Implementation workflow

- Give every task a narrow file scope and one writer.
- Use an isolated worktree for delegated implementation.
- Prefer small, reversible local commits.
- Do not push intermediate work.
- During implementation waves, use static checks or narrowly scoped probes;
  run the unified E2E and specialist validation after integration.
- Record planned and executed KKT/provider routes separately.
- Keep exact operator residuals and five-equation residuals above provider
  diagnostics.

## Verification layers

### Default package test

`Pkg.test()` runs the deterministic black-box solve-to-result E2E from
`benchmark/general/`.

### Manual validation

- `scripts/provider_smoke.sh` — installed MFLA/BFLA integration;
- `validation/newton_system_reference.jl` — independent Newton fixtures;
- `validation/providers/` — provider-specific probes;
- `benchmark/general/performance/` — allocation/performance checks;
- `benchmark/bootstrap/` — physical build/solve/profile campaigns.

## Final integration

1. merge reviewed local commits;
2. freeze one candidate SHA;
3. run black-box E2E;
4. run required provider/MOI/precision/certificate checks;
5. repair failures without weakening contracts;
6. run local and cluster benchmark campaigns on the same SHA;
7. obtain independent final review; and
8. push once. Do not publish until release gates are complete.
