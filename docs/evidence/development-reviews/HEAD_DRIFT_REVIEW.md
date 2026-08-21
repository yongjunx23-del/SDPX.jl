# SDPX v0.5 HEAD drift review

Review date: 2026-08-16

## Baseline and current candidate

- Review baseline: `eca719c62820fd7bf6c3317245af9f78978898f3`
- Development HEAD: `12b313efc5c984cebe7826689be7ac73aa0c5a5c`
- Development tree: `876086b25c7bfc7779c7ea1120de003de4fa72d7`
- Development commit date: `2026-08-16T20:46:27+08:00`
- Branch: `development/v0.5-execution-plan`
- Draft PR: `#8`
- Commits since the review baseline: 50

The current candidate is intentionally ahead of the fixed review baseline. The
baseline reports remain useful as problem statements, but their line numbers are
not used as implementation instructions. All subsequent evidence is tied to the
full development HEAD above or to a later explicitly recorded candidate.

The original local `main` checkout contains unrelated uncommitted work and was
not reset, switched, stashed, or overwritten. Development continues in a clean
repository-external checkout of the GitHub development branch.

## Name-status drift from the fixed baseline

```text
A .codex/ci-after-n3.marker
A .codex/ci-after-n4.marker
A .codex/family-direction-gates.patch.gz.b64
A .codex/n1-acceptance-fix.patch.gz.b64
A .codex/n2-diagnostic.marker
A .codex/n3-equality-scaling.patch.gz.b64
A .codex/n3-kkt-a.patch
A .codex/n3-kkt-b.patch
A .codex/n3-kkt-c.patch
A .codex/n3-kkt-d.patch
A .codex/n3-kkt.patch
A .codex/n3-proportional-rank.patch
A .codex/n3-proportional-tests.patch
A .codex/n3-tests.patch
A .codex/n3-workspace.patch
A .codex/wave1.patch.gz.b64
A .github/workflows/codex-apply-family-direction-gates.yml
A .github/workflows/codex-apply-n1-fix.yml
A .github/workflows/codex-apply-n3-equality-scaling.yml
A .github/workflows/codex-apply-wave1.yml
A .github/workflows/codex-export-source.yml
A docs/development/SDPX_v0.5_execution_plan.md
M src/kernels/mixed_precision_kkt.jl
M src/kkt.jl
M src/pipeline.jl
M src/soc_native.jl
M src/solver/interior_point.jl
M src/step.jl
M src/types.jl
M src/validation.jl
M src/workspace.jl
M test/adaptive_parameter_policy.jl
M test/kkt_regressions.jl
M test/mixed_precision_kkt_regressions.jl
M test/pipeline.jl
M test/result_certificate.jl
M test/soc_native_solver.jl
M test/sparse_schur_round7.jl
M test/v05_core_invariants.jl
```

## Task impact classification

| Task | Drift verdict | Current interpretation |
|---|---|---|
| N1 true SDP trial residual | Fixed on the development branch | Keep the unregularized structured residual and accepted-trial carry already present. |
| N1 acceptance/refinement separation | Fixed on the development branch | Do not collapse the outer direction-safety gate back into the strict correction target. |
| Runtime equality scaling | Fixed on the development branch | Preserve normalized equality coordinates and original-coordinate multiplier recovery. |
| Mixed equality scaling parity | Fixed on the development branch | Preserve rung-local scaling and recovery. |
| N3 FixedTrace rank parity | Fixed on the development branch | Keep requested-accuracy numerical-rank policy and plan-authorized RRQR only. |
| N4 fixed refinement rollback | Fixed on the development branch | Keep worsening/nonfinite rollback with the retained factor. |
| N5 sparse equality quarantine | Fixed on the development branch | Keep equality-bearing generic sparse Schur unsupported and fail closed. |
| D1 minimal original-coordinate Optimal gate | Fixed on the development branch | Keep strong status semantics even when detailed certification is disabled. |
| N7 LP/NativeSOC validated direction | Staged, not yet assumed complete | Inspect the staging payload and production source; production completion requires source, tests, and semantic review. |
| A0/O0 | Still applicable | Implement after N7 status is resolved; preserve existing diagnostics compatibility. |
| A1-A6 | Still applicable | Implement in dependency order after A0 schema is stable. |
| LA1-LA5 and P1-P5 | Still evidence-gated | Do not modify providers or keep performance work without current end-to-end evidence. |
| M1-M9 and release gates | Still applicable | Cleanup follows numerical and architectural stabilization. |

## Temporary machinery

The `.codex` payloads and `codex-*` workflows are historical staging machinery,
not production completion evidence. They remain temporarily because the N7
payload must first be inspected and, if valid, materialized into ordinary source
and tests. All such machinery must be removed before the final candidate is
validated.

## Decision

Drift review: `PASS` for continuing development from
`12b313efc5c984cebe7826689be7ac73aa0c5a5c`.

Release decision remains `NO-GO` until the required N7, A0-A6, provider,
performance, cleanup, CI, application, certification, and release-evidence gates
are closed on a clean final tree.
