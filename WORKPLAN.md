# WORKPLAN — SDPX v2 (zero-alloc FactorCache + optional LinearSolve + production HSD)

Coordinator: this file is the single source of truth for task dependencies, file
ownership, branches, and commit policy. Do NOT merge the old
refactor/zeroalloc-factorcache-hsd branch; it diverged from main (main rewrote
KKT/Schur/planner/interior-point/optimize).

## Branch
- refactor/zeroalloc-factorcache-hsd-v2 (created from origin/main @ 7febb46).
- Each subagent works on its own branch; coordinator reviews and cherry-picks.

## Old-branch commit classification (56 commits, merge-base d185450)

### Portable benchmark/test (Subagent 1 adapts to new main's benchmark/ + test/ infra)
- 5307a21 bench(baseline) multi-precision LP/SOCP/SDP
- ccf263a per-iteration allocation profiler + gate
- f8e02c2 per-phase allocation breakdown
- 66f9879 full-family per-iteration allocation regression gate
- eda737a fixed-precision contract gate
- 1199344 LP/SOCP multi-precision benchmark
- 63fc218 full-family infeasibility-certificate contract gate
- ec67976 tightened allocation ceilings
- aa6b9e1 ChordalReduction clique-cover validation (chordal.jl still exists)

### Low-risk perf (must RE-APPLY + re-verify on new main's hot loop; main rewrote step.jl)
- d481dff, db1cf53, bbf22a5, ebc4358, 6ee8397 (destructuring / closure-inlining /
  phase-times relocation) — re-derive on the new newton_step.
- e133e6e typed Workspace.backend — re-derive on new Workspace.

### Must-rewrite on new main (do NOT cherry-pick)
- 0c0c663 FactorCache; cc7e441/8c24e8f/3bc85e3/d05244b/1ce5245 HSD;
  b7104c0/ebdc23e/334e78e ConeAlgebra; 3173c9f/bd71fa0/d5f161f/8db3ea6/9b4a173/
  094c8dd/bfbe132/6d40e0e SymmetryReduction.

### Docs (port as reference; numbers are stale, regenerate from scripts)
- e8cb819, 74214ed, 3310a94, 902654d, 002b72e, 3634c9c, 933bfbb, 27db3dc,
  258564e, 2c80358, dc529ee.

## Frozen architecture
CanonicalProblem -> ReductionPlan -> ExecutionPlan{T,Route,Formulation,LAProvider,FactorCache} -> SolverState -> original-coordinate certificate.

## File ownership (no two subagents touch the same core file)
- Subagent 1: benchmark/, test/ baseline + fixed-precision + allocation gates.
- Subagent 2: ext/SDPXLinearSolveExt.jl, Project.toml weakdeps (optional only).
- Subagent 3: src/factor_cache.jl protocol + test/factor_cache.jl.
- Subagent 4: src/la_backends/mfla_cache.jl (MFLA 0.2 adapter).
- Subagent 5: src/la_backends/bfla_cache.jl (BFLA owned cache).
- Subagent 6: src/kkt*.jl, src/schur.jl, src/lp_solver.jl integration (after 3/4/5).
- Subagent 7: src/step.jl, src/workspace.jl (zero-alloc + type stability).
- Subagent 8: src/hsd.jl + LP HSD in lp_solver.jl.
- Subagent 9: src/cone_algebra.jl (math validation).
- Subagent 10: src/reduction_plan.jl, src/symmetry_reduction.jl, src/chordal.jl.
- Subagent 11: .github/workflows/*, Project.toml compat.

## Waves
A (parallel audit): 1 baseline, 2 LinearSolve spike, 3 FactorCache protocol,
  7 allocation/type audit, 9 ConeAlgebra math audit.
B (providers): 4 MFLA, 5 BFLA, 11 CI version matrix.
C (integration): 6 KKT, 7 allocation gate, 2 LinearSolve decision.
D (HSD): 8 LP HSD -> SOC -> SDP.
E (reduction): 10 Symmetry/Chordal (setup-only until C/D stable).

## Merge strategy (PR sequence, each independently revertible + before/after benchmark)
PR1 baseline+CI+version contract; PR2 FactorCache protocol; PR3 MFLA/BFLA adapters;
PR4 dense/equality KKT; PR5 LP/SOC/arrow/sparse; PR6 zero-alloc hard gate;
PR7 optional LinearSolve; PR8 LP HSD; PR9 SOC/SDP HSD; PR10 validated reduction.

## Hard gates
- Float64/x2/x4/BigFloat256 one_full_iteration == 0 Julia bytes (Subagent 7).
- factorizations_per_iteration == 1 (Subagent 6).
- Original-coordinate certificate re-verified at target precision (Subagent 8).
- No unauthorized fallback; no silent clip of significant negative eigenvalues (Subagent 9).

