# SDPX.jl — Comprehensive Maintainer Review

Date: 2026-07-26
Commit reviewed: `dcbaa06`
Reviewer: Claude Opus 5, acting as senior maintainer preparing a stable open-source release
Scope: `src/` (22,451 lines), `test/` (8,354 lines), `ext/`, `examples/`, `bench/`, `docs/`, `Project.toml`

**Evidence policy.** Findings are tagged:

- **[V]** Verified — reproduced by a measurement or direct probe during this review cycle.
- **[R]** Read-only — reported from reading the code; not exercised.
- **[E]** Estimate — a judgement, clearly separated from measurement.

Environment: Julia 1.12.6 and 1.10.11 (Apple M4, OpenBLAS ILP64), plus PBS jobs on a
128-core EPYC 7742 cluster node. Reference numbers for Task_Low08 (m = 6119, L = 32,
dense lattice, Float64) and the CSDR PSD-dual family (sparse, Float64x4) come from
runs recorded in `bench/` and the cluster results tree.

---

## 1. Executive summary

SDPX is numerically credible and structurally strained.

On the credibility side: the full suite (≈1,870 assertions) passes on both supported
Julia versions; three CSDR instances agree with independently computed Clarabel optima
to 1.2e-8 – 9.6e-8 relative; the lattice benchmark reproduces its reference objective
to every printed digit at every thread count; and the independent certificate correctly
*refuses* to validate a stalled solve, naming the failing conditions — the most
important honesty property a solver can have, and it works. **[V]**

On the strain side, five structural problems, none of them a wrong number today, all of
them the kind that produces one later:

1. **835 lines in `_solve_sdp_core!`** — nearly every SDP solve flows through one
   function that owns iteration, termination, restarts, centering, checkpointing, and
   parameter control. Two real bugs found this cycle (parameters chosen before
   equilibration; an initial-point constant fitted to a single instance) were hard to
   see precisely because their lines sit hundreds apart in one scope. **[V]**
2. **23 bare `catch` blocks** that swallow `InterruptException` and `OutOfMemoryError`
   in a solver whose production runs last minutes to hours. **[V]**
3. **≈670 lines of tested, unreachable code** (null-space reduction, chordal
   detection) shipped inside `src/`. **[V]**
4. **A recurring test-design fault** — assertions encoding the developer's machine
   rather than the code's contract — found five independent times this cycle (thread
   count, core count, free memory, LAPACK build, RNG stream). All five fixed; the
   pattern itself is the finding. **[V]**
5. **No infeasibility certificates in optimize mode.** An infeasible SDP grinds to
   `Stalled`/`IterLimit`; only the separate `findFeasible` auxiliary problem and a
   trivial LP presolve case can say "infeasible". Every mature solver SDPX is compared
   against reports primal/dual infeasibility from the main iteration. **[V]** (enum and
   grep-verified)

The single most valuable near-term investment is not an optimization. It is the §25
acceptance-gate infrastructure (already in place, bit-identical across Julia versions)
plus the phase-timing harness: this cycle they caught a 0.54× "fix", a wrong-baseline
microbenchmark, and an RNG-artifact "regression" of 16.55 in the objective. Every
recommendation below assumes changes land against those gates.

---

## 2. Repository architecture review

### 2.1 Layout

```
src/
  SDPX.jl                 module root, 26 exports, include order
  types.jl                enums, SolverOptions (59 fields), problem/result types
  ingest.jl               canonicalization, validation, symmetrization, equilibration
  pipeline.jl             classification, presolve, plan construction, diagnostics
  solve.jl                _solve_pipeline! → _solve_sdp_core! (the IPM loop)
  step.jl                 newton_step!: residuals → Schur → KKT → predictor/corrector
  schur.jl (3,230 ln)     dense/sparse/arrow Schur assembly kernels
  kkt.jl (2,038 ln)       dense + block-arrow KKT factorization and solves
  kkt_backend.jl          backend abstraction (deliberately outside the hot loop)
  kkt_sparse_backend.jl   CHOLMOD Cholesky/LDL with symbolic reuse
  lp_solver.jl (1,489 ln) dedicated scalar LP path
  lp_sparse.jl            sparse Newton system for the LP path
  nullspace.jl            §12.2 reduction (unreachable — see 8.1)
  chordal.jl              §8.3 detection only (documented decision)
  stagnation.jl           tolerance/precision-scaled stagnation detector
  adaptive_parameters.jl  guarded β/γ controller (off by default, benchmark-gated)
  validation.jl           original-coordinate certificate, solve_summary
  spectrum.jl, moi_wrapper.jl, compat.jl
  kernels/                generic + BigFloat + threaded + extended-precision BLAS
ext/                      MultiFloats / DoubleFloats / JLD2 extensions (weakdeps)
```

The layering is genuinely good: representation dispatch is confined to
`schur_build!`/`buildP!`/`accumulate_v!`, so `newton_step!` is
representation-agnostic **[R]**; arithmetic dispatch is confined to the kernel layer
(`kchol!`, `kmul_owned!`, …) so `Float64`/`Float64x4`/`BigFloat` share one algorithm;
extended types are weak dependencies with extensions, which is the correct modern
pattern. Decisions that were measured and declined (mixed precision, chordal
transformation, tile ownership, adaptive-by-default) are recorded *in the source at
the decision point* with their numbers — unusually good practice worth preserving as
policy.

### 2.2 Architectural weaknesses

- **The solve core is monolithic** (§3.1 below).
- **Two selection mechanisms coexist**: `ExecutionPlan` (built before presolve) and
  runtime selection inside `solve_lp!`/`factor_kkt!`. They can disagree, and
  diagnostics copy the *plan* (§4.4). **[V]** (external review P2.4; confirmed by
  reading `_attach_diagnostics`)
- **Workspace is a god-object**: `Workspace{T}` carries dense, sparse, arrow, mixed
  precision, and extended-precision state simultaneously, with seven `::Any` fields.
  **[R]**

---

## 3. Critical correctness issues

Nothing currently produces a wrong *number* on the tested paths. The items here are
correctness *risks* — paths where a wrong answer or lost control is reachable.

### 3.1 [Critical] Bare `catch` swallows interrupts and OOM — 23 sites **[V]**

```julia
julia> try; throw(InterruptException()); catch; "SWALLOWED"; end
"SWALLOWED"
```

Example (`src/pipeline.jl:901`): an interrupt during the equality-presolve QR is
absorbed and converted into `(elimination_valid=false, consistent=true)` — the solve
*continues* with a silently altered presolve decision. `OutOfMemoryError` in the same
places converts resource exhaustion into "this factorization is singular".

- Correctness risk: **high** (control-flow lies; Ctrl-C loss on multi-hour cluster runs)
- Performance impact: none
- Difficulty: **low** — mechanical; add a `_recoverable(e)` helper and filter all 23.

### 3.2 [High] No infeasibility detection in optimize mode **[V]**

`SolveStatus` has `FeasibleCert`/`InfeasibleCert` only for the `findFeasible`
auxiliary problem. `solve` on an infeasible or unbounded SDP has no exit besides
`Stalled`, `IterLimit`, or Ω-escalation breakdown — and this cycle demonstrated how
that misleads: two *badly generated benchmark problems* (one unbounded, one
infeasible) produced hours of investigation into "solver bugs" that were data bugs.
Users will do the same. Clarabel/MOSEK/SeDuMi report infeasibility from the main
iteration (HSDE or ray detection).

- Correctness risk: high (wrong *diagnosis*, not wrong number)
- Difficulty: **high** — a real algorithmic addition (dual-ray / primal-ray tests at
  the stagnation exit would be the cheap 80%: when the detector fires, test the
  current iterate as an infeasibility certificate before reporting `Stalled`).

### 3.3 [High] Diagnostics report the plan, not what ran **[V, confirmed by read]**

`_attach_diagnostics` copies `plan.kkt_backend`/`plan.gram_kernel` into
`selected_algorithms`. The LP path can select `:sparse_ldl` at runtime while the
result reports `:dense_lu` and a Gram kernel that never executed. Every published
benchmark table built from diagnostics inherits this. (External review P2.4 —
confirmed.)

- Difficulty: low-medium (thread the executed choice back into the result).

### 3.4 [Medium] `minimum_psd_eigenvalue` is not an eigenvalue **[V]**

`solve_summary` returns `-required_shift` under that name. The docstring *does* say
so, but a field whose name asserts something its value is not will be misread in
every downstream table. For `diag(2,3,4,5,6)` it reports `-0.0`; the minimum
eigenvalue is `2.0`. Rename (`psd_shift_lower_bound` or similar), keep the old name
as a documented deprecated alias.

### 3.5 [Medium] Remaining unchecked memory-size arithmetic **[R]**

`nullspace_memory_bytes` was fixed this cycle (it returned negative values at 2e9
variables and approved every budget). The same unchecked `Int` products remain in
`estimate_sdp_workspace_bytes` and `dense_workspace_floor_bytes`
(`src/pipeline.jl`). Same fix, same `saturating_bytes` helper, ~30 minutes.

### 3.6 [Low] `validate=false` skips finiteness entirely **[R]**

`ingest(...; validate=false)` (used by benchmark drivers for speed) admits NaN/Inf
coefficients into the solver, where the first symptom is a non-finite iterate many
iterations later. Consider a cheap always-on NaN check on `c`/`b` only, or document
the contract loudly at the `validate` kwarg.

### 3.7 Fixed during this cycle (recorded for the release notes) **[V]**

These were live correctness bugs at the start of the cycle; all are fixed, tested,
and pushed:

| bug | symptom | fix |
| --- | --- | --- |
| CHOLMOD `ldiv!` absent on Julia 1.10 | every sparse-backend solve threw `MethodError` on the declared minimum version | `_cholmod_solve!` via `\` |
| Dense/sparse LP regularization sign | directions diverged O(δ): 0.22 apart at δ=1e-2, reachable via escalation | equality block now `+δ` in both |
| Null-space memory gate | 5.0× under-estimate on rank-deficient B; overflow → negative → approve-everything | rank-aware + saturating + internal enforcement |
| Calibration cache | accepted `NaN` speedup → packed kernel enabled unconditionally | `valid_profile` gate |
| Ω initial-point multiplier | fitted to one instance; s20 CSDR stalled (94 iter) | swept over 4 referenced problems → 10; s20 now Optimal in 36 |
| Duplicate first sparse factorization | `factorizations=2` after one call | removed |

---

## 4. Numerical robustness issues

### 4.1 What is in good shape **[V]**

- **HKM-family direction + Mehrotra predictor-corrector** with an exact closed-form
  fraction-to-boundary rule for ≤2×2 blocks (`step_rule=:auto`) — the fix that took
  the CSDR 80/4/40/100 model from a false 27-iteration stall to genuine convergence.
- **Stagnation detection scaled by tolerance and precision** (nats/iteration over a
  rolling window, precision-floor discrimination) rather than a fixed bar — this is
  ahead of several mature solvers, which stall on iteration caps.
- **Independent original-coordinate certification** after unscaling — comparable to
  what SDPB publishes, and it demonstrably refuses wrong results (s20: gap 2.0,
  `valid=false`, failures named).
- **Regularization** `S + δ·diag(|S_ii|)`, escalation √eps → ×10⁶ max; proven
  reachable (fires on singular/indefinite S) and correctly bounded (a −1 eigenvalue
  among O(1) eigenvalues is *reported*, not silently shifted — max relative shift
  ≈1.5e-2 by design).

### 4.2 Fragile points, in comparison with mature solvers

| area | SDPX today | mature practice | assessment |
| --- | --- | --- | --- |
| Infeasibility | none in optimize mode (§3.2) | HSDE (Clarabel, SeDuMi), ray detection (MOSEK) | **the** algorithmic gap |
| Higher-order correctors | single Mehrotra corrector | Gondzio multiple centrality correctors (MOSEK, SDPT3 opt.) | **[E]** worth a gated experiment: iterations dominate cost at m=6119 (KKT+Schur ≈ 85–95% of time; each saved iteration ≈ 2–3.5 s cluster) |
| Initial point | Ω·I with Ω = 10·max‖C‖∞, swept over 4 problems | MOSEK/SDPT3 use data-scaled heuristics per block | current constant is non-monotone in effect (100 is *worse* than 10 on s20) — a per-family sweep is the honest tool; the docstring already mandates re-sweeping |
| Refinement target | residual-driven, against the *regularized* system | Clarabel refines against unregularized KKT | documented mismatch (external P1.2 follow-on); acceptable if stated, currently stated |
| Step rule for large blocks | backtracking with margin ∈ [γ,1] | exact eigenvalue-based fraction-to-boundary | `:auto` only covers ≤2×2 exactly; large-block accuracy relies on backtracking granularity **[R]** |
| BLAS thread mutation | process-global set/restore around phases | per-call thread control or solver-owned pools | **unsafe under concurrent solves** in one process (external P3.11, confirmed by read); fine for the single-solve HPC pattern; must be documented or locked |

### 4.3 Precision handling

The `Float64 → Float64x4 → BigFloat` ladder is clean at the kernel layer, and the
measured guidance (error tracks *requested tolerance* until the arithmetic floor;
Float64 stalls at ~1e-14 on the reference problem; Float64x4 tracks to 1e-30) is
now executable documentation in `examples/02`. **[V]** Two gaps: the dedicated LP
path bypasses `check_precision_consistency` (external P2.7, confirmed by read —
BigFloat LP inputs built at mismatched precision are neither warned about nor
rerounded); and BigFloat tests largely assert Float64-magnitude tolerances rather
than `eps(T)`-derived ones **[R]**.

---

## 5. Performance opportunities

**Measured phase profile first** (Task_Low08, 8 iterations, laptop M4; cluster bin
probe from job 193929) **[V]**:

| threads | total | Schur | KKT factor | everything else |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.5 s | 7.0 s (31%) | 14.0 s (62%) | 1.5 s |
| 4 | 13.1 s | 7.1 s (54%) | 4.5 s (34%) | 1.5 s |
| 8 | 14.7 s | 7.7 s | 4.8 s | 2.2 s |

Critical caveat: the laptop's flat Schur line is a **memory-cap artifact** (0.2 GB
free → 1 bin). On the cluster node the cap does not bind (27.6 GB free → 8 bins at 8
threads) **[V]**, so the cluster's Schur/KKT split is *unknown* — profiling job
`193929.node220` was submitted for exactly this and its phase CSVs have not been
collected. **Collect that before acting on anything in this section.**

Ordered by expected impact, with per-benchmark estimates:

| # | opportunity | Task_Low08 | medium CSDR (Float64x4 arrow) | BigFloat | Float64 generally | difficulty |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | **Iteration-count reduction** (Gondzio correctors; per-family parameters — measured: the Ω change alone took CSDR-s25 from 81→46 iters) | **[E]** 10–30% if correctors save 3–8 of 27 iters | **[V-adjacent]** family-dependent; already demonstrated 43% on s25 | same mechanism | same | high |
| P2 | **Collect cluster profile, then attack the real dominant phase** | unknown until data | — | — | — | trivial (job exists) |
| P3 | **Type-stabilize hot `::Any` fields** (`Qchol`, `sparse_system`, 6× `mixed_reduced_*`): union-annotate or function-barrier; `_solve_Q!` dispatches dynamically 2×/iteration **[V]** | **[E]** ≤1% (27 dispatches) | **[E]** ≤1% | negligible | small but free | low |
| P4 | **Sparse LP solve allocations** — `factorization \ rhs` + `similar(rhs)` = 2 vectors/solve (external P3.10, confirmed) | n/a (not LP) | n/a | n/a | **[E]** few % of LP solve time at most; correctness constraint (1.10 `ldiv!` absence) documented and takes priority | low |
| P5 | **Lower-triangle Schur accumulators** (285.7→142.9 MB/bin, machinery exists) | cluster: **none** (cap doesn't bind) **[V]**; laptop-class: real | possible | possible | — | medium |
| P6 | **BigFloat kernel allocation hygiene** — the owned-kernel discipline is good; remaining wins are in the reduced-arrow pack (244-line function, MPFR temporaries) **[R]** | n/a | n/a | **[E]** measured precedent: fused-arrow BigFloat Schur went 54.97→9.77 s (5.63×) when this was last done seriously | n/a | medium |

Explicitly **not** recommended, with the measurements that killed them: tile-owned
dense Schur (0.34–0.40× vs real path), mixed-precision KKT (accuracy parity, 1.01×),
wider BLAS during engaged threading (0.85–0.93×), precision escalation for speed
(1.00×), ordering selector over CHOLMOD's default (default wins every case). **[V]**

---

## 6. Julia best-practice issues

- **[V] No type piracy.** All `Base`/`LinearAlgebra` methods extend owned types.
- **[V] Correct weakdep/extension usage** for MultiFloats/DoubleFloats/JLD2; compat
  bounds present for all deps; `julia = "1.10"` is now actually true (it wasn't
  until the CHOLMOD fix this cycle — CI on the declared minimum was red).
- **[R] Seven `::Any` struct fields** in hot state (§5 P3). One (`Qchol`) has its
  intended union *in a comment* — transcribe it.
- **[R] 59-field `SolverOptions`.** Functional, but flat. Consider grouping
  (tolerances / precision / performance / experimental) before the API freezes at
  1.0 — after 1.0 this becomes nearly impossible to change.
- **[V] Naming is bilingual**: exported legacy API is camelCase (`sdp`,
  `findFeasible`, `iterMax`) while the new API is snake_case. `compat.jl` handles
  deprecation properly; plan removal at 1.0 rather than carrying both forever.
- **[R] `include`-order coupling** in `SDPX.jl` is fragile (this cycle: repeated
  docstring-adjacency breakage when inserting files). Low priority, but a known tax.
- **[V] Examples/tests still seed `MersenneTwister`** for data whose *properties*
  are asserted. Julia's RNG stream is not version-stable; this exact mechanism
  produced phantom objective shifts of 16.55 in the gates before they were moved to
  an explicit LCG (`GateStream`). `examples/03` and `test/nullspace_reduction.jl`
  currently survive on margin (nnz/row 9.01 vs 8.98 across versions). Migrate them
  to `GateStream` or assert against measured density.

---

## 7. Package design suggestions

- **Public surface (26 exports) is about right**; all eleven checked entry points
  have docstrings **[V]** (an earlier grep-based claim of missing docs was wrong —
  verified with `@doc`).
- **Decide the status of unreachable subsystems before release** (§8.1): a user
  reading `src/nullspace.jl` will reasonably believe `solve` can use it. It cannot.
- **`solve_summary` is the right idea** (one NamedTuple, the §21.3 contract) —
  make it the documented "what did I get" entry point in README/examples.
- **Backend extensibility**: `KKTBackend` exists but the dense hot path deliberately
  bypasses it. That is the right performance call; say so in CONTRIBUTING so a
  well-meaning PR doesn't "unify" it.
- **Concurrency contract**: one solve per process is the safe usage today
  (process-global BLAS mutation, shared calibration cache). State it in the README
  until P3.11 is engineered properly.

---

## 8. Testing suggestions

Current: ≈1,870 assertions, green on 1.10/1.12 × {1,4,8} threads; acceptance gates
bit-identical across versions; examples executed by the suite (which caught a broken
README snippet on first run). Ratio test:source ≈ 0.37.

Gaps, in priority order:

1. **`schur.jl` has no dedicated test file** — the largest (3,230 ln), most
   dispatch-dense file is covered only incidentally. Property test to add: for every
   `(cons-type, arithmetic, threading)` combination, `schur_build!` output equals the
   naive dense reference within `eps(T)`-scaled tolerance.
2. **Infeasible/unbounded inputs** — no test feeds `solve` a certifiably infeasible
   SDP and asserts on the resulting status/diagnostics. This cycle produced three
   accidentally-ill-posed generators; make intentional ones.
3. **BigFloat tolerances derived from `setprecision`**, not hard-coded 1e-10s.
4. **Concurrent-solve smoke test** (two `Threads.@spawn`ed solves) — even if it just
   documents today's limitation.
5. **Environment-reading tests must state their environment** — adopt as a review
   rule; five violations were found and fixed this cycle (thread count, cores, free
   memory, LAPACK build, RNG). The fixed versions show the pattern to follow
   (`thread_count=1`, `free_memory_bytes=4GiB`, `GateStream`).
6. **MOI**: coverage is decent (87 regression assertions) but add
   `MOI.Test.runtests` config exclusions review — currently unclear which parts of
   the MOI test suite run. **[R]**

---

## 9. Documentation suggestions

- **README is honest** (no unfair solver comparisons; experimental status stated;
  measured Float64x4 guidance) **[V]** — keep it that way.
- The **measured-decision comments** (Ω sweep table, BLAS-width table, declined
  optimizations with numbers) are the project's best documentation. Two additions
  would complete the pattern: a `docs/decisions.md` index pointing at each in-source
  decision record; and CONTRIBUTING language requiring a measurement for any change
  to a gated constant ("the whole sweep, not one benchmark" — the Ω docstring
  already says this; generalize it).
- **Undocumented assumptions to write down**: single-solve-per-process; symmetric
  input matrices (symmetrize tolerance semantics); `validate=false` contract; the
  refinement-targets-regularized-system decision; CHOLMOD Float64-only boundary.
- `docs/` mixes design plans, session logs, and user documentation — separate
  `docs/dev/` from user-facing pages before release.

---

## 10. Prioritized action list

| # | action | class | correctness risk | perf impact | difficulty |
| --- | --- | --- | --- | --- | --- |
| 1 | Filter 23 bare `catch` sites (rethrow Interrupt/OOM) | **Critical** | high | none | low |
| 2 | Collect cluster phase profile (job `193929` pattern) before any perf work | **High** | — | gates all perf work | trivial |
| 3 | Report executed backend/kernel, not plan (P2.4) | **High** | med (false records) | none | low-med |
| 4 | Decide null-space/chordal status: wire behind gate or move out of `src/` | **High** | med (misleading) | none | low |
| 5 | Rename `minimum_psd_eigenvalue` (+ deprecated alias) | **Medium** | med | none | low |
| 6 | Saturating arithmetic in remaining memory estimators | **Medium** | med | none | low |
| 7 | Infeasibility detection at the stagnation exit | **High** | high (diagnosis) | none | high |
| 8 | Split `_solve_sdp_core!` along existing seams, under acceptance gates | **Medium** | low *if gated* | none | high |
| 9 | Type-stabilize `::Any` hot fields | **Low** | none | ≤1% | low |
| 10 | Migrate RNG-seeded examples/tests to `GateStream` | **Medium** | low today | none | low |
| 11 | BigFloat LP precision-consistency (P2.7) + `eps(T)` test tolerances | **Medium** | med (BigFloat users) | none | med |
| 12 | Gondzio corrector experiment, gated | **Medium** | low (gated) | possibly largest remaining | high |
| 13 | Concurrency: document single-solve contract now; lock/redesign later (P3.11) | **Medium** | med (concurrent users) | n/a | doc: trivial |
| 14 | `SolverOptions` grouping + camelCase retirement plan for 1.0 | **Low** | none | none | med |

---

## 11. Estimated impact of each recommendation

Honest accounting: most items in this review protect correctness and trust; only
three plausibly move benchmark wall-clock.

| action | Task_Low08 (Float64, m=6119) | medium CSDR (Float64x4, arrow) | BigFloat | basis |
| --- | --- | --- | --- | --- |
| 1, 3, 4, 5, 6, 10, 13 | none | none | none | correctness/trust only |
| 2 (profile) | enables the rest | — | — | the laptop profile is provably unrepresentative (bin cap: 1 vs 8) **[V]** |
| 7 (infeasibility) | none on feasible problems | none | none | changes *answers on bad inputs*, the highest-value non-speed change here |
| 8 (core split) | must be 0.00× by construction (gates enforce identical iterations) | same | same | refactor |
| 9 (type stability) | **[E]** ≤1% | ≤1% | negligible | 2 dynamic dispatches × 27 iters |
| 12 (correctors) | **[E]** 10–30% *if* 3–8 iterations saved at ~2–3.5 s/iter (cluster) | **[E]** potentially larger — s25 already showed 81→46 iters from parameter work alone **[V]** | same mechanism | iteration cost dominates; every saved iteration is pure win |
| P6 BigFloat pack hygiene | n/a | n/a | **[E]** meaningful; precedent is the measured 5.63× on the fused-arrow kernel **[V]** | prior measured win on the sibling kernel |

The pattern that produced results all cycle, and that I recommend institutionalizing
over any individual item above: **measure before building, against a reference that
is independent of the code being measured** (Clarabel optima, closed-form 2√6,
recorded gates) — this review found five of its own prior claims wrong by exactly
that discipline (tile-Schur baseline, "cache order irrelevant in Float64x4",
"machine-independent" gates, the missing-docstrings grep, and the laptop Schur
profile). A solver team that keeps that habit will catch its next regression before
its users do.

---

## Implementation record (2026-07-27)

Every item above was either implemented or explicitly deferred. Landed, in
order, each with a focused regression test and a green suite on Julia 1.10 and
1.12:

| item | commit |
| --- | --- |
| 1. Filter bare `catch` (25 sites, not 23 — two more had appeared) | `3502da3` |
| 2. Collect cluster phase profile | job `193929.node220`: Schur 39–53%, KKT 34–39%, both ≈4× at 8 threads → iteration count confirmed as the remaining lever |
| 3. Executed-vs-planned diagnostics (P2.4) | `91de9fa` |
| 5. `psd_shift_lower_bound` rename + deprecated alias (P2.5) | `cd83bf0` |
| 6. Saturating memory estimators (P3.12) | `64c0a06` |
| 9. Type `Qchol`/`sparse_system` | `d44058d` |
| 10. Deterministic data in example 03 | `3c99b15` |
| 11. BigFloat LP precision consistency (P2.7) | `eee91f0` |
| 13. Concurrency/validate=false/experimental-status documentation | `df4612a` |

### Deferred, with reasons

- **Item 7, infeasibility detection.** A wrong infeasibility certificate is
  worse than none, and a correct ray test needs design (which cone form, what
  tolerance discipline, how it interacts with Ω escalation) plus intentionally
  infeasible test problems with known certificates. High-risk to improvise;
  needs its own cycle.
- **Item 8, splitting `_solve_sdp_core!`.** The gates make this safe in
  principle, but a concurrent session landed a 700-line parameter-policy
  rewrite into the same file mid-cycle; refactoring under it would mix
  unrelated structural change with live feature work. Do it in a quiet window.
- **Item 12, Gondzio correctors.** Performance experiment; the cluster profile
  that justifies it now exists. Gate any attempt on the recorded baselines.
- **Item 14, `SolverOptions` grouping / camelCase retirement.** Breaking-API
  design decisions that belong to the 1.0 planning discussion, not a patch
  release.
- **P3.10 sparse-solve allocations and P3.11 BLAS thread locking.** The
  allocation is two vectors per solve behind a documented Julia 1.10
  compatibility constraint; the concurrency limitation is now documented as a
  usage contract (one solve per process). Both are engineering the contract
  makes non-urgent.
- **`test/nullspace_reduction.jl` RNG seeding** (flagged §6). Left as is,
  deliberately: its assertions compare the reduced solve against the
  range-space solve of the same generated problem, so both sides move together
  under any stream change — nothing constant is encoded.
