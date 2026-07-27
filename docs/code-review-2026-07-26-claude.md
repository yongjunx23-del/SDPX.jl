# SDPX code review

Date: 2026-07-26
Reviewer: Claude Opus 5
Commit reviewed: `dcbaa06`
Scope: `src/` (22,451 lines), `test/` (8,354 lines), `examples/`, `bench/`

## Method

Findings below are marked **Verified** where I reproduced them with a
measurement or a direct probe, and **Read-only** where I am reporting what the
code says without having exercised it. I checked several suspicions that turned
out to be wrong; those are listed too, because a review that only reports hits
gives no sense of its own false-positive rate.

Environment for all measurements: Julia 1.12.6 and 1.10.11, OpenBLAS (ILP64),
Apple M4 (10 cores), plus one PBS job on the cluster (EPYC 7742, 32 GB
allocation).

## Summary

The codebase is in good shape numerically. The whole suite passes on both
supported Julia versions, the solver agrees with independent references
(Clarabel) to 1e-8 relative on three CSDR instances and reproduces the lattice
benchmark objective exactly, and the certificate machinery correctly refuses to
certify results that are wrong.

The problems are concentrated elsewhere:

1. one 835-line function that most of the solver's control flow passes through;
2. 23 exception handlers that silently swallow `InterruptException`;
3. ~670 lines of implemented, tested, unreachable code;
4. a recurring test-design fault — assertions that encode the developer's
   machine rather than the code's contract — found five separate times;
5. dynamic dispatch on a hot-loop field.

None of these is a numerical defect. All of them are the kind of thing that
turns into one later.

---

## Priority 1 — Correctness risk

### 1.1 Bare `catch` swallows interrupts and allocation failures

**Verified.** 23 handlers in `src/` are written `catch` with no exception
filter. Julia's bare `catch` catches everything:

```julia
julia> try; throw(InterruptException()); catch; "SWALLOWED"; end
"SWALLOWED"
```

This matters because SDPX solves run for a long time — 94.5 s for the lattice
benchmark on the cluster, and hours for the sparse CSDR models. A user pressing
Ctrl-C during a factorization can have the interrupt absorbed by, for example,
`src/pipeline.jl:901`:

```julia
coefficients = try
    qr(Bkeep) \ Bdropped
catch
    return (elimination_valid=false, consistent=true)
end
```

An interrupt there does not stop the solve; it silently reports that the
equality elimination is invalid and continues. `OutOfMemoryError` and
`StackOverflowError` are absorbed the same way, and the caller cannot
distinguish "this factorization is singular" — the case the handler is for —
from "the machine ran out of memory".

**Recommendation.** Filter every handler:

```julia
catch exception
    exception isa InterruptException && rethrow()
    exception isa OutOfMemoryError && rethrow()
    return (elimination_valid=false, consistent=true)
end
```

Mechanical, testable, and worth doing across all 23 sites at once. A helper
(`_recoverable(exception)`) would keep the intent in one place.

### 1.2 Null-space reduction is unreachable

**Verified.** `src/nullspace.jl` is 415 lines with 35 passing tests. Outside its
own file it has **zero call sites**:

```bash
$ grep -rn "nullspace_reduce\|should_use_nullspace" src/ | grep -v src/nullspace.jl
(no output)
```

Together with `src/chordal.jl` (256 lines, detection only — this one is
deliberate and documented) that is roughly 670 lines shipped to users that no
`solve` call can reach.

This is not an argument for wiring it in. On this repository's own benchmarks
the formulation does not apply: CSDR models have no equality rows at all, and
the lattice benchmark constrains 394 of 6,119 variables — a reduction ratio of
0.94 against a threshold of 0.5. Measured on constructed qualifying problems it
is worth 2.2–5.1×.

**Recommendation.** Either wire it behind the existing gate and state in the
release notes that it will not fire on bootstrap models, or move it out of
`src/` into an experimental module. What should not persist is code that reads
as part of the solver but cannot be reached from it.

---

## Priority 2 — Maintainability

### 2.1 `_solve_sdp_core!` is 835 lines

**Verified** by measurement:

| function | lines | file |
| --- | ---: | --- |
| `_solve_sdp_core!` | **835** | `src/solve.jl` |
| `solve_lp!` | 417 | `src/lp_solver.jl` |
| `_solve_pipeline!` | 275 | `src/solve.jl` |
| `_pack_native_bigfloat_reduced_arrow_block!` | 244 | `src/schur.jl` |

Nearly every SDP solve passes through the 835-line function. It owns the
iteration loop, termination, stagnation detection, restarts, centering,
best-iterate tracking, checkpointing, timing, and the parameter controller.

The cost is already visible in this session's history. Two separate bugs —
parameters selected before equilibration, and the initial-point multiplier
being fitted to one instance — were both hard to see because the relevant lines
are hundreds apart inside one scope.

**Recommendation.** Extract along seams that already exist as concepts:
termination assessment, restart/centering policy, iterate bookkeeping. Each is
already a distinct idea in the comments; none of them needs the other's locals.
This is a refactor with a real risk of introducing a numerical change, so it
should be done against the §25 acceptance gates, which will fail on any
iteration-count change.

### 2.2 Test-to-source ratio

**Verified**: 8,354 test lines against 22,451 source lines, a ratio of 0.37.
For a numerical library where most defects are silent, this is low. The ratio
matters less than where the gaps are, and the gaps are visible: `src/schur.jl`
is the largest file at 3,230 lines and has no dedicated test file.

---

## Priority 3 — Test design

### 3.1 Assertions that encode the machine, not the contract

**Verified.** This is the most frequently recurring fault in the codebase, found
five separate times in one session:

| test | encoded | failed on |
| --- | --- | --- |
| `test/sparse.jl` classification | thread count | 8 threads |
| `test/pipeline.jl` worker report | physical core count | 1-core CI runner |
| `test/pipeline.jl` Schur capping | free memory | 256 GB cluster node |
| `test/kkt_regressions.jl` regularization | LAPACK build | ubuntu x86_64 |
| `bench/gates.jl` baselines | Julia's RNG stream | Julia 1.10 |

Each was written by observing a value on one machine and asserting it. Each
passed locally and failed somewhere else. All five are now fixed, but the
pattern will recur unless it is named: **a test that reads the environment must
state the environment**, either by pinning it (`thread_count=1`,
`free_memory_bytes=4GiB`) or by deriving the expectation from the same source
the code reads.

The RNG case is the sharpest. `bench/gates.jl` generated its problems with
`MersenneTwister`, and Julia's random stream is not covered by the
compatibility guarantee — it changed between 1.10 and 1.12. The gates then
reported objective movements of 16.55 and 0.65, which read as catastrophic
solver regressions and were entirely an artifact of the generator.

### 3.2 The same fragility remains in examples and tests

**Verified, currently harmless.** `examples/03_sparse_lp.jl` and
`test/nullspace_reduction.jl` still generate data with `MersenneTwister`.
Measured across versions:

```text
julia 1.12.6  entries=2  nnz/row=9.01 -> sparse_normal
julia 1.10.11 entries=2  nnz/row=8.98 -> sparse_normal
julia 1.12.6  entries=3  nnz/row=24.20 -> dense_lu
julia 1.10.11 entries=3  nnz/row=24.30 -> dense_lu
```

The decisions agree only because the margin against the threshold of 13 is
wide. An example that asserts a *routing decision* from randomly generated data
is one threshold change away from being version-dependent.

**Recommendation.** Use the `GateStream` LCG already in `bench/gates.jl`, or
assert the decision against the measured density rather than against a constant.

---

## Priority 4 — Performance

### 4.1 Dynamic dispatch in the per-iteration path

**Read-only** — I have not measured the cost. `Workspace.Qchol` is declared
`::Any`, with a comment stating the intended union:

```julia
Qchol::Any    # ::Union{Nothing,Cholesky,CholeskyPivoted} — set by factor_kkt!
```

`_solve_Q!` has three methods, and `_solve_kkt_owned!` calls it twice per
iteration (predictor and corrector). Every such call is a dynamic dispatch. The
same pattern appears in six `mixed_reduced_*::Any` fields.

At 27 iterations the absolute cost is negligible; the reason to fix it is that
the comment already documents the correct type, so the annotation is a
transcription away.

### 4.2 The Schur accumulator cap is a laptop problem, not a cluster problem

**Verified on both.** `_schur_parallel_bins` caps task-local `m×m` accumulators
at 15% of free memory. On a memory-constrained laptop this forces a single bin
at `m = 6119`, making Schur assembly serial — 54% of runtime at 4 threads, and
it gets *worse* with more threads:

| threads | total | Schur | KKT factorization |
| ---: | ---: | ---: | ---: |
| 1 | 22.52 s | 7.00 s | 14.03 s |
| 4 | 13.11 s | 7.14 s | 4.46 s |
| 8 | 14.69 s | 7.69 s | 4.81 s |

On the cluster the cap does **not** bind:

```text
free_memory_gb=27.6   bytes_per_bin_mb=285.7
threads=1 selected_bins=1 capped=false
threads=2 selected_bins=2 capped=false
threads=4 selected_bins=4 capped=false
threads=8 selected_bins=8 capped=false
```

This is worth stating plainly because it invalidates an optimization I was
about to recommend. Halving accumulator storage to the lower triangle (285.7 MB
→ 142.9 MB) would double the affordable bins — real on a laptop, worth nothing
on the machine where the work actually runs.

**The cluster's phase breakdown is still unknown.** A profiling job
(`193929.node220`) was submitted for exactly this and its results were not
collected. No performance work on the Schur/KKT split should start before that
data exists.

---

## Suspicions that did not hold

Reported so the review's own reliability can be judged.

- **Exported API undocumented.** A grep heuristic suggested `classify_problem`,
  `build_execution_plan`, and `analyze_structure` lacked docstrings. Checking
  with `@doc` showed all eleven exported entry points are documented.
- **Tile-owned Schur assembly would be faster.** A microbenchmark showed
  2.68–3.27×. Against the real code path it ran at 0.34–0.40×, because the
  baseline in the microbenchmark had been pinned to one BLAS thread. The idea
  was discarded.
- **Precision escalation would speed up hard solves.** It cuts wide-precision
  iterations from 29 to 10, and delivers 1.00× wall clock. Not worth building.

---

## Still open from the external review

`docs/claude-code-review-findings-2026-07-26.md` raised twelve items. Priority 1
(null-space memory safety, LP KKT regularization consistency, calibration
validation) and one Priority 3 item are fixed and pushed. These remain:

| item | status |
| --- | --- |
| P2.4 diagnostics report the plan, not the executed algorithms | open |
| P2.5 `minimum_psd_eigenvalue` contains a shift, not an eigenvalue | open |
| P2.6 parameters reported may predate equilibration | open |
| P2.7 BigFloat LP bypasses precision-consistency handling | open |
| P2.8 status/provenance semantics inconsistent | open |
| P3.10 sparse solve allocates two vectors per solve | open |
| P3.11 `BLAS.set_num_threads` is process-global and concurrency-unsafe | open |
| P3.12 remaining memory estimators can overflow | open |

P2.4 and P2.5 are the two I would take first: both are cases where the solver
reports something that is not true, and both are cheap to fix relative to the
confusion they cause.

---

## Recommended order

1. Filter the 23 bare `catch` handlers (§1.1) — mechanical, and it is currently
   possible to lose an interrupt during a multi-hour cluster solve.
2. Collect the cluster profile from job `193929.node220` (§4.2) before any
   performance work.
3. Fix the two "reports something untrue" items, P2.4 and P2.5.
4. Decide null-space reduction's status: wired, or moved out of `src/` (§1.2).
5. Split `_solve_sdp_core!` against the acceptance gates (§2.1).

Items 1–4 are low-risk. Item 5 is not, and should be done only with the gates
in place to catch an accidental change in iteration behaviour.
