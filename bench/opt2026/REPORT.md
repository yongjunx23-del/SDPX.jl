# SDPX optimization pass — 2026-07-25

Profile-driven optimization of the sparse Schur/KKT path, measured on the two
benchmarks named in the task: the dense lattice-bootstrap `Task_Low08` model at
`Float64`, and the CSDR sparse PSD dual at `J/K/N_a/N_mu = 80/4/40/100`.

All numbers come from running code. Every change below was kept only after it
measured better; two changes that looked good on paper were measured, found
worse, and reverted (recorded in "Rejected changes" so they are not retried).

> **Historical optimization log.** The measurements and test counts in this
> file describe dated intermediate implementations and are preserved for
> reproducibility. They are not the current release baseline unless repeated
> in the [extended-precision report](../extended_precision_blas/REPORT.md),
> [CSDR results](../csdr_psd_dual/RESULTS.md),
> [automatic-pipeline report](../../docs/automatic-optimization-pipeline.md),
> or [threading results](../threading/RESULTS.md).

---

## 1. What the profile said

Sampling profile of a full `Task_Low08` solve (8 Julia threads, 4 BLAS threads),
before any change in this pass:

| Site | Self time |
|---|---:|
| `threaded_schur_build!` (sparse block loop) | 27.1% |
| `sparse_schur_block!` pair loop | 25.2% |
| `range.jl` iterate | 7.6% |
| `essentials.jl` getindex | 11.1% |
| `_dot_dense_sparse` | 7.5% |

Roughly **80% of solve time was sparse Schur assembly**, and the dense `m×m`
KKT Cholesky did not appear at all. The iterate/getindex overhead is the tell:
the inner kernel was spending most of its instructions on loop bookkeeping
rather than arithmetic.

The cause was visible in the data. `Task_Low08` has 32 PSD blocks with `k` from
23 to 74 and 1815–5290 active variables per block, but its coefficient matrices
hold only **2.4–6.4 stored entries each**. The inner product `⟨W, A_j⟩` read
those from `SparseMatrixCSC` with

```julia
for c in 1:size(A, 2), idx in nzrange(A, c)
```

so reaching 4 stored entries in a `52×52` block walked 52 columns and 104
`colptr` reads, and did so once per `(i,j)` pair — 2.3e8 times per iteration,
each time dereferencing a different heap object.

---

## 2. Changes kept

### 2.1 Flat coordinate storage for sparse coefficients (`SparseBlockCOO`)

`SparseCons` gained a derived `coo` field: per block, the coefficients in
`schur_order` packed into flat `ptr`/`lin`/`row`/`col`/`val` arrays. `lin` is a
precomputed column-major linear index into the `k×k` workspace, so the dot
product is one flat load per stored entry with no empty-column scan and no
pointer chasing.

`Asp` remains the source of truth for validation, MOI, equilibration, and every
non-hot path, so the four-argument `SparseCons` constructor is unchanged at all
existing call sites and `coo` is built automatically.

**Effect: `Task_Low08` 8.319 → 2.881 s/iter (2.89×).** Kernels verified bitwise
identical to the CSC versions they replaced.

### 2.2 Shared empty coefficient matrix at ingest

Sparse ingest allocated a distinct `SparseMatrixCSC` (three arrays) for every
structurally empty coefficient slot. Models with many blocks are usually models
where each block touches few variables, so most of the `L × m` grid is empty:
the 4100-block CSDR case has ~5 active variables per block out of 4484, i.e.
**18.4M coefficient slots of which ~20K are nonzero**. All empty slots now share
one canonical matrix per block. Safe because coefficient matrices are read-only
after ingest (equilibration and re-rounding both build new arrays).

This removed ~55M allocations from ingest on that model and cut peak memory
correspondingly.

### 2.3 Parallel Schur scatter (the scaling fix)

This was the largest single finding. Splitting the Schur build into its parallel
and serial parts on `Task_Low08`:

```
full threaded_schur_build!  = 0.9403 s
reduce_sparse_schur! alone  = 0.8877 s   (94.4%)
block compute (parallel)    = 0.0526 s
```

The per-block compute was already parallel and nearly free; the scatter of
2.26e8 packed entries into the dense Schur matrix was **entirely serial**. It
therefore both dominated runtime and capped multicore scaling regardless of how
many threads the assembly used — which is exactly what the "before" cluster
curve shows (Schur assembly improves only 1.87× from 1 to 16 threads while the
BLAS-backed KKT factorization improves 4.2×).

The scatter now partitions the **output columns** of `S` across tasks. Each task
owns a disjoint contiguous column range, so there are no atomics, no locks, and
no per-thread `m×m` buffers (dense per-thread copies would be 280 MB each here,
which does not scale to 128 threads). `schur_order` was changed from
nnz-descending to ascending variable id so that the positions feeding an owned
column range are contiguous and locatable by binary search, and a direct
`_packed_pair_base` index addresses packed entries without walking the
enumeration. Each task therefore reads only the entries it needs, and total work
is unchanged from the serial version.

The parallel scatter is **bitwise identical** to the serial one — asserted in
the test suite across randomized structures, and verified at 1, 4, and 8 threads.

### 2.4 Best-iterate retention (robustness)

An interior-point run can reach a good point and then wander away. On
`Task_Low08` at `ϵ=1e-8` the primal residual reaches `1.4e-12` around iteration
55 while the dual residual diverges to ~1.8; the restart rule then rescales the
collapsed side and the good iterate is destroyed, so the run reported
`NumericalBreakdown` with an objective that must not be trusted.

`solve!` now retains the best point visited, ranked by the largest of the three
*scaled* quantities the termination test uses (so "best" means "closest to
satisfying the stopping criteria", which a wildly infeasible point cannot fake).
Only non-optimal exits fall back to it, and only when it is meaningfully better
than the final point — a converged run already ends on its best iterate, so this
can never change a successful solve.

---

## 3. Rejected changes (measured, worse, reverted)

Recording these so they are not retried.

| Idea | Why it looked good | Measured result |
|---|---|---|
| Gather + BLAS `gemm` for the two-sided product `Y·A_i·X⁻¹` | Accumulating `nnz` rank-one terms re-reads the whole `k×k` output once per term; gathering makes it one `(k×nnz)(nnz×k)` product with ~11× less traffic | **2.25× slower** (Schur 1.299 → 1.720 s/iter). Inner dimension is only 2.4–6.4, so `gemm` overhead dominates across ~115k calls/iteration, and `dest` is ≤44 KB so it stays in L2 and the "extra" passes were nearly free. Reverted. |
| Group variables by sparsity pattern and replace the pair loop with per-group `gemm`s | Turns `Σ|I_l|²·nnz` scalar work into BLAS | Measured first: only **40.5%** of variable slots share a pattern and the largest group is 4–16, so the group-pair cost estimate was **4.9e9 vs 1.9e9** for the current form. Not implemented. |

Also measured and rejected on the data rather than implemented: converting the
sparse pair loop to the dense symmetric-square `syrk` form used by the dense
path. Work would rise from `Σ na²/2·nnz = 9.3e8` to `Σ na²/2·k² = 6.3e11`, i.e.
676× more arithmetic — the coefficients are far too sparse for that form.

---

### 2.5 Stall / precision-exhaustion stop

`Stalled` existed in the status enum but was never assigned — dead code. Two
conditions now set it:

1. `stall_iterations` (default 15) consecutive iterations with no meaningful
   improvement in the scaled merit.
2. A step collapse *after the solve has already made real progress* (best merit
   improved ≥1000× from the first iterate). A restart repairs bad **initial
   scaling**; once the solve has converged by orders of magnitude, a collapsed
   step means precision exhaustion, and multiplying the collapsed side by 1e5
   destroys the good iterate instead of rescuing it.

Condition 2 matters because the plain counter never fires in that regime: a
restart does not increment `iter`, so the restart budget is spent before the
stall counter reaches its threshold. This was found by measuring, not by
reading the code.

### 2.6 Phase-aware BLAS thread control

The two expensive phases want opposite BLAS settings. Block-parallel regions
issue many small BLAS calls from concurrent Julia tasks, so
`julia_threads × blas_threads` threads become runnable at once; the dense KKT
Cholesky is one large call that wants every core.

Leaving BLAS at full width during the parallel regions oversubscribes badly. On
a 128-core compute node, `16 Julia × 16 BLAS = 256` threads made the lattice
benchmark **hang** — the 16-thread point had to be killed after >10 minutes for
work that takes ~1.5 s/iteration, and this reproduced across two independent
job submissions. The same configuration ran fine on the smaller login node,
where the product stayed near the core count, which is exactly why it was
invisible in local testing.

`newton_step!` now pins BLAS to 1 thread inside every block-parallel region and
restores the caller's setting for the KKT factorization.

---

## 4. Correctness

- Test suite: **259/259 pass** (248 before; the additional assertions are the new
  parallel-vs-serial scatter equivalence test and the updated `schur_order`
  invariant).
- COO kernels: bitwise identical to the CSC versions (`0.000e+00` relative
  difference), including under a permuted `schur_order`.
- Full sparse Schur matrix versus the canonical `tr(Y A_i X⁻¹ A_j)` formula:
  `2.158e-16` relative error.
- Parallel scatter versus serial scatter: bitwise identical at 1, 4, 8 threads.
- `Task_Low08` at `1e-6`: objective `0.653291393898`, identical to 12 digits to
  the archived pre-optimization result, same 27 iterations, equality residual
  `2.060e-12`, minimum PSD eigenvalue `-2.616e-15`.
- Objective is stable across thread counts (1/2/4/8) to ~14 digits, with
  identical `p_res`/`d_res`; the last-digit variation is BLAS reduction order in
  the Cholesky, not the scatter, which is bitwise deterministic.

---

## 5. Benchmark results

### 5.1 `Task_Low08` — same machine, MOSEK vs SDPX (Float64)

Apple Silicon laptop, 8 Julia threads / 4 BLAS threads for SDPX, 8 threads for
MOSEK.

| Solver | Path | Tolerance | Status | Time | Objective |
|---|---|---:|---|---:|---|
| MOSEK 11.2 | CVXPY | 1e-8 | solved | **31.08 s** wall | — |
| SDPX (this pass) | native API + equality presolve | 1e-6 | `Optimal`, 27 iters | **33.01 s** solve / 33.53 s total | 0.653291393898 |
| SDPX (this pass) | JuMP + MOI wrapper | 1e-6 | `OPTIMAL` | 1.90 s build + 58.87 s solve = **60.78 s** | 0.653291393898 |
| SDPX (archived, pre-pass) | native | 1e-6 | `Optimal`, 27 iters | 180.24 s | 0.653291393898 |

**SDPX is 5.46× faster than before this pass** (180.24 → 33.01 s) at identical
objective and iteration count, which takes it from ~6× slower than MOSEK to
roughly **parity on solve time**.

Two honest caveats:

- The tolerances differ. SDPX certifies 1e-6; MOSEK certifies 1e-8. At 1e-8
  SDPX still cannot certify (it now reports `Stalled`/`MaxRestartsExceeded`
  with the retained best iterate, objective 0.65329124339, equality residual
  1.3e-14, min eigenvalue -2.5e-17 — a usable point, but not a certificate).
  Float64 precision is the binding constraint there, not speed.
- Through equivalent modeling layers (JuMP vs CVXPY) MOSEK is ~2× faster
  end-to-end, because the MOI path does not benefit from the equality presolve
  (482 → 394 independent equalities) that the native driver applies. Closing
  that is a concrete follow-up.

### 5.2 Multicore scaling — `Task_Low08` on the cluster

One node of the UCAS HIAS cluster: 2× AMD EPYC 7742 (128 cores), 251 GB usable
RAM, CentOS 7.8, PBS `normal` queue, `nodes=1:ppn=128`. Julia 1.12.6 via the
shared toolchain helper. Julia and BLAS thread counts swept together. Fixed
8-iteration solves, minimum of 3 timed runs after a warmup.

Before this pass (COO + shared-empty ingest, serial scatter):

| Threads | Time | Speedup | Schur | KKT |
|---:|---:|---:|---:|---:|
| 1 | 47.59 s | 1.00× | 28.93 | 16.59 |
| 2 | 29.20 s | 1.63× | 16.92 | 9.83 |
| 4 | 25.37 s | 1.88× | 14.38 | 6.86 |
| 8 | 26.28 s | 1.81× | 17.83 | 4.86 |
| 16 | 22.16 s | 2.15× | 15.45 | 3.93 |
| 32+ | job died | — | — | — |

After (parallel scatter + parallel partial reduction):

| Threads | Time | Speedup | Efficiency | Schur | KKT | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.65 s | 1.00× | 100% | 17.94 | 16.65 | 1.50 GB |
| 2 | 25.48 s | 1.44× | 72% | 13.09 | 9.94 | 3.33 GB |
| 4 | 19.38 s | 1.89× | 47% | 9.32 | 6.76 | 4.34 GB |
| 8 | 14.54 s | 2.52× | 32% | 6.63 | 4.64 | 5.94 GB |

The shape of the improvement is what matters: **Schur assembly went from
essentially non-scaling (28.93 → 15.45, i.e. 1.87× across 16 threads) to
scaling in step with the rest** (17.94 → 6.63, 2.70× across 8), and the
1-thread time itself dropped 23% because ascending `schur_order` also made the
serial scatter's writes sequential within a column.

Two caveats on this table, both real:

- Points at 1/2/4 threads use a *different* assembly mode than 8+: the workspace
  heuristic switches between dense per-bin `m×m` accumulators and packed
  per-block buffers based on `nbins × m²`, so the curve mixes two algorithms.
  It is what the solver actually does, but it is not a clean strong-scaling
  curve for one kernel.
- Memory grows with thread count (1.5 → 5.9 GB) because of those per-bin
  accumulators, and `allocated_bytes` (1.16 → 2.97 GB) is dominated by
  per-`solve!` workspace construction rather than per-iteration churn.

**Oversubscription finding.** 16+ threads reproducibly hung on the compute node
until §2.6 landed — see that section. The sweep is being re-run with the fix.

### 5.3 Cluster reproduction notes

The CSDR sparse problem is 2.35 GB serialized but is fully determined by four
parameters (`J/K/N_a/N_mu = 80/4/40/100`). It is therefore **regenerated on the
cluster** from ~300 KB of generator source rather than copied, and
`prepare_problem.jl` prints a SHA-256 so the regenerated instance can be proven
identical to the local one (`ed65290d3517ef24b334…`). Copying the artifact would
have taken ~25 minutes and given a weaker identity guarantee.

Scaling points are submitted as **one PBS job per thread count** rather than a
single looping job. Three successive looping sweeps each died partway through
(at 32, then 16, then 8 threads) with the shell exiting mid-loop and no
diagnostic, on an otherwise-idle exclusive node with 270 GB free — so the cause
was something cumulative in the loop rather than the thread count itself.
Independent jobs sidestep it and mean one bad configuration cannot destroy the
campaign.

---

## 5.4 CSDR sparse benchmark (`J/K/N_a/N_mu = 80/4/40/100`)

4100 `2×2` PSD blocks, 4484 variables, `n=0`, exact block-arrow structure with
**384 shared variables and one local per block** — so every block has 385 active
variables and 74,305 Schur pairs.

Local (16 GB laptop, 1 thread, Float64x4), 3-iteration phase timings:

| Stage | s/iter | Schur | KKT | Peak RSS |
|---|---:|---:|---:|---:|
| baseline | 122.22 | 89.6 | 26.4 | ~14 GB (swapping) |
| + fused arrow assembly | 73.47 | 28.2 | 43.9 | 4.92 GB |
| + rank-one arrow reduction | **43.99** | 28.0 | **14.6** | **4.81 GB** |

**2.78× faster, ~3× less memory.**

The decisive finding was memory, not arithmetic. Micro-benchmarks of the three
candidate hot loops accounted for only ~28 s of the measured ~90 s Schur time
(pair loop 21.4, scatter 6.0, transform 0.5 s/iter). The gap was the packed pair
buffer: `4100 × 74,305 × 32 B = 9.08 GB` of `Float64x4` on a 16 GB machine.
`fused_arrow_schur_block!` computes and scatters in one pass so the buffer is
never allocated, which is both the memory fix and most of the speedup.

Independent confirmation: the same pre-fusion code on a 251 GB cluster node ran
at **20.5 s/iter** versus 122 s/iter locally — a 6× difference from RAM alone.

Two "optimizations" were measured and reverted along the way, both for the same
reason — a matrix product whose inner dimension is 1–6 is all dispatch overhead:

- Arrow reduction via `kmul!` instead of loops: **26.4 → 43.9 s/iter** (worse).
  Replaced with an explicit rank-one update with the invariant hoisted and the
  stride-1 index innermost: **14.6 s/iter**, better than the original.
- Hoisting the `2×2` pair loop with `@simd` measured 1.37× faster *in isolation*
  but the full solve got slower, because at that point the run was
  memory-bound — a reminder that isolated kernel gains do not transfer when the
  bottleneck is elsewhere.

Reference point: Clarabel solved this instance `OPTIMAL` in 1114.9 s / 70
iterations (15.9 s/iter, single-threaded, same laptop, SHA-verified identical
problem). SDPX does **not** yet converge on it — it stops at
`MaxRestartsExceeded` — so the per-iteration comparison is not yet a solve-time
comparison. Convergence, not speed, is the open item.

---

## 6. Remaining bottlenecks

1. **Dense KKT Cholesky** — `m = 6119`, `S` is 84% structurally dense, so
   `m³/3 ≈ 7.6e10` flops with no sparse structure to exploit. At 8 threads it is
   32% of the solve and already runs near BLAS peak. The only real levers are
   more cores or a smaller `m`.
2. **Schur assembly two-sided product** — `Σ_i nnz_i·k² ≈ 1.3e9`, now 2.25× the
   pair-dot work. Gather+`gemm` and pattern-group `gemm` were both measured and
   are worse (§3). No further constant-factor idea has survived measurement.
3. **Float64 precision ceiling at 1e-8** — the binding limit for tight
   tolerances on this model. `Float64x2` is the principled answer; automatic
   escalation on `Stalled` would be a natural UX follow-up.
4. **MOI/JuMP path ~1.8× slower than native** — cause not yet identified. The
   first diagnosis here ("the wrapper does not apply the equality presolve") was
   wrong and is retracted: SDPX's presolve removes exactly 88 dependent rows
   (482 → 394) on this model through the MOI path too, matching the driver's own
   QR. The per-iteration overhead is real but lies elsewhere.
5. **Assembly-mode heuristic is thread-count dependent**, which makes scaling
   behavior discontinuous around 8 threads. Worth making explicit rather than
   emergent.

---

## 7. Clarabel-inspired round: why the sparse model never converged

Focus: the CSDR sparse 2×2-block model (`J/K/N_a/N_mu = 80/4/40/100`,
`L = 4100` blocks, `m = 4484`, 384 shared + 1 local variable per block). It was
already fast per iteration but **never converged** — every configuration ended
`MaxRestartsExceeded` or `Stalled`, far from Clarabel's answer.

### 7.1 The reference is real

Clarabel on the same problem hash (`ed65290d…`, Float64x4, 1 thread) genuinely
converged, so any SDPX objective above ~13.5809 is simply unconverged:

| | Clarabel |
|---|---|
| status | OPTIMAL / SOLVED, **70 iterations**, 1114.86 s |
| pObj / dObj | 13.580848604207631 / 13.580849742204832 |
| gap_rel | 8.38e-08 |
| primal / dual residual | 7.26e-26 / 2.80e-24 |
| min block slack | 6.44e-09 |

"Min slack" is the smallest eigenvalue of `Σᵢ xᵢAᵢ[l] − C_l` over all blocks —
the direct PSD feasibility measure. A tiny positive value is the *expected*
result at an SDP optimum, where the constraint is active.

### 7.2 Root cause: three compounding defects in termination, not in the algorithm

The solve was never running out of iterations (300–400 allowed, 13–39 used) and
never diverging. It was being *stopped*, by three independent defects:

**(a) The stall detector used a fixed per-iteration bar.** It required a
`stall_tolerance = 1e-3` improvement on *every* iteration and stopped after 15
consecutive misses. This model converges at ~0.05% per iteration near the end —
real progress needing ~70 iterations in total, exactly what Clarabel takes — so
every step missed the bar.

**(b) A collapsed step on an already-feasible side terminated everything.** The
trigger was `tX < min_step || tY < min_step`, on *either* side. Here the primal
reaches `p_res ≈ 1e-47` by iteration 2 and stays there, so `dX ≈ 0` and `tX`
duly collapses — which is the expected, harmless outcome for a side with
nowhere left to go. Primal and dual step lengths are independent in this method,
so the dual still had work to do. This is why disabling stall detection entirely
changed nothing: the run stopped at iteration 27 either way.

**(c) `step_rule` defaulted to `:backtrack`.** It accepts the first `γᵏ` that is
positive definite, so its effective fraction-to-boundary factor lands anywhere
in `[γ, 1]` and can put the iterate essentially *on* the cone boundary, after
which every later direction exits immediately. `:auto` selects the exact rule —
closed-form `det(X + t·dX) = 0` for 2×2 blocks with a consistent margin — exactly
when all blocks are ≤2×2, i.e. this model, but was not the default.

Isolating them on the cluster (Float64x4, 32 threads):

| configuration | iters | pObj | gap_rel |
|---|---|---|---|
| baseline (`:backtrack`, stall on) | 27 | 13.7452 | 9.02e-04 |
| stall detection disabled | 27 | 13.7452 | 9.02e-04 |
| `:fraction_to_boundary`, stall on | 44 | 13.7169 | 6.22e-05 |
| **`:fraction_to_boundary`, stall off** | **109** | **13.6516** | **8.45e-06** |

Two orders of magnitude on the gap, and the objective moves from 13.745 toward
Clarabel's 13.5808. The β/Ω sweeps that preceded this were measuring *where
premature termination happened to land*, which is why they looked chaotic and
hypersensitive; they are not the fix and were not kept as one.

### 7.3 What was verified clean

- **Arrow KKT elimination.** This model exercises the arrow path almost
  exclusively (`n = 0` + arrow structure; `Task_Low08` takes the standard path),
  so a defect there would hide from every other benchmark. Checked against the
  dense path at 3/20/100/400/1000 blocks: Schur and `dx` agree to **1e-61 …
  1e-64** at 212-bit precision, plus an independent `S·dx − rhs` residual check.
  Now covered in the suite rather than only at 3 blocks.
- **Precision.** The iterative-refinement residual runs ~1e-48 against
  `eps(Float64x4) = 2.4e-63` — 15 orders of headroom. Not precision-limited.
- **Direction symmetry.** `dX` is symmetric by construction; `dY` is explicitly
  symmetrized. `trial_isposdef!` applies Sylvester's criterion correctly.

### 7.4 Implemented

1. **Automatic stagnation detector** (`src/stagnation.jl`, new). Replaces the
   fixed `stall_tolerance` bar. All four convergence metrics — primal residual,
   dual residual, relative duality gap, complementarity — are divided by the
   tolerance requested for each, so the combined `merit` is dimensionless and
   `merit <= 1` means converged. Progress is measured in nats of `log(merit)`
   per iteration over a rolling window, and the run stops when that rate cannot
   reach the target within the remaining budget. Scaling is automatic in both
   senses: **with the tolerance**, because loosening an `ϵ` genuinely shortens
   the run rather than only changing the final check; and **with the
   precision**, because a metric within `PRECISION_FLOOR_ULPS·eps(T)` cannot
   improve at that width and is reported as `:precision_floor` (actionable:
   widen `T`) instead of an unexplained stall. The verdict, measured rate, and
   projected iteration count are recorded in `diagnostics.termination`.
   One calibration lesson worth recording: the budget-projection arm of this
   rule was first written to fire whenever the projected iteration count merely
   exceeded the remaining budget. That is a *marginal* call made from a noisy
   local rate, and it misfired immediately — on the CSDR model it ended the
   solve at iteration 34 (projected ~373 against 366 remaining) at gap 7.1e-3,
   worse than the very baseline it was meant to improve on. The arm now fires
   only at `PROJECTION_SLACK = 10×` the budget, i.e. when the rate is hopeless
   rather than merely tight, and `iter_max`/`max_time` enforce the budget —
   which is their job, not the detector's.
   The detector's own diagnostics make its decisions checkable, and they
   validated it in practice: on an `Ω = 3·max‖C_l‖∞` configuration it reports
   `rate = 0.002736` nats/iteration and `projected = 4613` iterations, so the
   stop is a correct reading of a genuinely hopeless rate rather than a
   premature give-up. Local and cluster runs agree exactly (35 iterations, gap
   7.081e-03), which also makes the result reproducible across machines. The
   fault that configuration exposes is in the Ω default, not the detector.
2. **Asymmetric step collapse is no longer fatal.** A stuck side that is already
   within its feasibility tolerance is frozen (`t = 0`) and the solve continues
   on the other side; only a stuck side that still has work to do — or both
   sides at once — counts as a real collapse.
3. **`step_rule` defaults to `:auto`**, selecting the exact 2×2
   fraction-to-boundary rule when every block is ≤2×2 and backtracking
   otherwise. Also much cheaper here: backtracking costs ~141 Cholesky sweeps
   over 4100 blocks to walk `t` down to 1e-10.
4. **Restart repairs only the infeasible side.** The old rule keyed off which
   *step* collapsed, a different question from which side is badly *scaled*; it
   walked `p_res` from 1e-48 up through 1e+23 over five restarts.
5. **Adaptive iterative refinement** (`refine_direction!`, `refine_policy`).
   Refines while the KKT residual exceeds `64·eps(T)` *and* each pass still
   halves it, capped at `refine_max_steps`; skips the correction solve entirely
   when the direction is already accurate, and **rolls back** a pass that
   increases the residual (refinement diverges when `κ(S)·eps(T)` is not small).
   Bit-identical results, 245.5 → 221.6 s sparse and 65.0 → 56.3 s dense.
6. **Recentering on step collapse** (`max_centering`). When a step collapses
   while residuals *and* the KKT direction are healthy, the iterate has hit the
   cone boundary far from the optimum — too little centering. Raising β is cheap
   and non-destructive, unlike a 1e5 rescale. Ordered *after* the
   precision-exhaustion test (reversed, it cost `Task_Low08` 64.1 → 75.0 s for
   an identical answer).
7. **Data-driven Ω** — `1·max‖C_l‖∞`, floored at the old 10. The old fixed
   `Ω = 10` against `max‖C_l‖∞ = 116.6` gave `pObj = 1.4e+12`.
8. **Adaptive β/γ bounds respect the profile's own β** — the generic floor
   `β ≥ 0.02` silently clamped away a profile that asked for 0.01. The
   controller itself remains **off** by default; see §7.5.

### 7.5 Measured negative results (deliberately not defaults)

- **Constants fitted before the termination bugs were fixed.** Two defaults were
  calibrated against runs that were stopping prematurely, and both reversed once
  termination was correct. Re-measured on the cluster (Float64x4, 32 threads,
  β=0.05, γ=0.85), the full grid is:

  | Ω | β/γ control | iters | pObj | gap_rel |
  |---|---|---|---|---|
  | **1·max‖C_l‖∞** | **fixed** | **47** | **13.6455** | **3.08e-04** |
  | 1·max‖C_l‖∞ | adaptive | 33 | 13.7472 | 6.08e-04 |
  | 3·max‖C_l‖∞ | fixed | 28 | 13.7090 | 2.06e-03 |
  | 3·max‖C_l‖∞ | adaptive | 32 | 13.6602 | 9.68e-04 |

  So `OMEGA_DATA_MULTIPLIER` is 1, not 3, and the adaptive β/γ controller is
  **not** enabled by any profile — the earlier "large win on the arrow profile"
  was an artefact of where premature termination happened to land. The general
  lesson: a constant fitted while a termination bug is live measures the bug.

- **BigFloat for this model.** BigFloat-256 costs **787.59 s/iteration** against
  Float64x4's **9.21 s** — 85× — and hit the time limit after 7 iterations.
  Since the refinement residual is ~1e-48 against `eps(Float64x4) = 2.4e-63`,
  precision was never the binding constraint, so the extra width buys nothing
  here. `Float64x4` is the right default; BigFloat-512 would only be slower.

- **Per-block initial scaling** (`omega_scaling = :per_block`), setting
  `X_l = ‖C_l‖∞·I` so each block starts commensurate with its own data. The
  reasoning is sound — the initial dual residual *is* `‖C_l − Ω_d·I‖` — but it is
  the wrong criterion: **pObj = 7.06e11, Stalled at 13 iterations**, versus
  `pObj = 14.78` for the scalar start. The initial point must *dominate* the
  data, not match it. Kept as an explicit opt-in.
- **Sparse equilibration** on this model: `Stalled` at 19 iterations
  (`pObj = 1.9e9`) versus 56 iterations unequilibrated.

### 7.6 Regression checks

**Dense lattice `Task_Low08`** (Float64, 4 threads). Its blocks are 23–74, so
`:auto` correctly keeps backtracking and the step-rule change is a no-op there:

| configuration | time | iters | pObj | gap |
|---|---|---|---|---|
| `step_rule=:backtrack` | 59.50 s | 40 | 0.653291222792 | 6.41e-07 |
| `step_rule=:auto` (new default) | 65.28 s | 40 | 0.653291222792 | 6.41e-07 |
| adaptive refinement off / on | 65.04 → **56.30 s** | 36 / 37 | agrees to 10 digits | — |

The new detector reports `reason=:too_slow` here, which is the honest verdict:
Float64 cannot reach a 1e-8 target on this model, and that is now stated rather
than surfacing as an unexplained stall.

**Per-iteration cost.** The exact step rule is also cheaper than backtracking on
the sparse model — 6.35 s/iteration versus 7.43 — because it replaces ~141
Cholesky sweeps over 4100 blocks with one closed-form root per block.

### 7.7 Next highest-priority opportunities

1. **Close the last 0.07 on the sparse objective.** At 109 iterations the run
   reaches `pObj = 13.6516` against Clarabel's 13.5808 and is still descending
   when it stops. The remaining question is whether it converges given budget,
   or needs a centering/corrector change.
2. **Per-iteration cost now dominates the comparison.** With ~110–150 iterations
   needed and ~6.4 s/iteration at 32 threads, the sparse solve lands near
   700–950 s against Clarabel's 1114.86 s. Schur assembly is still the largest
   phase.
3. **MOI/JuMP overhead — measured, and much smaller than claimed.** Timing
   construction and solve separately (Float64, 1 thread):

   | model | MOI solve vs native | `copy_to` vs native `ingest` |
   |---|---|---|
   | 2 blocks, side 8, m=30 | **1.00×** | 1.9× (0.2 ms vs 0.1 ms) |
   | 3 blocks, side 12, m=60 | **1.02×** | 2.5× (1.1 ms vs 0.5 ms) |

   The wrapper adds essentially nothing to the solve, and the construction gap
   is sub-millisecond against a 15 ms solve. The earlier "~1.8× slower via
   JuMP" figure does not reproduce at the MOI layer and is retracted, as is the
   "missing equality presolve" diagnosis that preceded it (presolve removes the
   same 88 dependent rows through both paths). Plan §7.3's code-level targets
   are already in place: one shared canonical empty CSC rather than an object
   per variable-constraint pair, sparse triplet construction, and
   `supports_incremental_interface = false`.
4. **BigFloat throughput.** At this historical checkpoint, fixed-width
   threading was repaired, while BigFloat remained serial and substantially
   slower than `Float64x4`. With precision ruled out as the binding constraint
   on this model, `Float64x4` was the appropriate arithmetic choice.

Historical test-suite checkpoint: 270 → **386** passing.
