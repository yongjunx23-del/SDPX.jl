# SDPX Research Worklog

This file records performance and numerical experiments in chronological order.
An optimization is retained only when its benchmark is reproducible and its
correctness gates pass. Unless stated otherwise, timings include warm-up only
when the benchmark explicitly reports it.

## 2026-08-01 — Baseline and next-step selection

### Repository state

- Branch: `agent/sdpx-v0.3-certificates`
- Local commit: `c93d485c07de00027ddfc2ac429ec2cebabd9539`
- Source tree: `c956e68b5ad1ce9134c03257ec9f68243699f313`
- GitHub PR head with the same tree:
  `b3b726dad6309e8647f01181d10bf08e78becd88`
- Package version: `0.3.0`

### Cluster regression baseline

The immutable candidate at
`<cluster-root>/SDPX.jl/releases/b3b726dad6309e8647f01181d10bf08e78becd88`
passed job `194948.node220` on a 32-core reservation. The validation commands
used eight Julia and eight BLAS threads for Task_Low08.

- Package tests: 2,594 / 2,594 passed.
- Task_Low08: `Optimal`, 29 iterations.
- Primal objective: `0.6532914234005835`.
- Dual objective: `0.653290438054924`.
- Relative gap: `9.853456595099175e-7`.
- Primal residual: `5.422811089061952e-10`.
- Dual residual: `2.4870661086140444e-11`.
- Maximum equality residual: `5.422811089061952e-10`.
- Minimum primal / dual PSD eigenvalues:
  `-6.563299502130475e-11` / `5.164806013981275e-15`.
- Original-coordinate optimality certificate: valid; primal and dual PSD
  certificate checks passed.
- Solve / end-to-end time: `48.695` / `50.387` seconds.
- Task_Low08 peak RSS: 4,341,788 KiB (about 4.14 GiB).
- Automatic profile selected `beta=0.075`, `gamma=0.8`,
  `Omega_p=100`, `Omega_d=0.001`, and the SDPB predictor.

Dominant recorded phases were:

| Phase | Seconds | Share of 47.609 s solver phase total |
|---|---:|---:|
| KKT factorization | 16.001 | 33.6% |
| Schur assembly | 12.871 | 27.0% |
| Schur Cholesky subphase | 11.486 | 24.1% |
| Corrector | 3.690 | 7.8% |
| Predictor | 2.629 | 5.5% |
| Workspace setup | 1.947 | 4.1% |

The phase totals overlap where a parent phase contains named subphases; they
must not be added as disjoint costs.

The sparse Schur scheduler regression retained numerical agreement and showed
the following warmed medium-case scaling:

| Arithmetic | 1 thread | 4 threads | 8 threads | 8-thread speedup | Relative error |
|---|---:|---:|---:|---:|---:|
| Float64 | 1.095 ms | 0.778 ms | 0.456 ms | 2.40x | `1.39e-15` |
| Float64x4 | 10.055 ms | 5.701 ms | 2.955 ms | 3.40x | `1.09e-64` |

### Decision

The next highest-value experiment is a source-level audit of dense Schur
assembly and dense KKT factorization. Together they dominate Task_Low08, while
the current result already passes all numerical gates. Candidate changes must
therefore preserve the exact lower-triangular storage contract, avoid nested
BLAS/Julia oversubscription, and keep the Float64 path numerically stable.

## 2026-08-01 — Task_Low08 bottleneck audit and adaptive-trajectory diagnosis

### Kernel audit

The current Task_Low08 implementation already contains the low-risk dense
Schur/KKT optimizations that would normally be attempted first:

- sparse coefficients are flattened and reduced by pair-balanced column work;
- only the lower Schur triangle is assembled and copied;
- dense Cholesky is selected because the measured Schur density is 84.26%;
- an earlier CHOLMOD experiment produced full fill and took 1.83x the dense
  conversion-plus-factorization time;
- short-dot unrolling, tiny-GEMM gathering, packed dense SYRK, MKL, and BLIS
  were previously benchmarked and rejected on the target AMD node;
- the KKT BLAS-width policy already prevents nested oversubscription and caps
  useful dense factorization parallelism near `m / 256` threads.

This makes another uncalibrated Schur rewrite higher risk than improving the
iteration trajectory. The 29-iteration adaptive result spends approximately
one extra Schur and KKT factorization for every avoidable iteration.

### Parameter-history analysis

The complete JSON history from cluster job `194948.node220` was downloaded and
examined. The transition out of the fixed cold-start profile is the clear
failure point:

| Iteration | sigma | mu_aff / mu | affine primal / dual step | accepted primal / dual step | backtracks |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.075 | 0.231 | 0.772 / 0.973 | 0.800 / 0.800 | 2 |
| 5 | 0.272 | 0.567 | 0.346 / 0.971 | 0.512 / 0.640 | 5 |
| 6 | 0.500 | 0.939 | 0.019 / 0.135 | 0.0374 / 0.00376 | 27 |
| 7 | 0.500 | 0.879 | 0.146 / 0.078 | 0.275 / 0.116 | 8 |
| 9 | 0.500 | 0.973 | 0.021 / 0.033 | 0.0490 / 0.000660 | 24 |

The generic controller squares `mu_aff / mu` after a short affine step and
allows `sigma <= 0.5`. On this structured lattice model that combination
over-centers the corrector exactly when the affine predictor is already poor.
The trajectory remains near the cap through iteration 13 and converges in 29
iterations. The independently validated structural fixed profile uses
`sigma = beta = 0.075` and reaches the requested certificate in 24 iterations.

### Next experiment

Add an expert-only bounded `adaptive_sigma_max` with a conservative automatic
value for the existing `large_lattice_dense_schur` profile. Compare caps of
0.15 and 0.20 against both the unchanged adaptive controller and the fixed
0.075/0.8 trajectory on one cluster node. Retain the automatic cap only if the
full original-coordinate certificate passes and total solve time improves
stably.

### Candidate implementation and local regression

Implemented an expert `SolverOptions.adaptive_sigma_max` override. Zero keeps
automatic selection; positive values are bounded below by the fixed fallback
beta. Automatic selection currently proposes 0.15 only for
`large_lattice_dense_schur`, and leaves every other profile on the existing
generic policy. The execution plan reports the selected cap.

Local tests after the change:

| Test target | Result |
|---|---:|
| Adaptive parameter policy | 41 / 41 passed |
| Automatic pipeline and LP path | 68 / 68 passed across three test sets |
| SDP solver regressions | 120 / 120 passed across four test sets |

The first local test attempt was rejected before package loading because the
sandboxed Julia process could not create a precompile pidfile under the user
depot. Re-running with a writable `/private/tmp` depot succeeded; this was an
environment failure, not a solver failure.

Cluster job `195492.node220` was submitted to the normal queue with a 32-core
reservation. It will run 16 Julia and 16 BLAS threads, warm the complete
first-iteration path, and then compare caps 0.15, 0.20, the fixed profile, and
the unchanged generic 0.50 cap sequentially on the same node. Production is
unchanged.

### Same-node Task_Low08 comparison — job 195492

The job completed successfully on one node after an explicit one-iteration
warm-up. All adaptive runs used identical inputs, equality basis, tolerance,
threading, and validation boundaries.

| Strategy | Status | Iterations | Solve time | Total backtracks | Relative gap | Primal / dual residual | Minimum primal / dual eigenvalue | Certificate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive, cap 0.15 | Optimal | 28 | 32.673 s | 134 | 5.102e-7 | 6.27e-10 / 7.02e-11 | -5.90e-10 / 1.38e-14 | valid |
| Adaptive, cap 0.20 | Optimal | 28 | 32.895 s | 119 | 2.176e-7 | 3.32e-10 / 9.74e-12 | -7.13e-11 / 1.98e-15 | valid |
| Adaptive, cap 0.50 (unchanged) | Optimal | 29 | 35.287 s | 174 | 9.853e-7 | 5.42e-10 / 2.48e-11 | -6.58e-11 / 5.16e-15 | valid |
| Current fixed path | Stalled | 23 | 32.988 s | 535 | 2.0 | 1.23e-4 / 1.76e5 | -5.55e-3 / 1.34e-4 | invalid |

Both adaptive caps eliminate one full Newton iteration. Cap 0.15 is 7.4%
faster than the unchanged controller in this run; cap 0.20 is 6.8% faster,
uses 15 fewer backtracking contractions than 0.15, and gives a substantially
tighter PSD certificate. The 0.22-second difference between the two capped
runs is too small to distinguish from noise.

The current fixed path must not be used as the performance reference: unlike
the older 24-iteration implementation, it stalls after a collapsed step and a
restart, with an invalid certificate. This also means that automatically
falling back from the modern Mehrotra direction to the legacy fixed direction
would be unsafe on the current source.

Job `195495.node220` was submitted to sweep caps 0.10, 0.15, 0.20, 0.25,
0.30, and 0.40, with a repeated 0.20 point, before selecting the retained
automatic value.

### Refined cap sweep — job 195495

The refined sweep ran on the same node and with the same 16/16 thread layout.
Every adaptive point returned `Optimal` with a valid original-coordinate
certificate.

| Sigma cap | Iterations | Solve time | Backtracks | Relative gap | Minimum primal eigenvalue |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 28 | 32.114 s | 127 | 4.243e-7 | -1.707e-7 |
| 0.15 | 28 | 32.246 s | 134 | 5.102e-7 | -5.897e-10 |
| 0.20 | 28 | 29.928 s | 119 | 2.176e-7 | -7.126e-11 |
| 0.20 repeat | 28 | 32.418 s | 119 | 2.176e-7 | -7.126e-11 |
| 0.25 | 29 | 32.421 s | 153 | 2.980e-7 | -3.360e-11 |
| 0.30 | 29 | 33.689 s | 157 | 3.729e-7 | -1.758e-9 |
| 0.40 | 29 | 32.653 s | 148 | 5.670e-7 | 2.754e-10 |

Cap 0.20 is retained for the `large_lattice_dense_schur` automatic profile.
It is the largest value that consistently removes an iteration, has the
fewest backtracks in the sweep, and gives a materially better PSD certificate
than the more aggressive 0.10/0.15 choices. The repeated solve-time spread
(29.928--32.418 seconds) shows why selection is based primarily on iteration
count, backtracking work, and numerical quality rather than the fastest single
wall-clock sample. All other structural profiles remain unchanged.

### Full local regression

`Pkg.test()` completed after selecting the retained cap:

- 2,547 tests passed;
- 1 test remained marked `Broken` as expected;
- no failures or errors;
- wall time reported by the test set: 3 minutes 52.5 seconds.

The run covered Float64, MultiFloats, BigFloat, JLD2 checkpointing, MOI/JuMP,
preprocessing, solver certificates, threading, and package-quality checks.
macOS `sysctl` calls emitted sandbox permission warnings while probing host
memory/CPU data, but the guarded fallback paths worked and all assertions
passed. A prior direct `test/runtests.jl` attempt did not enter the tests
because that entry point omits `[extras]`; `Pkg.test()` correctly constructed
the isolated test environment and is the authoritative result.

## 2026-08-01 — Ownership-safe BigFloat block-diagonal equality KKT

### Audit

The CSDR handoff correctly identified one remaining integration blocker, but
the low-level kernel situation was better than the handoff snapshot implied.
`ExtendedPrecisionBLAS.syrk!` already had an ownership-safe BigFloat
specialization with disjoint lower-triangular output tiles and task-local MPFR
scratch. The actual blockers were:

- `ArrowWorkspace` rejected every BigFloat problem with explicit equalities;
- the execution planner forced ordinary BigFloat equality problems to one
  thread;
- the arrow equality forward/back substitutions used allocating scalar
  arithmetic and disabled BigFloat task parallelism;
- equality-Gram telemetry labeled a requested multi-thread call as threaded
  even when the crossover selected one worker.

### Retained implementation

Enabled the new route only when all of the following are proven:

1. arithmetic is `BigFloat`;
2. explicit equality columns are present;
3. coefficients use sparse storage;
4. every PSD block is 2x2;
5. every Schur variable occurs in exactly one PSD block.

This is the independent-cell structure of the primal CSDR model. Shared-arrow
variables or larger PSD blocks still fall back to the established general KKT
path. The execution plan reports `:owned_bigfloat_equality_tiles`, and the
workspace retains the requested Julia thread count while keeping only one
generic vector partial to avoid `threads * m` BigFloat memory growth.

The equality forward and transpose solves now mutate independently owned MPFR
destinations with `MA.sub_mul` and `_mpfr_divide!`. Each task owns complete,
disjoint row blocks and a private multiplication buffer. The equality Gram
continues to assign complete lower-triangular tiles exclusively; diagnostics
now use the kernel's actual selected worker count.

### Local correctness gate

Targeted four-thread tests passed:

| Test group | Passed |
|---|---:|
| Existing BigFloat sparse/arrow regressions | 95 |
| New full-rank block-diagonal equality KKT | 16 |
| New rank-deficient equality fallback | 7 |

The new full-rank test compares the lower Gram triangle with an independent
`Btil' * Btil` reference, checks both KKT equations to `1e-65` relative error,
and verifies distinct object identities in every result slot. The duplicated
equality-column fixture correctly switches to rank-revealing QR and reports
rank `n - 1`. No Float64 route was changed.

The first isolated test command omitted the `MultiFloats` test dependency and
therefore could not exercise the existing mixed-arrow fixture; rerunning with
the package test dependency loaded passed. This was a harness issue, not a
numerical regression.

### Next experiment

Measure serial owned-arrow versus tiled-parallel Gram on the same synthetic
512-bit problem, then run the immutable CSDR artifact on one 128-core PBS
allocation with 1/2/4/8/16/32/64/128 solver-thread points. Retain automatic
parallelism only if the end-to-end median improves without changing the KKT
residual, rank decision, objective, or physical certificate.

A later direct `test/pipeline.jl` invocation likewise reached its first
`StableRNG` use without the imports normally established by `runtests.jl` and
errored there. All preceding pipeline assertions passed, and the same file had
already passed inside the authoritative full `Pkg.test()` run; no source
change was made for this test-harness-only invocation error.

### Final thread layout and fixed-path diagnosis — job 195497

The retained 0.20 cap was repeated with 32 Julia threads and 16 OpenBLAS
threads inside the 32-core reservation:

| Run | Status | Iterations | Solve time | Backtracks | Relative gap | Certificate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | Optimal | 28 | 33.114 s | 119 | 2.176e-7 | valid |
| 2 | Optimal | 28 | 30.110 s | 119 | 2.176e-7 | valid |

The median is 31.612 seconds, about 2.5% below the 32.418-second median of the
three 16-Julia/16-BLAS cap-0.20 runs. Peak RSS for the complete two-run process
was 6,545,048 KiB (about 6.24 GiB). The recommended Task_Low08 launch layout
is therefore 32 Julia threads and 16 BLAS threads on this 32-core reservation;
the gain over 16/16 is modest, so topology and load still matter.

A separate fresh-process fixed strategy with the modern automatic Ruiz
pipeline reproduced the earlier stall: 23 iterations, 48.370 seconds, relative
gap 2.0, dual residual 1.76e5, and an invalid certificate. This differs from
the historical 24-iteration result because that profile was calibrated before
automatic Ruiz scaling became the SDP pipeline default. Job `195509.node220`
was submitted with `scaling=:none` to verify that hypothesis rather than
misclassifying a pipeline/parameter interaction as model reuse corruption.

### Fixed-profile scaling result — job 195509

The unscaled fixed diagnostic completed successfully:

- status `Optimal` in 24 iterations;
- primal / dual objective `0.6532912859301279` / `0.6532908586918761`;
- relative gap `4.27238e-7`;
- reported primal / dual residual `1.09910e-10` / `1.13744e-10`;
- original equality residual `1.099e-10`;
- minimum primal / dual PSD eigenvalue `-1.574e-9` / `4.070e-14`;
- solve / end-to-end time 48.766 / 50.628 seconds including cold JIT;
- valid final certificate.

This confirms the fixed solver was not corrupted. Its parameters and the Ruiz
pipeline form an incompatible, uncalibrated combination. Retained policy:

- `scaling=:auto`, large-lattice profile, adaptive strategy: Ruiz + sigma cap
  0.20;
- `scaling=:auto`, large-lattice profile, fixed strategy: no scaling, restoring
  the 24-iteration calibrated trajectory;
- explicit `scaling=:equilibrate` or `:none`: always respected;
- every other profile: unchanged.

The focused policy/scaling regression passed 48 / 48 assertions after this
final automatic-pipeline change. `git diff --check` also passed.

A typed BigFloat smoke check confirmed that an explicit 0.20 cap remains a
`BigFloat` at the active MPFR precision. The controller introduces no Float64
round-trip for BigFloat or MultiFloat diagnostics.

### Automatic fixed-profile scaling validation — job 195512

The final source was installed in the isolated cluster candidate and run with
`parameter_policy=:fixed` and `scaling=:auto`. The automatic selector reproduced
the explicitly unscaled reference trajectory exactly:

- PBS exit status 0; completion marker and checksummed artifacts present;
- status `Optimal` in 24 iterations;
- primal / dual objective `0.6532912859301279` / `0.6532908586918761`;
- relative gap `4.2723825e-7`;
- primal / dual residual `1.0990980e-10` / `1.1374410e-10`;
- maximum equality residual `1.0990986e-10`;
- minimum primal / dual PSD eigenvalue `-1.5744484e-9` / `4.0699682e-14`;
- valid original-coordinate optimality certificate;
- solver / end-to-end time 52.585 / 54.539 seconds in a cold process that
  also precompiled the changed package;
- peak RSS 3,924,684 KiB (about 3.74 GiB).

The numerical values and iteration history match job 195509, so the automatic
pipeline is selecting the intended no-scaling compatibility path for the
large-lattice fixed policy. This closes the fixed-policy regression without
changing the default adaptive path. Production was not promoted during the
experiment; all runs used the isolated candidate installation.

## 2026-08-01 — Adaptive scaling interaction and next iteration-count lever

### Audit conclusion

The dense Task_Low08 kernels have already exhausted the obvious low-risk
changes: sparse Cholesky fills completely, OpenBLAS beats the tested MKL and
BLIS installations on the target EPYC nodes, and prior packed-SYRK,
tiny-GEMM, unrolling, and tile-owned assembly experiments were slower. The
remaining measured cost is dominated by one Schur assembly and dense KKT
factorization per Newton iteration. Iteration-count reduction is therefore
still the highest-value direction.

Before implementing a higher-order centrality corrector, the scaling and
parameter controller need one missing controlled comparison. The fixed
large-lattice profile reaches a valid certificate in 24 iterations only in
the original coordinates, whereas the retained adaptive profile uses Ruiz
scaling and takes 28 iterations. No same-process adaptive comparison of these
two scaling choices had been recorded.

The benchmark helper now accepts `SDPX_SCALING_MODES` and records the requested
mode for every run. Cluster job `195527.node220` was submitted to the normal
queue with a 32-core reservation, 32 Julia threads, 16 OpenBLAS threads, one
shared input/presolve/ingest stage, a full-path warm-up, and the ordered modes
`auto,none,auto,none`. The no-scaling policy will be retained only if both
repetitions preserve the original-coordinate certificate and improve the
iteration trajectory; otherwise the next experiment is a guarded
factorization-reusing higher-order corrector.

Job `195527.node220` exited before loading SDPX because the isolated cluster
depot did not contain the `StructUtils 2.8.2` source newly referenced by the
candidate manifest through `JSON 1.6.1`. PBS reported exit status 1 after
4.58 seconds and no numerical solve ran, so this is an environment setup
failure rather than a rejected solver configuration. The replacement job
instantiates and precompiles the exact candidate manifest on its compute node
before running the unchanged comparison; no full test or solve is moved onto
a login node.

The first replacement, job `195530.node220`, also exited before loading SDPX:
compute nodes cannot resolve GitHub, so an online `Pkg.instantiate()` is not a
valid cluster-deployment mechanism. Inspection of the earlier successful job
scripts found the intended offline environment at
`<cluster-root>/SDPX.jl/depot`, initialized by
`hpc_julia_clarabel_env.sh`. The corrected job uses that depot with
`JULIA_PKG_OFFLINE=true`, exactly matching jobs 195492--195512. This second
failure likewise contains no solver timing or numerical result.

The offline depot itself lacked only the manifest-selected `StructUtils 2.8.2`
source. That exact UUID/version was copied from the local Julia package cache
into the SDPX-specific cluster depot without replacing any existing package.
A package-load-only login-node check then returned SDPX `0.3.0` successfully
when the dedicated depot was prepended to the environment-provided depot, as
the successful PBS scripts do. The numerical rerun uses this verified offline
search path.

### Controlled scaling result (job 195531.node220)

The corrected job completed with PBS exit status 0 on node125. It alternated
`auto,none,auto,none` after one shared warm-up, so both policies saw the same
input, rank-394 equality basis, process, node, 32 Julia threads, and 16
OpenBLAS threads. All four solves returned `Optimal` with valid
original-coordinate certificates.

| Scaling | Repetition times (s) | Median (s) | Iterations | Backtracks | Gap |
|---|---:|---:|---:|---:|---:|
| Ruiz (`auto`) | 32.416, 32.708 | 32.562 | 28 | 119 | 2.176e-7 |
| None | 33.050, 32.675 | 32.863 | 24 | 102 | 9.303e-7 |

Removing scaling saves four Newton iterations and 17 backtracks, but is 0.92%
slower in median wall time and finishes much closer to the requested gap
limit. It also raises the per-iteration block-local cost: for example,
residual/block factorization is 0.129--0.138 s per unscaled iteration versus
0.071--0.079 s with Ruiz, and predictor/corrector direction recovery is about
0.062--0.073 s versus 0.036--0.039 s. Therefore the automatic adaptive policy
continues to use Ruiz; the no-scaling proposal is rejected as a runtime
optimization despite its lower iteration count. Peak RSS for the complete
four-run process was 6,484,908 KiB.

### Cost-aware block-loop crossover

The phase breakdown exposed a more direct low-risk opportunity. The existing
Julia block scheduler engaged only when a model contained at least 256 PSD
blocks. Task_Low08 has 32 blocks with dimensions 23--74 (sum of cubes
3,977,757), so residual construction, block Cholesky, predictor RHS/direction,
and corrector RHS/direction all ran serially even with 32 Julia threads.

`_block_loop_threading_profitable` now preserves the old 256-block rule and
also enables the block-local scheduler when the dimension-derived cubic work
exceeds a conservative arithmetic-aware threshold: 1.0e6 for Float64-family
arithmetic and 1.0e5 for fixed-width extended arithmetic. Mutable BigFloat is
still excluded by the existing ownership rule.

A four-thread synthetic benchmark used the exact Task_Low08 dimension vector,
32 sparse blocks, one BLAS thread, and an allocation-reusing cycle containing
residual/block factorization, predictor RHS, direction recovery, and Mehrotra
corrector construction. Seven post-warm-up samples gave median times of
5.480 ms serial and 2.715 ms scheduled, a 2.02x speedup, with matching output
at 1e-12 relative and absolute tolerance. The focused scheduler regression
suite passes 274/274 tests.

The full A/B ran as PBS job `195533.node220` on node125. The unchanged
baseline and candidate each executed two measured adaptive/Ruiz solves in
separate processes with 32 Julia threads and 16 OpenBLAS threads. PBS exited 0
and all four solves returned `Optimal`, 28 iterations, 119 backtracks, and
valid certificates.

| Variant | Repetition times (s) | Median (s) | Median internal total (s) |
|---|---:|---:|---:|
| Serial block crossover | 33.453, 30.672 | 32.062 | 31.358 |
| Cost-aware block tasks | 29.770, 27.106 | 28.438 | 27.724 |

The candidate is 11.3% faster by median measured solve time. Median phase
times changed as follows: residual/block factorization 1.671 -> 0.438 s,
predictor 2.691 -> 1.273 s, and corrector 3.207 -> 2.015 s. Schur assembly
was essentially unchanged (8.180 -> 7.898 s), while KKT factorization noise
moved 13.804 -> 14.297 s. The objective changed by 8.1e-15, the dual
objective by 2.0e-12, the relative gap by 2.0e-12, and the minimum primal PSD
eigenvalue by 1.2e-16. Peak process RSS increased from 6,591,948 to 6,645,504
KiB (0.81%), with no extra task-local matrices. This is a stable, clear
improvement and the crossover is retained.

The complete local package test target passed 2,611/2,611 tests on Julia 1.12
with four threads. A direct `test/runtests.jl` invocation initially failed
before loading the test suite because it does not activate the declared JLD2
test dependency; rerunning through `Pkg.test()` used the correct test
environment and passed. The benchmark helper now also accepts
`SDPX_SOLVER_THREADS` so the next experiment can calibrate block-task width
independently of the fixed 16-thread BLAS factorization.

A second nine-sample local calibration on the same Task-shaped sparse cycle
measured medians of 5.273 / 2.773 / 2.524 ms at 1 / 2 / 4 Julia workers.
Two workers already provide 1.90x and four provide 2.09x, indicating limited
additional parallelism on the four-core local runtime. This reinforces the
need for the cluster sweep rather than assuming all 32 block tasks are useful.

PBS job `195534.node220` then swept 4/8/16/32/32/16/8/4 solver workers in
one process on node125 while holding OpenBLAS at 16 threads. Every run returned
the same 28-iteration, 119-backtrack, valid certificate trajectory. Paired
median solve times were 36.708 / 32.158 / 29.780 / 27.424 seconds at
4 / 8 / 16 / 32 workers. The 32-worker medians were 0.412 s for residual and
block factorization, 1.318 s for the predictor, 1.967 s for the corrector,
8.013 s for Schur assembly, and 13.280 s for KKT factorization. The full sweep
peaked at 7,106,144 KiB RSS. One worker per Task_Low08 PSD block is therefore
the retained configuration; an automatic cap below the requested 32 workers
would lose performance.

With block-local work addressed, Schur and KKT account for about 80% of the
internal solve time. Previous kernel/backend experiments already rejected
wider BLAS for sparse Schur assembly, packed dense SYRK, sparse Cholesky, and
alternative BLAS libraries. The next bounded experiment tests whether the
15% free-memory cap on task-local Float64 Schur accumulators is now overly
conservative under an explicit 28 GiB scheduler-aware memory ceiling.

The controlled 15%-versus-25% run was PBS job `195537.node220` on node125,
again with 32 Julia and 16 OpenBLAS threads and two solves per configuration.
The 15% solve times were 33.184 and 27.449 seconds; the 25% times were 26.871
and 25.965 seconds. The first baseline repetition was noisy, but comparing the
stable second repetitions still gives a 5.4% end-to-end improvement. Median
Schur assembly fell from 8.140 to 6.701 seconds. All four runs used the same
28 iterations and 119 backtracks and the candidate objectives, residuals,
gap, minimum PSD eigenvalue, and certificate-valid flag exactly matched the
retained block-thread candidate at reported precision.

The wider policy did not raise observed peak memory: process `Sys.maxrss()`
was 6,737,629,184 bytes versus 6,778,216,448 bytes for the baseline, and
`/usr/bin/time` reported 6,590,048 versus 6,637,764 KiB. The difference is
consistent with lazy/triangular first-touch and run-to-run allocator noise,
not a claim that more logical partials require less virtual storage. Since the
job remains far below its 32 GiB allocation and the speedup is stable, a final
25%-versus-35% experiment will locate whether useful parallelism remains
before selecting a conservative cap.

### Schur accumulator ceiling and retained crossover — job 195538

The final ceiling experiment compared 25% and 35% in the same process on
node125 with 32 Julia threads, 16 OpenBLAS threads, the same 28 GiB explicit
ceiling, two measured solves per policy, and identical adaptive/Ruiz options.
PBS exited successfully and every solve returned the same 28-iteration,
119-backtrack, certificate-valid numerical trajectory.

| Memory share | Solve times (s) | Median (s) | Median Schur (s) |
|---|---:|---:|---:|
| 25% | 28.678, 25.232 | 26.955 | 7.154 |
| 35% | 27.220, 25.763 | 26.491 | 6.842 |

Although the 35% median is 1.7% lower, the stable second repetition is 2.1%
slower. Median Schur saves only 0.312 seconds, median KKT is unchanged within
noise (13.619 versus 13.648 seconds), and peak RSS rises from 6,580,096 to
6,609,016 KiB by `/usr/bin/time` (0.44%). The objectives, residuals, gap, PSD
margins, iteration count, and certificate flag match exactly at reported
precision. The 35% proposal is therefore rejected.

The retained implementation uses 25% only for `Float64` systems satisfying
all four conservative gates: Schur dimension at least 4,096, at least 16 PSD
blocks, at least 16 requested workers, and at least 16 GiB of scheduler-aware
available memory. All other cases, including `Float64x4` and native BigFloat,
keep the historical 15% limit. At Task_Low08's `m = 6119`, 32 blocks/workers,
and 28 GiB ceiling, the policy selects 25 task-local accumulators; at 15 GiB it
selects eight. Focused pipeline and Schur-scheduler regressions pass after the
policy was narrowed and made directly testable.

This concludes the low-risk Task_Low08 CPU scheduling sweep. The remaining
dominant phases are dense KKT factorization (roughly 13--14 seconds) and Schur
assembly (roughly 6.7--7.2 seconds). Earlier controlled work already rejected
sparse Cholesky because of full fill-in, wider KKT BLAS teams, MKL/BLIS on the
target EPYC platform, packed dense SYRK, tiny GEMM substitution, and more
aggressive task-local memory. A higher-order corrector or a new distributed
dense factorization would change the algorithmic risk profile and is not an
obvious incremental optimization.

### Final local verification and isolated cluster candidate

After narrowing the memory rule, the focused pipeline tests passed, the Schur
scheduler regression passed 274/274 assertions, and the complete package test
target passed 2,619/2,619 tests in 3m51.1s on Julia 1.12 with four threads.
The Documenter build, doctests, cross-reference checks, and `git diff --check`
also completed without error.

The exact tested worktree was synchronized to the isolated cluster directory
`candidates/final-task-optimization-20260801-1/source`. PBS job
`195541.node220` requests 32 normal-queue cores and 32 GiB, uses 32 Julia
threads, 16 OpenBLAS threads, a 28 GiB SDPX memory ceiling, and two warmed
Task_Low08 solves. The production `current` link remains unchanged while this
final certificate and timing gate runs.

### Final Task_Low08 validation — job 195541

The final job ran on node125 and completed with PBS exit status 0. The
`COMPLETE` marker was present and SHA-256 verification passed for the JSON,
log, and `/usr/bin/time` report. Both measured solves returned `Optimal` in 28
iterations with 119 backtracking contractions and identical validated
numerical results.

| Run | Solve (s) | Internal (s) | Schur (s) | KKT (s) | Predictor (s) | Corrector (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 30.679 | 29.724 | 8.131 | 15.247 | 1.471 | 2.491 |
| 2 | 26.010 | 25.282 | 6.870 | 12.742 | 1.361 | 2.049 |

The two-run solve median is 28.344 seconds; the second warmed solve is 26.010
seconds. Each run allocated 3.242 GB in Julia. The complete process, including
input, rank-394 equality presolve, ingest, warm-up, two solves, validation, and
package compilation, peaked at 6,622,180 KiB RSS and completed in 2m19.80s.
Input, equality presolve, and ingest took 0.516, 0.260, and 0.678 seconds.

Final numerical values were:

- primal / dual objective: 0.6532912655025932 / 0.6532910479398872;
- relative gap: 2.1756271e-7;
- primal / dual residual: 3.3158809e-10 / 9.7011062e-12;
- maximum original equality violation: 3.3158809e-10;
- minimum primal / dual PSD eigenvalue: -7.1259986e-11 / 1.9749806e-15;
- certificate: valid, with no failed primal or dual PSD blocks.

After the diagnostic report fields were added, the exact final worktree passed
the complete package target again: 2,623/2,623 tests in 3m54.1s. The final
cluster run therefore confirms both the retained performance behavior and the
original-coordinate numerical gates. Production remained unchanged; only the
isolated candidate was exercised.

## 2026-08-01 — Native BigFloat equality-arrow scaling

### A2/A3 baseline and 128-core calibration — job 195760

The ownership-safe BigFloat block-diagonal equality route from commit
`152b0c18f48df0b382b1a75f620199be24075e37` was installed as an immutable
candidate and benchmarked on node178, a dual-socket AMD EPYC 7742 node with
128 physical cores and eight NUMA domains. The job requested one 128-core
normal-queue node, 220 GiB, disabled nested BLAS/OMP threading, and used
`numactl --interleave=all`. Production `current` was not changed.

At 512 bits, the representative 3,400-variable / 144-equality system retained
the same `6.07e-153` relative KKT residual at every thread point. Gram time
fell from 4.154 seconds at one worker to 0.105 seconds at 128 workers (39.6x),
and complete factorization fell from 4.357 to 0.273 seconds (16.0x). The old
serial BigFloat GEMV path made the complete solve regress from 0.242 to 0.355
seconds as the requested thread count increased.

The full J80/Na20-shaped 16,400-variable / 230-equality system also retained
one `5.64e-152` residual at all thread points:

| Workers | Factor (s) | Equality Gram (s) | KKT solve (s) |
|---:|---:|---:|---:|
| 1 | 60.093 | 58.618 | 1.790 |
| 8 | 10.077 | 9.549 | 1.799 |
| 32 | 3.342 | 2.646 | 2.078 |
| 64 | 2.044 | 1.476 | 2.072 |
| 128 | 1.932 | 1.339 | 2.132 |

This is a 43.8x Gram speedup and a 31.1x complete-factor speedup at 128
workers. The process peak was 5,267,016 KiB, well below the 220 GiB request.
The complete result directory contains a `PASSED` marker and SHA-256 manifest.

The tile sweep showed that the static tile 8 remains reasonable but is not
optimal on this Zen 2 node. At 128 workers, tile 6 had a 0.114-second median
versus 0.128 seconds for tile 8; at 64 workers the medians were 0.186 and
0.190 seconds. At one worker tile 12 was best. Apple M4 calibration favored
tile 4 at four to eight workers, so no cross-platform hard-coded tile change
was retained. A host-calibrated tile field remains a bounded follow-up.

### Ownership-safe BigFloat equality GEMV

Profiling identified the two dense equality matrix-vector products in every
predictor/corrector KKT solve as the reason solve time did not scale. A new
internal BigFloat GEMV partitions complete output ranges among workers. Each
worker owns every MPFR destination it writes and has private accumulator and
multiplication buffers; each dot product keeps the serial reduction order.
The worker count is capped by an 18,000 multiply-add work threshold so small
panels stay serial.

A 512-bit local 3,400-by-144 KKT benchmark reduced solve time from 0.0775
seconds at one worker to 0.0190 seconds at eight workers (4.1x). Every thread
point retained the `6.07e-153` residual. Direct tests of both `B*x` and
`B'*x` are bit-for-bit equal to `kmul_owned!` and verify unique destination
object identities.

An attempted allocation reduction reused one MPFR multiplication buffer
across multiple local 2x2 triangular solves inside a parallel task. Although
small isolated checks sometimes matched, repeated 3,400-variable runs
produced relative KKT errors around `1e-3`. The proposal was rejected and
fully removed. The retained GEMV shares no mutable BigFloat object and leaves
the already validated per-block triangular scratch policy unchanged.

The focused BigFloat sparse regression file passes all 123 assertions. A
complete package run passed 2,650 numerical, API, and quality assertions; the
only failure was Aqua's subprocess-based persistent-task check because its
temporary wrapper could not reach `pkg.julialang.org` in the sandbox. No
ordinary test failed. A subsequent isolated, network-enabled run of Aqua
0.8.16's `test_persistent_tasks(SDPX)` completed successfully with exit code
zero, confirming that the sandbox failure was an infrastructure issue rather
than a persistent-task regression. The next gate is a cluster A/B of the
retained GEMV followed by the immutable J40 CSDR model.

### A3 equality-GEMV cluster validation — job 195853

Commit `d4ba4f2b6a1e2b6222605d2883b4121113f5f051` was installed as a second
immutable candidate and tested at BigFloat512 on node141 (AMD EPYC 7742),
using 64 physical cores, one BLAS thread, and interleaved NUMA allocation.
The production symlink remained unchanged.  The node also hosted an unrelated
job on the other half of the machine, so factor/Gram differences against the
node178 baseline are treated as secondary evidence; the GEMV improvement is
large enough to remain unambiguous.

| Geometry | Workers | Factor (s) | Equality Gram (s) | KKT solve (s) | Relative KKT residual |
|---|---:|---:|---:|---:|---:|
| 3,400 x 144 | 1 | 4.805 | 4.462 | 0.2513 | `6.068377e-153` |
| 3,400 x 144 | 64 | 0.262 | 0.129 | 0.02432 | `6.068377e-153` |
| 16,400 x 230 | 1 | 55.079 | 53.213 | 1.8663 | `5.637828e-152` |
| 16,400 x 230 | 64 | 2.021 | 1.468 | 0.07641 | `5.637828e-152` |

Against the pre-GEMV 64-worker measurements, KKT solve time improved from
0.3080 to 0.02432 seconds (12.7x) for 3,400 x 144 and from 2.0715 to 0.07641
seconds (27.1x) for 16,400 x 230.  One-worker time stayed within node-to-node
variation.  Every requested worker point from 1/2/4/8/16/32/64 retained the
same residual, and the complete job exited zero with a `PASSED` marker and
SHA-256 manifest.  Peak RSS was 4,481,652 KiB, although it is not directly
comparable to the earlier job that also exercised 128-worker configurations.
The change is retained.  The next gate is the strict real-model J40 CSDR
BigFloat512 solve and off-grid physical validation.

### A3 real J40 BigFloat512 baseline — job 195874

The first submitted driver (job 195869) stopped before model loading because
an escaped dictionary key inside a Julia string interpolation did not parse.
The driver was corrected, parsed locally and on the cluster, committed as
`28f928249f72f315118a36e28ca4626b33117e9a`, and installed as a new immutable
candidate. No solver code or numerical state ran in the failed job.

The corrected 64-worker run used the certified J40, Na20, Nmu200,
twice-subtraction maximum model. It solved the exact rounded Float64x4 model
with native BigFloat512 arithmetic, adaptive IPM parameters, Ruiz
equilibration, the owned block-diagonal arrow path, and normal-equation
equalities. It completed in 157 iterations with no restart or regularization:

- solver / end-to-end wall time: 734.47 / 825.81 seconds;
- `/usr/bin/time` peak RSS: 3,764,096 KiB (3.59 GiB);
- physical objective: `21.0253439247597371127872626908378907964`;
- relative gap: `1.88989002973937e-13`;
- primal / dual residual: `1.13e-153` / `1.99e-133`;
- maximum original linear residual: `3.90e-63`;
- maximum off-grid relative residual: `1.75049e-11`;
- minimum PSD eigenvalue: `1.76019e-34`;
- maximum disk violation: exactly zero.

The strict physical validator accepted the result. The objective differs from
the existing certified Float64x4 value by about `4.47e-12` (relative
`2.13e-13`), consistent with the two primal-dual gaps. The result directory
contains `PASSED`, complete timing and validation files, the solution, and a
SHA-256 manifest.

The new Gram/GEMV kernels changed the end-to-end bottleneck. Of the 593.46
seconds attributed to internal phases, residual plus local-block
factorization used 155.63 seconds (26.2%), explicit KKT refinement used 150.22
seconds (25.3%), and KKT factorization used 130.29 seconds (22.0%). Equality
Gram is now 66.28 seconds (11.2%); the equality triangular preparation and
factorization add 41.68 and 22.31 seconds. The native unregularized factor is
already accurate to roughly the BigFloat512 floor, but `refine_policy =
:adaptive` still evaluates the expensive exact KKT residual every iteration.
The next controlled A/B extends the existing conservative `:auto` refinement
skip from singleton arrows to the new exact all-local equality specialization.

The first version of that extension (job 195896, commit `06ee004`) produced a
useful negative control. It was numerically identical and again passed every
certificate, but solve time was 739.26 seconds and refinement still consumed
152.34 seconds. Telemetry showed only eight accepted correction steps across
157 iterations: almost all of the time was spent measuring a residual that
was already below the useful target. The adaptive controller had inserted an
iteration-local refinement tolerance before `_skip_automatic_refinement` ran;
the latter mistook this internal value for an explicit user override and
declined the skip. Commit `f6f5f12` moves the skip decision before controller
overrides while preserving all existing safety gates. A same-node rerun is in
progress; the ineffective `06ee004` policy wiring will not be retained as the
final implementation.

The corrected same-node run (job 195898, commit `f6f5f12`) retained the
optimization. Refinement fell from 152.34 to 0.034 seconds and the reported
accepted correction count fell from eight to zero. Solver time decreased from
739.26 to 589.84 seconds (20.2%, 1.25x), while internal timed phases decreased
from 593.88 to 451.66 seconds. End-to-end process time decreased from 833.66
to 680.95 seconds (18.3%). `/usr/bin/time` peak RSS increased from 3,888,916
to 4,020,652 KiB (3.4%), within the 5% acceptance gate and opposite in sign to
`Sys.maxrss`, so it is treated as run-to-run variation rather than a retained
memory cost.

The candidate needed 158 instead of 157 iterations but reached a tighter
`3.45e-14` gap. Its physical objective was
`21.0253439247613726121335648342943317108`; original-grid residual was
`3.90e-63`, off-grid relative residual `1.75050e-11`, minimum PSD eigenvalue
`3.00e-35`, and disk violation zero. The strict validator accepted the result
and the job produced `PASSED` plus a complete SHA-256 manifest. The objective
shift is smaller than the certified primal-dual envelopes. The conservative
automatic skip is therefore retained.

With refinement removed, residual/local-block factorization is the largest
single timed phase (160.02 seconds), followed by KKT factorization (131.57
seconds). The next experiment assigns complete BigFloat block workspaces and
their unique all-local variable destinations to workers. Sparse equality
products iterate CSC nonzeros directly; transpose products assign complete
columns to workers. No mutable BigFloat destination is shared.

### A4 owned BigFloat block scheduling — local preflight

The next candidate implements a deliberately narrow scheduler for native
BigFloat problems that satisfy the already validated all-local 2x2 equality
arrow predicate. Complete PSD blocks are assigned to workers, which gives each
task exclusive ownership of its `P`, `R`, `LX`, `MY`, `Z`, `dX`, and `dY`
matrices and every all-local vector destination that it updates. Sparse
equality products now iterate CSC nonzeros directly: `B*y` remains serial
because columns may share output rows, while `B'*x` assigns complete output
columns to workers. Every MPFR accumulator and multiplication buffer is
private to its worker.

On the Apple M4 local preflight at BigFloat512 with 3,400 variables, 144
equalities, and 1,700 independent 2x2 blocks, median phase times were:

| Julia threads | Residual + block factor (s) | Predictor RHS (s) | Direction recovery (s) | Corrector RHS (s) |
|---:|---:|---:|---:|---:|
| 1 | 0.02003 | 0.00553 | 0.00726 | 0.00980 |
| 2 | 0.01349 | 0.00281 | 0.00324 | 0.00530 |
| 4 | 0.01243 | 0.00234 | 0.00257 | 0.00461 |
| 8 | 0.01101 | 0.00172 | 0.00189 | 0.00298 |

The historical generic one-thread residual path took 0.12672 seconds on the
same fixture. Direct sparse equality traversal therefore accounts for a 6.3x
one-thread reduction, while block scheduling improves the complete residual
phase by 11.5x relative to that baseline at eight threads. Predictor,
direction, and corrector phases scale by 3.2x, 3.8x, and 3.3x respectively.
All serial/parallel matrices and vectors compared exactly. The focused
BigFloat regression file passes its existing cases plus 2,057 assertions for
the new scheduler, and the two related extended-precision BLAS files pass all
113 assertions. The next acceptance gate is a same-model 64/128-core J40
BigFloat512 cluster solve with the full physical certificate.

### A4 J40 rejection and MPFR alias root cause — jobs 195901--195903

Two 128-core attempts (jobs 195901 and 195902) were assigned to node135 and
terminated by PBS before the script started. Both accounting records report
`exit_status=-9`, zero CPU time, zero RSS, and no output file. This is a node
or MOM startup failure rather than a solver result, so repeated submission to
that node was stopped. The 64-core fallback ran normally on node143.

The first real candidate, commit `62dff7f`, was rejected. It reached iteration
2, then produced an unhealthy direction, exhausted ten restart repairs, and
returned `MaxRestartsExceeded`; the physical validator correctly rejected the
zero best iterate. This was a useful long-trajectory gate: 4,712 local tests
passed, including 2,057 one-shot serial/parallel equality checks, but they did
not reuse the direction buffer across iterations.

The root cause was mutable-object aliasing in the generic
`symmetrize_inplace!`. Assigning the same `avg::BigFloat` to both symmetric
positions made the two off-diagonal entries refer to one MPFR object. The
historical allocating `kmul!` replaced every output object on the next
iteration and accidentally repaired the alias. The new allocation-free
`kmul_owned!` correctly reused its destinations, exposing the latent bug: a
write to one off-diagonal entry silently changed the other during the next
matrix traversal. This explains why iteration 1 was plausible and iteration
2 diverged.

A BigFloat specialization now averages into private scratch, mutates two
already independent destinations in place, and defensively repairs an
arbitrary pre-aliased pair with independent copies. The scheduler regression
now runs direction recovery twice and requires all four 2x2 entries to have
unique object identities after each pass. The focused suite passes 3,081
assertions. The corrected candidate must repeat the full J40 certificate gate;
no result from `62dff7f` is accepted or promoted.

The alias-corrected follow-up `b3fb80f` also failed the full gate, although it
no longer collapsed. It reached 220 iterations in 535.30 seconds and returned
`IterLimit`; the best iterate had physical objective
`21.025343924720055`, relative gap `3.96e-12`, valid PSD/off-grid quantities,
but a large original dual residual and therefore a rejected status. Comparing
iteration logs with the accepted `f6f5f12` baseline localized the divergence:
iterations 1 and 2 have the same printed objectives, primal residuals, and
steps, while at iteration 3 only the candidate's dual residual jumps from the
BigFloat floor to `1.01997` and then grows. This implicates the combined
residual/block-factor specialization rather than the KKT factorization or
adaptive controller.

The next isolation candidate restores the validated serial residual and block
factorization path while retaining the ownership-fixed block-parallel
predictor, direction, and corrector phases. The rejected residual kernel
remains unreachable while the exact subcomponent is diagnosed; it will be
removed or repaired before final release.

Direct review then found the exact residual defect. Both new CSC `B*y` and
`B'*x` kernels called `MutableArithmetics.buffered_operate!` with the
destination both as the accumulator and as an extra multiplicand. The API
already treats its first post-operation argument as the mutable accumulator,
so the duplicate argument implemented the wrong recurrence; the transpose
case stayed identically zero from a zero start. The existing equality-GEMV
regression used a dense panel and therefore never dispatched to these
CSC-only helpers. Both calls now pass only `(value, scalar)` after the
accumulator. New direct sparse forward and transpose tests require exact
agreement with the established owned dense kernel and unique MPFR
destinations. The corrected fast residual path is re-enabled only for a new
immutable candidate and must pass J40 from scratch.

The isolation run itself (job 195915, commit `a3b9e2d`) completed successfully
on node58 and establishes the independent value of the remaining block
scheduler. It reproduced the accepted baseline exactly: 158 iterations,
physical objective
`21.0253439247613726121335648342943317107892849925173565643`, relative gap
`3.45084e-14`, original linear residual `3.89952e-63`, off-grid relative
residual `1.75050e-11`, minimum PSD eigenvalue `3.00255e-35`, and zero disk
violation. Solver time was 560.14 seconds versus 589.84 seconds for
`f6f5f12`, a 5.0% reduction. Predictor plus corrector time fell from 111.79
to 65.00 seconds, while the restored serial residual/block factor phase
remained dominant at 165.45 seconds. Peak RSS was 3,632,936 KiB. `PASSED`,
all expected artifacts, and every SHA-256 verification succeeded.

### A5 Corrected sparse residual accepted — job 195930

Commit `6fa4526c` repaired the sparse BigFloat CSC forward and transpose GEMV
recurrences, added direct exactness and MPFR-object-ownership regressions, and
re-enabled the all-local residual path. The complete local suite passed 5,739
tests before the immutable cluster candidate was installed.

The BigFloat512 J40 certificate run (job 195930 on node156, 64 Julia threads)
passed every gate in 158 iterations. It reproduced the accepted reference
bit-for-bit: objective
`21.0253439247613726121335648342943317107892849925173565642655547851283405741418584401534504196441654763365346531515530664755637606224505556451889536532886632`,
relative gap `3.45084e-14`, primal residual `1.39838e-153`, dual residual
`9.42437e-134`, original linear residual `3.89952e-63`, off-grid relative
residual `1.75050e-11`, minimum PSD eigenvalue `3.00255e-35`, and zero disk
violation. `PASSED`, every artifact, and every SHA-256 verification succeeded.

Solver time fell from 589.8396 to 423.7045 seconds (1.392x, 28.2% lower),
while measured internal time fell from 451.6624 to 281.2693 seconds (37.7%
lower). Residual plus block factorization fell from 160.0222 to 31.6752
seconds (5.05x, 80.2% lower). KKT remained the largest phase at 131.4683
seconds: 67.6314 seconds for equality Gram construction, 41.0518 seconds for
triangular work, and 22.7716 seconds for the equality factorization. Predictor
and corrector took 11.9450 and 55.6010 seconds; complementarity analysis took
42.5873 seconds and line search 17.8919 seconds. Peak RSS was 3,508,312 KiB
under `/usr/bin/time`, 12.7% below the accepted pre-scheduler baseline. This
candidate is retained. The next experiment targets complementarity diagnostics,
fraction-to-boundary search, and accepted-step updates using the same strict
complete-block ownership rule.

### A6 Owned predictor diagnostics and line-search preflight

The next candidate extends complete-block ownership to affine/legacy
complementarity diagnostics, fraction-to-boundary evaluation, and accepted
block updates. Per-block MPFR dot products write into independently allocated
`block_norms` slots, and every global sum retains the original block order.
The fraction-to-boundary scheduler uses separate primal and dual waves. An
initial test exposed a subtle mutable-reference hazard: `min` returned one of
the `block_norms` BigFloat objects, so the dual wave overwrote the saved primal
bound. Copying that one scalar between waves fixed the issue. The regression
now compares serial and parallel affine diagnostics, legacy diagnostics, step
lengths, updated matrices, complementarity, targets, and objectives exactly.
All 3,088 assertions in the owned scheduling test pass, as do all other
focused BigFloat tests.

At the J40 block count (1,700 2x2 blocks) and BigFloat512 on the Apple M4, an
eight-thread median microbenchmark measured:

| Phase | Serial (ms) | 8 threads (ms) | Speedup |
|---|---:|---:|---:|
| Adaptive predictor diagnostics | 5.904 | 2.358 | 2.50x |
| Fraction-to-boundary search | 3.580 | 1.271 | 2.82x |
| Accepted block update | 1.659 | 0.772 | 2.15x |
| Dual objective | 0.326 | 0.328 | 0.99x |
| Adaptive target update | 0.00096 | 0.0263 | 0.04x |

The dual-objective and target-update experiments were rejected and restored to
their serial implementations. Their scalar work is too small to amortize task
launches. Only the three consistently faster phases remain in the candidate.
The complete four-thread local suite then passed all 5,747 tests in 3m56.7s.

### A6 J40 acceptance — job 196046

Commit `27f4751` was installed as an immutable release and run on the same
node156 geometry as the preceding accepted candidate: 64 Julia threads, one
BLAS thread, BigFloat512, and interleaved NUMA allocation. The job passed in
158 iterations and reproduced the complete objective, residual, gap, PSD, and
off-grid certificate bit-for-bit. Solver time fell from 423.7045 to 368.7039
seconds (13.0% lower), and measured internal time fell from 281.2693 to
229.7291 seconds (18.3% lower). Complementarity analysis fell from 42.5873 to
19.4535 seconds (2.19x), fraction-to-boundary line search from 17.8919 to
4.5867 seconds (3.90x), and accepted updates from 4.9455 to 0.8688 seconds
(5.69x). KKT also varied down from 131.4683 to 121.5944 seconds on the same
node; this secondary change is not attributed to the block controller.

`/usr/bin/time` reported 457.82 seconds end to end and 3,907,204 KiB peak RSS.
The peak is 11.4% above job 195930 despite no persistent matrix addition, so
it is recorded as a possible runtime/GC variation rather than claimed as a
memory improvement. PBS exited zero, `PASSED` exists, and all six artifact
hashes verify. Production `current` remains unchanged.

### A7 Reusing BigFloat Cholesky reciprocals — rejected

The remaining KKT profile spends 37.7163 seconds applying thousands of 2x2
local Cholesky factors to 170 equality columns and 23.2830 seconds factoring
the 170x170 equality normal matrix. Two conservative reciprocal-reuse
prototypes were therefore evaluated:

- Each 2x2/3x3 local factor cached its diagonal reciprocals in otherwise
  unused upper-triangular factor slots plus the existing per-block `Dinv`
  scalar. At the actual two-variable-per-block J40 geometry, the equality
  panel transform fell from 29.945 to 21.669 ms on eight M4 threads (1.38x).
  The cached result differed from direct MPFR division by `7.75e-156`
  relative at 512 bits.
- For Cholesky factors of order at least 16, one reciprocal per column replaced
  repeated divisions below the diagonal. A 170x170 BigFloat512 factor fell
  from 64.052 to 61.456 ms (1.04x); relative factor difference was
  `9.66e-155` and reconstruction residual `9.88e-154`.

Focused tests passed, as did all 5,831 local tests in 3m58.1s. The immutable
cluster gate was job 196053 on the same node156, with the same 64 Julia
threads, one BLAS thread, BigFloat512 arithmetic, model, and validation.
All hashes and physical gates passed, but the candidate was not retained:

| Metric | Accepted A6 | Reciprocal A7 | Change |
|---|---:|---:|---:|
| Iterations | 158 | 157 | -1 |
| Solver time (s) | 368.704 | 370.705 | +0.5% |
| Internal time (s) | 229.729 | 235.201 | +2.4% |
| Constraint triangular (s) | 37.716 | 37.000 | -1.9% |
| Equality factor (s) | 23.283 | 21.744 | -6.6% |
| Equality Gram (s) | 60.580 | 67.151 | +10.8% |
| Relative gap | `3.45084e-14` | `1.88989e-13` | 5.48x worse |
| Peak RSS, `/usr/bin/time` (KiB) | 3,907,204 | 3,955,072 | +1.2% |

The final physical objective moved from the bit-for-bit reference
`21.02534392476137261213...` to `21.02534392475973711279...`. Although that
solution was still physically accepted, the changed trajectory, weaker gap,
and absent end-to-end speedup violate the retention criteria. Both reciprocal
paths and their tests were reverted. This negative result confirms that MPFR
division replacement must be evaluated at full-solver trajectory level rather
than inferred from isolated kernels.

### A8 128-core scaling and phase-aware BigFloat task cap

The restored A6 code was installed as immutable commit `d6b2198`. Job 196080
ran the same J40 BigFloat512 model on all 128 physical cores of node150, with
one BLAS thread and interleaved NUMA allocation. It reproduced the complete
158-iteration objective, gap, residuals, PSD margin, off-grid residual, and
disk certificate bit-for-bit; all hashes passed. Numerical determinism is
therefore preserved at 128 workers.

The unmodified scheduler did not scale as a complete solver:

| Phase | 64 workers, job 196046 (s) | 128 workers, job 196080 (s) |
|---|---:|---:|
| Solver | 368.704 | 495.811 |
| Internal total | 229.729 | 356.574 |
| KKT factorization | 121.594 | 110.009 |
| Equality Gram | 60.580 | 43.446 |
| Equality factorization | 23.283 | 19.794 |
| Constraint triangular | 37.716 | 46.755 |
| Residual and block factor | 31.282 | 47.694 |
| Predictor | 11.042 | 41.885 |
| Corrector | 31.817 | 100.628 |
| Complementarity analysis | 19.453 | 49.665 |
| Fraction-to-boundary line search | 4.587 | 18.934 |
| Accepted update | 0.869 | 7.992 |

The tiled equality Gram still benefits from the second socket, but the many
tiny 2x2 MPFR block tasks and equality GEMV/triangular tasks suffer launch,
synchronization, and NUMA costs. Solver time is 34.5% higher at 128 workers;
peak RSS also rises from 3,907,204 to 4,346,976 KiB. Using every requested
worker uniformly is therefore rejected.

The next candidate keeps the disjoint equality Gram at the full requested
width but automatically merges ownership-safe block bins into at most 64 task
streams. It applies the cap only to native BigFloat all-local equality
residual, predictor/corrector, direction recovery, complementarity, line
search, accepted updates, equality GEMV, and local triangular work. The 1--64
worker path and all other arithmetic types are unchanged. Every per-block
result remains exclusive, and global sums retain block order. Focused
BigFloat sparse/ownership tests passed 3,092 assertions; the complete suite
passed all 5,751 tests in 5m40.7s. A new immutable 128-core J40 run is required
to determine whether the hybrid schedule retains the 128-core Gram gain while
recovering the 64-core block-phase performance.

Job 196243 performed that gate on node146. It passed all hashes and reproduced
the complete objective and physical certificate bit-for-bit in 158 iterations.
The hybrid schedule improved 128-core solver time from 495.811 to 425.880
seconds (14.1%) and internal time from 356.574 to 286.429 seconds (19.7%).
The major phase changes were:

| Phase | Uniform 128 (s) | Hybrid 128 (s) | Change |
|---|---:|---:|---:|
| KKT factorization | 110.009 | 107.601 | -2.2% |
| Equality Gram | 43.446 | 41.672 | -4.1% |
| Residual and block factor | 47.694 | 39.929 | -16.3% |
| Predictor | 41.885 | 26.852 | -35.9% |
| Corrector | 100.628 | 68.341 | -32.1% |
| Complementarity analysis | 49.665 | 35.754 | -28.0% |
| Fraction-to-boundary line search | 18.934 | 10.605 | -44.0% |
| Accepted update | 7.992 | 4.272 | -46.5% |

Peak RSS fell from 4,346,976 to 4,058,792 KiB (6.6%). The hybrid candidate is
retained because it provides a clear improvement for over-wide requests and
leaves every 1--64 worker route unchanged. It is still 15.5% slower than the
64-worker solver result, so 64 remains the provisional recommendation for
this J40 geometry. A 96-worker crossover run is the final scaling check.

Job 196263 completed that check on node113. It again reproduced every
high-precision and physical certificate field bit-for-bit. Solver time was
398.303 seconds, internal time 265.944 seconds, and `/usr/bin/time` peak RSS
4,147,008 KiB. This is 6.5% faster than hybrid 128 but still 8.0% slower than
64 workers. The measured crossover is therefore unambiguous for J40:

| Requested workers | Fine-grained task cap | Solver (s) | Internal (s) | Equality Gram (s) | Peak RSS (KiB) |
|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 368.704 | 229.729 | 60.580 | 3,907,204 |
| 96 | 64 | 398.303 | 265.944 | 56.632 | 4,147,008 |
| 128, uniform baseline | 128 | 495.811 | 356.574 | 43.446 | 4,346,976 |
| 128, retained hybrid | 64 | 425.880 | 286.429 | 41.672 | 4,058,792 |

Recommend 64 Julia threads, one BLAS thread, and `numactl --interleave=all`
for this model. Wider allocations remain supported and safer with the retained
cap, but should be used only after a larger equality panel demonstrates enough
Gram savings to offset cross-socket scheduling costs.

### A8 final Task_Low08 Float64 regression — job 196277

The final cross-problem gate used the immutable `c9d6514` release and the
archived Task_Low08 input with every input and output hash checked. An initial
32-core reservation (job 196272) terminated on its execution host before the
PBS payload started: it recorded zero payload CPU, zero payload RSS, no driver
output, and no solver artifact. It is classified as infrastructure rather than
a candidate regression. The only retry, job 196277 on node58, reserved 64
cores and ran the established performance configuration with 16 Julia threads,
16 OpenBLAS threads, adaptive parameters, automatic scaling, and automatic
step and parameter policies.

The retry passed. It returned `Optimal` in 28 iterations with no restart or
regularization. The primal and dual objectives were `0.6532912655025964` and
`0.6532910479425099`; relative gap was `2.1756008650175573e-7`, primal
residual `3.315879792964438e-10`, and dual residual
`9.534220635210033e-12`. Equality presolve reduced 482 rows to rank 394 with
dependency residual `3.309223593712139e-16`. The maximum original equality
violation was `3.315879792964438e-10`, minimum primal PSD eigenvalue
`-7.12597888558375e-11`, and minimum dual PSD eigenvalue
`1.9753595193928247e-15`. The aggregate certificate is available and valid,
both PSD gates passed, and its failure list is empty.

Solver time was 33.846 seconds and driver total time was 35.585 seconds.
`/usr/bin/time` measured 94.80 seconds including input, startup, and full
validation, with 4,469,000 KiB peak RSS. The principal timed solver phases
were:

| Phase | Time (s) |
|---|---:|
| Schur assembly | 7.540 |
| KKT total | 9.512 |
| Schur factorization | 5.929 |
| Constraint triangular solve | 1.445 |
| Equality factorization | 0.987 |
| Predictor | 1.555 |
| Corrector | 2.660 |
| Refinement | 0.406 |
| Equilibration | 0.245 |

PBS exited zero, the `PASSED` marker exists, and the recorded SHA-256 manifest
verifies. This closes the required Task_Low08 regression without changing its
Float64 route.

### A8 BigFloat1024 support gate — job 196296

The retained 64-worker candidate was also run at a fixed 1,024-bit working
precision on the same immutable J40 mathematical model. Job 196296 used node141,
64 Julia threads, one BLAS thread, interleaved NUMA allocation, and 64 GiB. It
returned `Optimal` in 157 iterations without restart, regularization,
refinement, or fallback. PBS exited zero, the `PASSED` marker exists, and all
six recorded artifacts verify against their SHA-256 manifest.

The 1,024-bit solve took 553.959 seconds, with 360.413 seconds attributed to
instrumented solver phases. `/usr/bin/time` measured 644.37 seconds end to end
and 4,268,480 KiB peak RSS. The dominant phases were KKT/equality work:

| Phase | BigFloat512, 64 workers (s) | BigFloat1024, 64 workers (s) |
|---|---:|---:|
| Solver | 368.704 | 553.959 |
| Instrumented total | 229.729 | 360.413 |
| KKT factorization | 121.594 | 207.640 |
| Equality Gram | 60.580 | 115.491 |
| Equality factorization | 23.283 | 42.838 |
| Constraint triangular solve | 37.716 | 49.298 |
| Residual and block factor | 31.282 | 51.048 |
| Predictor | 11.042 | 15.102 |
| Corrector | 31.817 | 46.024 |

The two precision runs used different nodes, so the ratios are orientation
rather than a same-node speed claim. The phase ranking is nevertheless clear:
the equality Gram and native BigFloat Cholesky are the first 1,024-bit targets.

The physical certificate passed: original linear residual
`3.8995196507e-63`, off-grid relative residual `1.7504886435e-11`, minimum PSD
eigenvalue `1.7601915718e-34`, and zero disk violation. The primal and dual
solver residuals were `9.6515590158e-308` and `2.7038636179e-287`. The final
physical objective was
`21.02534392475973711278726269083789079642679721718820330510155065579...`
with relative gap `1.8898900297e-13`.

This trajectory is valid but not an accuracy improvement over the 512-bit
reference, whose relative gap was `3.4508398259e-14` and whose physical
objective was `21.02534392476137261213...`. The input coefficients were
precomputed at BigFloat1024 and rounded once to Float64x4 before either solve;
increasing the solver precision cannot restore discarded coefficient digits.
The adaptive path also terminates from measured tolerances rather than trying
to consume every available mantissa bit. BigFloat1024 is therefore confirmed
as supported, but 512 bits remains the recommended fixed precision for this
rounded model. No default precision or termination rule is changed from this
single valid trajectory.

### A9 Julia 1.10 rank-deficient equality compatibility

After commit `dcb84c4` was published to PR #3, every Julia 1.x, macOS,
Windows, documentation, quality, and benchmark-smoke check passed. Both Julia
1.10 Linux jobs failed at the new rank-deficient all-local BigFloat equality
test. The one- and four-thread logs had the same exception:

```text
ArgumentError: generic pivoted Cholesky factorization is not implemented yet
```

The normal-equation Cholesky had already reported insufficient numerical rank.
Automatic mode then computed a pivoted Cholesky only to obtain a diagnostic
rank before unconditionally selecting the existing rank-revealing QR factor.
Julia 1.10 has no generic `BigFloat` implementation for that redundant probe;
newer Julia versions do.

The fix sends automatic mode directly to rank-revealing QR after the rejected
normal-equation factor. The same change is applied to both the all-local arrow
and general dense KKT routes. It preserves explicit
`equality_solver=:normal_equations` semantics, removes one unnecessary
factorization, and does not touch a full-rank solve such as J40. Diagnostics
now record `reason=:normal_equation_rank_loss` with the QR rank and quality.

The BigFloat sparse/ownership plus general correctness suites passed at one and
four threads, including 3,092 exact ownership assertions and eight
rank-deficient equality assertions per run. The complete four-thread suite
then passed all 5,752 tests in 4m38.4s on Julia 1.12.6. The authoritative
Julia 1.10 validation remains the refreshed GitHub Actions matrix.

### A10 publication, CI, and immutable cluster staging

The Julia 1.10 compatibility fix was committed locally as
`ac22a53cf4686bd921ef60d5c2a244b0fd92341b`. Git's HTTPS transport repeatedly
stalled before uploading, so the same six-file tree was published atomically
through GitHub's Git-data API as PR #3 head
`0a71efed779617f5bd57af3359efc7523c9f0188`. Both commits have tree
`f83014ccbb97f2f5ca8234a7b44d184601d9c7b6`; only the commit metadata differs.

GitHub Actions run 30709269815 passed every required job: Julia 1.10 Linux at
one and four threads, current Julia Linux at one and four threads, current
Julia macOS and Windows at four threads, package quality, documentation,
benchmark smoke, and documentation deployment. This confirms that automatic
rank-deficient BigFloat equality fallback no longer calls unsupported generic
pivoted Cholesky on Julia 1.10.

The exact published tree was staged as an immutable cluster release at:

```text
<cluster-root>/SDPX.jl/releases/0a71efed779617f5bd57af3359efc7523c9f0188
```

The source archive SHA-256 is
`c1929eca65aa2bef9a29fafd087a1bf92df27d5657880f3cbb533281a17c5587`.
Its pinned environment was copied from the previously validated release and
only the SDPX source path was changed. Precompilation and a package-load smoke
test passed with SDPX 0.3.0, MultiFloats, and JLD2; the smoke test also verified
that a 128-worker all-local BigFloat request produces the retained 64-stream
fine-grained task cap.

The Codex and Claude `sdpx-cluster` skills were synchronized. Their site
reference now distinguishes the promoted v0.2.1 release from the staged v0.3.0
candidate and records the validated J40 BigFloat512 64/96/128 scaling,
BigFloat1024 support gate, Task_Low08 regression, phase-aware scheduler, and
recommended 64-Julia/one-BLAS-thread launch configuration.

Production `current` still resolves to
`ef1642272c17af63a468ca1adedc7c8735a95157/source`. It was intentionally not
changed because the preceding cluster-regression safety constraint required it
to remain fixed. Promotion therefore requires an explicit authorization; no
solver, queue, or unrelated job state was changed during staging.

## 2026-08-02 — Float64x4 parallel scaling audit

### P1 immutable baseline and measured resource use — job 196495

The medium CSDR J32/K4/Na16/Nmu100 model was benchmarked from the immutable
GitHub PR tree `f861593e6413e9ab487b2018d5caedb341122bd1` on one node with 128
allocated cores. The run used one BLAS thread, interleaved NUMA allocation,
one warm-up solve, and three measured fixed-trajectory solves at each Julia
thread count. Every measured solve returned `Optimal` in 41 iterations. The
relative gap was `9.210e-11`, the dual residual was `3.257e-14`, the minimum
PSD eigenvalue was `2.508e-18`, and cross-thread objective differences were
below `6.1e-38` relative to the one-thread result.

| Julia threads | Median solve (s) | Speedup | Schur (s) | KKT (s) | Peak RSS (GiB) | Process CPU (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.078 | 1.00 | 28.638 | 1.704 | 2.27 | 94 |
| 2 | 28.314 | 1.77 | 14.588 | 2.763 | 2.17 | 137 |
| 4 | 16.845 | 2.97 | 7.428 | 2.737 | 2.23 | 171 |
| 8 | 10.241 | 4.89 | 3.895 | 2.200 | 2.15 | 202 |
| 16 | 6.487 | 7.72 | 2.076 | 1.504 | 2.24 | 240 |
| 32 | 6.286 | 7.97 | 1.940 | 1.497 | 2.34 | 324 |
| 64 | 8.767 | 5.71 | 1.811 | 1.728 | 2.57 | 713 |
| 128 | 45.572 | 1.10 | 4.460 | 3.657 | 3.45 | 2,017 |

The process-level CPU percentage includes warm-up, compilation, model loading,
validation, and serialization, so it is not a solver-only utilization metric.
Per-thread sampling nevertheless confirmed that requested Julia workers were
created and became active: the maximum active compute-thread counts were 1,
3, 5, 9, 17, 33, and 65 for requests through 64. At 128, Julia's default GC
team temporarily made 256 OS threads active. BLAS remained at one thread in
all cases. NUMA sampling showed roughly even memory placement across all eight
nodes under interleaving.

The scaling limit is not an unfulfilled core allocation. The 3,400-by-144
reduced panel has only 78 triangular jobs with the existing twelve-column
tile, while residual, recovery, line-search, and update phases launch many
short per-block tasks. At 64 and 128 threads, task synchronization, GC helper
threads, and cross-NUMA traffic dominate the saved arithmetic. The retained
design candidate therefore keeps all requested workers for panel/SYRK work but
caps only short Float64x4 block phases at 32 task streams. It also records both
effective widths in solve diagnostics.

### P2 reduced-SYRK geometry experiment — job 196511

An allocation-free direct kernel calibration used a deterministic
3,400-by-144 Float64x4 panel and compared column tiles 4, 6, 8, 12, 16, and 24.
Every triangle was exactly equal to the one-thread twelve-column reference.
At 64 threads the median kernel times for tiles 8 and 12 were 0.02354 and
0.03244 seconds, respectively: tile 8 was 27.4% faster because it exposed 171
rather than 78 triangular jobs. At 32 threads the difference was only 1.5%,
and at 128 threads the additional task/NUMA overhead made tile 12 faster.
Six-column tiles were consistently poor because they leave unfavorable SIMD
tails.

The conservative candidate selects the eight-column tile only for 48--95
Float64x4 workers when the default tile would expose fewer than two jobs per
worker. Smaller teams, wider panels, and 96 or more workers retain the existing
configuration. Job 196511 is also measuring a launch-only control using exact
compute-thread counts, one GC mark thread, one BLAS thread, and compact NUMA
placement. No kernel selection or solver code is changed in that control.

### P3 Float64x4 reduced-factor screening

The 144-by-144 shared factorization becomes serially dominant once Schur and
block phases reach their scaling limit. A local eight-thread screening test
compared the current generic Cholesky with the package's existing blocked
fixed-precision factorization on a deterministic SPD Float64x4 matrix of the
same order. Copies were outside the timed region and seven post-warm-up samples
were collected.

| Factorization | Median (ms) | Speedup |
|---|---:|---:|
| Current native | 11.379 | 1.00 |
| Blocked, 1 worker | 10.152 | 1.12 |
| Blocked, 2 workers | 6.989 | 1.63 |
| Blocked, 4 workers | 6.778 | 1.68 |
| Blocked, 8 workers | 6.451 | 1.76 |

The native and blocked reconstruction errors were `9.758e-64` and
`9.781e-64`; their lower factors differed by only `3.248e-65` relatively.
This is promising but not yet a retained solver change: the screening machine
is not the cluster EPYC node, and the test matrix is representative in size
rather than the actual iteration matrix. The next same-node compute job will
repeat this geometry test before any reduced-factor integration is attempted.

### P4 launch, GC, and NUMA control — job 196511

The unchanged `f861593` solver completed the launch-only control on node60.
All artifact hashes passed, all 24 measured solves returned `Optimal` in 41
iterations, BLAS remained single-threaded, and every numerical bound matched
the original baseline: maximum relative objective variation `5.66e-38`, gap
`9.210e-11`, dual residual `3.257e-14`, and minimum PSD eigenvalue
`2.508e-18`. Julia was launched with exactly the requested compute pool, no
interactive pool, and one GC mark thread. CPUs and memory were limited to the
active NUMA subset through 64 threads; 128 threads used all eight nodes.

| Threads | Median solve (s) | Schur (s) | KKT (s) | Allocated (MiB) | Process CPU | Process peak RSS (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.696 | 28.339 | 1.657 | 89.6 | 98% | 2.24 |
| 2 | 25.937 | 14.411 | 1.500 | 88.3 | 135% | 2.26 |
| 4 | 14.323 | 7.265 | 1.410 | 90.5 | 167% | 2.23 |
| 8 | 8.581 | 3.708 | 1.377 | 94.8 | 190% | 2.32 |
| 16 | 5.819 | 1.972 | 1.383 | 103.6 | 209% | 2.33 |
| 32 | 5.017 | 1.416 | 1.396 | 120.8 | 231% | 2.67 |
| 64 | 6.926 | 1.780 | 1.692 | 155.9 | 391% | 2.91 |
| 128 | 45.959 | 4.500 | 3.603 | 223.6 | 1,801% | 3.35 |

`/usr/bin/time` process CPU covers loading, compilation, warm-up, three solves,
validation, and serialization. It demonstrates that allocating a wide pool is
not the same as using it efficiently: the 64-thread process averaged only 3.91
cores over its lifetime, while the 128-thread process averaged 18.01 cores and
incurred 46.4 million voluntary plus 2.09 million involuntary context
switches. The exact pool removed the extra 64/128 GC workers, but did not cure
the solver's 128-task oversubscription. At 64 threads, local elimination and
line search were 4.0x and 4.9x slower than at 32; at 128 they were 71x and
218x slower. This supports phase-specific task widths rather than a global
thread count.

### P5 EPYC reduced-factor test and A/B harness correction — jobs 196521/196523

Job 196521 ran the Cholesky screening on the same node60 EPYC 7742 before the
full solver harness stopped. The 144-by-144 native factor median was 31.967 ms.
The blocked factor took 24.836 ms with one worker (1.29x faster), but parallel
widths were noisy and slower: 40.543 ms at two, 34.436 ms at four, 26.362 ms at
eight, and about 28.5 ms at 16--128. Reconstruction errors remained
`9.758e-64` native and `9.781e-64` blocked, with a `3.248e-65` relative factor
difference. The retained next experiment is therefore a serial blocked
factor, not fine-grained parallel Cholesky, for this small shared system.

The solve A/B in job 196521 did not execute. Its copied runner resolved
`model_io.jl` relative to the campaign subdirectory rather than the benchmark
script directory and exited before loading a model. No result was accepted
from that failed harness. The runner was placed beside its unchanged helpers,
the original runner remained untouched, and corrected 128-core normal-queue
job 196523 was submitted with the same immutable baseline and candidate.

### P6 serial blocked-Cholesky crossover screening

A second local screening sweep varied both matrix order and panel size. The
blocked implementation stayed serial because job 196521 had already rejected
fine-grained parallel factorization on the target EPYC. Five post-warm-up
samples were collected for each deterministic SPD Float64x4 matrix. The best
observed panels were deliberately small:

| Order | Best panel | Native (ms) | Blocked (ms) | Speedup |
|---:|---:|---:|---:|---:|
| 64 | 16 | 1.021 | 0.949 | 1.08 |
| 96 | 16 | 3.466 | 2.784 | 1.25 |
| 128 | 16 | 8.093 | 6.063 | 1.33 |
| 144 | 16 | 11.523 | 8.362 | 1.38 |
| 160 | 24 | 17.630 | 11.188 | 1.58 |
| 192 | 16 | 31.230 | 18.425 | 1.69 |
| 256 | 24 | 67.779 | 41.806 | 1.62 |

Every reconstruction error remained near `1e-63`. This sweep is not yet a
cluster crossover rule: native medians at 160--256 showed some local jitter,
and the EPYC cache hierarchy differs. The same sweep script was staged for a
compute-node confirmation before choosing a conservative dimension and panel
threshold.

### P7 same-node phase-aware A/B and effective-worker correction — job 196523

The immutable baseline and phase-aware candidate ran back-to-back on node60
with exact compute pools, one GC mark thread, one BLAS thread, controlled NUMA
placement, one warm-up, and three measured solves. Artifact hashes passed and
all 18 solves returned `Optimal` in 41 iterations. The maximum relative gap
was `9.211e-11`, maximum dual residual was `3.258e-14`, and every reported
minimum PSD eigenvalue was positive (`2.508e-18`). The candidate's objectives
and certificates agreed with the baseline to the available Float64x4 output
precision.

| Threads | Baseline median (s) | Candidate median (s) | Baseline Schur (s) | Candidate Schur (s) | Effective block/SYRK tasks |
|---:|---:|---:|---:|---:|---:|
| 32 | 5.332 | 5.345 | 1.606 | 1.622 | 32 / 32 |
| 64 | 6.822 | 6.878 | 1.791 | 1.353 | 32 / 64 |
| 128 | 45.303 | 15.635 | 4.387 | 3.087 | 32 / 64 reported |

At 64 threads the eight-column triangular tile reduced the measured Schur
time by 24.5%, allocations by about 3.3%, and process voluntary context
switches from 1.19 million to 1.01 million. End-to-end time was 0.8% slower,
which is within the run-to-run phase noise and is not accepted as a standalone
solver speedup. At 128 threads, limiting short block phases reduced total time
by 2.90x and preserved the exact solve trajectory, but it remained much slower
than the 32-thread configuration.

Interval-level `pidstat` data confirmed that the requested pools existed and
became active. The 64-thread candidate reached 64 simultaneously active Julia
workers and 2,080% interval CPU, but averaged only 343% CPU across loading,
warm-up, solves, validation, and output. It used 3.01 GiB peak RSS and one BLAS
thread. The wide pool is therefore genuinely allocated and briefly used, but
the solver geometry cannot sustain it: the 144-column triangle, serial shared
factor, and synchronization between short phases limit useful parallelism.

Inspection after the 128-thread run found an instrumentation/execution
discrepancy: `schur_threads=64` was selected and reported, and panel packing
used 64 tasks, but the final exact SYRK still received `ws.thread_count` (128)
instead of the selected worker count. This explains why its Schur phase still
took about three seconds. The call now receives the actual selected `workers`,
so diagnostics and execution agree. Focused extended-BLAS (67 assertions) and
thread-safety (38 assertions) suites pass after the correction.

The next same-node job, `196530.node220`, compares the corrected candidate
against the phase-aware baseline with five measured solves per point. It also
validates a serial sixteen-column blocked Float64x4 Cholesky for 128--192 order
reduced systems and records interval CPU use, active worker counts, process
RSS, affinity, NUMA placement, wall time, and context switches automatically.

### P8 serial blocked reduced factor and zero-allocation hot path — job 196530

The EPYC crossover sweep confirmed that the serial sixteen-column blocked
factor is faster throughout the tested fixed-precision range. All lower-factor
reconstruction errors remained between `5.8e-64` and `1.1e-63`.

| Order | Native (ms) | Blocked-16 (ms) | Speedup |
|---:|---:|---:|---:|
| 64 | 2.871 | 2.100 | 1.37 |
| 96 | 9.547 | 5.852 | 1.63 |
| 128 | 22.487 | 12.431 | 1.81 |
| 144 | 31.947 | 16.902 | 1.89 |
| 160 | 43.731 | 22.465 | 1.95 |
| 192 | 75.468 | 36.717 | 2.06 |
| 256 | 178.666 | 80.899 | 2.21 |

The retained crossover is deliberately narrower than the whole measured
range: Float64x4 reduced-arrow systems of order 128--256 use the serial
blocked factor, while smaller and larger systems keep the native method. This
leaves latency-dominated tiny systems alone and avoids extrapolating beyond
the measured cache regime. A Boolean internal kernel now lets the direct
reduced path avoid constructing the mixed-precision factor wrapper; the hot
factor call allocates zero bytes locally (versus 16 bytes in the first
candidate). The wrapper-returning interface remains unchanged for existing
mixed-precision callers.

Job 196530 compared the phase-aware baseline and factor candidate five times
at each width. All 30 solve results were `Optimal` in 41 iterations. The
maximum relative gap was `9.211e-11`, the maximum dual residual was
`3.258e-14`, and all minimum PSD eigenvalues were `2.508e-18`. The changed
factor summation order moved the reported objective only at roughly the 38th
relative decimal digit and did not change any certificate decision.

| Requested threads | Baseline solve median (s) | Factor candidate median (s) | Speedup | Baseline factor (s) | Candidate factor (s) |
|---:|---:|---:|---:|---:|---:|
| 32 | 5.352 | 5.149 | 1.039 | 1.357 | 0.742 |
| 64 | 6.748 | 6.504 | 1.037 | 1.613 | 0.953 |
| 128 | 15.533 | 15.609 | 0.995 | 1.398 | 0.852 |

At 32 and 64 threads the factor phase improved by 45.3% and 40.9%, producing
repeatable end-to-end gains of 3.8% and 3.6%. At 128 threads, both factor and
Schur became faster and voluntary context switches fell from 20.6 million to
19.0 million, but residual, predictor, corrector, line-search, and update
overhead erased the savings. The candidate averaged 11.6 active cores over
its full process lifetime, briefly reached all 128 workers, used one BLAS
thread, and peaked at 3.69 GiB RSS. This is further evidence that 128 requested
threads are inappropriate for a 144-column reduced system; the 32-thread
launch remains the fastest validated configuration.

The actual-worker correction made execution agree with the reported
`schur_threads=64`, but did not cure wide-pool task migration: the 128-thread
Schur median remained near three seconds, versus about 1.6 seconds in the
32-thread pool. It is retained as a truthful resource bound, not claimed as an
end-to-end speedup. The next experiment restores 64 fine-grained tasks for a
64-thread request, where narrowing to 32 was neutral, and separately tests
NUMA interleaving and Julia's exclusive thread affinity.

### P9 task width, SIMD reuse, affinity, and NUMA controls — job 196534

Job 196534 completed on the same 128-core node60 EPYC 7742 with PBS exit
status zero, a `PASSED` marker, and valid SHA-256 checksums for all 38 measured
solutions. Every solve returned `Optimal` in 41 iterations. The maximum
relative gap was `9.211e-11`, maximum dual residual was `3.258e-14`, and the
minimum reported PSD eigenvalue was `2.508e-18`. Across every configuration,
the recorded primal objective varied by only `1.01e-40` in absolute terms
(`1.58e-29` relative to the small objective), far below any certificate gate.

The first A/B restored 64 fine-grained block tasks for a 64-thread request
while keeping the reduced Schur at 64 tasks. Seven post-warm-up solves gave:

| 64-thread configuration | Solve median (s) | Schur (s) | Reduced factor (s) | Residual (s) | Predictor (s) | Corrector (s) |
|---|---:|---:|---:|---:|---:|---:|
| 32 fine tasks | 6.654 | 1.417 | 0.955 | 0.572 | 0.977 | 1.208 |
| 64 fine tasks | 5.981 | 1.374 | 0.857 | 0.469 | 0.865 | 0.943 |

Restoring the full 64 tasks improved the end-to-end solve by 10.1%. The
benefit appears in every short block-local phase, so the earlier 64-to-32 cap
was removed. The cap remains at 96 or more requested workers, where the
128-thread A/B has already demonstrated a synchronization collapse. The
additional task partials increased solve allocation from 150.7 to 164.3 MiB,
but process peak RSS fell slightly within run noise (3.03 to 2.97 GiB).

An exact direct-kernel screen then compared the existing four-output-row SIMD
microkernel with a two-accumulator eight-row kernel. Both retained the scalar
reduction order in every lane and produced an exactly equal lower triangle.

| Workers / column tile | Vec4 median (s) | Vec8 median (s) | Vec8 speedup |
|---:|---:|---:|---:|
| 1 / 12 | 0.61118 | 0.55182 | 1.108x |
| 32 / 12 | 0.05526 | 0.05185 | 1.066x |
| 64 / 8 | 0.04035 | 0.03419 | 1.180x |

The vec8 kernel reuses one broadcast multiplier across two independent
`MultiFloatVec` accumulators. It does not change per-lane summation order,
output ownership, or hot-loop allocation. It is retained for a solver-level
A/B in the next compute job.

NUMA interleaving was not beneficial. Binding 64 cores and allocations to
NUMA nodes 0--3 solved in 5.981 seconds, while forced interleaving over those
nodes took 6.304 seconds (5.4% slower). `numastat` confirmed that interleaving
did distribute about 2.9 GiB evenly across the four nodes; the ordinary bound
run concentrated memory on nodes 0 and 1, but locality still won. Interleaving
is therefore rejected for this workload.

Julia's exclusive worker affinity was the largest launch-level improvement.
With otherwise identical code and memory binding it reduced the 64-thread
median from 5.981 to 4.791 seconds (19.9%). Per-thread samples showed that 56
of 62 observed workers migrated without exclusive affinity, visiting 9.32
CPUs each on average and up to 27. With `JULIA_EXCLUSIVE=1`, no sampled worker
migrated: each stayed on one CPU. Voluntary context switches fell from 2.45
million to 0.72 million, and process peak RSS remained about 2.97 GiB. The
process had 65 OS threads, reached 64 active Julia workers, and BLAS stayed at
one thread, so the speedup is genuine affinity/cache behavior rather than an
unrequested BLAS team.

Exclusive affinity also improved the 128-thread median from 15.595 to 12.645
seconds, but that remains 2.64 times slower than 64 exclusive threads. The
128-thread run reached all 128 workers yet averaged only about 12.4 cores over
the complete process and incurred 18.7 million voluntary context switches.
The 144-column reduced triangle and short 1,700-block phases cannot sustain a
128-worker team. This point is retained as a measured scaling limit, not a
recommended launch configuration.

Normal-queue job 196535 now brackets the vec8 reference around independent
seven-run tests of contiguous versus LPT block ownership at 32 and 64
exclusive threads. It also compares 48 versus 64 fine-grained tasks at 64
threads. The contiguous policy is restricted to uniform reduced-arrow
Float64x4 systems; heterogeneous and Float64 problems retain LPT. It will be
removed unless the same-node solve A/B shows a reproducible benefit.

### P10 measured locality crossover — job 196535

Job 196535 completed with PBS exit status zero, a `PASSED` marker, and valid
checksums for all 40 measured solutions. Every configuration remained
`Optimal` in 41 iterations; maximum relative gap was `9.211e-11`, maximum
dual residual was `3.258e-14`, and every minimum PSD eigenvalue was
`2.508e-18`. The primal objective span across all scheduler variants was
`3.23e-40` (`5.07e-29` relative), and no validation gate changed.

All processes used exclusive Julia affinity, one BLAS thread, exact compute
pools, and socket-local memory binding. The 64-thread LPT reference was run
both before and after the candidates; its combined twelve-solve median removes
most node drift.

| Pool / fine tasks / partition | Solve median (s) | Schur (s) | Factor (s) | Allocated (MiB) | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|
| 32 / 32 / LPT | 5.010 | 1.795 | 0.676 | 128.9 | 2.84 |
| 32 / 32 / contiguous | **4.555** | **1.504** | 0.678 | 128.9 | **2.75** |
| 64 / 64 / LPT, bracketed | 4.789 | 1.258 | 0.658 | 164.3 | 2.88--2.95 |
| 64 / 64 / contiguous | 4.866 | 1.254 | 0.658 | 164.2 | 2.91 |
| 64 / 48 / contiguous | 4.939 | 1.243 | 0.659 | 157.8 | 2.87 |

Contiguous ownership improves the 32-task solve by 9.1%, reduces process peak
memory by about 75 MiB, and lowers both voluntary and involuntary context
switches. The blocks are uniform `2x2` cells, so no LPT balance is lost and
each task traverses adjacent block objects and coefficient storage.

At 64 tasks the finer Schur time is already saturated. Contiguous ownership
was 1.6% slower than the bracketed LPT reference, and lowering the fine phase
to 48 tasks was 3.1% slower: both lost time in residual, predictor, and
corrector phases despite slightly faster Schur assembly. These alternatives
are rejected. The retained automatic rule therefore selects contiguous bins
only for uniform Float64x4 reduced-arrow systems with at least 256 blocks and
at most 32 actual fine-grained tasks. Wider teams retain LPT. The generic
Float64 path and heterogeneous block systems remain unchanged.

The next isolated candidate skips panel and coupling clears only when a block
contains exactly one local variable plus every shared variable. Such a block
overwrites every destination unconditionally; partial shared coverage keeps
the original clear. Sentinel-based full- and partial-coverage regressions
pass. Job 196536 measures whether removing roughly one GiB of redundant
Float64x4 writes per solve has a stable end-to-end benefit.

### P11 full-coverage clear-elision screen and warm-up audit — job 196536

Job 196536 completed on node60 with PBS exit status zero, a `PASSED` marker,
and valid checksums for all 33 measured solutions. Every result was `Optimal`
in 41 iterations. The maximum relative gap was `9.211e-11`, maximum primal
and dual residuals were `2.31e-62` and `3.258e-14`, and the minimum PSD
eigenvalue was `2.508e-18`. The primal objective span was `2.31e-40`, so the
candidate did not change a certificate decision or meaningful result digit.

The initial seven-run medians appeared favorable:

| Pool | Clear baseline (s) | Clear-elision candidate (s) | Apparent gain | Baseline Schur (s) | Candidate Schur (s) |
|---:|---:|---:|---:|---:|---:|
| 32 | 4.928 | 4.741 | 3.8% | 1.719 | 1.516 |
| 64 | 4.865 | 4.663 | 4.1% | 1.252 | 1.247 |

The bracketed 64-thread repeat showed why this is not yet sufficient evidence.
With one warm-up, the first two measured baseline solves contained about
0.25 seconds of uninstrumented compilation or GC latency. Later baseline
samples converged to 4.645--4.701 seconds, the same range as the candidate's
4.639--4.697 seconds. The repeated baseline had the same two-sample transient
before settling, while its Schur median remained 1.254 seconds. The measured
64-thread kernel effect is therefore neutral, and the apparent end-to-end
gain is a protocol artifact at that width.

Resource telemetry remained internally consistent: the 32-thread processes
created 33 OS threads, pinned one Julia worker to each requested CPU, reached
32 simultaneously active workers, kept BLAS at one thread, and peaked at
2.75--2.85 GiB RSS. The 64-thread processes reached all 64 workers, averaged
about 5.3 CPU cores over complete process lifetime, and peaked near 2.9 GiB.

The first confirmation attempt exposed a campaign-driver defect before its
output was used: `--warmup` was parsed as the Boolean expression
`parse(Int, value) == 1`, so requesting three warm-ups silently disabled them.
Job 196537 was cancelled after its first compilation-contaminated row. The
versioned v2 runner now accepts a nonnegative integer and logs every warm-up.
Job 196538 then failed in six CPU-seconds because the versioned runner had
been staged away from its relative `model_io.jl` dependency; no model or solve
ran. Both original files remain unchanged, and the corrected versioned runner
now sits beside the loader.

Normal-queue job 196539 repeats only the fastest 32-thread point with three
verified full-model warm-ups, five measured solves, and alternating
baseline/candidate/baseline/candidate process order. The source snapshots are
immutable and differ in the hot path only by full-coverage clear elision. The
change will be retained only if this stricter same-node confirmation produces
a stable gain.

A separate local screen tested static Julia-thread ownership for the
3,400-by-144 reduced SYRK. The lower triangle was bit-for-bit identical, but
the current dynamic task mapping took 106.406 ms versus 106.419 ms for the
static mapping (0.9999x), with essentially identical timing ranges. Static
mapping is therefore rejected without a cluster experiment or source change.

Job 196539 completed with PBS exit status zero, a `PASSED` marker, and valid
checksums for all 20 measured solution files. Each of the four source
processes executed three full-model warm-ups before five recorded solves. The
two baseline process medians were 4.631 and 4.627 seconds; the two
clear-elision medians were 4.389 and 4.580 seconds, so both alternating pairs
improved in the same direction despite the node's Schur timing spread.

| Combined ten-sample metric | Clear baseline | Clear elision | Improvement |
|---|---:|---:|---:|
| Solve time | 4.629 s | 4.442 s | 4.05% |
| Schur time | 1.542 s | 1.411 s | 8.46% |
| Reduced factor | 0.678 s | 0.678 s | neutral |
| Solve allocation | 128.85 MiB | 128.85 MiB | neutral |
| Per-solve peak memory | 3.105 GiB | 3.113 GiB | within noise |

Every solve was `Optimal` in 41 iterations. The primal objective was identical
across all rows to all 77 printed decimal digits; maximum relative gap was
`9.211e-11`, maximum dual residual was `3.258e-14`, and minimum PSD
eigenvalue was `2.508e-18`. Each process created 33 OS threads, pinned one
Julia worker per requested CPU, reached all 32 active workers, and used one
BLAS thread. Mean whole-process CPU use was 3.16--3.22 cores because model
loading, compilation, the serial reduced factor, and synchronization-bound
phases cannot occupy the full team continuously. Process peak RSS was
3.10--3.11 GiB.

Full-coverage clear elision is therefore retained. Partial shared coverage
continues to clear both panel rows and the coupling row, and sentinel-based
regressions cover both cases. The complete local package suite passes all
5,791 assertions in 3 minutes 57 seconds on eight Julia threads.

### P12 cached panel reciprocals and SIMD panel rows — local screen

The retained blocked Float64x4 Cholesky still divided every panel and
trailing-panel entry by the same pivot. A controlled Apple M4 microbenchmark
replaced those repeated extended-precision divisions with one reciprocal per
pivot and multiplication thereafter. Median factorization results, with 24
alternating measurements per implementation, were:

| Order | Repeated division (ms) | Cached reciprocal (ms) | Speedup |
|---:|---:|---:|---:|
| 128 | 5.857 | 5.169 | 1.133x |
| 144 | 8.056 | 7.186 | 1.121x |
| 192 | 17.639 | 16.133 | 1.093x |
| 256 | 39.733 | 37.033 | 1.073x |

Both methods had the same Float64-visible reconstruction error
(`9.6e-17`--`1.2e-16`). Low limbs need not be bit-identical because one
rounded reciprocal is reused, so solver-level certificates remain the
retention gate. The implementation stores reciprocals in the undefined upper
triangle of the in-place lower factor. This avoids a per-factor allocation;
threaded row owners only read that scratch after the serial panel factor has
completed. The zero-allocation factor regression and the 77 extended-BLAS
regressions pass, and the independent extended-precision BLAS suite passes
all 72 assertions. A same-node full-model A/B is required before retention.

Vectorizing independent panel rows in one or two four-lane
`MultiFloatVec{4,Float64,4}` groups improved the reciprocal version further.
Against the original repeated-division implementation, the final local
one-worker medians were 4.280, 6.036, 14.067, and 33.347 milliseconds at
orders 128, 144, 192, and 256: speedups of 1.368x, 1.332x, 1.253x, and
1.191x. Every SIMD lane preserves the scalar reduction order. The strict
Float64x4 reconstruction regression remains below `1e-60`, the lower-factor
hot path still allocates zero bytes, and both focused suites pass.

The SIMD balance also makes bounded trailing-update parallelism plausible.
One warmed Apple run at order 144 measured 6.05, 3.87, 3.24, and 2.64
milliseconds for 1/2/4/8 requested workers. A shorter standalone launch was
less stable and measured 5.98, 3.58, and 4.37 milliseconds at 1/2/4 exclusive
workers. This variation makes a local hard-coded optimum unsafe. The
experimental controller therefore exposes the executed factor width,
restricts the 128--256 range to at most eight workers, and leaves all other
dimensions and arithmetic types serial. A cluster micro-sweep and alternating
full-model A/B will decide whether the cap should be one, two, four, or eight.

A final local variant changed each panel solve from row-group-major traversal
to column-major traversal while preserving row ownership and SIMD arithmetic.
The measured speedups were only 1.002x, 1.006x, 1.003x, and 1.002x at
1/2/4/8 workers. This is below the retention threshold and would duplicate a
large specialized loop, so the variant is rejected without a source change.

Reducing the threaded panel-solve row grain from 64 to 32 improved the local
eight-worker factor from 2.658 to 2.607 milliseconds (1.020x). Because this
would save only about 0.3% of the measured full solve even if it transferred
perfectly, it is also rejected pending stronger cluster evidence; the current
64-row threshold remains unchanged.

After adding the bounded factor-width selector and executed diagnostics, the
complete eight-thread local package suite passes all 5,797 assertions in
3 minutes 57 seconds.

### P13 final Float64x4 scaling and resource audit — job 196540

Normal-queue job 196540 completed on node60 with PBS exit status zero and a
`PASSED` marker. All ten CSV files and 68 measured solution files pass their
recorded SHA-256 checksums. Each process used one BLAS thread, an exact Julia
compute pool, one GC mark thread, no interactive threads, exclusive worker
affinity, and explicit NUMA placement. Three complete model warm-ups preceded
every seven-run point; the 128-worker point used five measured solves.

| Julia workers | Solve median (s) | Speedup | Efficiency | Schur (s) | Factor (s) | Allocation (MiB) | Process peak RSS (GiB) | Whole-process mean cores | Maximum active workers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.293 | 1.00x | 100.0% | 24.532 | 0.656 | 89.5 | 2.380 | 1.00 | 1 |
| 2 | 23.382 | 1.89x | 94.7% | 12.532 | 0.656 | 96.1 | 2.303 | 1.52 | 2 |
| 4 | 12.969 | 3.42x | 85.4% | 6.601 | 0.657 | 98.4 | 2.346 | 2.11 | 4 |
| 8 | 7.526 | 5.88x | 73.6% | 3.358 | 0.659 | 102.7 | 2.546 | 2.62 | 8 |
| 16 | 4.880 | 9.08x | 56.7% | 1.746 | 0.671 | 111.5 | 2.834 | 3.01 | 16 |
| 32 | 4.709 | 9.41x | 29.4% | 1.591 | 0.676 | 128.9 | 3.120 | 3.62 | 32 |
| 64 | **4.614** | **9.60x** | 15.0% | **1.244** | 0.658 | 164.3 | 3.227 | 6.09 | 64 |
| 128 | 12.444 | 3.56x | 2.8% | 2.334 | 0.657 | 205.0 | 3.324 | 13.53 | 128 |

The 128-worker process genuinely created 129 OS threads, pinned one Julia
worker per CPU, reached all 128 workers, and used one BLAS thread. It still
averaged only 13.53 cores over its complete lifetime, peaked at 44.18 CPU
cores in a sampled interval, and incurred 24.89 million voluntary context
switches. Allocation rose to 205 MiB per solve. The 144-column triangle and
short phases therefore cannot amortize that width; the result is a measured
synchronization/NUMA limit rather than unused scheduler allocation.

All 54 scaling rows were `Optimal` in 41 iterations. The primal-objective
span was `1.46e-39`, maximum relative gap was `9.211e-11`, maximum primal and
dual residuals were `2.61e-62` and `3.258e-14`, and the minimum PSD
eigenvalue was `2.508e-18`. The requested arithmetic was Float64x4 (209
reported bits) at every point.

The repeat 32-worker process had a 4.508-second median; combining the two
seven-run 32-worker processes gives 4.549 seconds. The immutable pre-
optimization baseline at 32 workers took 4.939 seconds, while its reduced
factor took 1.362 seconds versus 0.677 seconds in the retained candidate. The
campaign confirms a 7.9% cumulative solve reduction and a 50.3% factor
reduction relative to that baseline, although the earlier alternating P11
experiment remains the cleaner isolation of clear elision.

The remaining pre-P12 factor is independent of solver width and consumes
0.66--0.68 seconds. Job 196541 therefore runs a same-node micro-sweep and four
alternating full-model processes to decide the cached-reciprocal, SIMD-row,
and bounded factor-team candidate.

### P14 cached/SIMD/bounded factor retention -- job 196541

Normal-queue job 196541 completed on node57 with PBS exit status zero, a
`PASSED` marker, and valid SHA-256 checksums for four CSV files and all 20
measured solutions. The immutable source comparison changed only the
lower-only blocked factor, its worker selector, and executed diagnostics; no
Schur assembly formula, model input, tolerance, or iteration policy changed.
Every process used three full-model warm-ups followed by five measurements,
and the order was baseline/candidate/baseline/candidate.

The isolated order-144 factor sweep established the architecture-specific
crossover before the full solve:

| Factor configuration | Effective workers | Median (ms) | Allocation/call | Float64-visible relative error |
|---|---:|---:|---:|---:|
| Previous blocked factor | 1 | 15.891 | 0 B | `1.218e-16` |
| Cached reciprocal + SIMD rows | 1 | 10.253 | 0 B | `1.218e-16` |
| New factor, request 2 | 2 | 14.228 | 15,008 B | `1.218e-16` |
| New factor, request 4 | 4 | 9.396 | 22,864 B | `1.218e-16` |
| New factor, request 8 | 8 | **6.890** | 35,184 B | `1.218e-16` |
| New factor, request 16/32 | capped at 8 | 6.889/6.888 | 35,184 B | `1.218e-16` |

The two-worker point is slower than the new serial kernel on this EPYC node,
but four and eight workers are beneficial. Eight is the measured optimum and
requests above eight reproduce it, so the automatic selector retains a hard
eight-worker maximum only for 128--256 order Float64x4 reduced-arrow factors.
Other dimensions and arithmetic types retain their established path. The
strict Float64x4 reconstruction error remained about `1.01e-63`; the looser
Float64 projection shown in the table is unchanged. The temporary task
allocation is bounded and accounts for the full-model allocation delta.

| Five-run process median | Previous factor A | New factor B | Previous factor A2 | New factor B2 |
|---|---:|---:|---:|---:|
| Solve (s) | 4.710 | **3.422** | 4.793 | **3.418** |
| Schur (s) | 1.606 | 1.145 | 1.676 | 1.147 |
| Factor (s) | 0.672 | **0.139** | 0.671 | **0.140** |
| Allocation (MiB/solve) | 128.85 | 130.23 | 128.85 | 130.23 |
| Process peak RSS (GiB) | 3.009 | 3.013 | 3.120 | 2.779 |

Combining ten samples per implementation gives a 4.768-to-3.420 second solve
reduction (28.3%) and a 0.671-to-0.140 second factor reduction (79.2%). The
candidate's factor width was reported as eight on every row. Several other
phases also ran faster during both alternating candidate processes; those
co-movements are reported rather than attributed to the factor formula. The
isolated factor measurement and the 0.53-second full-solve factor reduction
alone are sufficient for retention.

All 20 solves were `Optimal` in 41 iterations. The cross-source primal
objective span was `8.44e-40`; maximum relative gap was `9.211e-11`, maximum
dual residual was `3.258e-14`, and minimum PSD eigenvalue was `2.508e-18`.
The reciprocal/SIMD reduction order changes only sub-leading Float64x4 limbs
and did not change a status, tolerance gate, or certificate.

Resource records confirm that each process created 33 OS threads, pinned one
Julia compute worker to each CPU 0--31 under exclusive affinity, reached all
32 active workers, and kept every BLAS backend at one thread. Whole-process
CPU averaged 3.14--3.28 cores because model loading and many short phases are
serial or synchronization limited; the candidate reached about 22 sampled
cores versus 17 for the previous factor. NUMA policy bound allocation to
nodes 0 and 1, with first-touch placement concentrated on node 0. Candidate
process peak RSS was 2.80--3.01 GiB versus 3.01--3.12 GiB for the baseline.
Voluntary context switches increased from roughly 215 thousand to 294
thousand due to the bounded factor tasks, but the factor and total runtime
improvements substantially outweigh that overhead.

The candidate is retained. The next controlled experiment, job 196542,
compares the true 128-worker path against a transparent 64-worker whole-solver
cap for this narrow 1,700-block/144-shared-variable geometry. It includes 64-
worker controls so the cap is retained only if it removes the measured 128-
worker synchronization and NUMA collapse without harming the genuine 64-
worker optimum.

### P15 rejected first whole-solver cap -- job 196542

Job 196542 completed on node60 with all six processes returning zero, a
`PASSED` marker, and valid SHA-256 checksums for six CSV files and 30 measured
solutions. Every solve remained `Optimal` in 41 iterations. Across all rows,
the primal-objective span was `1.40e-40`, maximum relative gap was
`9.211e-11`, maximum dual residual was `3.258e-14`, and the minimum PSD
eigenvalue was `2.508e-18`. The candidate is rejected solely for performance,
not correctness.

The first implementation changed `Workspace.thread_count` from 128 to the
64-worker reduced-SYRK cap before constructing all per-phase schedules. This
saved task-local storage but accidentally asked the fine-grained selector to
interpret the already-capped width as a genuine 64-worker request. The
previous 128-worker path intentionally used 32 short-phase bins; the candidate
expanded them to 64 and changed their uniform-block partition away from the
validated 32-bin contiguous schedule.

| Combined ten-sample 128-worker metric | Previous factor source | First global cap | Change |
|---|---:|---:|---:|
| Solve (s) | 11.997 | 18.261 | **52.2% slower** |
| Schur (s) | 2.249 | 2.273 | neutral |
| Factor (s) | 0.738 | 0.750 | neutral |
| Residual (s) | 1.114 | 1.840 | 65.2% slower |
| Predictor (s) | 2.040 | 3.575 | 75.2% slower |
| Corrector (s) | 2.235 | 3.739 | 67.3% slower |
| Allocation (MiB/solve) | 206.39 | 179.15 | 13.2% lower |
| Peak result RSS median (GiB) | 3.430 | 3.257 | 5.0% lower |

The genuine 64-worker controls were effectively neutral: 3.725 seconds for
the source baseline and 3.640 seconds for the candidate, with overlapping
3.61--3.81 second ranges and nearly identical phase timings. The extra shared-
variable frequency scan does not materially penalize a non-capped solve.

Resource telemetry confirms that the regression was real work rather than an
idle allocation. Both 128-worker variants created 129 OS threads, pinned one
worker per CPU, reached all 128 workers, and used one BLAS thread. The first
cap averaged about 21.0 process cores versus 15.2 for the source baseline, but
voluntary context switches rose from roughly 27.4 million to 48.6 million and
wall time rose from 4:41 to 5:30 per process. More active threads therefore
made the synchronization-bound phases slower.

The first cap is rejected. The corrected selector keeps two widths: the
original requested pool chooses the phase-aware bin count, while the reduced
solver width is a hard upper bound on actual tasks and task-local storage. For
128 requested/64 effective workers this restores 32 contiguous fine-grained
bins. Focused tests pass 87/87, including a regression for this composed
crossover. Normal-queue job 196543 repeats the same alternating 128/64 A/B on
the corrected immutable source.

### P16 paired-column Float64x4 SYRK micro-tile -- rejected local screen

The retained eight-output SYRK micro-kernel computes two independent
four-lane row groups for one Gram column. A local experiment instead computed
one four-lane row group for two adjacent columns, reusing the four panel
streams across both multipliers. Every output retained its original reduction
order and the focused extended-BLAS suite passed all 87 assertions.

On the 3,400-by-144 reduced-panel geometry, separate warmed Apple M4 screens
measured 0.3542/0.1052/0.0692 seconds for the retained kernel and
0.3522/0.1030/0.0683 seconds for the paired-column candidate at 1/4/8 workers.
The apparent gains were only 0.6%, 2.1%, and 1.3%, respectively, below the
retention threshold for a substantially more complicated triangular edge
path and not established by an alternating comparison. Extended arithmetic,
rather than panel bandwidth, dominates this kernel. The experiment is
rejected and the source was restored to the simpler retained implementation.

### P17 corrected whole-solver cap retained -- job 196543

Normal-queue job 196543 completed on node61 with PBS exit state `C`, zero
exit status from every child process, a `PASSED` marker, and valid SHA-256
checksums for six CSV files and all 30 measured solutions. The corrected
candidate retains the 32 contiguous short-phase bins selected from the
original 128-worker request while limiting effective workspace and arrow
width to 64. The process order was baseline/candidate at 128 workers, a
baseline/candidate 64-worker control, then candidate/baseline at 128 workers.
Every process performed three complete warm-ups before five recorded solves.

| Pair median | 128 baseline (s) | 128 capped (s) | Change |
|---|---:|---:|---:|
| Forward order | 12.267 | 11.862 | 3.30% faster |
| Reverse order | 11.848 | 11.506 | 2.89% faster |
| Combined ten samples | 12.216 | 11.839 | 3.09% faster |

The genuine 64-worker control was neutral: 3.6305 seconds for the source and
3.6308 seconds for the candidate. Combined 128-worker phase medians changed
from 2.267 to 2.233 seconds for Schur, 1.119 to 1.105 seconds for residuals,
2.062 to 2.016 seconds for predictor work, and 2.271 to 2.208 seconds for
corrector work. The factor was unchanged at 0.743 seconds because this source
still requested its eight factor tasks from the 128-thread Julia pool.

Solve allocation fell from a combined 216.41 MB to 166.54 MB (23.0%), while
the median result peak changed from 3.518 to 3.503 GB. The two candidate
processes averaged 15.2--15.3 CPU cores, reached 128 active Julia workers,
and incurred 27.4--27.5 million voluntary context switches. The cap controls
solver work and task-local storage; it cannot shrink the Julia runtime's
already-created 128-thread pool. That distinction is now exposed as requested
versus effective thread diagnostics.

All 30 solves were `Optimal` in 41 iterations. There were only two printed
primal-objective strings, corresponding to the expected 64- versus 128-worker
reduction order; their difference is about `1.40e-40`. Maximum relative gap
was `9.211e-11`, maximum primal and dual residuals were `2.205e-62` and
`3.257e-14`, and the minimum PSD eigenvalue was `2.508e-18`. The corrected
whole-solver cap is retained because both alternating pairs improve, the
64-worker control is unchanged, allocation falls materially, and no numerical
gate changes.

### P18 Julia-pool sensitivity of factor and SYRK -- jobs 196544/196545

Two chained 128-core normal-queue jobs isolated the effect of the Julia pool
size from the number of numerical tasks. Both completed with `PASSED` markers
and valid checksums. All factor reconstructions had the same
`1.218e-16` Float64-visible relative error; every SYRK lower triangle was
bit-identical to the serial reference.

For the order-144 Float64x4 reduced factor, 51-run medians in milliseconds
were:

| Julia pool | Serial | 4 factor tasks | 8 factor tasks | Best |
|---:|---:|---:|---:|---:|
| 32 | 10.266 | 8.869 | **6.556** | 8 tasks |
| 64 | 10.243 | 8.441 | **6.118** | 8 tasks |
| 96 | **10.248** | 12.031 | 13.946 | serial |
| 128 | **10.239** | 14.855 | 19.029 | serial |

The work is identical, but small spawned teams migrate across a much wider
scheduler and NUMA domain at 96--128 runtime threads. The 128-pool process
incurred 773,346 voluntary context switches versus 66,720 in the 32-pool
process. This result motivated a complete-solver A/B before changing the
factor selector; isolated kernel timing was not accepted as sufficient.

The 3,400-by-144 SYRK sweep reached its best median at the full useful width
within each pool: 0.02205 seconds with 32 tasks in a 32-thread pool, 0.01383
seconds with 64 tasks in a 64-thread pool, 0.02529 seconds with 64 tasks in a
96-thread pool, and 0.03030 seconds with 64 tasks in a 128-thread pool. The
existing 64-task wide-pool cap is therefore correct; reducing it to 32 or 48
would lose useful tile parallelism. The two-fold 64-pool to 128-pool slowdown
with the same 64 tasks confirms that runtime-pool scheduling and cross-NUMA
wake-up overhead, not missing Schur task count, is the remaining wide-pool
bottleneck.

### P19 pool-aware serial factor -- rejected by full-solver A/B

Normal-queue job 196546 compared the retained eight-task factor against a
candidate that selected the serial SIMD factor when the Julia pool contained
96 or more workers. It completed on node60 with a `PASSED` marker, valid
checksums, and 20 `Optimal` 41-iteration solves. Each implementation ran in
both forward and reverse order after three complete warm-ups.

The candidate reduced combined median factor time from 0.742 to 0.425 seconds
and KKT time from 1.142 to 0.831 seconds. Nevertheless, combined median solve
time increased from 11.749 to 11.803 seconds (0.46% slower). The forward pair
was 0.23% slower and the reverse pair was 2.54% slower, so there was no
reproducible end-to-end benefit. Residual, Schur, predictor, corrector, line
search, and update phases collectively lost more time than the factor saved.
Allocation changed only from 173.62 to 172.18 MiB per solve.

All numerical gates were unchanged: maximum relative gap was `9.211e-11`,
maximum primal and dual residuals were `2.205e-62` and `3.257e-14`, and the
minimum PSD eigenvalue was `2.508e-18`. Resource telemetry showed 129 OS
threads and one BLAS thread in both variants. Average process use fell from
about 15.0 CPU cores to 13.7--14.2 cores and voluntary context switches fell
from roughly 27.5 million to 25.0 million, but neither translated into lower
wall time. The pool-aware serial factor is therefore rejected and the
validated at-most-eight-task selector is restored for every Julia pool.

### P20 128-worker sleep-policy full-solver A/B -- job 196548

Job 196548 ran the retained corrected-cap source on node53 with an exact
128-thread Julia pool, one BLAS thread, exclusive CPU affinity, and memory
interleaved across all eight NUMA nodes. The only changed process-start
setting was `JULIA_THREAD_SLEEP_THRESHOLD`: unset for the Julia default and
`infinite` for permanently awake workers. Each policy ran in forward and
reverse order after three full warm-ups, with five recorded solves per
process. PBS exited zero, the job emitted `PASSED`, and all four CSVs plus 20
solution files passed their SHA-256 checks.

| Combined ten-sample metric | Default sleep | Never sleep | Change |
|---|---:|---:|---:|
| Solve (s) | 11.890 | 3.550 | **70.1% lower** |
| End-to-end measured row (s) | 14.332 | 5.784 | 59.6% lower |
| Residual (s) | 1.126 | 0.271 | 76.0% lower |
| Schur (s) | 2.260 | 0.600 | 73.5% lower |
| KKT (s) | 1.150 | 0.182 | 84.1% lower |
| Reduced factor (s) | 0.748 | 0.147 | 80.4% lower |
| Predictor (s) | 2.060 | 0.483 | 76.5% lower |
| Corrector (s) | 2.230 | 0.667 | 70.1% lower |
| Line search (s) | 0.402 | 0.029 | 92.8% lower |
| Update (s) | 0.408 | 0.034 | 91.7% lower |
| Allocation (MiB/solve) | 158.83 | 152.07 | 4.3% lower |
| Result peak memory (GiB) | 3.175 | 3.281 | 3.3% higher |

The two never-sleep process medians were 3.762 and 3.384 seconds, versus
11.586 and 12.021 seconds for the bracketing default controls. The speedup is
therefore not a favorable single ordering. Every row was `Optimal` in 41
iterations, and the printed primal objective, dual objective, relative gap,
primal residual, dual residual, and minimum PSD eigenvalue were bit-for-bit
identical across all 20 rows. The common maximum relative gap was
`9.210e-11`, the dual residual was `3.257e-14`, and the minimum PSD eigenvalue
was `2.508e-18`.

Resource telemetry explains the wide-pool collapse. Both modes created 129
OS threads, pinned the 128 Julia compute threads to CPUs 0--127, reached all
128 workers, and kept BLAS at one thread. The default processes averaged
14.09/15.24 CPU-core equivalents and incurred 27.48/27.36 million voluntary
context switches. Never-sleep averaged 113.21/113.16 cores, its median sample
used about 127.2 cores, and voluntary switches fell to 128/129 thousand.
Involuntary switches rose from roughly 0.58 million to 1.57 million. Process
wall time fell from 4:59/4:41 to 3:53/3:50; process peak RSS was comparable.

This setting is not made a library default: Julia reads it before startup,
and continuous spinning consumes the whole reserved node even during serial
or compilation work. It establishes a measured speed ceiling and proves that
worker wake-up/sleep and scheduler synchronization, rather than missing
Float64x4 arithmetic parallelism, dominate the 128-pool default. The next
experiment searches finite spin thresholds for the same kernel speed with
less idle CPU consumption.

### P21 finite Julia sleep-threshold kernel sweep -- job 196549

Job 196549 completed on node60 with PBS exit zero, a `PASSED` marker, and
valid checksums for all sixteen factor/SYRK CSVs. It swept 100 microseconds,
1, 10, 100, and 1,000 milliseconds between bracketing default controls and
the never-sleep ceiling. Every 3,400-by-144 SYRK lower triangle was exact;
every order-144 factor had the same `1.218e-16` Float64-visible reconstruction
error.

| Sleep policy | 64-task SYRK (ms) | 8-task factor (ms) | SYRK process CPU | Factor process CPU |
|---|---:|---:|---:|---:|
| Default, first/repeat | 35.21 / 32.38 | 19.13 / 19.41 | 12.53 / 12.90 cores | 8.33 / 8.80 cores |
| 100 microseconds | 28.56 | 18.89 | 12.43 cores | 8.79 cores |
| 1 millisecond | 23.18 | 3.96 | 9.18 cores | 9.33 cores |
| 10 milliseconds | 11.63 | 3.28 | 18.19 cores | 8.65 cores |
| 100 milliseconds | **11.39** | **3.23** | 30.93 cores | 12.30 cores |
| 1 second | 11.55 | 3.49 | 63.23 cores | 28.10 cores |
| Never sleep | 11.56 | 3.36 | 109.54 cores | 112.58 cores |

Ten milliseconds reaches the useful 64-task SYRK and factor plateaus with
substantially less process CPU than permanent spinning. One hundred
milliseconds is the fastest isolated point but uses more spin. Both advance
to a complete-solver, repeated-order comparison; shorter thresholds and one
second are rejected at the kernel screen.

### P22 finite-threshold full solver -- job 196550

Job 196550 completed on node57 with PBS exit zero, `PASSED`, valid checksums,
and 35 `Optimal` 41-iteration solutions. The run identities exposed a driver
labeling issue before analysis: the shell pattern `ms10*` preceded `ms100*`,
so both nominal labels received the recorded value `10000000` nanoseconds.
The four finite processes are therefore four independent 10-millisecond
repeats, not a 10/100-millisecond comparison. The script order was corrected;
no result is assigned to an unrecorded policy, and the redundant repetitions
make the 10-millisecond conclusion stronger.

| Combined metric | Default, 10 rows | 10 ms, 20 rows | Never sleep, 5 rows |
|---|---:|---:|---:|
| Solve (s) | 11.635 | **3.409** | 3.375 |
| End-to-end measured row (s) | 13.958 | **5.600** | 5.615 |
| Residual (s) | 1.093 | 0.277 | 0.277 |
| Schur (s) | 2.199 | 0.610 | 0.587 |
| KKT (s) | 1.134 | 0.194 | 0.186 |
| Reduced factor (s) | 0.737 | 0.150 | 0.149 |
| Predictor (s) | 1.990 | 0.489 | 0.481 |
| Corrector (s) | 2.192 | 0.672 | 0.677 |
| Line search (s) | 0.401 | 0.028 | 0.029 |
| Update (s) | 0.407 | 0.034 | 0.034 |
| Allocation (MiB/solve) | 165.58 | 152.07 | 152.07 |
| Result peak memory (GiB) | 3.363 | 3.305 | 3.303 |

The 10-millisecond policy is 70.7% faster than Julia's default and only 1.0%
slower than never-sleep in solver time. Its end-to-end measured-row median is
slightly lower than never-sleep, allocation is 8.2% below default, and peak
memory is 1.7% lower. All objective and certificate strings are identical
across all 35 rows: relative gap `9.210e-11`, primal residual `2.205e-62`,
dual residual `3.257e-14`, and minimum PSD eigenvalue `2.508e-18`.

Every process created 129 OS threads, pinned 128 Julia workers to CPUs 0--127,
kept BLAS at one thread, and interleaved memory over NUMA nodes 0--7. The two
default controls averaged 15.16/15.19 CPU-core equivalents and incurred
27.53/27.49 million voluntary context switches. The four 10-millisecond
processes averaged only 12.44--12.61 cores over their complete lifetimes,
reached 124--126 cores during parallel bursts, and incurred 108--115 thousand
voluntary switches. Never-sleep averaged 113.09 cores and incurred 125
thousand voluntary switches. Thus 10 milliseconds removes the wake-up storm
without burning the whole node during serial work.

`JULIA_THREAD_SLEEP_THRESHOLD=10000000` is retained as the hardware-specific
cluster launch recommendation, not as package state: Julia consumes the
setting before SDPX is loaded. The final exact-pool sweep compares this policy
with Julia's default at 1--128 threads to select the smallest and fastest
resource request.

### P23 complete local regression after the retained selector revert

The standard package test command ran on Julia 1.12.6 with four Julia threads
and a writable isolated depot layered over the installed package cache. It
passed 5,803/5,803 checks in 5 minutes 10.9 seconds, including Aqua, public API
and extension checks, native BigFloat ownership regressions, Float64x4 Schur
and factor tests, diagnostics, certificates, and solver cases. Repeated macOS
`sysctl` permission messages came only from the sandboxed memory probe and did
not fail a check. The earlier focused runs also passed 87/87 extended-BLAS,
72/72 extended-precision-BLAS, and 18/18 executed-diagnostic assertions.

### P24 lazy reduced-arrow Schur partials and post-change regression

The direct reduced-panel path never consumes the legacy task-local
`Sredpartial` matrices: it forms the reduced shared triangle directly and the
later local-factor loop only prepares the local coupling solves.  Workspace
construction nevertheless allocated one dense `ng`-by-`ng` Float64x4 matrix
per selected worker in case the direct kernel later fell back.  For the
medium CSDR geometry (`ng = 144`), each unused matrix occupies 663,552 bytes;
48 copies occupy 30.375 MiB and 64 copies occupy 40.5 MiB before allocator
overhead.

`ArrowWorkspace` now accepts an explicit partial count and may omit the full
Schur partials.  Direct reduced workspaces retain only the small RHS partials;
`ensure_arrow_schur_partials!` allocates independent full matrices if the
structure-specific build declines the problem or if the factorization enters
the legacy fused path.  The fallback is outside the successful direct hot
path.  Construction also uses the already selected fine-grained bin count,
so a 96/128-thread request does not recreate 64 redundant RHS buffers after
the narrow-arrow scheduler selects 32 bins.

New tests verify that direct workspaces start without full Schur partials,
that lazy fallback matrices are independent, that full shared coverage safely
skips clearing the packed panel, and that incomplete shared coverage clears
sentinel data before rebuilding the exact reduced Schur matrix.  The focused
extended-precision test group passed 76/76 assertions.  The complete package
suite then passed 5,807/5,807 checks on Julia 1.12.6 with four Julia threads in
5 minutes 12.1 seconds.  The cluster candidate is isolated at
`<cluster-root>/SDPX.jl/experiments/float64x4-lazypartial-d9c287af-clean`;
production remains unchanged pending the repeated complete-solver A/B.

### P25 exact-pool Float64x4 scaling and resource audit -- job 196551

Normal-queue job 196551 completed on node60 (dual-socket AMD EPYC 7742,
128 CPUs, eight NUMA domains) with PBS exit zero, a `PASSED` marker, valid
SHA-256 checks for 22 CSV files and 110 solution files, and no
error marker in any solver log.  The job reserved one complete 128-core,
128-GiB node.  Every process used an exact Julia compute pool
(`--threads=N,0`), one GC mark thread, no interactive or sweep pool, one BLAS
thread, `JULIA_EXCLUSIVE=1`, and three complete warm-ups followed by five
recorded solves.  Runs through 16 workers were bound within NUMA node 0,
32 workers used CPUs 0--31 with memory nodes 0--1, 48/64 workers used CPUs
0--47/63 with memory nodes 0--3, and 96/128 workers used CPUs 0--127 with
memory interleaved over nodes 0--7.

| Julia workers | Default sleep (s) | 10 ms sleep (s) | 10 ms speedup vs 1 | Maximum active workers |
|---:|---:|---:|---:|---:|
| 1 | 44.112 | -- | 1.00x | 1 |
| 2 | 22.938 | 22.954 | 1.92x | 2 |
| 4 | 12.458 | 12.450 | 3.54x | 4 |
| 8 | 6.975 | 6.946 | 6.35x | 8 |
| 16 | 4.318 | 4.275 | 10.32x | 16 |
| 32 | 3.448 | 3.267 | 13.50x | 32 |
| 48 | 3.325 | **2.967** | **14.87x** | 48 |
| 64 | 3.824 | 3.074 | 14.35x | 64 |
| 96 | 8.910 | 3.540 | 12.46x | 96 |
| 128 | 12.061 | 3.403 | 12.96x | 128 |

The independent default 32-worker controls were 3.451, 3.448, and 3.437
seconds.  The two 10-millisecond 32-worker controls were 3.267 and 3.292
seconds.  Thus the wake-policy result is not a favorable single process, and
the campaign's fastest 48-worker point is 9.2% below the first 10-millisecond
32-worker result and 3.5% below 64 workers.  The 48-worker median phase split
was 0.222 seconds residual, 0.741 Schur, 0.160 KKT including 0.122 factor,
0.371 predictor, 0.513 corrector, 0.013 line search, and 0.020 update.  It
allocated 148.46 MiB per solve and reached 3.144 GiB process peak RSS.

Resource sampling confirms that the requested pools were genuinely active.
The 48-worker 10-millisecond process reached 45.38 CPU-core equivalents in a
sample and all 48 workers; the 96-worker process reached 94.87 cores and all
96 workers; the 128-worker process reached 125.78 cores and all 128 workers.
BLAS reported one worker in every CSV, and all BLAS/OMP environment limits
were one, so no nested BLAS pool explains the measurements.  Whole-process
mean use was lower (5.25, 9.42, and 12.39 cores respectively) because it also
includes package/model loading, compilation, serialization, and serial
regions around the repeated solves.

The wide-pool loss is synchronization, not an idle allocation.  At 96 workers
Julia's default sleep policy incurred 17,416,887 voluntary context switches
and took 8.910 seconds; the finite threshold reduced those to 86,666 and
3.540 seconds.  At 128 workers the corresponding counts were 27,383,741 and
111,341, with solve times of 12.061 and 3.403 seconds.  Even after eliminating
the wake-up storm, both pools lose to 48 because the 144-column shared
triangle has saturated and the wider runtime spans more NUMA domains.

All 110 recorded solves returned `Optimal` in 41 iterations.  Only expected
last-limb reduction-order differences occurred across pool widths.  In
Float64 projection the primal objective was
`6.3668480614993285e-12`, the dual objective was
`-8.5733398394969845e-11`, maximum relative gap was `9.2100e-11`, maximum
primal and dual residuals were `2.8562e-62` and `3.2572e-14`, and the minimum
PSD eigenvalue was `2.5085e-18`.  The retained launch recommendation for this
node and geometry is therefore a 128-core allocation, an exact 48-worker
Julia pool, one BLAS worker, local CPU binding across CPUs 0--47, memory bound
to NUMA nodes 0--3, and `JULIA_THREAD_SLEEP_THRESHOLD=10000000`.  The sleep
threshold remains an explicit cluster launch setting rather than package
state because Julia consumes it before SDPX loads.

### P26 fused Float64x4 singleton-local preparation -- candidate

At the retained 48-worker point, KKT local elimination still consumed a
median 36.95 milliseconds over 41 iterations.  The direct 2x2 reduced-panel
pack had already computed each singleton local diagonal and coupling row, but
`factor_arrow_kkt!` later launched another block pass to copy the diagonal,
take the same square root, cache its inverse, and form `D^-1*C`.

The candidate adds an arithmetic-extension hook that caches these values
while the block's coupling row is hot.  Only Float64x4 implements it.  It
stores `sqrt(D)`, computes the inverse in the historical
`one / sqrt(D) / sqrt(D)` order, and writes `D^-1*C` with the same scalar
multiplication as the old pass.  A per-block readiness vector and one
aggregate flag allow the KKT factor to skip the redundant pass only after
every direct-panel block succeeded.  Float64, BigFloat, mixed precision,
partial structure, regularization, and materialized fallback paths keep the
established factorization and explicitly clear the readiness state.

Focused tests check every cached factor, inverse, and solved coupling entry,
including a partial-coverage solve that catches stale uncoupled entries.  The
extended-precision BLAS group passed 95/95 assertions and the extended BLAS
regression group passed 87/87; the complete package suite passed 5,823/5,823
checks before the final three partial-coverage assertions were added.  This
candidate is not retained yet; it
must remove the measured local-elimination time and improve repeated full
solves at the final 48-worker scheduling configuration without changing the
objective or certificate.

### P27 lazy reduced-arrow partials retained -- job 196555

Normal-queue job 196555 completed on node26 with PBS exit zero and a
`PASSED` marker.  SHA-256 verification covered all eight CSVs and forty
solution files.  Baseline and candidate each ran in forward and reverse order
at the measured 48-worker optimum and in a 128-thread capped-pool control;
every process used the 10-millisecond sleep threshold, one BLAS worker, three
complete warm-ups, and five recorded solves.

| Metric | 48 baseline | 48 lazy | Change | 128 baseline | 128 lazy | Change |
|---|---:|---:|---:|---:|---:|---:|
| Solve (s) | 2.982 | **2.940** | **-1.39%** | 3.480 | **3.351** | **-3.69%** |
| Measured-row wall (s) | 4.860 | **4.692** | -3.47% | 5.752 | **5.601** | -2.63% |
| Workspace setup (ms) | 15.284 | **12.842** | -16.0% | 21.078 | **15.275** | -27.5% |
| Solve allocation (MiB) | 148.458 | **118.079** | -20.5% | 152.072 | **111.421** | -26.7% |
| Process peak RSS (GiB) | 3.166 | **3.028** | -4.36% | 3.360 | **3.243** | -3.48% |

Both candidate process medians were stable: 2.942/2.939 seconds at 48
workers and 3.342/3.361 seconds in the 128-thread pool.  Resource telemetry
reached every requested Julia worker (48 or 128), kept BLAS at one, and did
not trade memory for hidden oversubscription.  Within each pool width, all
twenty baseline/candidate result tuples were string-identical: `Optimal`, 41
iterations, equal primal/dual objectives, equal relative gap and residuals,
and equal minimum PSD eigenvalue.  The lazy full-Schur partial allocation is
therefore retained: it provides reproducible full-solver speedup and removes
the expected 30.375/40.5 MiB of unused raw Float64x4 matrix storage at the two
tested workspace widths.

### P28 block locality and longer worker-spin controls rejected -- job 196556

Normal-queue job 196556 ran on the same dual-EPYC node class with a complete
128-core reservation, an exact 48-worker Julia pool, one BLAS worker, CPUs
0--47, memory bound to NUMA nodes 0--3, and three warm-ups plus five measured
solves per process.  Forward and reverse brackets independently compared LPT
and contiguous ownership of the short block-local phases at 10- and
100-millisecond Julia sleep thresholds.  The job exited zero, wrote `PASSED`,
and verified all eight CSVs and forty solution files by SHA-256.

| Fine block partition | Worker spin | Samples | Solve (s) | Row wall (s) | Mean process cores | Peak RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| LPT | 10 ms | 10 | 2.9727 | 4.8185 | 5.07 | 2.871 |
| contiguous | 10 ms | 10 | 2.9572 | 4.7824 | 5.16 | 2.928 |
| LPT | 100 ms | 10 | 2.9584 | 4.7815 | 5.71 | 2.837 |
| contiguous | 100 ms | 10 | 2.9581 | 4.8153 | 5.66 | 2.881 |

Contiguous ownership's 0.52% advantage at 10 milliseconds disappeared at
100 milliseconds, while the longer spin policy provided no solve benefit and
increased mean whole-process CPU consumption by about 11%.  All configurations
reached all 48 workers and kept BLAS at one.  Every solve returned `Optimal`
in 41 iterations with equivalent objectives, residuals, gap, and PSD
certificate.  Neither change clears the retention threshold; the established
LPT partition and 10-millisecond cluster launch policy remain selected.

### P29 fused singleton-local factor preparation retained -- job 196580

Job 196580 isolated the P26 candidate against the retained lazy-partial
baseline on node57.  It reserved 128 cores but launched exactly 48 pinned
Julia workers and one BLAS worker.  Baseline/candidate and reverse
candidate/baseline processes each performed three full warm-ups followed by
seven measured solves.  PBS exit status was zero, `PASSED` was present, and
all four CSVs plus 28 serialized solutions passed SHA-256 verification.

| Metric | Baseline, 14 solves | Fused preparation, 14 solves | Change |
|---|---:|---:|---:|
| Solver time (s) | 2.92683 | **2.90423** | **-0.77%** |
| Measured-row wall (s) | 4.63878 | **4.60665** | **-0.69%** |
| KKT time (s) | 0.16143 | **0.12338** | **-23.6%** |
| Local elimination (ms) | 36.93 | **0.016** | **-99.96%** |
| Solve allocation (MiB) | 118.077 | **117.310** | **-0.65%** |
| Process peak RSS (GiB) | 2.933 | **2.885** | **-1.6%** |

The two independent candidate medians were 2.90327 and 2.90518 seconds;
the corresponding baselines were 2.92747 and 2.92382 seconds.  About 14
milliseconds of local work moved into the cache-hot Schur pack, but the
separate 37-millisecond local factor pass disappeared, giving the stable net
gain.  Every process reached all 48 requested workers, used one BLAS thread,
and returned the same 41-iteration optimal certificate.  The change is
therefore retained.

### P30 Float64x4 singleton-arrow SIMD solve retained -- job 196582

After P29, predictor and corrector linear solves still use repeated
singleton-local block-arrow RHS accumulation and local recovery.  The next
candidate opts in only `MultiFloats.Float64x4` reduced panels.  It updates up
to eight independent shared RHS entries in two four-lane `MultiFloatVec`
groups, and recovers four independent local variables in four lanes.  Each
lane retains the historical block/global reduction order, every task owns its
partial vector and local destinations exclusively, and generic Float64,
BigFloat, mixed-precision, and non-reduced paths remain on their existing
loops.

Direct tests require bit-for-bit agreement with the same threaded reduction
schedule and zero hot-loop allocations.  The widened threaded fixture covers
64 local blocks and 32 shared variables; the complete extended-precision
group passes 104/104 assertions with either one or four Julia threads.  The
extended-BLAS and generic KKT groups pass 87/87 and 74/74.

Job 196582 then used the same node57, 128-core reservation, exact 48-worker
pool, one BLAS worker, 10-millisecond sleep policy, CPU/NUMA placement, and
three-warm-up/seven-measurement reverse bracket as P29.  It exited zero with
`PASSED`; all four CSVs and 28 solutions passed their SHA-256 manifest.

| Metric | P29 baseline, 14 solves | SIMD solve, 14 solves | Change |
|---|---:|---:|---:|
| Solver time (s) | 2.91341 | **2.89516** | **-0.63%** |
| Predictor linear solve (s) | 0.13520 | **0.12606** | **-6.76%** |
| Corrector linear solve (s) | 0.13209 | **0.12036** | **-8.88%** |
| Solve allocation (MiB) | 117.312 | 117.369 | +0.05% |
| Process peak RSS (GiB) | 2.955 | 2.957 | +0.07% |

Both candidate process medians improved (2.89059 and 2.89973 seconds versus
2.91246 and 2.91437 seconds for the two baselines).  The measured-row wall
median moved from 4.62596 to 4.63999 seconds (+0.30%) because of untimed
loading/build variation; the solver-only gain and both targeted subphases are
stable.  Mean whole-process CPU use was 5.81/5.84 cores, both variants reached
all 48 workers, BLAS remained one, and memory was unchanged.  All 28 rows have
one exact certificate signature and all serialized solutions have one hash:
41 iterations, `Optimal`, relative gap `9.21002e-11`, dual residual
`3.25712e-14`, and minimum PSD eigenvalue `2.50847e-18`.  The SIMD solve is
therefore retained.

### P31 eight-block recovery grouping rejected by isolated screen

The remaining obvious microkernel variation grouped eight independent local
recoveries as two four-lane accumulators, reusing each shared-value broadcast.
A deterministic 1,700-block / 144-shared-variable fixture reproduced the 48
LPT-bin geometry, alternated 21 post-warm-up samples per implementation, and
required exact output equality plus zero allocations.  The retained four-
block kernel took 2.7258 milliseconds; the eight-block kernel took 2.7541
milliseconds and was 1.0% slower.  Register pressure and eight independent
weight streams offset the saved loop overhead.  The eight-block variation is
rejected without a full-solver run.

### P32 complete local regression after the retained Float64x4 changes

The final local Julia 1.12.6 regression executed with four compute threads.
All 5,834 solver, numerical-certificate, allocation, API, extension, and
thread-safety assertions passed.  The only failing assertion was Aqua's
persistent-task probe: its isolated temporary Julia environment attempted to
download the package registry, while the macOS execution sandbox intentionally
blocked DNS.  The subprocess then lacked its generated source and completion
log.  This is an environment-only failure rather than a package or numerical
regression.  The immutable cluster release must repeat the complete suite in
the preinstalled offline environment, where the Aqua probe is a hard release
gate.

### P33 finite-support LP audit and reduced-system candidate

The finite-energy-support handoff was reduced to its exact numerical core:
`min/max c'rho` subject to `B'rho=b, rho>=0`.  The primary exported model has
2,002 variables and 31 equalities; 1,001-by-17 and 91-by-17 models are controls.
The primary endpoint is non-strictly feasible and its unscaled moment rows span
about 52 decimal orders of magnitude, so both degeneracy and scale-aware
certification are required.

The audit found that the dedicated LP engine still expanded the identity cone
map and, for extended precision, built and factored a 2,033-by-2,033 dense KKT.
For a permutation-diagonal positive cone map, exact block elimination instead
gives the 31-by-31 positive-definite system

```text
D       = diag(g_i^2 z_i/s_i) + delta*I
Q       = B' inv(D) B + delta*I
Q dy    = ry - B' inv(D) rx
dx      = inv(D) (rx + B dy).
```

A candidate implementation now recognizes only this exact structure, avoids
the dense cone/Hessian/KKT allocations, assembles one lower triangle, and
reuses the Cholesky for predictor and corrector.  Float64 uses BLAS SYRK;
Float64x4 and BigFloat use disjoint packed-panel rows and disjoint output tiles.
The BigFloat pack uses owned MPFR destinations and in-place MPFR division and
square root.  Failed structure checks preserve the established LP route.
The same audit found that the LP regularization floor was clamped to `1e-12`
for every arithmetic type.  Binary64 keeps its established floor; fixed-width
and MPFR arithmetic now start at `max(eps(T)^(3/4), 1e-60)` and retain the
existing factor-failure escalation.  This is still awaiting the cluster A/B.

The cluster benchmark driver, PBS matrix, independent componentwise equality
audit, and full optimization proposal are recorded under
`bench/finite_support_lp/`; the retired design note is preserved in Git
history rather than the published documentation set.
No numerical result is recorded yet: two SSH attempts to the cluster alias
timed out at the TCP connection layer on 2026-08-02.  In accordance with the
request, no substitute solve or benchmark was run on the local Mac.  The next
step is an immutable cluster syntax/unit gate, followed by 91-by-17,
1,001-by-17, and 2,002-by-31 A/B runs for Float64, Float64x4, and BigFloat256.

### P34 cluster smoke gate and executed-kernel audit

The immutable candidate release `3ac5bcc836f31c434ef666b67bebf113ef4dfd73`
was run on a normal-queue compute node (`node93`, eight allocated CPUs; Julia
was intentionally started with one thread for the smoke gate).  The focused
LP regression file passed all 81 assertions: high-precision Hessian 10/10,
reduced standard-form system 26/26, and LP presolve 45/45.  The job exited
with status 0 and used about 1.69 GiB peak RSS.

The first smoke report appeared to show dense LU and pairwise extended
precision kernels for Float64x4/BigFloat.  A compute-node type trace showed
that both types do enter `LPDiagonalMatrix` and the reduced Cholesky route.  The
discrepancy was in the benchmark driver: `result.diagnostics.selected_algorithms`
is the conservative pre-solve plan, whereas the actual post-scaling dispatch
is recorded in `result.termination.executed`.  A tiny Float64x4 and BigFloat
solve confirmed `kkt=:diagonal_reduced_cholesky` and
`gram=:reduced_equality_syrk`.  The driver was corrected to report both planned
and executed labels; the executed labels are now the authoritative fields for
kernel comparisons.  No solver algorithm change was made for this finding.

The initial smoke numerical certificates remain false on the deliberately
degenerate endpoint (the control has a raw equality residual around `1e24` in
Float64 and the requested extended tolerances are stricter than the attainable
scaled residual).  Independent normalized equality and objective values remain
the required comparison gates; a full benchmark will report both raw and
normalized residuals.

### P35 certificate downgrade provenance fix

The first rerun of the corrected driver exposed a real diagnostics bug: when
`certify_final_result` downgraded an `Optimal` status to `Stalled`, it replaced
the entire termination record with only the certificate failure.  The solve
itself had completed the reduced LP route, but the benchmark then could not
read `termination.executed` for extended arithmetic and aborted.  The
certificate downgrade now merges its reason/failure fields into the existing
termination record, retaining executed-kernel, equality-system, refinement,
and fallback provenance.  The next smoke gate must verify this on all three
arithmetic modes before the full matrix is submitted.

### P36 provenance-fix smoke gate

Release `120d8585b697e15a86c0969f4c8e94a30fd5abea` ran on a normal-queue
128-core node (`node137`; Julia itself used one thread for the control models).
The job completed with exit status 0, peak RSS about 1.52 GiB, and the focused
LP suite passed 81/81 assertions.  All three logs now report the actual
`diagonal_reduced_cholesky` / `reduced_equality_syrk` route, including the
certificate-failure downgrade case.

For the 91-by-17 control (one timed solve after one warm-up): Float64 took
0.00548 s with 553,032 allocated bytes; Float64x4 took 0.06214 s with
1,862,888 bytes; BigFloat256 took 0.22297 s with 48,704,360 bytes.  Their
independent normalized equality residuals were `1.0`, `1.01924e-13`, and
`4.68436e-23`, respectively; these deliberately degenerate endpoint runs are
not certificates at the requested tolerances.  The objective values remain
near one and the nonnegative/PSD scalar checks are valid within each
arithmetic's reported residual.  This gate is retained; the full primary and
secondary matrix is now safe to submit with the corrected reporting.

### P37 true-primary cluster A/B and scheduler diagnosis

The first full “primary” sweep had used the 1,001-by-17 control. A corrected
run used the actual 2,002-by-31 `all_spins_fixed_y_mu12_1_min.txt` model. The
unchanged production Float64 path ran on node147 with 128 reserved CPUs and
used one, four, and eight BLAS threads. Solver times were 40.590 s, 19.132 s,
and 15.489 s; the dominant stages were Gram assembly (23.098 s, 11.334 s,
7.939 s) and dense KKT factorization (16.126 s, 6.264 s, 5.800 s). All three
runs reached the same 120-iteration limit and had the expected degenerate
endpoint certificate failure (`relative_gap` 0.01784, 0.01736, 0.01800 and
normalized equality residual approximately one).

Two unconstrained 128-core candidate allocations were killed by PBS with
SIGKILL before the shell wrote startup metadata (nodes 134 and 71). A minimal
probe reproduced the same scheduler/node behavior. Targeting the healthy
node147 allocation allowed the candidate to complete; production `current`
was never modified.

### P38 reduced LP kernel on the true primary

The candidate’s exact permutation-diagonal reduction selected
`diagonal_reduced_cholesky` and `reduced_equality_syrk` for all arithmetic
modes. With the same node and one warm-up, Float64 at 120 iterations took
0.04453 s (0.03625 s timed core, 8.32 MB allocations), versus 40.590 s for
the dense baseline; its gap was 0.01705, slightly better than the baseline
gap. The Float64x4 run took 6.235 s (6.001 s core, 22.70 MB allocations) and
reduced the normalized equality residual to 2.27e-12. Increasing Float64x4
threads was not beneficial for this 31-equality Schur: 24.22 s at two threads,
26.58 s at four, and 24.26 s at eight for the deliberately 600-iteration
stress run; the panel scheduler selected six workers at eight threads.

Native BigFloat256 was substantially more accurate in equality feasibility
but produced very large dual multipliers on this ill-conditioned endpoint
(`relative_gap` about 2 and dual objective 6.36e4 at 120 iterations), even
with the fixed parameter policy. This is a numerical robustness issue, not a
kernel-speed result, so BigFloat promotion is blocked pending the
regularization experiment. The 600-iteration stress run took 102.05 s core,
allocated 7.67 GB, and was not accepted as a certificate.

### P39 tolerance-aware BigFloat regularization and snapshot selection

The tolerance-aware native BigFloat regularization floor was tested on the
true 2,002-variable/31-equality endpoint.  A `1e-52` floor reduced the
120-iteration BigFloat relative gap from approximately `2` to `0.1699`; the
conservative `min(1e-48, 1e-8*ϵ_gap)` cap then reduced it to `5.35e-5` with a
normalized equality residual of `1.40e-12`.  Without an iterate snapshot, the
600-iteration run drifted back to a gap of `5.62e-4`, so returning the last
iterate was not robust on this degenerate endpoint.

### P40 gap-first best-iterate validation on the primary LP

Release `29bbfe164dc5540c8913850e0241c270f0e23cc` changes the LP snapshot
policy to minimize relative duality gap first, with normalized feasibility as
a tie-breaker.  The policy remains bounded and allocation-safe: snapshots
own their storage and are copied only when the merit improves.  A 128-core
normal-queue job on node147 ran Float64, Float64x4, and BigFloat256 with one
Julia thread each, adaptive parameters, a warm-up solve, and 600 iterations.

The first batch wrapper exited before Julia because its result-directory path
had a typo; PBS reported zero CPU time and no solver output.  The path was
corrected, the wrapper was resubmitted, and the successful measurements below
come only from the corrected job.

* Float64: 0.18234 s total (0.17525 s timed core), 9.02 MB allocated,
  1.355 GB peak RSS, gap `8.4422e-2`, normalized equality `1.0000`, and the
  same endpoint/iteration-limit behavior as the preceding run.
* Float64x4: 32.956 s total/core, 24.998 MB allocated, 1.833 GB peak RSS,
  gap `3.50969e-15`, normalized equality `6.12289e-18`, with objective
  values both within `4e-15` of one.  This is a large improvement over the
  previous late-iterate gap of about `4e-5`.
* BigFloat256: 102.129 s total/core, 7.676 GB allocated, 1.596 GB peak RSS,
  gap `3.50964e-15`, normalized equality `6.12290e-18`, with both objective
  values within `4e-15` of one.  The snapshot copies add allocation traffic
  but do not increase peak RSS; they prevent the late-iteration drift.

The endpoint intentionally stops at the iteration limit and therefore does
not produce a formal optimality certificate at the requested tolerance.  The
independent normalized residual, objective agreement, and nonnegative/PSD
checks are the meaningful gates for this degenerate model.  The scheduler
started the job on node147 with all 128 CPUs reserved; Julia and BLAS were
explicitly single-threaded, so the reported solve times are single-core
measurements rather than inflated allocations.

### P41 latest-code focused regression gate

The latest release passed `test/lp_regressions.jl` on node147 with exit status
zero: high-precision LP Hessian regressions 10/10, reduced standard-form
system 26/26, and LP presolve regressions 45/45 (81/81 total).  The test job
used about 1.06 GiB resident memory and completed in 84 seconds including
Julia startup and compilation.  The production `current` symlink was not
modified.

### P42 Chapter 3 finite-energy LP cluster route and precision/thread sweep

The isolated Chapter 3 project at
`<cluster-root>/chapter3-sdpx-lp` was compared with the
older release path.  The old path selected a serial weighted outer-product
Gram build and dense LU (`serial_weighted_outer_product`, `dense_lu`): a
20-iteration, 408-variable/16-equality pilot took 213.17 s end-to-end, with
about 77.34 s of steady LP core per boundary.  Candidate release `29bbfe1`
selected the reduced LP route (`reduced_equality_syrk`,
`diagonal_reduced_cholesky`) and completed the same pilot in 62.15 s including
startup; steady core was 0.141/0.100 s per boundary.  Objective, residual, and
minimum-weight fields were recorded independently; the 20-iteration pilot is
an iteration-limit diagnostic, not a certificate.

The isolated Chapter 3 driver now accepts explicit
`--precision=float64|float64x4|bigfloat256` and `--threads=N` controls.  A
small smoke job passed all three arithmetic paths.  Float64x4 and native
BigFloat256 agreed to the displayed high-precision digits, while Float64
showed the expected lower-precision early-iterate differences.

On the 1,584-variable/47-equality, 20-iteration pilot, Float64x4 steady
minimum-boundary LP core was 1.428/1.234/0.982/0.995/0.905/0.961 s at
1/2/4/8/16/32 requested threads.  The solver used reduced SYRK and diagonal
Cholesky throughout; its scheduler capped Schur workers at 10 for the 16- and
32-thread cases.  Numerical fields were identical across thread counts
(relative gap 3.1573e-3 for the minimum boundary and 6.3700e-4 for the
maximum at 20 iterations), with peak RSS 1.62--1.83 GiB.  The reproducible
best point is 16 Julia threads; 32 threads regressed slightly because
predictor/residual work and synchronization dominate after SYRK.  A larger
model sweep is queued before changing any solver default.

### P43 native BigFloat panel threading for the Chapter 3 LP

The production-sized Chapter 3 case (`n_mu=300`, `j_max=40`, `k_max=8`,
12,300 variables, 47 independent equalities, 20 iterations) was profiled on
node147 with construction precision 512 bits and a 256-bit native BigFloat
solve.  The serial candidate spent about 24.9--25.3 s of each boundary core
in weighted-panel/SYRK assembly; factorization was below 0.1 s.  The existing
planner nevertheless forced every native BigFloat solve serial.

The isolated release
`ch3-lp-bigfloat-threads-c1178eb` enables threads only for the dedicated
standard-form LP when each panel row and Schur tile has exclusive ownership.
The original serial path remains the fallback for all other BigFloat models.
On the same node, 20-iteration results were numerically identical at every
thread count: the maximum-boundary audited objective was
`-0.869841951302534453889472977476090943666931850166807968...`, the minimum
was `0.411369163632267446781178158004261889897083760098...`, the relative gaps
were `1.0354244890832694e-3` and `2.4805867005840241e-2`, and the 256-bit
dual residuals remained below `7e-45`/`2e-42`.

| requested threads | effective | LP core (s) | SYRK/panel (s) | end-to-end (s) | CPU | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 40.94 | 24.94 | 161.58 | 99% | 2.19 GiB |
| 2 | 2 | 36.63 | 18.95 | 155.23 | 124% | 2.10 GiB |
| 4 | 4 | 31.50 | 11.39 | 146.58 | 149% | 2.35 GiB |
| 8 | 8 | 26.90 | 6.73 | 145.30 | 190% | 2.22 GiB |
| 16 | 16 | 25.69 | 4.57 | 153.04 | 267% | 2.39 GiB |
| 32 | 32 | 26.85 | 3.48 | 157.08 | 492% | 2.35 GiB |
| 64 | 64 | 26.25 | 3.84 | 162.71 | 882% | 2.65 GiB |

The stable end-to-end crossover is 8 threads (10.1% faster than serial);
larger requests waste synchronization and memory even when SYRK alone keeps
getting faster.  The planner now uses work bands of 1/8/16/32 workers at
250k/1M/4M panel-entry thresholds, with this case selecting 8.  A validation
run requesting 128 threads reported `effective_threads=8`, `lp_pack_threads=8`,
and `schur_threads=8`, with identical certificates.  Its wall time was 229.5
s and CPU utilization 1,797% because the Julia process itself was launched
with a 128-thread pool; the recommended launch for this LP is therefore
`--threads=8` (the internal selector cannot shrink Julia's already-created
pool).  A small 256-bit smoke test after the schedule/diagnostic edit passed
with exit status 0 and selected the serial branch below the 250k crossover.

Conclusion: retain the ownership-safe LP BigFloat threading and conservative
automatic crossover in the isolated candidate; do not promote a 128-thread
launch for this problem.  The remaining high-priority optimization is to
remove full-pool overhead in pre-solve/certificate stages when callers start
Julia with far more threads than the selected LP worker count.

### P44 candidate regression gate

The corrected, fully instantiated candidate environment passed the focused LP
suite on node147: high-precision Hessian regressions 10/10, reduced
standard-form LP regressions 26/26, and LP presolve regressions 45/45 (81/81
total, PBS exit status 0).  The first attempt used the Chapter 3 environment,
which intentionally omits MathOptInterface; that harness failure was not a
solver failure and was not counted as a regression.

### P45 Convex.jl frontend integration and native/Convex benchmark harness

Convex 0.16.6 was audited against the existing non-incremental MOI optimizer.
No runtime extension is required: Convex's typed context instantiates
`SDPX.Optimizer{T}`, adds an MOI cache, and applies the standard bridges before
copying the completed conic model.  Formal test-only compatibility, examples,
documentation, and an isolated native-versus-Convex benchmark now cover affine
LP, SOCP, and real PSD models in Float64, Float64x4, and BigFloat.  Extended
arithmetic requires both `numeric_type=T` on the Convex problem and an
`SDPX.Optimizer{T}` factory.  Convex warm starts remain unsupported because
SDPX does not yet implement `MOI.VariablePrimalStart`.

The audit found one frontend integration bug: querying `MOI.RawSolver()`
through Convex's caching optimizer tried to remap an `SDPResult` as though it
contained MOI indices.  `SDPResult` is now passed through unchanged for that
attribute.  The focused frontend matrix passed 66/66 tests, including primal
and dual recovery and an analytic PSD check that does not downcast extended
arithmetic.

The full suite initially exposed nine existing dedicated-LP failures.  The LP
loop had reached `Optimal`, but result construction unconditionally replaced
the certified current iterate with a stale merit snapshot.  The stale result
was then correctly rejected by final certificate validation.  Returning the
current iterate after a successful termination, while retaining snapshots for
interrupted or unsuccessful exits, fixed the issue without changing any
tolerance.  Reproductions after the fix returned gaps of about `7.07e-9` for
the distant Float64 case, `3.2e-21` for Double64, and `2.56e-36` for a
BigFloat-256 case.  The complete local gate passed 5862 tests with one expected
broken test and zero failures in 5m46.5s.  The documentation build also passed.

Tiny one-repetition Mac smoke measurements validated timing boundaries before
the cluster comparison.  At these deliberately small sizes the SDPX core time
was similar through both frontends, while Convex canonicalization, bridging,
copy, and result recovery dominated end-to-end time.  The harness reports
build, frontend overhead, core solver, validation, allocations, process peak
RSS, core dimensions, objective, equality violation, and cone margin
separately.  Its SDP case also exposes the expected representation difference:
the native triangle has fewer variables and no square-cone symmetry rows.

| smoke case | frontend | core (s) | frontend overhead (s) | end-to-end (s) |
|---|---|---:|---:|---:|
| LP, 16 variables | native | 0.000767 | 0 | 0.000914 |
| LP, 16 variables | Convex | 0.000574 | 0.006127 | 0.006788 |
| SOCP, dimension 4 | native | 0.003585 | 0 | 0.003667 |
| SOCP, dimension 4 | Convex | 0.004632 | 0.022714 | 0.027559 |
| SDP, side 4 | native | 0.004213 | 0 | 0.004277 |
| SDP, side 4 | Convex | 0.004449 | 0.004077 | 0.008616 |

All six smoke certificates passed.  The native/Convex SDP objectives were
`2.83897400116` and `2.83897399202`; the small difference is within the
Float64 validation gate.  These numbers are functional smoke data, not the
final performance comparison; repeat medians on one cluster node are required
before drawing a speed conclusion.

The first cluster gate (`199298.node220`, node56) passed 5,865 assertions but
errored in the Float64 square-PSD integration case before benchmarks started.
A focused 3-by-3 diagnostic showed a hardware-stable numerical crossover:
the `1e-9` request ended in `NumericalBreakdown` after an accurate gap but a
`4.10e-7` dual residual, while `1e-8` and `1e-7` were `Optimal` in every run.
At `1e-8` the objective was `2.000000001600999`, the gap was `1.60e-9`, and
the dual residual was `2.13e-33`.  The frontend regression now uses the
package's standard Float64 `1e-8` integration tolerance and checks termination
before attempting variable recovery.  This does not weaken solver accuracy
tests: native Float64 and extended-precision tolerance stress remain in their
dedicated suites.  The failed gate produced no benchmark CSV and was excluded
from performance results.

The corrected candidate then passed the complete node56 package gate: 5,873
tests passed, one existing test remained intentionally broken, and no test
failed (`15m38.1s` test time; `16m06.9s` including the outer `Pkg.test`
process; 2.92 GiB process peak RSS).  Production `current` remained on
`68286ef` throughout.

Matched Float64 measurements used one Julia, solver, and BLAS thread on the
same AMD EPYC 7742 node.  LP and SOCP are five-sample medians; the certified
SDP side-6 crossover is a three-sample median.  All rows used a requested
`1e-8` tolerance and passed their independent equality/cone checks.

| case | frontend | core (ms) | frontend overhead (ms) | end-to-end (ms) | total allocation (MiB) | iterations |
|---|---|---:|---:|---:|---:|---:|
| LP, 512 variables | native | 2.425 | 0 | 6.009 | 2.91 | 11 |
| LP, 512 variables | Convex | 2.344 | 12.044 | 14.798 | 4.86 | 11 |
| SOCP, dimension 24 | native | 7.542 | 0 | 7.833 | 1.53 | 10 |
| SOCP, dimension 24 | Convex | 6.938 | 5.231 | 12.063 | 2.15 | 11 |
| SDP, side 6 | native | 6.040 | 0 | 6.214 | 1.34 | 11 |
| SDP, side 6 | Convex | 5.936 | 5.474 | 11.496 | 2.17 | 11 |

The core solve times agree within normal short-run noise.  End-to-end Convex
overhead is material for these small models: 2.46x for LP, 1.54x for SOCP,
and 1.85x for SDP.  Numerical agreement is strong: LP objectives are exactly
equal in the recorded precision, SOCP objectives differ by `1.80e-10`, and
SDP objectives differ by `2.06e-10`; equality violations are at most
`3.11e-15`, and every cone margin is above `-1e-8`.

The square-PSD representation is the important structural distinction.  At
side 6, native triangle modeling produces 21 variables and 6 equalities;
Convex/MOI square bridging produces 36 variables and 21 equalities.  A side-8
experiment made this 36/8 versus 64/36 and returned an accurate point through
Convex but only `SLOW_PROGRESS`, so it was rejected as a certified timing row.
The original side-12 default also stalled through the native route.  Side 6
is therefore the largest tested model for which both routes produce stable
formal certificates at identical tolerance on this platform, and is now the
benchmark default.  The harness also records primal/dual objectives,
relative gap, primal/dual residuals, and uses the requested tolerance rather
than the previous loose `1e-6` external Convex gate.

## 2026-08-07 — Native LP and Convex triangle frontend

### Retained changes

- Added `linear_program` and `solve_lp`, which accept standard LP matrices and
  construct active-only scalar cone rows. Sparse rows are traversed after one
  CSC transpose, so setup is proportional to stored coefficients.
- MOI multi-variable scalar inequalities now use
  `ActiveSparseCoefficientVector`; equality functions are accumulated once as
  sparse triplets and densified only above a 25% structural density crossover.
- Added an optional Convex extension with `convex_optimizer`, `solve_convex!`,
  and `convex_semidefinite`. The PSD helper defaults to triangle variables and
  retains `representation=:square`. A single sparse unpack map replaced an
  initially tested hcat/vcat expression tree because the tree added measurable
  canonicalization overhead.

### Model-copy benchmark

The deterministic local model has 512 variables, 1,024 four-active
inequalities, and 64 four-active equalities. Five warmed `MOI.copy_to` samples
were measured before and after the change.

| implementation | median time | allocated bytes |
|---|---:|---:|
| expanded rows + dense equality temporaries | 35.5 ms | 116,001,920 |
| active rows + sparse equality triplets | 11.1 ms | 23,026,064 |

This is a 3.2x construction speedup and a 5.0x allocation reduction.

### Convex PSD representation benchmark

Five warmed Float64 samples used one Julia, solver, and BLAS thread on the
same Apple host. All variants solved the deterministic side-6 SDP in 11
iterations and passed the `1e-8` certificate gate.

| frontend | core dimensions `(variables, equalities)` | core | end-to-end | total allocation |
|---|---:|---:|---:|---:|
| native SDPX | (21, 6) | 5.204 ms | 5.294 ms | 1,394,496 B |
| Convex native triangle | (21, 6) | 5.317 ms | 8.447 ms | 2,102,720 B |
| Convex legacy square | (36, 21) | 5.437 ms | 8.663 ms | 2,181,496 B |

The default triangle path is 2.5% faster end to end than the compatible square
path at this small stable size and avoids its quadratic symmetry expansion.
The objectives differ only within the independently accepted Float64 solve
tolerance. Focused validation passed 9 native-LP, 97 MOI, 72 API-surface, and
74 Convex assertions before the full package gate.

## 2026-08-07 — Fixed-trace and compact SOC foundation

### Audit conclusion

Every real symmetric `2x2` PSD block is linearly isomorphic to a three-entry
Lorentz cone. With a fixed trace, the scalar/head coordinate is constant; the
CSDR block `[[q,r],[r,2-q]]` is exactly the unit disk
`(q-1)^2+r^2 <= 1`. The existing solver already contains specialized scalar
`2x2` Cholesky, line-search, sparse contraction, and reduced-arrow kernels, so
an unvalidated wholesale replacement would be high risk. The exact Q3/PSD2
selector is retained after its cluster gates; a general-dimensional Lorentz
Newton backend remains unpromoted.

### Retained first-stage implementation

- Added compact SOC problem/result types and public construction/solve entry
  points. Q3 uses its exact two-by-two PSD isomorphism; other dimensions keep
  the exact PSD-arrow reference.
- Added Lorentz Jordan product, inverse, determinant/interior, and exact
  quadratic fraction-to-boundary primitives.
- Added conservative direct and equality-implied fixed-trace analysis. It
  classifies negative trace as infeasible, zero trace as a forced-zero block,
  positive `2x2` blocks as SOC candidates, and larger fixed-trace blocks as
  traceless-SDP candidates.
- Sparse `2x2` solve setup records in transient `BlockWS` storage which blocks
  have entirely traceless coefficient matrices. Dominant Schur and arrow
  contractions use the two independent coefficient coordinates while the
  persistent serialized model layout remains unchanged.
- Added a sequential `PreparedSolver` boundary that reuses ingested constraint
  storage across a few objectives and can warm start from a successful prior
  result. It is explicitly non-reentrant.

### Validation status

Local work was limited to source loading and formatting checks. Three PBS
compute-node jobs (`199414`, `199415`, `199416`) were submitted in `sugon` for
Float64, Float64x4, and 256-bit BigFloat correctness and certificate checks.
No production symlink was changed.

### Cluster validation update

The corrected compact frontend and fixed-trace analysis passed 26/26 focused
tests for Float64, Float64x4, and 256-bit BigFloat on PBS compute nodes
(`199426`--`199428`). Analytic `min(t)` with `(t,3,4) in Q3` returned
`Optimal` in all arithmetic types. Objective error was `8.53e-9`, `1.71e-14`,
and `1.50e-24`, respectively.

A second same-node representation gate (`199431`--`199433`) compared the
exact two-by-two Q3 isomorphism with the historical three-by-three arrow. The
median single-thread solver times were:

| arithmetic | exact PSD2 | arrow3 | speedup | PSD2 certificate |
|---|---:|---:|---:|---|
| Float64 | 0.004240 s | 0.010089 s | 2.38x | valid |
| Float64x4 | 0.013332 s | 0.042241 s | 3.17x | valid |
| BigFloat-256 | 0.019872 s | 0.514442 s | 25.9x | valid |

The BigFloat arrow3 reference stalled and failed its final certificate, while
the exact PSD2 representation was `Optimal`. The fixed-trace hot flag was then
moved from persistent `SparseCons` storage to transient `BlockWS`; this keeps
old serialized CSDR models and checkpoints layout-compatible. Native and mixed
BigFloat reduced-panel packing now contracts the two independent traceless
coordinates and preserves exclusive MPFR ownership.

### Real CSDR spectral-primal gates

The immutable `J=8, Na=8, Nmu=40, Nx=3` spectral-primal payloads were read
directly from the prior campaign, confirming backward compatibility of the
serialized `SparseCons` model. Production 0.3.1 and the candidate ran
sequentially on the same node with 16 Julia/solver threads, one BLAS thread,
one warm-up, and three measured solves.

For `zero_c00_min`, both versions were `Optimal` in 21 iterations. The
production median was 1.666438 s and the candidate median was 1.060194 s, a
1.57x speedup. For the harder `zero_c10_c11` direction, both versions were
`Optimal` in 108 iterations with bit-identical printed objectives
`-2.429066649954053...e-11`, relative gap `3.03858e-13`, and certificate
metrics. Median wall time improved from 4.891804 s to 4.791984 s. The isolated
Schur phase improved from 0.034755 s to 0.032619 s (6.1%); equality Gram and
factorization dominated the remaining time. Process peak RSS decreased from
1,567,420,416 to 1,543,401,472 bytes in that matched run.

The second direction's candidate certificate had primal/dual residuals
`3.16e-63` and `5.69e-49`; minimum reconstructed two-by-two primal/dual cone
margins were `-7.80e-24` and `1.74e-15`. The tiny negative primal margin is
well inside the requested original-coordinate tolerance. No production path
or symlink was changed.

### Thread scaling and dense-SDP regression

The same `zero_c00_min` CSDR smoke model was run sequentially on one PBS
compute node with one BLAS thread. Every run returned the same objective and
certificate; only the Julia/solver thread count changed.

| Julia threads | median solver time | speedup vs 1 thread |
|---:|---:|---:|
| 1 | 1.648903 s | 1.00x |
| 2 | 1.627795 s | 1.01x |
| 4 | 1.418084 s | 1.16x |
| 8 | 1.140819 s | 1.45x |
| 16 | 1.089007 s | 1.51x |

The last 8-to-16-thread step improved only 4.8%. This small problem contains
parallel bursts but is dominated by short serial phases, compilation, and
model loading; eight threads are the practical smoke-test recommendation.
Larger production models still need an independent thread crossover.

The final Float64 `Task_Low08` regression ran on `node180` with 16 Julia and
16 BLAS threads under interleaved NUMA placement. It returned `Optimal` in 28
iterations with a valid original-coordinate certificate. Solver and
end-to-end times were 43.656 s and 45.414 s; peak process RSS was 4,494,412
KiB. The primal/dual objectives were `0.6532912655025964` and
`0.6532910479425099`, relative gap `2.176e-7`, primal residual `3.316e-10`,
and dual residual `9.534e-12`. The diagnostic minimum primal/dual eigenvalues
were `-7.126e-11` and `1.975e-15`, both accepted by the scale-aware
certificate. Phase totals identify Schur assembly (10.209 s), Schur
factorization (9.496 s), and total KKT factorization (13.875 s) as the main
remaining dense-SDP costs. The fixed-trace equality scan crossover avoided a
new dense relation-analysis cost on this large model.

### Release-gate corrections

The first full v0.4 candidate gate passed 5,942 assertions and failed one
pre-existing LP termination-reason assertion. The new generic fixed-trace
stage correctly found the negative fixed trace of a contradictory scalar
row, but it returned the generic `structural_presolve_infeasibility` reason
before the dedicated LP presolve could return `lp_zero_row_infeasible`.
The final implementation retains fixed-trace infeasibility detection for a
direct `preprocess` call and maps a negative fixed scalar block back to the
specific LP zero-row reason at the solve boundary. A focused compute-node
rerun passed 97 LP, 53 SOC/fixed-trace, and 6 public-API assertions.

The release audit also added an explicit symmetry check to the SOC-arrow
classifier. This protects callers that intentionally disable ingest
validation from misclassifying a nonsymmetric arrow as a Lorentz cone.
Independent docs/examples/benchmark/CLI environments now admit SDPX 0.4,
all advanced API examples use the `SDPX.Experimental` namespace, and a clean
Documenter build completed successfully. The final full package regression
ran as PBS job `199463.node220` and passed all 5,945 assertions in 15m55.2s.
The complete process took 16m44.3s, exited zero, and reached 2,711,188 KiB
peak RSS (`/usr/bin/time`; PBS recorded 3,025,684 KiB). Production remains
unchanged.

## 2026-08-08 — Native fixed-trace Q3 development

### Geometry and preprocessing audit

The new benchmark geometry is deliberately larger than the historical J40
case recorded above.  The development case `J=40, Na=20, Nmu=200, Nx=3` has
4,200 fixed-trace two-by-two PSD cells, 8,400 reduced spectral variables, and
170 equalities.  The high-core case `J=80, Na=40, Nmu=800, Nx=3` has 32,800
cells, 65,600 variables, and 350 equalities.  In particular, the latter panel
is `65,600 x 350`; it is not the smaller historical J80/Na20 panel.

Fixed-trace analysis originally allocated and cleared an `m`-entry dense
trace vector for every block.  That is an O(Lm) preprocessing path: about
2.15 billion writes for the J80 benchmark before the first Newton step.  The
retained implementation now scans only each sparse block's active coefficient
list, recognizes packed traceless PSD2 coefficients directly, and allocates a
dense relation vector only for a genuinely small equality-implied trace
solve.  Direct fixed traces retain an empty relation vector.  The focused
fixed-trace regression passed all 53 assertions after this change.

### Compact scalar kernels and rejected barrier prototype

An allocation-conscious Q3 kernel module now implements the exact map

`(x0,x1,x2) -> [[x0+x1,x2],[x2,x0-x1]]`

together with determinant/margin tests, stable cone fraction-to-boundary,
full nonsymmetric products, inverse-left multiplication, fixed-trace Schur
metrics, predictor contractions, direction recovery, and second-order
corrector residuals.  Float64 and BigFloat256 comparisons against explicit
two-by-two matrix algebra pass 474/474 assertions.  Warm Float64 fraction,
Schur, and corrector calls allocate zero bytes.

A compact primal log-barrier prototype produced accurate Float64x4 and
BigFloat256 answers but required roughly 109 outer iterations at `1e-20`.
It was rejected and removed because it would not improve complete CSDR solve
time; its measured result remains recorded here as a negative experiment.

### Native compact Mehrotra engine

A narrow native engine now accepts only exactly certified sparse fixed-trace
PSD2 products with two nonsingular, block-local traceless coordinates per
cell.  It stores primal and dual cone states in three Q3 coordinates, stores
only two spectral directions per cell, uses closed-form local Schur metrics
and exact Q3 cone steps, and materializes PSD matrices only at the final
compatibility/certificate boundary.  It preserves the established two-row
local equality elimination and triangular equality Gram; fixed trace does not
justify an incorrect two-to-one row reduction.

The first predictor comparison against the unchanged matrix SDP workspace
found a sign error in dual-direction recovery.  The local Schur block,
residual, and primal direction agreed exactly, but the Q3 dual direction was
the negative of SDPX's Newton convention.  The corrected formula is
`dY = sym(X^-1 * (R - dX*Y))`.  A boundary disk then converged in six native
iterations, while a two-cell equality-constrained problem converged in six
iterations at `1e-20` in both Float64x4 and BigFloat256.  The latter returned
relative gap about `1.39e-22`; its primal and dual residuals were at or below
`1e-72` for Float64x4 and `1e-77` for BigFloat256.

The integrated SOC/fixed-trace suite passes 84/84 assertions, and the scalar
kernel suite passes 474/474.  The native engine performs an independent
reduced original-coordinate certificate gate before promotion; an
ineligible post-presolve model, unsupported warm start/checkpoint, numerical
failure, or certificate failure reruns the unchanged PSD2 path when time
remains.  `algorithm=:auto` deliberately continues to select the validated
PSD2 reference until the PBS J40/J80 performance and reliability gates are
complete.

### Equality-Gram scheduling experiment

The first compact backend reused the existing disjoint output-tile SYRK.  On
the `65,600 x 350` J80 panel each tile worker traverses all 65,600 rows, which
can reread a column panel many times and limit NUMA scaling.  An alternative
expert/automatic strategy now gives every worker a contiguous row bin and a
private packed lower triangle.  Each bin is accumulated without shared writes,
then merged in a deterministic binary tree and copied into the solver-owned
lower triangle.  BigFloat reduction mutates only independently owned MPFR
destinations.  A local four-core `2048 x 100` Float64x4 microbenchmark measured
0.0356 s for output tiles and 0.0734 s for row bins after enabling the packed
2x2 micro-kernel.  This rejected automatic row-bin promotion: `:auto` retains
output tiles, while the forced experiment remains subject to the configured
memory fraction.

The new `q3_gram_strategy=:output_tiles|:row_bins|:auto` expert control exists
solely to obtain matched J40/J80 evidence; `:auto` is conservative.  Local
four-thread regressions exercise the packed row-bin path and retain the exact
objective and certificate.  After this addition the scalar Q3 suite passed
474/474 assertions, the integrated fixed-trace/SOC suite passed 101/101, and
the related pipeline, KKT, and BigFloat ownership suites remained green.  Real
performance claims are deferred to PBS compute-node measurements.

### Cluster staging

Candidate `q3-fixed-trace-e9559eb8f4ca` was uploaded to an isolated cluster
path without changing the production symlink.  The first model-generation
jobs are `199489.node220` (J40, node57, 16 cores, 32 GiB) and
`199490.node220` (J80, node58, 16 cores, 64 GiB).  Both use the pinned CSDR
release, Float64x4 source construction, single-threaded BLAS, immutable output
directories, and model SHA-256 manifests.  They were still queued at the time
of the initial entry.  The nodes reported `free` but were actually substantially
occupied, so the two queued jobs were moved to idle node1/node7.  Both then
failed in under three seconds before model construction: PBS had copied the
script away from its repository-relative files and the release-local Project
lacked MultiFloats.  The retained fix requires an explicit `SDPX_SOURCE` and
uses the instantiated shared Julia project/depot; the failed directories remain
as audit evidence and new attempts use new immutable output directories.

The second generation attempt (`199491.node220` and `199492.node220`) also
failed before numerical work.  Its active project and depot were correct, but
the stacked load path placed the SDPX candidate first.  Because SDPX declares
MultiFloats as a weak dependency, that project entry shadowed the instantiated
shared manifest and Julia reported that MultiFloats was not installed.  A
login-node load-only reproduction isolated the ordering bug.  The verified
stack is now `shared environment -> SDPX candidate -> CSDR release -> stdlib`;
it loads MultiFloats from the pinned shared depot, SDPX from the candidate,
CSDRBootstrap from the immutable release, and activates the SDPX MultiFloats
extension.  Generator and solver launchers use this ordering and pass the CSDR
release explicitly to the deserializer.  No failed attempt changed production.

The corrected generation jobs `199493.node220` and `199494.node220` completed
on node1/node7 with exit status zero. J40 took 26.74 seconds in the generator
and produced a 43 MiB payload with SHA-256
`eb9072b252e32d39f00bef78f81c5cd9269c6f65a1d02150b7cf02a694fd46cd`.
J80 took 381.20 seconds and produced a 633 MiB payload with SHA-256
`bbad609457787dc72e92997b91cfc24cebb372e00d28c2f6d66ec1546c735020`;
its peak RSS was 3,927,132 KiB. Although each generator exposed 16 Julia
threads inside a 32-slot allocation, J80 averaged only 1.56 active CPU cores.
This serial source-model construction is reported separately and is excluded
from every solver-time comparison.

Before timing, the harness was hardened against false SOCP claims: a row is
accepted only when the solve is Optimal, the plan is `:socp_fixed_trace_q3`,
the executed KKT is `:q3_block_diagonal_equality`, the original-coordinate
certificate passes, and any forced Gram strategy actually executed. It also
requires the exact model hash, native eligibility, fixed BigFloat256 precision,
the pinned CSDR release, and the requested candidate source. The manifest now
records a serialized reduced-model hash, release/source/Julia-manifest hashes,
effective memory, actual CPU use, context switches, and NUMA residency. A new
preflight-only mode performs conversion and memory estimation without entering
a Newton iteration.

The final backend audit found two correctness gates before the first campaign.
A boundary iterate that already met all tolerances could be discarded when the
unused next Newton factor lost a positive pivot; residual certification now
precedes that failure classification, with a deterministic regression. The Q3
primal residual now uses the exact matrix infinity norm
`abs(p0)+abs(p1)+abs(p2)` and the same data scaling as the public certificate.
BigFloat local panel and triangular solves also now reuse destination-owned
MPFR objects plus one task-local MutableArithmetics scratch value, eliminating
per-panel-entry division temporaries.

### J40/J80 fixed-trace preflight and harness corrections

The first Float64x4 preflight pair (`199501.node220` and `199502.node220`)
failed before numerical work because Julia 1.12 would not call the dynamically
included release reducer from an older compiled world.  The retained loader
uses `Base.invokelatest` for both the release binding and the exact
low-energy-elimination function.  Corrected J40 job `199504.node220` then
passed with the declared reduced geometry: 4,200 PSD2 blocks, 8,400 variables,
and 170 equalities.  Its Q3 workspace estimate was 54,965,608 bytes, process
peak RSS was 2,400,198,656 bytes, model loading/reduction took 12.13 seconds,
and the native eligibility scan took 0.070 seconds.  Focused job
`199503.node220` simultaneously passed 97 LP, 105 SOC/fixed-trace, and six
public-API assertions.

The corresponding J80 Float64x4 preflight `199511.node220` passed on 64
threads with exactly 32,800 blocks, 65,600 variables, and 350 equalities.  It
estimated 800,369,696 bytes for the compact Q3 workspace; complete payload
loading, release-pinned reduction, hashing, conversion, and inspection peaked
at 7,475,322,880 bytes.  Hashing the 633 MiB input took 6.72 seconds and model
loading/reduction took 51.91 seconds, while native eligibility itself took
0.095 seconds.  The planned 64 GiB Float64x4 allocation is therefore safe;
the originally submitted 256 GiB request was cancelled while still queued and
replaced by the measured 64 GiB request.

Two first J40 timing pilots (`199507`/`199508`) exposed that PBS parses commas
inside `qsub -v` as new variables; the paired launcher now accepts the safe
tokens `sdp-socp` and `socp-sdp`.  The corrected launchers then reached the
timing boundary and exposed a second harness-only incompatibility: Julia 1.12
removed the undocumented `Base.cputime()` function.  No Newton iteration was
timed in either failed pilot.  Process CPU accounting now uses POSIX
`clock()`, which accumulates CPU time across all Julia threads and permits the
reported active-core and utilization checks.

J80 BigFloat256 preflight job `199519.node220` converted the immutable
Float64x4 payload inside a fixed 256-bit precision scope and passed the same
32,800/65,600/350 geometry and native-Q3 eligibility gates.  Deterministic
conversion took 24.12 seconds.  The conservative compact workspace estimate
was 4,201,801,504 bytes; the load/reduction/conversion process itself peaked
at 7,509,975,040 bytes (`/usr/bin/time`: 7,333,960 KiB).  Their conservative
sum remains far below the planned 128 GiB allocation, so a future full J80
BigFloat256 solve is not rejected by memory preflight.  The preflight consumed
991.9 process CPU seconds over 256.7 wall seconds (3.86 average active cores),
showing that source conversion is only weakly parallel and must remain outside
solver-time comparisons.

The first valid J40 Float64x4 single-thread SDP row completed in 536.425
seconds and 166 iterations.  It returned `Optimal`, objective
`-21.0253439247550203`, relative gap `6.38e-13`, and a valid original-
coordinate certificate.  Equality Gram construction consumed 315.79 seconds
(59% of solver wall time), constraint triangular solves 56.55 seconds, and
equality factorization 8.73 seconds.  This establishes that fixed-trace cone
algebra alone cannot deliver a large complete-solve speedup: equality-panel
construction/Gram and its downstream solves must also scale.  The paired
second process on node15 later received SIGBUS while executing release-side
model elimination, before a SOCP solve; that row is rejected as a runtime/
node anomaly and will be repeated elsewhere.

A direct no-fallback J40 Q3 diagnostic isolated the first native failure in
7.36 seconds.  The core completed one update, then reported a non-interior
cone state at block 202; the public solver had correctly discarded this state
and rerun PSD2.  The accepted first-step history exposed the mechanism: the
bounded fraction-to-boundary helper returned one for a dual full step.  That
value was ambiguous between a safely interior endpoint and an endpoint exactly
on the closed cone boundary, while the old caller skipped its 0.99 safety
factor whenever the value was one.  The retained fix tracks strict full-step
interiority separately for every worker and applies the safety factor when the
endpoint is only semidefinite.  A dedicated Q3 regression covers this exact
boundary endpoint.

### Native Q3 parallel closure race and repair

The strict-endpoint candidate still failed the direct J40 Float64x4
diagnostic (`199526.node220`) after one update. It reported block 5 as a
non-interior dual state even though the saved head (`0.893259`) and determinant
(`0.161784`) were both strictly positive. A deterministic four-block Float64
regression reproduced the same contradiction with four workers, while the
identical one-worker solve reached `Optimal` in six iterations with a valid
certificate. This separated the failure from CSDR data, MultiFloat arithmetic,
and the equality Gram kernel.

A phase-by-phase one-worker/four-worker differential run found divergence in
the local `2x2` metric before the first Gram construction. The cause was Julia
closure capture: the spawned assembly kernels used local names `x0:x2` and
`y0:y2`, while the enclosing failure-diagnostic branch assigned the same names.
Julia boxed those function-local variables, so every spawned task raced on the
same six scalar bindings despite owning disjoint output blocks. The parallel
step-bound reduction independently reused its task-local accumulator names in
the enclosing scope and had the same defect.

The retained repair gives the diagnostic and task accumulators distinct names,
so every spawned kernel owns its scalar temporaries as well as its output
range. After the change, one and four workers were bit-for-bit identical after
local metric construction, Gram/KKT factorization, predictor and corrector
directions, fraction-to-boundary, and all eight inspected updates. The public
four-worker minimal solve again returned `Optimal` in six iterations with a
valid certificate. The scalar Q3 suite passed 480/480 assertions and the
integrated SOC/fixed-trace suite passed 105/105 locally. Candidate
`q3-fixed-trace-72e335c04cf8` and its failed focused regression
`199525.node220` predate this race repair and remain rejected audit artifacts;
a new immutable candidate is required before repeating J40 timing.

The unaffected one-worker control from the preceding candidate
(`199527.node220`) completed the full J40 Float64x4 native solve in 490.850
seconds and 174 iterations. It returned `Optimal`, objective
`-21.0253439247582131`, relative certificate gap `3.73e-13`, and a valid
original-coordinate certificate. Its dominant phases were equality Gram
construction (334.13 seconds), local residual/metric factorization (32.95
seconds), constraint triangular solves (51.76 seconds), and the 170-by-170
equality factorization (9.22 seconds). Against the validated 536.425-second
one-worker SDP2 row, compact Q3 is only 1.09 times faster and therefore does
not satisfy the automatic-promotion threshold.

Race-fixed candidate `q3-fixed-trace-db9b5d8fd891` then completed the direct
J40 Float64x4 eight-worker diagnostic (`199530.node220`) in 134.093 seconds
with the identical 174 iterations, objective, and certificate. The executed
backend was `q3_block_diagonal_equality` with the output-tile triangular Gram;
no PSD fallback occurred. This is a 3.66-times strong-scaling speedup over the
one-worker Q3 row. Equality Gram time fell from 334.13 to 48.60 seconds
(6.88 times), while local metric work remained 33.50 seconds and the serial
equality factorization took 11.67 seconds. Solver CPU time was 608.28 seconds,
or 4.54 mean active cores out of eight requested, so additional scaling work
must target the remaining serial/local phases rather than the already-scaling
Gram alone. Process peak RSS was 1,301,384 KiB.

The first focused PBS rerun for that candidate failed before loading SDPX:
the script selected the intentionally unversioned `bench/Manifest.toml`, which
was not part of the immutable upload. No numerical test ran. The portable
launcher now reuses the pinned shared cluster environment, places the candidate
package next on `JULIA_LOAD_PATH`, and asserts `pathof(SDPX)` before including
tests. Locally, the integrated SOC suite now passes 111/111 assertions,
including deterministic multithreaded BigFloat256 Q3 execution.

### Residual GEMV scaling and isolated cluster compilation caches

The eight-worker J40 profile showed that the equality Gram already scaled by
6.88 times, but the combined residual/local-metric phase stayed essentially
flat at 33.50 seconds. Two dense residual products, `B*y` and `B' * x`, still
used the serial generic matrix-vector path for `Float64x4` and `BigFloat`.
The retained Q3-only helper partitions complete output rows among fixed tasks,
preserves the established summation order within every dot product, and never
shares a mutable `BigFloat` destination or MPFR scratch object between tasks.
Float32 and Float64 continue to use their unchanged BLAS path.

On a local J40-shaped `8,400 x 170` Float64x4 panel, the two residual products
were bit-for-bit identical to the owned serial kernel. Median elapsed time fell
from 0.05778 seconds to 0.03917, 0.02768, and 0.02019 seconds with 2, 4, and 8
workers, respectively (1.48, 2.09, and 2.86 times speedup). New phase counters
separate `q3_residual_gemv` from `q3_local_metric_factor`, so the cluster run
can determine whether this microkernel saves at least five percent of the full
solve. The direct Float64x4/BigFloat256 regression covers a `200 x 200` panel,
uninitialized MPFR output slots, deterministic results, and alias-safe
reference storage. The scalar Q3 suite passes 480/480 assertions and the
integrated SOC/fixed-trace suite passes 113/113 with four Julia threads.

Paired job `199531.node220` and focused regression `199534.node220` terminated
with SIGBUS in Julia compilation/release loading before a numerical solve.
Their logs showed concurrent `.ji.pidfile` activity in the shared depot from
several immutable candidates that have the same package UUID and version but
different source trees. Every retained PBS launcher now prepends a job-private
compiled depot under its result directory and keeps the shared read-mostly
depot second for installed packages and artifacts. This removes cross-candidate
cache writes without copying the full package store. The two failed jobs are
environmental failures and provide no benchmark row.

The first optional Float64 reference jobs (`199535.node220` and
`199536.node220`) also stopped before solver setup: Julia serialization needs
the concrete CSDRBootstrap payload types loaded before `deserialize`, not
after inspecting the payload. The direct-Q3 reference harness now includes the
pinned release in `Main` first and then delegates one load/reduction to the
shared fixed-trace loader. It records the source and reduced-model hashes,
requires exact Q3 eligibility, separates frontend/setup/solver/validation
times, and maps external SOC primal/dual iterates back to PSD2 for SDPX's
original-coordinate certificate. Clarabel uses one vector `Zeros` equality;
MosekTools uses equivalent scalar `EqualTo` rows because their current MOI
wrappers accept different equality encodings. No tolerance or model relaxation
is introduced by this frontend-only difference.

Focused cluster regression `199538.node220` then passed on node165 with the
job-private depot: 97 LP assertions, 113 compact-SOC/fixed-trace assertions,
and six public-API assertions, with no failures. PBS wall time was 3 minutes
44 seconds and reported peak memory was 1,465,004 KiB. A 16-block synthetic
Float64x4 native-Q3 benchmark also returned `Optimal` in five iterations with
an original-coordinate certificate; a deliberately forced Gram strategy on
the equality-free synthetic case was correctly rejected because no Gram
kernel actually executed.

The first paired rerun `199539.node220` exited in six seconds before Julia
startup because the portable `run_pair.sh` defaulted to an unqualified
`julia`, which is not on compute-node PATH. It now uses the same pinned Julia
1.12.6 executable as the other launchers while retaining an explicit override.
Clarabel reference `199540.node220` loaded the exact J40 payload and direct Q3
model but returned `SLOW_PROGRESS` at `1e-8`; the fail-closed harness rejected
it rather than presenting an uncertified reference. MOSEK reference
`199541.node220` did not start because the cluster environment contains
MosekTools but no Mosek.jl package or license installation. A filesystem audit
found no existing user-owned MOSEK installation, so no MOSEK timing is claimed.

A final harness audit closed several paths that could otherwise produce
mislabelled large runs. Tuple payload coefficients now use a concrete
`Vector{Array{T,3}}` accepted by `ingest`; pre-existing BigFloat tuple/model
objects are rejected instead of silently retaining an unknown MPFR precision.
Legacy same-arithmetic SparseCons blocks are repacked onto their active support
before hashing, avoiding more than two billion retained references at J80.
Geometry, model hash, exact fixed-trace/Q3 eligibility, executed backend, Gram
strategy, and a 70%-of-effective-free-memory workspace gate are mandatory for
every serialized timed run. Provenance hashes now cover the numerical source
and fixed-trace benchmark drivers/launchers, and the optional reference asserts
that `pathof(SDPX)` belongs to the requested immutable candidate. MOSEK and
Clarabel now share recorded iteration/time/thread controls and use a fresh
optimizer for each measured repetition.

### J40 paired result and fixed-extended equality factorization

Paired job `199545.node220` completed on node58 with eight Julia/solver
threads and one BLAS thread. Both modes used the same hash-pinned J40 model,
requested tolerance, adaptive policy, and original-coordinate certificate.
The native Q3 run returned `Optimal` in 174 iterations with primal objective
`-21.025343924758213146...`, relative certificate gap `3.73e-13`, and a valid
certificate. Its solver time was 107.081 seconds and end-to-end time was
117.076 seconds. The PSD2 reference returned `Optimal` in 166 iterations with
primal objective `-21.025343924755020299...`, a valid certificate, solver time
210.915 seconds, and end-to-end time 220.834 seconds. Thus the compact Q3 path
was 1.97 times faster in the solver and 1.89 times faster end to end on this
controlled pair. This is the first complete-pair result above the 1.5-times
promotion threshold, but more repetitions are still required to establish the
coefficient of variation before `:auto` may select Q3.

The residual-GEMV change explains a material part of the improvement over the
earlier 134.09-second Q3 run: the new eight-thread residual phase took 9.059
seconds and the separately measured local metric phase 1.684 seconds, compared
with 33.497 seconds when both were reported together. Equality Gram remained
the largest Q3 phase at 47.288 seconds. Equality Cholesky still took 11.789
seconds and reported one active factor thread; average whole-process CPU usage
was 5.52 cores of the requested eight.

A local 170-by-170 Float64x4 experiment then compared the existing scalar
Cholesky with the already available 24-column blocked kernel. The blocked
median was 0.01029, 0.00695, 0.00579, and 0.00484 seconds at 1, 2, 4, and 8
threads. It reconstructed the input with relative infinity error
`1.78e-63`, and its lower factor was bitwise identical at one and eight
threads. Earlier measurements put the generic factor at 0.02076 seconds, so
the eight-thread blocked kernel is about 4.3 times faster. A conservative Q3
selector now uses this kernel only for fixed-width extended arithmetic and
equality dimension at least 128, with panel size 24. Float64 retains LAPACK;
native BigFloat retains the MPFR kernel. Executed factor kernel and thread
count are recorded, and a failed blocked factor retries the previous generic
factor from the untouched Gram matrix.

The optional direct-Q3 Clarabel 0.11.1 run on the raw J40 model was rejected:
it stopped after one iteration with zero step and `SLOW_PROGRESS`, despite
equilibration, rather than producing a certificate. Its model had 8,400
variables, 170 zero-cone rows, 4,200 Q3 cones, and 1,222,200 nonzeros. The
cluster environment still lacks Mosek.jl and a MOSEK license installation, so
no MOSEK timing is claimed. These are reference-solver/environment outcomes,
not SDPX numerical rows.

Candidate `q3-fixed-trace-18136b04e1b6` contains the blocked equality factor
selector and passed its compute-node focused gate in job `199551.node220`: 97
LP assertions, 121 SOC/fixed-trace assertions, and six public-API assertions,
with no failures. Three-run J40 Float64x4 scaling points then executed the
native Q3 backend and the blocked factor on every applicable repetition. All
18 runs at 1/2/4/8/16/32 threads returned `Optimal`, used 174 iterations,
passed the original-coordinate certificate, and produced numerically identical
objectives and PSD margins.

| Julia/Q3 threads | Solver median (s) | End-to-end median (s) | Equality Gram median (s) | Equality factor median (s) | Mean active cores | End-to-end CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 475.957 | 485.461 | 333.189 | 2.927 | 1.00 | 0.17% |
| 2 | 289.561 | 298.937 | 170.997 | 1.984 | 1.89 | 1.25% |
| 4 | 163.375 | 172.957 | 90.575 | 1.352 | 3.50 | 0.22% |
| 8 | 95.444 | 105.018 | 49.280 | 1.105 | 6.00 | 0.67% |
| 16 | 53.238 | 63.074 | 26.726 | 0.849 | 10.26 | 1.22% |
| 32 | 34.165 | 43.850 | 15.368 | 0.817 | 16.04 | 2.92% |

Relative to one thread, solver-phase speedups are 1.64, 2.91, 4.99, 8.94, and
13.93 times at 2/4/8/16/32 threads. Parallel efficiency consequently declines
from 82% at two threads to 44% at 32, while measured active-core usage reaches
only about 16 cores at the 32-thread request. The 8-thread candidate improves
the preceding same-model run from 107.081 to
95.444 seconds in solver phases (12.2%) and from 117.076 to 105.018 seconds
end to end (11.5%). Equality factorization fell from 11.789 to 1.105 seconds,
accounting for the measured improvement without changing iteration count or
the certificate. From 8 to 16 threads the solver speedup is 1.79 times; from
16 to 32 it is 1.56 times. Mean active-core utilization declines from about
75% of the requested eight to 50% of the requested 32, so higher thread counts
remain experiments rather than automatic defaults. Some 2/4- and 16/32-thread
jobs shared node58 on disjoint scheduler slots, and are retained as exploratory
scaling points; the eventual best configuration needs an isolated repetition.

Two follow-up J40-shaped local kernel experiments were rejected rather than
added to the candidate. Replacing the two panel divisions with precomputed
reciprocal multiplications improved the isolated transform by 1.49, 1.30, and
1.09 times at 1/2/4 threads but was 2.4% slower at eight threads; the relative
result difference was `2.53e-64`. The high-thread path is already limited by
panel traffic, so this does not satisfy the complete-solve five-percent gate.
A column-owned Float64x4 GEMV preserved the result bit for bit and was about
1.20 times faster at 2/4/8 threads, but 1.82 times slower at one thread. Four-
and eight-row accumulator microkernels were slower at every tested thread
count. Because only part of the 9.1-second residual and 16.8-second KKT-solve
phases uses the strided orientation, the measured Amdahl benefit is below the
retention threshold on J40. The column-owned variant remains a J80-only NUMA
experiment; it is not enabled in solver code.

### BigFloat256 Q3 Gram profile and output-tile selector

The first complete J40 BigFloat256 native-Q3 run (`199547.node220`, four
requested solver threads) returned `Optimal` in 174 iterations and passed the
original-coordinate certificate. Solver phases took 1,818.215 seconds and
end-to-end time was 1,917.509 seconds. The primal objective was
`-21.025343924758213146233748041163...` and the relative certificate gap was
`3.73e-13`, consistent with the Float64x4 result. Peak process RSS was about
2.71 GiB, but the solve generated about 106.8 GiB of allocation traffic.

Profiling exposed one dominant selector error rather than an MPFR conversion
cost. Equality Gram construction took 1,585.792 seconds, 87.2% of all solver
phases, and diagnostics reported `pairwise_gram`; whole-process utilization
averaged only 1.25 of four requested cores. Model conversion took 1.411
seconds and reduced-model hashing 3.963 seconds, so neither is a worthwhile
target at this scale. The generic crossover assumes only 1.08 reuse for
BigFloat and assigns it no parallel gain, predicting 1.08 times speedup below
its 1.12 acceptance threshold. That model is inappropriate for native Q3:
the transformed equality panel already exists and the BigFloat triangular
SYRK assigns complete output tiles and private MPFR scratch to workers.

The compute-node microbenchmark `199560.node220` used the J40 equality-panel
shape, `8,400 x 170`, at BigFloat256 and four Julia threads. After warm-up,
pairwise Gram took 9.605 seconds, output tiles took 3.250 seconds, and four
private row bins plus deterministic merge took 3.327 seconds. Output tiles
were therefore 2.96 times faster; both alternatives agreed with the pairwise
lower triangle to the printed precision. The process used 669,156 KiB peak
RSS. Smaller local checks found 1.00, 2.06, and 3.40 times output-tile speedup
at one, two, and four workers on a 1,024-by-64 panel; a 512-by-48 check was
bit-for-bit identical and 3.21--3.29 times faster at four workers.

Follow-up compute-node jobs used the same J40-shaped panel. At one thread
(`199562.node220`), pairwise/output-tile/row-bin times were
8.651/8.568/8.761 seconds, so output tiles were only 1.01 times faster. At two
threads (`199563.node220`) they were 8.649/5.959/6.122 seconds, a 1.45-times
output-tile speedup. All reported lower-triangle relative errors were zero.
Row bins were 3--4% slower than output tiles at both two and four threads, so
they remain a forced experiment rather than an automatic choice. Automatic
BigFloat selection therefore requires at least two effective Gram workers;
the one-worker path stays pairwise.

The Q3 selector now honors an explicit `q3_gram_strategy=:output_tiles` even
when the generic packing model rejects it. In automatic mode, BigFloat uses
output tiles only when at least two workers are selected, the equality
dimension is at least 32, and triangular work is at least 250,000 scalar
contractions. Small panels retain pairwise Gram. Diagnostics now report the
actual Gram worker count and selection reason. A focused test uses a
400-by-40 BigFloat256 panel to verify automatic selection and exact lower-
triangle agreement, and a complete native solve verifies the forced strategy,
objective, iteration count, and certificate. The combined SOC/fixed-trace
suite passes 131/131 assertions with four Julia threads, including an explicit
check that `extended_precision_blas=:off` still disables the automatic Q3
override.

### Complete J40 BigFloat256 output-tile solve

Job `199571.node220` completed on node102 with four Julia/Q3 workers and one
BLAS worker.  The automatic selector executed
`threaded_output_tile_triangular_syrk` with four Gram workers and reported
`q3_bigfloat_parallel_output_tiles`; no PSD2 fallback occurred.  The result
was `Optimal` after 174 iterations and passed the original-coordinate
certificate.  The primal and dual objectives were
`-21.025343924758213146...` and `-21.025343924766050272...`, respectively,
with relative gap `3.7275e-13`, normalized primal/dual residuals
`9.651e-77`/`9.932e-32`, equality backward error `2.734e-79`, and reconstructed
primal/dual PSD2 margins `-1.355e-33`/`2.786e-17`.  These values agree with the
old pairwise-Gram run.

| J40 BigFloat256, four workers | Pairwise Gram | Output tiles | Speedup |
| --- | ---: | ---: | ---: |
| Solver phases | 1,818.215 s | 838.745 s | 2.17x |
| End to end | 1,917.509 s | 940.127 s | 2.04x |
| Equality Gram | 1,585.792 s | 581.201 s | 2.73x |
| Peak process RSS | 2.90 GB | 2.34 GB | 19.5% lower |

The timed process averaged 3.30 active cores, or 82.5% of the four requested
workers, compared with about 1.25 active cores for the serial pairwise
baseline.  Allocation traffic changed only from 114.64 to 114.16 GB, so the
wall-time and RSS improvements come from disjoint output ownership rather than
weakened validation or reduced arithmetic.  Schur assembly is still 680.477
seconds (81.1% of the solver); equality Gram alone is 581.201 seconds (69.3%).
The next retained experiment is therefore an eight-worker complete solve,
followed by higher widths only while measured Gram and complete-solve scaling
remain useful.

### Stable J40 Float64x4 SDP2 versus native Q3 comparison

Job `199573.node220` completed three timed SDP2 reference solves on node104
with eight solver workers and one BLAS worker.  All three returned `Optimal`,
took 166 iterations, passed the original-coordinate certificate, and produced
identical objectives.  The primal objective was
`-21.025343924755020298714...`; the corresponding three-run native Q3 result
used 174 iterations and differed by only `1.52e-13` relative to the objective
scale.  Every Q3 repetition also passed the certificate.

| J40 Float64x4, eight workers | SDP2 median | Native Q3 median | SDP2 / Q3 |
| --- | ---: | ---: | ---: |
| Solver phases | 187.627 s (CV 0.80%) | 95.444 s (CV 0.72%) | 1.97x |
| End to end | 196.935 s (CV 0.68%) | 105.018 s (CV 0.67%) | 1.88x |
| Equality Gram | 46.534 s | 49.280 s | 0.94x |
| Equality factorization | 12.274 s | 1.105 s | 11.11x |
| Residual and local block factor | 27.802 s | 9.477 s | 2.93x |
| Predictor | 22.690 s | 10.619 s | 2.14x |
| Corrector | 55.627 s | 9.488 s | 5.86x |
| Refinement | 30.424 s | 0.000 s | -- |
| Mean active cores | 3.17 | 6.00 | 1.89x |
| Allocation traffic | 532.94 MB | 463.31 MB | 13.1% lower |

This isolates where the compact formulation wins.  Its equality Gram has the
same `8,400 x 170` geometry and is slightly slower in this comparison; fixed
trace cannot legally reduce its two transformed rows per cell to one.  The
speedup instead comes from closed-form cone-local algebra, the blocked
extended-precision equality factor, cheaper direction recovery, and avoiding
the PSD2 refinement work.  Q3 also keeps twice as much of the requested CPU
capacity active.

Peak process RSS is dominated by compilation and the common serialized model:
the separate three-run jobs reported 2.46 GB for Q3 and 2.30 GB for SDP2, but
the controlled same-allocation pair reported 2.25 GB for Q3 versus 2.43 GB for
SDP2.  Its solver workspace estimates were 55.0 MB for Q3 versus 34.0 GB for
the deliberately conservative SDP campaign estimate.  Allocation traffic and
the paired RSS therefore support the no-memory-regression gate; process-level
RSS from different candidate compilation jobs is not treated as a solver
workspace measurement.  The speed and CV gates are now satisfied at eight
workers.  A same-allocation one-worker pair is required before changing the
automatic algorithm selector.

### J40 BigFloat256 scaling from four to eight workers

Job `199575.node220` completed on node165 with eight Julia/Q3 workers and one
BLAS worker.  It again selected `output_tiles`, used eight disjoint Gram jobs,
returned `Optimal` in 174 iterations, and passed the original-coordinate
certificate.  Its primal/dual objectives, `3.7275e-13` relative gap, residuals,
and reconstructed PSD2 margins were identical to both the four-worker output-
tile run and the old pairwise baseline.

| J40 BigFloat256 | 4 workers | 8 workers | 4-to-8 speedup |
| --- | ---: | ---: | ---: |
| Solver phases | 838.745 s | 485.913 s | 1.73x |
| End to end | 940.127 s | 584.476 s | 1.61x |
| Equality Gram | 581.201 s | 300.528 s | 1.93x |
| Complete Schur | 680.477 s | 370.552 s | 1.84x |
| Mean active cores | 3.30 | 5.63 | 1.71x |
| Peak process RSS | 2.34 GB | 2.51 GB | 7.2% higher |

Equality Gram achieved 96.7% parallel efficiency from four to eight workers,
and complete Schur achieved 91.8%.  Relative to the original four-worker
pairwise run, the eight-worker solver is 3.74 times faster and end to end is
3.27 times faster while peak RSS remains 13.7% lower.  The eight-worker run
allocated 114.17 GB over its lifetime, effectively unchanged from the
four-worker output-tile run; its modest RSS increase is the cost of additional
task stacks and MPFR scratch, not replicated output matrices.  Gram remains
61.8% of the solver and Schur 76.3%, so a sixteen-worker complete solve is the
next useful scaling point.  Wider BigFloat runs are justified only if that
point retains material complete-solve scaling.

### J40 BigFloat256 scaling from eight to sixteen workers

Job `199577.node220` completed on node8 with sixteen Q3 workers and one BLAS
worker.  Automatic selection used sixteen disjoint output-tile Gram workers.
The solve again returned `Optimal` after 174 iterations, passed the independent
certificate, and reproduced the four- and eight-worker objective, gap,
residuals, and PSD2 margins.

| J40 BigFloat256 | 8 workers | 16 workers | 8-to-16 speedup |
| --- | ---: | ---: | ---: |
| Solver phases | 485.913 s | 312.057 s | 1.56x |
| End to end | 584.476 s | 424.661 s | 1.38x |
| Equality Gram | 300.528 s | 167.539 s | 1.79x |
| Complete Schur | 370.552 s | 221.209 s | 1.68x |
| Mean active cores | 5.63 | 9.03 | 1.61x |
| Peak process RSS | 2.51 GB | 2.51 GB | effectively unchanged |

Gram retained 89.7% parallel efficiency and complete Schur 83.8% from eight
to sixteen workers.  Relative to the old four-worker pairwise run, solver
phases are 5.83 times faster and end to end is 4.50 times faster, with peak RSS
still 13.5% lower.  Lifetime allocation traffic remains essentially fixed at
114.19 GB.  Parallel efficiency of the complete solve is now falling because
predictor, local solves, and residual work do not scale as strongly, but Gram
still consumes 53.7% of solver time.  One thirty-two-worker confirmation is
therefore justified; wider BigFloat J40 runs require a material gain at that
point.

### J40 BigFloat256 thirty-two-worker confirmation

Job `199578.node220` completed on node15 with thirty-two Q3 workers and one
BLAS worker.  The automatic output-tile path used thirty-two Gram workers,
returned `Optimal` in 174 iterations, passed the original-coordinate
certificate, and reproduced every reported numerical quantity from the
four/eight/sixteen-worker solves.

| J40 BigFloat256 | 16 workers | 32 workers | 16-to-32 speedup |
| --- | ---: | ---: | ---: |
| Solver phases | 312.057 s | 216.944 s | 1.44x |
| End to end | 424.661 s | 346.028 s | 1.23x |
| Equality Gram | 167.539 s | 88.698 s | 1.89x |
| Complete Schur | 221.209 s | 138.748 s | 1.59x |
| Mean active cores | 9.03 | 15.34 | 1.70x |
| Peak process RSS | 2.51 GB | 2.97 GB | 18.1% higher |

Equality Gram still achieved 94.4% parallel efficiency from sixteen to
thirty-two workers and remains 40.9% of solver time.  The whole solve is less
efficient because the equality constraint solve, predictor, local residual
work, and serial BigFloat equality factor now account for a larger fraction.
Relative to the old four-worker pairwise run, solver phases are 8.38 times
faster and end to end is 5.53 times faster, while allocation traffic is still
essentially unchanged.  RSS has now risen 2.1% above that old baseline.
Thirty-two workers are therefore the conservative recommendation.  One final
sixty-four-worker experiment is permitted only as a crossover check: it must
improve end to end materially without an unacceptable RSS increase to replace
the thirty-two-worker recommendation.

### Same-allocation one-worker Float64x4 promotion gate

Paired job `199576.node220` ran native Q3 followed by the SDP2 reference in
separate Julia processes on node7 with one solver/BLAS worker.  Both modes used
the same model hash, tolerance, adaptive controller, scaling state, and
original-coordinate certificate.  Both returned `Optimal`; Q3 used 174
iterations and SDP2 used 166.

| J40 Float64x4, one worker | Native Q3 | SDP2 | Q3 improvement |
| --- | ---: | ---: | ---: |
| Solver phases | 512.567 s | 561.581 s | 9.6% |
| End to end | 523.575 s | 572.001 s | 9.2% |
| Allocation traffic | 700.75 MB | 1,994.03 MB | 64.9% lower |
| Peak process RSS | 2.395 GB | 2.452 GB | 2.3% lower |
| Mean active cores | 0.999 | 0.999 | equivalent |

The objectives agree to `1.52e-13` relative scale and both certificates are
valid.  Q3 therefore satisfies the no-single-worker-regression gate rather
than relying only on its 1.97-times eight-worker median speedup.  Together
with the sub-one-percent three-run coefficients of variation and the
same-allocation memory reduction, this supports a conservative automatic Q3
selector for large exact fixed-trace Float64x4 models.  Float64 and BigFloat
remain separate policy decisions until they have equivalent formulation-level
reference evidence.

The retained automatic policy encodes that evidence rather than recognizing
all mathematically eligible blocks as a performance win. It requires sparse
fixed-width arithmetic with at least four Float64 limbs, 4,096 blocks, 8,192
variables, and 128 equalities, in addition to the exact native-Q3 structural
certificate. Float64, BigFloat, smaller problems, explicit equilibration, and
unsupported structures continue to select PSD2 automatically. Explicit
`algorithm=:socp` remains the expert override. Dimension-boundary regressions
and the complete SOC/fixed-trace suite pass 138/138 assertions with four Julia
threads.

### J40 BigFloat256 sixty-four-worker crossover and thread cap

Job `199580.node220` completed on node19 with sixty-four Q3 workers and one
BLAS worker.  It used sixty-four output-tile Gram workers, returned `Optimal`
after 174 iterations, passed the original-coordinate certificate, and exactly
matched every numerical quantity in the 4/8/16/32-worker runs.

| J40 BigFloat256 | 32 workers | 64 workers | 32-to-64 speedup |
| --- | ---: | ---: | ---: |
| Solver phases | 216.944 s | 170.960 s | 1.27x |
| End to end | 346.028 s | 297.381 s | 1.16x |
| Equality Gram | 88.698 s | 54.265 s | 1.63x |
| Complete Schur | 138.748 s | 98.478 s | 1.41x |
| Mean active cores | 15.34 | 25.03 | 1.63x |
| Peak process RSS | 2.97 GB | 2.91 GB | 1.9% lower |

Relative to the original four-worker pairwise run, sixty-four workers are
10.64 times faster in solver phases and 6.43 times faster end to end.  RSS is
essentially equal to that old baseline and allocation traffic remains nearly
constant.  The final doubling has only 63.4% solver and 58.2% end-to-end
parallel efficiency, while voluntary/involuntary context switches rise to
about 2.01 million/0.59 million.  Equality Gram is now 31.7% of solver time;
constraint solves, predictor work, and serial equality factorization limit the
rest.  The measured J40 latency cap is therefore 64 workers.  Thirty-two is
the more efficient throughput choice, and 128 workers are reserved for the
larger J80 panel rather than tested on J40.

### J40 BigFloat256 PSD2 reference and formulation-fairness audit

The memory preflight for the legacy PSD2 formulation completed in job
`199581.node220`. It estimated a conservative 648.0 MB equality-arrow workspace
against 249.1 GB available memory, while the preflight process itself peaked at
about 2.41 GB. The complete 64-worker reference solve then ran as job
`199582.node220`. It returned `Optimal` after 166 iterations and passed the
original-coordinate certificate. The primal and dual objectives were
`-21.025343924759423761...` and `-21.025343924764005096...`, the relative gap
was `2.1790e-13`, the normalized primal/dual residuals were
`6.044e-77`/`2.206e-55`, and the primal/dual PSD margins were
`2.050e-34`/`1.888e-17`.

| J40 BigFloat256, 64 requested workers | PSD2 reference | Native Q3 |
| --- | ---: | ---: |
| Solver phases | 1,832.769 s | 170.960 s |
| End to end | 1,960.859 s | 297.381 s |
| Equality Gram | 1,654.373 s | 54.265 s |
| Iterations | 166 | 174 |
| Mean active cores | 4.18 | 25.03 |
| Peak process RSS | 3.139 GB | 2.909 GB |

The raw ratios are 10.72x in solver phases and 6.59x end to end, with a
`5.76e-14` relative primal-objective difference and valid certificates on both
sides. They are not yet accepted as the formulation-level performance result:
the PSD2 equality Gram used the old serial pairwise selector and consumed 90.3%
of its solver time, whereas Q3 used the new ownership-safe output-tile kernel.
That selector asymmetry is repairable and would exaggerate the benefit of the
compact formulation.

The next candidate therefore applies the measured BigFloat equality-panel
crossover to the generic block-arrow PSD2 path as well. A panel already stored
in `Btil` pays no sparse packing cost, so automatic mode now enables disjoint
output tiles when the equality dimension is at least 32, triangular work is at
least 250,000 scalar contractions, and at least two output workers are
available. One-worker and small-panel cases remain pairwise, and
`extended_precision_blas=:off` remains an absolute override. The focused
regression constructs a 400-by-40 BigFloat256 panel, verifies the threaded
kernel label, and requires exact lower-triangle equality with the pairwise
reference. The complete SOC/fixed-trace suite passes 143/143 assertions after
this change. A new complete PSD2 run is required before the selector is retained
or the Q3-versus-SDP speed ratio is reported as final.

### Candidate e52fb812 focused cluster gate

The candidate carrying the generic BigFloat equality-Gram crossover was frozen
at `<cluster-root>/SDPX.jl/candidates/q3-fixed-trace-e52fb812f7c9/source`.
Local and remote hashes matched for the package metadata, Q3/selector sources,
benchmark driver, regression tests, and this work log before the tree was made
read-only. Focused PBS job `199593.node220` ran on node104 and exited zero with a
`PASSED` marker. It passed 246 assertions: 97 LP tests, 143 compact SOC and
fixed-trace tests, and 6 public-API tests. The four-thread test process took
217.07 seconds wall time, peaked at 1,501,396 KiB RSS, and loaded the candidate
path explicitly rather than the production symlink.

### First J80 Float64x4 64-worker attempt and diagnostic retention

The first complete J80 attempt, job `199574.node220` on node103, reached the
configured 7,200-second solver limit and returned `TimeLimit`; it is therefore
not a valid benchmark row and supplies no objective or certificate claim. The
whole process elapsed 2:04:18 including loading and compilation, used
168,124.5 user seconds plus 1,348.0 system seconds (22.72 mean active cores),
and peaked at 7,849,700 KiB RSS. It also incurred 17.59 million voluntary and
10.14 million involuntary context switches. This is direct evidence that the
J80 workload is large enough for high-core experiments, but the requested 64
workers were only about 35.5% utilized on average and the current solver did not
finish within two hours.

The strict benchmark gate correctly exited nonzero, but it raised before
writing the result row, losing the partial iteration/phase diagnostics needed
to decide the next optimization. The harness now builds and writes the timed
row and manifest first, including `benchmark_valid`, `execution_valid`, and the
validation error, then exits nonzero for a non-`Optimal` status or invalid
certificate. A synthetic one-iteration regression exits one while preserving
an `IterLimit` row and both TOML artifacts; the normal five-iteration synthetic
solve remains `Optimal` and exits zero. Warm-up failures remain fail-fast.

### Rejected native-BigFloat blocked equality factorization

An isolated 256-bit factorization experiment tested the 170-by-170 equality
matrix size used by J40. The first driver attempt was invalid and was discarded:
it used Julia's shallow `copy(Matrix{BigFloat})`, so MPFR in-place operations
aliased and corrupted the nominally read-only source matrix. The corrected
`199597.node220` driver used `alloc_zeros` plus `copy_owned!` for every matrix
and vector copy. The generic and blocked factors were then deterministic, their
factor and solve residuals were approximately `1e-77`, and the blocked result
differed from the generic lower factor only at approximately `5e-77`.

| BigFloat256 equality Cholesky, n=170 | Time | Generic / blocked |
| --- | ---: | ---: |
| Generic `kchol!` | 0.0859 s | 1.00x |
| Blocked, 1 worker | 0.1646 s | 0.52x |
| Blocked, 2 workers | 0.2323 s | 0.37x |
| Blocked, 4 workers | 0.1313 s | 0.65x |
| Blocked, 8 workers | 0.1198 s | 0.72x |
| Blocked, 16 workers | 0.1189 s | 0.72x |
| Blocked, 32 workers | 0.1431 s | 0.60x |

The blocked kernel is 38--63% slower at every tested width. The generic native
BigFloat factorization therefore remains selected; no solver code changed. The
experiment also confirms that benchmark code must never use shallow array copies
around mutation-capable MPFR kernels.

### Formulation-fair J40 BigFloat256 PSD2 rerun

Job `199598.node220` ran the generic PSD2 block-arrow formulation on node107
with the retained BigFloat output-tile equality-Gram selector, 64 Julia/solver
workers, and one BLAS worker. It returned `Optimal` after 166 iterations and
passed the original-coordinate certificate. The primal/dual objectives were
`-21.025343924759423761...` and `-21.025343924764005096...`, the relative gap
was `2.1790e-13`, the normalized primal/dual residuals were
`6.044e-77`/`2.206e-55`, and both reconstructed PSD margins were positive.

| J40 BigFloat256, 64 requested workers | Fair PSD2 | Native Q3 | PSD2 / Q3 |
| --- | ---: | ---: | ---: |
| Timed solver phases | 235.833 s | 170.960 s | 1.38x |
| `solve!` wall time | 356.099 s | 290.853 s | 1.22x |
| End to end | 363.374 s | 297.381 s | 1.22x |
| Equality Gram | 80.018 s | 54.265 s | 1.47x |
| Iterations | 166 | 174 | 0.95x |
| Mean active cores | 23.79 | 25.03 | 0.95x |
| Peak process RSS | 2.929 GB | 2.909 GB | 1.01x |
| Allocated bytes | 73.135 GB | 114.284 GB | 0.64x |

The generic selector reduced the PSD2 equality-Gram phase from 1,654.373 to
80.018 seconds (20.68x) and the timed solver phases from 1,832.769 to 235.833
seconds (7.77x), without changing the 166-iteration numerical result. This is
a stable, correctness-preserving improvement and removes the earlier
formulation-fairness blocker. Compact Q3 remains faster, but its accepted
BigFloat256 advantage is now 1.22x by `solve!`/end-to-end wall time and 1.38x
by the sum of instrumented solver phases, not the misleading 6--11x raw ratio
against a serial PSD2 Gram. Q3 spends fewer seconds in equality Gram despite
eight more iterations, while PSD2 allocates substantially fewer total bytes;
peak RSS is essentially equal. One-run, cross-node noise still requires a
repeat before treating the precise ratio as final, but the formulation result
and the generic BigFloat selector both clear the five-percent retention gate.

Same-node node107 repeat `199604.node220` reproduced the exact 166-iteration
objectives, residuals, gap, and PSD margins. It measured 227.068 seconds in
instrumented phases, 346.594 seconds in `solve!`, 353.817 seconds end to end,
and 72.229 seconds in equality Gram. Across the two node107 runs, the medians
are 231.450/351.346/358.595/76.124 seconds respectively. Solver-wall and
end-to-end CVs are 1.91% and 1.88%; the individually shorter Gram phase is
noisier at 7.24% CV but the complete solve is reproducible. Peak process RSS
fell from 2.929 to 2.655 GB in the repeat. The generic BigFloat output-tile
selector is therefore retained; its complete-solve speedup over the historical
serial-Gram PSD2 run is not a favorable single-run artifact.

### J80 Float64x4 64-worker retained diagnostic row

Job `199599.node220` reran the J80 native-Q3 formulation on node109 with the
new failure-preserving benchmark harness, the default Julia thread pool, 64
solver workers, and one BLAS worker. The configured solve limit was 900
seconds. The solve returned `TimeLimit` after 170 iterations and the harness
correctly wrote the complete row and manifest before exiting nonzero. This is
diagnostic evidence only: `benchmark_valid=false`, and no objective is accepted
as a solved CSDR result.

| J80 Float64x4 default pool, 64 requested workers | Measurement |
| --- | ---: |
| `solve!` wall time | 905.975 s |
| End to end around the timed solve | 917.763 s |
| Instrumented solver phases | 650.541 s |
| Equality Gram | 486.260 s |
| Complete Schur assembly | 568.488 s |
| Residual/block factorization | 26.826 s |
| KKT constraint triangular solves | 55.401 s |
| Mean active cores | 31.29 |
| Peak process RSS | 7.993 GB |
| Voluntary/involuntary switches during `solve!` | 220,466 / 34,603 |

The equality Gram is 74.7% of instrumented solver time and 85.5% of Schur
assembly, so it is the primary arithmetic bottleneck. Mean utilization is
48.9% of the 64 requested workers, materially better than the first two-hour
attempt but still leaves half the allocation idle on average. The terminal
iterate already has primal/dual normalized residuals of `1.99e-61` and
`8.29e-33`; the rejected certificate is caused by the `1.997e-5` relative gap.
Thus the J80 difficulty combines expensive Gram construction with slow late
complementarity reduction rather than primal or equality infeasibility.

The complete process took 19:45 because model loading, precompilation, setup,
validation, and failure reporting are outside the 900-second `solve!` limit.
The solver returned only at an iteration boundary (5.98 seconds beyond the
nominal limit), which is acceptable for this iteration cost but should remain
visible in time-limit documentation. A same-node exact-pool control was
submitted as job `199603.node220` with 64 default threads, zero interactive
threads, one GC thread, `JULIA_EXCLUSIVE=1`, and otherwise identical solver
and model settings. The weighted-contiguous SYRK experiment remains blocked
until that no-code control is complete.

### Native-Q3 adaptive-control audit and terminal-limit correctness

A line-by-line audit found no sign, factor-of-two, complementarity, cone-root,
or residual-scaling error in the compact HKM/Mehrotra equations. The affine
predictor uses `R=-XY`, the corrector uses
`R=sigma*mu*I-XY-dX_aff*dY_aff`, primal and dual fraction-to-boundary roots
are exact for Q3, and `mu=dot(qx,qy)` is exactly `tr(XY)/2` for a PSD2 block.
The J80 terminal residual/gap split is therefore a genuine complementarity
tail, not a certificate normalization artifact.

The audit did expose an interface gap: the native Q3 loop currently applies
automatic cold-start `OmegaP/OmegaD` only. It does not call the general
`AdaptiveIPMController`; `beta`, `gamma`, `parameter_strategy`, and
`adaptive_sigma_max` do not affect its iteration controller. Q3 instead uses
the local Mehrotra rule `clamp((mu_aff/mu)^3, 1e-6, 0.9)` and a fixed 0.99
fraction. The benchmark harness now records compact first/last/min/max history
for sigma, mu, affine and accepted steps, backtracking, and fallbacks. A local
synthetic Q3 smoke returned `Optimal` in five iterations and verified all new
history fields in both report and manifest. The full 151-assertion SOC suite
passed locally with four Julia threads.

One independent correctness issue was fixed immediately. If the last allowed
Q3 update satisfied every tolerance, the loop exited on `iter_max` before the
next top-of-loop residual assembly and mislabeled the certified point
`IterLimit`; the analogous case existed at the time boundary. The finalization
path now reuses the original-coordinate residuals it already computes and
promotes an `IterLimit`/`TimeLimit` point to `Optimal` only when primal, dual,
and gap tolerances all pass. A regression sets the iteration cap exactly to
the known convergence update, requires the terminal-boundary message, and
passes the independent certificate.

Two algorithm experiments remain evidence-gated rather than enabled. First,
Q3 initializes the primal head to `OmegaP*data_scale` even though fixed trace
makes the exact interior head `layout.head`; using that exact head removes an
artificial affine residual and saved one iteration on small audited examples.
Second, the standard Mehrotra sigma upper bound is one rather than the current
0.9. The safer order is to measure the J40 `Omega=1/2/5/10/20` histories, then
test exact-head initialization, and only then test the sigma cap. None becomes
default without the five-percent complete-solve and certificate gates.

### J80 exact Julia-pool control and invalid first omega sweep

The no-code exact-pool control `199603.node220` completed on the same node109
as the default-pool J80 run. It used 64 default Julia threads, no interactive
thread, one GC thread, `JULIA_EXCLUSIVE=1`, the 10 ms thread sleep threshold,
one BLAS thread, the same immutable model and candidate, and the same 900-second
solver limit. It returned `TimeLimit` after 257 iterations, so this row remains
diagnostic and is not an accepted solved benchmark. The certificate failed only
the duality-gap gate: the relative gap fell to `7.428e-8`, while normalized
primal and dual residuals were `1.178e-61` and `9.421e-18`.

| J80 Float64x4, 64 requested workers | Default pool | Exact pool |
| --- | ---: | ---: |
| Iterations in approximately 905 s | 170 | 257 |
| Relative gap at time limit | `1.997e-5` | `7.428e-8` |
| Mean active cores | 31.29 | 36.18 |
| Equality Gram, total | 486.260 s | 492.956 s |
| Equality Gram per iteration | 2.860 s | 1.918 s |
| Schur assembly per iteration | 3.344 s | 2.271 s |
| Process peak RSS | 7.993 GB | 7.717 GB |

The exact pool improved completed iterations by 51.2% and equality-Gram
throughput by 49.1% without a solver-code change. It reduced the gap by about
269x, but it did not solve the model to `1e-12`; the late-complementarity issue
is therefore still algorithmic. Mean utilization improved only from 31.29 to
36.18 active cores, so the remaining high-core scaling bottleneck is real. The
exact pool is the required launcher configuration for subsequent J80 kernel
experiments, but this single time-limited control is not used as an accuracy or
final speed benchmark.

The first J40 `OmegaP=OmegaD=1/2/5/10/20` sweep, job `199606.node220`, did not
enter the solver. Every process failed the same benchmark-driver geometry check
because it referenced the nonexistent `problem.dims.kmax` field. No numerical
row from that allocation is retained. The driver now checks
`maximum(problem.dims.k)` and was resubmitted from its verified SHA-256
`f5e162eeda8d4e99fe94d0b966f992abda3bd9a48410a6f99cd25a0fc0b36770`.

### Rejected J80 weighted-contiguous output-tile schedule

Job `199612.node220` isolated the equality-Gram kernel on the real J80
`65,600 x 350` Float64x4 panel with 64 exact-pool workers, one BLAS thread,
physical CPUs 0--63, memory nodes 0--3, and two repetitions in each of four
fresh Julia processes. The order was interleaved, weighted, weighted,
interleaved. All four lower triangles had the identical SHA-256
`7bd61f5afc6e708856b3f9fc07721aebda2273cf6eec1076a13a3f4d038c6b89`,
so the alternative ownership schedule was numerically exact.

| J80 Float64x4 equality Gram | Process median | Within-process CV |
| --- | ---: | ---: |
| Interleaved first | 6.175 s | 0.30% |
| Weighted first | 6.306 s | 0.04% |
| Weighted second | 6.883 s | 2.24% |
| Interleaved second | 6.385 s | 0.04% |

Across all four repetitions per strategy, the aggregate medians were 6.286 s
for the established interleaved scheduler and 6.541 s for weighted contiguous
ownership. Weighted ownership was 4.1% slower. Its indivisible tile ranges also
left a measured theoretical work spread of about 1.28x at 64 workers. The
experiment therefore failed the five-percent improvement gate in the wrong
direction. The weighted scheduler, selector, diagnostics, and unit-only test
were removed; the established interleaved output-tile scheduler remains the
default. Peak RSS was essentially unchanged at approximately 7.47 GB.

### Experimental fixed-extended column-owned GEMV

After rejecting weighted SYRK ownership, the next measured J80 phases outside
the Gram were rectangular products with `B` or `Btil`. The established
row-owned kernel gives each output to one task, which is race-free but reads a
column-major `m x n` panel with a stride of `m` inside every dot product. An
experimental fixed-extended-only kernel retains exclusive row ownership and
the same ascending-column accumulation order, but loops over columns outside
the owned row interval. It therefore streams each panel slice contiguously.
BigFloat and Float64 remain on their existing paths.

A local four-thread Float64x4 preflight on an `8,192 x 64` panel measured
4.829 ms for the row-major traversal and 3.312 ms for the column-streaming
traversal, a 1.46x kernel speedup. Every output was bitwise identical. The SOC
residual form measured 5.062 ms versus 3.687 ms (1.37x) with the same exact
result. The SOC
suite, including a direct long-panel GEMV and residual comparison, passed
197/197; the scalar Q3 kernels passed 480/480 and the KKT regressions passed
74/74. This is only a local preflight, not retention evidence. The change
remains experimental pending an isolated real-J80 panel A/B and a complete
solve showing that task launches and NUMA placement do not erase the gain.

The follow-up correctness audit found one internal edge case: the
column-owned residual helper clears its output before consuming the
right-hand side, so directly aliasing those two vectors would destroy the
right-hand side. Solver workspaces are disjoint, but the dispatcher now
explicitly falls back to the established row-owned implementation when the
buffers may alias, while a direct call to the specialized helper rejects the
unsupported alias. A regression verifies both the alias-safe dispatcher and
the fail-fast helper. The selector also requires at least 20,000 panel
contractions per selected worker. This retains J40 through 64 workers and J80
through 128 workers, while avoiding the J40 128-worker launch regime with only
about 11,000 contractions per task. The focused SOC suite passed 202/202 on
four local threads after these changes. After adding the executed
`forward_gemv_kernel` diagnostic, the same suite passed 205/205. Real J80
cluster timing is still the retention gate. An additional trajectory
regression confirms that `OmegaP=1` and `OmegaP=100` produce identical native
Q3 iterates when `OmegaD` is held fixed; the focused suite now passes 211/211.

### Native-SOC reference implementation audit

The Clarabel and ECOS native SOC implementations were audited from their
official source trees before considering a larger algorithm change. Both use
the Lorentz residual `x0^2 - ||x_tail||^2`, Nesterov--Todd primal-dual scaling,
and a scaling Hessian of the form
`eta^2 * (2*w*w' - diag(1,-I))`. For Q3 this is only six packed triangular
entries and the Hessian-vector product needs one dot product plus vector
updates. Clarabel deliberately reserves the rank-two sparse expansion for
larger SOCs; adding two extension variables to every three-dimensional CSDR
cone would increase KKT dimension and likely fill rather than help it.

The transferable implementation lessons are therefore: keep Q3 state and its
six scaling values contiguous, freeze the global KKT sparsity pattern and
numeric index maps once, update only numeric values per iteration, use
sign-aware regularization and improving-only iterative refinement, and make
any cone-parallel reductions deterministic. The current SDPX Q3 engine is an
HKM-equivalent fixed-trace implementation, so NT scaling is the main
algorithmic experiment after the exact-head and data-layout measurements. It
will not be introduced during the current kernel A/B because that would mix
iteration-count and per-iteration effects.

The standalone Q3 layer now contains the corresponding NT construction
without changing the solver path. It computes owned `(w, lambda)` state,
`eta`, the six-entry packed Q3 scaling Hessian, and an allocation-free
Hessian-vector product. The implementation rescales Lorentz residual checks to
avoid avoidable overflow, rejects non-interior or aliased inputs before
writing, and keeps BigFloat outputs independently owned. Explicit identities
`H_s*z = s` and `lambda = W*z`, random interior points, central rays spanning
large magnitude ratios, in-place Hessian application, and Float64 allocation
gates pass for Float64, Float64x4, and BigFloat256. The complete scalar kernel
suite passes 1,251/1,251 assertions. These kernels remain unconnected to the
Newton/KKT engine until a controlled HKM-versus-NT solve benchmark is possible.

### Experimental fused equality-panel transform

The established Q3 equality setup first copies the complete `B` panel into
`Btil` serially and then transforms two rows per cone in parallel. A standalone
fixed-extended kernel now permits a controlled A/B in which each worker reads
its immutable source rows and writes the transformed destination directly. It
preserves the two divisions and subtraction order exactly, removes the serial
copy, and can distribute first touch across NUMA nodes. Its conservative
selector accepts only the adjacent `(2l-1,2l)` CSDR row layout and sufficient
work per worker; it is not connected to solve dispatch yet.

On the local four-core J40-shaped `8,400 x 170` Float64x4 panel, ten alternating
passes produced bitwise-identical output. Median copy-plus-transform time was
47.933 ms versus 46.696 ms fused, only a 1.027x improvement (CV 0.87%/1.16%).
This fails the five-percent local retention gate, so no production selector was
changed. The standalone function remains only long enough to test whether the
much larger J80 panel gains materially from distributed NUMA first touch; it
will be removed if that cluster test also fails.

### Reused local-pivot reciprocals and fused panel candidate

The first fused-panel measurement was dominated by the two triangular
divisions per panel entry, so it understated the value of removing the serial
copy. Each Q3 local Cholesky pivot is reused for every equality column and in
both predictor/corrector triangular solves. A controlled four-worker local
microbenchmark therefore compared the established division expressions with
one reciprocal per pivot followed by multiplication.

| Arithmetic / panel | Direct divisions | Reciprocal multiply | Speedup |
| --- | ---: | ---: | ---: |
| Float64x4, `8,400 x 170` | 47.683 ms | 12.760 ms | 3.74x |
| BigFloat256, `2,048 x 64` | 3.825 ms | 1.982 ms | 1.93x |

The Float64x4 relative transform difference was `2.68e-65`; the BigFloat256
difference was `6.13e-78`. Both are below the respective rounding scale, and
the direct Float64 path remains unchanged. The workspace now stores two owned
reciprocals per cone. BigFloat computes them by mutating disjoint MPFR slots,
and every subsequent panel/forward/transpose solve mutates only its owned
destination. Small Float64, Float64x4, and BigFloat native-Q3 solves retain
their analytic objectives and independent certificates.

Once reciprocal multiplication removed most local arithmetic, the J40-shaped
Float64x4 copy-plus-transform median was 16.608 ms while a fused source-read,
destination-write transform was 12.760 ms, a further 1.30x phase speedup. The
conservative adjacent-row selector is therefore connected for fixed-width
extended arithmetic only; broader row layouts, Float64, and BigFloat retain
copy-then-transform. Executed diagnostics report both `local_pivot_kernel` and
`equality_panel_transform`. The complete SOC suite passes 257/257 assertions,
including reciprocal ownership, original division agreement, exact fused
output, analytic objectives, residuals, and PSD/SOC certificates. This remains
a candidate until the real J40/J80 same-node solve gate confirms at least a
five-percent complete-solve gain; production and the cluster current symlink
are unchanged.

### Native-Q3 dual cold-start sensitivity preflight

The exact fixed-trace primal head makes `OmegaP` intentionally inactive, but
the dual head still uses `OmegaD`. A deterministic product-of-disks smoke with
64 Q3 cells/eight dense equalities (Float64x4, tolerance `1e-18`) and a second
32-cell/six-equality case (BigFloat256, tolerance `1e-24`) swept
`OmegaD=0.001/0.01/0.1/1/10/100`. Every row was `Optimal` with a valid original
certificate.

| Arithmetic | `0.001` | `0.01` | `0.1` | `1` | `10` | `100` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Float64x4 iterations | 25 | **21** | 21 | 23 | 21 | 26 |
| BigFloat256 iterations | 35 | **28** | 30 | 33 | 28 | 29 |

This is not representative enough to change the automatic policy, but it
confirms that the existing `OmegaD=0.001` calibration can be a poor Q3 cold
start even after the primal-head correction. The real J40 sweep remains the
promotion gate; the intended next comparison is `0.001/0.01/0.1/1/10` with
identical model hash, exact thread pool, and full sigma/step history. No default
parameter was changed from this local preflight.

### Final ownership and fallback-diagnostic audit

A fresh mutable-arithmetic audit found that the Q3 compatibility boundary and
two scalar conversion kernels assigned the same BigFloat object to both
off-diagonal entries of a symmetric matrix. Values were numerically correct,
but mutating one triangle during later export or validation could silently
change the other. Materialized primal/dual matrices, Q3-to-symmetric
conversion, and full direction recovery now create two independently owned
MPFR values. The Schur-metric mutator also rejects output/coefficient aliasing,
and the BigFloat equality GEMV initializes previously unassigned destination
slots before worker launch. Dedicated object-identity and fresh-destination
regressions cover these cases.

The global `extended_precision_blas=:off` switch now remains authoritative
even when a conflicting Q3 output-tile or row-bin strategy is requested.
Zero-equality native solves report `gram=:none`, and Q3-to-PSD2 fallback results
report the solver that actually executed (`:sdp`) while preserving the planned
Q3 algorithm separately. After these fixes, the combined focused suites pass
1,251 scalar-Q3 assertions and 257 native-SOC assertions on four local threads.

### Complete local regression after the Q3 equality-kernel changes

The complete package test suite was run from a clean temporary test environment
with four Julia threads after the reciprocal-pivot, fused equality-panel,
BigFloat ownership, GEMV initialization, and executed-backend diagnostic fixes.
It passed **7,401/7,401 assertions** in **21 minutes 53.7 seconds**. This includes
the focused Q3 scalar, native SOC, KKT, MOI/Convex, LP, checkpoint, precision,
and public-interface regressions. The run used the package's declared test
dependencies rather than a reduced include-only smoke environment.

This result closes the local correctness gate. It does not close the performance
gate: the reciprocal/fused and column-owned kernels remain candidates until
same-node J40/J80 cluster runs demonstrate reproducible complete-solve gains
with unchanged original-coordinate objectives, residuals, gaps, and PSD/SOC
certificates. No production symlink or published release was changed.

The benchmark source-tree fingerprint after this gate and the J80 launcher
retention below is
`6da04947620b6c54d21adac9e59f272555d2aa8435afe58ddc1cd9e70e8d3bd1`.

### J80 exact-pool launcher retention

The earlier node109 control already established a reproducible launcher-level
improvement: the exact `N,0` Julia pool with one GC worker and exclusive CPU
affinity increased the number of completed J80 iterations in the same
approximately 900-second limit from 170 to 257, and reduced equality-Gram time
per iteration from 2.860 to 1.918 seconds. The J80 SDP, SOCP, and 128-core PBS
templates now select this policy by default while retaining an explicit
override. J40 and general launchers remain unchanged. This is a scheduling
default, not a new accepted solved J80 benchmark; both compared rows ended at
the time limit, and the exact-pool row still failed only the duality-gap gate.

The documentation build was also run after the launcher and Q3 documentation
updates. Documenter completed doctests, template expansion, cross-reference
checking, inventory generation, and HTML rendering successfully for version
0.4.0.

### Rejected cached direct cone inverse experiment

The Q3 predictor, corrector, and direction-recovery phases reuse the same three
entries of each primal cone inverse. An experiment cached those entries once
per assembly for Float64x4 and BigFloat while preserving Float64's established
direct-division path.

A J40-sized four-use scalar-kernel A/B produced bitwise-identical final values:

| Arithmetic | Direct divisions | Cached direct inverse | Kernel speedup |
| --- | ---: | ---: | ---: |
| Float64x4 | 3.826 ms | 1.739 ms | 2.20x |
| BigFloat256 | 28.244 ms | 17.505 ms | 1.61x |

The experimental branch passed 278/278 focused SOC assertions and 7,422/7,422
complete package assertions. BigFloat ownership checks found no MPFR aliasing.
The microkernel improvement is nevertheless confined to a small fraction of
the measured J40/J80 runtime; equality Gram and convergence dominate. Per the
renewed requirement to stop low-impact work, the complete-solve ABBA job was
terminated before a candidate timing row was accepted, and the cache, its
`3L` storage, diagnostics, and tests were removed from the working candidate.
The restored direct path passes 257/257 focused SOC assertions.

Candidate source
`<cluster-root>/SDPX.jl/candidates/q3-fixed-trace-417f5198081c/source`
was verified on the cluster with source fingerprint
`417f5198081c7c125b7029b0fe0429a01f1ec306591b9ae5626b325b3eefd270`
and made read-only. Compute-node focused regression job `199640.node220`
finished on node155 with exit status zero and a `PASSED` marker: 381/381 LP,
SOC, and public-API assertions passed in 3 minutes 57.2 seconds with peak RSS
1,500,812 KiB. It is retained only as a read-only experimental artifact, not a
release candidate. Job `199641.node220` completed one certified no-cache
baseline row (191 iterations, 109.269 seconds solve wall, 100.238 seconds in
instrumented phases, 67.753 seconds in equality Gram) and was then stopped.
The companion OmegaD job `199642.node220` was also stopped before numerical
output so the main investigation can concentrate on convergence and the
dominant Gram/KKT path.

### J40 native-Q3 dual-scale sweep

Compute-node job `199643.node220` ran an exact-pool Float64x4 sweep on the
canonical J40 model (`4,200` fixed-trace PSD2/Q3 blocks, `8,400` variables,
`170` equalities; model SHA-256
`eb9072b252e32d39f00bef78f81c5cd9269c6f65a1d02150b7cf02a694fd46cd`).
Every row used eight Julia/solver threads, one BLAS thread, tolerance `1e-12`,
the exact fixed primal head, and the same source fingerprint
`6da04947620b6c54d21adac9e59f272555d2aa8435afe58ddc1cd9e70e8d3bd1`.
All seven rows returned `Optimal` through the native Q3 KKT, passed the
original-coordinate certificate, and agreed on the physical objective.

| `OmegaD` | Iterations | Core phases (s) | Equality Gram (s) | Schur (s) | Relative gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 188 | 78.350 | 48.312 | 59.887 | `7.31e-14` |
| 0.1 | 186 | 77.540 | 47.843 | 59.145 | `7.54e-13` |
| 0.5 | 182 | 76.214 | 46.789 | 58.125 | `4.99e-13` |
| 1 | 175 | 73.389 | 45.039 | 55.853 | `8.71e-13` |
| 2 | 191 | 79.634 | 49.086 | 60.952 | `9.46e-13` |
| 5 | 172 | 72.480 | 44.250 | 55.148 | `9.94e-13` |
| 10 | **168** | **70.726** | **43.220** | **53.680** | `9.83e-14` |

The best measured point reduces iterations by 12.0% and instrumented core
time by 11.2% relative to the automatic `OmegaD=2` trajectory. The response is
not monotone, however, so this single-model grid is evidence for a Q3-specific
initializer, not a justification for hard-coding `10`. Fresh-process elapsed
times include roughly constant compilation overhead and are therefore not the
retention metric; a same-allocation warm A/B remains required.

### J40 feasible-start geometry diagnostic

Read-only compute-node job `199646.node220` evaluated initialization geometry
without changing the solver. The current `x=0` is already exactly equality
feasible because `b=0`. Moving every block variable to its disk center would
instead create equality residual `845.056` in infinity norm, so that candidate
was rejected without a full solve. The minimum-tail exact equality correction
also leaves the cone: its maximum tail/head ratio is `1.9483` and its minimum
PSD margin is `-0.9483`.

The dual side has a useful exact construction. A Float64 weighted least-squares
solve of the fixed `170 x 170` equality system, followed by target-arithmetic
local recovery, gives maximum dual-tail norm `0.45084` and stationarity
residual `1.46e-11` in the diagnostic arithmetic. Common determinant slacks
`0.1` and `1` give initial complementarity `0.1215` and `1.0034`, respectively.
The next controlled experiment therefore keeps `x=0`, tests a strictly
interior radial primal tail that reduces the affine cone residual, and tests
the weighted dual-feasible start. It does not retain the rejected disk-center
or infeasible exact-equality candidates.

### J40 initialization A/B and rejected feasible-start variants

The first runtime-override attempt (`199658.node220`) was discarded: Julia's
world-age rules left the original initializer active, which was detected
because the purported radial row reproduced the baseline bit for bit and
reported the unchanged unit cone residual.  The job was cancelled, the driver
was corrected by installing the experimental method before compiling its
entry point, and no result from that invalid attempt was used.

Corrected exact-pool job `199660.node220` then ran all four rows on node19 with
eight Julia/solver threads, one BLAS thread, source fingerprint
`6da04947620b6c54d21adac9e59f272555d2aa8435afe58ddc1cd9e70e8d3bd1`,
and the canonical J40 model hash.  Every row executed the native Q3 KKT,
returned `Optimal`, and passed the original-coordinate certificate.

| Initializer | Iterations | `solve!` (s) | Core phases (s) | Equality Gram (s) | Peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold, `OmegaD=10` | 168 | 80.718 | 70.795 | 43.352 | 1.84 |
| radial primal tail `0.9`, `OmegaD=10` | 157 | 76.850 | 66.844 | 40.484 | 2.15 |
| weighted dual, determinant slack `0.1` | 192 | 89.893 | 79.901 | 49.395 | 2.13 |
| radial plus weighted dual | 174 | 106.612 | 96.657 | 61.178 | 2.14 |

The radial start was genuinely active: its original cone mismatch fell from
`1.0` to `0.1`.  It reduced iterations by 6.5% and core phases by 5.6%, but
improved `solve!` wall time by only 4.8% while increasing peak RSS by about
17%.  It therefore fails the complete-solve and memory retention gates; no
tail-fraction sweep or solver change is justified.  The exactly stationary
weighted dual start was worse, increasing iterations by 14.3%, and its
combination with the radial start was also slower.  Both weighted variants are
rejected.  The cold `OmegaD=10` trajectory remains the only initialization
candidate clearing the requested ten-percent core-time threshold, pending its
BigFloat256 and J80 checks.

### Structural pivot: native Nesterov--Todd direction for fixed-trace Q3

The parameter and initializer sweep above is now closed. No tuned `OmegaD`,
sigma cap, radial start, or low-precision continuation is being promoted. The
next candidate instead uses the fixed-trace geometry itself: every eligible
real PSD2 block is represented as one Lorentz vector, and the native Q3
predictor/corrector can select a Nesterov--Todd (NT) scaled search direction.
The established matrix/HKM direction remains the default reference and the
per-solve fallback.

The NT implementation computes a single scaling point per cone and uses the
closed Q3 inverse metric

`K = (W'W)^(-1) = eta^(-2) * (2 J*w*w'*J - J)`.

It preserves the existing block-local variable elimination and the shared
equality Gram/KKT system. Predictor recovery uses `Y + K*P`; the corrector
uses the exact Lorentz Jordan shift
`(W^(-1)dX_aff) circ (W*dY_aff) - sigma*mu*e` and solves the Jordan linear
system `L_lambda*u = d_s`. It does not approximate that solve with an
element-wise cone inverse. The final PSD2 matrices are reconstructed only at
the compatibility/certificate boundary.

All NT storage is independently owned for BigFloat. Invalid scaling,
nonpositive local metrics, or direction recovery restart the same unmodified
iterate through a fresh HKM assembly; a later native-Q3 failure still retains
the established PSD2 fallback. Execution diagnostics record requested and
executed directions and the exact fallback reason. The benchmark gate rejects
any nominal NT row that actually used HKM or PSD2.

Local verification on four Julia threads passed:

- 1,724/1,724 scalar-Q3 assertions for Float64, Float64x4, and BigFloat256,
  including `W`, `W^(-1)`, `H_s^(-1)`, Jordan-solve, alias, ownership, and
  zero-allocation checks;
- 386/386 native-SOC assertions, including exact scaled Newton residuals for
  predictor and corrector directions in all three arithmetic families;
- small analytic boundary solves at Float64, Float64x4 (`1e-18`), and
  BigFloat256 (`1e-30`), all `Optimal`, certificate-valid, and without
  direction fallback.

The immutable cluster candidate has source/benchmark/test fingerprint
`d38b3f9114815aba932e4142059ef3427928e9308b5ace5d5fcfccc635b87943`
at
`<cluster-root>/SDPX.jl/candidates/q3-nt-d38b3f911481/source`.
Production remains unchanged. The cluster retention sequence is a focused
compute-node regression, followed by same-node J40 `HKM -> NT -> NT -> HKM`
comparisons at identical model hash, tolerance, thread pool, BLAS width, and
timing boundary. J80 is deferred until J40 establishes both numerical
reliability and a complete-solve benefit.

### J40 NT/HKM ABBA result: NT rejected for this CSDR trajectory

Cluster job `199664.node220` first passed the focused compute-node gate on
node155: 489/489 LP, SOC, and public-API assertions, including 386 native-SOC
assertions, completed with exit status zero and a `PASSED` marker. Production
remained unchanged. The first performance submission landed on known-faulty
node70 and was killed before the script or benchmark started; it produced no
measurement and is excluded.

Replacement job `199666.node220` ran on idle node101 (AMD EPYC 7742) in one
32-slot allocation. Each numerical process used the exact eight-thread Julia
pool, one GC worker, one BLAS thread, the same J40 model SHA-256, tolerance
`1e-12`, automatic Gram selection, and one warm-up. The order was
`HKM -> NT -> NT -> HKM`.

| Direction | Iterations | `solve!` median (s) | Core median (s) | Gram median (s) | E2E median (s) | Mean active cores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HKM | 191 | **82.361** | **73.635** | **49.006** | **83.199** | 7.162 |
| NT | 225 | 96.232 | 87.171 | 57.694 | 97.078 | 7.272 |

The two HKM solve times were 82.397 and 82.325 seconds; the two NT times were
96.124 and 96.339 seconds. Thus the order-reversed result is highly
reproducible, but NT is 16.84% slower in timed solve, 18.38% slower in core
phases, and 16.68% slower end to end. Per-iteration Gram cost is essentially
unchanged, so the regression is predominantly the 17.8% increase in Newton
iterations plus the extra scaled corrector work. NT used about 4.1% less peak
RSS, which is not enough to offset the runtime loss.

All four rows returned `Optimal`, used the native
`q3_block_diagonal_equality` KKT and `output_tiles` Gram, passed their
original-coordinate certificates, and reported no direction or PSD2
fallback. HKM and NT physical objectives agree to much better than the
`1e-10` relative comparison gate; primal/dual/equality residuals and PSD2
margins pass. Therefore NT is numerically valid but fails the performance
retention gate. It remains an explicit research/reference direction and will
not be selected automatically. No NT parameter tuning or J80 NT campaign is
justified.

The measured bottleneck after rejection remains equality Gram construction:
49.006 seconds, or 66.6% of HKM core phases. The next structural experiment
must reduce the number or width of fixed-trace equality-Gram evaluations; a
small local-cone microkernel cannot meet the complete-solve gate.

### Boundary support-function reduction: measured and rejected

A second structural experiment tested whether the fixed-trace disk could be
eliminated at the boundary.  At a rank-one primal block, the support-function
Jacobian is rank one, so an idealized equality panel has one row per cone
instead of the two rows required by the interior Q3 Newton system.  This is an
exact boundary identity, but it is not a uniformly nonsingular interior-point
coordinate system.

Read-only job `199668.node220` reconstructed the converged J40 HKM result in
support coordinates.  The support reconstruction agreed with the objective to
`5.86e-13` relative error and its equality residual was `1.45e-11`.  The
rank-one Gram kernel itself took `0.127` seconds, about half the ordinary
per-iteration Gram time.  However, its Float64 projection was numerically
singular at roughly the `1e16` condition scale, and the dual-tail norms ranged
from `8.39e-9` to `1.46e18`.  A trajectory scan in job `199669.node220` showed
that the support residual was still `9.10e-1`, `4.55e-1`, and `1.11e-3` after
10, 40, and 80 ordinary HKM steps respectively; the boundary system becomes
useful only very late, exactly when its conditioning is worst.

The guarded semismooth crossover in `199670.node220` therefore failed its
complete-solve gate.  Starting after 80 HKM iterations reduced the support
residual from `1.11e-3` to `1.15e-4`, but line-search steps collapsed to below
`1e-6` and failed after 16 support iterations.  Starting after 120 HKM
iterations still left `8.77e-6` after 20 support iterations.  Neither path
produced a final certificate.  No damping or crossover tuning is justified;
the support-function/rank-one path is rejected and is not present in the
repository.

### Revised fixed-trace CSDR plan: implementation audit

The requested plan was compared line by line with the current candidate.

- Compact fixed-trace Q3 state, exact Lorentz fraction-to-boundary, a
  Mehrotra/HKM predictor-corrector, PSD2 reconstruction, original-coordinate
  certification, and PSD2 fallback are complete.  NT scaling is also complete
  and tested, but the J40 ABBA result above proves that it must remain an
  explicit research direction rather than the production default.
- Strict equality reduction is already implemented before native Q3.  Exact
  zero/duplicate/proportional cleanup is followed by an arithmetic-aware rank
  proposal; every proposed dependency and its right-hand side is checked in
  the solve arithmetic, and the equality multiplier is mapped back before the
  original-coordinate certificate.  J40 retains all 170 columns, so there is
  no rank reduction to exploit on that instance.  J80 remains to be measured.
- Float64x4 triangular output-tile SYRK, static ownership, the blocked equality
  Cholesky, BigFloat output tiles, Q3 block mapping, compact active-support
  storage, and immutable J40/J80 benchmark provenance are complete.
- Exact equality-row template grouping has not been implemented because no
  J40/J80 measurement yet proves repeated proportional templates.  It will be
  attempted only if a target-arithmetic structural scan shows a materially
  smaller number of two-row templates than cone blocks.
- `PreparedSolver` currently reuses the ingested problem and previous result,
  but it does not yet cache a reduced Q3 workspace.  This is useful for future
  objective scans, not the first priority for the single large solve.
- Guarded Float64x4 construction/factorization of a BigFloat equality Gram is
  still experimental work.  It will not be enabled without target-precision
  residual/refinement and final certificate gates.

The experimental campaign is now capped at 32 solver threads.  Parameter,
initializer, 64/96/128-core, row-bin, support-function, and NT trajectory
sweeps are closed.  The next decision is driven by measured equality-rank
presolve cost, exact template counts, and the J80 32-thread profile.

### Current 32-thread baseline and J80 memory preflight

Job `199683.node220` established the current J40 Float64x4/HKM baseline on
node103 with 32 Julia/solver threads, one GC worker, one BLAS thread, one
warm-up, and three timed solves.  All rows used the native
`q3_block_diagonal_equality` KKT and output-tile Gram, returned `Optimal`, and
passed the original-coordinate certificate with identical objectives,
residuals, gap, and PSD2 margins.

| Metric | Timed values | Median | CV |
| --- | --- | ---: | ---: |
| `solve!` | 32.491, 32.475, 32.577 s | 32.491 s | 0.17% |
| core phases | 23.702, 23.668, 23.749 s | 23.702 s | 0.17% |
| equality Gram | 14.496, 14.473, 14.532 s | 14.496 s | 0.21% |
| end to end | 33.338, 33.215, 33.314 s | 33.314 s | 0.20% |

Every run used 191 iterations.  Equality Gram is 61.2% of core time, while the
170-by-170 equality factorization is only 0.625 seconds per complete solve.
The process averaged 22.68 active cores out of 32, used 2.91 GB peak RSS, and
reported no regularization or fallback.  Relative to the matched eight-thread
HKM median, 32 threads improve `solve!` by 2.54x and Gram by 3.38x; 32 remains
worthwhile for J40 latency even though total CPU utilization is about 71%.

J80 preflight job `199684.node220` passed on node102 with the immutable model
hash and reduced geometry 32,800 blocks, 65,600 variables, and 350 equalities.
The estimated compact Q3 workspace is 810,863,712 bytes and the whole
load/convert/preflight process peaked at 7.93 GB RSS, so a 64 GB allocation has
ample headroom.  The serialized model is 663 MB.  NUMA telemetry showed about
5.65 GB of 6.45 GB resident data on node 0 immediately after serial loading;
the complete J80 test must therefore compare the default placement with a
basic interleaved/parallel-first-touch control before attributing scaling loss
to the Gram arithmetic alone.

### J80 equality-structure audit and retained preprocessing fast path

Read-only structure job `199685.node220` measured the exact J80 equality
system before any new transformation.  The 65,600-by-350 matrix has
19,319,200 nonzeros (84.14% density).  Arithmetic-aware equality presolve
retained all 350 columns; there was no rank reduction.  Its 65,600 rows formed
65,600 distinct exact proportional templates, and the 32,800 ordered two-row
block templates were also all distinct.  A deterministic sampled full-rank
proposal took 11.43 seconds, versus 5.38 seconds for the existing complete
target-arithmetic rank presolve, and did not certify a shortcut.  Rank
sampling and equality-support grouping are therefore rejected for this
benchmark.

Stage-level profiling job `199686.node220` then isolated a separate setup
bottleneck.  Of 224.75 seconds in `preprocess`, exact equality cleanup consumed
224.42 seconds even though it removed no equality.  The cause was the
diagnostic-only near-proportional scan: columns with the same dense support
were compared pairwise in Float64x4, producing O(m*n^2) work.  Exact zero,
duplicate, and proportional cleanup was not the source of a useful model
reduction.

The retained implementation caches every equality support signature in one
matrix pass and reuses those collision-checked buckets.  Exact cleanup still
validates complete target-arithmetic values before removing anything.
Near-proportional detection remains diagnostic-only and now has a fail-closed
budget of eight equivalent equality-matrix passes; exceeding the budget emits
an explicit warning and leaves the exact cleanup and model unchanged.
Work estimates use saturating arithmetic, dense pivot scans are charged, and
nonfinite candidate arithmetic is rejected from the diagnostic.

The first immutable A/B candidate, source fingerprint
`89694b25700dc91727c6552c795f16f2407dd9dae691176f3938c3a68993108e`,
ran as job `199687.node220` on the same node103 and same J80 model hash:

| J80 Float64x4 preprocessing | Baseline | Bounded diagnostic | Speedup |
| --- | ---: | ---: | ---: |
| Complete `preprocess` | 224.754 s | 0.395 s | 568.7x |
| Exact-constraint-cleanup stage | 224.424 s | 0.0735 s | 3053x |
| Whole profiler process | 315.81 s | 80.39 s | 3.93x |
| Peak RSS | 7,245,096 KiB | 7,196,584 KiB | 1.01x |

Both runs retained exactly 350 equalities and passed their immutable source,
model, and process-exit gates.  The candidate writes one warning explaining
that only the bounded near-proportional diagnostic was skipped.  This is a
large, reproducible end-to-end setup improvement with no numerical model
change and is retained for the final regression and solve campaign.

The complete-solve gate was then run as job `199690.node220` on node103 with
the exact 32-thread Julia pool, one GC worker, one BLAS thread, one warm-up,
and three cold J40 Float64x4/HKM-Q3 measurements.  All three rows returned
`Optimal`, executed the native Q3 block-diagonal-equality KKT and output-tile
Gram, completed in 191 iterations, and passed the original-coordinate
certificate.  Printed primal/dual objectives, gap, residuals, equality error,
and reconstructed PSD2 margins are identical to the pre-change baseline.

| J40 Float64x4, 32 workers | Baseline | Bounded diagnostic | Change |
| --- | ---: | ---: | ---: |
| `solve!` median | 32.491 s | 25.302 s | **22.1% faster** |
| End-to-end median | 33.314 s | 26.047 s | **21.8% faster** |
| Instrumented core median | 23.702 s | 23.580 s | 0.5% faster |
| Equality Gram median | 14.496 s | 14.442 s | 0.4% faster |
| Mean active cores | 22.68 | 28.72 | 26.6% higher |
| Peak process RSS | 2.906 GB | 2.890 GB | 0.6% lower |

The three candidate `solve!` times were 25.3024, 25.1835, and 25.3021 seconds
(CV 0.27%); end-to-end CV was 0.46%.  Candidate preprocessing itself was
0.255 seconds.  The essentially unchanged core/Gram times prove that the gain
is removal of wasted setup work rather than a trajectory or timing-boundary
change.  This clears the five-percent complete-solve, CV, memory, objective,
and certificate retention gates.

The retained source then passed two independent focused gates. PBS job
`199691.node220` ran on node104 with four Julia threads, one BLAS thread, and
16 reserved slots; it finished in 3:59 wall time with scheduler exit status
zero and about 1.48 GiB peak scheduler RSS. The job executes both
`preprocessing_regressions.jl` and `soc_regressions.jl` and writes `PASSED`
only after both return successfully. A separate local Julia 1.12.6 process
with an isolated writable depot passed 90 preprocessing, 1,724 scalar-Q3, 386
native-SOC, and 74 KKT assertions (2,274 total). The only messages were
expected sandbox `sysctl` warnings; the process exit status was zero.

### Same-node J40 PSD2 versus native HKM-Q3 at 32 threads

PBS job `199692.node220` ran the immutable bounded-preprocessing candidate on
node103 in `sdp-socp` order. Each formulation used the same J40 model/hash,
`Float64x4`, `scaling=:none`, tolerance `1e-12`, an exact 32-thread Julia pool,
one GC worker, one BLAS thread, one warm-up, and three timed solves. The job
finished with PBS exit zero and a `PASSED` marker. Every row was `Optimal`,
certificate-valid, and benchmark-valid; PSD2 executed `block_arrow`, while Q3
executed `q3_block_diagonal_equality` with HKM and output-tile Gram.

| J40 Float64x4 median | Optimized PSD2 | Native HKM-Q3 | PSD2 / Q3 |
| --- | ---: | ---: | ---: |
| iterations | 166 | 191 | 0.87x |
| `solve!` | 135.465 s | 25.454 s | **5.32x** |
| end to end | 136.213 s | 26.204 s | **5.20x** |
| instrumented core | 134.377 s | 23.710 s | **5.67x** |
| equality Gram | 12.847 s | 14.481 s | 0.89x |
| mean active cores | 7.83 | 28.72 | 3.67x |
| utilization of 32-thread request | 24.5% | 89.8% | 3.67x |
| `/usr/bin/time` peak RSS | 2,425,472 KiB | 2,572,088 KiB | 0.94x |

The PSD2 and Q3 solve CVs were 0.11% and 0.94%; end-to-end CVs were 0.16%
and 0.70%. Their primal objectives differ by about `4.7e-12` absolute
(`2.2e-13` relative), comfortably inside the `1e-9` gate. Both relative gaps
are below `1e-12`; equality errors are around `5e-67`; both reconstructed
primal/dual PSD2 margins are positive. Q3's compact workspace estimate is
56.3 MB versus the conservative 115.2 GB SDP estimate and its per-solve
allocation is lower (519 MB versus 721 MB). Whole-process RSS was 6.0% higher
for the second subprocess, so reverse-order job `199694.node220` was submitted
on the same node before interpreting that small process-level difference.

The equality Gram itself is not faster in Q3: Q3 performs more iterations and
spends 14.48 rather than 12.85 seconds there. The 5.3x complete-solve gain is
therefore the intended fixed-trace structural benefit: compact Lorentz local
algebra removes PSD2 matrix factor/contraction work and exposes enough regular
work to keep the requested pool active.

Reverse-order job `199694.node220` also finished with PBS exit zero and a
`PASSED` marker. Across all six timed rows per formulation, PSD2 and Q3 solve
medians were 135.102 and 25.706 seconds, respectively (5.26x); end-to-end
medians gave 5.14x. Combined CVs were 0.29% and 0.88%. The first/second-process
RSS pairs were 2.484/2.533 GB for PSD2 and 2.634/2.609 GB for Q3, so order does
not explain the roughly 3--6% process-level difference. Q3 nevertheless owns a
far smaller solver workspace and allocates less per solve; the absolute process
RSS remains below 2.7 GB and inside the non-material memory gate.

The phase waterfall explains the speedup. PSD2 spent median 54.46 seconds in
corrector work, 30.32 seconds in refinement, 29.33 seconds in residual/block
factorization, and about 20.96 seconds in each predictor/corrector linear solve.
Q3 spent 17.30 seconds in total Schur assembly, of which 14.48 seconds was the
equality Gram; its remaining major phases were only 3.50 seconds of KKT solves
and 2.65/2.06 seconds of predictor/corrector work. Equality Gram is therefore
the next Q3 bottleneck, while further 170-by-170 factor tuning is not valuable.

### J40 BigFloat256 formulation gate at 32 threads

PBS job `199695.node220` ran the same immutable J40 model on node103 in
Q3-then-PSD2 order. Both isolated processes used 256-bit BigFloat arithmetic,
the exact 32-thread Julia pool, one GC worker, one BLAS thread,
`scaling=:none`, tolerance `1e-12`, one warm-up, and three timed solves. Both
`/usr/bin/time` processes exited zero. Every timed row was `Optimal`,
certificate-valid, and benchmark-valid; Q3 executed HKM with the compact
block-diagonal-equality KKT and output-tile Gram, with no direction or PSD2
fallback.

| J40 BigFloat256 median | Optimized PSD2 | Native HKM-Q3 | Q3 / PSD2 |
| --- | ---: | ---: | ---: |
| iterations | 166 | 191 | 1.15x |
| `solve!` | 184.993 s | 197.066 s | **1.065x slower** |
| end to end | 189.537 s | 201.488 s | **1.063x slower** |
| instrumented phases | 179.637 s | 186.998 s | 1.041x slower |
| equality Gram | 91.867 s | 100.562 s | 1.095x slower |
| mean active cores | 17.51 | 18.03 | 1.03x |
| process peak RSS | 2.703 GB | 2.890 GB | 1.069x |
| allocated bytes / solve | 29.09 GB | 77.48 GB | 2.66x |

Solve CV was 1.35% for PSD2 and 1.09% for Q3; end-to-end CV was also below
1.4% for both. The primal objectives differ by about `9.1e-12` absolute
(`4.3e-13` relative), both relative gaps are below `1e-12`, equality backward
errors are about `1e-79`, and all reconstructed primal/dual PSD2 margins are
positive. Thus the numerical comparison is valid, but Q3 does not clear the
five-percent BigFloat performance or memory gates. The existing policy is
confirmed: BigFloat remains on optimized PSD2 under `algorithm=:auto`; native
Q3 remains an explicit reference/research option. The fixed-trace Q3 benefit
is strongly arithmetic-dependent: it is 5.26x faster for J40 Float64x4, but
its extra iterations and higher BigFloat allocation outweigh the compact cone
algebra at 256 bits.

### Final harness and example hygiene before the release candidate

An independent read-only release audit passed package loading, a generic SOC
smoke, a native fixed-trace Q3 smoke, 328 focused native-SOC assertions, 1,724
scalar-Q3 assertions, Julia parsing, shell syntax, and `git diff --check`. It
identified three non-numerical inconsistencies, all corrected before freezing
the final candidate: the direct J40/J80 driver now enforces the campaign's
32-thread cap; successful single-mode launchers write an explicit `PASSED`
file; and portable launchers no longer embed the maintainer's private cluster
paths, instead accepting Julia/environment/depot/release locations through
environment variables. The general SDPX solver remains unrestricted by this
benchmark-only thread policy.

`examples/08_soc_fixed_trace.jl` was added to the examples-as-tests set. It
solves a direct Lorentz norm problem, then an explicit constant-trace unit-disk
PSD2 model and asserts that the latter executed
`:q3_block_diagonal_equality`, returned `Optimal`, and passed the independent
certificate. The standalone run exited zero with objectives
`5.0000000170575625` and `-0.9999999996296093`.

### Recycled equality-PCG prototype: deterministic J40 rejection

The proposed matrix-free equality experiment was implemented behind the
explicit `q3_equality_solver=:recycled_pcg` option, with direct Gram remaining
the default and final fallback.  Its operator applies the current
`Btil' * (Btil * v)` in the target arithmetic type, while the most recent
exact target-arithmetic Gram Cholesky is used only as a preconditioner.  The
predictor and corrector form one transaction: failure of either target KKT
residual gate rebuilds the current exact Gram and recomputes both directions
before any outer iterate is updated.  Three consecutive fallback iterations
lock the remainder of the solve to the direct path.  Focused Float64,
Float64x4, and BigFloat256 tests, including BigFloat storage-ownership checks,
passed (420 assertions after the final diagnostic fixes).

PBS job `199698.node220` ran on node103 with the immutable prototype candidate,
the pinned J40 model/hash, Float64x4, tolerance `1e-12`, 32 Julia/solver
threads, one GC thread, one BLAS thread, and one warm-up per isolated process.
The planned A/B/B/A order was stopped after both recycled arms completed,
because they produced the same decisive failure mode; the redundant final
direct process was cancelled and 32 slots released.  The first direct arm was
`Optimal` and certificate-valid in 191 iterations, with timed solve/core/Gram
times of 25.696/23.582/14.396 seconds, mean 28.269 active cores, and 2,863,304
KiB process peak RSS.

Both recycled replicates were also `Optimal` and produced the same objective,
gap, equality residual, and positive reconstructed PSD2 margins as direct, but
the equality experiment did not recycle successfully:

| J40 Float64x4 | PCG replicate 1 | PCG replicate 2 |
| --- | ---: | ---: |
| PCG attempts / accepted inner solves | 5 / 2 | 5 / 2 |
| fallback iterations | 3 | 3 |
| accumulated PCG iterations | 17 | 17 |
| final pre-lock residual | 0.1257786763 | 0.1257786763 |
| direct lock | yes | yes |
| exact Gram builds / skipped builds | 192 / 0 | 192 / 0 |
| timed solve | 26.070 s | 25.787 s |
| instrumented core | 24.245 s | 23.650 s |
| equality Gram | 14.490 s | 14.337 s |
| mean active cores | 28.276 | 28.312 |

Thus the previous factor is not a sufficiently close preconditioner for even
the early J40 HKM metric changes under the target-precision gate.  The bounded
five-iteration PCG budget saved no Gram at all, and increasing its iteration
budget or weakening the residual is precisely the unproductive parameter
tuning excluded by the experiment plan.  The direction is rejected before
J80 and will not be enabled automatically or retained in the release
candidate.

The audit also found two measurement issues while the prototype was live.
PCG successes were initially counted after the Krylov residual but before the
complete transformed-KKT residual; success accounting now occurs only after
the latter passes.  The Q3 workspace preflight now includes the four
equality-sized Krylov vectors plus the variable-sized matrix-free panel.  The
benchmark execution gate now reads the actual equality solver rather than its
requested label, so a run locked back to direct cannot be marked as a valid
recycled-PCG benchmark.  These fixes passed the 420 focused SOC regressions.

### Float64x4 off-diagonal 4-by-2 equality-Gram microkernel

After rejecting recycled PCG, profiling returned to the dominant operation
that is still exact and unavoidable: the triangular Float64x4
`Btil' * Btil` contraction. The retained experiment pairs two columns in each
off-diagonal output tile and updates four output rows at once. At every
reduction index it loads one four-lane row vector and two broadcast column
values, then advances two independent four-lane accumulators. Reduction
indices remain strictly ascending for every output, diagonal tiles and odd
tails retain the previous implementation, and output ownership is unchanged.
Consequently the optimized lower triangle is bit-for-bit identical to the
scalar-order reference rather than merely tolerance-equivalent.

Focused regression added odd dimensions and tile tails at `(11,4)` and
`(13,5)`, checks every lower-triangle value exactly, and verifies that the
upper triangle remains untouched. The complete extended-precision BLAS
focused file passed 108/108 assertions.

The first cleaned deployment candidate contained 284 tracked/unignored files;
local and remote ordered file digests both equaled
`48f13cf9175fd273af4c0921001a1d039a40f1bb1e7ffa3f9714f7d4a7453edf`.
It was frozen at `candidates/v0.4.0-syrk4x2-5d43218617bc/source`, but the
packager had deliberately omitted ignored files and therefore also omitted
the root and environment `Manifest.toml` lock files. J40 scaling jobs
`199711.node220` and `199712.node220` each completed their old arm, then the
new-source subprocess failed during package loading because MOI could not be
resolved without the root manifest. This is a deployment-only failure before
model loading, not numerical or performance evidence. The incomplete
candidate is retained read-only for audit but is rejected for further solver
jobs. A replacement must include all four lock files and pass file-for-file
verification before the affected jobs are rerun. Production throughout still
points to release `46b8b9733d89f8a5eea4231e6e88249692b64057`.

The corrected replacement is
`candidates/v0.4.0-syrk4x2-8eeaefa925ef/source`. It contains the 284
tracked/unignored files plus the root, benchmark, Convex-frontend, and docs
manifests (288 files total). Local and remote ordered file digests are both
`8eeaefa925ef211f67dfad0290e1302d16518f4d8be82c28cedd049d4ccd6e5a`;
all four lock files are nonempty, a lightweight Julia 1.12.6 package load
returned zero, and the source was then made read-only. Full and scaling jobs
use only this manifest-complete replacement.

For complete-solve timing, the earlier `v0.4.0-bench-r3` baseline was found to
differ from the replacement in both the extension and a native-Q3 failure
branch. That branch is not exercised by successful J40 HKM solves, and it
cannot affect the direct Gram microbench, but it would make a full-solve A/B
less clean than necessary. A strict control was therefore created by copying
the 288-file replacement and substituting only the previous
`ext/SDPXMultiFloatsExt.jl`. Its source tree is otherwise byte-identical, it
is read-only, and its ordered digest is
`f3144c35213ab5574d15d39e38eb7ea4f9d4f59f5999c427a51293eb1c4ed08e`.
Complete-solve acceptance uses this strict control rather than the older
multi-difference candidate.

PBS job `199708.node220` provided the first valid same-node J40 kernel A/B on
node103. It used the exact J40 model/hash, a `8400-by-170` transformed panel,
32 Julia workers, one BLAS thread, one warm-up, and five calls in each
old/new/new/old arm. Attempt `199706` was cancelled before measurement when
the scheduler co-located it with the long J80 run, and attempt `199707` failed
before loading the model because the private harness used invalid Julia 1.12
tuple interpolation; neither is numerical evidence.

| J40 32-thread Gram kernel | old arm 1 | old arm 2 | new arm 1 | new arm 2 |
| --- | ---: | ---: | ---: | ---: |
| median seconds | 0.073526 | 0.073698 | 0.067462 | 0.067362 |
| within-arm CV | 1.00% | 1.17% | 0.97% | 0.61% |
| effective cores | 30.664 | 30.610 | 30.684 | 30.659 |

The median-of-arms improvement is about **8.4%**. All four arms produced the
same lower-triangle SHA-256
`2334bb9722d9c5b0c41c86e08d3e73030b601b59a40ba1c2fcaf1a6e712b4c0c`;
there was no material RSS increase. This clears the isolated-kernel gate but
does not by itself authorize retention. A same-node full J40 native-Q3 A/B
and the larger J80 panel A/B remain required to show a measurable complete
solve benefit and exclude cache/order artifacts.

Corrected manifest-complete scaling jobs `199713.node220` (node4, eight
allocated/eight Julia workers) and `199714.node220` (node7, sixteen
allocated/sixteen Julia workers) both completed with scheduler exit zero and
`PASSED`. They repeated the old/new/new/old J40 kernel comparison with nine
timed calls per arm. The lower-triangle hash remained exactly
`2334bb9722d9c5b0c41c86e08d3e73030b601b59a40ba1c2fcaf1a6e712b4c0c`
for every arm.

| J40 kernel scaling | old median of arms | new median of arms | reduction |
| --- | ---: | ---: | ---: |
| 8 workers | 0.264571 s | 0.242873 s | **8.20%** |
| 16 workers | 0.192680 s | 0.178737 s | **7.24%** |
| 32 workers | 0.073612 s | 0.067412 s | **8.42%** |

At eight workers both kernels used about 7.95 active cores. At sixteen the
kernel used only about 11.5--11.8 active cores, whereas the 32-worker isolated
run used about 30.6; this nonmonotone placement/scheduling behavior is why the
release selector must use measured complete-solve curves rather than assuming
that the nominal thread count predicts utilization. Process RSS varied by
ordinary compilation/order noise and did not increase systematically for the
new kernel. Thus the improvement reproduces at every requested J40 thread
count while preserving exact arithmetic output.

PBS job `199709.node220` completed the corresponding J80 old/new/new/old
kernel comparison on node103 with scheduler exit zero and a `PASSED` marker.
All arms used the exact `65600-by-350` panel, 32 Julia workers, one BLAS
thread, one warm-up, and five timed calls. Every old and new call produced the
same lower-triangle SHA-256
`681a2c248f0a69cd6f5a5fddbc79acae33b11b6ec3ca37e84ecb3e67067dca27`.

| J80 32-thread Gram kernel | old arm 1 | old arm 2 | new arm 1 | new arm 2 |
| --- | ---: | ---: | ---: | ---: |
| median seconds | 2.760024 | 2.521486 | 1.980615 | 1.994238 |
| within-arm CV | 3.27% | 5.40% | 0.28% | 1.12% |
| effective cores | 27.979 | 29.543 | 30.475 | 30.397 |
| peak RSS (KiB) | 7,136,880 | 7,225,488 | 7,143,608 | 7,133,296 |

The old kernel exhibits a measurable order/load effect, so the decision does
not rely on its more favorable first arm. Comparing the second new arm to the
faster second old arm still gives a **20.9%** reduction; median-of-arm medians
gives 2.640755 versus 1.987427 seconds, or **24.7%**. New-arm CV is below
1.2%, effective utilization rises to about 30.4 of 32 cores, and RSS is not
materially higher. The J80 panel therefore independently clears the kernel
retention gate. A complete J40 solve A/B is still required because only that
test includes the nonlinear trajectory, objectives, residuals, gap, SOC/PSD
certificates, and end-to-end timing.

### Target-residual-gated Float64x2 equality shadow preflight

After recycled PCG failed, the next structural experiment used a fresh
low-cost factor every outer iteration rather than reusing an obsolete target
factor. PBS job `199719.node220` ran on node4 with exactly 16 allocated/Julia
workers and one BLAS thread. It used the current J40 Float64x4 Q3 state and
the actual first predictor and corrector equality right-hand sides. The
`8400-by-170` target panel was converted in 1,024-row chunks, accumulated into
a Float64x2 lower-triangle Gram, and factored in Float64x2. No shadow panel was
retained. Every acceptance residual was recomputed as
`Btil'*(Btil*x)` in Float64x4.

The native Float64x4 Gram/factor took 0.163480/0.021805 seconds. Three shadow
builds had median 0.028500 seconds (CV 7.34%, about 15.8 effective cores), and
the shadow factor took 0.005292 seconds. The initial shadow solutions already
had target-operator relative residuals `8.71e-29` and `1.19e-27` for the
predictor/corrector. One target refinement reduced them to `6.65e-49` and
`7.58e-48`; two reached `1.02e-62` and `1.14e-61`. Warm target-operator
applications cost about 0.0126 seconds per RHS. The final solutions differ
from the direct-Gram solutions by only `9.4e-54` and `2.0e-53` relative.
Peak process RSS was 1,335,592 KiB.

Thus a residual gate with no correction would reduce equality build/factor
plus two verification products from roughly 0.185 to 0.059 seconds; even two
refinement corrections to near-Float64x4 arithmetic limits project about
0.109 seconds, a 41% reduction. This is preflight evidence only: it does not
yet prove nonlinear trajectory or certificate reliability. It clears the J40
gate for the same actual-RHS experiment on J80 (`199720.node220`); solver
integration remains disabled until that result and transactional fallback
tests pass.

The J80 follow-up `199720.node220` completed on node103 with scheduler exit
zero and `PASSED`, using exactly 32 allocated/Julia workers and one BLAS
thread. On the `65600-by-350` panel, the current Float64x4 Gram/factor took
2.727507/0.023223 seconds. Three 1,024-row streaming Float64x2 shadow builds
had median 0.427191 seconds and CV 0.97%, with 31.8--31.9 effective cores;
the shadow factor took 0.045811 seconds. Peak RSS was 7,231,212 KiB, which is
essentially the same as the direct J80 kernel processes because only one small
chunk is retained.

The actual predictor/corrector shadow solutions began at target-operator
residuals `2.17e-27` and `1.04e-26`; one refinement reached `5.36e-44` and
`4.08e-44`, and two reached `1.90e-61` and `7.41e-61`. Warm target-operator
products cost about 0.099 seconds per RHS. With two conservative correction
passes for both Newton right-hand sides, projected per-iteration equality
cost is about `0.427 + 0.046 + 6*0.099 = 1.067` seconds versus 2.751 seconds
for the target Gram/factor, a **61%** reduction. At the requested `1e-12`
outer tolerance, even the unrefined shadow solutions already exceed the
needed residual by fourteen digits; nevertheless implementation acceptance
will retain target-arithmetic residual checks and at least one guarded
correction opportunity.

Both J40 and J80 therefore support integrating an expert-only transactional
`Float64x2 -> Float64x4` equality path. Integration must build a fresh shadow
factor for the current `Btil`, certify predictor and corrector before any IPM
update, and rebuild/recompute both with the exact target Gram on nonfinite
conversion, nonpositive factorization, insufficient residual contraction, or
either RHS gate failure. No automatic/default selection is authorized by
these preflights alone.

### Strict full-solve retention gate for the Float64x4 4-by-2 SYRK kernel

PBS job `199718.node220` completed on node9 with scheduler exit zero and an
explicit `PASSED` marker.  It used the manifest-complete strict control and
candidate trees, the exact J40 model/hash, 32 Julia/solver workers, one BLAS
thread, one warm-up per process, and an old/new/new/old ordering.  All four
rows executed native fixed-trace HKM-Q3 with the output-tile Gram and no
fallback.  Each returned `Optimal` after 191 iterations with valid execution,
benchmark, and original-coordinate certificates.

The primal and dual objectives, relative gap, primal/dual/equality residuals,
and reconstructed PSD2 margins were bit-for-bit identical across all four
processes.  In particular, the primal objective was
`-21.0253439247503263337953061827573472424515723866581163695625404356`,
the relative gap was `9.45653417936792e-13`, the equality backward error was
`5.10810430998658e-67`, and both reconstructed primal and dual PSD2 margins
were positive.

| J40 full solve, 32 workers | strict control median | 4-by-2 candidate median | reduction |
| --- | ---: | ---: | ---: |
| equality Gram | 14.463329 s | 13.321284 s | **7.90%** |
| Schur assembly | 17.252378 s | 16.284591 s | **5.61%** |
| native core phase total | 23.727391 s | 22.824172 s | **3.81%** |
| timed solve wall | 25.652054 s | 24.550100 s | **4.30%** |
| solve plus validation | 26.513257 s | 25.410440 s | **4.16%** |

Within-arm CV was below 1% for all of these primary timings.  Mean active
cores were 28.44 for the control and 28.26 for the candidate (no material
utilization loss).  Candidate peak RSS across its two arms was 2.965 GB,
slightly below the control maximum of 2.974 GB.  The complete-solve gain is
smaller than the isolated Gram gain, as predicted by Amdahl's law, but it is
stable, order-controlled, numerically exact, and measurable.  Together with
the 7--8% J40 kernel gains at 8/16/32 workers and the conservative 20.9% J80
kernel gain, this clears retention for the 4-by-2 Float64x4 off-diagonal
microkernel.  It does not change the selected algorithm or authorize any
mixed-precision/default-policy change.

Jobs `199710`, `199711`, `199712`, `199716`, and `199717` were deployment-only
failures before a model solve (respectively incomplete frozen manifests,
missing exported Julia path, or an unsuitable candidate-local project
environment).  Job `199715` was explicitly superseded before measurement.
None supplied numerical evidence and none is included in the A/B result.

### BigFloat256-to-Float64x4 shadow-Gram preflight

PBS job `199721.node220` completed on node101 with scheduler exit zero and a
`PASSED` marker.  It used J40, 32 Julia workers, one BLAS thread, BigFloat at
exactly 256 bits, and the same actual first predictor/corrector equality
right-hand sides as the direct target factor.  A 1,024-row scratch panel was
converted to Float64x4 and accumulated into a Float64x4 shadow Gram; every
acceptance residual was recomputed by the BigFloat256 matrix-free target
operator.

Numerical quality was excellent: the unrefined predictor/corrector target
residuals were `1.37e-61` and `2.91e-60`; one correction reduced them to
`2.14e-75` and `6.67e-74`.  Relative errors against the direct BigFloat
solutions were about `1.0e-65` and `1.8e-65`.  This easily clears both the
`1e-12` and `1e-20` residual gates.

Performance, however, failed.  The native BigFloat256 Gram/factor took
0.547722/0.101544 seconds, while the streaming Float64x4 shadow build/factor
took 0.992702/0.054286 seconds.  Shadow-build CV was 20.3% and effective CPU
use only 4.4--5.8 cores out of 32 because serial conversion and nine small
SYRK launches dominated.  Peak RSS was 1,389,632 KiB.  Therefore the
1,024-row BigFloat shadow design is rejected despite its accuracy.  One
bounded larger-chunk experiment is justified to separate launch/conversion
overhead from the arithmetic idea; absent a clear total build/factor gain,
BigFloat256 remains on its native target-precision Gram.

Job `199722.node220` tested a single 8,400-row scratch panel but retained the
serial conversion.  It also passed numerically, yet shadow build/factor was
0.957423/0.054599 seconds versus 0.549417/0.101942 seconds for the target
path.  Effective use fell to 3.2--3.7 cores and CV remained 18.8%.  This
isolated the dominant cost as BigFloat-to-Float64x4 conversion, not chunked
SYRK accumulation.

Job `199723.node220` therefore made exactly one structural change: the same
single scratch panel was converted by 32 statically column-owned tasks before
the same output-tile SYRK.  The job completed on node103 with exit zero and
`PASSED`; all residual and direct-solution comparisons were unchanged.  The
shadow build median fell to 0.207798 seconds and its best arm to 0.144733
seconds, with effective use up to 25.15 cores.  Shadow factorization was
0.054769 seconds, while target BigFloat Gram/factor was 0.534798/0.091054
seconds.  Two target-operator checks cost about 0.037 seconds total, so the
median unrefined shadow transaction is roughly 0.300 seconds versus 0.626
seconds direct, a projected **52% equality-path reduction**.  Peak RSS was
1,445,548 KiB, not materially above the other preflights.

The parallel build still had 58.8% three-sample CV (0.145/0.208/0.440 s), so
this is not retention evidence.  It does reverse the previous BigFloat
NO-GO: parallel conversion is a necessary part of any later BigFloat256
mixed-refine prototype.  Float64x4 integration and complete J40/J80 gates
remain first; only after they pass should BigFloat use a bounded parallel
chunk size and full-solve A/B.  Native BigFloat Gram remains the default and
unconditional fallback.

### Expert Float64x2-to-Float64x4 equality refinement candidate

The measured J40/J80 preflights were integrated as an explicit
`q3_equality_solver=:mixed_refine` research path; `:direct` remains the
default.  Each native-Q3 iteration first transforms the current equality
panel in Float64x4, then streams at most 1,024 rows at a time into an owned
Float64x2 scratch panel.  Output-tile triangular SYRK constructs a fresh
shadow Gram and Float64x2 Cholesky factor.  Predictor and corrector solutions
are accepted only after target-Float64x4 applications of
`Btil'*(Btil*x)` pass the existing refinement tolerance.  Up to three
corrections are allowed, and each must reduce the target residual by at least
a factor of two.

The transaction is fail closed.  A nonfinite conversion, shadow-factor
failure, weak residual contraction, or either RHS failure permanently locks
that solve to the direct Float64x4 Gram.  No primal/dual IPM state is updated
before both directions succeed; the unchanged iterate rebuilds the current
target factor and recomputes predictor and corrector.  A stale pending marker
is explicitly cleared as soon as the direct rebuild begins, so a later
unrelated direct KKT failure cannot be misclassified as a mixed fallback.
Unsupported arithmetic, including BigFloat256 at this stage, executes the
ordinary direct path.  The benchmark rejects unsupported mixed requests
rather than spending a full solve merely to report that fallback.

The compact-Q3 memory estimator now includes both Float64x2 Gram/factor
matrices, the bounded conversion panel, and the Float64x4 target residual
buffers.  Benchmark reporting treats type/module/function objects as scalar
provenance strings; this fixes a post-solve stack overflow when flattening the
recorded `Float64x2` shadow type.  SDP rows reject Q3-only controls, while the
paired SDP/SOCP launcher forwards those controls only to its SOCP leg.

Local verification after these audit fixes passed 424 native-SOC assertions
at four Julia threads, 1,724 scalar-Q3 kernel assertions, and six dedicated
benchmark-report/option regressions.  Static Julia parsing, shell syntax, and
`git diff --check` also passed.  This is implementation evidence only: the
expert path is not retained or automatically selected until strict same-node
J40 direct/mixed A/B and then J80 complete-solve gates pass.  Production
remains unchanged.

### Rejected J80 NUMA comparison attempt

The older direct-path NUMA job `199696.node220` terminated after only its
default-placement arm, so it is not an A/B result.  On node104 it requested
32 Julia workers and one BLAS worker, used the exact J80 model/hash, and
reached the 10,800-second per-solve limit after 359 reported iterations.  The
preserved row was `TimeLimit`, failed the duality-gap certificate at
`1.4638e-10` versus the requested `1e-12`, and was therefore correctly marked
invalid.  The final execution provenance was the SDP `block_arrow` backend
despite a planned native-Q3 backend, so it also cannot be used as native-Q3
performance evidence.  The process used about 34.1 GB peak RSS and averaged
about 10.03 active cores (31.3% of the requested pool); `/usr/bin/time`
reported 108,779 user seconds over 3:05:03 elapsed.

Because the first arm exited nonzero, the wrapper did not start the
interleaved-placement arm.  No NUMA conclusion is drawn and the job is not
included in any direct/mixed acceptance ratio.  It does reinforce two gates
already imposed on the new harness: a requested native-Q3 benchmark must
verify the actually executed KKT backend, and a failed certificate must stop
the campaign rather than be interpreted as a timing sample.

### Rejected Float64x2 equality refinement complete-solve path

The implementation candidate was frozen read-only at
`candidates/v0.4.1-mixedrefine-b78db9e15dcf-v2/source`: 313 files, all four
manifests present, and local/remote ordered digest
`b78db9e15dcf2b1a3477873a2afce9b6395cd1b8d2fab836f5ddad14da631954`.
The first archive attempt without macOS metadata suppression created
AppleDouble `._*` files and failed the remote count/digest gate before Julia
loaded; it is a deployment-only rejected artifact.  The v2 archive was first
re-extracted locally, then verified again remotely and made read-only.

PBS job `199724.node220` ran strict direct/mixed/mixed/direct J40 arms on
otherwise idle node9.  Every arm used the same immutable candidate and model
hash, 32 exclusive Julia workers, one BLAS worker, HKM-Q3, no scaling, and a
`1e-12` target.  All four rows were `Optimal`, used
`q3_block_diagonal_equality`, passed execution/benchmark/original-coordinate
certificates, and had zero fallback.  Direct took 191 iterations; mixed took
190.  The mixed objective differed from direct by only `4.7393e-13` relative
and its certified gap was tighter (`1.7321e-13` versus `9.4565e-13`).  RSS was
essentially unchanged.

Performance failed decisively:

| J40 32-worker A/B median | direct | Float64x2 mixed | mixed/direct |
| --- | ---: | ---: | ---: |
| timed solve | 24.328 s | 35.298 s | 1.451 |
| native core | 22.564 s | 33.592 s | 1.489 |
| solve + validation | 25.199 s | 36.187 s | 1.436 |
| active cores | 28.38 | 25.45 | 0.897 |

Direct target Gram/factor consumed 13.251/0.612 seconds.  Mixed avoided those
phases, but fresh shadow preparation consumed 23.250 seconds and target
residual checks another 2.667 seconds.  All 380 mixed RHS attempts passed
without a single refinement correction, proving that convergence/refinement
was not the cost: construction of the Float64x2 shadow itself was slower than
the retained raw-limb Float64x4 Gram.  Within-arm primary timing CV was below
1.4%, so this is not order noise.

Job `199726.node220` then tested the only plausible low-risk rescue: larger
conversion/SYRK chunks on the same real J40 transformed panel, 32 workers,
two warm-ups and five repeats.  Median complete shadow prepare times were
about 0.0948/0.0950/0.1582/0.1040 seconds for 1,024/4,096/8,192/8,400 rows.
All factors were finite with positive pivots; grouping changed the Float64x2
Gram only at roughly `1e-28` relative and Cholesky reconstruction stayed near
machine accuracy.  None beats the approximately 0.073-second direct
Float64x4 Gram-plus-factor cost, even before target-residual products.
Job `199725` was a deployment-only failure caused by a malformed chunk-list
environment value and supplied no data.

Therefore the Float64x2-to-Float64x4 mixed-refinement implementation is
rejected before J80 complete-solve testing and will be removed from the
source, options, benchmark interface, and tests.  The preflight and full-solve
evidence supersede the optimistic arithmetic projection.  The robust report
flattening, Q3-option labeling gates, and already retained 4-by-2 Float64x4
SYRK kernel remain independently useful.  BigFloat256 mixed refinement is
also deferred: it has higher conversion cost and cannot be justified after
the lower-cost Float64x4 target case failed.  Production remains unchanged.

### Rejected Float64x4 4-by-4 equality-Gram microkernel

The next bounded experiment doubled the retained off-diagonal register tile
from four rows by two columns to four rows by four columns.  It kept the same
ascending reduction order for every output, left diagonal and tail paths
unchanged, and produced the same J40 lower-triangle SHA-256
`2334bb9722d9c5b0c41c86e08d3e73030b601b59a40ba1c2fcaf1a6e712b4c0c`.
The strict control and candidate contained 289 files and differed only in the
reduced-arrow `micro_tile=2` versus `micro_tile=4` selector.  Their independently
verified ordered digests were
`ad6e6e78c59f4a77530e84985a758981d5021f3606003a429b22765c16af3102`
and
`973dbcb24795bd52eac1cb6d046ee9a7616f32391b68289fb8b435662a64f41f`.

Jobs `199729`--`199731` and the 8/16-worker portions of
`199732`--`199733` exited before model loading because the private PBS wrappers
first omitted exported model metadata and then the exported thread count.
They are deployment-only failures and contain no timing evidence.  Valid
control/candidate/candidate/control runs were `199735.node220` (node4, eight
workers), `199736.node220` (node7, sixteen workers), and `199734.node220`
(node9, thirty-two workers).  Each used one BLAS worker, two warm-ups and nine
timed calls per arm.  All jobs exited zero with `PASSED` markers.

| J40 Float64x4 Gram | 4-by-2 control | 4-by-4 candidate | reduction |
| --- | ---: | ---: | ---: |
| 8 workers | 0.244122 s | 0.240766 s | 1.37% |
| 16 workers | 0.178244 s | 0.175796 s | 1.37% |
| 32 workers | 0.067441 s | 0.066753 s | 1.02% |

Every arm had the same output hash.  Eight- and thirty-two-worker within-arm
CVs were below 0.7%; the noisier sixteen-worker arms remained below 4.2%.
Effective-core counts were essentially unchanged, and RSS showed only process
startup/order variation rather than a systematic increase.  The gain is
stable but far below the five-percent kernel retention threshold, so no full
solve or J80 campaign is warranted.  The 4-by-4 code and selector are removed;
the already validated 4-by-2 kernel remains the release path.
