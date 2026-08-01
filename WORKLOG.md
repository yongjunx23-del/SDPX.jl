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
`/public/home/yongjunxu/projects/SDPX.jl/releases/b3b726dad6309e8647f01181d10bf08e78becd88`
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
`/public/home/yongjunxu/projects/SDPX.jl/depot`, initialized by
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
ordinary test failed, and the same Aqua gate had passed in the preceding
network-enabled full run. The next gate is a same-node cluster A/B of the
retained GEMV followed by the immutable J40 CSDR model.
