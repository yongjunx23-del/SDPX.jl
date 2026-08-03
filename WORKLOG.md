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
/public/home/yongjunxu/projects/SDPX.jl/releases/0a71efed779617f5bd57af3359efc7523c9f0188
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
`/public/home/yongjunxu/projects/SDPX.jl/experiments/float64x4-lazypartial-d9c287af-clean`;
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
`bench/finite_support_lp/` and `docs/finite-support-lp-optimization.md`.
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
`/public/home/yongjunxu/projects/chapter3-sdpx-lp` was compared with the
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
