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
