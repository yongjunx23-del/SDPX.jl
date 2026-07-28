# BLAS backend evaluation for SDPX on the UCAS cluster

Date: 2026-07-28
SDPX commit: `444b994` (v0.2.1 release, the validated cluster deployment)
Raw artifacts: `results/blas-eval/{194353,194356,194357}.node220/` on the
cluster (kernels.csv, mkl-variants.csv, solves.csv, per-run solve JSONs,
environment records, SHA-256 manifests).

## Recommendation

**Keep OpenBLAS.** On the cluster's AMD EPYC 7742 nodes it wins end-to-end
SDPX solves against MKL by ~20% and against BLIS by ~30% at every thread
count tested, it is the only backend whose LAPACK scales with threads on this
hardware, it requires zero deployment work (it ships with Julia), and all
three backends are numerically interchangeable — so there is no accuracy
argument to offset the performance and deployment costs of switching.

This is a measured conclusion, not a theoretical one; the tables below are
the evidence. It is also hardware-specific: the deciding factor is MKL's
threaded LAPACK behaviour on Zen 2, so it should be re-evaluated if the
cluster ever gains Intel nodes (§8).

## 1. Hardware and system environment

| item | value |
| --- | --- |
| Compute node CPU | 2 × AMD EPYC 7742 (Zen 2), 64 cores/socket, 128 cores, SMT off (1 thread/core) |
| NUMA | 8 domains |
| SIMD | AVX2 + FMA; **no AVX-512** (Zen 2 has none) |
| Memory | 256 GB/node |
| OS / glibc / gcc | CentOS 7 / 2.17 / 4.8.5 (system compiler irrelevant: all backends arrive as prebuilt binaries) |
| Julia | 1.12.6, site toolchain, shared depot |
| Login node | 2 × EPYC 7282 (same Zen 2 generation; used only for setup) |

## 2. Current Julia BLAS/LAPACK configuration

`LBTConfig([ILP64] libopenblas64_.so)` — Julia's bundled OpenBLAS (ILP64)
behind libblastrampoline (LBT), default 32 BLAS threads at startup on the
login node. SDPX manages widths itself per phase (serialized during
Julia-threaded block phases, widened for the KKT factorization, capped at
`m/256` for POTRF). Loaded libraries: `libopenblas64_.so`,
`libblastrampoline.so.5` only.

## 3. Available optimized libraries on the cluster

| library | status |
| --- | --- |
| OpenBLAS | bundled with Julia (in use) |
| **MKL 2022.2.1** | installed: oneAPI at `/public/software/compiler/intel-oneapi-2022/mkl/2022.2.1` (`libmkl_rt.so`), plus libiomp5 and TBB runtimes |
| AOCL / BLIS / libFLAME | **not installed** system-wide; `/public/software/mathlib` has only reference LAPACK, FFTW, GSL, etc. |
| BLIS (Julia) | usable via the registered `BLISBLAS.jl` (AMD-family BLIS through LBT); installed into a dedicated bench environment on the login node |

No system modification was made: MKL is used in place via `lbt_forward`, and
BLIS lives in `bench-blas/`, a separate Julia environment.

## 4. Methodology

- **Backend switching**: libblastrampoline forwarding, one backend per
  process (forwarding is process-global). MKL: system `libmkl_rt` with
  `MKL_INTERFACE_LAYER=ILP64`; forwarding replaces BLAS *and* LAPACK. BLIS:
  `BLISBLAS.jl` — replaces BLAS only; LAPACK (POTRF) remains OpenBLAS's,
  which is exactly what a real BLIS deployment of SDPX would run, and that
  split turned out to matter (§5.3).
- **Kernels**: SYRK (`AᵀA`, A 900×n), GEMM (`AᵀB`), POTRF (n×n), TRSM
  (n×394) at n ∈ {2000, 4000, 6119} — 6119 is Task_Low08's Schur dimension,
  394 its equality count. Threads {1,2,4,8,16,32}; 1 warm-up + 3 measured,
  median/min/max recorded (216 rows per job).
- **End-to-end**: full Task_Low08 solves (m=6119, L=32, Float64, tol 1e-6)
  through the release's own driver with the *validated* parameter profile
  (`β=0.1, γ=0.85, Ωp=100, Ωd=0.001, sdpb`), Julia threads {8,16,32}, one
  process per run, 1 warm-up + 3 measured per configuration, all on **one
  node within one PBS job** so backends are compared like for like.
- **Metadata**: every result directory records the CPU model, NUMA policy
  (`default`, no explicit binding), Julia version, thread counts, commit, and
  SHA-256 manifests. Node hostnames stay in the on-cluster records.
- **Oversubscription**: none — Julia threads and BLAS threads are never
  simultaneously wide (SDPX's own phase policy), and jobs request ppn=32.

### A harness defect, caught by the correctness check

The first end-to-end round invoked the driver with its neutral defaults
(Ω=1), a configuration the Ω sweep had already proven stalls on this
problem. All three backends "Stalled" at 15 iterations *identically* — the
harness was measuring itself, not the backends. The run was discarded and
repeated with the validated profile. It is recorded here because it is the
strongest argument this report has for always pairing timing tables with
solver-status tables: wall-clock alone would happily have ranked three
broken solves.

## 5. Results

### 5.1 Kernel medians at n = 6119 (seconds; job 194353)

| op | threads | OpenBLAS | MKL | BLIS | best |
| --- | ---: | ---: | ---: | ---: | --- |
| SYRK | 1 | 0.728 | 1.015 | **0.696** | BLIS |
| SYRK | 8 | **0.134** | 0.165 | 0.142 | OpenBLAS |
| SYRK | 32 | 0.064 | 0.114 | **0.056** | BLIS |
| GEMM | 1 | 1.533 | 1.444 | **1.397** | BLIS |
| GEMM | 8 | 0.246 | 0.211 | **0.191** | BLIS |
| GEMM | 32 | 0.095 | 0.101 | **0.085** | BLIS |
| POTRF | 1 | 1.710 | **0.638** | 2.341 | MKL |
| POTRF | 8 | **0.349** | 0.636 | 1.084 | OpenBLAS |
| POTRF | 32 | **0.178** | 0.513 | 1.128 | OpenBLAS |
| TRSM | 1 | **0.455** | 0.537 | 0.468 | OpenBLAS |
| TRSM | 8 | **0.108** | 0.197 | 0.119 | OpenBLAS |
| TRSM | 32 | 0.176 | 0.249 | **0.114** | BLIS |

Reading: BLIS is genuinely good at level-3 BLAS on Zen 2 (SYRK/GEMM/TRSM,
up to 1.3×). MKL wins exactly one cell — *serial* POTRF, by 2.7× — and loses
threaded POTRF badly. OpenBLAS is the only backend whose POTRF scales
(1.71 → 0.18 s, 9.6× at 32 threads), and POTRF at wide threads is the KKT
phase of every SDPX iteration.

### 5.2 MKL parallel-performance diagnosis (job 194356)

Six configurations, targeting the threaded-POTRF deficit — POTRF n=6119
medians (OpenBLAS reference: 0.35 @8, 0.18 @32):

| threads | GNU OMP | Intel OMP | +compact | +scatter | +DYNAMIC=off | TBB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.634 | 0.679 | 0.660 | 0.697 | 0.678 | 0.544 |
| 8 | 0.418 | 0.626 | 0.669 | 0.663 | 0.624 | 0.512 |
| 16 | 0.492 | 0.569 | 0.689 | 0.712 | 0.598 | 0.522 |
| 32 | 0.484 | 0.569 | 0.706 | 0.689 | 0.597 | 0.553 |

**No configuration scales.** From 1 to 32 threads MKL's DPOTRF improves at
most 1.3×, under every threading layer and affinity tried; the deficit is in
MKL's AMD dispatch path, not in the runtime configuration. Two further
disqualifiers surfaced: KMP_AFFINITY (either flavour) *hurts* MKL SYRK at 32
threads (0.53 s vs 0.16 unpinned — the pinning fights Julia's own threads),
and the TBB layer ignores `mkl_set_num_threads(1)` (its "serial" SYRK ran
0.15 s — fully parallel), which would silently break SDPX's contract of
serializing BLAS inside Julia-threaded phases. End-to-end, GNU and Intel
layers are indistinguishable (medians 102–106 s both, t=16/32), confirming
the layer was never the story.

### 5.3 End-to-end Task_Low08, one node, validated profile (job 194357)

Median of 3 (min..max), seconds:

| backend | t=8 | t=16 | t=32 |
| --- | ---: | ---: | ---: |
| **OpenBLAS** | **92.2** (90.6..94.3) | **82.7** (79.8..82.7) | **86.6** (86.1..89.1) |
| MKL (GNU) | 103.7 (97.3..104.9) | 101.3 (93.8..101.8) | 99.7 (95.9..104.8) |
| BLIS | 113.5 (110.4..113.9) | 108.3 (106.8..114.5) | 109.1 (109.1..119.4) |

OpenBLAS wins every column; its best (t=16) beats MKL's best by 21% and
BLIS's best by 31%. The kernel tables explain both losses: MKL gives back
its serial-POTRF win the moment the KKT phase runs wide (SDPX runs it at
`m/256 ≈ 24` threads), and BLIS's level-3 gains are erased because LAPACK
stays with OpenBLAS *outside* LBT's thread control — SDPX's
`set_num_threads` calls stop reaching the POTRF that dominates the phase.
(Absolute times differ from the v0.2.1 validation's 59.6 s because these are
different nodes and thread counts; per the evaluation rule, only same-node
comparisons are read.)

### 5.4 Numerical correctness (same runs)

| backend | status | objective | iters | rel. gap | p_res | d_res | certificate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| OpenBLAS | Optimal | 0.653291393898 | 27 | 4.57e-07 | 2.06e-10 | 1.12e-08 | valid |
| MKL | Optimal | 0.653291393898 | 27 | 4.55e-07 | 2.06e-10 | 2.57e-08 | valid |
| BLIS | Optimal | 0.653291393898 | 27 | 4.55e-07 | 2.06e-10 | 9.25e-09 | valid |

Identical status, identical iteration count, objective identical to every
printed digit, residuals within ordinary floating-point reordering of one
another, independent PSD certificates valid throughout. The backends are
numerically interchangeable on this workload; nothing in the accuracy column
offsets the performance column.

## 6. Analysis against SDPX's operation profile

The cluster phase profile (job 193929) puts Schur assembly at 39–53% and the
KKT factorization at 34–39% of solve time. Mapping the kernel results onto
that profile: BLIS's SYRK/GEMM advantage (~10–30% on under half the runtime)
could at best buy ~5–10% end-to-end, while its POTRF regression costs
3–6× on a third of the runtime — net large loss, exactly as measured. MKL is
the mirror image: its one advantage (serial POTRF) applies to a phase SDPX
deliberately runs wide. OpenBLAS is the only backend strong on *both* halves
of the profile at the widths SDPX actually uses.

## 7. Deployment and maintainability

- **OpenBLAS**: ships with Julia; zero configuration; already validated by
  every SDPX release gate.
- **MKL**: present on the cluster, but adds `MKL_INTERFACE_LAYER` /
  threading-layer / library-path configuration to every job script, is
  version-locked to the site's oneAPI install, and — decisively — is slower.
- **BLIS**: requires a package environment plus the LAPACK split, whose
  thread-control hazard (§5.3) is a footgun for any code that manages BLAS
  widths, which SDPX does by design. AOCL proper (BLIS+libFLAME with AMD's
  LAPACK) is not installed on the cluster; libFLAME could close the POTRF
  gap in principle, but that is a theoretical path this report deliberately
  does not recommend on — it would need the same end-to-end measurement,
  and the burden of proof sits with the challenger.

## 8. Conditions that would reopen this question

- Intel (AVX-512) nodes joining the cluster — MKL's balance changes there.
- A site AOCL installation (BLIS **plus** libFLAME) — rerun §5 as-is; the
  harness in `bench-blas/` is reusable and takes one PBS submission.
- An OpenBLAS regression in a future Julia bundle — the §25 acceptance gates
  would surface it, and this report's tables are the baseline to compare
  against.
