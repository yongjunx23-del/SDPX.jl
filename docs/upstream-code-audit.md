# Upstream Code Audit

Date: 2026-07-25

## Scope

This audit compares the SDPX source tree with upstream SDPJSolver.jl at commit
`51f363b5103a9b89ed1f341f157ef12a97f89e1c`. It distinguishes public interface
compatibility from retained numerical implementation.

The comparison covered:

- every function in upstream `src/functions.jl`;
- the upstream module entry point;
- current files under `src/`;
- exact normalized source-line matches;
- Git rename/copy similarity;
- exported legacy names and their call paths.

## Result

No upstream numerical solver implementation remains in `src/`.

The only retained source-level interface is the shape of the public `sdp` and
`findFeasible` calls and the deprecated setter names. Their implementations were
rewritten as thin front ends over `ingest`, `SolverOptions`, and `solve!`. They
do not execute the upstream Newton-step, sparse-step, line-search, restart, or
BFGS code.

The compatibility front ends remain because SDPX promises source compatibility
for existing callers. Removing them would break the public API without removing
any numerical implementation.

## Function-by-function disposition

| Upstream item | SDPX disposition |
|---|---|
| `module SDPJSolver` | Removed; replaced by the independent `SDPX` module |
| global `T`, `mode`, and `sparseMode` | Removed from the solver; options are per solve |
| `getResidue` | Removed; replaced by typed, in-place residual kernels and reusable workspace |
| `NewtonStep` | Removed; replaced by separate Schur, KKT, predictor/corrector, and line-search modules |
| `NewtonStepSparse` | Removed; replaced by incidence-aware sparse assembly, compact active sets, fused arrow kernels, and automatic sparse/dense selection |
| `sdp` | Name and keyword-compatible call surface retained; implementation delegates to the new pipeline |
| `findFeasible` | Name retained; formulation and implementation rewritten, including a bounded auxiliary problem |
| `setArithmeticType` | Deprecated compatibility shim; no numerical algorithm |
| `setSparseMode` | Deprecated compatibility shim; no numerical algorithm |
| `setMode` | Deprecated no-op compatibility shim; no global solver mode |
| `f1`, `df`, and `f` | Removed |
| `sdpBFGS` | Removed |
| `findFeasibleBFGS` | Removed |
| upstream `src/functions.jl` | Removed |
| upstream `src/SDPJSolver.jl` | Removed |

## Replacement efficiency

The replacement solver avoids the main costs in the upstream implementation:

- no explicit matrix inverse in the Newton step;
- no pairwise dense Frobenius-dot construction for supported Gram/Schur paths;
- one triangular Schur part is computed and mirrored only when required;
- Cholesky/KKT workspaces are allocated once and reused;
- sparse blocks store active constraint incidence instead of dense zero blocks;
- exact block-arrow structure is eliminated locally before the global solve;
- all-`2x2` sparse blocks use a fused compute-and-scatter kernel;
- Float64x4 uses blocked, packed, multithreaded extended-precision kernels;
- BigFloat uses serial allocation-free MPFR scalar kernels;
- LP models use a dedicated predictor-corrector engine rather than `1x1` PSD
  blocks in the general SDP engine.

## Retained non-core derivation

The MIT notice remains necessary and appropriate:

- compatibility signatures and keyword names follow the upstream interface;
- the README structure follows the upstream presentation;
- two small analytic test fixtures retain upstream problem data.

These items are documented in `THIRD_PARTY_NOTICES.md`. They are not numerical
solver implementations and do not affect runtime performance.

## Regression guard

The package-quality CI job fails if:

- `src/functions.jl` or `src/SDPJSolver.jl` reappears;
- any former numerical implementation function is defined under `src/`;
- the former `SDPJSolver` module is defined under `src/`.

This guard complements the normal correctness, sparse-path, extended-precision,
and one-thread/four-thread test matrix.
