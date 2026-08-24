# Third-Party Notices

SDPX.jl is a **derivative work** of SDPJSolver.jl. This file records what is
reused, what is original, and the licence obligations that follow.

## SDPJSolver.jl — the upstream project SDPX is derived from

- Project: <https://github.com/FishboneChiang/SDPJSolver.jl>
- Original author: Li-Yuan Chiang
- Licence: MIT
- Upstream base commit: `51f363b5103a9b89ed1f341f157ef12a97f89e1c`

SDPJSolver.jl's MIT licence permits derivative works provided the original
copyright and permission notice are preserved. They are, in [LICENSE](LICENSE),
which carries **both** the upstream copyright and the SDPX copyright.

### What is directly derived

| SDPX file | Derived from | Nature of derivation |
|---|---|---|
| `README.md` | upstream `README.md` | **Structural derivation.** Problem statement, the primal/dual formulation, and the argument tables follow the upstream document. |
| `test/correctness.jl` (T1/T2 fixtures) | upstream `test/runtests.jl` | The 2×2 toy problem (`A[1,1,1]=1`, `A[2,2,2]=1`, `C[1,2]=C[2,1]=1`, `c=[2,3]`) is upstream's test case, kept so results stay comparable. |
| `.github/workflows/TagBot.yml` | upstream | Standard Julia TagBot workflow, adapted to use `GITHUB_TOKEN` only. |

### Removed rather than kept

`src/legacy_bfgs.jl` — a close derivation of upstream's `sdpBFGS`,
`findFeasibleBFGS`, `f1` and `df`, retaining their algorithm, control flow,
variable names and console output — was **deleted** before the first public
release. It was exported but never called, never tested and never documented
(upstream's own test for it was commented out), so removing it cost nothing and
eliminates the most substantial piece of copied implementation.

The upstream `src/functions.jl` and `src/SDPJSolver.jl` files are also absent.
None of their numerical implementation functions (`getResidue`, `NewtonStep`,
`NewtonStepSparse`, `f1`, `df`, `f`, `sdpBFGS`, or `findFeasibleBFGS`) is
defined by SDPX. The former compatibility names `sdp`, `findFeasible`, and
`setArithmeticType` were retired with the v0.5 single-interface API.

The release CI checks both the removed filenames and the former numerical
function definitions so they cannot be reintroduced accidentally. The
function-by-function migration review remains available in the repository's
Git history; this notice records the maintained release boundary.

### What is original to SDPX

Everything else in `src/` is new: `types/`, `ingest.jl`, `pipeline/`,
`workspace.jl`, `schur.jl`, `kkt.jl`, `step.jl`,
`solver/interior_point.jl`, `stagnation.jl`,
`adaptive_parameters.jl`, `validation.jl`, `lp_solver.jl`, `moi_wrapper.jl`,
`spectrum.jl`, all of `kernels/` (including
`kernels/extended_precision_blas/`), and all package extensions. Upstream had
no typed solver state, no MathOptInterface layer, no threading kernels, no
sparse Schur / arrow KKT path, no extended-precision backends, no dedicated LP
engine, and no structured termination or final-certificate diagnostics.

### Why the upstream copyright is still retained

The README structure and the T1/T2 test fixtures remain derived, and the
project began as a fork of the upstream repository. MIT requires the original
notice to accompany substantial portions of the original software, and
retaining it costs nothing, so [LICENSE](LICENSE) carries both copyright lines.
This is a deliberate choice, not an oversight.

### Algorithmic lineage (ideas, not code)

Neither of the following has any source code vendored in SDPX. They are credited
because the algorithms are theirs:

- **SDPA** — <https://sdpa.sourceforge.net/> — the HRVW/KSH/M search direction
  and the general primal–dual interior-point structure. Upstream SDPJSolver.jl
  credits SDPA likewise.
- **SDPB** — <https://github.com/davidsd/sdpb> (GPL-3.0) — the Cholesky block
  elimination of the KKT system and the symmetric-square Schur construction were
  implemented from the published description of the method. **No SDPB source
  code is copied, translated, or linked**, so SDPX does not inherit SDPB's
  GPL-3.0 terms. If SDPB source is ever vendored, this conclusion must be
  revisited — GPL-3.0 is incompatible with MIT redistribution.

## Clarabel.jl

Used only as an independently installed reference solver in benchmarks, and as
design inspiration for equilibration, adaptive regularisation, and
residual-driven iterative refinement. **No Clarabel source is vendored.**

- Project: <https://github.com/oxfordcontrol/Clarabel.jl>
- Licence: Apache-2.0

## Runtime dependencies

All are permissively licensed and used as ordinary package dependencies, not
vendored:

| Package | Licence | Role |
|---|---|---|
| MathOptInterface.jl | MIT | runtime |
| **MutableArithmetics.jl** | **MPL-2.0** | runtime |
| LinearAlgebra, Printf, Serialization, SparseArrays (Julia stdlib) | MIT | runtime |
| MultiFloats.jl | MIT | weak dep (extension) |
| JLD2.jl | MIT | weak dep (extension) |
| JuMP.jl | MPL-2.0 | test only |
| StableRNGs.jl | MIT | test only |

Two dependencies are MPL-2.0 (Mozilla Public License 2.0), a **file-level**
copyleft: its obligations attach to modified copies of MPL-covered *files*, not
to independent code that merely calls the library. SDPX copies no source from
either project — MutableArithmetics is used only through its public API
(`MA.operate!`, `MA.operate_to!`, `MA.buffered_operate!`) in
`src/kernels/bigfloat.jl` and `src/kernels/extended_precision_blas/`, and JuMP
is used only in tests. MPL-2.0 §3.3 expressly permits distributing a "Larger
Work" under other terms, so SDPX's own MIT licensing is unaffected.

This would change if SDPX ever vendored or modified files from either package;
in that case those files must remain MPL-2.0 and be marked as such.

## Benchmark comparison solvers

MOSEK (commercial, licence required) and Hypatia.jl (MIT) appear in benchmark
scripts as optional external comparisons. Neither is a package dependency, and
no MOSEK code or licence material is included.
