# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A native Julia [semidefinite programming](https://en.wikipedia.org/wiki/Semidefinite_programming)
(SDP) solver: primal-dual interior-point method with the HRVW/KSH/M search
direction and a Mehrotra predictor-corrector, in arbitrary precision.

Motivated by the S-matrix/EFT bootstrap, the modular bootstrap, and the lattice
bootstrap, where the resulting programs are badly conditioned enough to need
more than `Float64`.

> **Status: experimental.** The public API may change between minor versions
> before 1.0. The solver is under active development and is not yet recommended
> for production use. See [Known limitations](#known-limitations).

## Features

- **Arbitrary precision** — `Float64`, `BigFloat`, `MultiFloats.Float64xN`,
  and `DoubleFloats.Double64`, the last three via package extensions.
- **Multithreaded fixed-width arithmetic** — `Float64` and
  `MultiFloats.Float64xN` use cost-aware block scheduling, triangular Schur
  reduction, and phase-aware BLAS thread control. `BigFloat` deliberately uses
  one solver thread.
- **Structure-aware**: sparse constraint storage, a block-arrow KKT path for
  models with shared plus per-block local variables, and a fused kernel for
  models whose blocks are all 2x2. The fused path does not allocate the
  otherwise-large transformed-panel or pair-buffer storage.
- **Automatic solve planning** — problem classification, equality and LP-row
  presolve, scaling, arithmetic-aware kernel selection, memory budgeting, and
  conservative parameter profiles happen before factorization.
- **High-precision owned-storage kernels** — the `BigFloat` path reuses
  independently owned MPFR values for matrix products, triangular solves,
  Cholesky factors, KKT right-hand sides, and LP Hessian assembly.
- **JuMP / MathOptInterface** wrapper alongside a native typed API.
- **Diagnostics that explain themselves** — a solve that stops short reports
  *why*, with the measured convergence rate and whether the limit was the
  arithmetic precision.

## Installation

Not yet registered in the Julia General registry. Install directly from the
repository:

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

## Quick start

```julia
using SDPX

# minimise cᵀx  subject to  Σᵢ xᵢ Aᵢ − C ⪰ 0
A = zeros(2, 2, 2)
A[1, 1, 1] = 1
A[2, 2, 2] = 1
C = [0.0 1.0; 1.0 0.0]
c = [2.0, 3.0]

result = solve(c, [A], [C], Matrix{Float64}(undef, 2, 0), Float64[]; verbosity=0)

result.status        # Optimal
result.pObj          # 4.898979506633980  (exact optimum 2√6)
result.termination   # why it stopped, with the measured convergence rate
```

Drop `verbosity=0` to see the per-iteration log. For higher precision, build the
inputs at the element type you want — it is inferred from the data:

```julia
using MultiFloats                     # enables the Float64xN backend
result = solve(Float64x4.(c), [Float64x4.(A)], [Float64x4.(C)],
               Matrix{Float64x4}(undef, 2, 0), Float64x4[]; verbosity=0)
```

## Public API

The package is pre-1.0 and marked experimental, but these are the entry points
intended for users, and changes to them will be noted in
[CHANGELOG.md](CHANGELOG.md):

| Stable-intent | Purpose |
|---|---|
| `solve`, `solve!` | run a solve |
| `ingest`, `SDPProblem` | build a typed problem |
| `SolverOptions`, `SDPResult`, `SolveStatus` | configure and inspect |
| `Optimizer` (MathOptInterface) | use from JuMP |

| Experimental | Caveat |
|---|---|
| `analyze_structure`, `structure_summary`, `classify_problem`, `build_execution_plan` | introspection; shapes may change |
| `recommended_parameters` | heuristic profiles, actively being recalibrated |
| `reconstruct_spectrum`, `export_spectrum` | bootstrap-specific helpers |
| `sdp`, `findFeasible` | legacy interface inherited from SDPJSolver.jl |
| `setArithmeticType`, `setSparseMode`, `setMode` | deprecated global setters; use `SolverOptions` |

Anything not listed, and anything prefixed with `_`, is internal and may change
without notice.

## Acknowledgements

SDPX began as a fork of
[SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl) by Li-Yuan
Chiang (MIT) and has since been substantially rewritten into a modular solver
core. `src/compat.jl` still
preserves the upstream `sdp`/`findFeasible` interface for source compatibility,
so the upstream copyright is retained in [LICENSE](LICENSE) alongside SDPX's.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records exactly what is derived
and what is original.

The algorithms come from [SDPA](https://sdpa.sourceforge.net/) (the HRVW/KSH/M
direction) and [SDPB](https://github.com/davidsd/sdpb) (Cholesky block
elimination of the KKT system, symmetric-square Schur construction). Neither
project's source code is included.
[Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) informed the design
of equilibration, adaptive regularisation, and residual-driven iterative
refinement, and is used as a reference solver in benchmarks.

Development also benefited from ChatGPT by OpenAI and Claude by Anthropic for
code review, algorithm exploration, performance analysis, test design,
documentation, and release preparation. They are credited as AI-assisted
development tools, not as legal authors or copyright holders. See
[CONTRIBUTORS.md](CONTRIBUTORS.md) for the complete contributor and
acknowledgement record.

## The optimization problem
The function
```julia
sdp(c, A, C, B, b;
    β=0.1, γ=0.9, Ωp=1, Ωd=1,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300,
    restart=true, minStep=1e-10, maxOmega=1e50, OmegaStep=1e5)
```
solves the following SDP:

### Primal
$$
    \begin{aligned}
        \text{Minimize } \quad & c^T x \\
        \text{subject to } \quad & X^{(l)} = \sum_i x_i A_i^{(l)} - C^{(l)} \geq 0, \quad l = 1, 2, ..., L \\
        & B^T x = b
    \end{aligned}
$$

### Dual
$$
    \begin{aligned}
        \text{Maximize } \quad & \sum_l tr(C^{(l)} Y^{(l)}) + b^T y \\
        \text{subject to } \quad & \sum_l tr(A_{\star}^{(l)} Y^{(l)}) + B y - c = 0 \\
        & Y^{(l)} \geq 0, \quad l = 1, 2, ..., L
    \end{aligned}
$$

### Domain
$$
    \begin{aligned}
        x & \in \mathbb{R}^m \\
        B & \in \mathbb{R}^{m \times n} \\
        b, y & \in \mathbb{R}^n \\
        A_i^{(l)}, C^{(l)}, X^{(l)}, Y^{(l)} & \in \mathbb{S}^{k^{(l)}} \\
    \end{aligned}
$$

## Interior-point method
In each iteration, the program solves the following deformed KKT conditions to determine the Newton step:
- Primal feasibility

$$ X^{(l)} = \sum_i x_i A_i^{(l)} - C^{(l)} $$

- Dual feasibility

$$ \sum_l tr(A_{\star}^{(l)} Y^{(l)}) + B y - c = 0 $$

- Complementarity

$$ X^{(l)} Y^{(l)} = \mu^{(l)} I $$

Mehrotra's predictor-corrector method is used to accelerate convergence, with one step of iterative refinement on the predictor/corrector solve by default.

After a search direction is obtained, the step size is determined by requiring that $X$ and $Y$ remain positive (allocation-free backtracking line search).

## The feasibility problem
The function
```julia
findFeasible(A, C, B, b;
    β=0.1, Ωp=1, Ωd=1, γ=0.9,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300, restart=true, minStep=1e-10, t_max=nothing)
```
determines whether the SDP above is feasible. Note that the arguments are basically the same as `sdp()` except no vector `c` for the objective function is needed. The function converts the feasibility problem to the following optimization problem:

$$
    \begin{aligned}
        \text{Minimize } \quad & t \\
        \text{subject to } \quad & X^{(l)} = \sum_i x_i A_i^{(l)} - C^{(l)} + t I\geq 0, \quad l = 1, 2, ..., L \\
        & B^T x = b \\
        & t \leq t_{max}
    \end{aligned}
$$

If $t^* \geq 0$, the problem is infeasible; otherwise, the problem is feasible. `t_max` (default `10·(1+maxₗ‖C^{(l)}‖∞)`) keeps this auxiliary problem bounded, so — unlike earlier versions — `findFeasible()` terminates even when the original feasible set (if any) is unbounded.

## Inputs

`prec`: arithmetic precision in base-10, which is equivalent to
```julia
setprecision(prec, base = 10)
```
The default arithmetic type is `BigFloat`, which supports arbitrary precision. The type is now inferred from the input arrays themselves — build `A`/`C`/`B`/`b`/`c` at the type you want (`Float64`, `BigFloat`, or a `MultiFloats.jl`/`DoubleFloats.jl` type) and `sdp`/`findFeasible` will solve at that type. `setArithmeticType(Float64)` still works as a deprecated fallback for inputs that don't carry their own type (e.g. all-`Int` literals).

`c`: $m$-element `Vector{T}`

`A`: $L$-element `Vector{Array{T, 3}}`

`C`: $L$-element `Vector{Matrix{T}}`

`B`: $m$ x $n$ `Matrix{T}`

`b`: $n$-element `Vector{T}`

`β`: factor of reduction in μ in each step

`γ`: factor of reduction in the step size for backtracking line search

`Ωp` and `Ωd` are initial values for the matrices X and Y: $X = Ω_p I, Y = Ω_d I$

`restart`: `true` or `false`. If at any step the primal/dual step sizes are smaller than `minStep`, the *current iterate is kept* and the collapsed side (`X` and/or `Y`) is rescaled by `OmegaStep`, up to `max_restarts` times (default 5; the legacy `maxOmega` kwarg is translated into an equivalent restart cap). Earlier versions discarded the whole iterate on restart; this version keeps it.

`sparse`: `:auto` (default), `true`/`:sparse`, or `false`/`:dense`.
Automatic mode measures coefficient density, aggregate PSD-block pattern
density, and Schur structural density separately. This distinction lets SDPX
keep sparse coefficient matrices while using a dense PSD kernel and dense
Cholesky when, as in lattice-bootstrap models, the latter structures are
nearly dense.

`equilibrate`: `true` or `false` (default `false`). Opt-in Ruiz-style diagonal equilibration, useful for badly-scaled bootstrap data.

`termination`: `:relative` (default) or `:legacy`. `:relative` normalizes the
gap/residual tests by problem scale, fixing a case where the original absolute
test could silently run to `iterMax` when the gap overshot to a small negative
number. `:legacy` is a legacy-like absolute/nonnegative-gap convention; it
still retains modern inclusive boundaries and post-solve certification.

The iteration terminates in one of: `Optimal`, `Feasible`/`Infeasible` (for
`findFeasible`), `Stalled`, `IterLimit`, `TimeLimit`,
`MaxRestartsExceeded`, `NumericalBreakdown`, or `UserStopped` (via
`callback`) — every run ends in an informative status, never a silent
`iterMax` burn.

## Outputs
`sdp()`/`findFeasible()` return an `SDPResult`, which keeps the original `Dict`-style access for compatibility:
- `prob["x"]`, `prob["X"]`, `prob["y"]`, `prob["Y"]`
- `prob["pObj"]`, `prob["dObj"]`
- `prob["status"]`: `"Optimal"`, `"Feasible"`, `"Infeasible"`, or a message describing why the run stopped

New code can use the typed fields directly (`prob.status::SolveStatus`, `prob.iterations`, `prob.restarts`, `prob.gap_rel`, `prob.p_res`, `prob.d_res`).

## New API

The recommended interface exposes only common controls:

```julia
result = solve(
    c, A, C, B, b;
    tolerance=1e-8,
    maximum_iterations=300,
    time_limit=60.0,
    threads=4,
    precision=:float64,
    precision_bits=256, # used only when precision=:bigfloat
    verbosity=1,
    diagnostics=true,
    warm_start=nothing,
)
```

The automatic pipeline performs problem classification, equality and
redundancy presolve, scaling, kernel and schedule selection, and safe
parameter-profile selection. Models containing only scalar cones use a
dedicated LP predictor-corrector engine instead of the PSD matrix path.
The public `time_limit` covers that pipeline setup, not only barrier
iterations; the raw-array one-call form above also charges input ingestion
against the same limit. A supplied `warm_start` is expressed in original input
coordinates and is mapped through equality presolve and the selected
equilibration automatically.

The legacy functions are thin wrappers over the expert typed core:

```julia
prob = ingest(c, A, C, B, b; sparse=:auto)         # -> SDPProblem{T}, T inferred from inputs
structure_summary(prob)                            # inspect the selected execution plan
opts = SolverOptions{Float64}(β=0.1, verbosity=1, equilibrate=true, refine_steps=1)
result = solve!(prob, opts)                         # -> SDPResult{T}
```

Low-level `beta`, `gamma`, `omega_p`, and `omega_d` settings remain available
through `SolverOptions` for expert use. `parameter_strategy=:adaptive`
enables bounded feedback control and records its values per iteration; fixed
parameters remain the default until broader benchmarks show a stable win.
`SolverOptions` also exposes: `callback` (per-iteration `(state) -> Bool`,
`true` stops the solve), `checkpoint_every`/`checkpoint_path` (crash-safe
iterate-level warm restart via `resume=path` on the SDP `solve!` path),
`max_time`,
`predictor=:classic|:sdpb`, and `convert_inputs` (normalize independent
`BigFloat` storage to `precision_bits`; this cannot recover digits already
lost in the source data).

A checkpoint restores the primal/dual iterate, centering targets, and
iteration/restart counters. Adaptive-controller history, stagnation windows,
phase-timing history, and best-iterate history restart empty, so a resumed run
is not a bit-for-bit continuation of an uninterrupted solve. The dedicated LP
path does not currently support checkpoint resume.

## JuMP and MathOptInterface

`SDPX.Optimizer` is a non-incremental MathOptInterface optimizer. It supports
scalar linear inequalities, affine equalities, second-order cones, and
positive-semidefinite triangle constraints, including JuMP's `PSDCone()`
syntax:

```julia
using JuMP, SDPX

model = Model(() -> SDPX.Optimizer(sparse=:auto, verbosity=0))
@variable(model, x[1:2])
@constraint(model, Symmetric([x[1] -1.0; -1.0 x[2]]) in PSDCone())
@objective(model, Min, 2x[1] + 3x[2])
optimize!(model)

termination_status(model)
objective_value(model)
value.(x)
```

Use `GenericModel{T}` and `SDPX.Optimizer{T}` for fixed-width extended
precision such as `Float64x4`. The interface defaults to `sparse=:auto` and
builds sparse coefficient matrices directly from MOI terms.
See the [JuMP and MathOptInterface guide](docs/julia-interface.md) for the
supported forms, options, precision types, and current limitations.

## Precision backends

Beyond `Float64`/`BigFloat`, `Float64x2`/`Float64x4`/… (via
`MultiFloats.jl`) and `Double64` (via `DoubleFloats.jl`) work as drop-in
element types once the corresponding package is loaded (`using MultiFloats`).
These are fixed-width bitstypes with no MPFR allocation overhead.

`BigFloat` inputs should be constructed inside the intended
`setprecision(BigFloat, bits) do ... end` scope. `solve!` establishes
`precision_bits` for the complete solve, but it cannot recover digits that
were already rounded away when the input data was created. The current
`BigFloat` implementation is serial and uses ownership-aware,
allocation-reusing scalar kernels; requesting more solver threads does not
parallelize it.

The packed triangular Schur/Gram backend for `Float64x4` and `BigFloat` is
available through `extended_precision_blas=:auto`, but remains off by default
until each workload clears a conservative runtime and memory crossover.
Exact-arrow models with only `2x2` PSD blocks bypass that optional backend for
both types: the fused direct kernel needs no transformed panel or pair buffer
and takes precedence even when packed extended BLAS is requested. Diagnostics
report `gram_kernel=:fused_arrow_2x2` and
`gram_kernel_reason=:fused_arrow_specialized` for this case.
Memory planning uses the smallest available signal among host free memory,
Linux cgroup limits, and the optional `SDPX_MEMORY_LIMIT_BYTES` environment
limit. For example:

```bash
SDPX_MEMORY_LIMIT_BYTES=64GiB julia --project=. -t 8 solve_problem.jl
```

The limit is a planning ceiling, not a request to reserve that amount.

Dense, non-arrow `Float64x4` and `BigFloat` problems can also opt into
`mixed_precision_kkt=:auto`. It uses Float64 factorization with
target-precision residuals and iterative refinement, guarded by memory,
conditioning, rank, and convergence checks. Any failed guard recomputes with
the native extended-precision factorization. The feature remains off by
default while large complete-solve validation is still being collected.

## Threading

Start Julia with `-t N` to enable threaded block factorisation, dense and sparse
Schur assembly, residual construction, direction recovery, arrow
factorisation/solves, and line search.

This applies to immutable fixed-width arithmetic. `BigFloat` is deliberately
kept serial: mutable scalar ownership, allocator pressure, and per-worker
high-precision workspace growth make the serial owned-storage path the only
release configuration currently validated. This is a solver policy, not a
general statement that MPFR can never be called concurrently.

Scaling depends strongly on problem size — small models do not have enough work
per block to amortise the synchronisation. Small Float64 Schur builds therefore
stay serial automatically. On the current scheduler microbenchmarks,
Float64x4 reached 4.08x on a medium dense Schur case and 3.73x on a medium
sparse exact-arrow case at eight requested workers; the host exposed four
hardware threads, so these are scheduler measurements rather than an
eight-core claim. See the [threading guide](docs/threading.md) and its linked
raw protocol.

`Float64x4` and `BigFloat` can opt into the blocked triangular Schur backend
with `extended_precision_blas=:auto`. It is off by default, never redirects
`Float64`, keeps BigFloat serial, and retains sparse outer products where panel
packing is not predicted to pay for itself. The fused exact-arrow `2x2` route
is a higher-priority specialization and therefore does not allocate those
panels for either arithmetic type.

## Benchmarks

Benchmark drivers live in `bench/`. Their inputs and outputs are **not**
committed — the serialised problem instances run to gigabytes — so regenerate
them locally with `bench/generate.jl`. Only the `:small` tier runs in CI, as a
smoke test.

```bash
julia --project=bench -e 'include("bench/run.jl"); main(tiers=(:small,))'
```

**No claim is made that SDPX is faster than MOSEK, SDPB, Clarabel, or any other
solver.** Cross-solver comparisons depend on the problem, tolerance, precision,
thread count, and hardware, and the benchmarks here are not broad enough to
support a general statement. `bench/opt2026/REPORT.md` records measurements —
including cases where SDPX does worse — with the configuration for each.

## Known limitations

- The package is **experimental**; the API may change before 1.0.
- The sparse conformal-bootstrap benchmark in `bench/` does not yet converge to
  the tolerance a reference solver reaches on the same instance.
  `bench/csdr_psd_dual/RESULTS.md` records current evidence, while
  `bench/opt2026/REPORT.md` preserves the historical optimization log.
- `Float64` is precision-limited on ill-conditioned bootstrap models; use
  `Float64x2`/`Float64x4` when a solve reports `:precision_floor`.
- Non-arrow SDP problems still use a dense Schur/KKT fallback. Large dense
  high-precision instances can therefore be limited by quadratic workspace
  and cubic factorization cost before arithmetic accuracy becomes the issue.
- BigFloat kernels are serial, and distributed Schur assembly/factorization is
  not implemented. On a cluster, run independent BigFloat solves as separate
  jobs or job-array elements.

## Design and benchmark notes

- [Solver parameter reference](docs/parameters.md)
- [Adaptive dense/sparse optimization and Task_Low08 results](docs/adaptive-dense-sparse-optimization.md)
- [Performance and accuracy roadmap](docs/performance-roadmap.md)
- [JuMP and MathOptInterface guide](docs/julia-interface.md)
- [Julia interface roadmap](docs/julia-interface-plan.md)
- [Open-source release checklist](docs/open-source-release-checklist.md)
- [Cluster deployment and execution guide](docs/cluster-workflow.md)
- [Matched CSDR PSD-dual benchmark](bench/csdr_psd_dual/README.md)
- [Extended-precision BLAS benchmark report](bench/extended_precision_blas/REPORT.md)
- [Guarded mixed-precision KKT benchmark](bench/mixed_precision_kkt/RESULTS.md)
