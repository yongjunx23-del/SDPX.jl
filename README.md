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
  reduction, and phase-aware BLAS thread control. Native `BigFloat` kernels
  deliberately use one solver thread; an opt-in reduced-arrow preconditioner
  can use Float64x4 workers while retaining BigFloat residual checks.
- **Structure-aware**: sparse constraint storage, a block-arrow KKT path for
  models with shared plus per-block local variables, a no-pair-buffer fused
  kernel for `2x2` blocks, and an optional combined reduced shared panel for
  exact singleton-local arrows.
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

On this problem that changes nothing — at the default tolerance both
arithmetics return the same digits, and the wider one only costs time.
[`examples/02_extended_precision.jl`](examples/02_extended_precision.jl) shows
where the difference actually appears: `Float64` stalls at a tolerance of
1e-14 while `Float64x4` goes on tracking the requested tolerance down to
1e-30.

## Examples

[`examples/`](examples/) holds runnable versions of the above and more — the
LP path and its sparse/dense selector, independent certificate checking
including a solve the certificate refuses to accept, and the JuMP interface.
They are executed by the test suite, so they do not go stale.

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/01_basic_sdp.jl
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
using JuMP, LinearAlgebra, SDPX   # `Symmetric` comes from LinearAlgebra

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
were already rounded away when the input data was created. Native `BigFloat`
assembly and solves are serial and use ownership-aware, allocation-reusing
scalar kernels. The exception is the opt-in exact singleton-arrow mixed path
described below: its independent BigFloat block metric preparation and
Float64x4 reduced panel may use multiple workers, while residual evaluation,
refinement, and fallback remain in BigFloat.

The packed triangular Schur/Gram backend for `Float64x4` and `BigFloat` is
available through `extended_precision_blas=:auto` only when a workload clears
a conservative runtime and memory crossover. Both types use that conservative
policy by default; other arithmetic types remain `:off` unless requested.
Exact singleton-local arrows with only `2x2` PSD blocks have a stronger
Float64x4 specialization: they eliminate each local coefficient in its
three-dimensional symmetric-coefficient space, pack two reduced rows per
block, and form only the lower shared Schur triangle with one blocked SYRK.
`:auto` selects it only after arithmetic, dimension, active-density,
shared-Schur-density, thread, packing-cost, and memory checks. Rejected cases
retain the fused direct kernel and its no-panel/no-pair-buffer storage.
Diagnostics distinguish the Float64x4
`:reduced_arrow[_threaded]_multifloatvec4_syrk`, generic reduced-arrow, and
`:fused_arrow_2x2` paths.
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

For exact singleton-local BigFloat arrows, loading `MultiFloats` adds a
separate guarded `mixed_precision_kkt=:on` path. It builds the coefficient
metric and all residuals in BigFloat, forms and factors the reduced shared
system in Float64x4, applies BigFloat iterative refinement, and automatically
reconstructs and factors the native BigFloat Schur matrix if a guard fails.
The exact refinement guard rejects this path on the medium CSDR benchmark and
safely reconstructs the native factorization, so it is not enabled by default.

BigFloat also has an opt-in staged precision policy:

```julia
result = solve!(
    problem,
    SolverOptions{BigFloat}(
        precision_bits=256,
        working_precision_policy=:auto,
        minimum_working_precision_bits=192,
    ),
)
```

A lower-precision attempt is accepted only after the usual
original-coordinate certificate; otherwise SDPX retries at the requested
precision while time remains. The fixed policy remains the default. The
[native BigFloat report](bench/opt2026/BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md)
records the profile, 1/2/4/8 scaling, precision A/B, memory, and validation.

## Threading

Start Julia with `-t N` to enable threaded block factorisation, dense and sparse
Schur assembly, residual construction, direction recovery, arrow
factorisation/solves, and line search.

This applies generally to immutable fixed-width arithmetic. Most native
`BigFloat` phases remain serial because mutable scalar ownership, allocator
pressure, and per-worker high-precision workspace growth make unrestricted
threading unsafe. Exact singleton-local `2x2` arrows are the validated
exception: block preparation owns disjoint per-block storage and triangular
SYRK tasks own disjoint Schur tiles, so those phases may use the requested
workers without sharing a writable MPFR object.

Scaling depends strongly on problem size — small models do not have enough work
per block to amortise the synchronisation. Small Float64 Schur builds therefore
stay serial automatically. On the cluster medium exact-arrow benchmark, the
final Float64x4 solve took 51.48 / 31.34 / 19.35 / 11.73 seconds with
1 / 2 / 4 / 8 Julia threads. See the
[threading guide](docs/threading.md) and its linked raw protocol.

`Float64x4` and `BigFloat` use the conservative
`extended_precision_blas=:auto` policy by default. The policy never redirects
`Float64` and retains sparse outer products where panel packing is not
predicted to pay for itself. Exact singleton-arrow problems use their
dedicated reduced panel only when its separate structure, memory, density,
dimension, arithmetic, and thread-count crossover predicts a win; otherwise
the fused `2x2` route remains active.

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
- General native BigFloat kernels and distributed Schur
  assembly/factorization remain serial. The ownership-safe native
  reduced-arrow Schur path is a narrow shared-memory exception; the guarded
  reduced-arrow mixed path remains opt-in. On a cluster, use separate jobs or
  job-array elements for non-arrow native BigFloat solves.

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
