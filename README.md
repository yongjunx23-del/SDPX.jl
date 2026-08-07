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
  reduction, and phase-aware BLAS thread control. General native `BigFloat`
  kernels deliberately use one solver thread; exact singleton-local `2x2`
  arrows and all-local `2x2` equality cells may use ownership-safe native
  block, triangular, GEMV, and Schur/Gram-tile workers.
- **Structure-aware**: sparse constraint storage, a block-arrow KKT path for
  models with shared plus per-block local variables, a no-pair-buffer fused
  kernel for `2x2` blocks, and an optional combined reduced shared panel for
  exact singleton-local arrows. Large Float64 SDPs with sparse equality
  operators and a sufficiently sparse Schur pattern can instead use a
  lower-triangle CSC Schur matrix, symbolic-reuse sparse Cholesky, and a dense
  multi-right-hand-side equality elimination.
- **Automatic solve planning** — problem classification, equality and LP-row
  presolve, adaptive-pass Ruiz scaling, arithmetic-aware kernel selection,
  memory budgeting, guarded Mehrotra iteration control, and conservative
  parameter profiles happen before factorization.
- **Rank-aware equality solves** — eligible dense systems switch from normal
  equations to rank-revealing QR when factor diagnostics justify its cost.
  Large systems retain the fast Gram path under conservative dimension and
  memory crossovers, with numerical rank exposed in diagnostics. Exactly
  block-diagonal sparse Schur systems apply their local factors directly to
  the equality panel; wider immutable arithmetic uses an automatically gated
  blocked triangular SYRK instead of pairwise column contractions.
- **High-precision owned-storage kernels** — the `BigFloat` path reuses
  independently owned MPFR values for matrix products, triangular solves,
  Cholesky factors, KKT right-hand sides, and LP Hessian assembly.
- **Guarded mixed KKT hierarchy** — an automatically selected dense
  `Float64x4` solve can use
  Float64 first, promote to a cache-blocked `Float64x2` preconditioner when
  needed, and retain native `Float64x4` as the final fallback. Every promoted
  solve must pass a target-precision residual check.
- **JuMP, Convex.jl, and MathOptInterface** modeling support alongside a
  native typed API.
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

## Mathematica

[`mathematica/SDPXLink.wl`](mathematica/README.md) calls SDPX from
Mathematica through a command-line bridge: the problem is exported as JSON
(schema in [`docs/bridge-schema.md`](docs/bridge-schema.md)), Julia runs
`bin/sdpx_solve.jl`, and the result is imported back — with numbers as
strings above `Float64`, so a 256-bit `BigFloat` solve round-trips ~30
correct digits. `SDPXOptimize[c, A, C]` returns an `Association` with the
solution, residuals, and the independent certificate.

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
| `SDPX.Experimental` | namespace for advanced preprocessing, parameter policies, introspection, and backend controls |
| `SDPX.infeasibility_diagnosis` | normalized optimize-mode ray checks; schema may change |
| `SDPX.Experimental.recommended_parameters` | heuristic profiles, actively being recalibrated |
| `reconstruct_spectrum`, `export_spectrum` | bootstrap-specific helpers |
| `sdp`, `findFeasible` | legacy interface inherited from SDPJSolver.jl |
| `setArithmeticType`, `setSparseMode`, `setMode` | deprecated global setters; use `SolverOptions` |

Version 0.3 retains the historical top-level experimental exports for one
deprecation cycle. They are scheduled to stop being exported in 0.4; use
`SDPX.Experimental.name` now. Legacy SDPJSolver-style exports retain their
longer 1.0 compatibility window. `SDPX.api_surface()` returns the exact policy.

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

`B`: $m$ x $n$ `Matrix{T}` or `SparseMatrixCSC{T,Int}`

`b`: $n$-element `Vector{T}`

`β`: factor of reduction in μ in each step

`γ`: factor of reduction in the step size for backtracking line search

`Ωp` and `Ωd` are initial values for the matrices X and Y: $X = Ω_p I, Y = Ω_d I$

`restart`: `true` or `false`. If at any step the primal/dual step sizes are smaller than `minStep`, the *current iterate is kept* and the collapsed side (`X` and/or `Y`) is rescaled by `OmegaStep`, up to `max_restarts` times (default 5; the legacy `maxOmega` kwarg is translated into an equivalent restart cap). Earlier versions discarded the whole iterate on restart; this version keeps it.

`sparse`: `:auto` (default), `true`/`:sparse`, or `false`/`:dense`.
Automatic mode measures coefficient density, aggregate PSD-block pattern
density, and Schur structural density separately. This distinction lets SDPX
keep sparse coefficient and equality matrices while choosing the KKT backend
from the predicted Schur structure. Dense lattice-bootstrap Schur matrices
retain dense Cholesky; sufficiently large Float64 systems at or below the
guarded 10% Schur-density crossover can use sparse Schur Cholesky.

`equilibrate`: legacy compatibility flag. The public `scaling=:auto` default
applies adaptive-pass Ruiz equilibration to SDP models; use `scaling=:none`
only as an expert override.

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
redundancy presolve, adaptive Ruiz scaling, kernel and schedule selection,
safe parameter-profile selection, and guarded per-iteration adaptation.
Models containing only scalar cones use a
dedicated LP predictor-corrector engine instead of the PSD matrix path.
The public `time_limit` covers that pipeline setup, not only barrier
iterations; the raw-array one-call form above also charges input ingestion
against the same limit. A supplied `warm_start` is expressed in original input
coordinates and is mapped through equality presolve and the selected
equilibration automatically.

The frontend also includes conservative typed preprocessing for scalar bounds,
exactly fixed variables, and structural equality cleanup. It preserves
`Float64`, `Float64x4`, and BigFloat arithmetic, maps warm starts through the
reduction, and certifies the returned solution in the original coordinates.
Primal/dual formulation changes and chordal decomposition remain
analysis-only. See [conservative preprocessing](docs/preprocessing.md) for the
stage interfaces, options, reports, and target-model behavior.

The legacy functions are thin wrappers over the expert typed core:

```julia
prob = ingest(c, A, C, B, b; sparse=:auto)         # -> SDPProblem{T}, T inferred from inputs
structure_summary(prob)                            # inspect the selected execution plan
opts = SolverOptions{Float64}(β=0.1, verbosity=1, equilibrate=true, refine_steps=1)
result = solve!(prob, opts)                         # -> SDPResult{T}
```

The same expert options can be constructed without Unicode input:

```julia
opts = SolverOptions(
    Float64;
    tolerance=1e-9,
    maximum_iterations=300,
    time_limit=120.0,
    beta=0.1,
    gamma=0.9,
    verbosity=0,
)
```

Linear programs have a compact native frontend. It accepts ordinary LP rows
directly and stores only active coefficients:

```julia
G = [1.0 0.0; 0.0 1.0; 1.0 1.0]
problem = linear_program(
    [1.0, 2.0], G, [1.0, 1.0, 3.0];
    Aeq=[1.0 1.0], beq=[3.0],
)
result = solve(problem; tolerance=1e-8, threads=4)

# One-call equivalent:
result = solve_lp(
    [1.0, 2.0], G, [1.0, 1.0, 3.0];
    Aeq=[1.0 1.0], beq=[3.0], tolerance=1e-8,
)
```

The convention is `G*x >= h` and `Aeq*x = beq`. Existing scalar-block
`ingest` calls remain fully supported.

For very large block-arrow inputs where each PSD block touches only a small
subset of the global variables, callers can avoid allocating an `L × m`
mostly-empty reference grid:

```julia
block = ActiveSparseCoefficientVector(
    Float64x4,
    m,
    active_variable_ids,       # sorted global indices
    active_sparse_matrices,
    block_dimension,
)
prob = ingest(c, [block, ...], C, B, b; sparse=true)
```

The representation is read-only and preserves the historical
`AbstractVector{SparseMatrixCSC}` interface. Existing dense and expanded
sparse inputs behave unchanged.

Low-level `beta`, `gamma`, `omega_p`, and `omega_d` settings remain available
through `SolverOptions` for expert use. `parameter_strategy=:adaptive`
enables a typed, bounded Mehrotra controller with independent primal/dual
fractions, refinement selection, and complete fixed-path fallback. It records
its diagnostics and selected values per iteration. The adaptive strategy is
the public default; `:fixed` is retained for historical trajectory
reproduction and controlled A/B benchmarks. The expert
`adaptive_sigma_max=0` default delegates the centering cap to the structural
policy; Task_Low08-like dense-Schur lattice systems use the separately
validated 0.20 cap while other profiles retain the generic 0.50 bound.
For the large-lattice profile, `scaling=:auto` also preserves the original
coordinates when `parameter_strategy=:fixed`, because the historical fixed
0.075/0.8 trajectory was calibrated without Ruiz scaling. Adaptive mode keeps
the default Ruiz pipeline.
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

Eligible failed optimize-mode runs check whether the returned iterate defines
a normalized homogeneous ray, regardless of whether verbose diagnostics are
enabled. The report is stored at
`result.termination.infeasibility_diagnosis` and can be recomputed with
`SDPX.infeasibility_diagnosis(prob, result, opts)`. A ray that passes the
independent original-coordinate checks upgrades the result to
`PrimalInfeasible` or `DualInfeasible`; an undetermined candidate leaves the
original stopped status unchanged. The generator is currently direct
primal-dual rather than a full HSD `tau`/`kappa` iteration.

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

## Convex.jl

Convex.jl support is loaded as an optional extension. The high-level helper
keeps solver configuration concise, while `convex_semidefinite` defaults to an
SDPX-native upper-triangle representation:

```julia
import Convex
using SDPX

x = Convex.Variable(2)
problem = Convex.minimize(
    2x[1] + x[2],
    [x >= 0, sum(x) == 1],
)
SDPX.solve_convex!(problem; tolerance=1e-8, threads=4)

X = SDPX.convex_semidefinite(3)
sdp = Convex.minimize(Convex.tr(X), [X[1, 2] == 1])
SDPX.solve_convex!(sdp; tolerance=1e-8, threads=4)

problem.status
problem.optval
Convex.evaluate(x)
```

Use `convex_semidefinite(n; representation=:square)` or the original
`Convex.Semidefinite(n)` for compatibility. The native default uses
`n(n+1)/2` variables and avoids bridge-generated symmetry equalities.

Convex automatically caches the completed model and applies MOI bridges for
SDPX's non-incremental interface. Its affine, second-order-cone, and real PSD
canonicalizations are supported; atoms requiring exponential, power, or other
unsupported cones are not. For `Float64x4` or `BigFloat`, set both
`numeric_type=T` on the Convex problem and use `SDPX.Optimizer{T}`.

See the [Convex.jl interface guide](docs/convex-interface.md) and the
[matched native-versus-Convex benchmark](bench/convex_frontend/README.md).

## Precision backends

Beyond `Float64`/`BigFloat`, `Float64x2`/`Float64x4`/… (via
`MultiFloats.jl`) and `Double64` (via `DoubleFloats.jl`) work as drop-in
element types once the corresponding package is loaded (`using MultiFloats`).
These are fixed-width bitstypes with no MPFR allocation overhead.

`BigFloat` inputs should be constructed inside the intended
`setprecision(BigFloat, bits) do ... end` scope. `solve!` establishes
`precision_bits` for the complete solve, but it cannot recover digits that
were already rounded away when the input data was created. General native
`BigFloat` assembly and solves are serial and use ownership-aware,
allocation-reusing scalar kernels. Exact singleton-local `2x2` arrows and
block-diagonal `2x2` cell systems with all-local Schur variables plus explicit
equalities are the native exceptions. Their independent block work and
complete lower-triangular Schur or equality-Gram tiles may run concurrently
without sharing writable MPFR objects.

For all-local equality cells, fine-grained block, triangular, GEMV,
predictor/corrector, line-search, and update phases use at most 64 ownership
tasks. The equality Gram may still use a wider requested allocation because
its lower-triangular tiles contain enough work to scale across a second
socket. This phase-aware cap is numerical-order preserving: block results are
exclusive and global reductions remain in block order.

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
For dense non-arrow Float64x4 systems that store only the lower Schur
triangle, repeated refinement products use disjoint output-row ownership once
the Schur dimension reaches 1,024. Aliased input/output vectors and smaller
systems retain the established multiplication path.
Memory planning uses the smallest available signal among host free memory,
Linux cgroup limits, and the optional `SDPX_MEMORY_LIMIT_BYTES` environment
limit. For example:

```bash
SDPX_MEMORY_LIMIT_BYTES=64GiB julia --project=. -t 8 solve_problem.jl
```

The limit is a planning ceiling, not a request to reserve that amount.

Dense, non-arrow `Float64x4` and `BigFloat` problems use
`mixed_precision_kkt=:auto` by default. It may select Float64 factorization with
target-precision residuals and iterative refinement, guarded by memory,
conditioning, rank, and convergence checks. Any failed guard recomputes with
the native extended-precision factorization. Mixed refinement targets the
tighter of the arithmetic floor and
the square of the requested solver tolerance, so a moderately conditioned
Float64 preconditioner is not rejected merely because it cannot reproduce
unused Float64x4 or BigFloat digits. The predictor is corrected only to its
separate safety guard; stalled or worsening corrections still trigger the
native fallback. For fixed-width arithmetic, explicit `:on` is a measured
expert mode: conservative condition and predicted-step estimates remain in
diagnostics, while actual target-precision residual reduction decides whether
to continue. `:auto` and BigFloat retain the static condition cutoff.

For exact singleton-local BigFloat arrows, loading `MultiFloats` adds a
separate guarded `mixed_precision_kkt=:on` path. It builds the coefficient
metric and all residuals in BigFloat, forms and factors the reduced shared
system in Float64x4, applies BigFloat iterative refinement, and automatically
reconstructs and factors the native BigFloat Schur matrix if a guard fails.
The exact refinement guard rejects this path on the medium CSDR benchmark and
safely reconstructs the native factorization, so it is not enabled by default.

BigFloat also uses a staged precision policy by default:

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
precision while time remains. Use `working_precision_policy=:fixed` for one
requested-precision attempt. The
[native BigFloat report](bench/opt2026/BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md)
records the profile, 1/2/4/8 scaling, precision A/B, memory, and validation.

## Threading

Start Julia with `-t N` to enable threaded block factorisation, dense and sparse
Schur assembly, residual construction, direction recovery, arrow
factorisation/solves, and line search.

This applies generally to immutable fixed-width arithmetic. Most native
`BigFloat` phases remain serial because mutable scalar ownership, allocator
pressure, and per-worker high-precision workspace growth make unrestricted
threading unsafe. Exact singleton-local `2x2` arrows and all-local 2x2 cell
systems with explicit equalities are the validated exceptions: block work owns
disjoint storage, and triangular SYRK tasks own disjoint Schur or equality-Gram
tiles, so those phases may use the requested workers without sharing a
writable MPFR object.

On the certified J40 BigFloat512 CSDR model, a uniform 128-worker schedule was
34.5% slower than 64 workers even though equality Gram time improved. Capping
the fine-grained phases at 64 tasks reduced the 128-worker solver from 495.81
to 425.88 seconds and peak RSS from 4,346,976 to 4,058,792 KiB, with a
bit-for-bit identical certificate. The 64-worker solve remains faster at
368.70 seconds; a 96-worker crossover took 398.30 seconds. Therefore 64 is the
recommended width for this geometry, and wider runs should be reserved for
larger equality panels after measurement.

A fixed 1,024-bit run of the same J40 model also passed its complete physical
certificate in 157 iterations. It took 553.96 seconds at 64 workers versus
368.70 seconds for 512 bits, while its relative gap (`1.89e-13`) was not tighter
than the 512-bit result (`3.45e-14`). Because the archived coefficients were
rounded once to `Float64x4`, extra solver precision cannot recover input digits.
Use 512 bits for this model unless a tighter, genuinely higher-precision input
or certificate requirement justifies the additional cost.

Scaling depends strongly on problem size — small models do not have enough work
per block to amortise the synchronisation. Small Float64 Schur builds therefore
stay serial automatically. Block-local residual, factorization, predictor, and
corrector work uses both block count and estimated cubic work, so models such
as Task_Low08 with only 32 but moderately large PSD blocks still use safe
disjoint-block parallelism. A same-node 32-Julia-thread / 16-BLAS-thread A/B
reduced its median adaptive solve from 32.062 to 28.438 seconds without
changing the iteration trajectory or certificate. On the 1,700-block /
144-shared-variable medium CSDR model, a resource-instrumented Float64x4 sweep
measured 44.112 / 22.954 / 12.450 / 6.946 / 4.275 / 3.286 / 2.967 / 3.074 /
3.540 / 3.403 seconds with 1 / 2 / 4 / 8 / 16 / 32 / 48 / 64 / 96 / 128
Julia workers and one BLAS thread. All requested pools were observed active;
48 workers were fastest (14.87x over one worker), while wider pools lost to
synchronization and NUMA traffic. The retained cache-hot local factor and SIMD
arrow-solve changes later reduced the controlled 48-worker median to 2.895
seconds with the same 41-iteration certificate. See the
[threading guide](docs/threading.md) for exact affinity, sleep-policy, memory,
and validation details.

Dense Schur accumulation is also memory-aware. The generic per-solve limit is
15% of available memory. Large `Float64` systems may use 25% only when they
have at least 4,096 variables, 16 PSD blocks, 16 requested workers, and an
explicitly visible budget of at least 16 GiB. This narrow Task_Low08-calibrated
rule reduced median Schur assembly from 8.140 to 6.701 seconds; MultiFloat,
BigFloat, smaller problems, and memory-constrained jobs retain 15%.

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
The narrowly scoped
[`Task_Low08` cluster report](bench/opt2026/TASK_LOW08_PRECISION_CLUSTER_REPORT_2026-07-27.md)
adds same-node MOSEK comparisons, 1--128-thread SDPX scaling, extended-precision
memory gates, and full numerical certificates.

## Known limitations

- The package is **experimental**; the API may change before 1.0.
- Optimize mode reports formal `PrimalInfeasible` and `DualInfeasible`
  statuses when an independently normalized homogeneous ray passes the
  original-coordinate certificate. The current direct primal-dual iteration
  does not yet carry HSD `τ` and `κ`, so it may fail to generate a ray for an
  infeasible model and return an ordinary stopped status instead.
- The sparse conformal-bootstrap benchmark in `bench/` does not yet converge to
  the tolerance a reference solver reaches on the same instance.
  `bench/csdr_psd_dual/RESULTS.md` records current evidence, while
  `bench/opt2026/REPORT.md` preserves the historical optimization log.
- `Float64` is precision-limited on ill-conditioned bootstrap models; use
  `Float64x2`/`Float64x4` when a solve reports `:precision_floor`.
- Non-arrow extended-precision SDP problems and Float64 problems outside the
  guarded sparse-Schur crossover still use a dense Schur/KKT fallback. Large
  dense instances can therefore be limited by quadratic workspace and cubic
  factorization cost before arithmetic accuracy becomes the issue. The sparse
  equality-aware path is currently Float64-only.
- General native BigFloat kernels and distributed Schur
  assembly/factorization remain serial. The ownership-safe native
  reduced-arrow Schur path is a narrow shared-memory exception; the guarded
  reduced-arrow mixed path is conservatively auto-gated. On a cluster, use
  separate jobs or job-array elements for non-arrow native BigFloat solves.
- **One solve per process.** The solver adjusts the process-global BLAS thread
  count around its phases, so two concurrent `solve` calls in one Julia
  process can interleave those adjustments and oversubscribe or finish with a
  stale setting. Run concurrent solves in separate processes (the pattern the
  cluster scripts already use).
- `ingest(...; validate=false)` skips finiteness checking entirely: NaN or Inf
  coefficients enter the solver unchecked and surface only as a non-finite
  iterate many iterations later. It exists for benchmark drivers that validate
  inputs by checksum; leave validation on otherwise.
- The null-space reduction (`src/nullspace.jl`) and chordal detection
  (`src/chordal.jl`) are **experimental, opt-in building blocks** — tested,
  but not reachable from `solve`. No benchmark in this repository qualifies
  for the null-space formulation (CSDR models carry no equality rows; the
  lattice benchmark constrains 6% of variables against a 50% threshold), so
  wiring it into the automatic pipeline is deferred until a qualifying model
  family exists to gate it against.

## Design and benchmark notes

- [Solver parameter reference](docs/parameters.md)
- [Adaptive interior-point parameter policy](docs/adaptive-parameter-policy.md)
- [Adaptive dense/sparse optimization and Task_Low08 results](docs/adaptive-dense-sparse-optimization.md)
- [Performance and accuracy roadmap](docs/performance-roadmap.md)
- [JuMP and MathOptInterface guide](docs/julia-interface.md)
- [Julia interface roadmap](docs/julia-interface-plan.md)
- [Open-source release checklist](docs/open-source-release-checklist.md)
- [Cluster deployment and execution guide](docs/cluster-workflow.md)
- [Matched CSDR PSD-dual benchmark](bench/csdr_psd_dual/README.md)
- [Extended-precision BLAS benchmark report](bench/extended_precision_blas/REPORT.md)
- [Guarded mixed-precision KKT benchmark](bench/mixed_precision_kkt/RESULTS.md)
