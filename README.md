# SDPX.jl

[![CI](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yongjunx23-del/SDPX.jl/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SDPX.jl is a native Julia solver for semidefinite programs (SDPs).  It uses a
primal-dual interior-point method with the HRVW/KSH/M search direction and a
Mehrotra predictor-corrector.  The typed API follows the arithmetic of its
inputs, including `Float64`, fixed-width `MultiFloats` types, `Double64`, and
`BigFloat`.

The package is designed for bootstrap calculations whose conditioning can be
too difficult for `Float64`, while retaining a fast path for ordinary sparse
and dense SDPs.  It also has native LP and compact SOC frontends, JuMP/MOI and
Convex.jl adapters, preprocessing, diagnostics, and independent result
certificates.

> **Status: experimental, pre-1.0.** The public API is usable but may change
> between minor versions.  Check [CHANGELOG.md](CHANGELOG.md) before upgrading
> and treat benchmark results as workload-specific evidence.

## Installation

SDPX is not yet registered in the Julia General registry.  Install the current
repository directly:

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

For a checkout under development, use `Pkg.develop(path=".")`.  Optional
frontends and arithmetic backends are loaded as Julia package extensions when
their packages are present.

## First solve

The native model is

\[
\min_x c^T x \quad\text{subject to}\quad
\sum_i x_i A_i^{(l)}-C^{(l)}\succeq0,
\qquad B^T x=b.
\]

`A` is a vector of three-dimensional block arrays, `C` a vector of symmetric
constant blocks, and `B,b` optional affine equalities.  A minimal solve is:

```julia
using SDPX

# minimise cᵀx subject to Σᵢ xᵢ Aᵢ − C ⪰ 0
A = zeros(2, 2, 2)
A[1, 1, 1] = 1
A[2, 2, 2] = 1
C = [0.0 1.0; 1.0 0.0]
c = [2.0, 3.0]

result = solve(c, [A], [C], Matrix{Float64}(undef, 2, 0), Float64[];
               verbosity=0)

result.status       # Optimal
result.pObj         # 4.898979506633980 (the exact value is 2√6)
result.termination  # stopping reason and convergence diagnostics
```

The raw-array call infers the element type from the data.  For repeated solves,
call `ingest` once and reuse the resulting typed `SDPProblem` or `PreparedSolver`.

The complete runnable set is in [`examples/`](examples/).  It is exercised by
the test suite; setup and commands are in [`examples/README.md`](examples/README.md).


## Command-line frontend

The v0.5 development snapshot adds a small SDPB-style command-line policy
layer.  All ordinary options default to `auto`; backend-specific IPM/KKT
parameters stay internal.  After one-time setup:

```bash
julia bin/setup_cli.jl
./bin/sdpx problem.json
```

A high-precision run can be written in the familiar form:

```bash
./bin/sdpx problem.json result.json \
  --precision=840 \
  --dualityGapThreshold=1e-80 \
  --primalErrorThreshold=1e-80 \
  --dualErrorThreshold=1e-80
```

An integer precision is a BigFloat **bit count**.  The result JSON records both
`resolved_options` and the executed automatic `plan`, so `auto` is inspectable.
See [`docs/cli.md`](docs/cli.md).

## Public API

These are the stable-intent entry points.  The package remains experimental,
so code that needs long-term compatibility should pin a release and read the
changelog.

| Entry point | Use |
| --- | --- |
| `solve`, `solve!` | Solve a raw model, `SDPProblem`, or prepared session |
| `ingest`, `SDPProblem` | Validate and store typed block-SDP data |
| `linear_program`, `solve_lp` | Build or solve a native LP |
| `second_order_program`, `solve_socp` | Build or solve LP plus Lorentz-cone models |
| `prepare`, `PreparedSolver` | Reuse an ingested model across objective directions |
| `SolveOptions` | Small all-auto user policy for LP/SOCP/SDP solves |
| `SolverOptions` | Fully resolved typed controls and expert overrides |
| `SDPResult`, `SolveStatus` | Structured result, status, residuals, and timings |
| `solve_summary`, `result_certificate` | Public summary and independent validation |
| `Optimizer` | MathOptInterface optimizer for JuMP and other MOI clients |

Advanced inspection and policy controls live under `SDPX.Experimental`,
including structure classification, preprocessing reports, execution plans,
adaptive parameter recommendations, and fixed-trace analysis.  The
`SDPX.infeasibility_diagnosis` ray check is also experimental.  Spectrum
helpers (`reconstruct_spectrum`, `export_spectrum`) are bootstrap-specific.

Names not listed here, and names beginning with `_`, are internal.  The legacy
`sdp` and `findFeasible` interfaces remain for compatibility; new code should
use the typed API.  Deprecated global setters such as `setArithmeticType` and
`setSparseMode` should be replaced with `SolverOptions`.

## Automatic solve workflow

With the default `:auto` settings SDPX:

1. classifies cone, storage, arithmetic, size, and predicted Schur density;
2. removes exact zero, duplicate, and verified dependent equalities where safe;
3. extracts scalar bounds and exactly fixed variables;
4. selects scaling and an arithmetic-aware kernel/factorization;
5. estimates memory and scheduling costs;
6. runs the guarded predictor-corrector iteration; and
7. reconstructs and certifies the result in the original coordinates.

The common controls are deliberately small:

```julia
result = solve(problem;
    tolerance=1e-8,
    maximum_iterations=300,
    time_limit=60.0,
    threads=4,
    verbosity=1,
    diagnostics=true,
)
```

Use `scaling=:none`, `algorithm=:sdp|:socp`, `sparse=:dense|:sparse`, or typed
`SolverOptions` only when a measured experiment needs an explicit policy.
Primal-to-dual conversion and chordal decomposition are analysis-only; they are
not silently applied to a solve.  See the [automatic pipeline guide](https://yongjunx23-del.github.io/SDPX.jl/pipeline/),
[preprocessing guide](docs/preprocessing.md), and [parameter reference](docs/parameters.md).

## JuMP and MathOptInterface

`SDPX.Optimizer` is a non-incremental MathOptInterface optimizer.  JuMP models
can use affine equalities, scalar bounds and inequalities, PSD triangle
constraints (`PSDCone()`), and second-order cones through the exact PSD-arrow
lift.  Construct `GenericModel{T}` with `SDPX.Optimizer{T}` when the model uses
`Float64x4`, `Double64`, or another supported coefficient type.

See the [JuMP/MOI guide](docs/julia-interface.md) for supported attributes,
typed models, and result access.

## Convex.jl

Convex.jl support is an optional extension.  `SDPX.solve_convex!` solves
completed DCP models through the same MOI optimizer.  Affine LP forms,
Euclidean-norm/SOC forms, and real PSD forms are supported.  Use
`SDPX.convex_semidefinite(n)` for the packed upper-triangle representation, or
`representation=:square` for compatibility with the original square form.

For extended precision, set the Convex problem's `numeric_type` and the SDPX
optimizer's type to the same `T`.  Exponential, power, and other unsupported
cones are not accepted, and Convex warm starts are not currently implemented.
The [Convex interface guide](docs/convex-interface.md) documents atoms,
precision, options, and result fields.

## SOC and fixed-trace PSD blocks

`second_order_program` accepts standard Lorentz constraints directly.  The
compact frontend uses an exact PSD compilation as its correctness reference:

- Dimension-three `Q3` is exactly isomorphic to a real `2×2` PSD cone.
- Generic SOC dimensions still use the PSD-arrow lift.  There is no certified
  native general-dimensional Lorentz Newton method yet.
- The native `algorithm=:socp` path is certified only for a narrow fixed-trace
  `2×2` PSD structure: real symmetric blocks, direct positive traces, exactly
  two local variables, traceless coefficients, and a nonsingular two-coordinate
  tail map.  Unsupported models fall back to the PSD reference when possible.

For a block

\[
X=\begin{bmatrix}a&b\\b&c\end{bmatrix},\qquad a+c=\tau,
\]

PSD is equivalent to `(τ, a-c, 2b) ∈ Q₃`.  The common unit-trace block
`[q r; r 2-q]` is therefore the disk `(q-1)^2+r^2 ≤ 1`.  Use
`SDPX.Experimental.analyze_fixed_trace(problem)` to detect direct or
equality-implied traces; detection is conservative and does not itself change
the model.

The native fixed-trace implementation uses a compact Mehrotra/HKM direction.
`q3_direction=:nt` selects a Nesterov–Todd direction for controlled research;
it is not an automatic target.  HKM remains the default because the matched J40
gate was 16.8% slower with NT.  The equality Gram remains part of the native
path, so fixed trace removes one cone coordinate but does not collapse generic
transformed equality rows.

Automatic selection is intentionally narrower than explicit selection: the
validated release gate requires an exact sparse fixed-width model with at least
4,096 blocks, 8,192 variables, 128 equalities, and `Float64x4`-width arithmetic.
`Float64`, `BigFloat`, smaller models, and unsupported structures retain the
PSD2 reference under `algorithm=:auto`.  Use `algorithm=:socp, scaling=:none`
for an explicit comparison and inspect `result.termination.executed` to see
what actually ran.  The [fixed-trace guide](docs/src/soc-fixed-trace.md) gives
the full scope, diagnostics, and fallback rules.

## Precision guidance

SDPX follows the element type of the input arrays; there is no process-global
arithmetic mode in the typed API.

| Type | Typical use |
| --- | --- |
| `Float64` | Fast baseline for well-scaled models |
| `MultiFloats.Float64x2`/`Float64x4` | Fixed-width extra precision with threaded kernels |
| `DoubleFloats.Double64` | Fixed-width alternative backend |
| `BigFloat` | Arbitrary precision or difficult exponent ranges |

Construct `BigFloat` data inside the desired precision scope:

```julia
setprecision(BigFloat, 256) do
    c = BigFloat[1, 2]
    # Construct A, C, B, and b here as BigFloat values too.
end
```

Raising `precision_bits` after coefficients have been rounded cannot recreate
lost digits.  If a `Float64` solve reports a precision floor, try
`Float64x2`/`Float64x4`; use `BigFloat` when fixed-width range or accuracy is
not enough.  BigFloat defaults to conservative staged working precision and
retries at the requested width if original-coordinate certification fails.

General native BigFloat kernels are serial because MPFR values are mutable.
Ownership-safe independent blocks, triangular tiles, and a few exact local
`2×2` arrow phases may use requested workers.  Mixed-precision factorization is
guarded by residual, condition, rank, memory, and refinement checks and falls
back to native arithmetic when a guard fails.  See the [precision guide](docs/precision.md)
for the full policy.

## Threading guidance

Start Julia with `-t N`; the solver can use threads for block work, Schur
assembly, factorization, residuals, and line search when the arithmetic and
structure make that safe.  BLAS thread count is adjusted around solver phases,
so concurrent solves should run in separate Julia processes rather than sharing
one process.

The fixed-trace release campaign is capped at 32 Julia/solver workers, with one
BLAS thread: J40 uses `1/2/4/8/16/32` and J80 uses `8/16/32`.  This is a
benchmark and launcher policy, not a global solver API limit.  `threads` remains
available to callers for other models; choose a width from complete-solve
measurements and available memory, not from scheduler allocation alone.

The [threading guide](docs/threading.md) records affinity, memory, ownership,
and validation details.  The [cluster workflow](docs/cluster-workflow.md) has
portable PBS templates; keep site-specific paths in the job environment.

## Validation and certificates

Every `SDPResult` contains status, objectives, residuals, iterations, timing,
and diagnostics.  `result_certificate` independently recomputes objective and
affine residuals, componentwise backward errors, complementarity, and PSD
margins in the original coordinates.  A solver status is not treated as
authoritative when this independent check fails.

For optimize-mode failures, `SDPX.infeasibility_diagnosis` checks normalized
homogeneous primal or dual rays.  Only a ray that passes PSD, stationarity,
objective, and contradiction-margin checks can promote a stopped result to
`PrimalInfeasible` or `DualInfeasible`; `:undetermined` is not a feasibility
claim.  The current Newton method does not carry HSD `τ` and `κ` variables, so
it may fail to produce a ray for some infeasible models.

The fixed-trace benchmark harness applies `result_certificate` to every warmup
and timed solve, records the model hash and geometry, and rejects non-`Optimal`
or certificate-invalid rows.  This is the validation boundary used by the
retained performance numbers below.

## Current benchmark summary

These are matched same-node J40 fixed-trace measurements, not cross-solver
claims.  The model has 4,200 PSD2 blocks, 8,400 variables, and 170 equalities;
the release campaign uses one BLAS thread and complete-solve medians.

| Arithmetic / backend | Median solve | Result |
| --- | ---: | --- |
| `Float64x4`, optimized PSD2 | 135.10 s | `Optimal`, certificate-valid |
| `Float64x4`, native HKM-Q3 | 25.71 s | `Optimal`, certificate-valid; 5.26× solver speedup and 5.14× end-to-end |
| `BigFloat256`, optimized PSD2 | 184.99 s | `Optimal`, certificate-valid |
| `BigFloat256`, native HKM-Q3 | 197.07 s | `Optimal`, certificate-valid; 6.5% slower than PSD2 |

The Float64x4 objectives agree to about `2.2e-13` relative; the BigFloat
objectives agree to about `4.3e-13` relative.  Q3 is therefore selected
automatically only for the validated fixed-width gate, while BigFloat remains
on PSD2 under `:auto`.  The equality Gram is still the dominant native-Q3
target on this model.

Benchmark drivers, configuration, and provenance are in
[`bench/soc_fixed_trace/README.md`](bench/soc_fixed_trace/README.md).  The
repository's small smoke tier and retained historical context are indexed in
[`bench/RESULTS.md`](bench/RESULTS.md); do not generalize either table to other
solvers, hardware, tolerances, or problem families.

## Known limitations

- The package is experimental and the API may change before 1.0.
- General-dimensional native SOC Newton steps are not implemented; generic SOC
  models use the exact PSD-arrow reference.
- Native Q3 is limited to certified fixed-trace `2×2` blocks and has a narrow
  automatic gate.  Explicit requests can still fall back after a failed factor
  or certificate check.
- `Float64` can hit an arithmetic floor on ill-conditioned bootstrap models;
  large non-arrow extended-precision models can be limited by dense Schur
  memory and factorization cost.
- General BigFloat parallelism and distributed Schur assembly remain limited by
  mutable MPFR ownership.  Use separate processes for independent solves.
- The direct primal-dual iteration is not a full homogeneous self-dual
  embedding and cannot certify every infeasible model.
- `ingest(...; validate=false)` skips finiteness checks and is intended only for
  trusted benchmark inputs.
- Null-space reduction and chordal detection are tested experimental building
  blocks, not automatic solve transformations.

## Documentation and development

The [Documenter manual](https://yongjunx23-del.github.io/SDPX.jl/) has the
quick start, precision, JuMP/MOI, Convex, SOC, pipeline, parameter, diagnostics,
API, and development pages.  Operational markdown references remain in
[`docs/`](docs/), including the [bridge schema](docs/bridge-schema.md),
[cluster workflow](docs/cluster-workflow.md), [preprocessing](docs/preprocessing.md),
and [threading](docs/threading.md).

For contributions, run the package tests and the small benchmark tier in the
appropriate environment; cluster validation is preferred for expensive
numerical campaigns.  Read [CONTRIBUTING.md](CONTRIBUTING.md), preserve the
certificate boundary, and report arithmetic, tolerance, hardware, thread
configuration, timing boundary, and repeated-run statistics with performance
claims.

The [WORKLOG.md](WORKLOG.md) keeps detailed implementation and measurement
history.  It is intentionally separate from this user-facing overview.

## Acknowledgements and license

SDPX began as a fork of
[SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl); upstream
copyright and derived components are recorded in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).  The HRVW/KSH/M direction is
from the SDPA tradition, while the KKT block-elimination design also draws on
SDPB.  Clarabel.jl is a reference for several equilibration and refinement
ideas.  Contributors and AI-assisted development acknowledgements are listed
in [CONTRIBUTORS.md](CONTRIBUTORS.md).
