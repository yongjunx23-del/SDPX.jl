# Architecture

SDPX has one production mathematical path:

```text
Model / MOI / CLI
        |
        v
canonical product-cone program
        |
        v
presolve + exact reconstruction stack
        |
        v
product-cone HSD state (x, s, y, tau, kappa)
        |
        v
five-equation NewtonSystem
        |
        +-- bordered
        +-- expanded exact operator
        +-- sparse reduced Schur
        |
        v
same-iterate fallback + refinement
        |
        v
original-coordinate certificate
        |
        v
Result / MOI status
```

LP, SOC, RSOC, PSD, exponential, and power models are not separate solver
engines. They are different block compositions in the same canonical program.
KKT routes and linear-algebra providers may change implementation details, but
they may not change the Newton equations or certificate policy.

## Frontends

The typed `Model` API is the direct public frontend. `variable!`, `constraint!`,
and `objective!` record affine expressions and cone domains; `optimize!`
compiles and solves the completed model.

The MathOptInterface optimizer and JSON CLI are adapters over the same
canonicalization and solve path. They do not own alternative numerical
engines. The public legacy-engine selector is rejected with a migration error.

## Canonical program

Every accepted model lowers to

```text
minimize    c'x
subject to  A*x + s = b
            s in K
```

`K` is an ordered product of native blocks. Free variables stay free. Zero
blocks represent equalities. Nonpositive rows and rotated Lorentz rows use
exact typed transforms. Every transform records enough information to recover
primal points, dual points, slacks, and rays in the original model coordinates.

The detailed contract is the repository's [canonical-form design](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/design/CANONICAL_FORM.md).

## Product-cone HSD state

The production iterate is `(x,s,y,tau,kappa)`. One state machine owns:

- cold start;
- cone scaling and local metrics;
- affine predictor and Mehrotra corrector;
- neighborhood/progress line search;
- recovery and typed exhaustion;
- terminal candidate construction; and
- original-coordinate certificate verification.

The HSD sign and complementarity conventions are frozen in the
[HSD formulation](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/design/HSD_FORMULATION.md). Raw `tau`, `kappa`, iteration
limits, or factorization success never establish a terminal mathematical
status.

## Newton system

All KKT implementations consume the same five-equation `NewtonSystem`. The
right-hand side contains primal affine, dual affine, homogeneous-gap, cone
complementarity, and scalar `tau*kappa` equations.

Available implementation routes are:

- `:bordered`: conservative default;
- `:expanded`: exact nonsymmetric expanded operator;
- `:sparse_schur`: reduced sparse Schur route with frozen pattern ownership.

The exact expanded HSD operator is nonsymmetric. A symmetric quasidefinite
companion may provide inertia evidence only when its assumptions are verified;
it does not replace the exact solve or exact-operator residual check.

Same-iterate fallback is explicit and recorded. A route failure cannot silently
change arithmetic, model coordinates, or certificate requirements. See
the [Newton-system design](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/design/NEWTON_SYSTEM.md).

## Cone runtime

Each block supplies dimension-preserving cone operations and local scaling:

- nonnegative orthants;
- Lorentz cones;
- packed PSD cones;
- exponential cones;
- power cones.

RSOC-to-SOC and Nonpositive-to-Nonnegative are exact canonical transforms.
PSD panel kernels, fixed-size 3x3 kernels, and fixed-trace/Q3 reductions are
structural specializations inside the shared runtime; they are not separate
solvers.

## Presolve and reconstruction

Presolve is conservative and typed. Equality reduction, exact fixed-variable
handling, row normalization, optional equilibration, and sparse setup each
produce reversible metadata. If a reduction cannot justify its rank or
consistency decision in the active arithmetic, it keeps the original system or
fails closed.

Every candidate certificate is reconstructed through the complete inverse
stack before terminal status promotion.

## Linear algebra and providers

SDPX owns equations, assembly, route planning, refinement, fallback, and
certification. Providers own factor/solve operations:

- Julia/SuiteSparse for supported Float64 dense and sparse routes;
- MultiFloatLinearAlgebra for fixed-width extended dense/local solves;
- BigFloatLinearAlgebra for BigFloat dense/local solves;
- PureKLU for exact nonsymmetric high-precision sparse LU;
- QDLDL for applicable symmetric-companion inertia evidence.

A `FactorReceipt` binds factor generation, operator generation, route,
provider, and factor outcome. It does not certify a mathematical solution.
Every right-hand side still requires finite checks and residual verification.

See [Linear-algebra providers](providers.md) and the
[high-precision sparse provider decision](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md).

## Arithmetic ownership

Float64, MultiFloat, and BigFloat follow the same mathematical path. Conversion
to Float64 is never an implicit fallback. BigFloat workspaces own independent
mutable MPFR values, and fixed-width provider state remains provider-owned.

Threading is governed by a deterministic thread budget: Julia outer workers,
BLAS threads, and provider threads may not all own the same parallel region.

## Certification boundary

A terminal result is authoritative only after independent verification in the
original coordinates:

- optimality: primal/dual feasibility, cone membership, objective gap, and
  complementarity;
- primal infeasibility: a normalized original-coordinate dual ray;
- dual infeasibility: a normalized original-coordinate primal recession ray.

NaN, infinity, overflow-hidden residuals, invalid tolerances, stale factors, or
unavailable certificate data fail closed.

## Deliberate defaults

- `:bordered` remains the default KKT route until a complete benchmark and
  certificate matrix justifies promotion of another route.
- Ruiz equilibration remains opt-in while physical probes show regressions.
- Sparse and high-precision routes require capability and memory preflight.
- No hidden PSD lift or legacy-engine retry is permitted.

Current implementation work and release gates are tracked in
[`docs/PLAN.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/PLAN.md). Remaining legacy source dependencies are tracked in the
[legacy reference manifest](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/LEGACY_ENGINE_REFERENCES.md).
