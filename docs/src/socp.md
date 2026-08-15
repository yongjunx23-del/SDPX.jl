# SOCP and fixed-trace PSD blocks

SDPX accepts compact standard second-order cone constraints without requiring
users to construct PSD arrow matrices:

```julia
using LinearAlgebra, SDPX

# min t, subject to (t, x, y) in Q3 and x=3, y=4
problem = second_order_program(
    [1.0, 0.0, 0.0],
    Matrix{Float64}(I, 3, 3),
    zeros(3);
    Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
    beq=[3.0, 4.0],
)
result = solve_socp(problem; verbosity=0)
```

Each `SOCConstraint(A, b)` means `A*x+b in Q`, with the scalar/head coordinate
first. Multiple cones can be passed to `second_order_program(c, cones)`.

JuMP/MOI rotated cones are supported through the exact isomorphism

```text
(u, v, w) in Qr  <=>  (u+v, u-v, sqrt(2)w) in Q.
```

The bridge builds sparse rows directly, maps duals by the transpose of this
linear transformation, and maps primal values back through its inverse. The
native `SOCConstraint` API itself remains a standard-Lorentz API; it does not
introduce a second rotated-cone solver.

## Native Lorentz execution

`solve_socp` and pure-SOC MOI models run directly in Lorentz coordinates:

```text
Frontend / MOI
  -> ConicProblem
  -> ConeRepresentationPlan
       -> GeneralLorentzExecution
       -> FixedTraceQ3Execution (verified specialization)
  -> FormulationPlan
       -> DenseNormalEquations
       -> DenseAugmentedKKT
  -> LABackendConfiguration
  -> NativeSOCWorkspace
  -> original-coordinate certificate
```

Production native SOC never constructs PSD matrices. The historical SOC-to-PSD
transform lives only in `test/helpers/soc_psd_reference.jl` for correctness
and benchmark comparisons. The compact API always selects a native execution
and exposes no representation switch. For the general `solve`/`SolverOptions`
path, the automatic planner may select exact PSD-lift reference formulations
(`:socp_psd2`, `:socp_psd_lift`) for SOC-shaped input; those are explicit
exact reference routes, not hidden fallbacks, and the executed algorithm is
recorded in diagnostics.

For `x = (t,u)`, SDPX uses the Euclidean Lorentz pairing, determinant
`t^2 - dot(u,u)`, margin `t - norm(u)`, and Jordan product

```text
(t,u) o (s,v) = (t*s + dot(u,v), t*v + s*u).
```

Nesterov--Todd scaling, `W`, `W^-1`, and `(W'W)^-1` operate directly on
vectors. Determinants and boundary roots are scale-normalized, with a stable
quadratic `q` formula. Actual trial points are rechecked for strict interior.
Scalar nonnegative blocks contribute barrier degree one; proper Lorentz blocks
contribute degree two.

Dense normal equations are the default. The primal metric is factored once per
iteration and reused for predictor/corrector right-hand sides. Dependent
equalities use rank-revealing QR only when `equality_solver=:auto` and the
frozen LA plan authorizes it; explicit normal equations fail closed.

`formulation=:augmented` assembles the symmetric-indefinite system

```text
[ H  -Aeq' ] [dx] = [r ]
[-Aeq   0   ] [dy]   [-p]
```

and requires pivoted symmetric LDLT, multi-RHS solve, and inertia metadata.
Standard LA rejects this route. MFLA and BFLA execute it through provider-owned
factor handles without a generic retry.

## Q3 equivalence and fixed-trace reduction

For a real symmetric matrix

```math
X=\begin{bmatrix}a&b\\b&c\end{bmatrix},\qquad a+c=\tau,
```

the PSD constraint is exactly equivalent to

```math
(\tau,\ a-c,\ 2b)\in\mathcal Q_3.
```

Thus the common bootstrap block

```math
\begin{bmatrix}q&r\\r&2-q\end{bmatrix}\succeq0
```

is the unit disk `(q-1)^2+r^2 <= 1`. The `Q3 <-> S_+^2` isomorphism is
retained for derivation and tests only; production Q3 stays in Lorentz
coordinates.

`SDPX.Experimental.analyze_fixed_trace(problem)` detects direct constant traces
and, when the estimated relation-analysis work is conservative, traces implied
by `B'x=b`. Relations are accepted only after an original-arithmetic residual
check. Expensive large-block equality scans are recorded as skipped because no
automatic larger-block basis reduction is promoted yet. Negative trace is
reported as infeasible; zero trace implies a zero PSD block; positive `2×2`
blocks are marked as SOC candidates; larger direct fixed-trace blocks are
marked for a future traceless-basis reduction.

For sparse `2×2` models, solve setup records whether every coefficient is
traceless. Dominant Schur and reduced-arrow contractions then use
`A11*(M11-M22)+2*A12*M12` and avoid the redundant second diagonal multiply
without changing the persistent model or checkpoint layout.

## FixedTraceQ3 specialization

FixedTraceQ3 is not a second solver. It is a payload inside `NativeSOCPlan`
and shares residuals, NT scaling, predictor/corrector policy, line search,
result type, diagnostics, providers, and certificate with GeneralLorentz.

Promotion is strict: every block must have dimension three, a positive
constant head, zero head coefficients, exactly two nonsingular local tail
variables, no shared variables, and complete variable coverage. Under
`specialization=:auto`, the compact execution is selected whenever the exact
reduction verifies; otherwise `:auto` uses GeneralLorentz. A forced
`specialization=:fixed_trace` throws when the reduction does not verify.
Unsupported provider/formulation pairs fail during planning.

For eligible blocks SDPX stores three local metric entries and a local 2-by-2
Cholesky factor. It eliminates each block locally and forms only the global
equality panel/Gram when equalities exist. Diagnostics report
`executed_factorization=:native_local_cholesky` and
`executed_backend=:fixed_trace_local_elimination`.

Fixed trace removes the primal identity coordinate, but it does **not** reduce
the two transformed equality rows of a generic `(q,r)` cell to one. The native
backend therefore retains the validated block-diagonal equality elimination
and forms the same triangular equality Gram from a `2L x n` transformed panel.
On large equality-heavy models the equality Gram can dominate after cone-local
matrix work has been removed.

For Float64x4 and BigFloat, the two local Cholesky pivot reciprocals are
computed once per cone and reused across every equality column plus both
predictor/corrector triangular solves. Float64x4 CSDR layouts with exact
adjacent two-row ownership additionally fuse the panel copy and local
transform, so each worker reads immutable source rows and writes a disjoint
destination range. Broader layouts and BigFloat keep the copy-then-transform
path. Executed diagnostics expose `local_pivot_kernel` and
`equality_panel_transform`.

## Q3 direction and controller

The default method is a compact Mehrotra predictor-corrector with an
HKM-equivalent local direction. An ownership-safe Lorentz Nesterov--Todd
direction is also implemented internally for Q3, but it is not exposed as a
solve option; HKM remains the validated execution path after matched
certificate and performance gates.

The compact backend currently owns its iteration controller: it selects
`sigma` from the affine complementarity ratio and uses a 0.99 exact
fraction-to-boundary safety. The general SDP `AdaptiveIPMController` options
(`beta`, `gamma`, `parameter_strategy`, and `adaptive_sigma_max`) do not yet
change this Q3 trajectory. For `parameter_policy=:auto`, NativeSOC first solves
its identity-metric affine KKT system, then shifts each Lorentz head to strict
interior, raises aggregate head mass to the Lorentz product identity mass
`<e,e>` only when a side remains within the typed cone-vertex envelope, and
cross-centers complementarity. Balanced nonvertex affine points are not
renormalized. Barrier degree
continues to normalize complementarity and does not change the identity-mass
floor. `OmegaP` and `OmegaD`
therefore do not participate in the automatic Q3 start. Explicit fixed policy
retains the historical head scales exactly. Executed initialization diagnostics
report the affine residuals, shifts, margins, complementarity, factor count,
and two RHS solves.

## Precision, results, and certification

NativeSOC workspaces use `alloc_zeros`; mutable BigFloat slots receive owned
copies. Model coefficients are never installed by reference into scratch
storage. MFLA/BFLA factor handles retain provider state and configured
precision.

`ConicResult` stores primal slacks, cone duals, and equality multipliers in the
coordinates of its native `ConicProblem`. For MOI rotated cones those native
coordinates are the exact canonical Lorentz image; MOI getters return the
original rotated coordinates. Timings, termination counters, and diagnostics
come directly from the NativeSOC run; no PSD `SDPResult` is stored.
Test/reference PSD helpers keep their own `SDPResult` apart from the
`ConicResult`. Certification independently recomputes affine/equality
residuals, stationarity, primal and dual Lorentz margins, objectives, gap,
and complementarity. Pure-SOC MOI input stays native; mixed PSD+SOC input
fails clearly instead of silently lifting.

Lightweight development evidence for the native routes is summarized in
[benchmarks.md](benchmarks.md), with full reports under
[`bench/soc_fixed_trace/README.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/bench/soc_fixed_trace/README.md).

### Native-SOC implementation references

The explicit Q3 NT implementation follows published, open implementations
rather than proprietary behavior. Clarabel's native SOC implementation stores
the Nesterov--Todd point and applies `W'W` as a rank-one Lorentz update; for
Q3 its dense packed scaling block has only six entries. Clarabel and ECOS use
a rank-two KKT expansion for larger SOCs, but adding two extension variables
per Q3 cell is not automatically advantageous for fixed-trace CSDR.

- [Clarabel.rs SOC operations](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/cones/socone.rs)
- [Clarabel.rs KKT numeric maps](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/kktsolvers/direct/quasidef/datamaps.rs)
- [Clarabel.jl SOC operations](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/src/cones/coneops_socone.jl)
- [ECOS cone scaling](https://github.com/embotech/ecos/blob/develop/src/cone.c)

## Reusing a model

For a small number of objective directions, reuse the ingested constraints and
optionally the previous solution:

```julia
session = prepare(problem, options)
first = solve!(session; objective=c1, warm_start=nothing)
second = solve!(session; objective=c2, warm_start=:previous)
```

`PreparedSolver` is sequential and non-reentrant. It caches only immutable
preprocessing structure and an `ExecutionPlan` when planning is invariant to
the permitted objective/RHS updates; every solve still creates fresh numeric
workspace and factor state. RHS-sensitive automatic LP planning is deliberately
recomputed. Use one session per concurrent worker.
