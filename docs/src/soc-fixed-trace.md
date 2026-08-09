# SOC and fixed-trace PSD blocks

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

The compact frontend uses an exact PSD compilation as its correctness
reference. Dimension-three Lorentz cones use the exact `Q3 <-> S_+^2`
isomorphism directly, while other dimensions retain the general arrow lift.
The compact API and original-coordinate `ConicResult` remain stable boundaries
for a future general-dimensional Lorentz Newton backend.

## Fixed trace

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

is the unit disk `(q-1)^2+r^2 <= 1`.

`SDPX.Experimental.analyze_fixed_trace(problem)` detects direct constant traces
and, when the estimated relation-analysis work is conservative, traces implied
by `B'x=b`. Relations are accepted only after an original-arithmetic residual
check. Expensive large-block equality scans are recorded as skipped because no
automatic larger-block basis reduction is promoted yet. Negative trace is
reported as infeasible; zero trace implies a zero PSD block; positive `2x2`
blocks are marked as SOC candidates; larger direct fixed-trace blocks are
marked for a future traceless-basis reduction.

For sparse `2x2` models, solve setup records in the transient workspace whether
every coefficient is traceless. Dominant Schur and reduced-arrow contractions
then use `A11*(M11-M22)+2*A12*M12` and avoid the redundant second diagonal
multiply without changing the persistent model or checkpoint layout.

## Native fixed-trace Q3 backend

`algorithm=:socp` selects a compact Q3 Newton backend only for an exactly
certified narrow structure: every block is real symmetric `2x2`, has a direct
positive fixed trace, owns exactly two local variables, has traceless
coefficients, and has a nonsingular two-coordinate tail map. The hot iterate
stores `(t,u,v)` cone coordinates and two directions per block; it does not
construct PSD matrices until the compatibility result and original-coordinate
certificate boundary. Local Schur metrics, predictor/corrector residuals, and
fraction-to-boundary steps use closed-form Q3 algebra. The default method is a
compact Mehrotra predictor-corrector with an HKM-equivalent local direction.
An ownership-safe Lorentz Nesterov--Todd direction is also implemented for Q3
and can be selected explicitly with `q3_direction=:nt`; it is a research path,
not an automatic selector target. In the same-node J40 Float64x4 ABBA gate it
needed 225 rather than 191 iterations and was 16.8% slower than HKM despite
passing the same original-coordinate certificate. HKM therefore remains the
validated default. A general-dimensional native Lorentz NT backend remains
future work.

The compact backend currently owns its iteration controller: it selects
`sigma` from the affine complementarity ratio and uses a 0.99 exact
fraction-to-boundary safety. The general SDP `AdaptiveIPMController` options
(`beta`, `gamma`, `parameter_strategy`, and `adaptive_sigma_max`) do not yet
change this Q3 trajectory. The primal cone head starts at its exact fixed-trace
value, so `OmegaP` is intentionally inactive for Q3; automatic `OmegaD` still
sets the initial dual head. Executed parameter history reports the actual Q3
values. This scope is stated explicitly so an expert override is never
mistaken for an active control.

Fixed trace removes the primal identity coordinate, but it does **not** reduce
the two transformed equality rows of a generic `(q,r)` cell to one. The native
backend therefore retains the validated block-diagonal equality elimination
and forms the same triangular equality Gram from a `2L x n` transformed panel.
This is important for both correctness and performance interpretation: on
large J80-like models the equality Gram can dominate after cone-local matrix
work has been removed.

For Float64x4 and BigFloat, the two local Cholesky pivot reciprocals are
computed once per cone and reused across every equality column plus both
predictor/corrector triangular solves. This replaces repeated high-precision
division with multiplication while leaving the Float64 path unchanged.
Float64x4 CSDR layouts with exact adjacent two-row ownership additionally fuse
the panel copy and local transform, so each worker reads immutable source rows
and writes a disjoint destination range. Broader layouts and BigFloat keep the
copy-then-transform path. Executed diagnostics expose
`local_pivot_kernel` and `equality_panel_transform`; cluster promotion still
requires the complete-solve certificate and performance gates.

The expert option `q3_gram_strategy=:output_tiles|:row_bins|:auto` controls the
extended-precision Gram scheduler. Output tiles own disjoint result triangles
without replicated storage. Row bins own contiguous panel rows and private
packed lower triangles followed by a deterministic tree reduction. `:auto`
uses output tiles for fixed-width arithmetic after the general crossover and
for BigFloat only when at least two workers, 32 equality columns, and 250,000
triangular contractions are available. Smaller BigFloat panels keep the
serial pairwise kernel. Row-bin replication remains an experimental expert
override until complete cluster solves show that it pays for itself. BigFloat
tasks mutate only disjoint MPFR objects and use private preallocated scalar
scratch. Executed diagnostics report `gram_strategy`, `gram_threads`, and the
selection reason rather than merely the requested worker count.

Unsupported models, failed native factorization, or a failed independent
certificate fall back to the exact PSD2 reference when time remains. After
the controlled J40 gate, `algorithm=:auto` selects native Q3 only for an exact
sparse fixed-trace model with at least 4,096 blocks, 8,192 variables, 128
equalities, fixed-width arithmetic at least as wide as `Float64x4`, and no
explicit Ruiz equilibration. The gate measured a 1.97x eight-worker solver
speedup, sub-one-percent CV, a 9.6% one-worker improvement, equivalent
certificates, and lower same-allocation memory. Float64, BigFloat, smaller
models, and all unsupported structures keep the PSD2 reference. Request
`algorithm=:socp, scaling=:none` to override the performance policy, and
inspect `result.termination.executed.kkt` to confirm the executed backend.

In the final same-node 32-thread/BLAS-1 J40 comparison, six timed rows per
formulation gave median solve times of 135.10 seconds for the optimized PSD2
path and 25.71 seconds for native HKM-Q3 (5.26x). End-to-end speedup was 5.14x.
All rows were `Optimal` and certificate-valid; primal objectives agreed to
about `2.2e-13` relative. PSD2 averaged 7.83 active cores while Q3 averaged
28.73. Q3 still spent most of its core time in the equality Gram, so the next
structural target is reducing or accelerating that Gram—not further tuning of
the 170-by-170 equality Cholesky.

The result is arithmetic-dependent. In the matched J40 BigFloat256 gate at 32
threads, optimized PSD2 solved in a 184.99-second median while HKM-Q3 needed
197.07 seconds (6.5% slower) because it took 191 rather than 166 iterations.
Both three-run CVs were below 1.4%, objectives agreed to `4.3e-13` relative,
and every original-coordinate certificate passed, but Q3 also used 6.9% more
process RSS and 2.66x more allocation per solve. BigFloat therefore remains on
PSD2 under `algorithm=:auto`; Q3 is still available explicitly for controlled
comparisons and future kernel work.

### Native-SOC implementation references

The explicit Q3 NT implementation follows published, open implementations
rather than proprietary behavior. Clarabel's native SOC implementation stores
the Nesterov--Todd point and applies `W'W` as a rank-one Lorentz update; for Q3 its
dense packed scaling block has only six entries. Clarabel and ECOS use a
rank-two KKT expansion for larger SOCs, but adding two extension variables per
Q3 cell is not automatically advantageous for fixed-trace CSDR. SDPX will keep
the compact Q3 representation unless a matched fill and factorization
benchmark proves otherwise.

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

`PreparedSolver` is sequential and non-reentrant. Use one session per
concurrent worker.
