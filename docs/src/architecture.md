# Architecture

SDPX is organized as a canonical cone frontend, a deterministic midend, and a
typed numerical backend behind one `ExecutionPlan`.

```text
                 Frontend
 Julia arrays / JuMP-MOI / Convex / CLI / file loaders
                       |
                       v
             Canonical cone semantics
          Linear | Lorentz | PositiveSemidefinite
                       |
                       v
                  Midend
 analyze -> presolve -> scale -> formulate -> plan
                       |
                  ExecutionPlan
                       |
            +----------+----------+
            |          |          |
            v          v          v
           LP         SOCP        SDP
          kernel     kernel      kernel
            \          |          /
             +---------+---------+
                       v
             system assembly / KKT
                       v
        factorization -> refinement
                       v
          original-coordinate certificate
```

## Frontend boundaries

The native APIs (`ingest`, `linear_program`, `second_order_program`) validate
and canonicalize typed data into `SDPProblem{T}` or `ConicProblem{T}`. JuMP
and MathOptInterface are adapters over the same core: MOI caches a completed
model and `copy_to` finalizes PSD incidence data, then SDPX converts it to the
native typed representation. Convex.jl uses the same MOI optimizer after DCP
canonicalization. The CLI parses JSON into the same typed model boundary and
is not a separate numerical route.

The compact SOC frontend crosses the public boundary into `ConicProblem`
without any representation transform; production SOC solves stay in Lorentz
coordinates. The historical SOC-to-PSD transform is a test-only reference.

## Midend: presolve, scale, formulate, plan

Every public solve runs a conservative preparation pipeline:

1. classify cone, storage, arithmetic, size, and predicted Schur density;
2. rank-revealing equality presolve in the problem arithmetic;
3. merge scalar bounds and eliminate exactly fixed variables;
4. select scaling or Ruiz equilibration;
5. estimate formulation, kernel, factorization, memory, and scheduling costs;
6. build an immutable `ExecutionPlan` that freezes the mathematical
   formulation, storage plan, LA backend, provider, and thread schedule;
7. solve; and
8. reconstruct and certify the result in the original coordinates.

`ExecutionPlan` is authoritative. Deterministic choices made after plan
construction are architecture bugs. Runtime numerical fallback is allowed only
where the plan authorized it, and it must be recorded as planned-versus-
executed backend, formulation, and fallback reason. The original-coordinate
certificate remains the final status authority.

## Cone formulation ownership

SOC-to-PSD is a **formulation transform**, not a frontend fact. Native SOC and
the arrow specialized Q3 implementation share the same Lorentz semantic IR.
The planner may choose an exact PSD-arrow reference route when necessary, but
diagnostics and benchmarks must label that formulation explicitly.

The automatic planner uses structure facts first, then checks backend
feasibility. Providers and arithmetic names never choose the mathematical
formulation. Unsupported provider/formulation pairs fail closed during
planning; no silent route substitution is performed at execution time.

## KKT formulations

For the general dense SDP path, `formulation=:auto` makes a real pre-execution
choice between two implemented mathematical formulations:

- `DenseNormalEquations`, followed by Cholesky;
- `DenseAugmentedKKT`, followed by pivoted symmetric LDLT.

The planner is deliberately small, conservative, and deterministic. It does
not add a third formulation and never changes formulation after numerical
execution begins. The first policy uses only two numerical-risk indicators:

1. the scale spread of retained equality rows; and
2. the relative diagonal quality from the normalized RRQR that equality
   presolve already performs for correctness.

It does not compute a condition number, SVD, eigendecomposition, or a second
RRQR. If a verified retained equality basis is unavailable, auto planning
conservatively keeps dense normal equations. Augmented KKT is selected only
when strong normal-equation risk is present and the candidate passes
structural, backend, and memory feasibility.

Explicit requests are never overridden. `formulation=:normal_equations` fixes
dense normal equations and fails closed for dedicated LP. `formulation=:augmented`
fixes dense augmented KKT or fails during planning. `formulation=:primal`
preserves the historical primal orientation without disabling sparse or
block-arrow routes. The dual KKT form remains an internal cost estimate only;
production formulation values exclude `:dual`.

The dense augmented KKT system is the equality-augmented Schur system

```text
K = [ S   -B ]       rhs = [ r ]
    [-B'   0 ]             [-p]
```

with unknowns `u = [dx; dy]`, not the much larger full cone KKT matrix
containing explicit `dX_l` and `dY_l` unknowns. Solving it produces the same
`dx` and `dy` as the dense normal-equation elimination; block recovery and
certification are formulation-independent. Only the lower triangle of `K` is
authoritative. On LDLT failure the augmented route regularizes only the
primal `S` block; the equality block remains exactly zero, residual and
refinement always evaluate the original unregularized `K`, and failure never
switches formulation, provider, or precision.

`SDPX.Experimental.plan_formulation` is the dense-KKT formulation step used by
`build_execution_plan`; it returns a `FormulationDecision` with
requested/preferred/selected formulation, reasons, required capabilities,
feasibility, equality scale/RRQR evidence, and memory estimates. The execution
plan and result diagnostics expose the summary through
`parameters.formulation_decision` and
`selected_algorithms.formulation_decision`.

## Dedicated routes

The dedicated LP path owns scalar-cone preprocessing, its normal matrix and
Cholesky/LU choices, and the sparse normal-equation backend. NativeSOC owns
Lorentz algebra, the Q3 reduction, its equality Gram, and original-Lorentz
certification. Block-arrow elimination, sparse Schur assembly, fixed-trace Q3,
native SOC kernels, reduced LP systems, mixed-precision KKT logic, and
structured refinement are solver algorithms, not duplicate provider linear
algebra. They stay SDPX-owned even when a provider supplies the underlying
factor operations.

## Validation boundary

`result_certificate` recomputes objectives, affine residuals, componentwise
backward errors, complementarity, and PSD margins in the original coordinates.
A solver status is not authoritative when this independent check fails. The
direct primal-dual iteration is not a full homogeneous self-dual embedding,
so infeasibility certificates are produced only when a validated homogeneous
ray is found. See [diagnostics.md](diagnostics.md).

## Current limitations

- The public API is experimental and may change before 1.0.
- The direct primal-dual iteration does not carry HSD `τ`/`κ` variables, so it
  may fail to produce a ray for some infeasible models.
- Equality-constrained LPs use dense LU; null-space/range-space selection is
  future work.
- Sparse augmented/indefinite LDL is not implemented; unsupported sparse
  requests fail closed.
- General BigFloat work is serial; ownership-safe independent blocks and exact
  local arrow phases may use workers.
- Nested solves in one process are not supported because BLAS thread count is
  process-global.

Operational details and measured trade-offs live in
[providers.md](providers.md), [benchmarks.md](benchmarks.md),
[preprocessing.md](preprocessing.md), and the root research notes
[`docs/adaptive-dense-sparse-optimization.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-dense-sparse-optimization.md)
and [`docs/adaptive-parameter-policy.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-parameter-policy.md).
