# Architecture

SDPX is organized as a canonical cone frontend, a deterministic midend, and a
typed numerical backend behind one `ExecutionPlan`.

```text
                 Frontend
   Model builder / JuMP-MOI / CLI / file loaders
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

The public boundary is one typed `Model` builder. `variable!`, `constraint!`,
and `objective!` record affine expressions and cone domains; `optimize!`
compiles that model once, classifies its native cone family, and dispatches to
the corresponding typed lowerer. The public domains are `Reals`, the scalar
orthants and `ZeroCone` (LP), `LorentzCone`/`RotatedLorentzCone` (SOC), and
`PSDCone` (SDP). `Settings` and `Outputs` carry the typed solve policy and
retention policy; result accessors read the single typed result boundary.

JuMP and MathOptInterface remain adapters over the same boundary: MOI caches a
completed model and `copy_to` finalizes cone incidence before the SDPX
optimizer lowers it. The CLI parses JSON through the mature qualified loader
into the same numerical pipeline; it is not a separate solver route.

Route classification is fail-closed. A model containing more than one nonfree
family (for example LP + SOC, LP + SDP, or SOC + SDP) raises a typed error
before lowering. There is no production conversion from a Lorentz block to a
PSD block; SOC models stay in native Lorentz coordinates.

## Midend: presolve, scale, formulate, plan

Every public `optimize!` runs a conservative preparation pipeline:

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

Cone family and numerical formulation are separate decisions. Native SOC and
the arrow-specialized Q3 implementation share one Lorentz semantic IR; the
specialization is selected only inside the native SOC route. A PSD model stays
on the SDP route even when its blocks have a structure that admits a Lorentz
interpretation. Derivation and test fixtures may compare the two mathematical
cones, but execution never silently changes family.

The automatic planner uses structure facts first, then checks backend
feasibility. Providers and arithmetic names never choose the mathematical
formulation; execution uses the provider/formulation pair frozen in the plan.

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

Explicit requests are never overridden. Public `Settings.formulation` accepts
`:auto`, `:variable_space_schur`, and `:dense_augmented_kkt`; a request that is
incompatible with the classified route fails during planning. The first two
names lower to the mature normal-equation and augmented-KKT implementations.
Qualified low-level option records retain additional compatibility spellings,
but those records are internal and are not a second public modeling API.

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

`SDPX.plan_formulation` is a qualified internal dense-KKT formulation step used
by `build_execution_plan`; it returns a `FormulationDecision` with
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

The `certificate(result)` accessor exposes a post-solve check that recomputes
objectives, affine residuals, componentwise backward errors, complementarity,
and PSD margins in the original coordinates. A solver status is not
authoritative when this independent check fails. The direct primal-dual
iteration is not a full homogeneous self-dual embedding, so infeasibility
certificates are produced only when a validated homogeneous ray is found. See
[diagnostics.md](diagnostics.md).

## Current limitations

- The public API is experimental and may change before 1.0.
- LP, native SOC/RSOC, and SDP are implemented as separate typed routes.
- The direct primal-dual iteration does not carry HSD `τ`/`κ` variables, so it
  may fail to produce a ray for some infeasible models.
- Equality-constrained LPs use dense LU; null-space/range-space selection is
  future work.
- Sparse augmented/indefinite LDL is not implemented.
- General BigFloat work is serial; ownership-safe independent blocks and exact
  local arrow phases may use workers.
- Nested solves in one process are not supported because BLAS thread count is
  process-global.

Operational details and measured trade-offs live in
[providers.md](providers.md), [benchmarks.md](benchmarks.md),
[preprocessing.md](preprocessing.md), and the root research notes
[`docs/adaptive-dense-sparse-optimization.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-dense-sparse-optimization.md)
and [`docs/adaptive-parameter-policy.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-parameter-policy.md).
