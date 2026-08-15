# Sparse execution

Sparse execution keeps formulation, storage, arithmetic, and factorization as
separate decisions. `KKTStoragePlan` records `:dense` or `:sparse`; sparse
matrices are owned as frozen CSC (`SparseKKTStorage`) and numeric iterations
update only `nzval`. `SparseSymbolicAnalysis` stores a pattern-only ordering,
elimination tree, and factor row structure, so the same symbolic object can be
reused for Float64, MultiFloat, and BigFloat values.

## Providers

Float64 uses the CHOLMOD provider through Julia's SuiteSparse.
`Float64x2`/`Float64x3`/`Float64x4` and BigFloat use the provider-neutral
simplicial generic sparse Cholesky. Generic factors retain only their CSC
factor pattern and values; they never construct an `n×n` dense factor.
BigFloat operands are checked for uniform precision and factor/workspace
destinations are independently owned MPFR objects.

`analyze_sparse_pattern`, `numeric_factorize!`, `sparse_factor_solve!`, and
`sparse_factor_diagnostics` form the provider-neutral API. A changed CSC
pattern fails closed, while unchanged patterns support numeric
refactorization without repeating symbolic ordering. Sparse augmented/
indefinite LDL is not part of the generic route; explicit unsupported requests
report an error instead of silently falling back to dense.

The current generic ordering is a deterministic approximate minimum-degree
ordering for small patterns (natural order for very large setup graphs). The
CHOLMOD numeric provider retains SuiteSparse's AMD ordering. Fill diagnostics
include dimension, input/factor nonzeros, fill ratio, ordering, provider,
refactorization count, and (where available) minimum factor diagonal.

## Dedicated LP sparse path

The dedicated LP path forms `H = G' * Diagonal(z ./ s) * G` and may select a
sparse Newton system when `G` really is sparse. `select_lp_formulation` uses
`:auto`, `:dense`, or `:sparse`; the crossover is a property of the fill-in of
`G'*D*G` rather than of `G`, so a sparse factorization is used only when it is
predicted to pay for itself:

- explicit `:dense` always selects `:dense_lu`;
- generic extended arithmetic remains explicit-first: package availability
  must not silently change an `:auto` execution plan;
- `storage=:sparse` with equalities raises because sparse execution supports
  only SPD normal equations (the sparse augmented LDL route is retired);
- otherwise `:auto` selects `:sparse_normal` only at or above the dimension
  and density crossover.

The equality-free sparse Newton system is positive definite, so SDPX freezes
its CSC pattern, computes a fill-reducing ordering and elimination tree once,
and refactors only numeric values in later iterations. The final LP
`termination.sparse_schur_backend` payload reports actual input/factor counts
and reuse counters; factorization elapsed time remains the separate
`kkt_factorization` timing.

## SDP sparse Schur

For eligible SDP models, the provider-neutral sparse-Schur path is selected
before workspace allocation. `ExecutionPlan.storage_plan` is created before
the post-presolve pattern exists, so `storage_input_nnz=0` and density fields
are explicit "estimate unavailable" markers, not measured matrix counts.

## Scope

Sparse code remains inside SDPX. A future extraction boundary would be the
pattern/order/assembly-map and provider-neutral factor/solve API; formulation
planning, LP/SDP assembly policy, scaling, refinement, and certification
should remain SDPX-owned.
