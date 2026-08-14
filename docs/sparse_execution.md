# Sparse execution

Round 6 keeps formulation, storage, arithmetic, and factorization as separate
decisions. `KKTStoragePlan` records `:dense` or `:sparse`; sparse matrices are
owned as frozen CSC (`SparseKKTStorage`) and numeric iterations update only
`nzval`. `SparseSymbolicAnalysis` stores a pattern-only ordering, elimination
tree, and factor row structure, so the same symbolic object can be reused for
Float64, MultiFloat, and BigFloat values.

Float64 uses the CHOLMOD provider.  `Float64x2`/`Float64x3`/`Float64x4` and
BigFloat use the provider-neutral simplicial generic sparse Cholesky.  Generic
factors retain only their CSC factor pattern and values; they never construct
an `n×n` dense factor.  BigFloat operands are checked for uniform precision and
factor/workspace destinations are independently owned MPFR objects.

`analyze_sparse_pattern`, `numeric_factorize!`, `sparse_factor_solve!`, and
`sparse_factor_diagnostics` form the provider-neutral API.  A changed CSC
pattern fails closed, while unchanged patterns support numeric refactorization
without repeating symbolic ordering.  Sparse augmented/indefinite LDL is not
part of the generic Round 6A route; explicit unsupported requests report an
error instead of silently falling back to dense.

The current generic ordering is a deterministic approximate minimum-degree
ordering for small patterns (natural order for very large setup graphs).  The
CHOLMOD numeric provider retains SuiteSparse's AMD ordering.  Fill diagnostics
include dimension, input/factor nonzeros, fill ratio, ordering, provider,
refactorization count, and (where available) minimum factor diagonal.

`ExecutionPlan.storage_plan` is created before the LP's post-presolve sparse
pattern exists. Its `storage_input_nnz=0` and density are therefore explicit
"estimate unavailable" markers, not measured matrix counts. The final LP
`termination.sparse_schur_backend` payload reports actual input/factor counts
and reuse counters; factorization elapsed time remains the separate
`kkt_factorization` timing.

Sparse code remains inside SDPX for this round.  A future extraction boundary
would be the pattern/order/assembly-map and provider-neutral factor/solve API;
formulation planning, LP/SDP assembly policy, scaling, refinement, and
certification should remain SDPX-owned.
