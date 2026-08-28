#=====================================================================#
#    Route-specific FactorCache implementations (Subagent E).
#
#    The concrete Float64 route caches live in `routes/` and share the
#    provider-neutral protocol (prepare! / factorize! / solve! / solve_multi!
#    / refine_once! / invalidate! / factor_status / factor_*_epoch /
#    factor_diagnostics) and the state machine from state.jl.  Each cache:
#      * commits ALL capacity in `prepare!` (the single allocation point);
#      * reuses owned storage across epochs — a new `matrix_epoch`
#        re-factorizes reusing storage, the same epoch skips the numeric
#        factor entirely;
#      * stamps a real `factor_epoch` (and a monotone actual-factorization
#        counter) at each numeric factor call;
#      * is fail-closed (any exception → `Failed`, solve rejected);
#      * uses owned factor solves where the backend exposes a public in-place
#        solve; the Float64 CHOLMOD `SparseSymbolicNumericCache` explicitly
#        uses the public allocating `factor \ rhs` (see that file), so the
#        allocation-free claim is cache-specific and never asserted there.
#====================================================================#
include("routes/common.jl")
include("routes/dense_schur_cholesky.jl")
include("routes/dense_augmented_ldlt.jl")
include("routes/lp_lu.jl")
include("routes/equality_rrqr.jl")
include("routes/arrow_local.jl")
include("routes/arrow_reduced.jl")
include("routes/sparse_symbolic_numeric.jl")
