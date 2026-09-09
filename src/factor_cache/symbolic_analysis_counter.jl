#=====================================================================#
#    R2-A: process-local count of REAL provider symbolic analyses.
#
#    This counter is incremented ONLY at a true sparse symbolic-analysis
#    call reached by a solve path — never at a cache lookup, pattern
#    build, numeric refactor, or same-epoch reuse:
#
#      * `:cholmod` — `SparseSymbolicNumericCache.factorize!` first-factor
#        branch (`ldlt(Symmetric(K, :L))`; see
#        `src/factor_cache/routes/sparse_symbolic_numeric.jl`).  This is
#        the Float64 symmetric-augmented-core provider reached by the
#        public prepared/solve path (`kkt_route=:bordered`).  Later
#        `factorize!` calls on the same pattern reuse the retained CHOLMOD
#        factor object through `ldlt!` and do NOT increment.
#      * `:qdldl` — `SparseQDLDLCache{T}` construction (the provider-owned
#        QDLDL object is built there; see
#        `src/factor_cache/routes/qdldl_sparse.jl`).  QDLDL is an optional
#        provider for already-quasi-definite operators; it is NOT on the
#        default public Float64 bordered path (the symmetric core stores a
#        structural zero x-diagonal, which violates QDLDL's
#        quasi-definite precondition), but when it IS used its
#        construction is the real symbolic analysis and is counted here.
#
#    Deliberately NOT counted (documented, not provider symbolic work):
#
#      * `SymmetricCorePattern` CSC construction / structure-cache
#        hits-misses (`structure_cache_stats()`): frozen sparsity-pattern
#        assembly, not provider ordering/analysis.
#      * `DisconnectedLDLTCache`: dense per-component LDL with no sparse
#        ordering or symbolic phase.
#      * dense MFLA/BFLA symmetric-core LDL (`build_symmetric_core_ldlt_cache`):
#        dense pivoted LDL with no sparse symbolic phase.
#      * UMFPACK `sparse_schur` route: not reached by the default prepared
#        path (the entrypoint bridge fixes `kkt_route=:bordered`); no
#        counter is wired there yet (recorded gap, not silent reuse).
#
#    The counter is process-local, monotonic, and lock-guarded.  It never
#    affects routing or numerics: every increment site already performed
#    the provider call it records.  Recovery after a failed factor
#    re-runs the analysis (`cache.factor === nothing` again) and counts
#    again, truthfully.
#=====================================================================#

const _SYMBOLIC_ANALYSIS_LOCK = ReentrantLock()
const _SYMBOLIC_ANALYSIS_TOTAL = Ref{Int}(0)
const _SYMBOLIC_ANALYSIS_BY_PROVIDER = Dict{Symbol,Int}()

"""Record one completed real provider symbolic analysis (internal)."""
function _record_symbolic_analysis!(provider::Symbol)
    lock(_SYMBOLIC_ANALYSIS_LOCK) do
        _SYMBOLIC_ANALYSIS_TOTAL[] += 1
        _SYMBOLIC_ANALYSIS_BY_PROVIDER[provider] =
            get(_SYMBOLIC_ANALYSIS_BY_PROVIDER, provider, 0) + 1
    end
    return nothing
end

"""
    symbolic_analysis_count() -> Int

Process-local monotonic total of REAL provider symbolic analyses observed
in this process (see the file header for exactly which calls count).
Read-only: it never triggers, skips, or alters an analysis.
"""
function symbolic_analysis_count()
    return lock(_SYMBOLIC_ANALYSIS_LOCK) do
        _SYMBOLIC_ANALYSIS_TOTAL[]
    end
end

"""
    symbolic_analysis_counts() -> NamedTuple

Read-only per-provider breakdown `(total, cholmod, qdldl)`.  Providers
with no observed analysis report zero.
"""
function symbolic_analysis_counts()
    return lock(_SYMBOLIC_ANALYSIS_LOCK) do
        by = _SYMBOLIC_ANALYSIS_BY_PROVIDER
        (
            total=_SYMBOLIC_ANALYSIS_TOTAL[],
            cholmod=get(by, :cholmod, 0),
            qdldl=get(by, :qdldl, 0),
        )
    end
end

"""
    symbolic_analysis_delta(f::Function) -> NamedTuple

Run `f()`, returning `(result=f(), delta, before, after)` where
`delta = after - before` is the number of real provider symbolic analyses
`f` triggered.  Read-only reporting around an arbitrary solve/update
closure; routing and numerics are untouched.
"""
function symbolic_analysis_delta(f::Function)
    before = symbolic_analysis_count()
    result = f()
    after = symbolic_analysis_count()
    return (result=result, delta=after - before, before=before, after=after)
end
