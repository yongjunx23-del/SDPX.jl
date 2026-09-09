# R2-A real symbolic/numeric separation (implementation evidence + tests).
#
# Scope (docs/design/SCIENTIFIC_CORE_ROADMAP.md, R2-A): count symbolic
# analysis at the provider's REAL symbolic-analysis entry; same-structure
# 100 c/b updates must not repeat symbolic analysis.  Existing metadata
# reuse only proves metadata reuse and does NOT satisfy the gate.
#
# What this file proves, truthfully, with the committed counter
# (`src/factor_cache/symbolic_analysis_counter.jl`,
# `SDPX.symbolic_analysis_count()`):
#
#   1. The counter sits exactly at the true provider entries:
#      CHOLMOD `ldlt(Symmetric(K,:L))` first-factor branch
#      (`src/factor_cache/routes/sparse_symbolic_numeric.jl`) and the QDLDL
#      provider construction (`src/factor_cache/routes/qdldl_sparse.jl`).
#      Numeric refills, same-epoch reuse, structure-cache lookups, the
#      disconnected dense path, and dense MFLA/BFLA LDL do NOT increment.
#   2. One prepared problem + 100 sequential numeric-only c/b updates:
#      the test records the truthful observed delta.  The current prepared
#      path builds a FRESH symmetric-core workspace + provider cache per
#      solve, so a sparse-provider solve re-analyzes once per solve; the
#      disconnected dense path analyzes never.  Either way the R2-A reuse
#      gate (delta == 1) does NOT pass today, and this file reports that
#      explicitly instead of faking the counter.
#   3. Structure change invalidates (`:structure_changed`) and a fresh
#      structure analyzes fresh; provider/precision/thread-budget changes
#      are recorded with their truthful current behavior.
#   4. Negative control: a corrupted/stale FactorReceipt does not validate
#      (`factor_receipt_owned` is false); a Failed cache never solves stale
#      data and recovery re-runs the analysis (counted again, truthfully).
#
# No numerical behavior, tolerance, routing, or default-path change is made
# or asserted here: every solve must still certify Optimal with the same
# residual gates as the R2 lifecycle qualification.
#
# Run (bounded, single-threaded):
#
#   JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#   julia --startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G \
#     --project=<isolated-env-with-SDPX-developed> \
#     validation/scientific_core/test_r2a_symbolic_numeric_separation.jl
#
# The driver exports SDPX_EXPECT_ROOT / SDPX_EXPECT_HEAD; this process
# asserts it loaded exactly that committed source tree.

using Test
using Pkg
using SDPX
using LinearAlgebra
using SparseArrays

# --------------------------------------------------------------------------
# In-process provenance + loaded-source assertions.
# --------------------------------------------------------------------------

const _EXPECT_ROOT = get(ENV, "SDPX_EXPECT_ROOT", "")
const _EXPECT_HEAD = get(ENV, "SDPX_EXPECT_HEAD", "")

const _LOADED_ROOT = Base.pkgdir(SDPX)
const _LOADED_HEAD =
    chomp(read(`git -C $(_LOADED_ROOT) rev-parse HEAD`, String))
const _LOADED_CLEAN =
    isempty(chomp(read(`git -C $(_LOADED_ROOT) status --porcelain`, String)))

println("provenance: pkgdir(SDPX)=", _LOADED_ROOT)
println("provenance: HEAD=", _LOADED_HEAD)
println("provenance: clean=", _LOADED_CLEAN)
println("provenance: Threads.nthreads()=", Base.Threads.nthreads())
println("provenance: BLAS threads=", LinearAlgebra.BLAS.get_num_threads())
println("provenance: VERSION=", VERSION)

if !isempty(_EXPECT_ROOT)
    @test realpath(_LOADED_ROOT) == realpath(_EXPECT_ROOT)
end
if !isempty(_EXPECT_HEAD)
    @test _LOADED_HEAD == _EXPECT_HEAD
end
@test _LOADED_CLEAN

# --------------------------------------------------------------------------
# Fixture: the same small Float64 LP as the R2 lifecycle qualification.
# min c'x s.t. 0 <= x <= 1, sum(x) == 1.5.
# --------------------------------------------------------------------------

const _C0 = Float64[1.0, 2.0, 3.0]
const _G0 = Float64[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
const _H0 = Float64[0.0, 0.0, 0.0, -1.0, -1.0, -1.0]
const _AEQ0 = Float64[1.0 1.0 1.0]
const _BEQ0 = Float64[1.5]

_r2a_options() = SDPX.SolverOptions{Float64}(; verbosity=0, timing=false, threads=1)

function _r2a_problem(c=_C0, G=_G0, h=_H0, Aeq=_AEQ0, beq=_BEQ0)
    return SDPX.linear_program(c, G, h; Aeq=Aeq, beq=beq)
end

@testset "R2-A: counter sits at the true provider symbolic entry (direct cache)" begin
    # Frozen 2x2 lower-triangle pattern with a structural diagonal in every
    # column (the CHOLMOD cache admission rule).
    lower = sparse(Float64[2 0; 1 -2])
    mkcache() = begin
        cache = SDPX.SparseSymbolicNumericCache{Float64}()
        SDPX.prepare!(
            cache,
            SDPX.SparseSymbolicRequirements(lower; dsigns=[1, -1]),
        )
        cache
    end

    cache = mkcache()
    base = SDPX.symbolic_analysis_count()
    base_breakdown = SDPX.symbolic_analysis_counts()
    @test base_breakdown.total == base

    # First factorize: the public `ldlt` call performs the sole symbolic
    # analysis for this pattern.  Exactly one global count, provider :cholmod.
    SDPX.factorize!(cache, lower, 1)
    @test SDPX.symbolic_analysis_count() == base + 1
    @test SDPX.symbolic_analysis_counts().cholmod == base_breakdown.cholmod + 1
    @test SDPX.factor_diagnostics(cache).symbolic_count == 1
    @test SDPX.factor_diagnostics(cache).numeric_count == 1

    # Numeric-only refill, new epoch, same pattern: CHOLMOD `ldlt!` reuses
    # the retained symbolic object.  No new analysis anywhere.
    nudged = copy(lower)
    nudged.nzval .*= 1.5
    SDPX.factorize!(cache, nudged, 2)
    @test SDPX.symbolic_analysis_count() == base + 1
    @test SDPX.factor_diagnostics(cache).symbolic_count == 1
    @test SDPX.factor_diagnostics(cache).numeric_count == 2

    # Same-epoch reuse: no provider call at all.
    SDPX.factorize!(cache, nudged, 2)
    @test SDPX.symbolic_analysis_count() == base + 1
    @test SDPX.factor_diagnostics(cache).numeric_count == 2

    # Non-finite input is rejected BEFORE any provider call: no count.
    nonfinite = copy(lower)
    nonfinite.nzval[1] = NaN
    @test_throws ArgumentError SDPX.factorize!(cache, nonfinite, 3)
    @test SDPX.factor_status(cache) === SDPX.Failed
    @test SDPX.symbolic_analysis_count() == base + 1

    # A fresh cache over the same pattern performs its own one analysis
    # (no cross-cache symbolic reuse is claimed or counted as reuse).
    cache2 = mkcache()
    SDPX.factorize!(cache2, lower, 1)
    @test SDPX.symbolic_analysis_count() == base + 2

    # A singular operator against a cache that already holds a factor goes
    # through the NUMERIC `ldlt!` refactor on the retained object (no new
    # symbolic analysis exists to count) and fails closed with the typed
    # zero-pivot error.  The failed object is detached; recovery presents
    # the same pattern again, re-runs the real `ldlt` analysis (counted
    # again, truthfully), and returns to Fresh.
    singular = copy(lower)
    fill!(singular.nzval, 0.0)
    @test_throws ArgumentError SDPX.factorize!(cache2, singular, 2)
    @test SDPX.factor_status(cache2) === SDPX.Failed
    @test SDPX.symbolic_analysis_count() == base + 2
    @test SDPX.factor_diagnostics(cache2).symbolic_count == 1
    # Failed never solves stale data.
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        cache2, zeros(2), Float64[3.0, -1.0],
    )
    SDPX.factorize!(cache2, lower, 2)
    @test SDPX.factor_status(cache2) === SDPX.Fresh
    @test SDPX.symbolic_analysis_count() == base + 3
    @test SDPX.factor_diagnostics(cache2).symbolic_count == 2
    @test SDPX.factor_diagnostics(cache2).numeric_count == 2

    println("R2-A-direct-cache: base=", base,
        " final=", SDPX.symbolic_analysis_count(),
        " breakdown=", SDPX.symbolic_analysis_counts())
end

@testset "R2-A: QDLDL entry is fail-closed without a loaded provider" begin
    # No MFLA/BFLA QDLDL extension is loaded in the isolated env, so the
    # constructor must fail closed BEFORE any analysis — and count nothing.
    @test SDPX.SparseQDLDLProviderAvailable(Float64) == false
    before = SDPX.symbolic_analysis_count()
    upper = sparse(triu(Float64[2 0; 0 -2]))
    @test_throws ArgumentError SDPX.SparseQDLDLCache{Float64}(
        upper, [1, -1],
    )
    @test SDPX.symbolic_analysis_count() == before
    println("R2-A-qdldl: provider_available=false, delta=0 (fail closed, no phantom count)")
end

@testset "R2-A: stale/corrupted receipts never validate (negative control)" begin
    precision_bits = SDPX.factor_receipt_precision(Float64)
    receipt = SDPX.FactorReceipt(
        1, 1, UInt64(123), :symmetric_augmented_core, :cholmod,
        Float64, precision_bits, 0.0, :none, :factored, 0.0, false, 0, 0,
    )
    good_kwargs = (
        matrix_epoch=1, factor_epoch=1, pattern_signature=UInt64(123),
        route=:symmetric_augmented_core, provider=:cholmod,
        regularization=0.0, factor_status=:factored,
    )
    @test SDPX.factor_receipt_owned(receipt; good_kwargs...)
    # Stale epoch (e.g. a receipt kept across a refactor).
    @test !SDPX.factor_receipt_owned(
        receipt; matrix_epoch=2, factor_epoch=1,
        pattern_signature=UInt64(123), route=:symmetric_augmented_core,
        provider=:cholmod, regularization=0.0, factor_status=:factored,
    )
    # Corrupted pattern signature.
    @test !SDPX.factor_receipt_owned(
        receipt; matrix_epoch=1, factor_epoch=1,
        pattern_signature=UInt64(999), route=:symmetric_augmented_core,
        provider=:cholmod, regularization=0.0, factor_status=:factored,
    )
    # Wrong provider / route.
    @test !SDPX.factor_receipt_owned(
        receipt; matrix_epoch=1, factor_epoch=1,
        pattern_signature=UInt64(123), route=:symmetric_augmented_core,
        provider=:qdldl, regularization=0.0, factor_status=:factored,
    )
    # Provider receipts are evidence only (`proof_valid=false`): demanding a
    # proof fails closed instead of reusing the receipt as a certificate.
    @test !SDPX.factor_receipt_owned(receipt; good_kwargs..., require_proof=true)
    # `nothing` (no receipt) has no `factor_receipt_owned` dispatch at all:
    # every production call site guards `receipt === nothing` first (see
    # `src/hsd/native_hsd_public.jl` and `src/hsd/product_cone_hsd.jl`), so
    # there is no receipt object to corrupt or reuse in that state.  This
    # absence-of-dispatch is recorded as a gap in the report rather than
    # asserted as a validation outcome here.
    println("R2-A-receipt: exact-match validates; stale/corrupted/proof-demanding do not")
end

@testset "R2-A: prepared 100 c/b updates — truthful provider symbolic count" begin
    problem = _r2a_problem()
    options = _r2a_options()
    fp0 = SDPX.structure_fingerprint(problem, options)

    prepared = SDPX.prepare(problem, options)
    @test prepared.structure.fingerprint == fp0

    # Warm-up solve (covers JIT) + record the ACTUAL executed provider for
    # this structure.  The counter delta below is interpreted against that
    # fact, not against an assumed provider.
    warm = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(prepared; objective=_C0, rhs=_BEQ0)
    end
    @test warm.result.status == SDPX.Optimal
    provider = warm.result.diagnostics.memory.symmetric_core_actual_provider
    d_warm = warm.delta
    println("R2-A-prepared: executed_provider=", provider,
        " warmup_solve_symbolic_delta=", d_warm)
    @test provider in (:cholmod, :native_disconnected_ldlt)
    # Per-solve symbolic work is deterministic for a fixed structure: one
    # more identical numeric-only update must analyze exactly as many times
    # as the warm-up solve did (fresh workspace per solve today).
    probe = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(
            prepared;
            objective=Float64[1.01, 2.0, 3.0],
            rhs=Float64[1.5],
        )
    end
    @test probe.result.status == SDPX.Optimal
    d1 = probe.delta
    @test d1 == d_warm
    println("R2-A-prepared: per_update_symbolic_delta d1=", d1)

    # Structure-cache metadata behavior (NOT a symbolic claim): the frozen
    # pattern is shared across same-structure solves.
    cache_before_updates = SDPX.structure_cache_stats()

    n_updates = 100
    before = SDPX.symbolic_analysis_count()
    for k in 1:n_updates
        c_new = Float64[
            _C0[1] + 0.01 * sin(k + 1.0),
            _C0[2] + 0.01 * sin(k + 2.0),
            _C0[3] + 0.01 * sin(k + 3.0),
        ]
        b_new = Float64[1.5 + 0.005 * cos(k)]
        result = SDPX.solve!(prepared; objective=c_new, rhs=b_new)
        @test result.status == SDPX.Optimal
        @test result.p_res <= options.ϵ_primal
        @test result.d_res <= options.ϵ_dual
        @test result.gap_rel <= options.ϵ_gap
        # The changed data was actually consumed (not a stale certificate).
        @test abs(sum(result.x) - only(b_new)) <=
            options.ϵ_primal * max(1.0, abs(only(b_new)))
        @test abs(dot(c_new, result.x) - result.pObj) <=
            16eps(Float64) * max(1.0, abs(result.pObj))
    end
    delta_100 = SDPX.symbolic_analysis_count() - before
    @test prepared.state.solve_count == n_updates + 2
    @test prepared.state.structure_invalidations == 0
    @test prepared.structure.fingerprint == fp0

    # Truthful linearity law for the CURRENT implementation: every
    # same-structure solve performs the same provider symbolic work because
    # each solve builds a fresh symmetric-core workspace + provider cache.
    # Structure-cache entries stay bounded (metadata reuse is real), while
    # provider analyses repeat (numeric/symbolic separation is absent).
    @test delta_100 == n_updates * d1
    cache_after_updates = SDPX.structure_cache_stats()
    @test cache_after_updates.entries - cache_before_updates.entries <= 1

    gate_passed = (delta_100 == 1)
    println("R2-A-prepared: updates=", n_updates,
        " d1=", d1,
        " delta_100=", delta_100,
        " provider=", provider,
        " cache_entries_delta=",
        cache_after_updates.entries - cache_before_updates.entries)
    if provider === :cholmod
        println("R2-A-GATE: NOT PASSED — each of the ", n_updates,
            " same-structure updates re-ran one CHOLMOD symbolic analysis",
            " (delta_100=", delta_100,
            " == 100*d1); cross-solve symbolic reuse is not implemented.")
    else
        println("R2-A-GATE: NOT APPLICABLE at delta==1 for this structure —",
            " provider=", provider,
            " performs dense per-component LDL with no sparse symbolic",
            " phase (delta_100=", delta_100, ");",
            " a sparse-provider fixture is still needed to test reuse.")
    end
    @test !gate_passed
end

@testset "R2-A: structure change invalidates; new structure analyzes fresh" begin
    prepared = SDPX.prepare(_r2a_problem(), _r2a_options())
    warm = SDPX.solve!(prepared; objective=_C0, rhs=_BEQ0)
    @test warm.status == SDPX.Optimal
    provider = warm.diagnostics.memory.symmetric_core_actual_provider
    d1 = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(prepared; objective=_C0, rhs=_BEQ0)
    end.delta

    # Structural change: one extra inequality row.
    G_altered = vcat(_G0, Float64[1.0 1.0 0.0])
    h_altered = vcat(_H0, Float64[1.2])
    altered = SDPX.linear_program(_C0, G_altered, h_altered;
        Aeq=_AEQ0, beq=_BEQ0)
    @test SDPX.structure_fingerprint(altered, _r2a_options()) !=
        prepared.structure.fingerprint
    threw = false
    reason = :unobserved
    try
        SDPX.solve!(prepared, altered)
    catch err
        @test err isa SDPX.PreparedStructureMismatch
        threw = true
        reason = err.reason
    end
    @test threw
    @test reason == :structure_changed
    @test prepared.state.structure_invalidations == 1

    # A fresh prepared session over the new structure solves and analyzes
    # exactly like any fresh structure (d1 per solve for this provider).
    prepared2 = SDPX.prepare(altered, _r2a_options())
    @test prepared2.structure.fingerprint != prepared.structure.fingerprint
    measured = SDPX.symbolic_analysis_delta() do
        SDPX.solve!(prepared2; objective=_C0, rhs=_BEQ0)
    end
    @test measured.result.status == SDPX.Optimal
    @test measured.delta == d1
    @test measured.result.diagnostics.memory.symmetric_core_actual_provider ==
        provider
    println("R2-A-structure-change: reason=", reason,
        " invalidations=", prepared.state.structure_invalidations,
        " new_structure_delta=", measured.delta, " provider=", provider)
end

@testset "R2-A: provider/precision/thread-budget changes (truthful behavior)" begin
    problem = _r2a_problem()
    fp_auto = SDPX.structure_fingerprint(problem, _r2a_options())

    # Thread budget is part of the structural fingerprint: a different
    # budget is a different structure (R2-B invalidation semantics present
    # at the fingerprint layer).  The native bridge still executes with
    # threads=1; that routing fact is recorded, not changed.
    options_t2 = SDPX.SolverOptions{Float64}(;
        verbosity=0, timing=false, threads=2,
    )
    @test SDPX.structure_fingerprint(problem, options_t2) != fp_auto

    # Provider request is part of the fingerprint as well.
    options_bfla = SDPX.SolverOptions{Float64}(;
        verbosity=0, timing=false, threads=1,
        linear_algebra_backend=:bfla,
    )
    @test SDPX.structure_fingerprint(problem, options_bfla) != fp_auto

    # Precision change is invalidation-by-construction: the fingerprint
    # mixes the arithmetic type, and a Float64 session cannot consume a
    # BigFloat problem (method-level type separation, no silent reuse).
    big_problem = SDPX.linear_program(
        BigFloat.(_C0), BigFloat.(_G0), BigFloat.(_H0);
        Aeq=BigFloat.(_AEQ0), beq=BigFloat.(_BEQ0),
    )
    fp_big = SDPX.structure_fingerprint(
        big_problem,
        SDPX.SolverOptions{BigFloat}(; verbosity=0, timing=false, threads=1),
    )
    @test fp_big != fp_auto
    prepared = SDPX.prepare(problem, _r2a_options())
    @test_throws MethodError SDPX.solve!(prepared, big_problem)

    println("R2-A-invalidation: threads/provider/precision all change the",
        " fingerprint; precision mismatch is a MethodError (no silent reuse).",
        " Native execution threads remain 1 via the entrypoint bridge",
        " (routing unchanged).")
end
