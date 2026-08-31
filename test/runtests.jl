# Sole SDPX regression suite: black-box modeling-to-certified-result E2E.

using Test
using SDPX
using LinearAlgebra
using SparseArrays

include(joinpath(
    @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
))
using .GenericConicBenchmark

const E2E_CASE_IDS = (
    :lp_afiro_style,
    :lp_infeasible,
    :lp_unbounded,
    :socp_portfolio_small,
    :socp_ill_scaled_small,
    :rsoc_epigraph_small,
    :sdp_maxcut_k4,
    :psd_blockdiag_small,
    :psd_dense_maxcut_k5,
    :psd_ill_scaled_small,
    :exp_unit_small,
    :power_epigraph_small,
    :mixed_orthant_exp_small,
)
function e2e_spec(id::Symbol)
    matches = filter(
        spec -> spec.id === id,
        GenericConicBenchmark.inventory(; tier=:small),
    )
    length(matches) == 1 || error(
        "expected exactly one general E2E case $id, found $(length(matches))",
    )
    return only(matches)
end

@testset "LPLU factor cache lifecycle" begin
    # Keep this direct protocol fixture independent of the modeling E2Es:
    # Julia 1.10 does not provide LAPACK.getrf!(F, ipiv), so the production
    # cache must retain its own pivot storage while using the LAPACK kernel.
    cache = SDPX.LPLUCache{Float64}()
    @test SDPX.prepare!(cache, SDPX.FactorRequirements(2)) === cache
    @test SDPX.factor_status(cache) === SDPX.Prepared

    A = Float64[0 2; 1 3]  # requires a row pivot
    A2 = Float64[2 1; 1 3]
    @test SDPX.factorize!(cache, A, 1) === cache
    @test SDPX.factor_status(cache) === SDPX.Fresh
    @test SDPX.factor_epoch(cache) == 1

    x = zeros(2)
    rhs = A * [1.0, 2.0]
    @test SDPX.solve!(cache, x, rhs) === x
    @test x ≈ [1.0, 2.0]

    correction = zeros(2)
    residual = [1.0, -2.0]
    @test SDPX.refine_once!(cache, residual, correction) === correction
    @test A * correction ≈ residual

    # Numeric factorization and all solve variants remain allocation-free after
    # preparation/warm-up, including a fresh matrix epoch.
    SDPX.factorize!(cache, A2, 2)
    rhs2 = A2 * [2.0, -1.0]
    SDPX.solve!(cache, x, rhs2)
    @test x ≈ [2.0, -1.0]
    # Warm the new-epoch factorization call before measuring it so the check
    # observes cache behavior rather than first-call JIT compilation.
    SDPX.factorize!(cache, A, 3)
    @test @allocated(SDPX.factorize!(cache, A2, 4)) == 0
    @test @allocated(SDPX.solve!(cache, x, rhs)) == 0
    @test @allocated(SDPX.refine_once!(cache, residual, correction)) == 0
end

@testset "Sparse LDL failed-refactor restoration" begin
    lower=sparse(Float64[2 0; 1 -2])
    requirements=SDPX.SparseSymbolicRequirements(lower;dsigns=[1,-1])
    cache=SDPX.SparseSymbolicNumericCache{Float64}()
    SDPX.prepare!(cache,requirements)
    SDPX.factorize!(cache,lower,1)
    rhs=[3.0,-1.0]; solution=zeros(2)
    SDPX.solve!(cache,solution,rhs)
    @test Symmetric(lower,:L)*solution ≈ rhs

    nonfinite=copy(lower); nonfinite.nzval[1]=NaN
    @test_throws ArgumentError SDPX.factorize!(cache,nonfinite,2)
    @test SDPX.factor_status(cache) === SDPX.Failed
    @test cache.factor_view.nzval == lower.nzval

    SDPX.factorize!(cache,lower,2)
    singular=copy(lower); fill!(singular.nzval,0.0)
    @test_throws ArgumentError SDPX.factorize!(cache,singular,3)
    @test SDPX.factor_status(cache) === SDPX.Failed
    @test cache.factor_view.nzval == singular.nzval
    SDPX.factorize!(cache,lower,3)
    SDPX.solve!(cache,solution,rhs)
    @test Symmetric(lower,:L)*solution ≈ rhs
end

@testset "Disconnected small-component LDL lifecycle" begin
    block = Float64[2 1 0; 1 -3 0.5; 0 0.5 -2]
    full = zeros(6,6)
    full[1:3,1:3] .= block
    full[4:6,4:6] .= block
    lower = sparse(tril(full))
    cache = SDPX.DisconnectedLDLTCache(
        lower, [1,-1,-1,1,-1,-1]; max_size=4,
    )
    @test cache !== nothing
    @test SDPX.factor_status(cache) === SDPX.Prepared
    SDPX.factorize!(cache,lower,1)
    @test SDPX.factor_status(cache) === SDPX.Fresh
    rhs = collect(1.0:6.0)
    solution = zeros(6)
    SDPX.solve!(cache,solution,rhs)
    @test full*solution ≈ rhs

    drift=copy(lower)
    drift[4,1]=0.25
    @test_throws ArgumentError SDPX.factorize!(cache,drift,2)
    @test SDPX.factor_status(cache) === SDPX.Failed
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(cache,solution,rhs)
    SDPX.factorize!(cache,lower,2)
    @test SDPX.factor_status(cache) === SDPX.Fresh

    SDPX.revoke_numeric!(cache)
    @test SDPX.factor_status(cache) === SDPX.Prepared
    SDPX.factorize!(cache,lower,3)
    correction = zeros(6)
    SDPX.refine_once!(cache,rhs,correction)
    @test full*correction ≈ rhs
    SDPX.invalidate!(cache)
    @test_throws SDPX.FactorCacheStateError SDPX.factorize!(cache,lower,4)
    @test SDPX.factor_status(cache) === SDPX.Invalid
end

@testset "Scaled disjoint equalities retain structural rank" begin
    model=SDPX.Model(Float64)
    x=SDPX.variable!(model,:x,2;domain=SDPX.Reals())
    SDPX.constraint!(model,:unit,x[1]-1.0,SDPX.ZeroCone())
    SDPX.constraint!(model,:tiny,1e-20*x[2]-1e-20,SDPX.ZeroCone())
    SDPX.objective!(model,SDPX.Minimize(),x[1]+x[2])
    canonical=SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    reduction=SDPX.hsd_equality_reduce(canonical)
    @test reduction.status === SDPX.HSDEqualityReady
    @test reduction.rank==2
    @test size(reduction.range_basis)==(2,2)
    @test size(reduction.null_basis)==(2,0)
end

@testset "Pure orthant core uses disconnected LDL" begin
    spec=e2e_spec(:lp_random_small)
    result=SDPX.optimize!(
        GenericConicBenchmark.build(spec.problem,Float64,spec.params);
        settings=SDPX.Settings(Float64;verbosity=0),
    )
    certificate=SDPX.certificate(result)
    @test SDPX.status(result) === :optimal
    @test certificate.valid
    @test result.diagnostics.memory.symmetric_core_actual_provider ===
        :native_disconnected_ldlt
    @test isapprox(
        Float64(certificate.primal_objective),spec.known_objective;
        atol=spec.objective_tolerance,rtol=spec.objective_tolerance,
    )
end

@testset "Fixed-trace Q3 admits barrier-free columns" begin
    # One fixed-head Q3 pair (u,v) plus a free Wilson-style column f that
    # appears only in the equality row.  Substituting f = u + 1/2 gives the
    # same problem without the free column, so the pre-existing fixed-trace
    # path is the trusted reference for the saddle-point border.
    reference = begin
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :unit,
            Any[1.0, x[1] - 1.0, x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Maximize(), x[1] + 0.5)
        SDPX.optimize!(model; settings=SDPX.Settings(Float64; verbosity=0))
    end
    @test SDPX.status(reference) === :optimal
    @test SDPX.certificate(reference).valid
    @test SDPX.certificate(reference).primal_objective ≈ 2.5 atol=1e-8

    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 3; domain=SDPX.Reals())
    SDPX.constraint!(model, :link, x[3] - x[1] - 0.5, SDPX.ZeroCone())
    SDPX.constraint!(model, :unit,
        Any[1.0, x[1] - 1.0, x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Maximize(), x[3])
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    plan = SDPX.fixed_trace_q3_canonical_plan(canonical)
    @test plan !== nothing
    @test plan.free_ids == [3]
    result = SDPX.optimize!(model; settings=SDPX.Settings(Float64; verbosity=0))
    certificate = SDPX.certificate(result)
    @test SDPX.status(result) === :optimal
    @test certificate.valid
    @test certificate.primal_objective ≈ 2.5 atol=1e-8
    @test result.diagnostics.termination.reason === :verified_terminal_newton_trial ||
          result.diagnostics.termination.reason === :verified_accepted_step
end

@testset "Precision benchmark contract" begin
    precisions=precision_specs(Float64,Float64,Float64)
    @test length(precisions)==7
    @test precisions[1].solver_tolerance=="1e-8"
    @test precisions[5].bits==256
    spec=e2e_spec(:lp_afiro_style)
    row=run_precision_case(first(precisions),spec)
    @test row.passed
    @test row.objective isa String
    setprecision(BigFloat,192) do
        model=GenericConicBenchmark.build(
            spec.problem,BigFloat,merge(spec.params,(precision_bits=384,)),
        )
        @test model.arithmetic.precision_bits==384
    end
end

@testset "SDPX public modeling-to-certified-result E2E" begin
    for id in E2E_CASE_IDS
        @testset "$id" begin
            spec = e2e_spec(id)
            result = GenericConicBenchmark.run_one(spec, Float64)
            @test result.status === spec.expected_status
            @test result.certificate_valid
            @test result.expectation_met
        end
    end
end

@testset "Symmetric-core structure cache lifecycle" begin
    # Review slices 2/4: cross-solve structure cache. The cache stores ONLY
    # the frozen structural arrays keyed by (type, structure signature); a
    # hit must share the structure and allocate a FRESH zero numeric buffer.
    # Structural changes (dimension, cone partition, sparsity pattern) and
    # arithmetic-type changes must miss. clear_structure_cache! must drop
    # every entry and the next build must be an owned rebuild.
    SDPX.clear_structure_cache!()
    build_A(n; shift=0) = begin
        A = spzeros(Float64, n, 18)
        for j in 1:4, i in 1:n
            mod(i + 5 * j + shift, 11) == 0 && (A[i, j] = 0.25 * i + 0.5 * j)
        end
        sparse(A)
    end
    ranges = [1:60, 61:100, 101:120]
    shapes = Symbol[:dense_lower, :dense_lower, :dense_lower]

    Ar1 = build_A(120)
    p1 = SDPX.SymmetricCorePattern{Float64}(Ar1, ranges, shapes)
    st1 = SDPX.structure_cache_stats()
    @test st1.misses >= 1

    Ar1b = sparse(Ar1)  # identical structure
    p2 = SDPX.SymmetricCorePattern{Float64}(Ar1b, ranges, shapes)
    st2 = SDPX.structure_cache_stats()
    @test st2.hits == st1.hits + 1
    @test SDPX.symmetric_core_signature(p2) == SDPX.symmetric_core_signature(p1)
    @test p2.colptr === p1.colptr        # frozen structure shared
    @test p2.rowval === p1.rowval
    @test p2.nzval !== p1.nzval          # numeric buffer NOT shared
    @test all(iszero, p2.nzval)          # fresh zeros: no value survives reuse

    # sparsity-pattern change: different structure => miss => different key
    p3 = SDPX.SymmetricCorePattern{Float64}(build_A(120; shift=1), ranges, shapes)
    @test SDPX.symmetric_core_signature(p3) != SDPX.symmetric_core_signature(p2)
    @test SDPX.structure_cache_stats().misses > st2.misses

    # cone-partition change: same Ar, different block ranges => miss
    p4 = SDPX.SymmetricCorePattern{Float64}(Ar1, [1:50, 51:100, 101:120], shapes)
    @test SDPX.symmetric_core_signature(p4) != SDPX.symmetric_core_signature(p2)

    # arithmetic-type change: distinct (T, signature) key, hit on repeat
    Ar1_big = SparseMatrixCSC{BigFloat,Int}(Ar1)
    pbig1 = SDPX.SymmetricCorePattern{BigFloat}(Ar1_big, ranges, shapes)
    pbig2 = SDPX.SymmetricCorePattern{BigFloat}(
        SparseMatrixCSC{BigFloat,Int}(Ar1), ranges, shapes,
    )
    @test pbig2.colptr === pbig1.colptr
    stats_big = SDPX.structure_cache_stats()
    @test stats_big.hits == st2.hits + 1

    # explicit invalidation drops every entry
    SDPX.clear_structure_cache!()
    @test SDPX.structure_cache_stats().entries == 0

    # pattern built after clear is rebuilt (not shared) and still consistent
    p5 = SDPX.SymmetricCorePattern{Float64}(Ar1, ranges, shapes)
    @test SDPX.symmetric_core_signature(p5) == SDPX.symmetric_core_signature(p1)
    @test p5.colptr !== p1.colptr
end
