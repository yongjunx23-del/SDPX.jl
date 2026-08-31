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
