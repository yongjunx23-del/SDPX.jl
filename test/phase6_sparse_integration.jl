using SDPX
using Test
using SparseArrays
using LinearAlgebra

include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "BootstrapBenchmark.jl"))
const _P6_BB = Main.BootstrapBenchmark

const _P6_SETTINGS = SDPX.Settings{Float64}(
    engine=:native_hsd,
    kkt_route=:sparse_schur,
    tolerances=SDPX.Tolerances{Float64}(primal=1e-8, dual=1e-8, gap=1e-8),
    limits=SDPX.Limits(iterations=400, time=60.0, threads=1),
    verbosity=0,
)

function _p6_lp()
    return _P6_BB.build(_P6_BB.PROBLEMS[:lp], Float64, (sites=4,))
end

function _p6_socp()
    return _P6_BB.build(
        _P6_BB.PROBLEMS[:socp], Float64,
        (partial_waves=2, grid_points=4, analytic_coefficients=2),
    )
end

function _p6_rank_one_psd()
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :trace, X[1, 1] + X[2, 2] - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 2])
    return model
end

function _p6_mixed_psd()
    model = SDPX.Model(Float64)
    t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())
    M = SDPX.variable!(model, :M, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :link, M[1, 1] - t[1], SDPX.ZeroCone())
    SDPX.constraint!(
        model, :upper,
        [1.0 - M[1, 1] -M[1, 2]; -M[1, 2] 1.0 - M[2, 2]],
        SDPX.PSDCone(),
    )
    SDPX.objective!(model, SDPX.Maximize(), t[1])
    return model
end

function _p6_bounded_nonpositive()
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :sum, y[1] + y[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :lower1, y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :lower2, y[2], SDPX.Nonnegative())
    SDPX.constraint!(model, :upper1, y[1] - 1.0, SDPX.Nonpositive())
    SDPX.constraint!(model, :upper2, y[2] - 1.0, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), y[1] + 2.0 * y[2])
    return model
end

function _p6_dependent_scaled_lp()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :unit, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(
        model, :scaled_duplicate,
        1.0e6 * x[1] + 1.0e6 * x[2] - 1.0e6,
        SDPX.ZeroCone(),
    )
    SDPX.objective!(model, SDPX.Minimize(), x[1] + 2.0 * x[2])
    return model
end

function _p6_reduced(model)
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    return SDPX.hsd_equality_reduce(canonical).reduced
end

function _p6_solve(model)
    return SDPX.optimize!(model; settings=_P6_SETTINGS)
end

function _p6_assert_optimal(model, expected; atol=3e-6)
    result = _p6_solve(model)
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    @test isapprox(SDPX.primal_objective(result), expected; atol=atol)
    return result
end

@testset "Phase 6 sparse production integration" begin
    @test SDPX.Settings{Float64}().kkt_route === :bordered

    @testset "public solve parity and original-coordinate certificates" begin
        _p6_assert_optimal(_p6_lp(), -10.346; atol=3e-6)
        _p6_assert_optimal(_p6_socp(), 2.7272; atol=3e-6)
        _p6_assert_optimal(_p6_rank_one_psd(), -0.5)
        _p6_assert_optimal(_p6_mixed_psd(), 1.0)
        _p6_assert_optimal(_p6_bounded_nonpositive(), 2.0)
        _p6_assert_optimal(_p6_dependent_scaled_lp(), 1.0)
    end

    @testset "per-state pattern ownership and epoch counters" begin
        reduced = _p6_reduced(_p6_socp())
        first_state = SDPX.ProductConeHSDState(
            reduced; kkt_route=:sparse_schur,
        )
        second_state = SDPX.ProductConeHSDState(
            reduced; kkt_route=:sparse_schur,
        )
        @test first_state.sparse_schur !== nothing
        @test first_state.sparse_schur !== second_state.sparse_schur
        @test first_state.sparse_schur.schur !== second_state.sparse_schur.schur
        @test !first_state.sparse_schur.symbolic_reuse_supported

        SDPX.product_hsd_cold_start!(first_state)
        @test SDPX.product_hsd_step!(first_state) === SDPX.HSDStepOK
        session = first_state.sparse_schur
        @test session.structural_assembly_count == 1
        @test session.numeric_assembly_count == 1
        @test session.numeric_factor_count == 1
        @test session.rhs_assembly_count == 2
        @test session.schur isa SparseMatrixCSC
        @test nnz(session.schur) <= length(session.schur)

        @test SDPX.product_hsd_step!(first_state) === SDPX.HSDStepOK
        @test session.structural_assembly_count == 1
        @test session.numeric_assembly_count == 2
        @test session.numeric_factor_count == 2
        @test session.rhs_assembly_count == 4
        @test session.pattern_reuse_count == 1
    end

    @testset "same-iterate sparse to expanded fallback" begin
        reduced = _p6_reduced(_p6_socp())
        state = SDPX.ProductConeHSDState(reduced; kkt_route=:sparse_schur)
        SDPX.product_hsd_cold_start!(state)
        SDPX.hsd_residual!(state.base)
        @test SDPX.try_update_scaling!(
            state.runtime, state.base.s, state.base.y, state.base.mu,
        )
        snapshot = (
            copy(state.base.x), copy(state.base.s), copy(state.base.y),
            state.base.tau, state.base.kappa,
        )
        state.sparse_schur.status = SDPX.SPARSE_SCHUR_FACTOR_FAILED
        state.sparse_schur.last_reason = :manufactured_factor_failure
        code = SDPX._product_hsd_retry_expanded_same_iterate!(state, false)
        @test code === SDPX.HSDStepOK
        @test state.base.x == snapshot[1]
        @test state.base.s == snapshot[2]
        @test state.base.y == snapshot[3]
        @test state.base.tau == snapshot[4]
        @test state.base.kappa == snapshot[5]
        @test state.kkt_route === :expanded
        @test state.diagnostic === :sparse_to_expanded_same_iterate_fallback
    end

    @testset "unsupported arithmetic never downcasts" begin
        float32_session = SDPX.SparseSchurSession(Float32, 1, 1)
        float32_session.schur = sparse(Float32[2 1; 1 2])
        @test !SDPX.factor_sparse_schur!(float32_session)
        @test float32_session.last_reason === :sparse_factor_type_unsupported
        @test eltype(float32_session.schur) === Float32

        setprecision(BigFloat, 128) do
            session = SDPX.SparseSchurSession(BigFloat, 1, 1)
            session.schur = sparse(BigFloat[2 1; 1 2])
            @test !SDPX.factor_sparse_schur!(session)
            @test session.status === SDPX.SPARSE_SCHUR_FACTOR_FAILED
            @test session.last_reason === :sparse_factor_type_unsupported
            @test eltype(session.schur) === BigFloat
        end
    end

    @testset "typed exhaustion" begin
        reduced = _p6_reduced(_p6_lp())
        result = SDPX.product_hsd_solve!(
            SDPX.ProductConeHSDState(reduced; kkt_route=:sparse_schur);
            max_iterations=0,
        )
        @test result.status === SDPX.ProductHSDMaxIterations
        @test result.reason === SDPX.ProductHSDIterationLimitReached
        @test result.factorizations == 0
    end
end
