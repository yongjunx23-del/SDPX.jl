using SDPX
using Test
using SparseArrays
using LinearAlgebra

if !isdefined(Main, :GenericConicBenchmark)
    include(joinpath(
        @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
    ))
end
const _P6_GENERAL = Main.GenericConicBenchmark

function _p6_general_model(id::Symbol)
    spec = only(filter(
        candidate -> candidate.id === id,
        _P6_GENERAL.inventory(; tier=:small),
    ))
    return _P6_GENERAL.build(spec.problem, Float64, spec.params)
end

const _P6_SETTINGS = SDPX.Settings{Float64}(
    engine=:native_hsd,
    kkt_route=:sparse_schur,
    tolerances=SDPX.Tolerances{Float64}(primal=1e-8, dual=1e-8, gap=1e-8),
    limits=SDPX.Limits(iterations=400, time=60.0, threads=1),
    verbosity=0,
)

_p6_lp() = _p6_general_model(:lp_afiro_style)
_p6_socp() = _p6_general_model(:socp_portfolio_small)

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

function _p6_block_system(
    block_count::Int=3; block_dimension::Int=2, n::Int=3,
)
    T = Float64
    m = block_count * block_dimension
    A = spzeros(T, m, n)
    for block_index in 1:block_count
        rows = ((block_index - 1) * block_dimension + 1):(
            block_index * block_dimension
        )
        for (local_row, row) in enumerate(rows)
            column = mod1(block_index + local_row - 1, n)
            A[row, column] = one(T) + T(block_index + local_row) / T(17)
        end
    end
    ranges = UnitRange{Int}[
        ((index - 1) * block_dimension + 1):(index * block_dimension)
        for index in 1:block_count
    ]
    operators = Matrix{T}[
        T(2 + index / 10) .* Matrix{T}(I, block_dimension, block_dimension)
        for index in 1:block_count
    ]
    cone = SDPX.BlockProductConeLinearization{T}(
        operators, zeros(T, m), ranges,
    )
    rhs = SDPX.HSDNewtonRHS(
        fill(T(0.2), m), fill(T(0.1), n), T(0.3),
        fill(T(0.4), m), T(0.5),
    )
    return SDPX.NewtonSystem(
        A, fill(T(0.5), m), fill(T(0.25), n), cone,
        T(1.3), T(0.9), rhs,
    )
end

@testset "Phase 6 sparse production integration" begin
    @test SDPX.Settings{Float64}().kkt_route === :bordered

    @testset "public solve parity and original-coordinate certificates" begin
        lp_result = _p6_assert_optimal(_p6_lp(), 9.0; atol=3e-6)
        _p6_assert_optimal(_p6_socp(), 1.0; atol=3e-6)
        rank_one_result = _p6_assert_optimal(_p6_rank_one_psd(), -0.5)
        _p6_assert_optimal(_p6_mixed_psd(), 1.0)
        _p6_assert_optimal(_p6_bounded_nonpositive(), 2.0)
        _p6_assert_optimal(_p6_dependent_scaled_lp(), 1.0)

        planned = lp_result.diagnostics.plan.payload
        @test planned.kkt_execution isa SDPX.NativeHSDKKTDescriptor
        @test planned.storage === :sparse
        @test planned.factorization === :sparse_lu
        @test planned.provider === :suitesparse_umfpack
        executed = lp_result.diagnostics.selected_algorithms
        @test executed.planned_kkt_route === :sparse_schur
        @test executed.executed_kkt_route === :sparse_schur
        @test executed.executed_kkt_storage === :sparse
        @test executed.executed_backend === :sparse_reduced_schur
        @test executed.executed_factorization === :sparse_lu
        @test executed.la_executed_provider === :suitesparse_umfpack
        @test executed.fallback_reason === :none
        @test executed.attempted_kkt_routes === (:sparse_schur,)
        @test executed.executed_fallback_chain === (:sparse_schur,)
        @test lp_result.diagnostics.plan.parameters.factorization_reuse ===
              :one_numeric_factor_per_predictor_corrector_epoch

        fallback = rank_one_result.diagnostics.selected_algorithms
        @test fallback.planned_kkt_route === :sparse_schur
        @test fallback.executed_kkt_route === :expanded
        @test fallback.executed_kkt_storage === :dense
        @test fallback.executed_backend ===
              :native_hsd_expanded_quasidefinite
        @test fallback.executed_factorization === :quasidefinite_ldlt
        @test fallback.fallback_reason ===
              :sparse_factor_or_refinement_failure
        @test fallback.attempted_kkt_routes ===
              (:sparse_schur, :expanded)
        @test fallback.executed_fallback_chain ===
              (:sparse_schur, :expanded)

        bordered_result = SDPX.optimize!(
            _p6_lp(); settings=SDPX.Settings{Float64}(
                engine=:native_hsd, kkt_route=:bordered,
                limits=SDPX.Limits(iterations=400, time=60.0, threads=1),
                verbosity=0,
            ),
        )
        bordered_metadata = bordered_result.diagnostics.selected_algorithms
        @test bordered_metadata.executed_kkt_route === :bordered
        @test bordered_metadata.attempted_kkt_routes === (:bordered,)
        @test bordered_metadata.executed_fallback_chain === (:bordered,)

        canonical = SDPX.canonicalize(
            SDPX.compile_product_cone_model(_p6_lp()),
        )
        reduction = SDPX.hsd_equality_reduce(canonical)
        triple = SDPX._native_hsd_diagnostics(
            lp_result.diagnostics.plan, reduction, SDPX.Optimal,
            :verified_accepted_step, 1, 1, 0.0, 0.0, 0.0;
            executed_kkt_route=:bordered,
            executed_kkt_attempts=(:sparse_schur, :expanded, :bordered),
        ).selected_algorithms
        @test triple.executed_kkt_route === :bordered
        @test triple.attempted_kkt_routes ===
              (:sparse_schur, :expanded, :bordered)
        @test triple.executed_fallback_chain ===
              (:sparse_schur, :expanded, :bordered)
        @test triple.fallback_reason === :sparse_and_expanded_failure
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

    @testset "block-local cone storage is O(max block squared)" begin
        system = _p6_block_system(80; block_dimension=2, n=8)
        @test system.cone isa SDPX.BlockProductConeLinearization
        @test !hasproperty(system.cone, :operator)
        @test all(size(operator) == (2, 2) for operator in system.cone.operators)
        session = SDPX.SparseSchurSession(Float64, 8, 160)
        @test SDPX.assemble_sparse_schur_operator!(session, system)
        @test session.maximum_block_dimension == 2
        @test length(session.block_inverses) == 80
        @test all(size(inverse) == (2, 2) for inverse in session.block_inverses)
        @test size(session.block_augmented) == (2, 4)
        dense_workspace_sizes = Int[]
        for name in fieldnames(typeof(session))
            value = getfield(session, name)
            value isa Matrix && push!(dense_workspace_sizes, length(value))
        end
        append!(dense_workspace_sizes, length.(session.block_inverses))
        @test maximum(dense_workspace_sizes) <= 2 * 2^2
        @test all(size(value) != (session.m, session.m) for value in (
            session.block_augmented, session.block_inverses...,
        ))
    end

    @testset "block metadata and stale-factor rejection" begin
        T = Float64
        rhs3 = zeros(T, 3)
        H3 = Matrix{T}(I, 3, 3)
        @test_throws ArgumentError SDPX.ProductConeLinearization{T}(
            H3, rhs3, UnitRange{Int}[1:1, 3:3],
        )
        @test_throws ArgumentError SDPX.ProductConeLinearization{T}(
            H3, rhs3, UnitRange{Int}[1:2, 2:3],
        )
        @test_throws DimensionMismatch SDPX.ProductConeLinearization{T}(
            H3, rhs3, UnitRange{Int}[1:2, 3:4],
        )
        @test_throws DimensionMismatch SDPX.BlockProductConeLinearization{T}(
            Matrix{T}[ones(T, 2, 2), ones(T, 2, 2)], rhs3,
            UnitRange{Int}[1:1, 2:3],
        )

        system = _p6_block_system()
        session = SDPX.SparseSchurSession(Float64, 3, 6)
        @test SDPX.assemble_sparse_schur!(session, system)
        @test SDPX.factor_sparse_schur!(session)
        @test session.factor !== nothing
        session.factor_numeric_epoch -= 1
        @test !SDPX.solve_sparse_schur!(session, zeros(4))
        @test session.factor === nothing
        @test session.last_reason === :sparse_factor_stale

        condition_session = SDPX.SparseSchurSession(Float64, 3, 6)
        @test SDPX.assemble_sparse_schur!(condition_session, system)
        condition_session.condition_floor = 1.0
        @test !SDPX.factor_sparse_schur!(condition_session)
        @test condition_session.factor === nothing
        @test condition_session.last_reason === :sparse_condition_rejected

        drift_session = SDPX.SparseSchurSession(Float64, 3, 6)
        @test SDPX.assemble_sparse_schur!(drift_session, system)
        @test SDPX.factor_sparse_schur!(drift_session)
        drift_A = copy(system.A)
        drift_A[1, 2] = 0.125
        drifted = SDPX.NewtonSystem(
            drift_A, system.b, system.c, system.cone,
            system.tau, system.kappa, system.rhs,
        )
        @test !SDPX.assemble_sparse_schur_rhs!(drift_session, drifted)
        @test drift_session.factor === nothing
        @test drift_session.last_reason === :sparse_pattern_drift

        singular_session = SDPX.SparseSchurSession(Float64, 3, 6)
        @test SDPX.assemble_sparse_schur!(singular_session, system)
        fill!(singular_session.schur.nzval, 0.0)
        @test !SDPX.factor_sparse_schur!(singular_session)
        @test singular_session.factor === nothing
        @test singular_session.last_reason in (
            :sparse_factor_singular, :sparse_condition_rejected,
        )
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
        @test state.kkt_route_attempts == [:sparse_schur, :expanded]
        @test state.diagnostic === :sparse_to_expanded_same_iterate_fallback
    end

    @testset "sparse expanded bordered attempt sequence preserves iterate" begin
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
        @test SDPX._product_hsd_record_route_attempt!(state, :expanded) ===
              (:sparse_schur, :expanded)
        state.kkt_route = :expanded
        code = SDPX._product_hsd_retry_bordered_same_iterate!(state, false)
        @test code === SDPX.HSDStepOK
        @test Tuple(state.kkt_route_attempts) ===
              (:sparse_schur, :expanded, :bordered)
        @test state.kkt_route === :bordered
        @test state.base.x == snapshot[1]
        @test state.base.s == snapshot[2]
        @test state.base.y == snapshot[3]
        @test state.base.tau == snapshot[4]
        @test state.base.kappa == snapshot[5]
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
