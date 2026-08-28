using SDPX
using Test
using LinearAlgebra
using SparseArrays
using Random
using MultiFloats: Float64x2

const _RECEIPT_RNG = Random.Xoshiro(0xface7)

function _receipt_system(::Type{T}; n::Int=3, m::Int=4) where {T<:AbstractFloat}
    A = T.(randn(_RECEIPT_RNG, m, n))
    b = T.(randn(_RECEIPT_RNG, m))
    c = T.(randn(_RECEIPT_RNG, n))
    H = zeros(T, m, m)
    ranges = UnitRange{Int}[1:2, 3:4]
    for rows in ranges
        block = T.(randn(_RECEIPT_RNG, 2, 2))
        block = block * block' + T(2) * Matrix{T}(I, 2, 2)
        H[rows, rows] .= block
    end
    cone = SDPX.ProductConeLinearization{T}(H, zeros(T, m), ranges)
    rhs = SDPX.HSDNewtonRHS(
        T.(randn(_RECEIPT_RNG, m)), T.(randn(_RECEIPT_RNG, n)),
        T(0.3), T.(randn(_RECEIPT_RNG, m)), T(0.4),
    )
    return SDPX.NewtonSystem(A, b, c, cone, T(1.3), T(0.9), rhs)
end

function _receipt_lp(::Type{T}=Float64) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :sum, x[1] + x[2] - one(T), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + T(2) * x[2])
    return model
end

function _receipt_reduced(model)
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    return SDPX.hsd_equality_reduce(canonical).reduced
end

@testset "factor-epoch proof receipt" begin
    @testset "immutable typed receipt" begin
        for T in (Float64, Float64x2, BigFloat)
            receipt = SDPX.FactorReceipt(
                1, 2, UInt64(3), :bordered, :test_provider, T,
                SDPX.factor_receipt_precision(T), zero(T), :none,
                :factored, zero(T), true,
            )
            @test receipt.scalar_type === T
            @test receipt.precision_bits == SDPX.factor_receipt_precision(T)
            @test SDPX.factor_receipt_owned(
                receipt; matrix_epoch=1, factor_epoch=2,
                pattern_signature=UInt64(3), route=:bordered,
                provider=:test_provider, regularization=zero(T),
                require_proof=true,
            )
            @test_throws ErrorException setfield!(receipt, :matrix_epoch, 4)
        end
    end

    @testset "sparse factor owns one receipt across RHS solves" begin
        system = _receipt_system(Float64)
        session = SDPX.SparseSchurSession(Float64, 3, 4)
        @test SDPX.assemble_sparse_schur_operator!(session, system)
        @test SDPX.factor_sparse_schur!(session)
        @test session.numeric_factor_count == 1
        @test session.receipt_build_count == 1
        receipt = session.factor_receipt
        @test receipt !== nothing
        @test receipt.route === :sparse_schur
        @test !receipt.proof_valid # UMFPACK exposes no factor-wide proof.
        @test SDPX.assemble_sparse_schur_rhs!(session, system)
        solution = zeros(4)
        @test SDPX.solve_sparse_schur!(session, solution)
        @test session.receipt_build_count == 1
        # A second RHS at the same matrix/factor epoch cannot rebuild proof.
        session.status = SDPX.SPARSE_SCHUR_FACTORED
        @test SDPX.assemble_sparse_schur_rhs!(session, system)
        @test SDPX.solve_sparse_schur!(session, solution)
        @test session.numeric_factor_count == session.receipt_build_count == 1
        # Falsifying ownership test: stale epoch with a present factor is rejected.
        session.status = SDPX.SPARSE_SCHUR_FACTORED
        session.factor_numeric_epoch += 1
        @test session.factor !== nothing
        @test !SDPX.solve_sparse_schur!(session, solution)
        @test session.factor === nothing
        @test session.factor_receipt === nothing
    end

    @testset "expanded factor receipt is route-local and RHS-independent" begin
        system = _receipt_system(Float64)
        session = SDPX.ExpandedKKTSession(Float64, 3, 4; rhs_count=2)
        @test SDPX.factor_expanded_kkt!(session, system)
        @test session.numeric_factor_count == session.receipt_build_count == 1
        receipt = session.factor_receipt
        @test receipt !== nothing
        @test receipt.route === :expanded
        @test receipt.provider === :standard_pivoted_lu
        rhs = zeros(8)
        SDPX.expanded_rhs!(rhs, system)
        first_solution = similar(rhs)
        second_solution = similar(rhs)
        @test SDPX.solve_expanded!(first_solution, session, rhs)
        session.status = SDPX.EXPANDED_KKT_FACTORED
        @test SDPX.solve_expanded!(second_solution, session, rhs)
        @test first_solution == second_solution
        @test session.receipt_build_count == 1
        # Matrix epoch mutation leaves the factor object present but revokes it.
        session.matrix_epoch += 1
        @test session.factor.success
        @test !SDPX.solve_expanded!(second_solution, session, rhs)
        @test session.receipt_build_count == 1
    end

    @testset "bordered expensive proof is built once per numeric factor" begin
        reduced = _receipt_reduced(_receipt_lp())
        state = SDPX.ProductConeHSDState(reduced; kkt_route=:bordered)
        SDPX.product_hsd_cold_start!(state)
        SDPX.hsd_residual!(state.base)
        @test SDPX.try_update_scaling!(
            state.runtime, state.base.s, state.base.y, state.base.mu,
        )
        state.base.epoch += 1
        @test SDPX._product_hsd_bordered_route_direction!(state, false) ===
              SDPX.HSDStepOK
        workspace = state.symmetric_bordered
        @test workspace.factor_receipt !== nothing
        @test workspace.factor_receipt.proof_valid
        @test workspace.solves >= 2 # predictor and dependent corrector RHS
        @test workspace.receipt_build_count ==
              SDPX.product_hsd_factor_count(state) == 1
        @test SDPX.product_hsd_receipt_build_count(state) == 1
        @test SDPX.product_hsd_factor_receipt(state) === workspace.factor_receipt
        builds = workspace.receipt_build_count
        @test SDPX._product_bordered_factor_receipt_current(workspace)
        # Falsifying ownership test: retain the factor object, mutate its epoch.
        workspace.driver.route.factor_epoch += 1
        @test workspace.driver.route.status === SDPX.Fresh
        @test !SDPX._product_bordered_factor_receipt_current(workspace)
        @test workspace.receipt_build_count == builds
    end
end
