using Test
using LinearAlgebra
using SparseArrays
using SDPX

function _kkti_layout(::Type{T}) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for (cone, dimension) in ((:nonnegative, 2), (:soc, 3), (:psd, 2))
        block = SDPX.ConeBlockDescriptor(T, cone, dimension; offset=offset)
        push!(blocks, block)
        offset += block.length
    end
    return SDPX.canonical_layout(blocks)
end

function _kkti_program(::Type{T}) where {T<:AbstractFloat}
    layout = _kkti_layout(T)
    m = layout.dimension
    A = zeros(T, m, 3)
    for i in 1:m
        A[i, 1] = T(i + 1) / T(7)
        A[i, 2] = (isodd(i) ? -one(T) : one(T)) * T(i + 2) / T(11)
        A[i, 3] = T(1) / T(i + 3)
    end
    b = [T(i - 4) / T(5) for i in 1:m]
    c = T[-2, 1, 3] / T(7)
    reconstruction = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    bits = T === BigFloat ? precision(BigFloat) : 53
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout,
        reconstruction,
    )
end

@testset "KKT-derived product HSD cold start" begin
    state = SDPX.ProductConeHSDState(_kkti_program(Float64))
    identity_x = copy(state.base.x)
    report = SDPX.kkt_derived_start!(state)
    @test report.ok
    @test report.reason == :none
    @test report.factor_count == 1
    @test report.rhs_solves == 2
    @test report.regularization > 0
    @test all(isfinite, state.base.x)
    @test state.base.x != identity_x
    @test SDPX.product_strictly_interior(
        state.runtime, state.base.s, state.base.y,
    )
    @test state.runtime.valid
    @test state.base.tau == 1
    @test state.base.kappa == 1
    @test isfinite(report.primal_residual_before_shift)
    @test isfinite(report.dual_residual_before_shift)
    @test isfinite(report.primal_residual_after_shift)
    @test isfinite(report.dual_residual_after_shift)
    @test report.primal_interior_shift >= 0
    @test report.dual_interior_shift >= 0
    @test report.primal_mass_shift >= 0
    @test report.dual_mass_shift >= 0
    @test report.primal_centering_shift > 0
    @test report.dual_centering_shift > 0
end

@testset "KKT-derived start generic precision" begin
    setprecision(BigFloat, 192) do
        state = SDPX.ProductConeHSDState(_kkti_program(BigFloat))
        report = SDPX.kkt_derived_start!(state)
        @test report.ok
        @test report.factor_count == 1
        @test report.rhs_solves == 2
        @test SDPX.product_strictly_interior(
            state.runtime, state.base.s, state.base.y,
        )
        @test precision(state.base.x[1]) == 192
    end
end
