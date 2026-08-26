using SDPX
using SparseArrays
using Test

function _time_limit_canonical(::Type{T}=Float64) where {T<:AbstractFloat}
    descriptor = SDPX.ConeBlockDescriptor(T, :nonnegative, 2; offset=1)
    layout = SDPX.canonical_layout([descriptor])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1,
        zero(T),
        SDPX.VariableRef[],
        SDPX.ConstraintRef[],
        SDPX.VariableRef[],
        0,
    )
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T),
        bits,
        T[-1, -1],
        sparse(T[1 0; 0 1]),
        T[1, 1],
        layout,
        chain,
    )
end

@testset "product HSD explicit time limit" begin
    state = SDPX.ProductConeHSDState(_time_limit_canonical())
    result = SDPX.product_hsd_solve!(state; max_time=0.0)
    @test result.status === SDPX.ProductHSDTimeLimit
    @test result.reason === SDPX.ProductHSDTimeLimitReached
    @test result.iterations == 0
    @test result.factorizations == 0
    @test all(iszero, result.x)
    @test all(iszero, result.s)
    @test all(iszero, result.y)

    @test_throws ArgumentError SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(_time_limit_canonical()); max_time=-1,
    )
    @test_throws ArgumentError SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(_time_limit_canonical()); max_time=NaN,
    )

    ordinary = SDPX.product_hsd_solve(
        _time_limit_canonical(); max_time=Inf, max_iterations=300,
    )
    @test ordinary.status === SDPX.ProductHSDOptimal
    @test ordinary.reason in (
        SDPX.ProductHSDVerifiedAcceptedStep,
        SDPX.ProductHSDVerifiedTerminalNewtonTrial,
    )
end
