using SDPX
using Test
using SparseArrays

function _wave_h_nonnegative_program()
    layout = SDPX.ConeProductLayout([
        SDPX.CanonicalConeBlock(:nonnegative, 1, 1, 1,
            SDPX.CanonicalBlockMap(:constraint, 1, 1, 1, 1)),
    ])
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(Float64), 53, [1.0], sparse(reshape([1.0], 1, 1)),
        [1.0], layout, SDPX.ReconstructionChain(), 0.0,
    )
end

@testset "Wave H route storage ownership" begin
    state = SDPX.HSDState(_wave_h_nonnegative_program())
    mathematical_fields = fieldnames(typeof(state))
    for route_field in (
        :Ad, :Ar, :Atr, :rank_basis, :rank_null_objective,
        :rank_ambiguous, :rank_incompatible, :rank_ray, :H, :rhs,
        :qr, :rvec, :u, :w, :dxr, :driver,
    )
        @test route_field ∉ mathematical_fields
        @test hasfield(typeof(state.workspace), route_field)
    end
    @test state.Ar === state.workspace.Ar
    @test state.rank_basis === state.workspace.rank_basis
    @test state.H === state.workspace.H
end
