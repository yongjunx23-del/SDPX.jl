# Setup-time RRQR/PivotedQR precision gate for the LP-HSD reduction.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end
using Test
using SparseArrays
using MultiFloats

function _hsd_rank_precision_case(::Type{T}) where {T<:AbstractFloat}
    A = sparse(T[1 1; -1 -1])
    c = T[0, 0]
    return SDPX._hsd_column_reduction(A, c)
end

@testset "HSD setup RRQR across supported precision types" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        reduction = _hsd_rank_precision_case(T)
        @test length(reduction.cols) == 1
        @test reduction.dependent == [2]
        @test !reduction.ambiguous
        @test !reduction.incompatible
    end
    setprecision(BigFloat, 256) do
        reduction = _hsd_rank_precision_case(BigFloat)
        @test length(reduction.cols) == 1
        @test reduction.dependent == [2]
        @test !reduction.ambiguous
        @test !reduction.incompatible
    end
end
