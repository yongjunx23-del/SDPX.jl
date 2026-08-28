# Actual native product-cone HSD hot-step allocation gate
# (ProductConeHSDState + product_hsd_cold_start! + product_hsd_step!, not the
# legacy HotStepState or HSDState/nonnegative_hsd step gates).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using SparseArrays
using MultiFloats

function _hsdza_program(::Type{T}) where {T<:AbstractFloat}
    m, n = 20, 10
    A = Matrix{T}(undef, m, n)
    b = Vector{T}(undef, m)
    c = Vector{T}(undef, n)
    @inbounds for j in 1:n, i in 1:m
        A[i, j] = T(mod(7i + 11j + 3i * j, 29) - 14) / T(13)
    end
    @inbounds for i in 1:m
        b[i] = T(10) + T(i) / T(7)
    end
    @inbounds for j in 1:n
        c[j] = T(mod(5j, 13) - 6) / T(5)
    end
    descriptor = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
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
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout, chain,
    )
end

# Returning an enum to an interpreted top-level `@allocated` expression boxes
# the return value by 16 bytes.  The production caller consumes the code in
# compiled code, so this gate stores it in caller-owned memory and returns
# `nothing`; all allocations measured below are therefore from `product_hsd_step!`.
@inline function _hsdza_step_noreturn!(codes, index::Int, state)
    codes[index] = SDPX.product_hsd_step!(state)
    return nothing
end

@testset "actual HSD fixed-width step is allocation-free" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        state = SDPX.ProductConeHSDState(_hsdza_program(T))
        SDPX.product_hsd_cold_start!(state)
        warm = Vector{SDPX.HSDStepCode}(undef, 1)
        _hsdza_step_noreturn!(warm, 1, state)
        @test warm[1] === SDPX.HSDStepOK

        codes = Vector{SDPX.HSDStepCode}(undef, 10)
        samples = Vector{Int}(undef, 10)
        factors_before = SDPX.product_hsd_factor_count(state)
        epoch_before = state.base.epoch
        @inbounds for sample in 1:10
            samples[sample] = @allocated _hsdza_step_noreturn!(codes, sample, state)
        end

        @test all(==(SDPX.HSDStepOK), codes)
        @test samples == zeros(Int, 10)
        @test SDPX.product_hsd_factor_count(state) - factors_before == 10
        @test state.base.epoch - epoch_before == 10
        # The active bordered route keeps its own factor epoch, and the
        # numeric factor must be current for the epoch that assembled it.
        @test state.symmetric_bordered.factor_epoch ==
              state.symmetric_bordered.assembly_epoch == state.base.epoch
        @test isfinite(state.base.record.p_res)
        @test isfinite(state.base.record.d_res)
        @test isfinite(state.base.record.mu)
    end
end
