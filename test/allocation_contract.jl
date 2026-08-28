# Phase 4 allocation-contract regression gate (full arithmetic family).
#
# One full predictor-corrector Newton iteration (SDP route) in steady state
# must stay below a documented per-precision allocation ceiling. This is a
# regression gate (fail on serious allocation blowups), not the final
# zero-allocation target. This file retains a bounded-allocation regression for
# the legacy SDP route; the real native-HSD hard gate is
# `test/hsd_zeroalloc.jl`, reproducible with
# `benchmark/general/performance/hsd_allocation.jl --check`. The semantic gate (Optimal + valid
# certificate) pins the objective/iteration/residual/gap/certificate side.
using SDPX
using Test
using LinearAlgebra

const _MULTIFLOATS = try
    @eval import MultiFloats
    true
catch
    false
end

# Ceilings keep ~3-4x headroom over the current steady-state measurements
# (Float64 ~6.5 KB, Float64x2/x3/x4 ~10-15 KB, BigFloat256 ~102 KB) so minor
# GC/compiler noise cannot fail the gate while a real hot-loop allocation
# regression is caught. Ceilings are documented per arithmetic.
const ALLOC_CEILINGS = Dict{Type,Int}(
    Float64 => 24_000,
    BigFloat => 256_000,
)
const _MULTIFLOAT_CEILING = 48_000

function _gate_sdp_problem(::Type{T}) where {T}
    k = 3
    m = k * (k + 1) ÷ 2
    c = zeros(T, m)
    c[1] = -one(T)
    A = zeros(T, m, k, k)
    A[1, 1, 1] = one(T)
    A[2, 2, 2] = one(T)
    A[3, 3, 3] = one(T)
    A[4, 1, 2] = one(T); A[4, 2, 1] = one(T)
    A[5, 1, 3] = one(T); A[5, 3, 1] = one(T)
    A[6, 2, 3] = one(T); A[6, 3, 2] = one(T)
    B = zeros(T, m, 1)
    B[1, 1] = one(T); B[2, 1] = one(T); B[3, 1] = one(T)
    return SDPX.ingest(c, [A], [zeros(T, k, k)], B, T[3];
        T=T, sparse=false, verbosity=0)
end

function _steady_state_iteration_alloc(::Type{T}) where {T}
    prob = _gate_sdp_problem(T)
    opts = SDPX.SolverOptions{T}(
        algorithm=:sdp, presolve=false, scaling=:none, verbosity=0,
        iter_max=200,
    )
    ws = SDPX.Workspace(prob; thread_count=1)
    init = SDPX._kkt_cold_start_initialization(ws, prob, opts)
    @test init.ok
    x, X, y, Y, mu = init.x, init.X, init.y, init.Y, init.μ
    SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=1)  # JIT warm-up
    return minimum(
        @allocated(SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2))
        for _ in 1:3
    )
end

@testset "allocation contract (full arithmetic family)" begin
    @testset "Float64" begin
        alloc = _steady_state_iteration_alloc(Float64)
        @test alloc < ALLOC_CEILINGS[Float64]
        @info "Float64 per-iteration Julia allocation" alloc_bytes=alloc ceiling=ALLOC_CEILINGS[Float64]
    end
    @testset "BigFloat256" begin
        setprecision(BigFloat, 256) do
            alloc = _steady_state_iteration_alloc(BigFloat)
            @test alloc < ALLOC_CEILINGS[BigFloat]
            @info "BigFloat256 per-iteration Julia allocation" alloc_bytes=alloc ceiling=ALLOC_CEILINGS[BigFloat]
        end
    end
    if _MULTIFLOATS
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4)
            @testset "$T" begin
                alloc = _steady_state_iteration_alloc(T)
                @test alloc < _MULTIFLOAT_CEILING
                @info "$T per-iteration Julia allocation" alloc_bytes=alloc ceiling=_MULTIFLOAT_CEILING
            end
        end
    end
end
