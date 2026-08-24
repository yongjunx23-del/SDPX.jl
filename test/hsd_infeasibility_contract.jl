# Phase 6 infeasibility / HSD certificate contract gate (full arithmetic family).
using SDPX
using Test

const _MULTIFLOATS = try
    @eval import MultiFloats
    true
catch
    false
end

function _primal_infeasible(::Type{T}) where {T}
    return SDPX.ingest(
        T[0],
        [reshape(T[1], 1, 1, 1), reshape(T[-1], 1, 1, 1)],
        [reshape(T[1], 1, 1), reshape(T[0], 1, 1)],
        zeros(T, 1, 0), T[];
        verbosity=0,
    )
end

function _unbounded(::Type{T}) where {T}
    return SDPX.ingest(
        T[-1],
        [reshape(T[1], 1, 1, 1)],
        [reshape(T[0], 1, 1)],
        zeros(T, 1, 0), T[];
        verbosity=0,
    )
end

function _ray_result(::Type{T}, problem; x=zeros(T, problem.dims.m),
    y=zeros(T, problem.dims.n),
    Y=[zeros(T, d, d) for d in problem.dims.k]) where {T}
    X = [zeros(T, d, d) for d in problem.dims.k]
    return SDPX.SDPResult{T}(SDPX.IterLimit, "ray candidate", x, X, y, Y,
        zero(T), zero(T), T(Inf), T(Inf), T(Inf), 1, 0, 0, nothing)
end

function _assert_infeasible_certificate(::Type{T}) where {T}
    options = SDPX.SolverOptions{T}(
        ϵ_gap=T(1e-12), ϵ_primal=T(1e-12), ϵ_dual=T(1e-12), verbosity=0,
    )
    pinf = _primal_infeasible(T)
    dual_ray = _ray_result(T, pinf; Y=[reshape(T[one(T)],1,1), reshape(T[one(T)],1,1)])
    diag = SDPX.infeasibility_diagnosis(pinf, dual_ray, options)
    @test diag.kind === :primal_infeasible
    @test diag.primal_infeasibility.valid
    promoted, _, message = SDPX.certify_optimize_infeasibility(pinf, dual_ray, options)
    @test promoted.status === SDPX.PrimalInfeasible
    @test message !== nothing
    cert = SDPX.result_certificate(pinf, promoted, options)
    @test cert.valid
    @test cert.kind === :primal_infeasibility

    unb = _unbounded(T)
    primal_ray = _ray_result(T, unb; x=T[1])
    diag2 = SDPX.infeasibility_diagnosis(unb, primal_ray, options)
    @test diag2.kind === :dual_infeasible_or_primal_unbounded
    @test diag2.dual_infeasibility.valid
    promoted2, _, message2 = SDPX.certify_optimize_infeasibility(unb, primal_ray, options)
    @test promoted2.status === SDPX.DualInfeasible
    @test message2 !== nothing
    cert2 = SDPX.result_certificate(unb, promoted2, options)
    @test cert2.valid
    @test cert2.kind === :dual_infeasibility
    return nothing
end

@testset "Phase 6 infeasibility certificate contract" begin
    for T in (Float64, BigFloat)
        @testset "$T" begin
            setprecision(BigFloat, 256) do
                _assert_infeasible_certificate(T)
            end
        end
    end
    if _MULTIFLOATS
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4)
            @testset "$T" begin
                _assert_infeasible_certificate(T)
            end
        end
    end
end