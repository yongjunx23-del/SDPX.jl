# Phase 5 fixed-precision contract gate.
#
# With `working_precision_policy=:fixed` (the spec's default target) a solve
# must use one precision end-to-end: no precision ladder, no unauthorized
# fallback, a valid original-coordinate certificate, and for BigFloat the
# recorded working precision must equal the requested bits. This is a
# regression gate on the fixed-precision contract (diagnostics record the
# precision; no silent precision change).
using SDPX
using Test

const _HAVE_MULTIFLOATS = try
    @eval import MultiFloats
    true
catch
    false
end

function _fixed_precision_problem(T::Type)
    k = 3
    m = k * (k + 1) ÷ 2
    c = zeros(T, m); c[1] = -one(T)
    A = zeros(T, m, k, k)
    A[1,1,1]=one(T); A[2,2,2]=one(T); A[3,3,3]=one(T)
    A[4,1,2]=one(T); A[4,2,1]=one(T)
    A[5,1,3]=one(T); A[5,3,1]=one(T)
    A[6,2,3]=one(T); A[6,3,2]=one(T)
    B = zeros(T, m, 1); B[1,1]=one(T); B[2,1]=one(T); B[3,1]=one(T)
    return SDPX.ingest(c, [A], [zeros(T,k,k)], B, T[3]; T=T, sparse=false, verbosity=0)
end

function _assert_fixed_precision_contract(::Type{T}; bits::Union{Nothing,Int}=nothing) where {T}
    prob = _fixed_precision_problem(T)
    options = T === BigFloat ?
        SDPX.SolverOptions{T}(precision_bits=bits, working_precision_policy=:fixed,
            verbosity=0, diagnostics=true, iter_max=200) :
        SDPX.SolverOptions{T}(working_precision_policy=:fixed, verbosity=0,
            diagnostics=true, iter_max=200)
    result = SDPX.solve!(prob, options)
    @test result.status == SDPX.Optimal
    cert = SDPX.result_certificate(prob, result, options)
    @test cert.valid
    diag = result.diagnostics
    if T === BigFloat
        ladder = diag.precision_ladder
        @test ladder !== nothing
        @test ladder.plan.policy === :fixed
        @test ladder.plan.selected_bits == bits
        @test [rung.bits for rung in ladder.plan.rungs] == [bits]
        @test length(ladder.attempts) == 1
    else
        # Fixed-width arithmetic never runs a BigFloat ladder.
        @test diag.precision_ladder === nothing
    end
    # No unauthorized fallback events in any attempt.
    for attempt in diag.attempts
        @test all(event -> event.authorized, attempt.fallback_events)
    end
    return result
end

@testset "Phase 5 fixed-precision contract" begin
    @testset "Float64" begin
        _assert_fixed_precision_contract(Float64)
    end
    @testset "BigFloat256" begin
        setprecision(BigFloat, 256) do
            _assert_fixed_precision_contract(BigFloat; bits=256)
        end
    end
    if _HAVE_MULTIFLOATS
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x4)
            @testset "$T" begin
                _assert_fixed_precision_contract(T)
            end
        end
    end
end