#=====================================================================#
#    Hard zero-allocation gate (one full predictor-corrector Newton
#    step, SDP route) across the arithmetic family.
#
#    Per the frozen spec (docs/design/CANONICAL_FORM.md §5) "0 Julia
#    bytes warm step" is measured over 10 consecutive WARM samples and
#    **all** samples must equal 0. There is NO `minimum` and NO
#    tolerance: a single non-zero sample fails the gate. BigFloat/MPFR
#    native memory is tracked separately from the Julia allocation
#    measured here and is NOT counted.
#
#    This is the aspirational hard gate (WORKPLAN PR6). The current
#    `newton_step!` hot loop still allocates, so it is expected to
#    report FAIL and exit non-zero until the zero-allocation work lands.
#    The ceiling-based `test/allocation_contract.jl` is the regression
#    gate that must stay green meanwhile.
#
#    Usage:
#      julia --project=... benchmark/allocation_zero_gate.jl
#=====================================================================#

include(joinpath(@__DIR__, "SDPXBenchmarkRegistry.jl"))
using SDPX
using LinearAlgebra

const _MULTIFLOATS = try
    @eval import MultiFloats
    true
catch
    false
end

# 10 consecutive warm samples; ALL must be 0 (no minimum, no tolerance).
const _ZERO_SAMPLES = 10
# A few calls before the measured samples to JIT the full hot path so the
# measured calls are steady-state.
const _JIT_WARMUP = 5

function _zero_gate_problem(::Type{T}) where {T}
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

"""
    _zero_alloc_samples(::Type{T}) -> Union{Nothing,Vector{Int}}

Return `_ZERO_SAMPLES` consecutive steady-state Julia-allocation samples of
one full `newton_step!`. Returns `nothing` when the warm start fails (caller
records that as a failed arithmetic rather than a zero gate).
"""
function _zero_alloc_samples(::Type{T}) where {T}
    prob = _zero_gate_problem(T)
    opts = SDPX.SolverOptions{T}(
        algorithm=:sdp, presolve=false, scaling=:none, verbosity=0,
        iter_max=200,
    )
    ws = SDPX.Workspace(prob; thread_count=1)
    init = SDPX._kkt_cold_start_initialization(ws, prob, opts)
    init.ok || return nothing
    x, X, y, Y, mu = init.x, init.X, init.y, init.Y, init.μ
    # JIT warm-up: run the step several times before measuring so the samples
    # below are steady-state, not compilation.
    for _ in 1:_JIT_WARMUP
        SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2)
    end
    return [
        @allocated(SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2))
        for _ in 1:_ZERO_SAMPLES
    ]
end

# Use `all(sample == 0)` — never `minimum`. One non-zero sample fails the gate.
_zero_ok(samples::Vector{Int}) = !isempty(samples) && all(==(0), samples)

function _run()
    arithmetics = Any[
        (:float64, Float64),
    ]
    if _MULTIFLOATS
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4)
            push!(arithmetics, (string(T), T))
        end
    end
    push!(arithmetics, (:bigfloat256, BigFloat))

    rows = NamedTuple[]
    any_failure = false
    for (label, T) in arithmetics
        result = try
            samples = T === BigFloat ?
                setprecision(BigFloat, 256) do
                    _zero_alloc_samples(T)
                end :
                _zero_alloc_samples(T)
            if samples === nothing
                (label=label, ok=false, all_zero=false, samples=Int[], reason="warm start failed")
            else
                (label=label, ok=_zero_ok(samples), samples=samples, reason="ok")
            end
        catch exception
            (label=label, ok=false, samples=Int[], reason=sprint(showerror, exception))
        end
        push!(rows, result)
        any_failure |= !result.ok
    end

    for row in rows
        println(rpad(string(row.label), 16),
                " all_zero=", row.ok,
                " samples=", row.samples === Int[] ? "n/a" : join(row.samples, ","),
                " (", row.reason, ")")
    end
    return any_failure
end

_any_failure = _run()
if _any_failure
    println("ZERO-ALLOC GATE FAILED: not all warm samples are 0 Julia bytes")
else
    println("ZERO-ALLOC GATE PASSED: all warm samples are 0 Julia bytes")
end
# A failed gate must exit non-zero — never swallow it.
exit(_any_failure ? 1 : 0)
