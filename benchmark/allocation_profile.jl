#=
# Per-iteration allocation profile for one full predictor-corrector
# Newton step (SDP route) across the arithmetic family.
#
# Warm-up: a first `newton_step!` call JIT-compiles the hot path (its
# allocations are discarded). Then `@allocated` around a second call
# reports the steady-state Julia heap allocation of one full iteration.
#
# BigFloat: the Julia allocation reported here excludes the MPFR-native
# heap the factor/solve kernels allocate; that native behavior must be
# measured separately (Phase 4 note).
#
# Usage:
#   julia --project=... benchmark/allocation_profile.jl
=#

include(joinpath(@__DIR__, "SDPXBenchmarkRegistry.jl"))
using .SDPXBenchmarkRegistry
using SDPX
using LinearAlgebra
import MultiFloats
import MultiFloatLinearAlgebra
import BigFloatLinearAlgebra

const _ARTITHMETICS = (
    (:float64, Float64, :auto),
    (:float64x2, MultiFloats.Float64x2, :auto),
    (:float64x3, MultiFloats.Float64x3, :auto),
    (:float64x4, MultiFloats.Float64x4, :multifloat),
    (:bigfloat256, BigFloat, :auto),
)

function _small_sdp_problem(::Type{T}) where {T}
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

function _profile_one_iteration(::Type{T}) where {T}
    prob = _small_sdp_problem(T)
    opts = SDPX.SolverOptions{T}(
        algorithm=:sdp, presolve=false, scaling=:none, verbosity=0,
        iter_max=200,
    )
    ws = SDPX.Workspace(prob; thread_count=1)
    init = SDPX._kkt_cold_start_initialization(ws, prob, opts)
    init.ok || return (ok=false, reason=string(init.reason))
    x, X, y, Y, mu = init.x, init.X, init.y, init.Y, init.μ
    SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=1)
    alloc = minimum(
        @allocated(SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2))
        for _ in 1:3
    )
    return (ok=true, alloc=alloc)
end

function _run()
    rows = Any[]
    for (label, T, provider) in _ARTITHMETICS
        try
            profile = T === BigFloat ?
                setprecision(BigFloat, 256) do
                    _profile_one_iteration(T)
                end :
                _profile_one_iteration(T)
            push!(rows, (arithmetic=label, provider=provider, ok=profile.ok,
                        alloc_bytes=profile.ok ? profile.alloc : missing))
        catch exception
            push!(rows, (arithmetic=label, provider=provider, ok=false,
                        alloc_bytes=missing, error=sprint(showerror, exception)))
        end
    end
    for row in rows
        println(rpad(string(row.arithmetic), 12), rpad(string(row.provider), 10),
                " ok=", row.ok, " allocB=",
                row.alloc_bytes === missing ? "error" : string(row.alloc_bytes))
    end
    return rows
end

rows = _run()
any_failure = any(row -> !row.ok, rows)
if any_failure
    println("ALLOCATION PROFILE: at least one arithmetic failed")
end
# A failure must exit non-zero — never swallow it.
exit(any_failure ? 1 : 0)