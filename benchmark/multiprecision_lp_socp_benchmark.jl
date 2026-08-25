# Multi-precision LP/SOCP benchmark (Phase 1/6 coverage).
using SDPX
using Printf

const _MF = try
    @eval import MultiFloats
    true
catch
    false
end

function _lp_problem(::Type{T}) where {T}
    A = zeros(T, 1, 1, 1); A[1, 1, 1] = one(T)
    C = reshape([-one(T)], 1, 1)
    return SDPX.ingest(T[1], [A], [C], zeros(T, 1, 0), T[]; verbosity=0)
end

function _soc_problem(::Type{T}) where {T}
    A = zeros(T, 2, 2, 2); A[1, 1, 1] = one(T); A[2, 2, 2] = one(T)
    return SDPX.ingest(T[1, 1], [A], [zeros(T, 2, 2)], zeros(T, 2, 0), T[]; verbosity=0)
end

function _solve(prob, ::Type{T}) where {T}
    opts = SDPX.SolverOptions{T}(verbosity=0, ϵ_gap=T(1e-10), ϵ_primal=T(1e-10), ϵ_dual=T(1e-10))
    result = SDPX.solve!(prob, opts)
    return result.status, result.pObj
end

function _run()
    arith = vcat([Float64, BigFloat], _MF ? [MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4] : Type[])
    println("=== LP (1x1 block, opt 1.0) ===")
    for AT in arith
        AT === BigFloat && setprecision(BigFloat, 256)
        st, po = _solve(_lp_problem(AT), AT)
        println(rpad(string(AT), 24), " status=", st, " pobj=", po)
    end
    println("=== SOCP-style (2x2 block, opt 0.0) ===")
    for AT in arith
        AT === BigFloat && setprecision(BigFloat, 256)
        st, po = _solve(_soc_problem(AT), AT)
        println(rpad(string(AT), 24), " status=", st, " pobj=", po)
    end
end

_run()
