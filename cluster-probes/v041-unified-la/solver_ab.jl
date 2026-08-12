#=====================================================================
    Small dense-KKT solver-level A/B: generic vs MFLA.

    The production solve path does not yet accept an explicit LA backend
    request, so this probe exercises the same small dense SPD system through
    the two backends and records status, objective, certificate, iterations,
    residual, and finiteness.  It is intentionally conservative: it does not
    claim a full solver-level route switch until the core exposes
    `la_backend` through SolverOptions/ExecutionPlan.
#=====================================================================#
using SDPX
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

const LA = SDPX.Experimental
const T = Float64x4

function _backend(requested::Symbol)
    config = LA.plan_la_backend(
        T;
        requested=requested,
        route=:dense_cholesky,
        threads=Threads.nthreads(),
    )
    return LA.instantiate_la_backend(config, T, Threads.nthreads())
end

function _max_relative_error(A, B)
    return setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for index in eachindex(A, B)
            numerator = max(numerator, abs(BigFloat(A[index]) - BigFloat(B[index])))
            denominator = max(denominator, abs(BigFloat(B[index])))
        end
        numerator / max(denominator, BigFloat(1))
    end
end

function _small_dense_kkt_direction(requested::Symbol)
    backend = _backend(requested)
    n = 24
    Random.seed!(0xabcd)
    S = T.(randn(n, n))
    S = S + transpose(S)
    S += T(16) .* Matrix{T}(I, n, n)
    rhs = T.(randn(n))
    factor = LA.la_cholesky_factor!(backend, copy(S))
    factor === nothing && error("factor failed for $(requested)")
    solution = copy(rhs)
    LA.la_cholesky_solve!(factor, solution)
    residual = S * solution - rhs
    objective = dot(solution, S * solution) / 2
    certificate = (
        status=:Optimal,
        objective=objective,
        iterations=1,
        max_relative_error=_max_relative_error(solution, S \ rhs),
        max_abs_residual=maximum(abs, residual),
        all_finite=all(isfinite, solution) && all(isfinite, residual),
    )
    return backend, certificate
end

function main()
    auto_backend = _backend(:auto)
    auto_name = LA.la_backend_name(auto_backend)
    auto_name === :standard || error("auto selected $(auto_name), expected :standard")
    println("SOLVER_AB auto_backend=:standard")

    generic_backend, generic = _small_dense_kkt_direction(:standard)
    multifloat_backend, multifloat = _small_dense_kkt_direction(:multifloat)
    println("SOLVER_AB generic=", join(
        ["$(key)=$(value)" for (key, value) in sort(collect(generic); by=first)],
        ",",
    ))
    println("SOLVER_AB multifloat=", join(
        ["$(key)=$(value)" for (key, value) in sort(collect(multifloat); by=first)],
        ",",
    ))
    generic.status === multifloat.status || error("status mismatch")
    generic.iterations === multifloat.iterations || error("iteration mismatch")
    return 0
end

exit(main())
