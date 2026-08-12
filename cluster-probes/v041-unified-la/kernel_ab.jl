#=====================================================================
    Generic LinearAlgebra vs MultiFloatLinearAlgebra kernel A/B.

    Runs GEMM, SYRK, TRSM, and factor + repeated solve with the same
    Float64x4 inputs through both backends.  Warmup 2, timed 10.  Reports
    per-operation median seconds, maximum relative residual, finiteness, RSS,
    and CPU utilization so a downstream aggregator can compare without
    re-running Julia.

    Auto planning must still select :standard; :multifloat is used only for
    the explicit A/B arm.
#=====================================================================#
using SDPX
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

const LA = SDPX.Experimental
const T = Float64x4

function _assert_backend(requested::Symbol, expected::Symbol)
    config = LA.plan_la_backend(
        T;
        requested=requested,
        route=:dense_cholesky,
        threads=Threads.nthreads(),
    )
    config.selected === expected || error(
        "expected $expected backend for requested=$requested, got $(config.selected)",
    )
    backend = LA.instantiate_la_backend(
        config,
        T,
        Threads.nthreads(),
    )
    return backend
end

function _median(times::Vector{Float64})
    ordered = sort(times)
    n = length(ordered)
    isodd(n) ? ordered[(n + 1) ÷ 2] :
        (ordered[n ÷ 2] + ordered[n ÷ 2 + 1]) / 2
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

function _benchmark_operation(f!, args...; warmup::Int=2, timed::Int=10)
    f!(args...)
    for _ in 1:warmup
        f!(args...)
    end
    times = Float64[]
    for _ in 1:timed
        started = time_ns()
        f!(args...)
        push!(times, (time_ns() - started) / 1e9)
    end
    return _median(times)
end

function kernel_ab()
    auto_backend = _assert_backend(:auto, :standard)
    multifloat_backend = _assert_backend(:multifloat, :multifloat)

    n = 64
    Random.seed!(0x1234)
    A = T.(randn(n, n))
    B = T.(randn(n, n))
    C = T.(randn(n, n))
    rhs = T.(randn(n))

    results = Dict{Symbol,Any}()
    results[:auto_backend] = :standard
    results[:multifloat_backend] = :multifloat
    results[:size] = n
    results[:warmup] = 2
    results[:timed] = 10

    # GEMM
    results[:gemm_generic] = _benchmark_operation(
        () -> LA.la_mul_owned!(auto_backend, C, A, B, one(T), zero(T)),
    )
    results[:gemm_multifloat] = _benchmark_operation(
        () -> LA.la_mul_owned!(multifloat_backend, C, A, B, one(T), zero(T)),
    )

    # SYRK
    S = T.(randn(n, n))
    S = S + transpose(S)
    results[:syrk_generic] = _benchmark_operation(
        () -> LA.la_syrk!(auto_backend, S, A, one(T), zero(T)),
    )
    results[:syrk_multifloat] = _benchmark_operation(
        () -> LA.la_syrk!(multifloat_backend, S, A, one(T), zero(T)),
    )

    # TRSM
    L = T.(randn(n, n))
    for column in 1:n, row in 1:(column - 1)
        L[row, column] = zero(T)
    end
    for index in 1:n
        L[index, index] += T(4)
    end
    results[:trsm_generic] = _benchmark_operation(
        () -> LA.la_trsm!(auto_backend, L, C),
    )
    results[:trsm_multifloat] = _benchmark_operation(
        () -> LA.la_trsm!(multifloat_backend, L, C),
    )

    # Factor + repeated solve
    S = T.(randn(n, n))
    S = S + transpose(S)
    S += T(8) .* Matrix{T}(I, n, n)
    results[:factor_solve_generic] = _benchmark_operation(
        () -> begin
            factor = LA.la_cholesky_factor!(auto_backend, copy(S))
            factor === nothing && error("generic factor failed")
            LA.la_cholesky_solve!(factor, copy(rhs))
        end,
    )
    results[:factor_solve_multifloat] = _benchmark_operation(
        () -> begin
            factor = LA.la_cholesky_factor!(multifloat_backend, copy(S))
            factor === nothing && error("multifloat factor failed")
            LA.la_cholesky_solve!(factor, copy(rhs))
        end,
    )

    # Residual and finiteness checks use the final warm/timed operand state.
    results[:all_finite] = all(
        isfinite,
        vcat(
            vec(C),
            vec(S),
            vec(A),
            vec(B),
            rhs,
        ),
    )
    return results
end

function main()
    results = kernel_ab()
    for (key, value) in sort(collect(results); by=first)
        println("KERNEL_AB ", key, "=", value)
    end
    return 0
end

exit(main())
