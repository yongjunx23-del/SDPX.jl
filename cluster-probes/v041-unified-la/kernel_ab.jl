#=====================================================================
    Generic LinearAlgebra vs MultiFloatLinearAlgebra kernel A/B.

    Deterministic microbenchmark for GEMM, SYRK, TRSM, and factor +
    repeated solve on the same Float64x4 inputs.

    Inputs are generated once from a fixed seed and kept as immutable
    base copies.  Every warmup/timed repetition resets each destination
    buffer from those bases, so both arms and all repetitions see the
    same input state.  TRSM in particular must never inherit the previous
    arm's solution.

    SYRK is benchmarked under an explicit lower-triangle contract: the
    authoritative triangle is the lower one and residuals are computed
    only over row >= column.

    The MFLA arm always uses the explicit provider.  When the SDPX
    planner accepts the pinned provider, it runs through the instantiated
    `MultiFloatLABackend` (`mfla_route=sdpx_provider`); otherwise it calls
    the pinned upstream MFLA kernels directly (`mfla_route=direct_upstream`)
    so the kernel A/B remains runnable while the planner API is not yet
    landed.  The route is recorded in the output.

    Auto planning is recorded for audit but never changes the explicit
    `:standard` / `:multifloat` arms.
#=====================================================================#
using SDPX
using LinearAlgebra
using Random
using MultiFloats
using MultiFloatLinearAlgebra

const LA = SDPX.Experimental
const T = Float64x4

const WARMUP = 2
const TIMED = 10
const N = 64
const ALPHA = T(0.75)
const BETA = T(-0.5)
const DELTA = T(8)
const RESIDUAL_TOL = 1.0e-12

const MFLA_CONFIG = MultiFloatLinearAlgebra.KernelConfig(
    thread_count=max(Threads.nthreads(), 1),
)
const MFLA_WORKSPACE = MultiFloatLinearAlgebra.GemmWorkspace(
    T;
    thread_count=max(Threads.nthreads(), 1),
)

function _runtime_identity_checks!(threads::Int)
    expected_root = realpath(ENV["SDPX_CANDIDATE"])
    expected_src = realpath(joinpath(expected_root, "src", "SDPX.jl"))
    actual_src = realpath(pathof(SDPX))
    actual_root = realpath(joinpath(dirname(actual_src), ".."))
    println("CANDIDATE_PATHOF ", actual_src)
    println("CANDIDATE_ROOT ", actual_root)
    actual_src == expected_src || error(
        "loaded SDPX from $(actual_src) (root $(actual_root)); expected $(expected_src)",
    )
    actual_root == expected_root || error("SDPX root mismatch")

    mfla_src = realpath(pathof(MultiFloatLinearAlgebra))
    mfla_root = realpath(joinpath(dirname(mfla_src), ".."))
    println("MFLA_PATHOF ", mfla_src)
    println("MFLA_ROOT ", mfla_root)
    expected_mfla_root = realpath(ENV["MFLA_CANDIDATE"])
    mfla_root == expected_mfla_root || error(
        "loaded MFLA from $(mfla_root); expected $(expected_mfla_root)",
    )

    Threads.nthreads() == threads || error(
        "Julia thread mismatch: $(Threads.nthreads()) vs $threads",
    )
    LinearAlgebra.BLAS.get_num_threads() == 1 || error(
        "BLAS thread mismatch: $(LinearAlgebra.BLAS.get_num_threads())",
    )
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
        get(ENV, var, "") == "1" || error(
            "$var must be 1, got $(repr(get(ENV, var, "")))",
        )
    end
    println("RUNTIME_CONTRACT julia=$threads plan=$threads blas=1")
end

function _plan(requested::Symbol)
    return LA.plan_la_backend(
        T;
        requested=requested,
        route=:dense_cholesky,
        threads=Threads.nthreads(),
    )
end

function _backend(config)
    return LA.instantiate_la_backend(config, T, Threads.nthreads())
end

function _multifloat_route(multifloat_backend)
    return multifloat_backend isa LA.MultiFloatLABackend ?
           :sdpx_provider : :direct_upstream
end

function _mfla_factor!(backend, A, route::Symbol)
    if route === :sdpx_provider
        provider = backend.provider
        hasproperty(provider, :cholesky_factor!) || error(
            "MFLA provider lacks cholesky_factor!",
        )
        factor = getproperty(provider, :cholesky_factor!)(A)
        factor === nothing && error("MFLA provider factor failed")
        return factor
    end
    factor = MultiFloatLinearAlgebra.cholesky!(
        A;
        check=false,
        config=MFLA_CONFIG,
    )
    issuccess(factor) || error("MFLA cholesky failed")
    return factor
end

function _mfla_solve!(factor, rhs, route::Symbol)
    if route === :sdpx_provider
        hasproperty(factor, :solve!) || error(
            "MFLA factor handle lacks solve!",
        )
        return getproperty(factor, :solve!)(rhs)
    end
    column = reshape(rhs, length(rhs), 1)
    MultiFloatLinearAlgebra.trsm!(
        column,
        factor.factors,
        one(T);
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=MFLA_CONFIG,
    )
    MultiFloatLinearAlgebra.trsm!(
        column,
        factor.factors,
        one(T);
        side=:left,
        uplo=:lower,
        trans=:T,
        diag=:nonunit,
        config=MFLA_CONFIG,
    )
    return rhs
end

function _mfla_gemm!(C, A, B, α, β, backend, route::Symbol)
    if route === :sdpx_provider
        return LA.la_mul_owned!(backend, C, A, B, α, β)
    end
    return MultiFloatLinearAlgebra.gemm!(
        C,
        A,
        B,
        α,
        β;
        config=MFLA_CONFIG,
        workspace=MFLA_WORKSPACE,
    )
end

function _mfla_syrk!(S, P, α, β, backend, route::Symbol)
    if route === :sdpx_provider
        return LA.la_syrk!(backend, S, P, α, β)
    end
    return MultiFloatLinearAlgebra.syrk!(S, P, α, β; config=MFLA_CONFIG)
end

function _mfla_trsm!(L, X, backend, route::Symbol)
    if route === :sdpx_provider
        return LA.la_trsm!(backend, L, X)
    end
    return MultiFloatLinearAlgebra.trsm!(
        X,
        L,
        one(T);
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=MFLA_CONFIG,
    )
end

function _median(times::Vector{Float64})
    ordered = sort(times)
    n = length(ordered)
    isodd(n) ? ordered[(n + 1) ÷ 2] :
        (ordered[n ÷ 2] + ordered[n ÷ 2 + 1]) / 2
end

function _benchmark(reset!, call!; warmup::Int=WARMUP, timed::Int=TIMED)
    reset!()
    call!()
    for _ in 1:warmup
        reset!()
        call!()
    end
    times = Float64[]
    result = nothing
    for _ in 1:timed
        reset!()
        started = time_ns()
        result = call!()
        push!(times, (time_ns() - started) / 1e9)
    end
    reset!()
    alloc = @allocated call!()
    reset!()
    result = call!()
    return (;
        median=_median(times),
        alloc_bytes=alloc,
        result=copy(result),
    )
end

function _gemm_reference(A0, B0, C0)
    return setprecision(BigFloat, 512) do
        Ab = BigFloat.(A0)
        Bb = BigFloat.(B0)
        Cb = BigFloat.(C0)
        return BigFloat(ALPHA) .* (Ab * Bb) .+ BigFloat(BETA) .* Cb
    end
end

function _syrk_reference(P0, S0)
    return setprecision(BigFloat, 512) do
        Pb = BigFloat.(P0)
        Sb = BigFloat.(S0)
        return BigFloat(ALPHA) .* (transpose(Pb) * Pb) .+
               BigFloat(BETA) .* Sb
    end
end

function _trsm_reference(L0, X0)
    return setprecision(BigFloat, 512) do
        Lb = BigFloat.(L0)
        Xb = BigFloat.(X0)
        return Lb \ Xb
    end
end

function _factor_solve_reference(SPD0, rhs0)
    return setprecision(BigFloat, 512) do
        Sb = BigFloat.(SPD0)
        rb = BigFloat.(rhs0)
        return Sb \ rb
    end
end

function _residual_metrics(actual, reference)
    return setprecision(BigFloat, 512) do
        max_abs = BigFloat(0)
        max_ref = BigFloat(0)
        for index in eachindex(actual, reference)
            diff = abs(BigFloat(actual[index]) - reference[index])
            max_abs = max(max_abs, diff)
            max_ref = max(max_ref, abs(reference[index]))
        end
        relative = max_abs / max(max_ref, BigFloat(1))
        return (max_abs=Float64(max_abs), max_relative=Float64(relative))
    end
end

function _lower_triangle_residual_metrics(actual, reference)
    return setprecision(BigFloat, 512) do
        max_abs = BigFloat(0)
        max_ref = BigFloat(0)
        n = size(actual, 1)
        for column in 1:n, row in column:n
            diff = abs(BigFloat(actual[row, column]) - reference[row, column])
            max_abs = max(max_abs, diff)
            max_ref = max(max_ref, abs(reference[row, column]))
        end
        relative = max_abs / max(max_ref, BigFloat(1))
        return (max_abs=Float64(max_abs), max_relative=Float64(relative))
    end
end

function _assert_finite(metrics, label)
    all(isfinite, (metrics.max_abs, metrics.max_relative)) || error(
        "non-finite residual for $label",
    )
    metrics.max_relative <= RESIDUAL_TOL || error(
        "residual degradation for $label: $(metrics.max_relative) > $RESIDUAL_TOL",
    )
    return metrics
end

function main()
    started = time()
    _runtime_identity_checks!(Threads.nthreads())
    threads = Threads.nthreads()

    auto_plan = _plan(:auto)
    standard_backend = _backend(_plan(:standard))
    multifloat_config = _plan(:multifloat)
    multifloat_backend = _backend(multifloat_config)
    mfla_route = _multifloat_route(multifloat_backend)
    mfla_route in (:sdpx_provider, :direct_upstream) || error(
        "unexpected MFLA route $(mfla_route)",
    )

    Random.seed!(0x1234)
    A0 = T.(randn(N, N))
    B0 = T.(randn(N, N))
    C0 = T.(randn(N, N))
    P0 = T.(randn(N, N))
    S0 = T.(randn(N, N))
    S0 = S0 + transpose(S0)
    L0 = T.(randn(N, N))
    for column in 1:N, row in 1:(column - 1)
        L0[row, column] = zero(T)
    end
    for index in 1:N
        L0[index, index] += T(4)
    end
    R0 = T.(randn(N, N))
    SPD0 = transpose(R0) * R0
    SPD0 = (SPD0 + transpose(SPD0)) / 2
    SPD0 = SPD0 + DELTA * Matrix{T}(I, N, N)
    rhs0 = T.(randn(N))
    X0 = T.(randn(N, N))

    C = copy(C0)
    S = copy(S0)
    X = copy(X0)
    x = copy(rhs0)

    gemm_reset!() = (copyto!(C, C0); C)
    gemm_generic!() = LA.la_mul_owned!(standard_backend, C, A0, B0, ALPHA, BETA)
    gemm_multifloat!() =
        _mfla_gemm!(C, A0, B0, ALPHA, BETA, multifloat_backend, mfla_route)
    gemm_generic = _benchmark(gemm_reset!, gemm_generic!)
    gemm_multifloat = _benchmark(gemm_reset!, gemm_multifloat!)

    syrk_reset!() = (copyto!(S, S0); S)
    syrk_generic!() = LA.la_syrk!(standard_backend, S, P0, ALPHA, BETA)
    syrk_multifloat!() =
        _mfla_syrk!(S, P0, ALPHA, BETA, multifloat_backend, mfla_route)
    syrk_generic = _benchmark(syrk_reset!, syrk_generic!)
    syrk_multifloat = _benchmark(syrk_reset!, syrk_multifloat!)

    trsm_reset!() = (copyto!(X, X0); X)
    trsm_generic!() = LA.la_trsm!(standard_backend, L0, X)
    trsm_multifloat!() =
        _mfla_trsm!(L0, X, multifloat_backend, mfla_route)
    trsm_generic = _benchmark(trsm_reset!, trsm_generic!)
    trsm_multifloat = _benchmark(trsm_reset!, trsm_multifloat!)

    factor_reset!() = (copyto!(x, rhs0); x)
    factor_generic!() = begin
        factor = LinearAlgebra.cholesky(Symmetric(copy(SPD0), :L))
        LinearAlgebra.ldiv!(factor, x)
        x
    end
    factor_multifloat!() = begin
        factor = _mfla_factor!(multifloat_backend, copy(SPD0), mfla_route)
        _mfla_solve!(factor, x, mfla_route)
        x
    end
    factor_generic = _benchmark(factor_reset!, factor_generic!)
    factor_multifloat = _benchmark(factor_reset!, factor_multifloat!)

    gemm_generic_residual = _assert_finite(
        _residual_metrics(gemm_generic.result, _gemm_reference(A0, B0, C0)),
        "gemm_generic",
    )
    gemm_multifloat_residual = _assert_finite(
        _residual_metrics(gemm_multifloat.result, _gemm_reference(A0, B0, C0)),
        "gemm_multifloat",
    )
    syrk_generic_residual = _assert_finite(
        _lower_triangle_residual_metrics(
            syrk_generic.result,
            _syrk_reference(P0, S0),
        ),
        "syrk_generic",
    )
    syrk_multifloat_residual = _assert_finite(
        _lower_triangle_residual_metrics(
            syrk_multifloat.result,
            _syrk_reference(P0, S0),
        ),
        "syrk_multifloat",
    )
    trsm_generic_residual = _assert_finite(
        _residual_metrics(trsm_generic.result, _trsm_reference(L0, X0)),
        "trsm_generic",
    )
    trsm_multifloat_residual = _assert_finite(
        _residual_metrics(trsm_multifloat.result, _trsm_reference(L0, X0)),
        "trsm_multifloat",
    )
    factor_generic_residual = _assert_finite(
        _residual_metrics(
            factor_generic.result,
            _factor_solve_reference(SPD0, rhs0),
        ),
        "factor_solve_generic",
    )
    factor_multifloat_residual = _assert_finite(
        _residual_metrics(
            factor_multifloat.result,
            _factor_solve_reference(SPD0, rhs0),
        ),
        "factor_solve_multifloat",
    )

    generic_finite = all(
        isfinite,
        vcat(
            vec(gemm_generic.result),
            vec(syrk_generic.result),
            vec(trsm_generic.result),
            factor_generic.result,
        ),
    )
    multifloat_finite = all(
        isfinite,
        vcat(
            vec(gemm_multifloat.result),
            vec(syrk_multifloat.result),
            vec(trsm_multifloat.result),
            factor_multifloat.result,
        ),
    )

    results = Dict{Symbol,Any}()
    results[:size] = N
    results[:warmup] = WARMUP
    results[:timed] = TIMED
    results[:threads] = threads
    results[:blas_threads] = LinearAlgebra.BLAS.get_num_threads()
    results[:auto_selected] = auto_plan.selected
    results[:auto_provider] = auto_plan.provider
    results[:multifloat_plan_selected] = multifloat_config.selected
    results[:multifloat_plan_fallback_reason] =
        multifloat_config.fallback_reason
    results[:mfla_route] = mfla_route
    results[:syrk_contract] = :lower_triangle

    for (label, run) in (
        ("gemm_generic", gemm_generic),
        ("gemm_multifloat", gemm_multifloat),
        ("syrk_generic", syrk_generic),
        ("syrk_multifloat", syrk_multifloat),
        ("trsm_generic", trsm_generic),
        ("trsm_multifloat", trsm_multifloat),
        ("factor_solve_generic", factor_generic),
        ("factor_solve_multifloat", factor_multifloat),
    )
        results[Symbol(label, "_median_s")] = run.median
        results[Symbol(label, "_alloc_bytes")] = run.alloc_bytes
    end
    for (label, metrics) in (
        ("gemm_generic", gemm_generic_residual),
        ("gemm_multifloat", gemm_multifloat_residual),
        ("syrk_generic", syrk_generic_residual),
        ("syrk_multifloat", syrk_multifloat_residual),
        ("trsm_generic", trsm_generic_residual),
        ("trsm_multifloat", trsm_multifloat_residual),
        ("factor_solve_generic", factor_generic_residual),
        ("factor_solve_multifloat", factor_multifloat_residual),
    )
        results[Symbol(label, "_max_abs_residual")] = metrics.max_abs
        results[Symbol(label, "_max_relative_residual")] =
            metrics.max_relative
    end

    results[:generic_all_finite] = generic_finite
    results[:multifloat_all_finite] = multifloat_finite
    results[:rss_bytes] = Sys.maxrss()
    results[:elapsed_wall_seconds] = time() - started

    for (key, value) in sort(collect(results); by=first)
        println("KERNEL_AB ", key, "=", value)
    end
    generic_finite || error("generic arm produced non-finite output")
    multifloat_finite || error("multifloat arm produced non-finite output")
    return 0
end

exit(main())
