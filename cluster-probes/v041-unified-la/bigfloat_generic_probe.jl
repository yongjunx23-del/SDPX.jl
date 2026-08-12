#=
    BigFloat generic-vs-legacy dense LA probe.

    Cluster-only, one Julia/solver thread and one BLAS thread.  Working
    precision is 256 bits; every numerical reference is recomputed at 512
    bits.  The probe compares the Standard generic backend
    (`linear_algebra_backend=:standard`) with the Legacy exact dense
    production kernels (`linear_algebra_backend=:legacy`) on the same owned
    BigFloat inputs.

    Owned BigFloat inputs are constructed only through SDPX ownership-aware
    helpers (`SDPX.alloc_zeros` and `SDPX.copy_owned!`) or elementwise
    assignment of freshly allocated BigFloat values.  Shallow copies are
    never used for destination slots.

    Checks:
      - owned destination slots are not aliased to source slots;
      - source inputs are unchanged by every kernel;
      - results are deterministic across repeated owned-input runs;
      - all outputs are finite;
      - NaN, Inf and indefinite factor inputs fail closed for both backends;
      - GEMM/SYRK/TRSM/factor-solve residuals against 512-bit references;
      - a real small dense SDPX solve under `:standard` and `:legacy`,
        validating status, objective, gap, original-coordinate certificate,
        iterations, diagnostics and absence of fallback.

    Warmup is 1 and the number of timed repetitions is 3.  Every timed
    repetition resets the destination from immutable base inputs.
=#
using SDPX
using LinearAlgebra
using Random

const LA = SDPX.Experimental
const T = BigFloat

const WORKING_BITS = 256
const REFERENCE_BITS = 512
const WARMUP = 1
const TIMED = 3
const N = 24
const RESIDUAL_TOL = BigFloat("1e-40")
const SOLVE_TOL = BigFloat("1e-30")

function _identity_checks!()
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

    Threads.nthreads() == 1 || error(
        "Julia thread mismatch: $(Threads.nthreads()) vs 1",
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
    println("RUNTIME_CONTRACT julia=1 plan=1 blas=1")
end

function _plan(requested::Symbol)
    return LA.plan_la_backend(
        T;
        requested=requested,
        route=:dense_cholesky,
        threads=1,
    )
end

function _backend(requested::Symbol)
    return LA.instantiate_la_backend(_plan(requested), T, 1)
end

function _random_owned(rows::Int, cols::Int)
    A = SDPX.alloc_zeros(T, rows, cols)
    for index in eachindex(A)
        A[index] = BigFloat(randn())
    end
    return A
end

function _random_owned_vector(length::Int)
    A = SDPX.alloc_zeros(T, length)
    for index in eachindex(A)
        A[index] = BigFloat(randn())
    end
    return A
end

function _owned_copy(A)
    destination = SDPX.alloc_zeros(T, size(A)...)
    return SDPX.copy_owned!(destination, A)
end

function _spd_owned(n::Int, delta::BigFloat)
    R = _random_owned(n, n)
    product = transpose(R) * R
    S = SDPX.alloc_zeros(T, n, n)
    for column in 1:n, row in 1:n
        symmetric = (product[row, column] + product[column, row]) / 2
        diagonal = row == column ? delta : BigFloat(0)
        S[row, column] = symmetric + diagonal
    end
    return S
end

function _lower_owned(n::Int)
    L = _random_owned(n, n)
    for column in 1:n, row in 1:(column - 1)
        L[row, column] = BigFloat(0)
    end
    for index in 1:n
        L[index, index] = L[index, index] + BigFloat(4)
    end
    return L
end

function _median(times::Vector{Float64})
    ordered = sort(times)
    n = length(ordered)
    isodd(n) ? ordered[(n + 1) ÷ 2] :
        (ordered[n ÷ 2] + ordered[n ÷ 2 + 1]) / 2
end

function _benchmark(reset!, call!)
    reset!()
    call!()
    times = Float64[]
    result = nothing
    for _ in 1:TIMED
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
        result=_owned_copy(result),
    )
end

function _residual_metrics(actual, reference)
    return setprecision(BigFloat, REFERENCE_BITS) do
        max_abs = BigFloat(0)
        max_ref = BigFloat(0)
        for index in eachindex(actual, reference)
            diff = abs(BigFloat(actual[index]) - BigFloat(reference[index]))
            max_abs = max(max_abs, diff)
            max_ref = max(max_ref, abs(BigFloat(reference[index])))
        end
        relative = max_abs / max(max_ref, BigFloat(1))
        return (max_abs=Float64(max_abs), max_relative=Float64(relative))
    end
end

function _max_relative_error(actual, reference)
    return setprecision(BigFloat, REFERENCE_BITS) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for index in eachindex(actual, reference)
            diff = abs(BigFloat(actual[index]) - BigFloat(reference[index]))
            numerator = max(numerator, diff)
            denominator = max(denominator, abs(BigFloat(reference[index])))
        end
        numerator / max(denominator, BigFloat(1))
    end
end

function _assert_residual(metrics, label)
    all(isfinite, (metrics.max_abs, metrics.max_relative)) || error(
        "non-finite residual for $label",
    )
    metrics.max_relative <= Float64(RESIDUAL_TOL) || error(
        "residual degradation for $label: $(metrics.max_relative)",
    )
    return metrics
end

function _assert_owned_independent!(destination, source, label)
    destination_slots = collect(destination)
    source_slots = collect(source)
    for left in eachindex(destination_slots)
        for right in (left + 1):length(destination_slots)
            destination_slots[left] === destination_slots[right] && error(
                "$label destination slots $left and $right alias",
            )
        end
        for right in eachindex(source_slots)
            destination_slots[left] === source_slots[right] && error(
                "$label destination slot $left aliases source slot $right",
            )
        end
    end
    return true
end

function _assert_source_unchanged!(source, snapshot, label)
    for index in eachindex(source, snapshot)
        source[index] == snapshot[index] || error(
            "$label source changed at $index",
        )
    end
    return true
end

function _assert_deterministic(first, second, label)
    for index in eachindex(first, second)
        first[index] == second[index] || error(
            "$label is not deterministic at $index",
        )
    end
    return true
end

function _expect_factor_failure(backend, kind::Symbol)
    A = _spd_owned(N, BigFloat(8))
    if kind === :nan
        A[1, 1] = BigFloat(NaN)
    elseif kind === :inf
        A[1, 1] = BigFloat(Inf)
    elseif kind === :indefinite
        A[1, 1] = BigFloat(-1)
    else
        error("unknown failure kind $kind")
    end
    failed = false
    try
        factor = LA.la_cholesky_factor!(backend, A)
        failed = factor === nothing
    catch
        failed = true
    end
    failed || error("factor did not fail closed for $kind")
    return true
end

function _gemm_reference(A0, B0, C0, alpha, beta)
    return setprecision(BigFloat, REFERENCE_BITS) do
        Ab = BigFloat.(A0)
        Bb = BigFloat.(B0)
        Cb = BigFloat.(C0)
        return BigFloat(alpha) .* (Ab * Bb) .+ BigFloat(beta) .* Cb
    end
end

function _syrk_reference(P0, S0, alpha, beta)
    return setprecision(BigFloat, REFERENCE_BITS) do
        Pb = BigFloat.(P0)
        Sb = BigFloat.(S0)
        return BigFloat(alpha) .* (transpose(Pb) * Pb) .+
               BigFloat(beta) .* Sb
    end
end

function _trsm_reference(L0, X0)
    return setprecision(BigFloat, REFERENCE_BITS) do
        Lb = BigFloat.(L0)
        Xb = BigFloat.(X0)
        return Lb \ Xb
    end
end

function _factor_solve_reference(S0, rhs0)
    return setprecision(BigFloat, REFERENCE_BITS) do
        Sb = BigFloat.(S0)
        rb = BigFloat.(rhs0)
        return Sb \ rb
    end
end

function _small_dense_sdp()
    k = 3
    variables = k * (k + 1) ÷ 2
    c = SDPX.alloc_zeros(T, variables)
    c[1] = BigFloat(-1)
    A = SDPX.alloc_zeros(T, variables, k, k)
    A[1, 1, 1] = BigFloat(1)
    A[2, 2, 2] = BigFloat(1)
    A[3, 3, 3] = BigFloat(1)
    A[4, 1, 2] = BigFloat(1)
    A[4, 2, 1] = BigFloat(1)
    A[5, 1, 3] = BigFloat(1)
    A[5, 3, 1] = BigFloat(1)
    A[6, 2, 3] = BigFloat(1)
    A[6, 3, 2] = BigFloat(1)
    C = [SDPX.alloc_zeros(T, k, k)]
    B = SDPX.alloc_zeros(T, variables, 1)
    B[1, 1] = BigFloat(1)
    B[2, 1] = BigFloat(1)
    B[3, 1] = BigFloat(1)
    b = SDPX.alloc_zeros(T, 1)
    b[1] = BigFloat(3)
    return SDPX.ingest(
        c,
        [A],
        C,
        B,
        b;
        T=T,
        sparse=false,
        verbosity=0,
    )
end

function _full_solve(requested::Symbol)
    prob = _small_dense_sdp()
    options = SolveOptions(;
        verbosity=0,
        diagnostics=true,
        timing=true,
        duality_gap_threshold=BigFloat(SOLVE_TOL),
        primal_error_threshold=BigFloat(SOLVE_TOL),
        dual_error_threshold=BigFloat(SOLVE_TOL),
        linear_algebra_backend=requested,
    )
    result = SDPX.solve(prob, options)
    resolved = SDPX.resolve_solve_options(T, options)
    core_opts = resolved.core
    result.diagnostics === nothing && error("diagnostics disabled")
    selected = result.diagnostics.selected_algorithms
    certificate = SDPX.result_certificate(prob, result, core_opts)
    verification = (
        status=result.status,
        p_obj=Float64(result.pObj),
        d_obj=Float64(result.dObj),
        gap_rel=Float64(result.gap_rel),
        p_res=Float64(result.p_res),
        d_res=Float64(result.d_res),
        iterations=result.iterations,
        restarts=result.restarts,
        regularizations=result.regularizations,
        certificate_available=certificate.valid,
        certificate_valid=certificate.valid,
        executed_la_backend=get(selected, :la_backend, :not_executed),
        planned_la_backend=get(selected, :planned_la_backend, :not_executed),
        fallback_reason=get(selected, :la_fallback_reason, :not_recorded),
        all_finite=all(
            isfinite,
            (result.pObj, result.dObj, result.gap_rel, result.p_res, result.d_res),
        ),
    )
    verification.status == SDPX.Optimal || error(
        "full solve for $requested is not Optimal: $(verification.status)",
    )
    verification.all_finite || error(
        "full solve for $requested produced non-finite values",
    )
    verification.gap_rel <= Float64(SOLVE_TOL) || error(
        "full solve for $requested gap too large: $(verification.gap_rel)",
    )
    verification.p_res <= Float64(SOLVE_TOL) || error(
        "full solve for $requested primal residual too large: $(verification.p_res)",
    )
    verification.d_res <= Float64(SOLVE_TOL) || error(
        "full solve for $requested dual residual too large: $(verification.d_res)",
    )
    verification.certificate_available || error(
        "full solve for $requested has no certificate",
    )
    verification.certificate_valid || error(
        "full solve for $requested has an invalid certificate",
    )
    verification.planned_la_backend == requested || error(
        "planned backend for $requested was $(verification.planned_la_backend)",
    )
    verification.fallback_reason === :none || error(
        "full solve for $requested fell back: $(verification.fallback_reason)",
    )
    return verification
end

function main()
    return setprecision(BigFloat, WORKING_BITS) do
        started = time()
        _identity_checks!()
        Random.seed!(0x2a17)

        standard_backend = _backend(:standard)
        legacy_backend = _backend(:legacy)

        alpha = BigFloat("0.75")
        beta = BigFloat("-0.5")

        A0 = _random_owned(N, N)
        B0 = _random_owned(N, N)
        C0 = _random_owned(N, N)
        P0 = _random_owned(N, N)
        S0 = _random_owned(N, N)
        for column in 1:N, row in 1:(column - 1)
            S0[row, column] = S0[column, row]
        end
        L0 = _lower_owned(N)
        SPD0 = _spd_owned(N, BigFloat(8))
        rhs0 = _random_owned_vector(N)
        X0 = _random_owned(N, N)

        A_snapshot = _owned_copy(A0)
        B_snapshot = _owned_copy(B0)
        C_snapshot = _owned_copy(C0)
        P_snapshot = _owned_copy(P0)
        S_snapshot = _owned_copy(S0)
        L_snapshot = _owned_copy(L0)
        X_snapshot = _owned_copy(X0)
        rhs_snapshot = _owned_copy(rhs0)
        SPD_snapshot = _owned_copy(SPD0)

        C = _owned_copy(C0)
        S = _owned_copy(S0)
        X = _owned_copy(X0)
        rhs = _owned_copy(rhs0)
        Awork = _owned_copy(SPD0)

        _assert_owned_independent!(C, C0, "gemm destination")
        _assert_owned_independent!(S, S0, "syrk destination")
        _assert_owned_independent!(X, X0, "trsm destination")
        _assert_owned_independent!(rhs, rhs0, "factor-solve rhs")

        gemm_reset!() = (SDPX.copy_owned!(C, C0); C)
        gemm_standard!() =
            LA.la_mul_owned!(standard_backend, C, A0, B0, alpha, beta)
        gemm_legacy!() =
            LA.la_mul_owned!(legacy_backend, C, A0, B0, alpha, beta)
        gemm_standard = _benchmark(gemm_reset!, gemm_standard!)
        gemm_legacy = _benchmark(gemm_reset!, gemm_legacy!)

        syrk_reset!() = (SDPX.copy_owned!(S, S0); S)
        syrk_standard!() =
            LA.la_syrk!(standard_backend, S, P0, alpha, beta)
        syrk_legacy!() =
            LA.la_syrk!(legacy_backend, S, P0, alpha, beta)
        syrk_standard = _benchmark(syrk_reset!, syrk_standard!)
        syrk_legacy = _benchmark(syrk_reset!, syrk_legacy!)

        trsm_reset!() = (SDPX.copy_owned!(X, X0); X)
        trsm_standard!() = LA.la_trsm!(standard_backend, L0, X)
        trsm_legacy!() = LA.la_trsm!(legacy_backend, L0, X)
        trsm_standard = _benchmark(trsm_reset!, trsm_standard!)
        trsm_legacy = _benchmark(trsm_reset!, trsm_legacy!)

        factor_reset!() = begin
            SDPX.copy_owned!(Awork, SPD0)
            SDPX.copy_owned!(rhs, rhs0)
            rhs
        end
        factor_standard!() = begin
            factor = LA.la_cholesky_factor!(standard_backend, Awork)
            factor === nothing && error("standard factor failed")
            LA.la_cholesky_solve!(factor, rhs)
            rhs
        end
        factor_legacy!() = begin
            factor = LA.la_cholesky_factor!(legacy_backend, Awork)
            factor === nothing && error("legacy factor failed")
            LA.la_cholesky_solve!(factor, rhs)
            rhs
        end
        factor_standard = _benchmark(factor_reset!, factor_standard!)
        factor_legacy = _benchmark(factor_reset!, factor_legacy!)

        _assert_source_unchanged!(A0, A_snapshot, "gemm left source")
        _assert_source_unchanged!(B0, B_snapshot, "gemm right source")
        _assert_source_unchanged!(C0, C_snapshot, "gemm source")
        _assert_source_unchanged!(P0, P_snapshot, "syrk panel source")
        _assert_source_unchanged!(S0, S_snapshot, "syrk source")
        _assert_source_unchanged!(L0, L_snapshot, "trsm factor source")
        _assert_source_unchanged!(X0, X_snapshot, "trsm source")
        _assert_source_unchanged!(rhs0, rhs_snapshot, "rhs source")
        _assert_source_unchanged!(SPD0, SPD_snapshot, "SPD source")

        gemm_standard_twice = _benchmark(gemm_reset!, gemm_standard!)
        _assert_deterministic(
            gemm_standard.result,
            gemm_standard_twice.result,
            "gemm standard",
        )
        gemm_legacy_twice = _benchmark(gemm_reset!, gemm_legacy!)
        _assert_deterministic(
            gemm_legacy.result,
            gemm_legacy_twice.result,
            "gemm legacy",
        )

        _expect_factor_failure(standard_backend, :nan)
        _expect_factor_failure(standard_backend, :inf)
        _expect_factor_failure(standard_backend, :indefinite)
        _expect_factor_failure(legacy_backend, :nan)
        _expect_factor_failure(legacy_backend, :inf)
        _expect_factor_failure(legacy_backend, :indefinite)

        gemm_standard_residual = _assert_residual(
            _residual_metrics(
                gemm_standard.result,
                _gemm_reference(A0, B0, C0, alpha, beta),
            ),
            "gemm standard",
        )
        gemm_legacy_residual = _assert_residual(
            _residual_metrics(
                gemm_legacy.result,
                _gemm_reference(A0, B0, C0, alpha, beta),
            ),
            "gemm legacy",
        )
        syrk_standard_residual = _assert_residual(
            _residual_metrics(
                syrk_standard.result,
                _syrk_reference(P0, S0, alpha, beta),
            ),
            "syrk standard",
        )
        syrk_legacy_residual = _assert_residual(
            _residual_metrics(
                syrk_legacy.result,
                _syrk_reference(P0, S0, alpha, beta),
            ),
            "syrk legacy",
        )
        trsm_standard_residual = _assert_residual(
            _residual_metrics(trsm_standard.result, _trsm_reference(L0, X0)),
            "trsm standard",
        )
        trsm_legacy_residual = _assert_residual(
            _residual_metrics(trsm_legacy.result, _trsm_reference(L0, X0)),
            "trsm legacy",
        )
        factor_standard_residual = _assert_residual(
            _residual_metrics(
                factor_standard.result,
                _factor_solve_reference(SPD0, rhs0),
            ),
            "factor-solve standard",
        )
        factor_legacy_residual = _assert_residual(
            _residual_metrics(
                factor_legacy.result,
                _factor_solve_reference(SPD0, rhs0),
            ),
            "factor-solve legacy",
        )

        results = Dict{Symbol,Any}()
        results[:precision_bits] = WORKING_BITS
        results[:reference_bits] = REFERENCE_BITS
        results[:warmup] = WARMUP
        results[:timed] = TIMED
        results[:threads] = 1
        results[:blas_threads] = LinearAlgebra.BLAS.get_num_threads()
        results[:auto_selected] = _plan(:auto).selected
        results[:auto_provider] = _plan(:auto).provider
        results[:auto_ownership] = _plan(:auto).ownership
        results[:standard_plan] = _plan(:standard).selected
        results[:standard_provider] = _plan(:standard).provider
        results[:standard_ownership] = _plan(:standard).ownership
        results[:legacy_plan] = _plan(:legacy).selected

        for (label, run) in (
            ("gemm_standard", gemm_standard),
            ("gemm_legacy", gemm_legacy),
            ("syrk_standard", syrk_standard),
            ("syrk_legacy", syrk_legacy),
            ("trsm_standard", trsm_standard),
            ("trsm_legacy", trsm_legacy),
            ("factor_solve_standard", factor_standard),
            ("factor_solve_legacy", factor_legacy),
        )
            results[Symbol(label, "_median_s")] = run.median
            results[Symbol(label, "_alloc_bytes")] = run.alloc_bytes
        end
        for (label, metrics) in (
            ("gemm_standard", gemm_standard_residual),
            ("gemm_legacy", gemm_legacy_residual),
            ("syrk_standard", syrk_standard_residual),
            ("syrk_legacy", syrk_legacy_residual),
            ("trsm_standard", trsm_standard_residual),
            ("trsm_legacy", trsm_legacy_residual),
            ("factor_solve_standard", factor_standard_residual),
            ("factor_solve_legacy", factor_legacy_residual),
        )
            results[Symbol(label, "_max_abs_residual")] = metrics.max_abs
            results[Symbol(label, "_max_relative_residual")] =
                metrics.max_relative
        end

        results[:owned_slots_independent] = true
        results[:source_unchanged] = true
        results[:deterministic] = true
        results[:fail_closed] = true
        results[:rss_bytes] = Sys.maxrss()
        results[:elapsed_wall_seconds] = time() - started

        for (key, value) in sort(collect(results); by=first)
            println("BIGFLOAT_AB ", key, "=", value)
        end

        std_full = _full_solve(:standard)
        legacy_full = _full_solve(:legacy)
        println(
            "BIGFLOAT_AB full_solve_standard=",
            join(
                ["$(key)=$(value)" for (key, value) in sort(collect(std_full); by=first)],
                ",",
            ),
        )
        println(
            "BIGFLOAT_AB full_solve_legacy=",
            join(
                ["$(key)=$(value)" for (key, value) in sort(collect(legacy_full); by=first)],
                ",",
            ),
        )
        relative_objective =
            abs(std_full.p_obj - legacy_full.p_obj) / max(abs(std_full.p_obj), 1.0)
        relative_objective <= 1.0e-10 || error(
            "full-solve objective mismatch: $(relative_objective)",
        )
        std_full.iterations > 0 || error("standard full solve has no iterations")
        legacy_full.iterations > 0 || error("legacy full solve has no iterations")
        println("BIGFLOAT_AB full_solve=ok")
        return 0
    end
end

exit(main())
