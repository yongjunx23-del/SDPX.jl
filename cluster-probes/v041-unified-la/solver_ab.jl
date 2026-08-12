#=
    Small dense-KKT and full-solve solver-level A/B: generic vs MFLA.

    The KKT direction verification always runs.  The same SPD system and
    RHS are generated once from a fixed seed and both arms reset from
    those immutable inputs.  The generic arm uses
    `LinearAlgebra.cholesky`/`ldiv!`; the MFLA arm uses the explicit
    provider (SDPX dispatch when the planner accepts it, otherwise the
    pinned upstream kernels directly).  Real residuals are reported; no
    status or certificate is fabricated for this kernel-level check.

    The full SDPX solve A/B is capability-gated.  Once the core exposes
    the expert `SolverOptions.linear_algebra_backend` option, this probe
    runs the same small dense SDP through the public `solve` frontend with
    `linear_algebra_backend=:standard` and `:multifloat`, and validates
    status, objective, gap, primal/dual residual, original-coordinate
    certificate, and iterations from the real solve results.  Until that
    option is landed, it prints an explicit `full_solve=SKIPPED` marker;
    the PBS layer treats that as FAILED so a missing expert option can
    never be mistaken for a passed solver-level gate.
=#
using SDPX
using LinearAlgebra
using Random
using MultiFloats
using MultiFloatLinearAlgebra

const LA = SDPX.Experimental
const T = Float64x4

const KKT_SIZE = 24
const KKT_DELTA = T(16)
const RESIDUAL_TOL = 1.0e-12
const FULL_SOLVE_TOL = T(1e-12)

const MFLA_CONFIG = MultiFloatLinearAlgebra.KernelConfig(
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

function _kkt_verification(requested::Symbol, SPD0, rhs0)
    if requested === :standard
        factor = LinearAlgebra.cholesky(Symmetric(copy(SPD0), :L))
        solution = copy(rhs0)
        LinearAlgebra.ldiv!(factor, solution)
        route = :standard
    else
        backend = _backend(_plan(:multifloat))
        route = _multifloat_route(backend)
        factor = _mfla_factor!(backend, copy(SPD0), route)
        solution = copy(rhs0)
        _mfla_solve!(factor, solution, route)
    end
    residual = SPD0 * solution - rhs0
    max_abs_residual = maximum(abs, residual)
    max_relative_residual =
        max_abs_residual / max(maximum(abs, rhs0), one(T))
    reference = setprecision(BigFloat, 512) do
        Sb = BigFloat.(SPD0)
        rb = BigFloat.(rhs0)
        Sb \ rb
    end
    max_relative_error_vs_reference = setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for index in eachindex(solution, reference)
            numerator = max(
                numerator,
                abs(BigFloat(solution[index]) - reference[index]),
            )
            denominator = max(denominator, abs(reference[index]))
        end
        numerator / max(denominator, BigFloat(1))
    end
    return (
        route=route,
        max_abs_residual=Float64(max_abs_residual),
        max_relative_residual=Float64(max_relative_residual),
        max_relative_error_vs_reference=Float64(
            max_relative_error_vs_reference,
        ),
        all_finite=all(isfinite, solution) && all(isfinite, residual),
    )
end

function _small_dense_sdp(::Type{T}) where {T}
    k = 3
    variables = k * (k + 1) ÷ 2
    c = zeros(T, variables)
    c[1] = -one(T)
    A = zeros(T, variables, k, k)
    A[1, 1, 1] = one(T)
    A[2, 2, 2] = one(T)
    A[3, 3, 3] = one(T)
    A[4, 1, 2] = one(T)
    A[4, 2, 1] = one(T)
    A[5, 1, 3] = one(T)
    A[5, 3, 1] = one(T)
    A[6, 2, 3] = one(T)
    A[6, 3, 2] = one(T)
    C = [zeros(T, k, k)]
    B = zeros(T, variables, 1)
    B[1, 1] = one(T)
    B[2, 1] = one(T)
    B[3, 1] = one(T)
    b = T[3]
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

function _full_solve_verification(requested::Symbol, tolerance::T)
    prob = _small_dense_sdp(T)
    options = SolveOptions(;
        verbosity=0,
        diagnostics=true,
        timing=true,
        duality_gap_threshold=tolerance,
        primal_error_threshold=tolerance,
        dual_error_threshold=tolerance,
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
    verification.gap_rel <= Float64(tolerance) || error(
        "full solve for $requested gap too large: $(verification.gap_rel)",
    )
    verification.p_res <= Float64(tolerance) || error(
        "full solve for $requested primal residual too large: $(verification.p_res)",
    )
    verification.d_res <= Float64(tolerance) || error(
        "full solve for $requested dual residual too large: $(verification.d_res)",
    )
    verification.certificate_available || error(
        "full solve for $requested has no certificate",
    )
    verification.certificate_valid || error(
        "full solve for $requested has an invalid certificate",
    )
    return verification
end

function _expert_option_available(::Type{T}) where {T}
    return hasfield(SolverOptions{T}, :linear_algebra_backend)
end

function main()
    _runtime_identity_checks!(Threads.nthreads())

    Random.seed!(0xabcd)
    R0 = T.(randn(KKT_SIZE, KKT_SIZE))
    SPD0 = transpose(R0) * R0
    SPD0 = (SPD0 + transpose(SPD0)) / 2
    SPD0 = SPD0 + KKT_DELTA * Matrix{T}(I, KKT_SIZE, KKT_SIZE)
    rhs0 = T.(randn(KKT_SIZE))

    standard = _kkt_verification(:standard, SPD0, rhs0)
    multifloat = _kkt_verification(:multifloat, SPD0, rhs0)
    println(
        "SOLVER_AB kkt_standard=",
        join(
            ["$(key)=$(value)" for (key, value) in sort(collect(standard); by=first)],
            ",",
        ),
    )
    println(
        "SOLVER_AB kkt_multifloat=",
        join(
            ["$(key)=$(value)" for (key, value) in sort(collect(multifloat); by=first)],
            ",",
        ),
    )
    for (name, verification) in (("standard", standard), ("multifloat", multifloat))
        verification.all_finite || error("non-finite KKT direction for $name")
        verification.max_relative_residual <= RESIDUAL_TOL || error(
            "KKT residual too large for $name: $(verification.max_relative_residual)",
        )
        verification.max_relative_error_vs_reference <= RESIDUAL_TOL || error(
            "KKT reference error too large for $name: $(verification.max_relative_error_vs_reference)",
        )
    end
    println("SOLVER_AB kkt_verification=ok")

    if _expert_option_available(T)
        std_full = _full_solve_verification(:standard, FULL_SOLVE_TOL)
        mf_full = _full_solve_verification(:multifloat, FULL_SOLVE_TOL)
        println(
            "SOLVER_AB full_solve_standard=",
            join(
                ["$(key)=$(value)" for (key, value) in sort(collect(std_full); by=first)],
                ",",
            ),
        )
        println(
            "SOLVER_AB full_solve_multifloat=",
            join(
                ["$(key)=$(value)" for (key, value) in sort(collect(mf_full); by=first)],
                ",",
            ),
        )
        relative_objective =
            abs(std_full.p_obj - mf_full.p_obj) / max(abs(std_full.p_obj), 1.0)
        relative_objective <= 1.0e-10 || error(
            "full-solve objective mismatch: $(relative_objective)",
        )
        iterations_match = std_full.iterations == mf_full.iterations
        std_full.iterations > 0 || error("standard full solve has no iterations")
        mf_full.iterations > 0 || error("multifloat full solve has no iterations")
        println("SOLVER_AB iterations_match=$iterations_match")
        println("SOLVER_AB full_solve=ok")
    else
        println("SOLVER_AB expert_linear_algebra_backend_option=unavailable")
        println("SOLVER_AB full_solve=SKIPPED reason=expert_linear_algebra_backend_option_not_landed")
    end
    return 0
end

exit(main())
