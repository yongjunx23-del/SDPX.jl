#=
    Small dense BigFloat standard-vs-legacy identity smoke for the bundled
    legacy LA provider milestone.

    Cluster-only, Julia/solver threads 4, BLAS/OMP/MKL/BLIS threads 1.
    Working precision is 256 bits and numerical references are recomputed at
    512 bits.  The smoke is intentionally small: one SPD factor/solve identity
    and one real dense SDP solve through each backend.  It does not benchmark
    and does not duplicate the full regression suite.

    Owned BigFloat storage uses the existing shared ownership helpers
    (`SDPX.alloc_zeros`, `SDPX.copy_owned!`).  Those helpers are not asserted
    as provider capabilities; see the migration document.

    Output contract (checked by the PBS runner):
      LEGACY_PROVIDER_AB full_solve_standard=ok
      LEGACY_PROVIDER_AB full_solve_legacy=ok
      LEGACY_PROVIDER_AB identity=ok
=#
using SDPX
using LinearAlgebra
using Random

const LA = SDPX.Experimental
const T = BigFloat

const WORKING_BITS = 256
const REFERENCE_BITS = 512
const N = 12
const RESIDUAL_TOL = BigFloat("1e-40")
const SOLVE_TOL = BigFloat("1e-30")
const EXPECTED_PROVIDER_SYMBOL = Symbol(
    get(ENV, "SDPX_LEGACY_PROVIDER_SYMBOL", "sdpx_legacy_la"),
)
const EXPECTED_LEGACY_OWNERSHIP = Symbol(
    get(ENV, "SDPX_LEGACY_OWNERSHIP", "owned_mutable_scalars"),
)

function _identity_checks!()
    Threads.nthreads() == 4 || error(
        "Julia thread mismatch: $(Threads.nthreads()) vs 4",
    )
    LinearAlgebra.BLAS.get_num_threads() == 1 || error(
        "BLAS thread mismatch: $(LinearAlgebra.BLAS.get_num_threads())",
    )
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
        get(ENV, variable, "") == "1" || error(
            "$variable must be 1, got $(repr(get(ENV, variable, "")))",
        )
    end
    println("RUNTIME_CONTRACT julia=4 plan=4 blas=1")
end

function _owned_copy(A)
    destination = SDPX.alloc_zeros(T, size(A)...)
    return SDPX.copy_owned!(destination, A)
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

function _spd_owned(n::Int, delta::BigFloat)
    R = _random_owned(n, n)
    product = transpose(R) * R
    S = SDPX.alloc_zeros(T, n, n)
    for column in 1:n, row in 1:n
        symmetric = (product[row, column] + product[column, row]) / 2
        S[row, column] = symmetric + (row == column ? delta : BigFloat(0))
    end
    return S
end

function _backend(requested::Symbol)
    plan = LA.plan_la_backend(
        T;
        requested=requested,
        route=:dense_cholesky,
        threads=4,
    )
    return LA.instantiate_la_backend(plan, T, 4)
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

function _factor_solve(requested::Symbol, SPD0, rhs0)
    backend = _backend(requested)
    A = _owned_copy(SPD0)
    rhs = _owned_copy(rhs0)
    factor = LA.la_cholesky_factor!(backend, A)
    factor === nothing && error("$requested factor failed")
    LA.la_cholesky_solve!(factor, rhs)
    return rhs
end

function _factor_solve_reference(SPD0, rhs0)
    return setprecision(BigFloat, REFERENCE_BITS) do
        return BigFloat.(SPD0) \ BigFloat.(rhs0)
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
    certificate = SDPX.result_certificate(prob, result, resolved.core)
    result.diagnostics === nothing && error("diagnostics disabled")
    selected = result.diagnostics.selected_algorithms
    verification = (
        status=result.status,
        p_obj=Float64(result.pObj),
        d_obj=Float64(result.dObj),
        gap_rel=Float64(result.gap_rel),
        p_res=Float64(result.p_res),
        d_res=Float64(result.d_res),
        iterations=result.iterations,
        certificate_valid=certificate.valid,
        executed_la_backend=get(selected, :la_backend, :not_executed),
        planned_la_backend=get(selected, :planned_la_backend, :not_executed),
        la_provider=get(selected, :la_provider, :not_recorded),
        la_executed_provider=get(
            selected,
            :la_executed_provider,
            :not_recorded,
        ),
        la_ownership=get(selected, :la_ownership, :not_recorded),
        la_executed_ownership=get(
            selected,
            :la_executed_ownership,
            :not_recorded,
        ),
        planned_la_fallback_reason=get(
            selected,
            :planned_la_fallback_reason,
            :not_recorded,
        ),
        runtime_la_fallback_reason=get(
            selected,
            :la_fallback_reason,
            :not_recorded,
        ),
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
    verification.certificate_valid || error(
        "full solve for $requested has an invalid certificate",
    )
    verification.planned_la_backend == requested || error(
        "planned backend for $requested was $(verification.planned_la_backend)",
    )
    verification.executed_la_backend == requested || error(
        "executed backend for $requested was $(verification.executed_la_backend)",
    )
    expected_planned_fallback = requested === :legacy ? :requested_legacy : :none
    verification.planned_la_fallback_reason === expected_planned_fallback || error(
        "planned LA fallback for $requested was $(verification.planned_la_fallback_reason)",
    )
    allowed_runtime_fallbacks = requested === :legacy ?
        (:none, :requested_legacy) : (:none,)
    verification.runtime_la_fallback_reason in allowed_runtime_fallbacks || error(
        "full solve for $requested had an unauthorized runtime LA fallback: " *
        "$(verification.runtime_la_fallback_reason)",
    )
    if requested === :legacy
        verification.la_provider == EXPECTED_PROVIDER_SYMBOL || error(
            "legacy plan provider $(verification.la_provider); " *
            "expected $(EXPECTED_PROVIDER_SYMBOL)",
        )
        verification.la_executed_provider == EXPECTED_PROVIDER_SYMBOL || error(
            "legacy executed provider $(verification.la_executed_provider); " *
            "expected $(EXPECTED_PROVIDER_SYMBOL)",
        )
        verification.la_ownership == EXPECTED_LEGACY_OWNERSHIP || error(
            "legacy ownership $(verification.la_ownership); " *
            "expected $(EXPECTED_LEGACY_OWNERSHIP)",
        )
        verification.la_executed_ownership == EXPECTED_LEGACY_OWNERSHIP || error(
            "legacy executed ownership $(verification.la_executed_ownership); " *
            "expected $(EXPECTED_LEGACY_OWNERSHIP)",
        )
    end
    return verification
end

function main()
    return setprecision(BigFloat, WORKING_BITS) do
        _identity_checks!()
        Random.seed!(0x3141)
        SPD0 = _spd_owned(N, BigFloat(8))
        rhs0 = _random_owned_vector(N)

        standard_solution = _factor_solve(:standard, SPD0, rhs0)
        legacy_solution = _factor_solve(:legacy, SPD0, rhs0)
        reference = _factor_solve_reference(SPD0, rhs0)
        standard_relative = _max_relative_error(standard_solution, reference)
        legacy_relative = _max_relative_error(legacy_solution, reference)
        cross_relative = _max_relative_error(legacy_solution, standard_solution)
        standard_relative <= Float64(RESIDUAL_TOL) || error(
            "standard factor-solve residual $(standard_relative)",
        )
        legacy_relative <= Float64(RESIDUAL_TOL) || error(
            "legacy factor-solve residual $(legacy_relative)",
        )
        cross_relative <= Float64(RESIDUAL_TOL) || error(
            "standard-vs-legacy factor-solve mismatch $(cross_relative)",
        )
        println(
            "LEGACY_PROVIDER_AB factor_solve ",
            "standard=", standard_relative,
            " legacy=", legacy_relative,
            " cross=", cross_relative,
        )

        standard_full = _full_solve(:standard)
        legacy_full = _full_solve(:legacy)
        println("LEGACY_PROVIDER_AB full_solve_standard=ok")
        println("LEGACY_PROVIDER_AB full_solve_legacy=ok")
        objective_delta = abs(standard_full.p_obj - legacy_full.p_obj) /
                          max(abs(standard_full.p_obj), 1.0)
        objective_delta <= 1.0e-10 || error(
            "full-solve objective mismatch: $(objective_delta)",
        )
        standard_full.iterations > 0 || error("standard full solve has no iterations")
        legacy_full.iterations > 0 || error("legacy full solve has no iterations")
        println("LEGACY_PROVIDER_AB identity=ok")
        return 0
    end
end

exit(main())
