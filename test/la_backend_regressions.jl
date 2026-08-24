using MultiFloats: Float64x4
using LinearAlgebra

# Provider-neutral contract doubles.  The real MFLA payload lives only behind
# the optional extension; these types prove that MultiFloat core dispatch is
# a fixed semantic protocol and cannot discover operations dynamically.
struct _ContractOnlyProvider end

SDPX.la_provider_capability_model(::_ContractOnlyProvider) =
    SDPX.LAProviderCapabilities(
        dot=true,
        mul_owned=true,
        syrk=true,
        cholesky=true,
        factor_solve=true,
    )

SDPX.la_mfla_dot(::_ContractOnlyProvider, x, y) = sum(x .* y)

function SDPX.la_mfla_syrk!(
    ::_ContractOnlyProvider,
    S::AbstractMatrix,
    P::AbstractMatrix,
    α,
    β,
)
    S .= α .* (transpose(P) * P) .+ β .* S
    return S
end

function SDPX.la_mfla_mul_owned!(
    ::_ContractOnlyProvider,
    C::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    α,
    β,
)
    C .= α .* (A * B) .+ β .* C
    return C
end

function SDPX.la_mfla_mul_owned!(
    ::_ContractOnlyProvider,
    C::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    C .= A * B
    return C
end

function SDPX.la_mfla_cholesky_factor!(
    ::_ContractOnlyProvider,
    A::AbstractMatrix,
)
    factor = LinearAlgebra.cholesky!(Symmetric(A, :L); check=false)
    LinearAlgebra.issuccess(factor) || return nothing
    return _ContractCholeskyPayload(factor)
end

struct _ContractCholeskyPayload{F<:LinearAlgebra.Cholesky}
    factor::F
end

SDPX.la_factor_provider_identity(::_ContractCholeskyPayload) =
    :multifloat_linear_algebra

SDPX.la_provider_factor_matrix(payload::_ContractCholeskyPayload) =
    payload.factor.factors

function SDPX.la_provider_factor_solve!(
    payload::_ContractCholeskyPayload,
    rhs,
)
    LinearAlgebra.ldiv!(payload.factor, rhs)
    return rhs
end

struct _UnsupportedProvider end

@testset "linear algebra backend planning" begin
    f64 = SDPX.plan_la_backend(Float64)
    @test f64.selected === :standard
    @test f64.provider === :blas_lapack
    @test f64.fallback_chain === (:rank_revealing_qr,)
    @test SDPX.plan_la_backend(
        Float64;
        equality_solver=:normal_equations,
    ).fallback_chain === ()
    @test SDPX.instantiate_la_backend(f64, Float64) isa
          SDPX.StandardLABackend

    bf = SDPX.plan_la_backend(BigFloat)
    @test bf.selected === :standard
    @test bf.provider === :generic_linear_algebra
    @test bf.fallback_chain === (:rank_revealing_qr,)
    @test SDPX.instantiate_la_backend(bf, BigFloat) isa
          SDPX.StandardLABackend

    fixed = SDPX.plan_la_backend(
        Float64x4;
        requested=:fixed_extended,
    )
    @test fixed.selected === :standard
    @test fixed.provider === :generic_linear_algebra
    auto_fixed = SDPX.plan_la_backend(Float64x4)
    @test auto_fixed.selected === :standard
    @test auto_fixed.provider === :generic_linear_algebra

    # Non-dense routes stay on the historical backend for automatic/legacy
    # planning.  An explicit migrated backend must fail closed instead of
    # being silently rewritten to legacy.
    block_arrow = SDPX.plan_la_backend(
        Float64;
        route=:block_arrow,
    )
    @test block_arrow isa SDPX.LABackendConfiguration
    @test block_arrow.selected === :legacy
    @test block_arrow.fallback_reason === :route_not_migrated
    @test block_arrow.fallback_chain === (:rank_revealing_qr,)
    @test SDPX.plan_la_backend(
        Float64;
        route=:block_arrow,
        equality_solver=:normal_equations,
    ).fallback_chain === ()
    @test SDPX.plan_la_backend(
        Float64;
        route=:block_arrow,
        equality_solver=:qr,
    ).fallback_chain === ()
    @test_throws ArgumentError SDPX.plan_la_backend(
        Float64;
        route=:block_arrow,
        equality_solver=:bogus,
    )
    @test SDPX.plan_la_backend(
        Float64;
        route=:block_arrow,
        requested=:legacy,
    ).selected === :legacy
    for request in (:standard, :multifloat, :fixed_extended)
        @test_throws ArgumentError SDPX.plan_la_backend(
            Float64;
            route=:block_arrow,
            requested=request,
        )
    end
    dense_legacy = SDPX.plan_la_backend(
        Float64;
        requested=:legacy,
    )
    @test dense_legacy.fallback_chain === (:rank_revealing_qr,)
    @test SDPX.plan_la_backend(
        Float64;
        requested=:legacy,
        equality_solver=:normal_equations,
    ).fallback_chain === ()

    # Positional plans from the pre-LA API carry the classification symbol
    # (`:fixed_extended`) rather than the concrete Float64x4 LA symbol.  The
    # Workspace compatibility seam normalizes only this explicitly marked
    # legacy descriptor before applying the modern exact arithmetic guard.
    classification = SDPX.ProblemClassification(
        :sdp,
        :dense,
        :fixed_extended,
        :small,
        2,
        1,
        1,
        2,
        0.5,
        0.5,
    )
    dense_config = SDPX.BackendConfiguration(
        :dense_cholesky,
        :auto,
        false,
        false,
        :off,
        (),
        false,
    )
    positional = SDPX.ExecutionPlan(
        classification,
        :sdp_primal_dual,
        :none,
        :dense_cholesky,
        dense_config,
        :pairwise_gram,
        :static,
        1,
        :general,
        0,
        (equality_solver=:auto,),
    )
    @test positional.la_config.arithmetic === :fixed_extended
    normalized = SDPX._normalize_compatibility_execution_plan(
        positional,
        Float64x4,
    )
    @test normalized.la_config.arithmetic ===
          SDPX._la_arithmetic_symbol(Float64x4)
    @test normalized.la_config.selected === :legacy
    @test normalized.la_config.fallback_reason === :compatibility

    A = [4.0 1.0; 1.0 3.0]
    backend = SDPX.StandardLABackend(:float64)
    # Keep the dense-matrix signatures concrete enough that the generic
    # AbstractLABackend/AbstractArray fallback cannot become ambiguous for
    # Float64, BigFloat, or an optional MultiFloat provider.
    @test hasmethod(
        SDPX.la_cholesky_factor!,
        Tuple{SDPX.StandardLABackend,Matrix{Float64}},
    )
    @test hasmethod(
        SDPX.la_cholesky_factor!,
        Tuple{SDPX.StandardLABackend,Matrix{BigFloat}},
    )
    @test hasmethod(
        SDPX.la_cholesky_factor!,
        Tuple{SDPX.MultiFloatLABackend{Any},Matrix{Float64x4}},
    )
    @test SDPX.la_chol!(backend, A)
    rhs = [1.0, 2.0]
    SDPX.la_trsv_lower!(backend, A, rhs)
    @test all(isfinite, rhs)

    legacy_float = SDPX.LegacyLABackend(
        :float64,
        :compatibility,
    )
    legacy_float_matrix = [4.0 1.0; 1.0 3.0]
    legacy_float_factor = SDPX.la_cholesky_factor!(
        legacy_float,
        legacy_float_matrix,
    )
    @test legacy_float_factor isa
          SDPX.LegacyLACholeskyFactor{Float64}
    legacy_float_rhs = [1.0, 2.0]
    SDPX.la_factor_solve!(legacy_float_factor, legacy_float_rhs)
    @test all(isfinite, legacy_float_rhs)
    @test SDPX.la_backend_provider(legacy_float) === :sdpx_legacy_la
    rank_loss_float = SDPX.LegacyLACholeskyFactor(
        legacy_float.provider,
        [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0e-12],
    )
    rank_loss_handle = SDPX.la_factor_handle_matrix(rank_loss_float)
    @test !SDPX._la_factor_has_numerical_rank(
        rank_loss_float,
        rank_loss_handle,
        SDPX.SolverOptions{Float64}(),
    )

    factored = SDPX.la_cholesky_factor!(backend, [4.0 1.0; 1.0 3.0])
    @test factored isa SDPX.StandardLACholeskyFactor
    solved = [1.0, 2.0]
    SDPX.la_factor_solve!(factored, solved)
    @test all(isfinite, solved)

    generic = SDPX.StandardLABackend(
        :bigfloat,
        :generic_linear_algebra,
        :owned_mutable_scalars,
    )
    for value in (BigFloat(NaN), BigFloat(Inf))
        bad = BigFloat[4 1; 1 3]
        bad[1, 1] = value
        @test SDPX.la_cholesky_factor!(generic, bad) === nothing
        bad = BigFloat[4 1; 1 3]
        bad[1, 1] = value
        @test !SDPX.la_chol!(generic, bad)
    end
    @test SDPX.la_cholesky_factor!(generic, BigFloat[-1 0; 0 1]) === nothing
    @test !SDPX.la_chol!(generic, BigFloat[-1 0; 0 1])

    legacy = SDPX.LegacyLABackend(:bigfloat, :requested_legacy)
    @test legacy.provider isa SDPX.SDPXLegacyLAProvider
    @test SDPX.la_backend_provider(legacy) === :sdpx_legacy_la
    @test SDPX.la_backend_ownership(legacy) === :owned_mutable_scalars
    @test isbitstype(typeof(legacy.provider))
    @test isempty(fieldnames(typeof(legacy.provider)))
    @test SDPX.legacy_la_provider_ownership(
        legacy.provider,
    ) === :owned_mutable_scalars
    @test SDPX.legacy_la_provider_arithmetic(
        legacy.provider,
    ) === :bigfloat
    @test_throws ArgumentError SDPX.LegacyLABackend(
        :float64,
        :requested_legacy,
        legacy.provider,
    )
    @test_throws ArgumentError SDPX.SDPXLegacyLAProvider(
        :bigfloat,
        :unknown_ownership,
    )
    @test SDPX.legacy_la_provider_capabilities(legacy.provider) ===
          SDPX.SDPX_LEGACY_LA_CAPABILITIES
    @test all(
        operation -> operation in SDPX.legacy_la_provider_capabilities(
            legacy.provider,
        ),
        (:dot, :mul_owned, :syrk, :chol, :solve, :trsm),
    )
    @test SDPX.legacy_la_provider_supports(
        legacy.provider,
        :cholesky_factor!,
    )
    @test !SDPX.legacy_la_provider_supports(
        legacy.provider,
        :eigen,
    )
    for value in (BigFloat(NaN), BigFloat(Inf))
        bad = BigFloat[4 1; 1 3]
        bad[1, 1] = value
        @test SDPX.la_cholesky_factor!(legacy, bad) === nothing
        bad = BigFloat[4 1; 1 3]
        bad[1, 1] = value
        @test !SDPX.la_chol!(legacy, bad)
    end
    @test SDPX.la_cholesky_factor!(legacy, BigFloat[-1 0; 0 1]) === nothing
    @test !SDPX.la_chol!(legacy, BigFloat[-1 0; 0 1])

    setprecision(BigFloat, 256) do
        source = BigFloat[4 1; 1 3]
        provider_matrix = SDPX.alloc_zeros(BigFloat, 2, 2)
        direct_matrix = SDPX.alloc_zeros(BigFloat, 2, 2)
        SDPX.copy_owned!(provider_matrix, source)
        SDPX.copy_owned!(direct_matrix, source)
        provider_factor = SDPX.la_cholesky_factor!(legacy, provider_matrix)
        @test provider_factor isa SDPX.LegacyLACholeskyFactor
        @test provider_factor.provider === legacy.provider
        @test provider_factor.factors === provider_matrix
        @test SDPX.kchol!(direct_matrix)
        @test provider_matrix == direct_matrix
        provider_rhs = SDPX.alloc_zeros(BigFloat, 2)
        direct_rhs = SDPX.alloc_zeros(BigFloat, 2)
        SDPX.copy_owned!(provider_rhs, BigFloat[1, 2])
        SDPX.copy_owned!(direct_rhs, BigFloat[1, 2])
        SDPX.la_factor_solve!(provider_factor, provider_rhs)
        SDPX.kcholsolve_owned!(direct_matrix, direct_rhs)
        @test provider_rhs == direct_rhs
        @test provider_rhs[1] !== provider_rhs[2]

        stale_upper = SDPX.alloc_zeros(BigFloat, 2, 2)
        SDPX.copy_owned!(stale_upper, source)
        stale_upper[1, 2] = BigFloat(NaN)
        @test SDPX.la_cholesky_factor!(legacy, stale_upper) !== nothing

        # A failed legacy factor may partially overwrite the lower triangle,
        # but it must never retry through StandardLA inside the provider.
        failed = SDPX.alloc_zeros(BigFloat, 2, 2)
        SDPX.copy_owned!(failed, BigFloat[-1 0; 0 1])
        @test SDPX.la_cholesky_factor!(legacy, failed) === nothing
        @test SDPX.la_backend_provider(legacy) === :sdpx_legacy_la
        @test SDPX.la_backend_reason(legacy) === :requested_legacy
    end
end

@testset "MultiFloat provider protocol is explicit and fail-closed" begin
    # The arbitrary Symbol/getproperty escape hatch no longer exists: core
    # dispatch must name each semantic operation up front.
    @test !isdefined(SDPX, :_la_provider_call)
    @test !hasproperty(_ContractOnlyProvider(), :cholesky_factor!)

    contract_only = SDPX.MultiFloatLABackend(
        :float64x4,
        _ContractOnlyProvider(),
    )
    @test SDPX.la_backend_name(contract_only) === :multifloat
    @test SDPX.la_dot(contract_only, [1.0, 2.0], [3.0, 4.0]) == 11.0

    S = [1.0 0.0; 0.0 1.0]
    P = [1.0 2.0; 3.0 4.0]
    output = copy(S)
    @test SDPX.la_syrk!(contract_only, output, P, 0.5, -0.25) === output
    @test output ≈ 0.5 .* (transpose(P) * P) .- 0.25 .* S

    C = zeros(2, 2)
    @test SDPX.la_mul_owned!(
        contract_only,
        C,
        P,
        P,
        0.5,
        0.0,
    ) === C
    @test C ≈ 0.5 .* (P * P)

    # Cholesky is advertised and implemented by the explicit hook; the core
    # factor wrapper still validates provider identity and storage.
    factor = SDPX.la_cholesky_factor!(
        contract_only,
        [4.0 1.0; 1.0 3.0],
    )
    @test factor !== nothing
    @test SDPX.la_factor_provider_identity(factor.provider) ===
          :multifloat_linear_algebra
    rhs = [1.0, 2.0]
    SDPX.la_factor_solve!(factor, rhs)
    @test all(isfinite, rhs)
    # The payload contract is custom to this test; no Julia `Cholesky` type
    # piracy leaks into Standard factor behavior.
    @test_throws ArgumentError SDPX.la_provider_factor_matrix(
        LinearAlgebra.cholesky([4.0 1.0; 1.0 3.0]),
    )

    # An op the model does not advertise, and for which no hook exists, must
    # fail closed instead of being discovered dynamically.
    @test !SDPX.la_backend_capabilities(contract_only).triangular_solve
    @test_throws ArgumentError SDPX.la_chol!(
        contract_only,
        [4.0 1.0; 1.0 3.0],
    )
    @test_throws ArgumentError SDPX.la_trsm!(
        contract_only,
        [1.0 0.0; 2.0 1.0],
        [1.0 0.0; 0.0 1.0],
    )

    unsupported = SDPX.MultiFloatLABackend(
        :float64x4,
        _UnsupportedProvider(),
    )
    @test SDPX.la_backend_capabilities(unsupported) ==
          SDPX.LAProviderCapabilities()
    @test !SDPX.la_provider_supports(
        SDPX.la_backend_capabilities(unsupported),
        :dot,
    )
    A = [4.0 1.0; 1.0 3.0]
    for operation in (
        () -> SDPX.la_dot(unsupported, [1.0, 2.0], [3.0, 4.0]),
        () -> SDPX.la_syrk!(unsupported, copy(A), P, 1.0, 0.0),
        () -> SDPX.la_chol!(unsupported, copy(A)),
        () -> SDPX.la_cholesky_factor!(unsupported, copy(A)),
        () -> SDPX.la_trsm!(unsupported, copy(A), copy(A)),
        () -> SDPX.la_trsv_lower!(unsupported, copy(A), [1.0, 2.0]),
        () -> SDPX.la_trsv_transpose!(unsupported, copy(A), [1.0, 2.0]),
        () -> SDPX.la_mul_owned!(unsupported, copy(A), copy(A), copy(A)),
    )
        @test_throws ArgumentError operation()
    end
    # No hidden Standard or Legacy retry: an unsupported MultiFloat op must
    # not be rewritten into another backend.
    @test SDPX.la_backend_name(unsupported) === :multifloat
    @test SDPX.la_backend_provider(unsupported) ===
          :multifloat_linear_algebra
end

@testset "Standard pivoted QR capability is executable" begin
    # This file deliberately does not import GenericLinearAlgebra.  BigFloat
    # RRQR therefore exercises the production Standard/LinearAlgebra contract
    # that the planner advertises on the minimum supported Julia line.
    for T in (Float64, BigFloat)
        setprecision(BigFloat, 256) do
            backend = SDPX.StandardLABackend(
                SDPX._la_arithmetic_symbol(T),
            )
            capabilities = SDPX.standard_la_provider_capabilities(T)
            @test capabilities.qr
            @test capabilities.rank_revealing_qr
            @test capabilities.factor_solve
            @test capabilities.multi_rhs

            source = T[4 1 0; 1 3 1; 0 1 2]
            equality_factor = SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(T, source);
                pivoted=true,
                relative_tolerance=T(100) * eps(T),
            )
            @test equality_factor isa SDPX.EqualityQRFactor{T}
            @test SDPX.la_factor_provider(equality_factor) === backend.provider
            @test size(SDPX.la_factor_packed_factors(equality_factor)) ==
                  size(source)
            @test SDPX.la_factor_rank(equality_factor) == size(source, 2)
            @test sort(SDPX.la_factor_permutation(equality_factor)) ==
                  collect(1:size(source, 2))
            @test isfinite(SDPX.la_factor_quality(equality_factor))
            if T === BigFloat
                @test all(
                    value -> precision(value) == 256,
                    SDPX.la_factor_packed_factors(equality_factor),
                )
            end

            vector_rhs = T[1, 2, 3]
            vector_solution = SDPX._owned_array_copy(T, vector_rhs)
            scratch = SDPX.alloc_zeros(T, length(vector_rhs))
            SDPX.la_factor_solve!(
                equality_factor,
                vector_solution,
                scratch,
            )
            @test transpose(source) * source * vector_solution ≈ vector_rhs

            # The same Standard QR capability also returns a conventional
            # LinearAlgebra wrapper when no equality-normal-equation tolerance
            # is requested. That handle owns vector and multi-RHS solves.
            factor = SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(T, source);
                pivoted=true,
            )
            @test factor isa SDPX.StandardLAQRFactor{T}
            @test factor.pivoted
            @test SDPX.la_factor_provider(factor) === backend.provider
            matrix_rhs = T[1 2; 2 1; 3 4]
            matrix_solution = SDPX._owned_array_copy(T, matrix_rhs)
            SDPX.la_factor_solve!(factor, matrix_solution)
            @test source * matrix_solution ≈ matrix_rhs

            rank_deficient = T[1 1; 2 2; 3 3]
            failed_rank = SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(T, rank_deficient);
                pivoted=true,
                relative_tolerance=T(100) * eps(T),
            )
            @test failed_rank isa SDPX.EqualityQRFactor{T}
            @test SDPX.la_factor_rank(failed_rank) == 1
            @test SDPX.la_factor_provider(failed_rank) === backend.provider
            @test SDPX.la_backend_name(backend) === :standard
        end
    end
end
