#=
    MultiFloatLinearAlgebra focused backend contract.

    The MFLA package is unregistered and is developed into an independent
    environment (see cluster-probes/v041-unified-la).  This file is therefore
    extension-gated: without MFLA loaded it only verifies that explicit
    :multifloat requests fail closed while :auto / :standard / :legacy remain
    stable.  With MFLA loaded it covers planning, dense factors, residuals,
    and factor-lifetime ownership with deliberately small matrices.
=#
using SDPX
using Test
using LinearAlgebra
using Random
using MultiFloats: Float64x2, Float64x3, Float64x4

const _MFLA_LOADED = try
    @eval begin
        import MultiFloats
        import MultiFloatLinearAlgebra
    end
    Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt) !== nothing
catch
    false
end

const _MFLA_TYPES = (Float64x2, Float64x3, Float64x4)

function _expect_multifloat_backend(::Type{T}) where {T}
    LA = SDPX.Experimental
    config = LA.plan_la_backend(
        T;
        requested=:multifloat,
        route=:dense_cholesky,
        threads=1,
    )
    @test config.selected === :multifloat
    @test config.provider === :multifloat_linear_algebra
    backend = LA.instantiate_la_backend(config, T, 1)
    @test backend isa LA.MultiFloatLABackend
    ext = Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt)
    @test ext !== nothing
    @test backend.provider isa ext._Provider{T}
    return backend
end

function _spd_matrix(::Type{T}, rng::AbstractRNG, n::Int) where {T}
    R = T.(randn(rng, n, n))
    A = transpose(R) * R
    return A + T(8) .* Matrix{T}(I, n, n)
end

function _max_abs(A, B)
    return maximum(abs(A[index] - B[index]) for index in eachindex(A, B))
end

@testset "MFLA optional provider core contract" begin
    LA = SDPX.Experimental

    if !_MFLA_LOADED
        @testset "fail closed without optional package" begin
            for T in _MFLA_TYPES
                @test_throws ArgumentError LA.plan_la_backend(
                    T;
                    requested=:multifloat,
                    route=:dense_cholesky,
                )
            end
            @test_throws ArgumentError LA.plan_la_backend(
                Float64;
                requested=:multifloat,
                route=:dense_cholesky,
            )
            @test_throws ArgumentError LA.plan_la_backend(
                BigFloat;
                requested=:multifloat,
                route=:dense_cholesky,
            )
        end
    end

    @testset "standard and legacy remain stable reference routes" begin
        for T in _MFLA_TYPES
            @test LA.plan_la_backend(
                T;
                requested=:standard,
                route=:dense_cholesky,
            ).selected === :standard
            @test LA.plan_la_backend(
                T;
                requested=:legacy,
                route=:dense_cholesky,
            ).selected === :legacy
            @test LA.plan_la_backend(
                T;
                requested=:auto,
                route=:dense_cholesky,
            ).selected === (_MFLA_LOADED ? :multifloat : :standard)
        end
    end
end

if _MFLA_LOADED
    @testset "MFLA capability and planning" begin
        LA = SDPX.Experimental
        for T in _MFLA_TYPES
            descriptor = LA.la_provider_descriptor(T, 1)
            @test descriptor.available
            @test descriptor.provider === :multifloat_linear_algebra
            config = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_cholesky,
                threads=1,
            )
            @test config.selected === :multifloat
            @test config.provider === :multifloat_linear_algebra
            @test config.ownership === :provider_owned
            @test :cholesky_factor! in config.capabilities
            @test :mul_owned in config.capabilities
            for capability in (
                :cholesky,
                :lu,
                :pivoted_symmetric_ldlt,
                :factor_solve,
                :multi_rhs,
                :threading,
                :dot,
                :mul,
                :mul_owned,
                :syrk,
                :triangular_solve,
                :rank_revealing_qr,
                :refinement_correction,
            )
                @test capability in config.capability_model
            end
            @test (:mixed_precision_residual in config.capability_model) ==
                  (T !== Float64x4)
            @test (:mixed_precision_residual in descriptor.capabilities) ==
                  (T !== Float64x4)
            for absent in (
                :qr,
                :iterative_refinement,
                :higher_precision_residual,
                :sparse_factorization,
                :norminf,
                :axpby,
            )
                @test !(absent in config.capability_model)
            end

            # Normal equations need Cholesky only and must not claim an RRQR
            # fallback.  Equality=:qr is supported because MFLA advertises
            # rrqr as a rank-revealing capability; it is a required
            # capability, never an unpivoted QR request.
            normal = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_cholesky,
                equality_solver=:normal_equations,
            )
            @test normal.selected === :multifloat
            @test normal.fallback_chain == ()
            @test :rank_revealing_qr ∉ normal.required_capabilities
            @test :lu ∉ normal.required_capabilities
            @test :pivoted_symmetric_ldlt ∉ normal.required_capabilities
            equality = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_cholesky,
                equality_solver=:qr,
            )
            @test equality.selected === :multifloat
            @test :rank_revealing_qr in equality.required_capabilities
            @test :qr ∉ equality.required_capabilities
            @test equality.fallback_chain == ()
            automatic = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_cholesky,
                equality_solver=:auto,
            )
            @test automatic.fallback_chain == (:rank_revealing_qr,)
            @test :lu ∉ automatic.required_capabilities
            @test :pivoted_symmetric_ldlt ∉ automatic.required_capabilities
        end
    end

    @testset "MFLA Cholesky multi-RHS and provenance" begin
        for T in _MFLA_TYPES
            rng = MersenneTwister(0x5eed + sizeof(T))
            backend = _expect_multifloat_backend(T)
            n = 5
            A = _spd_matrix(T, rng, n)
            borrowed = copy(A)
            for column in 1:n, row in 1:(column - 1)
                borrowed[row, column] = T(NaN)
            end

            factor = SDPX.la_cholesky_factor!(backend, borrowed)
            @test factor isa SDPX.ProviderLACholeskyFactor{T}
            @test SDPX.la_factor_handle_matrix(factor) === borrowed

            rhs_two = T.(randn(rng, n, 2))
            solution_two = copy(rhs_two)
            SDPX.la_factor_solve!(factor, solution_two)
            @test _max_abs(A * solution_two, rhs_two) < T(1e-20)

            # One borrowed provider handle serves a second, wider RHS set.
            rhs_three = T.(randn(rng, n, 3))
            solution_three = copy(rhs_three)
            SDPX.la_factor_solve!(factor, solution_three)
            @test _max_abs(A * solution_three, rhs_three) < T(1e-20)

            bad_lower = copy(A)
            bad_lower[2, 1] = T(NaN)
            @test_throws ArgumentError SDPX.la_cholesky_factor!(
                backend,
                bad_lower,
            )
        end
    end

    @testset "MFLA equality RRQR SDPX seam" begin
        rng = MersenneTwister(0x7171)
        T = Float64x4
        backend = _expect_multifloat_backend(T)

        # Exact full-rank: packed factors stay provider-owned and SDPX wraps
        # the provider payload with the equality handle.
        M = T.(randn(rng, 6, 4))
        factor = SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, M);
            pivoted=true,
            relative_tolerance=T(1e-30),
        )
        @test factor isa SDPX.EqualityQRFactor{T}
        # The equality handle retains the opaque _QRPayload (which owns the
        # lease-bearing MFQR factor and its workspace), not the backend payload.
        @test SDPX.la_factor_provider(factor) !== backend.provider
        @test SDPX.la_factor_provider_identity(
            SDPX.la_factor_provider(factor),
        ) === :multifloat_linear_algebra
        @test SDPX.la_factor_rank(factor) == size(M, 2)
        @test sort(SDPX.la_factor_permutation(factor)) == collect(1:4)
        @test SDPX.la_factor_packed_factors(factor) isa Matrix{T}
        @test SDPX.la_factor_quality(factor) > zero(T)

        rhs = T.(randn(rng, size(M, 2)))
        direction = SDPX.alloc_zeros(T, size(M, 2))
        scratch = SDPX.alloc_zeros(T, size(M, 2))
        SDPX._solve_Q!(direction, factor, rhs, scratch)
        relative_residual =
            norm(transpose(M) * (M * direction) - rhs) / norm(rhs)
        @test relative_residual <= T(1_000) * eps(T)

        # Rank deficiency is a successful factorization; SDPX owns the
        # relative rank policy, not MFLA.
        Bdeficient = T[
            1 2 3
            2 4 6
            3 6 9
            4 8 12
            1 1 1
        ]
        deficient = SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, Bdeficient);
            pivoted=true,
            relative_tolerance=T(1e-30),
        )
        @test SDPX.la_factor_rank(deficient) == 2

        # Scaling the same near-dependent geometry must not change the
        # selected relative rank.
        Bnear = T[
            1 0
            0 T(1e-40)
            0 0
        ]
        near = SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, Bnear);
            pivoted=true,
            relative_tolerance=T(1e-30),
        )
        near_scaled = SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, T(1e20) .* Bnear);
            pivoted=true,
            relative_tolerance=T(1e-30),
        )
        @test SDPX.la_factor_rank(near) == 1
        @test SDPX.la_factor_rank(near_scaled) == 1
        @test sort(SDPX.la_factor_permutation(near)) == [1, 2]
        @test sort(SDPX.la_factor_permutation(near_scaled)) == [1, 2]

        # Fail closed: no unpivoted or untoleranced equality QR request.
        @test_throws ArgumentError SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, M);
            pivoted=false,
        )
        @test_throws ArgumentError SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, M);
            pivoted=true,
        )
    end

    @testset "MFLA equality RRQR workspace factor lifetime separation" begin
        T = Float64x4
        backend = _expect_multifloat_backend(T)
        rng = MersenneTwister(0x9191)
        A = T.(randn(rng, 8, 3))

        first = SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, A);
            pivoted=true,
            relative_tolerance=T(1e-30),
        )
        second = SDPX.la_qr_factor!(
            backend,
            SDPX._owned_array_copy(T, A);
            pivoted=true,
            relative_tolerance=T(1e-30),
        )
        @test first !== nothing
        @test second !== nothing
        @test SDPX.la_factor_provider(first) !== backend.provider
        @test SDPX.la_factor_provider(second) !== backend.provider
        @test SDPX.la_factor_provider_identity(
            SDPX.la_factor_provider(first),
        ) === :multifloat_linear_algebra
        @test SDPX.la_factor_provider_identity(
            SDPX.la_factor_provider(second),
        ) === :multifloat_linear_algebra
        @test SDPX.la_factor_rank(first) == size(A, 2)
        @test SDPX.la_factor_rank(second) == size(A, 2)

        # Each factor owns an independent workspace lease, so both must remain
        # valid after the second factorization starts its own workspace.
        for factor in (first, second)
            rhs = T.(randn(rng, size(A, 2)))
            direction = SDPX.alloc_zeros(T, size(A, 2))
            scratch = SDPX.alloc_zeros(T, size(A, 2))
            SDPX._solve_Q!(direction, factor, rhs, scratch)
            @test all(isfinite, direction)
            relative_residual =
                norm(transpose(A) * (A * direction) - rhs) / norm(rhs)
            @test relative_residual <= T(1_000) * eps(T)
        end
    end

    @testset "MFLA LU internal seam" begin
        for T in _MFLA_TYPES
            backend = _expect_multifloat_backend(T)
            A = T[4 1; 2 3]
            factor = SDPX.la_lu_factor!(backend, copy(A))
            @test factor isa SDPX.ProviderLALUFactor{T}
            @test SDPX.la_factor_kind(factor) === :lu
            @test SDPX.la_factor_handle_matrix(factor) isa Matrix{T}
            @test SDPX.la_factor_provider_identity(
                SDPX.la_factor_provider(factor),
            ) === :multifloat_linear_algebra

            rhs_vector = T[1, 2]
            solution = copy(rhs_vector)
            SDPX.la_factor_solve!(factor, solution)
            @test _max_abs(A * solution, rhs_vector) < T(1e-25)

            rhs_matrix = T[1 2; 3 4]
            solution_matrix = copy(rhs_matrix)
            SDPX.la_factor_solve!(factor, solution_matrix)
            @test _max_abs(A * solution_matrix, rhs_matrix) < T(1e-25)
        end
    end

    @testset "MFLA LU two live factor leases" begin
        T = Float64x4
        backend = _expect_multifloat_backend(T)
        A = T[4 1; 2 3]
        rhs = T[1, 2]
        first = SDPX.la_lu_factor!(backend, copy(A))
        second = SDPX.la_lu_factor!(backend, copy(A))
        @test first !== nothing
        @test second !== nothing
        for factor in (first, second)
            solution = copy(rhs)
            SDPX.la_factor_solve!(factor, solution)
            @test _max_abs(A * solution, rhs) < T(1e-25)
        end
    end

    @testset "MFLA pivoted LDLT internal seam" begin
        for T in _MFLA_TYPES
            backend = _expect_multifloat_backend(T)
            A0 = T[0 1; 1 0]
            factor = SDPX.la_ldlt_factor!(backend, copy(A0))
            @test factor isa SDPX.ProviderLALDLTFactor{T}
            @test SDPX.la_factor_kind(factor) === :ldlt
            @test SDPX.la_factor_handle_matrix(factor) isa Matrix{T}
            @test SDPX.la_factor_provider_identity(
                SDPX.la_factor_provider(factor),
            ) === :multifloat_linear_algebra
            @test SDPX.la_ldlt_inertia(factor) ==
                  (positive=1, negative=1, zero=0)
            @test SDPX.la_ldlt_blocks(factor) == [2]
            permutation = SDPX.la_ldlt_permutation(factor)
            @test sort(permutation) == [1, 2]

            rhs_vector = T[1, 2]
            solution = copy(rhs_vector)
            SDPX.la_ldlt_factor_solve!(factor, solution)
            @test _max_abs(A0 * solution, rhs_vector) < T(1e-25)

            rhs_matrix = T[1 2; 3 4]
            solution_matrix = copy(rhs_matrix)
            SDPX.la_ldlt_factor_solve!(factor, solution_matrix)
            @test _max_abs(A0 * solution_matrix, rhs_matrix) < T(1e-25)

            # Only the lower triangle is authoritative; a poisoned upper
            # triangle cannot change the result.
            poisoned = T[0 T(NaN); 1 0]
            poisoned_factor = SDPX.la_ldlt_factor!(backend, copy(poisoned))
            @test poisoned_factor !== nothing
            poisoned_solution = copy(rhs_vector)
            SDPX.la_ldlt_factor_solve!(poisoned_factor, poisoned_solution)
            @test _max_abs(A0 * poisoned_solution, rhs_vector) < T(1e-25)

            bad_lower = T[0 1; T(NaN) 0]
            @test SDPX.la_ldlt_factor!(backend, copy(bad_lower)) === nothing
        end
    end

    @testset "MFLA LDLT two live factor leases" begin
        T = Float64x4
        backend = _expect_multifloat_backend(T)
        A0 = T[0 1; 1 0]
        rhs = T[1, 2]
        first = SDPX.la_ldlt_factor!(backend, copy(A0))
        second = SDPX.la_ldlt_factor!(backend, copy(A0))
        @test first !== nothing
        @test second !== nothing
        @test SDPX.la_ldlt_inertia(first) ==
              (positive=1, negative=1, zero=0)
        @test SDPX.la_ldlt_inertia(second) ==
              (positive=1, negative=1, zero=0)
        for factor in (first, second)
            solution = copy(rhs)
            SDPX.la_ldlt_factor_solve!(factor, solution)
            @test _max_abs(A0 * solution, rhs) < T(1e-25)
        end
    end

    @testset "MFLA residuals and one requested correction" begin
        for T in _MFLA_TYPES
            backend = _expect_multifloat_backend(T)
            A = T[4 1; 1 3]
            exact = T[1, 2]
            rhs = A * exact
            approximate = exact .+ T[1e-8, -1e-8]
            residual = zeros(T, 2)
            SDPX.la_residual!(
                backend, :N, A, approximate, rhs, residual,
            )
            @test _max_abs(residual, rhs - A * approximate) <= T(32) * eps(T)
            before = SDPX.la_normwise_backward_error(
                backend, :N, A, approximate, rhs, residual,
            )
            @test isfinite(before)
            @test before > zero(T)
            @test_throws ArgumentError SDPX.la_residual!(
                backend, :T, A, approximate, rhs, residual,
            )

            factors = (
                SDPX.la_cholesky_factor!(backend, copy(A)),
                SDPX.la_lu_factor!(backend, copy(A)),
                SDPX.la_ldlt_factor!(backend, copy(A)),
            )
            for factor in factors
                correction = zeros(T, 2)
                SDPX.la_refinement_correction!(
                    factor,
                    residual,
                    correction,
                )
                corrected = approximate + correction
                corrected_residual = rhs - A * corrected
                @test maximum(abs, corrected_residual) < maximum(abs, residual)
                @test_throws ArgumentError SDPX.la_refinement_correction!(
                    factor,
                    T[NaN, 0],
                    zeros(T, 2),
                )
            end
        end
    end

    @testset "MFLA explicit mixed-limb residual pairs" begin
        for (Source, Residual) in (
            (Float64x2, Float64x3),
            (Float64x2, Float64x4),
            (Float64x3, Float64x4),
        )
            backend = _expect_multifloat_backend(Source)
            A = Source[2 1; 1 3]
            x = Source[1, 2]
            rhs = A * x
            residual = zeros(Residual, 2)
            SDPX.la_mixed_residual!(backend, A, x, rhs, residual)
            @test residual == zeros(Residual, 2)
        end

        source_backend = _expect_multifloat_backend(Float64x2)
        A2 = Float64x2[2 1; 1 3]
        x2 = Float64x2[1, 2]
        b2 = A2 * x2
        @test_throws ArgumentError SDPX.la_mixed_residual!(
            source_backend, A2, x2, b2, zeros(Float64x2, 2),
        )
        high_backend = _expect_multifloat_backend(Float64x4)
        A4 = Float64x4[2 1; 1 3]
        x4 = Float64x4[1, 2]
        b4 = A4 * x4
        @test_throws ArgumentError SDPX.la_mixed_residual!(
            high_backend, A4, x4, b4, zeros(Float64x3, 2),
        )
        # The selected backend arithmetic is authoritative; it cannot be used
        # as a generic dispatcher for a different source arithmetic.
        @test_throws ArgumentError SDPX.la_mixed_residual!(
            high_backend, A2, x2, b2, zeros(Float64x3, 2),
        )
        A3 = Float64x3[2 1; 1 3]
        x3 = Float64x3[1, 2]
        b3 = A3 * x3
        @test_throws ArgumentError SDPX.la_mixed_residual!(
            source_backend, A3, x3, b3, zeros(Float64x4, 2),
        )
    end

    @testset "MFLA Float64x4 end-to-end smoke" begin
        T = Float64x4

        function _multifloat_options(::Type{T}, algorithm::Symbol) where {T}
            return SDPX.SolveOptions(
                verbosity=0,
                diagnostics=true,
                timing=false,
                algorithm=algorithm,
                duality_gap_threshold=T(1e-10),
                primal_error_threshold=T(1e-10),
                dual_error_threshold=T(1e-10),
                linear_algebra_backend=:multifloat,
            )
        end

        function _assert_multifloat_execution(selected)
            @test selected.planned_la_backend === :multifloat
            @test selected.la_backend === :multifloat
            @test selected.planned_la_provider ===
                  :multifloat_linear_algebra
            @test selected.la_executed_provider ===
                  :multifloat_linear_algebra
            @test selected.planned_la_fallback_reason === :none
            @test selected.la_fallback_reason === :none
            @test selected.backend_resolution === :planned
            @test selected.certificate.valid
        end

        @testset "tiny SDP" begin
            k = 3
            m = k * (k + 1) ÷ 2
            c = zeros(T, m)
            c[1] = -one(T)
            A = zeros(T, m, k, k)
            A[1, 1, 1] = one(T)
            A[2, 2, 2] = one(T)
            A[3, 3, 3] = one(T)
            A[4, 1, 2] = one(T)
            A[4, 2, 1] = one(T)
            A[5, 1, 3] = one(T)
            A[5, 3, 1] = one(T)
            A[6, 2, 3] = one(T)
            A[6, 3, 2] = one(T)
            B = zeros(T, m, 1)
            B[1, 1] = one(T)
            B[2, 1] = one(T)
            B[3, 1] = one(T)
            problem = SDPX.ingest(
                c,
                [A],
                [zeros(T, k, k)],
                B,
                T[3];
                T=T,
                sparse=false,
                verbosity=0,
            )
            result = SDPX.solve(
                problem,
                _multifloat_options(T, :sdp),
            )
            @test result.status == SDPX.Optimal
            @test isapprox(Float64(result.pObj), -3.0; atol=1e-6)
            @test isapprox(Float64(result.dObj), -3.0; atol=1e-6)
            @test result.p_res <= T(1e-6)
            @test result.d_res <= T(1e-6)
            _assert_multifloat_execution(
                result.diagnostics.selected_algorithms,
            )
        end

        @testset "tiny SOCP" begin
            problem = SDPX.second_order_program(
                T[1, 0, 0],
                Matrix{T}(I, 3, 3),
                zeros(T, 3);
                Aeq=T[0 1 0; 0 0 1],
                beq=T[3, 4],
            )
            result = SDPX.solve_socp(
                problem,
                _multifloat_options(T, :socp),
            )
            @test result.status == SDPX.Optimal
            @test isapprox(Float64(result.pObj), 5.0; atol=1e-6)
            @test isapprox(Float64(result.dObj), 5.0; atol=1e-6)
            @test result.p_res <= T(1e-6)
            @test result.d_res <= T(1e-6)
            _assert_multifloat_execution(
                result.diagnostics.selected_algorithms,
            )
        end

        @testset "LP is not migrated to MFLA" begin
            problem = SDPX.linear_program(
                T[1, 2],
                Matrix{T}(I, 2, 2),
                T[3, 4];
                T=T,
                sparse=false,
            )
            legacy_plan = SDPX.build_execution_plan(
                problem,
                SDPX.SolverOptions{T}(
                    algorithm=:lp,
                    linear_algebra_backend=:auto,
                    verbosity=0,
                ),
            )
            @test legacy_plan.algorithm === :lp_primal_dual
            @test legacy_plan.la_config.selected === :legacy
            @test legacy_plan.la_config.provider === :sdpx_legacy_la
            @test legacy_plan.la_config.fallback_reason ===
                  :route_not_migrated

            @test_throws ArgumentError SDPX.build_execution_plan(
                problem,
                SDPX.SolverOptions{T}(
                    algorithm=:lp,
                    linear_algebra_backend=:multifloat,
                    verbosity=0,
                ),
            )

            result = SDPX.solve(
                problem,
                SDPX.SolveOptions(
                    verbosity=0,
                    diagnostics=true,
                    algorithm=:lp,
                    duality_gap_threshold=T(1e-10),
                    primal_error_threshold=T(1e-10),
                    dual_error_threshold=T(1e-10),
                    linear_algebra_backend=:legacy,
                ),
            )
            @test result.status == SDPX.Optimal
            @test isapprox(Float64(result.pObj), 11.0; atol=1e-6)
            selected = result.diagnostics.selected_algorithms
            @test selected.planned_la_backend === :legacy
            @test selected.planned_la_provider === :sdpx_legacy_la
            @test selected.planned_la_fallback_reason ===
                  :route_not_migrated
            @test selected.certificate.valid
        end
    end

end
