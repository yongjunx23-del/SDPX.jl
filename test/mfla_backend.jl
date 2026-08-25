#=
    MultiFloatLinearAlgebra focused backend contract.

    The MFLA package is unregistered and is tested in an independent provider
    environment (see scripts/dev_v05_provider_smoke.sh).  This file is therefore
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

@testset "Float64x2 equality QR crossover keeps wider types conservative" begin
    @test SDPX._equality_qr_maximum_elements(Float64) == 50_000_000
    @test SDPX._equality_qr_maximum_elements(Float64x2) == 100_000_000
    @test SDPX._equality_qr_maximum_elements(Float64x3) == 20_000_000
    @test SDPX._equality_qr_maximum_elements(Float64x4) == 20_000_000
    @test SDPX._equality_qr_maximum_elements(BigFloat) == 2_000_000

    opts = SDPX.SolverOptions(Float64x2; equality_solver=:qr, verbosity=0)
    @test SDPX._equality_qr_allowed(
        zeros(Float64x2, 1, 8_192),
        opts,
    )
    @test !SDPX._equality_qr_allowed(
        zeros(Float64x2, 1, 8_193),
        opts,
    )
end

@testset "MultiFloat duplicate-column fingerprints preserve exact decisions" begin
    left = Float64x2((1.0, 0.0))
    colliding_but_distinct = Float64x2((1.0, eps(Float64)))
    signed_negative_zero = Float64x2((-0.0, 0.0))
    signed_positive_zero = Float64x2((0.0, 0.0))

    # The extension deliberately permits a leading-limb collision. The full
    # equality scan must still reject columns that differ in a lower limb.
    @test SDPX._duplicate_column_fingerprint(left, UInt(0)) ==
          SDPX._duplicate_column_fingerprint(
              colliding_but_distinct,
              UInt(0),
          )
    @test !SDPX._has_exact_duplicate_columns([
        left colliding_but_distinct
        Float64x2(2) Float64x2(2)
        signed_negative_zero signed_positive_zero
    ])

    # Signed zero normalization and exact duplicate detection retain the
    # backend-independent behavior exercised by the generic tests.
    @test SDPX._has_exact_duplicate_columns([
        left left
        Float64x2(2) Float64x2(2)
        signed_negative_zero signed_positive_zero
    ])
end

function _expect_multifloat_backend(::Type{T}) where {T}
    LA = SDPX
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

function _mfla_block_arrow_fixture(
    ::Type{T};
    block_count::Int=8,
    equality_count::Int=8,
    rank_deficient::Bool=false,
) where {T}
    variable_count = 3 * block_count
    equality_count <= variable_count || throw(ArgumentError(
        "equalities cannot exceed variables",
    ))
    coefficients = [
        zeros(T, variable_count, 2, 2)
        for _ in 1:block_count
    ]
    for block in 1:block_count
        first = 3 * block - 2
        coefficients[block][first, 1, 1] = one(T)
        coefficients[block][first + 1, 1, 2] = one(T)
        coefficients[block][first + 1, 2, 1] = one(T)
        coefficients[block][first + 2, 2, 2] = one(T)
    end
    equality = zeros(T, variable_count, equality_count)
    for column in 1:equality_count
        equality[column, column] = one(T)
        for row in (equality_count + 1):variable_count
            equality[row, column] =
                T(mod(17 * row + 11 * column, 29) - 14) / T(257)
        end
    end
    if rank_deficient && equality_count >= 2
        equality[:, equality_count] .= equality[:, 1]
    end
    problem = SDPX.ingest(
        ones(T, variable_count),
        coefficients,
        [zeros(T, 2, 2) for _ in 1:block_count],
        equality,
        zeros(T, equality_count);
        sparse=true,
        verbosity=0,
    )
    X = [
        T[2 + block / 100 1 / 31; 1 / 31 3 / 2 + block / 200]
        for block in 1:block_count
    ]
    Y = [
        T[7 / 5 + block / 150 1 / 37; 1 / 37 19 / 10 + block / 250]
        for block in 1:block_count
    ]
    return problem, X, Y
end

@testset "MFLA optional provider core contract" begin
    LA = SDPX

    if !_MFLA_LOADED
        @testset "fail closed without optional package" begin
            for T in _MFLA_TYPES
                for route in (
                    :dense_cholesky,
                    :positive_definite_cholesky,
                    :dense_lu,
                    :block_arrow,
                )
                    @test_throws ArgumentError LA.plan_la_backend(
                        T;
                        requested=:multifloat,
                        route=route,
                    )
                end
            end
            for T in (Float64, BigFloat)
                for route in (
                    :dense_cholesky,
                    :positive_definite_cholesky,
                    :dense_lu,
                )
                    @test_throws ArgumentError LA.plan_la_backend(
                        T;
                        requested=:multifloat,
                        route=route,
                    )
                end
            end
        end
    end

    @testset "standard and legacy remain stable reference routes" begin
        for T in _MFLA_TYPES, route in (
            :dense_cholesky,
            :positive_definite_cholesky,
            :dense_lu,
        )
            @test LA.plan_la_backend(
                T;
                requested=:standard,
                route=route,
            ).selected === :standard
            @test LA.plan_la_backend(
                T;
                requested=:legacy,
                route=route,
            ).selected === :legacy
            @test LA.plan_la_backend(
                T;
                requested=:auto,
                route=route,
            ).selected === (_MFLA_LOADED ? :multifloat : :standard)
        end
        for T in _MFLA_TYPES
            @test LA.plan_la_backend(
                T;
                requested=:auto,
                route=:block_arrow,
            ).selected === (_MFLA_LOADED ? :multifloat : :legacy)
            @test LA.plan_la_backend(
                T;
                requested=:legacy,
                route=:block_arrow,
            ).selected === :legacy
            @test_throws ArgumentError LA.plan_la_backend(
                T;
                requested=:standard,
                route=:block_arrow,
            )
        end
    end

    @testset "legacy block-arrow equality tail is unchanged" begin
        T = Float64
        problem, X, Y = _mfla_block_arrow_fixture(T)
        options = SDPX.SolverOptions{T}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            sparse=true,
            formulation=:auto,
            equality_solver=:normal_equations,
            linear_algebra_backend=:legacy,
            extended_precision_blas=:off,
            mixed_precision_kkt=:off,
            threads=1,
            verbosity=0,
        )
        plan = SDPX.build_execution_plan(problem, options)
        @test plan.kkt_backend === :block_arrow
        @test plan.la_config.selected === :legacy
        workspace = SDPX.Workspace(problem; execution_plan=plan)
        @test workspace.la_backend isa SDPX.LegacyLABackend
        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(workspace, problem, problem.cons, X, Y)
        factorization = SDPX.factorize!(SDPX.select_backend(workspace), workspace, problem, options)
        @test factorization.ok
        @test workspace.equality_gram_kernel === :blas_syrk
        @test workspace.Qchol isa SDPX.LegacyLACholeskyFactor{T}
        @test workspace.executed_la_provider === :sdpx_legacy_la
    end
end

if _MFLA_LOADED
    @testset "MFLA capability and planning" begin
        LA = SDPX
        for T in _MFLA_TYPES
            descriptor = LA.la_provider_descriptor(T, 1)
            @test descriptor.available
            @test descriptor.provider === :multifloat_linear_algebra
            upstream = MultiFloatLinearAlgebra.capabilities(T)
            @test upstream.reusable_workspace
            @test upstream.factor_metadata_ownership === :factor_owned
            @test upstream.shared_gemm_workspace_concurrency === :serialized_safe
            @test !upstream.concurrent_factor_workspace
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
                :ldlt_inertia,
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

            block_arrow = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:block_arrow,
                equality_solver=:normal_equations,
            )
            @test block_arrow.selected === :multifloat
            @test block_arrow.provider === :multifloat_linear_algebra
            @test block_arrow.required_capabilities == (
                :cholesky,
                :factor_solve,
                :syrk,
            )
            @test block_arrow.fallback_chain == ()
            @test :triangular_solve ∉ block_arrow.required_capabilities
            block_arrow_auto = LA.plan_la_backend(
                T;
                requested=:auto,
                route=:block_arrow,
                equality_solver=:auto,
            )
            @test block_arrow_auto.selected === :multifloat
            @test block_arrow_auto.fallback_chain == (:rank_revealing_qr,)

            augmented = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_augmented_ldlt,
                equality_solver=:normal_equations,
                threads=1,
            )
            @test augmented.selected === :multifloat
            @test augmented.fallback_chain == ()
            @test augmented.required_capabilities == (
                :pivoted_symmetric_ldlt,
                :ldlt_inertia,
                :factor_solve,
                :multi_rhs,
            )
            @test :pivoted_symmetric_ldlt in augmented.capabilities
            @test :multi_rhs in augmented.capabilities
        end
    end


    @testset "MFLA block-arrow equality tail" begin
        T = Float64x2
        problem, X, Y = _mfla_block_arrow_fixture(T)
        options = SDPX.SolverOptions{T}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            sparse=true,
            formulation=:auto,
            equality_solver=:normal_equations,
            linear_algebra_backend=:multifloat,
            extended_precision_blas=:off,
            mixed_precision_kkt=:off,
            threads=1,
            verbosity=0,
        )
        plan = SDPX.build_execution_plan(problem, options)
        @test plan.kkt_backend === :block_arrow
        @test plan.la_config.selected === :multifloat
        @test plan.la_config.required_capabilities == (
            :cholesky,
            :factor_solve,
            :syrk,
        )

        workspace = SDPX.Workspace(problem; execution_plan=plan)
        @test workspace.la_backend isa SDPX.MultiFloatLABackend
        @test workspace.arrow !== nothing
        @test isempty(workspace.arrow.global_ids)
        @test SDPX.factor_blocks!(workspace, X, Y)
        SDPX.schur_build!(workspace, problem, problem.cons, X, Y)
        schur = zeros(T, problem.dims.m, problem.dims.m)
        SDPX.materialize_schur!(schur, workspace)

        factorization = SDPX.factorize!(SDPX.select_backend(workspace), workspace, problem, options)
        @test factorization.ok
        @test !factorization.q_pivoted
        @test !factorization.q_rank_deficient
        @test workspace.equality_gram_kernel === :multifloat_syrk
        @test workspace.Qchol isa SDPX.ProviderLACholeskyFactor{T}
        @test SDPX.la_factor_provider_identity(
            SDPX.la_factor_provider(workspace.Qchol),
        ) === :multifloat_linear_algebra
        @test workspace.executed_la_backend === :multifloat
        @test workspace.executed_la_provider ===
              :multifloat_linear_algebra

        expected_gram = transpose(workspace.Btil) * workspace.Btil
        gram_scale = max(maximum(abs, expected_gram), one(T))
        @test maximum(
            abs,
            LowerTriangular(workspace.Q) - LowerTriangular(expected_gram),
        ) / gram_scale <= T(1e-24)

        primal_rhs = T.(range(-0.7, 1.1; length=problem.dims.m))
        equality_rhs = T.(range(-0.2, 0.3; length=problem.dims.n))
        dx = zeros(T, problem.dims.m)
        dy = zeros(T, problem.dims.n)
        SDPX.solve_kkt!(
            workspace,
            problem.dims.n,
            primal_rhs,
            equality_rhs,
            dx,
            dy,
        )
        first_residual = schur * dx - problem.B * dy - primal_rhs
        second_residual = transpose(problem.B) * dx - equality_rhs
        residual_scale = max(
            maximum(abs, primal_rhs),
            maximum(abs, equality_rhs),
            one(T),
        )
        @test max(
            maximum(abs, first_residual),
            maximum(abs, second_residual),
        ) / residual_scale <= T(1e-20)

        deficient, Xdef, Ydef = _mfla_block_arrow_fixture(
            T;
            rank_deficient=true,
        )
        normal_plan = SDPX.build_execution_plan(deficient, options)
        normal_workspace = SDPX.Workspace(
            deficient;
            execution_plan=normal_plan,
        )
        @test SDPX.factor_blocks!(normal_workspace, Xdef, Ydef)
        SDPX.schur_build!(
            normal_workspace,
            deficient,
            deficient.cons,
            Xdef,
            Ydef,
        )
        rejected = SDPX.factorize!(
            SDPX.select_backend(normal_workspace),
            normal_workspace,
            deficient,
            options,
        )
        @test !rejected.ok
        @test normal_workspace.la_fallback_reason ===
              :la_equality_factor_failed
        @test !(normal_workspace.Qchol isa LinearAlgebra.CholeskyPivoted)

        auto_options = SDPX._replace_solver_options(
            options;
            equality_solver=:auto,
        )
        auto_plan = SDPX.build_execution_plan(deficient, auto_options)
        @test auto_plan.la_config.fallback_chain == (:rank_revealing_qr,)
        auto_workspace = SDPX.Workspace(
            deficient;
            execution_plan=auto_plan,
        )
        @test SDPX.factor_blocks!(auto_workspace, Xdef, Ydef)
        SDPX.schur_build!(
            auto_workspace,
            deficient,
            deficient.cons,
            Xdef,
            Ydef,
        )
        accepted = SDPX.factorize!(
            SDPX.select_backend(auto_workspace),
            auto_workspace,
            deficient,
            auto_options,
        )
        @test accepted.ok
        @test accepted.q_pivoted
        @test accepted.q_rank_deficient
        @test auto_workspace.Qchol isa SDPX.EqualityQRFactor{T}
        @test auto_workspace.la_fallback_reason ===
              :la_equality_factor_failed
        @test SDPX.la_factor_provider_identity(
            SDPX.la_factor_provider(auto_workspace.Qchol),
        ) === :multifloat_linear_algebra
    end

    @testset "MFLA Cholesky multi-RHS and provenance" begin
        for T in _MFLA_TYPES
            rng = MersenneTwister(0x5eed + sizeof(T))
            backend = _expect_multifloat_backend(T)
            @test SDPX.la_backend_owns_equality_gram(backend)
            @test SDPX.la_equality_gram_kernel(backend, T) ===
                  :multifloat_syrk
            n = 5
            A = _spd_matrix(T, rng, n)
            borrowed = copy(A)
            for column in 1:n, row in 1:(column - 1)
                borrowed[row, column] = T(NaN)
            end

            factor = SDPX.la_cholesky_factor!(backend, borrowed)
            @test factor isa SDPX.ProviderLACholeskyFactor{T}
            @test !SDPX.la_cholesky_rank_authoritative(factor)
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
        # The equality handle retains the opaque provider factor payload, not
        # the reusable backend workspace itself.
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

    @testset "MFLA equality RRQR survives workspace reuse" begin
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

        # MFLA snapshots factor metadata, so both factors remain valid after
        # the provider workspace has been reused by the second factorization.
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
            @test SDPX.la_factor_provider(factor) !== backend.provider

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

    @testset "MFLA LU factors survive workspace growth and reuse" begin
        T = Float64x4
        backend = _expect_multifloat_backend(T)
        A = T[4 1; 2 3]
        rhs = T[1, 2]
        first = SDPX.la_lu_factor!(backend, copy(A))
        ext = Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt)
        capacity_before = MultiFloatLinearAlgebra.workspace_capacity(
            backend.provider.workspace,
        ).factor_capacity
        wider = Matrix{T}(I, 7, 7) + T(0.05) .* ones(T, 7, 7)
        second = SDPX.la_lu_factor!(backend, copy(wider))
        capacity_after = MultiFloatLinearAlgebra.workspace_capacity(
            backend.provider.workspace,
        ).factor_capacity
        @test first !== nothing
        @test second !== nothing
        @test capacity_after >= 7
        @test capacity_after >= capacity_before
        @test ext !== nothing
        for factor in (first,)
            solution = copy(rhs)
            SDPX.la_factor_solve!(factor, solution)
            @test _max_abs(A * solution, rhs) < T(1e-25)
        end
        wider_rhs = ones(T, 7)
        wider_solution = copy(wider_rhs)
        SDPX.la_factor_solve!(second, wider_solution)
        @test _max_abs(wider * wider_solution, wider_rhs) < T(1e-25)
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
            payload = factor.provider
            @test SDPX.la_provider_factor_status(payload) ==
                  MultiFloatLinearAlgebra.factor_status(payload.factor)
            @test SDPX.la_provider_ldlt_inertia(payload) ==
                  MultiFloatLinearAlgebra.factor_inertia(payload.factor)
            @test SDPX.la_provider_ldlt_blocks(payload) == [2]
            @test SDPX.la_provider_ldlt_permutation(payload) ==
                  MultiFloatLinearAlgebra.factor_permutation(payload.factor)
            # Full diagnostics remains available as one explicit observation;
            # classification and metadata use the lightweight accessors above.
            diagnostics = SDPX.la_factor_diagnostics(factor)
            @test diagnostics.kind === :ldlt
            @test diagnostics.success
            @test diagnostics.status == 0
            @test diagnostics.inertia == (positive=1, negative=1, zero=0)

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

    @testset "MFLA LDLT two live factors after workspace reuse" begin
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

        function _assert_multifloat_execution(
            selected;
            backend_resolution::Symbol=:planned,
        )
            @test selected.planned_la_backend === :multifloat
            @test selected.la_backend === :multifloat
            @test selected.planned_la_provider ===
                  :multifloat_linear_algebra
            @test selected.la_executed_provider ===
                  :multifloat_linear_algebra
            @test selected.planned_la_fallback_reason === :none
            @test selected.la_fallback_reason === :none
            @test selected.backend_resolution === backend_resolution
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

        @testset "explicit dense augmented SDP" begin
            coefficients = zeros(T, 3, 3, 3)
            @inbounds for index in 1:3
                coefficients[index, index, index] = one(T)
            end
            problem = SDPX.ingest(
                ones(T, 3),
                [coefficients],
                [zeros(T, 3, 3)],
                T[1 0; 0 1; 1 -1],
                T[1, 1];
                sparse=false,
                verbosity=0,
            )
            options = SDPX.SolverOptions{T}(
                algorithm=:sdp,
                presolve=false,
                scaling=:none,
                sparse=false,
                formulation=:augmented,
                equality_solver=:normal_equations,
                linear_algebra_backend=:multifloat,
                threads=1,
                verbosity=0,
                diagnostics=true,
                timing=true,
                iter_max=100,
                ϵ_gap=T(1e-10),
                ϵ_primal=T(1e-10),
                ϵ_dual=T(1e-10),
            )
            result = SDPX.solve!(problem, options)
            @test result.status == SDPX.Optimal
            @test isapprox(Float64(result.pObj), 2.0; atol=1e-8)
            selected = result.diagnostics.selected_algorithms
            @test selected.requested_kkt_formulation === :augmented
            @test selected.planned_kkt_formulation === :dense_augmented_kkt
            @test selected.executed_kkt_formulation === :dense_augmented_kkt
            @test selected.planned_factorization === :pivoted_symmetric_ldlt
            @test selected.executed_factorization === :pivoted_symmetric_ldlt
            @test selected.planned_regularization === :schur_diagonal_retry
            @test selected.executed_regularization == zero(T)
            @test selected.la_backend === :multifloat
            @test selected.la_executed_provider ===
                  :multifloat_linear_algebra
            @test selected.la_fallback_reason === :none
            @test result.termination.augmented_kkt.inertia ==
                  (positive=3, negative=2, zero=0)
            @test selected.certificate.valid

            dependent = SDPX.ingest(
                ones(T, 3),
                [coefficients],
                [zeros(T, 3, 3)],
                T[1 2; 0 0; 1 2],
                T[1, 2];
                sparse=false,
                verbosity=0,
            )
            rejected = SDPX.solve!(dependent, options)
            @test rejected.status != SDPX.Optimal
            @test rejected.diagnostics.selected_algorithms.la_fallback_reason ===
                  :la_factor_failed
            @test !rejected.termination.augmented_kkt.available
            @test !rejected.termination.augmented_kkt.rank_deficient
            @test rejected.termination.augmented_kkt.inertia === nothing
            reduced_options = SDPX._replace_solver_options(
                options;
                presolve=true,
            )
            reduced = SDPX.solve!(dependent, reduced_options)
            @test reduced.status == SDPX.Optimal
            @test reduced.diagnostics.selected_algorithms.certificate.valid
        end

        @testset "automatic formulation uses verified equality evidence" begin
            coefficients = zeros(T, 3, 3, 3)
            @inbounds for index in 1:3
                coefficients[index, index, index] = one(T)
            end
            problem = SDPX.ingest(
                ones(T, 3),
                [coefficients],
                [zeros(T, 3, 3)],
                T[1 0; 0 1e8; 1 -1e8],
                T[1, 1e8];
                sparse=false,
                verbosity=0,
            )
            options = SDPX.SolverOptions{T}(
                algorithm=:sdp,
                presolve=true,
                scaling=:none,
                sparse=false,
                formulation=:auto,
                equality_solver=:normal_equations,
                linear_algebra_backend=:multifloat,
                threads=1,
                verbosity=0,
                diagnostics=true,
                timing=true,
                iter_max=100,
                ϵ_gap=T(1e-10),
                ϵ_primal=T(1e-10),
                ϵ_dual=T(1e-10),
            )
            result = SDPX.solve!(problem, options)
            @test result.status == SDPX.Optimal
            selected = result.diagnostics.selected_algorithms
            decision = selected.formulation_decision
            @test decision.requested === :auto
            @test decision.preferred === :dense_augmented_kkt
            @test decision.selected === :dense_augmented_kkt
            @test decision.reason === :large_equality_scale_spread
            @test decision.equality_evidence.available
            @test decision.equality_evidence.basis_verified
            @test selected.planned_kkt_formulation === :dense_augmented_kkt
            @test selected.executed_kkt_formulation === :dense_augmented_kkt
            @test selected.la_fallback_reason === :none
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
                backend_resolution=:native_soc_plan,
            )

            augmented_options = SDPX.SolveOptions(
                verbosity=0,
                diagnostics=true,
                timing=false,
                algorithm=:socp,
                presolve=false,
                sparse=false,
                scaling=:none,
                formulation=:augmented,
                equality_solver=:normal_equations,
                duality_gap_threshold=T(1e-10),
                primal_error_threshold=T(1e-10),
                dual_error_threshold=T(1e-10),
                linear_algebra_backend=:multifloat,
            )
            augmented = SDPX.solve_socp(problem, augmented_options)
            @test augmented.status == SDPX.Optimal
            @test isapprox(Float64(augmented.pObj), 5.0; atol=1e-6)
            augmented_selected =
                augmented.diagnostics.selected_algorithms
            @test augmented_selected.executed_kkt_formulation ===
                  :dense_augmented_kkt
            @test augmented_selected.executed_factorization ===
                  :pivoted_symmetric_ldlt
            @test augmented_selected.la_backend === :multifloat
            @test augmented_selected.la_fallback_reason === :none
            @test augmented_selected.certificate.valid
        end

        @testset "tiny dense LP uses MFLA" begin
            for (with_equality, expected_kkt, expected_factor) in (
                (false, :positive_definite_cholesky, :cholesky),
                (true, :dense_lu, :lu),
            )
                problem = SDPX.linear_program(
                    T[1, 2],
                    T[1 0; 0 1; 1 1],
                    T[1, 1, 3];
                    Aeq=with_equality ? T[1 1] : nothing,
                    beq=with_equality ? T[3] : nothing,
                    T=T,
                    sparse=false,
                )
                options = SDPX.SolverOptions{T}(
                    verbosity=0,
                    diagnostics=true,
                    timing=false,
                    algorithm=:lp,
                    presolve=false,
                    scaling=:none,
                    ϵ_gap=T(1e-10),
                    ϵ_primal=T(1e-10),
                    ϵ_dual=T(1e-10),
                    linear_algebra_backend=:multifloat,
                )
                plan = SDPX.build_execution_plan(problem, options)
                @test plan.kkt_backend === expected_kkt
                @test plan.la_config.selected === :multifloat
                @test plan.la_config.provider ===
                      :multifloat_linear_algebra
                @test expected_factor in
                      plan.la_config.required_capabilities
                result = SDPX.solve!(problem, options)
                @test result.status == SDPX.Optimal
                @test isapprox(Float64(result.pObj), 4.0; atol=1e-6)
                selected = result.diagnostics.selected_algorithms
                @test selected.kkt === expected_kkt
                @test selected.lp_formulation === expected_kkt
                @test selected.la_backend === :multifloat
                @test selected.la_executed_provider ===
                      :multifloat_linear_algebra
                @test selected.la_factorization === expected_factor
                @test selected.planned_la_backend === :multifloat
            end

            auto_plan = SDPX.build_execution_plan(
                SDPX.linear_program(
                    T[1, 2],
                    T[1 0; 0 1; 1 1],
                    T[1, 1, 3];
                    T=T,
                    sparse=false,
                ),
                SDPX.SolverOptions{T}(
                    verbosity=0,
                    algorithm=:lp,
                    presolve=false,
                    scaling=:none,
                    linear_algebra_backend=:auto,
                ),
            )
            @test auto_plan.la_config.selected === :multifloat
        end
    end

end
