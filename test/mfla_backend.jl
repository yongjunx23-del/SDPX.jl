#=
    MultiFloatLinearAlgebra focused backend contract.

    The MFLA package is unregistered and is developed into an independent
    environment (see cluster-probes/v041-unified-la).  This file is therefore
    extension-gated: without MFLA loaded it only verifies that explicit
    :multifloat requests fail closed while :auto / :standard / :legacy remain
    stable.  With MFLA loaded it covers planning (advanced capabilities stay
    false until the core adapters land) and Cholesky multi-RHS and ownership.
    RRQR, LDLT, LU, workspace lifetime, mixed residual, and end-to-end
    coverage follows the core adapter work.
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
    const MFLA = MultiFloatLinearAlgebra

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
                :factor_solve,
                :multi_rhs,
                :threading,
                :dot,
                :mul,
                :mul_owned,
                :syrk,
                :triangular_solve,
                :rank_revealing_qr,
            )
                @test capability in config.capability_model
            end
            for absent in (
                :lu,
                :qr,
                :pivoted_symmetric_ldlt,
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
            equality = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_cholesky,
                equality_solver=:qr,
            )
            @test equality.selected === :multifloat
            @test :rank_revealing_qr in equality.required_capabilities
            @test :qr ∉ equality.required_capabilities
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
        @test SDPX.la_factor_provider(factor) === backend.provider
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
        @test SDPX.la_factor_provider(first) === backend.provider
        @test SDPX.la_factor_provider(second) === backend.provider
        @test SDPX.la_factor_rank(first) == size(A, 2)
        @test SDPX.la_factor_rank(second) == size(A, 2)
    end

end
