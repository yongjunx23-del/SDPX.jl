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
            )
                @test capability in config.capability_model
            end
            for absent in (
                :lu,
                :qr,
                :rank_revealing_qr,
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
            @test_throws ArgumentError LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_cholesky,
                equality_solver=:qr,
            )
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

end
