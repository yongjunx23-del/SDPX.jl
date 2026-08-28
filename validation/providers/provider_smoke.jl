#=
    Explicit installed-provider smoke for MFLA and BFLA.

    This file is intentionally outside the ordinary `Pkg.test` target:
    `test/runtests.jl` does not include it, and the default test environment
    never clones or resolves MultiFloatLinearAlgebra or BigFloatLinearAlgebra.
    Run it through `scripts/provider_smoke.sh` (or an equivalent
    environment that has SDPX, MultiFloatLinearAlgebra,
    BigFloatLinearAlgebra, and GenericLinearAlgebra loaded together). CI may
    set `SDPX_PROVIDER_SMOKE_TARGET=mfla` to exercise the public MFLA provider
    independently; the default target remains `all` and requires both.

    The smoke deliberately uses the real installed/local MFLA and BFLA
    packages: no provider result is fabricated and no package is cloned by
    this file.  `SDPX_MFLA_PROJECT` / `SDPX_BFLA_PROJECT`, when set, are only
    used to assert that the loaded modules come from those checkouts.

    Provider presence is part of the contract: this file must never report a
    silent pass when the providers are absent, and a broken loaded provider
    must fail the process.
=#
using SDPX
using Test
using LinearAlgebra
using Random
using MultiFloats: Float64x2, Float64x3, Float64x4

const _SMOKE_TARGET = Symbol(lowercase(get(
    ENV, "SDPX_PROVIDER_SMOKE_TARGET", "all",
)))
_SMOKE_TARGET in (:all, :mfla, :bfla) || error(
    "SDPX_PROVIDER_SMOKE_TARGET must be all, mfla, or bfla",
)

const _MFLA_LOADED = try
    @eval import MultiFloatLinearAlgebra
    Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt) !== nothing
catch
    false
end

const _BFLA_LOADED = try
    @eval import BigFloatLinearAlgebra
    Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) !== nothing
catch
    false
end

const _GENERIC_LOADED = try
    @eval import GenericLinearAlgebra
    true
catch
    false
end

const _REQUIRE_MFLA = _SMOKE_TARGET in (:all, :mfla)
const _REQUIRE_BFLA = _SMOKE_TARGET in (:all, :bfla)

_REQUIRE_MFLA && !_MFLA_LOADED && error(
    "provider smoke target $(_SMOKE_TARGET) requires " *
    "MultiFloatLinearAlgebra; run scripts/provider_smoke.sh",
)
_REQUIRE_BFLA && !_BFLA_LOADED && error(
    "provider smoke target $(_SMOKE_TARGET) requires " *
    "BigFloatLinearAlgebra; run scripts/provider_smoke.sh",
)

const _PROVIDERS_LOADED =
    (!_REQUIRE_MFLA || _MFLA_LOADED) && (!_REQUIRE_BFLA || _BFLA_LOADED)

const LA = SDPX

function _assert_provider_root(environment::AbstractString, mod)
    project = get(ENV, environment, nothing)
    project === nothing && return
    root = realpath(project)
    @test startswith(realpath(pathof(mod)), root)
end

function _max_abs_error(x, y)
    return maximum(abs.(x .- y)) / max(maximum(abs.(y)), one(eltype(y)))
end

function _exercise_mfla_factors(::Type{T}) where {T}
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

    rng = MersenneTwister(2026_0813)
    n = 5
    R = T.(randn(rng, n, n))
    A = transpose(R) * R + T(8) * Matrix{T}(I, n, n)
    b = T.(randn(rng, n))
    chol = SDPX.la_cholesky_factor!(backend, copy(A))
    @test chol !== nothing
    x = copy(b)
    SDPX.la_factor_solve!(chol, x)
    @test _max_abs_error(A * x, b) <= T(1e-12)

    B = T.(randn(rng, n, n)) + T(2) * Matrix{T}(I, n, n)
    lu = SDPX.la_lu_factor!(backend, copy(B))
    @test lu !== nothing
    @test SDPX.la_factor_kind(lu) === :lu
    x = copy(b)
    SDPX.la_factor_solve!(lu, x)
    @test _max_abs_error(B * x, b) <= T(1e-12)

    C = T.(randn(rng, n, n))
    qr = SDPX.la_qr_factor!(
        backend,
        copy(C);
        pivoted=true,
        relative_tolerance=T(1e-12),
    )
    @test qr !== nothing
    @test SDPX.la_factor_rank(qr) == n
    @test length(SDPX.la_factor_permutation(qr)) == n
    permuted = SDPX.alloc_zeros(T, n)
    x = copy(b)
    SDPX.la_factor_solve!(qr, x, permuted)
    @test _max_abs_error(transpose(C) * (C * x), b) <= T(1e-10)

    D = T[
        4 1 0
        1 3 1
        0 1 -2
    ]
    ldlt = SDPX.la_ldlt_factor!(backend, copy(D))
    @test ldlt !== nothing
    @test SDPX.la_factor_kind(ldlt) === :ldlt
    @test sum(SDPX.la_ldlt_inertia(ldlt)) == size(D, 1)
    b2 = T[1, -1, 2]
    x = copy(b2)
    SDPX.la_ldlt_factor_solve!(ldlt, x)
    @test _max_abs_error(D * x, b2) <= T(1e-12)
    return backend
end

function _exercise_bfla_factors()
    config = LA.plan_la_backend(
        BigFloat;
        requested=:bfla,
        route=:dense_cholesky,
        threads=1,
    )
    @test config.selected === :bfla
    @test config.provider === :bigfloat_linear_algebra
    backend = LA.instantiate_la_backend(config, BigFloat, 1)
    @test backend isa LA.BFLALABackend

    rng = MersenneTwister(2026_0813)
    n = 5
    R = BigFloat.(randn(rng, n, n))
    A = transpose(R) * R + BigFloat(8) * Matrix{BigFloat}(I, n, n)
    b = BigFloat.(randn(rng, n))
    chol = SDPX.la_cholesky_factor!(
        backend,
        SDPX._owned_array_copy(BigFloat, A),
    )
    @test chol !== nothing
    x = SDPX._owned_array_copy(BigFloat, b)
    SDPX.la_factor_solve!(chol, x)
    @test _max_abs_error(A * x, b) <= big"1e-12"

    B = BigFloat.(randn(rng, n, n)) + BigFloat(2) * Matrix{BigFloat}(I, n, n)
    lu = SDPX.la_lu_factor!(
        backend,
        SDPX._owned_array_copy(BigFloat, B),
    )
    @test lu !== nothing
    @test SDPX.la_factor_kind(lu) === :lu
    x = SDPX._owned_array_copy(BigFloat, b)
    SDPX.la_factor_solve!(lu, x)
    @test _max_abs_error(B * x, b) <= big"1e-12"

    C = BigFloat.(randn(rng, n, n))
    qr = SDPX.la_qr_factor!(
        backend,
        SDPX._owned_array_copy(BigFloat, C);
        pivoted=true,
        relative_tolerance=big"1e-12",
    )
    @test qr !== nothing
    @test SDPX.la_factor_rank(qr) == n
    @test length(SDPX.la_factor_permutation(qr)) == n
    permuted = SDPX.alloc_zeros(BigFloat, n)
    x = SDPX._owned_array_copy(BigFloat, b)
    SDPX.la_factor_solve!(qr, x, permuted)
    @test _max_abs_error(transpose(C) * (C * x), b) <= big"1e-10"

    D = BigFloat[
        4 1 0
        1 3 1
        0 1 -2
    ]
    D_saved = SDPX._owned_array_copy(BigFloat, D)
    ldlt = SDPX.la_ldlt_factor!(
        backend,
        SDPX._owned_array_copy(BigFloat, D),
    )
    @test ldlt !== nothing
    @test SDPX.la_factor_kind(ldlt) === :ldlt
    @test sum(SDPX.la_ldlt_inertia(ldlt)) == size(D, 1)
    b2 = SDPX._owned_array_copy(BigFloat, BigFloat[1, -1, 2])
    x = SDPX._owned_array_copy(BigFloat, b2)
    SDPX.la_ldlt_factor_solve!(ldlt, x)
    @test _max_abs_error(D_saved * x, b2) <= big"1e-12"
    return backend
end

function _exercise_bfla_repeated_solve_correction()
    config = LA.plan_la_backend(
        BigFloat;
        requested=:bfla,
        route=:dense_cholesky,
        threads=1,
    )
    backend = LA.instantiate_la_backend(config, BigFloat, 1)

    rng = MersenneTwister(2026_0815)
    n = 4
    R = BigFloat.(randn(rng, n, n))
    A = transpose(R) * R + BigFloat(8) * Matrix{BigFloat}(I, n, n)
    rhs = BigFloat.(randn(rng, n))
    chol = SDPX.la_cholesky_factor!(
        backend,
        SDPX._owned_array_copy(BigFloat, A),
    )
    @test chol !== nothing
    first = SDPX._owned_array_copy(BigFloat, rhs)
    second = SDPX._owned_array_copy(BigFloat, rhs)
    SDPX.la_factor_solve!(chol, first)
    SDPX.la_factor_solve!(chol, second)
    @test _max_abs_error(A * first, rhs) <= big"1e-12"
    @test _max_abs_error(A * second, rhs) <= big"1e-12"

    square = BigFloat.(randn(rng, n, n)) + BigFloat(2) * Matrix{BigFloat}(I, n, n)
    lu = SDPX.la_lu_factor!(
        backend,
        SDPX._owned_array_copy(BigFloat, square),
    )
    @test lu !== nothing
    first = SDPX._owned_array_copy(BigFloat, rhs)
    second = SDPX._owned_array_copy(BigFloat, rhs)
    SDPX.la_factor_solve!(lu, first)
    SDPX.la_factor_solve!(lu, second)
    @test _max_abs_error(square * first, rhs) <= big"1e-12"
    @test _max_abs_error(square * second, rhs) <= big"1e-12"

    D = BigFloat[
        4 1 0
        1 3 1
        0 1 -2
    ]
    ldlt = SDPX.la_ldlt_factor!(
        backend,
        SDPX._owned_array_copy(BigFloat, D),
    )
    @test ldlt !== nothing
    first = SDPX._owned_array_copy(BigFloat, BigFloat[1, -1, 2])
    second = SDPX._owned_array_copy(BigFloat, first)
    SDPX.la_ldlt_factor_solve!(ldlt, first)
    SDPX.la_ldlt_factor_solve!(ldlt, second)
    @test _max_abs_error(D * first, BigFloat[1, -1, 2]) <= big"1e-12"
    @test _max_abs_error(D * second, BigFloat[1, -1, 2]) <= big"1e-12"

    for (factor, matrix, residual_source) in (
        (chol, A, rhs),
        (lu, square, rhs),
        (ldlt, D, BigFloat[1, -1, 2]),
    )
        residual = SDPX._owned_array_copy(BigFloat, residual_source)
        correction = SDPX.alloc_zeros(BigFloat, length(residual_source))
        SDPX.la_refinement_correction!(factor, residual, correction)
        @test all(isfinite, correction)
        @test _max_abs_error(matrix * correction, residual) <= big"1e-12"
    end
    return backend
end

function _exercise_mfla_mixed_residual()
    config = LA.plan_la_backend(
        Float64x2;
        requested=:multifloat,
        route=:dense_cholesky,
        threads=1,
    )
    backend = LA.instantiate_la_backend(config, Float64x2, 1)
    A = Float64x2[2 1; 1 3]
    x = Float64x2[1, 2]
    rhs = A * x
    residual = zeros(Float64x4, 2)
    SDPX.la_mixed_residual!(backend, A, x, rhs, residual)
    @test residual == zeros(Float64x4, 2)
    return residual
end

function _dense_lp_matrix(::Type{T}) where {T}
    G = SDPX.alloc_zeros(T, 3, 2)
    G[1, 1] = one(T)
    G[2, 2] = one(T)
    G[3, 1] = one(T)
    G[3, 2] = one(T)
    return G
end

@testset "installed provider smoke" begin
    if _REQUIRE_MFLA
        @testset "MFLA Float64x2/x3/x4 factors" begin
            _assert_provider_root("SDPX_MFLA_PROJECT", MultiFloatLinearAlgebra)
            for T in (Float64x2, Float64x3, Float64x4)
                @testset "factors $T" begin
                    _exercise_mfla_factors(T)
                end
            end
        end

        @testset "MFLA mixed residual Float64x2 -> Float64x4" begin
            _exercise_mfla_mixed_residual()
        end

    end

    if _REQUIRE_BFLA
        @testset "BFLA BigFloat factors" begin
            setprecision(BigFloat, 256) do
                _assert_provider_root("SDPX_BFLA_PROJECT", BigFloatLinearAlgebra)
                _exercise_bfla_factors()
                _exercise_bfla_repeated_solve_correction()
            end
        end

    end

    @testset "GenericLinearAlgebra reference role" begin
        for T in (Float64x4, BigFloat)
            config = LA.plan_la_backend(
                T;
                requested=:standard,
                route=:dense_cholesky,
                threads=1,
            )
            @test config.selected === :standard
            @test config.provider === :generic_linear_algebra
            @test config.provider_implementation ===
                  :julia_generic_with_gla_loaded
            @test config.ownership === :owned_mutable_scalars
        end
        @test _GENERIC_LOADED
    end
end
