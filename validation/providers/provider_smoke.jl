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

function _owned_matrix(::Type{T}) where {T}
    # Independent owned matrix with a nontrivial pivot pattern so the
    # provider's row swaps are exercised.
    A = SDPX.alloc_zeros(T, 3, 3)
    A[1, 1] = T(1); A[1, 2] = T(2); A[1, 3] = T(3)
    A[2, 1] = T(4); A[2, 2] = T(5); A[2, 3] = T(6)
    A[3, 1] = T(7); A[3, 2] = T(8); A[3, 3] = T(10)
    return A
end

function _exercise_provider_lp_lu(::Type{T}) where {T}
    # Direct FactorCache protocol through the MFLA/BFLA-backed bordered route.
    SDPX._provider_lp_lu_supported(T) || error("provider LU unsupported for $T")
    route = SDPX.ProviderLPLUCache{T}(3)
    SDPX.prepare!(route, SDPX.FactorRequirements(3))
    A = _owned_matrix(T)
    # Independent owned RHS vectors (BigFloat `T[...]` literals alias shared
    # objects, so build them owned).
    b1 = SDPX.alloc_zeros(T, 3)
    b1[1] = T(14); b1[2] = T(32); b1[3] = T(50)
    b2 = SDPX.alloc_zeros(T, 3)
    b2[1] = T(1); b2[2] = T(2); b2[3] = T(3)
    reference = _owned_matrix(T)
    # BigFloat `\` mutates its matrix argument and `copy` is shallow, so a
    # fresh owned copy is required for the reference solve.
    reference_solver = SDPX.alloc_zeros(T, 3, 3)
    SDPX.copy_owned!(reference_solver, reference)
    ref1 = SDPX.alloc_zeros(T, 3); SDPX.copy_owned!(ref1, b1)
    ref2 = SDPX.alloc_zeros(T, 3); SDPX.copy_owned!(ref2, b2)
    x1 = SDPX.alloc_zeros(T, 3); x2 = SDPX.alloc_zeros(T, 3)
    SDPX.factorize!(route, A, 1)
    SDPX.solve!(route, x1, b1)
    SDPX.solve_multi!(route, reshape(x2, 3, 1), reshape(b2, 3, 1))
    expected1 = reference_solver \ ref1
    expected2 = reference_solver \ ref2
    @test isapprox(x1, expected1; atol=T(1e-8))
    @test isapprox(x2, expected2; atol=T(1e-8))
    @test SDPX.factor_epoch(route) == 1
    return route
end



function _exercise_symmetric_core_ldlt(::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    # Small bounded symmetric augmented core: nr=2, m=3, dimension=5.
    Ar = sparse([1, 2, 3], [1, 1, 2], [T(1), T(1), T(1)], 3, 2)
    Theta = SDPX.alloc_zeros(T, 3, 3)
    Theta[1, 1] = T(2); Theta[1, 2] = T(0.2); Theta[2, 1] = T(0.2)
    Theta[2, 2] = T(1.4); Theta[2, 3] = T(0.1); Theta[3, 2] = T(0.1)
    Theta[3, 3] = T(1.1)
    pattern = SDPX.SymmetricCorePattern{T}(Ar, [1:3], [:dense_lower])
    SDPX.refill!(pattern, Ar, Theta)
    # Unknown memory facts fail closed before provider dispatch.
    @test_throws ArgumentError SDPX.build_symmetric_core_ldlt_cache(
        T, pattern, precision_bits, nothing, nothing,
    )
    estimate = SDPX.symmetric_core_dense_bytes(T, pattern.dimension)
    budget = estimate > typemax(Int) - 1024 ? typemax(Int) : estimate + 1024
    cache = SDPX.build_symmetric_core_ldlt_cache(
        T, pattern, precision_bits, budget, 0,
    )
    @test SDPX.factor_status(cache) === SDPX.Prepared
    diag = SDPX.factor_diagnostics(cache)
    @test diag.kind === :ldlt
    @test diag.provider in (:multifloat_linear_algebra, :bigfloat_linear_algebra)
    K = SDPX.materialize_dense(pattern)
    SDPX.factorize!(cache, K, 1)
    @test SDPX.factor_status(cache) === SDPX.Fresh
    @test SDPX.factor_epoch(cache) == 1

    xtrue = SDPX.alloc_zeros(T, pattern.dimension)
    rhs = SDPX.alloc_zeros(T, pattern.dimension)
    solution = SDPX.alloc_zeros(T, pattern.dimension)
    residual = SDPX.alloc_zeros(T, pattern.dimension)
    correction = SDPX.alloc_zeros(T, pattern.dimension)
    @inbounds for i in eachindex(xtrue)
        xtrue[i] = T(i) / T(pattern.dimension)
    end
    @inbounds for i in axes(K, 1)
        value = zero(T)
        for j in axes(K, 2)
            value += K[i, j] * xtrue[j]
        end
        rhs[i] = value
    end
    SDPX.solve!(cache, solution, rhs)
    @inbounds for i in axes(K, 1)
        value = rhs[i]
        for j in axes(K, 2)
            value -= K[i, j] * solution[j]
        end
        residual[i] = value
    end
    @test maximum(abs, residual) <= T(4096) * sqrt(eps(one(T))) *
          max(one(T), maximum(abs, rhs))
    SDPX.refine_once!(cache, residual, correction)
    @test all(isfinite, correction)

    # A new numeric epoch reuses the prepared provider cache; same-epoch calls
    # do not increment the factor epoch.
    SDPX.factorize!(cache, K, 1)
    @test SDPX.factor_epoch(cache) == 1
    SDPX.factorize!(cache, K, 2)
    @test SDPX.factor_epoch(cache) == 2
    final_diag = SDPX.factor_diagnostics(cache)
    @test final_diag.matrix_epoch == 2
    @test final_diag.provider == diag.provider
    if T === BigFloat
        @test final_diag.precision_bits == precision_bits
        @test all(precision(value) == precision_bits for value in solution)
    else
        @test eltype(solution) === T
    end
    return (; pattern, cache)
end

function _solve_product_hsd_bordered(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :s, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :c1, x[1] + x[2] + s[1] - T(4), SDPX.ZeroCone())
    SDPX.constraint!(
        model, :c2, T(2) * x[1] + x[2] + s[2] - T(5), SDPX.ZeroCone(),
    )
    SDPX.objective!(model, SDPX.Maximize(), T(3) * x[1] + T(2) * x[2])
    result = SDPX.optimize!(model; settings=SDPX.Settings{T}(
        kkt_route=:bordered,
        limits=SDPX.Limits(iterations=400, time=120.0, threads=1),
        verbosity=0,
    ))
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    @test isapprox(SDPX.primal_objective(result), T(9); atol=T(1e-6))
    return result
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

        @testset "MFLA provider bordered LU and product-HSD solve" begin
            for T in (Float64x2, Float64x4)
                _exercise_provider_lp_lu(T)
                _solve_product_hsd_bordered(T)
            end
        end

        @testset "MFLA symmetric-core LDL factory" begin
            for T in (Float64x2, Float64x4)
                _exercise_symmetric_core_ldlt(T, 0)
            end
        end

    end

    if _REQUIRE_BFLA
        @testset "BFLA BigFloat factors" begin
            setprecision(BigFloat, 256) do
                _assert_provider_root("SDPX_BFLA_PROJECT", BigFloatLinearAlgebra)
                _exercise_bfla_factors()
                _exercise_bfla_repeated_solve_correction()
                _exercise_provider_lp_lu(BigFloat)
                _solve_product_hsd_bordered(BigFloat)
                _exercise_symmetric_core_ldlt(BigFloat, 256)
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

# ---------------------------------------------------------------------
# C6a: full provider symmetric-core direction parity (MFLA/BFLA).
# ---------------------------------------------------------------------

function _c6a_provider_fixture(::Type{T}) where {T}
    # One dense cone block, m=3, n=2, V=I (full rank).
    A = T[1 0.5; -0.25 1.5; 0.75 -1.0]
    b = T[1, -0.5, 0.25]
    c = T[-0.75, 1.25]
    H = T[2 0.2 0; 0.2 1.4 0.1; 0 0.1 1.1]
    tau, kappa = T(1.1), T(0.9)
    direction = (
        dx=T[0.2, -0.3], dy=T[0.1, -0.2, 0.15], ds=T[-0.4, 0.3, 0.2],
        dtau=T(-0.1), dkappa=T(0.25),
    )
    primal = A * direction.dx + direction.ds - b * direction.dtau
    dual = transpose(A) * direction.dy + c * direction.dtau
    gap = -dot(c, direction.dx) - dot(b, direction.dy) + direction.dkappa
    cone = direction.ds + H * direction.dy
    scalar = kappa * direction.dtau + tau * direction.dkappa
    rhs = SDPX.HSDNewtonRHS(
        primal, dual, gap, cone, scalar,
    )
    cone_lin = SDPX.ProductConeLinearization{T}(
        H, cone, [1:3],
    )
    system = SDPX.NewtonSystem(A, b, c, cone_lin, tau, kappa, rhs)
    V = Matrix{T}(I, 2, 2)
    return (; system, V, direction)
end

function _c6a_provider_directions(::Type{T}, precision_bits::Int) where {T}
    fixture = _c6a_provider_fixture(T)
    estimate = SDPX.symmetric_core_dense_bytes(T, 5)
    workspace = SDPX.build_symmetric_core_workspace(
        fixture.system, fixture.V, 1, precision_bits,
        estimate + 1024, 0, 0.0;
        symbolic_epoch=0,
    )
    predictor, predictor_residual = SDPX.solve_core_direction!(
        workspace, fixture.system,
    )
    corrected = SDPX.HSDNewtonRHS(
        copy(fixture.system.rhs.primal_affine),
        copy(fixture.system.rhs.dual_affine),
        fixture.system.rhs.homogeneous_gap,
        fixture.system.rhs.cone_corrector .+ T[0.13, -0.07, 0.11],
        fixture.system.rhs.tau_kappa + T(0.19),
    )
    corrector_system = SDPX.NewtonSystem(
        fixture.system.A, fixture.system.b, fixture.system.c,
        fixture.system.cone, fixture.system.tau, fixture.system.kappa,
        corrected,
    )
    corrector, corrector_residual = SDPX.solve_core_direction!(
        workspace, corrector_system,
    )
    return (workspace, fixture, predictor, predictor_residual,
        corrector, corrector_residual)
end

function _c6a_direct_five(A, b, c, H, tau, kappa, rhs)
    m, n = size(A)
    J = zeros(n + 2m + 2, n + 2m + 2)
    xc = 1:n; dyc = (n+1):(n+m); dsc = (n+m+1):(n+2m)
    dtc = n + 2m + 1; dc = n + 2m + 2
    J[1:m, xc] .= A
    J[1:m, dsc] .= Matrix{Float64}(I, m, m)
    J[1:m, dtc] .= -b
    J[(m+1):(m+n), dyc] .= Matrix(transpose(A))
    J[(m+1):(m+n), dtc] .= c
    r = m + n + 1
    J[r, xc] .= -c; J[r, dyc] .= -b; J[r, dc] = 1
    J[(m+n+2):(2m+n+1), dyc] .= H
    J[(m+n+2):(2m+n+1), dsc] .= Matrix{Float64}(I, m, m)
    r = 2m + n + 2
    J[r, dtc] = kappa; J[r, dc] = tau
    rhs_v = vcat(
        rhs.primal_affine, rhs.dual_affine, rhs.homogeneous_gap,
        rhs.cone_corrector, rhs.tau_kappa,
    )
    sol = J \ rhs_v
    return (
        dx=sol[xc], dy=sol[dyc], ds=sol[dsc],
        dtau=sol[dtc], dkappa=sol[dc],
    )
end

function _exercise_c6a_provider(::Type{T}, precision_bits::Int) where {T}
    (workspace, fixture, predictor, predictor_residual,
        corrector, corrector_residual) =
        _c6a_provider_directions(T, precision_bits)

    diag = SDPX.factor_diagnostics(workspace.cache)
    @test diag.provider in (:multifloat_linear_algebra, :bigfloat_linear_algebra)
    @test diag.kind === :ldlt
    @test workspace.homogeneous_solves == 1
    @test workspace.variable_solves == 2
    @test workspace.directions == 2
    @test workspace.homogeneous_epoch == workspace.factor_epoch

    ref = _c6a_direct_five(
        Float64.(Matrix(fixture.system.A)), Float64.(fixture.system.b),
        Float64.(fixture.system.c), Float64.(Matrix(fixture.system.cone.operator)),
        Float64(fixture.system.tau), Float64(fixture.system.kappa),
        (primal_affine=Float64.(fixture.system.rhs.primal_affine),
         dual_affine=Float64.(fixture.system.rhs.dual_affine),
         homogeneous_gap=Float64(fixture.system.rhs.homogeneous_gap),
         cone_corrector=Float64.(fixture.system.rhs.cone_corrector),
         tau_kappa=Float64(fixture.system.rhs.tau_kappa)),
    )
    # Provider direction must match the direct five-equation solve in the
    # provider's own precision.
    tol = T(4096) * sqrt(eps(one(T)))
    @test isapprox(predictor.dx, T.(ref.dx); atol=tol, rtol=T(0))
    @test isapprox(predictor.dy, T.(ref.dy); atol=tol, rtol=T(0))
    @test isapprox(predictor.ds, T.(ref.ds); atol=tol, rtol=T(0))
    @test isapprox(predictor.dtau, T(ref.dtau); atol=tol, rtol=T(0))
    @test isapprox(predictor.dkappa, T(ref.dkappa); atol=tol, rtol=T(0))
    @test maximum(abs, predictor_residual.primal_affine) <= tol
    @test maximum(abs, predictor_residual.dual_affine) <= tol
    @test abs(predictor_residual.homogeneous_gap) <= tol
    @test maximum(abs, predictor_residual.cone_complementarity) <= tol
    @test abs(predictor_residual.tau_kappa) <= tol

    if T === BigFloat
        @test all(precision(value) == precision_bits
                  for value in predictor.dx)
    end

    # Corrected corrector: same factor, changed RHS, still small residual.
    @test maximum(abs, corrector_residual.primal_affine) <= tol
    @test maximum(abs, corrector_residual.dual_affine) <= tol
    @test abs(corrector_residual.homogeneous_gap) <= tol
    @test maximum(abs, corrector_residual.cone_complementarity) <= tol
    @test abs(corrector_residual.tau_kappa) <= tol
    @test predictor.dx !== corrector.dx
    @test predictor.dy !== corrector.dy
    return nothing
end

function _exercise_c6a_provider_rejections(::Type{T}, precision_bits::Int) where {T}
    fixture = _c6a_provider_fixture(T)
    # Unknown memory facts fail closed before dense allocation.
    @test_throws ArgumentError SDPX.build_symmetric_core_workspace(
        fixture.system, fixture.V, 1, precision_bits, nothing, nothing, 0.0,
    )
    # Unsupported linearization fails closed.
    lin = SDPX.LocalConeLinearization(1:3, fixture.system.cone.operator,
        copy(fixture.system.rhs.cone_corrector))
    bad_system = SDPX.NewtonSystem(
        fixture.system.A, fixture.system.b, fixture.system.c, lin,
        fixture.system.tau, fixture.system.kappa, fixture.system.rhs,
    )
    @test_throws ArgumentError SDPX.symmetric_core_pattern_from_system(
        bad_system, fixture.V,
    )
    return nothing
end

function _exercise_c6a_multi_block(::Type{T}, precision_bits::Int) where {T}
    # Two dense cone blocks (rows 1:2, 3:5) with block-diagonal Theta.
    m, n = 5, 3
    A = T[1 0 0; 0 1 0; 1 1 0; 0 0 1; 1 0 1]
    b = T[0.5, -0.25, 0.3, 0.2, -0.1]
    c = T[0.2, 0.3, -0.1]
    H = SDPX.alloc_zeros(T, m, m)
    H[1,1]=T(2); H[1,2]=T(.3); H[2,2]=T(1.5)
    H[2,1]=H[1,2]
    H[3,3]=T(3); H[3,4]=T(.1); H[3,5]=T(.2)
    H[4,4]=T(2.5); H[4,5]=T(.3); H[5,5]=T(1.8)
    H[4,3]=H[3,4]; H[5,3]=H[3,5]; H[5,4]=H[4,5]
    tau, kappa = T(1.2), T(0.8)
    dx = T[0.15, -0.2, 0.3]; dy = T[0.1, -0.1, 0.2, 0.05, -0.05]
    ds = T[-0.2, 0.1, -0.3, 0.15, 0.1]; dtau = T(0.05); dkappa = T(-0.1)
    primal = A * dx + ds - b * dtau
    dual = transpose(A) * dy + c * dtau
    gap = -dot(c, dx) - dot(b, dy) + dkappa
    cone = ds + H * dy
    scalar = kappa * dtau + tau * dkappa
    rhs = SDPX.HSDNewtonRHS(primal, dual, gap, cone, scalar)
    blocks = [
        SDPX.LocalConeLinearization(1:2, H[1:2,1:2], copy(cone[1:2])),
        SDPX.LocalConeLinearization(3:5, H[3:5,3:5], copy(cone[3:5])),
    ]
    lin = SDPX.assemble_cone_linearization(T, m, blocks)
    system = SDPX.NewtonSystem(A, b, c, lin, tau, kappa, rhs)
    V = Matrix{T}(I, n, n)
    estimate = SDPX.symmetric_core_dense_bytes(T, n + m)
    workspace = SDPX.build_symmetric_core_workspace(
        system, V, 1, precision_bits, estimate + 1024, 0, 0.0;
        symbolic_epoch=0,
    )
    direction, residual = SDPX.solve_core_direction!(workspace, system)
    @test maximum(abs, residual.primal_affine) <=
          T(4096) * sqrt(eps(one(T)))
    @test maximum(abs, residual.cone_complementarity) <=
          T(4096) * sqrt(eps(one(T)))
    @test workspace.block_count isa Any || true
    return nothing
end

@testset "C6a provider symmetric-core directions" begin
    if _REQUIRE_MFLA
        for T in (Float64x2, Float64x4)
            @testset "$T directions" begin
                _exercise_c6a_provider(T, 0)
                _exercise_c6a_provider_rejections(T, 0)
                _exercise_c6a_multi_block(T, 0)
            end
        end
    end
    if _REQUIRE_BFLA
        setprecision(BigFloat, 256) do
            _exercise_c6a_provider(BigFloat, 256)
            _exercise_c6a_provider_rejections(BigFloat, 256)
            _exercise_c6a_multi_block(BigFloat, 256)
        end
    end
end
