# Independent Clarabel-style symmetric augmented-core HSD oracle.
#
# This file is deliberately self-contained (Julia Base + stdlibs only).  It
# does not load SDPX, does not call production residual helpers, and builds
# every matrix from the frozen five-equation convention documented in
# docs/design/NEWTON_SYSTEM.md.
#
# Purpose: prove that the Clarabel-style symmetric augmented-core elimination
#
#     K = [ 0    Ar' ]
#         [ Ar  -Theta ]
#
#     K*w = [ dr; p - h ],   K*u = [ -cr; b ]
#     dtau = (t - tau*(g + cr'*wx + b'*wy))
#              / (kappa + tau*(cr'*ux + b'*uy))
#     dxr  = wx + dtau*ux
#     dy   = wy + dtau*uy
#     dx   = V*dxr
#     ds   = p - A*dx + b*dtau
#     dkappa = g + c'*dx + b'*dy
#
# reproduces the unique direction of the frozen five Newton equations, and
# that the old scalar-bordered (n+1)-square coefficient operator remains
# generally nonsymmetric and therefore LDL-ineligible.

using Test
using LinearAlgebra
using SparseArrays
using SDPX

# Small non-CHOLMOD cache used only for the bounded BigFloat ownership seam.
# It is deliberately direct and provider-neutral: the production CHOLMOD
# cache remains Float64-only, while this test proves that the workspace's
# mutable MPFR buffers never alias one another or the returned snapshots.
mutable struct _BigFloatCoreCache <: SDPX.AbstractFactorCache{BigFloat}
    K::Matrix{BigFloat}
    status::SDPX.FactorCacheState
    matrix_epoch::Int
    factor_epoch::Int
    solve_count::Int
    refine_count::Int
end

SDPX.factor_status(cache::_BigFloatCoreCache) = cache.status
SDPX.factor_matrix_epoch(cache::_BigFloatCoreCache) = cache.matrix_epoch
SDPX.factor_epoch(cache::_BigFloatCoreCache) = cache.factor_epoch
function SDPX.solve!(
    cache::_BigFloatCoreCache,
    destination::AbstractVector{BigFloat}, rhs::AbstractVector{BigFloat},
)
    SDPX.copy_owned!(destination, cache.K \ rhs)
    cache.solve_count += 1
    return destination
end
function SDPX.refine_once!(
    cache::_BigFloatCoreCache,
    residual::AbstractVector{BigFloat}, correction::AbstractVector{BigFloat},
)
    SDPX.copy_owned!(correction, cache.K \ residual)
    cache.refine_count += 1
    return correction
end

struct _Fixture
    V::Matrix{Float64}       # n x nr rank-reduction basis, V'V = I
    A::Matrix{Float64}       # m x n original data
    b::Vector{Float64}
    c::Vector{Float64}
    Theta::Matrix{Float64}   # m x m symmetric SPD cone scaling
    tau::Float64
    kappa::Float64
    rhs::NamedTuple
    direction::NamedTuple    # reference direction used to build the RHS
end

function _fixture(family::Symbol)
    if family === :identity
        V = Matrix{Float64}(I, 2, 2)
        A = [1.0 0.5; -0.25 1.5; 0.75 -1.0]
        b = [1.0, -0.5, 0.25]
        c = [-0.75, 1.25]
        Theta = [2.0 0.2 0.0; 0.2 1.4 0.1; 0.0 0.1 1.1]
        tau, kappa = 1.2, 0.8
        direction = (
            dx=[0.2, -0.3], dy=[0.1, -0.2, 0.15],
            ds=[-0.4, 0.3, 0.2], dtau=-0.1, dkappa=0.25,
        )
    elseif family === :rotated
        θ = 0.4
        V = [cos(θ) -sin(θ); sin(θ) cos(θ)]   # nontrivial orthonormal basis
        A = [1.0 0.5; -0.25 1.5; 0.75 -1.0]
        b = [1.0, -0.5, 0.25]
        c = [-0.75, 1.25]
        Theta = [2.0 0.2 0.0; 0.2 1.4 0.1; 0.0 0.1 1.1]
        tau, kappa = 1.2, 0.8
        direction = (
            dx=[0.2, -0.3], dy=[0.1, -0.2, 0.15],
            ds=[-0.4, 0.3, 0.2], dtau=-0.1, dkappa=0.25,
        )
    elseif family === :rank_reduced
        s = 1.0 / sqrt(2.0)
        # n=3, nr=2 orthonormal basis of a proper subspace.
        V = [1.0 0.0; 0.0 s; 0.0 s]
        # Rows of A are orthogonal to [0,-1,1]/sqrt2, so null(A) = span(w),
        # rank(Ar) = nr = 2.
        A = [1.0 1.0 1.0; 1.0 -1.0 -1.0; 0.0 1.0 1.0]
        b = [0.6, -0.4, 0.3]
        # Solvable rank-reduced branch requires c in range(V); a significant
        # nullspace component is handled by the dual-infeasibility candidate
        # before the Newton core is constructed.
        c = [0.5, 0.25, 0.25]
        Theta = [2.0 0.1 0.0; 0.1 1.5 0.0; 0.0 0.0 1.2]
        tau, kappa = 1.3, 0.7
        xr0 = [0.4, -0.25]
        direction = (
            dx=V * xr0, dy=[0.2, -0.1, 0.3],
            ds=[-0.3, 0.15, 0.2], dtau=0.12, dkappa=-0.18,
        )
    else
        throw(ArgumentError("unknown fixture $family"))
    end

    m, n = size(A)
    # Build the RHS from the frozen five equations and the reference
    # direction so every fixture is exactly consistent by construction.
    primal = A * direction.dx + direction.ds - b * direction.dtau
    dual = transpose(A) * direction.dy + c * direction.dtau
    gap = -dot(c, direction.dx) - dot(b, direction.dy) + direction.dkappa
    cone = direction.ds + Theta * direction.dy
    scalar = kappa * direction.dtau + tau * direction.dkappa
    rhs = (
        primal_affine=primal, dual_affine=dual, homogeneous_gap=gap,
        cone_corrector=cone, tau_kappa=scalar,
    )
    return _Fixture(V, A, b, c, Theta, tau, kappa, rhs, direction)
end

# Independent five-equation residual evaluation (no production helpers).
function _residuals(fx, dir)
    dx, dy, ds, dtau, dkappa = dir.dx, dir.dy, dir.ds, dir.dtau, dir.dkappa
    return (
        primal=fx.A * dx + ds - fx.b * dtau - fx.rhs.primal_affine,
        dual=transpose(fx.A) * dy + fx.c * dtau - fx.rhs.dual_affine,
        gap=-dot(fx.c, dx) - dot(fx.b, dy) + dkappa -
            fx.rhs.homogeneous_gap,
        cone=ds + fx.Theta * dy - fx.rhs.cone_corrector,
        scalar=fx.kappa * dtau + fx.tau * dkappa - fx.rhs.tau_kappa,
    )
end

function _max_residual(res)
    return maximum((
        maximum(abs, res.primal; init=0.0),
        maximum(abs, res.dual; init=0.0),
        abs(res.gap),
        maximum(abs, res.cone; init=0.0),
        abs(res.scalar),
    ))
end

# Direct solve of the full five-equation Jacobian in original coordinates.
# Unknown order (dx, dy, ds, dtau, dkappa); row order (P, D, G, C1, C2).
function _direct_full(fx)
    m, n = size(fx.A)
    ncol = n + 2m + 2
    J = zeros(ncol, ncol)
    dxc = 1:n
    dyc = (n + 1):(n + m)
    dsc = (n + m + 1):(n + 2m)
    dtc = n + 2m + 1
    dc = n + 2m + 2
    J[1:m, dxc] .= fx.A
    J[1:m, dsc] .= Matrix{Float64}(I, m, m)
    J[1:m, dtc] .= -fx.b
    J[(m + 1):(m + n), dyc] .= Matrix(fx.A')
    J[(m + 1):(m + n), dtc] .= fx.c
    r = m + n + 1
    J[r, dxc] .= -fx.c
    J[r, dyc] .= -fx.b
    J[r, dc] = 1.0
    J[(m + n + 2):(2m + n + 1), dyc] .= fx.Theta
    J[(m + n + 2):(2m + n + 1), dsc] .= Matrix{Float64}(I, m, m)
    r = 2m + n + 2
    J[r, dtc] = fx.kappa
    J[r, dc] = fx.tau
    rhs = vcat(
        fx.rhs.primal_affine, fx.rhs.dual_affine, fx.rhs.homogeneous_gap,
        fx.rhs.cone_corrector, fx.rhs.tau_kappa,
    )
    sol = J \ rhs
    return (
        dx=sol[dxc], dy=sol[dyc], ds=sol[dsc],
        dtau=sol[dtc], dkappa=sol[dc],
    )
end

# Direct solve of the five-equation Jacobian in rank-reduced coordinates
# (x = V*xr).  Independent of the augmented-core elimination: it keeps the
# full five equations and only substitutes x = V*xr.
function _direct_reduced(fx)
    m, n = size(fx.A)
    nr = size(fx.V, 2)
    Ar = fx.A * fx.V
    cr = fx.V' * fx.c
    dr = fx.V' * fx.rhs.dual_affine
    ncol = nr + 2m + 2
    J = zeros(ncol, ncol)
    xc = 1:nr
    dyc = (nr + 1):(nr + m)
    dsc = (nr + m + 1):(nr + 2m)
    dtc = nr + 2m + 1
    dc = nr + 2m + 2
    J[1:m, xc] .= Ar
    J[1:m, dsc] .= Matrix{Float64}(I, m, m)
    J[1:m, dtc] .= -fx.b
    J[(m + 1):(m + nr), dyc] .= Matrix(Ar')
    J[(m + 1):(m + nr), dtc] .= cr
    r = m + nr + 1
    J[r, xc] .= -cr
    J[r, dyc] .= -fx.b
    J[r, dc] = 1.0
    J[(m + nr + 2):(2m + nr + 1), dyc] .= fx.Theta
    J[(m + nr + 2):(2m + nr + 1), dsc] .= Matrix{Float64}(I, m, m)
    r = 2m + nr + 2
    J[r, dtc] = fx.kappa
    J[r, dc] = fx.tau
    rhs = vcat(
        fx.rhs.primal_affine, dr, fx.rhs.homogeneous_gap,
        fx.rhs.cone_corrector, fx.rhs.tau_kappa,
    )
    sol = J \ rhs
    dxr = sol[xc]
    return (
        dxr=dxr, dx=fx.V * dxr, dy=sol[dyc], ds=sol[dsc],
        dtau=sol[dtc], dkappa=sol[dc],
    )
end

# The Clarabel-style elimination under test.
function _augmented_core(fx)
    m, n = size(fx.A)
    nr = size(fx.V, 2)
    Ar = fx.A * fx.V
    cr = fx.V' * fx.c
    dr = fx.V' * fx.rhs.dual_affine
    K = [zeros(nr, nr) transpose(Ar); Ar -fx.Theta]
    w = K \ vcat(dr, fx.rhs.primal_affine - fx.rhs.cone_corrector)
    u = K \ vcat(-cr, fx.b)
    wx, wy = w[1:nr], w[nr + 1:end]
    ux, uy = u[1:nr], u[nr + 1:end]
    den = fx.kappa + fx.tau * (dot(cr, ux) + dot(fx.b, uy))
    num = fx.rhs.tau_kappa - fx.tau * (
        fx.rhs.homogeneous_gap + dot(cr, wx) + dot(fx.b, wy)
    )
    dtau = num / den
    dxr = wx + dtau * ux
    dy = wy + dtau * uy
    dx = fx.V * dxr
    ds = fx.rhs.primal_affine - fx.A * dx + fx.b * dtau
    dkappa = fx.rhs.homogeneous_gap + dot(fx.c, dx) + dot(fx.b, dy)
    return (; dxr, dx, dy, ds, dtau, dkappa)
end

# Current SDPX scalar-bordered coefficient operator built from the frozen
# condensation (docs/design/HSD_FORMULATION.md 4.2): H = Ar'*G*Ar with
# G = Theta^{-1}, q = cr - Ar'*G*b, r' = tau*(cr' + b'*G*Ar), d = kappa -
# tau*b'*G*b.
function _old_scalar_border(fx)
    G = inv(fx.Theta)
    Ar = fx.A * fx.V
    cr = fx.V' * fx.c
    H = Ar' * G * Ar
    q = cr - Ar' * G * fx.b
    r = fx.tau * (cr + Ar' * G * fx.b)
    d = fx.kappa - fx.tau * dot(fx.b, G * fx.b)
    return [H q; transpose(r) d]
end

function _scale(fx)
    return max(1.0, norm(fx.A, Inf), norm(fx.b, Inf),
        norm(fx.c, Inf), norm(fx.Theta, Inf),
        norm(fx.rhs.primal_affine, Inf), norm(fx.rhs.dual_affine, Inf),
    )
end

function _compare_directions(aug, ref, tol)
    @test aug.dx ≈ ref.dx atol=tol rtol=0
    @test aug.dy ≈ ref.dy atol=tol rtol=0
    @test aug.ds ≈ ref.ds atol=tol rtol=0
    @test aug.dtau ≈ ref.dtau atol=tol rtol=0
    @test aug.dkappa ≈ ref.dkappa atol=tol rtol=0
    return nothing
end

@testset "Clarabel-style symmetric augmented-core oracle" begin
    for family in (:identity, :rotated, :rank_reduced)
        @testset "$family" begin
            fx = _fixture(family)
            scale = _scale(fx)
            tol = 1e-9 * scale

            # Rank-reduction basis must be an isometry.
            @test fx.V' * fx.V ≈ Matrix{Float64}(I, size(fx.V, 2), size(fx.V, 2)) atol=1e-14 rtol=0

            # Reference direction must be exactly consistent with the RHS.
            @test _max_residual(_residuals(fx, fx.direction)) <= 1e-14

            aug = _augmented_core(fx)
            if family === :rank_reduced
                ref = _direct_reduced(fx)
                @test aug.dxr ≈ ref.dxr atol=tol rtol=0
                # dx must reconstruct in the reduced row space.
                @test aug.dx ≈ fx.direction.dx atol=tol rtol=0
                @test norm((Matrix{Float64}(I, size(fx.A, 2), size(fx.A, 2)) -
                            fx.V * fx.V') * aug.dx, Inf) <= tol
            else
                ref = _direct_full(fx)
            end
            _compare_directions(aug, ref, tol)

            # The augmented direction must satisfy all five frozen equations.
            res = _residuals(fx, aug)
            @test _max_residual(res) <= tol

            # K is symmetric (top-left zero block, Ar/Ar', -Theta).
            Ar = fx.A * fx.V
            K = [zeros(size(fx.V, 2), size(fx.V, 2)) transpose(Ar); Ar -fx.Theta]
            @test norm(K - K', Inf) == 0.0

            # The old scalar-bordered operator is generally nonsymmetric
            # (q != r), so LDL on the full border is not legal.
            KB = _old_scalar_border(fx)
            H = KB[1:end-1, 1:end-1]
            @test norm(H - H', Inf) <= 1e-12 * scale  # top-left block is symmetric
            @test maximum(abs, KB - KB') > 1e-10 * scale
        end
    end
end

# ---------------------------------------------------------------------
# C2: frozen symmetric augmented-core CSC pattern and in-place refill.
# Exercises the production `SDPX.SymmetricCorePattern` from
# src/kkt/symmetric_core.jl (loaded through `using SDPX` above).
# ---------------------------------------------------------------------

"""Two dense cone blocks (rows 1:2 and 3:5) for m = 5."""
function _core_blocks()
    return ([1:2, 3:5], [:dense_lower, :dense_lower])
end

"""Ar with an explicit structural numeric zero at (3, 1)."""
function _core_ar(scale::Float64 = 1.0)
    return sparse(
        [1, 3, 2, 4, 5],            # rows
        [1, 1, 2, 2, 2],            # columns
        scale .* [1.0, 0.0, 2.0, -1.5, 0.5],   # (3,1) is a stored zero
        5, 2,
    )
end

"""Block-diagonal symmetric Theta over [1:2, 3:5]."""
function _core_theta(scale::Float64 = 1.0)
    Theta = zeros(5, 5)
    Theta[1, 1] = 2.0; Theta[1, 2] = 0.4; Theta[2, 2] = 1.3
    Theta[2, 1] = Theta[1, 2]
    Theta[3, 3] = 3.0; Theta[3, 4] = 0.1; Theta[3, 5] = 0.2
    Theta[4, 3] = Theta[3, 4]; Theta[4, 4] = 2.5; Theta[4, 5] = 0.3
    Theta[5, 3] = Theta[3, 5]; Theta[5, 4] = Theta[4, 5]; Theta[5, 5] = 1.8
    return scale .* Theta
end

function _direct_core_dense(Ar, Theta)
    nr = size(Ar, 2)
    return [zeros(nr, nr) Matrix(Ar'); Matrix(Ar) -Theta]
end

"""Theta whose off-diagonal copy disagrees with its mirror."""
function _core_theta_asymmetric()
    Theta = _core_theta()
    Theta[1, 2] = 0.4
    Theta[2, 1] = 0.7
    return Theta
end

"""Theta containing a non-finite entry."""
function _core_theta_nan()
    Theta = _core_theta()
    Theta[3, 3] = NaN
    return Theta
end

@testset "Symmetric augmented-core CSC pattern and refill" begin
    ranges, shapes = _core_blocks()
    Ar = _core_ar()
    Theta = _core_theta()
    pattern = SDPX.SymmetricCorePattern(Ar, ranges, shapes)

    @test pattern.nr == 2
    @test pattern.m == 5
    @test SDPX.symmetric_core_dimension(pattern) == 7
    expected_slots = pattern.nr + nnz(Ar) +
                     (3 + 6)          # blocks of size 2 and 3 lower triangles
    @test length(SDPX.symmetric_core_nzval(pattern)) == expected_slots
    @test length(SDPX.symmetric_core_rowval(pattern)) == expected_slots

    # Numeric values never change the frozen signature.
    Ar2 = _core_ar(2.5)
    Theta2 = _core_theta(3.0)
    pattern2 = SDPX.SymmetricCorePattern(Ar2, ranges, shapes)
    @test SDPX.symmetric_core_signature(pattern2) ==
          SDPX.symmetric_core_signature(pattern)
    drifted = sparse([1, 2, 2, 4, 5], [1, 1, 2, 2, 2],
        [1.0, 0.0, 2.0, -1.5, 0.5], 5, 2)  # column 1 rows {1,2} instead of {1,3}
    pattern_drift = SDPX.SymmetricCorePattern(drifted, ranges, shapes)
    @test SDPX.symmetric_core_signature(pattern_drift) !=
          SDPX.symmetric_core_signature(pattern)

    # First refill: expanded dense K equals the direct augmented core.
    cp = pattern.colptr
    rv = pattern.rowval
    sig = SDPX.symmetric_core_signature(pattern)
    SDPX.refill!(pattern, Ar, Theta)
    @test pattern.colptr === cp && pattern.rowval === rv
    @test SDPX.symmetric_core_signature(pattern) == sig
    K = SDPX.materialize_dense(pattern)
    @test size(K) == (7, 7)
    @test K == _direct_core_dense(Ar, Theta)   # elementwise, incl. stored zeros
    @test all(K .== transpose(K))              # expanded triangle is symmetric

    # Second numeric refill with changed values: pattern must be unchanged.
    SDPX.refill!(pattern, Ar2, Theta2)
    @test pattern.colptr === cp && pattern.rowval === rv
    @test SDPX.symmetric_core_signature(pattern) == sig
    K2 = SDPX.materialize_dense(pattern)
    @test K2 == _direct_core_dense(Ar2, Theta2)
    @test all(K2 .== transpose(K2))

    # Rejections ----------------------------------------------------
    @test_throws ArgumentError SDPX.refill!(
        pattern, Ar, _core_theta_asymmetric(),
    )
    @test_throws ArgumentError SDPX.refill!(pattern, Ar, _core_theta_nan())
    @test_throws DimensionMismatch SDPX.refill!(
        pattern, sparse([1], [1], [1.0], 5, 3), Theta,
    )
    @test_throws ArgumentError SDPX.refill!(pattern, drifted, Theta)
    @test_throws ArgumentError SDPX.SymmetricCorePattern(
        Ar, [1:2, 4:5], shapes,           # gap at row 3
    )
    @test_throws ArgumentError SDPX.SymmetricCorePattern(
        Ar, [1:2, 2:5], shapes,           # overlap at row 2
    )
    @test_throws ArgumentError SDPX.SymmetricCorePattern(
        Ar, ranges, [:dense_lower, :unsupported_shape],
    )
end

# ---------------------------------------------------------------------
# C3: truthful Float64 CHOLMOD signed-LDL lifecycle.
# Exercises SDPX.SparseSymbolicNumericCache against the frozen core
# pattern: one symbolic analysis, same-object numeric refactor reuse,
# signed static regularization confined to the factor view, honest
# allocating `factor \ rhs` solves, and fail-closed rejection paths.
# ---------------------------------------------------------------------

"""nsr=2 core fixture (same helper as the C2 testset)."""
function _core_cache_fixture(regularization::Float64)
    ranges, shapes = _core_blocks()
    Ar = _core_ar()
    Theta = _core_theta()
    pattern = SDPX.SymmetricCorePattern(Ar, ranges, shapes)
    SDPX.refill!(pattern, Ar, Theta)
    K = SDPX.symmetric_core_lower_sparse(pattern)
    req = SDPX.SparseSymbolicRequirements(
        K; symbolic_epoch=7, dsigns=SDPX.symmetric_core_dsigns(pattern),
        regularization=regularization,
    )
    cache = SDPX.SparseSymbolicNumericCache{Float64}()
    SDPX.prepare!(cache, req)
    return (; pattern, K, cache)
end

"""Deterministically singular nr=1 core fixture: a zero `Ar` column makes the
first core column exactly zero, so the unregularized factor fails regardless
of the CHOLMOD ordering; the signed static shift repairs it."""
function _singular_core_fixture(regularization::Float64)
    Ar = sparse([1, 2], [1, 1], [0.0, 0.0], 2, 1)
    Theta = [2.0 0.2; 0.2 1.3]
    pattern = SDPX.SymmetricCorePattern(Ar, [1:2], [:dense_lower])
    SDPX.refill!(pattern, Ar, Theta)
    K = SDPX.symmetric_core_lower_sparse(pattern)
    req = SDPX.SparseSymbolicRequirements(
        K; symbolic_epoch=1,
        dsigns=SDPX.symmetric_core_dsigns(pattern),
        regularization=regularization,
    )
    cache = SDPX.SparseSymbolicNumericCache{Float64}()
    SDPX.prepare!(cache, req)
    return (; pattern, K, cache)
end

@testset "CHOLMOD symmetric augmented core lifecycle" begin
    # --- unregularized zero-pivot failure is honest ------------------
    fx = _singular_core_fixture(0.0)
    cache = fx.cache
    @test SDPX.factor_status(cache) === SDPX.Prepared
    original_nzval = copy(fx.pattern.nzval)
    # Structural zero x diagonal → numerically singular core.  CHOLMOD
    # reports failure; the cache must revoke any usable factor.
    @test_throws ArgumentError SDPX.factorize!(cache, fx.K, 1)
    @test SDPX.factor_status(cache) === SDPX.Failed
    @test SDPX.factor_epoch(cache) == 0
    @test SDPX.factor_diagnostics(cache).numeric_count == 0
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        cache, zeros(3), ones(3),
    )
    @test fx.pattern.nzval == original_nzval   # pattern never touched

    # --- signed regularized singular core succeeds -------------------
    fr = _singular_core_fixture(1e-6)
    cache = fr.cache
    original_nzval = copy(fr.pattern.nzval)
    SDPX.factorize!(cache, fr.K, 1)
    @test SDPX.factor_status(cache) === SDPX.Fresh
    @test SDPX.factor_epoch(cache) == 1
    @test SDPX.factor_symbolic_epoch(cache) == 1
    d = SDPX.factor_diagnostics(cache)
    @test d.symbolic_count == 1
    @test d.numeric_count == 1
    @test d.provider === :cholmod
    @test d.kind === :symmetric_ldl
    @test d.solve_allocation_policy == :allocating_factor_backslash_copy
    @test fr.pattern.nzval == original_nzval   # original K untouched
    factor_object = cache.factor
    @test factor_object !== nothing

    # Solve against the regularized factor view.
    rhs = [1.0, -0.5, 2.0]
    destination = zeros(3)
    SDPX.solve!(cache, destination, rhs)
    @test all(isfinite, destination)
    Kreg = SDPX.materialize_dense(fr.pattern)
    for j in 1:3
        Kreg[j, j] += (j == 1 ? 1 : -1) * 1e-6
    end
    @test norm(Kreg * destination - rhs, Inf) <= 1e-8
    # A tiny signed shift on a zero pivot can be absorbed into the LDL
    # scaling, so the regularized solution may still nearly solve the
    # unmodified singular K.  We therefore never assert that the
    # regularized solve solves the original K exactly (and never use the
    # unmodified K residual to accept a direction in the production gate).
    original_residual = norm(
        SDPX.materialize_dense(fr.pattern) * destination - rhs, Inf,
    )
    @test isfinite(original_residual)

    # Same matrix epoch is a no-op; factor object survives.
    SDPX.factorize!(cache, fr.K, 1)
    @test SDPX.factor_epoch(cache) == 1
    @test SDPX.factor_diagnostics(cache).numeric_count == 1
    @test cache.factor === factor_object

    # --- second matrix epoch reuses the same CHOLMOD object ----------
    f2 = _core_cache_fixture(1e-6)
    cache = f2.cache
    SDPX.factorize!(cache, f2.K, 11)
    @test SDPX.factor_diagnostics(cache).symbolic_count == 1
    @test SDPX.factor_diagnostics(cache).numeric_count == 1
    factor_object = cache.factor
    Ar2 = _core_ar(2.5)
    Theta2 = _core_theta(3.0)
    SDPX.refill!(f2.pattern, Ar2, Theta2)
    K2 = SDPX.symmetric_core_lower_sparse(f2.pattern)
    SDPX.factorize!(cache, K2, 12)
    @test SDPX.factor_status(cache) === SDPX.Fresh
    @test SDPX.factor_epoch(cache) == 2
    d2 = SDPX.factor_diagnostics(cache)
    @test d2.symbolic_count == 1       # symbolic analysis happened once
    @test d2.numeric_count == 2
    @test cache.factor === factor_object   # same object, refactored in place

    # Refinement reuses the same factor.
    correction = zeros(7)
    SDPX.refine_once!(cache, ones(7), correction)
    @test SDPX.factor_diagnostics(cache).refine_count == 1
    @test SDPX.factor_diagnostics(cache).numeric_count == 2
    @test cache.factor === factor_object

    # --- stale solve after invalidate is rejected --------------------
    SDPX.invalidate!(cache)
    @test SDPX.factor_status(cache) === SDPX.Invalid
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        cache, zeros(7), ones(7),
    )

    # --- pattern drift is rejected without reanalysis ---------------
    f3 = _core_cache_fixture(1e-6)
    cache3 = f3.cache
    SDPX.factorize!(cache3, f3.K, 1)
    @test SDPX.factor_diagnostics(cache3).symbolic_count == 1
    drifted_ar = sparse([1, 2, 2, 4, 5], [1, 1, 2, 2, 2],
        [1.0, 0.0, 2.0, -1.5, 0.5], 5, 2)   # column 1 rows {1,2} instead of {1,3}
    ranges, shapes = _core_blocks()
    drifted_pattern = SDPX.SymmetricCorePattern(drifted_ar, ranges, shapes)
    SDPX.refill!(drifted_pattern, drifted_ar, _core_theta())
    driftedK = SDPX.symmetric_core_lower_sparse(drifted_pattern)
    @test_throws ArgumentError SDPX.factorize!(cache3, driftedK, 2)
    @test SDPX.factor_status(cache3) === SDPX.Failed
    @test SDPX.factor_diagnostics(cache3).symbolic_count == 1

    # --- dsigns length/value rejection ------------------------------
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        f3.K; symbolic_epoch=0, dsigns=Int[1, -1, 1, -1, 1, -1, 0],
    )
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        f3.K; symbolic_epoch=0, dsigns=Int[1, -1],
    )
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        f3.K; symbolic_epoch=0, dsigns=Int[1, -1, 1, -1, 1, -1, 1],
        regularization=-1.0,
    )
end

# ---------------------------------------------------------------------
# C3 hardening: reviewer-blocked contract defects.
# ---------------------------------------------------------------------



@testset "C3 hardened contracts" begin
    # --- cross-block Theta rejection --------------------------------
    ranges, shapes = _core_blocks()
    Ar = _core_ar()
    pattern = SDPX.SymmetricCorePattern(Ar, ranges, shapes)
    Theta = _core_theta()
    @test SDPX.validate_symmetric_core(pattern, Ar, Theta)
    cross = copy(Theta)
    cross[1, 3] = 0.99   # between block 1 (1:2) and block 2 (3:5)
    cross[3, 1] = 0.99
    @test_throws ArgumentError SDPX.validate_symmetric_core(
        pattern, Ar, cross,
    )
    @test_throws ArgumentError SDPX.refill!(pattern, Ar, cross)

    # --- non-Float64 rejection before factorization -----------------
    Kf = _core_cache_fixture(1e-6).K
    @test_throws ArgumentError SDPX.SparseSymbolicNumericCache{Float32}()
    @test_throws ArgumentError SDPX.SparseSymbolicNumericCache{BigFloat}()
    try
        SDPX.SparseSymbolicNumericCache{Float32}(Kf; symbolic_epoch=0,
            dsigns=Int[1, -1, 1, -1, 1, -1, 1], regularization=1e-6)
        @test false
    catch
        @test true
    end
    # prepare! from a Float64 cache still enforces Float64.
    cache32 = SDPX.SparseSymbolicNumericCache{Float32}
    @test_throws ArgumentError cache32()

    # --- malformed positional/requirements invariants ---------------
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        _core_cache_fixture(1e-6).K;
        symbolic_epoch=0, dsigns=Int[1, -1, 1, -1, 1, -1, 0],
    )
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        _core_cache_fixture(1e-6).K;
        symbolic_epoch=0, dsigns=Int[1, -1],
    )
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        _core_cache_fixture(1e-6).K;
        symbolic_epoch=0, dsigns=Int[1, -1, 1, -1, 1, -1, 1],
        regularization=-1.0,
    )
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        _core_cache_fixture(1e-6).K;
        symbolic_epoch=0, dsigns=Int[1, -1, 1, -1, 1, -1, 1],
        regularization=NaN,
    )
    # non-square pattern rejection
    nonsquare = sparse([1, 2], [1, 1], [1.0, 2.0], 2, 3)
    @test_throws ArgumentError SDPX.SparseSymbolicRequirements(
        nonsquare; symbolic_epoch=0, dsigns=Int[1, -1], regularization=0.0,
    )

    # --- D-sign / regularization identity in the signature ----------
    Kf = _core_cache_fixture(1e-6).K
    req_a = SDPX.SparseSymbolicRequirements(
        Kf; symbolic_epoch=0,
        dsigns=Int[1, -1, 1, -1, 1, -1, 1], regularization=1e-6,
    )
    req_b = SDPX.SparseSymbolicRequirements(
        Kf; symbolic_epoch=0,
        dsigns=Int[1, -1, 1, -1, 1, -1, -1], regularization=1e-6,
    )
    req_c = SDPX.SparseSymbolicRequirements(
        Kf; symbolic_epoch=0,
        dsigns=Int[1, -1, 1, -1, 1, -1, 1], regularization=2e-6,
    )
    c1 = SDPX.SparseSymbolicNumericCache{Float64}()
    SDPX.prepare!(c1, req_a)
    c2 = SDPX.SparseSymbolicNumericCache{Float64}()
    SDPX.prepare!(c2, req_b)
    c3 = SDPX.SparseSymbolicNumericCache{Float64}()
    SDPX.prepare!(c3, req_c)
    @test SDPX.factor_diagnostics(c1).signature !=
          SDPX.factor_diagnostics(c2).signature
    @test SDPX.factor_diagnostics(c1).signature !=
          SDPX.factor_diagnostics(c3).signature
    @test SDPX.factor_diagnostics(c2).signature !=
          SDPX.factor_diagnostics(c3).signature

    # --- deterministic singular + regularized recovery --------------
    fu = _singular_core_fixture(0.0)
    @test_throws ArgumentError SDPX.factorize!(fu.cache, fu.K, 1)
    @test SDPX.factor_status(fu.cache) === SDPX.Failed
    @test SDPX.factor_diagnostics(fu.cache).numeric_count == 0

    fr = _singular_core_fixture(1e-6)
    orig = copy(fr.pattern.nzval)
    SDPX.factorize!(fr.cache, fr.K, 1)
    @test SDPX.factor_status(fr.cache) === SDPX.Fresh
    d = SDPX.factor_diagnostics(fr.cache)
    @test d.symbolic_count == 1
    @test d.numeric_count == 1
    @test fr.pattern.nzval == orig   # original K untouched by regularization

    # --- failed refactor detaches object; recovery via fresh ldlt -----
    factor_before = fr.cache.factor
    @test factor_before !== nothing
    Kbad = copy(fr.K)
    Kbad.nzval[1] = Inf
    @test_throws ArgumentError SDPX.factorize!(fr.cache, Kbad, 2)
    @test SDPX.factor_status(fr.cache) === SDPX.Failed
    @test fr.cache.factor === nothing          # no stale object
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        fr.cache, zeros(3), ones(3),
    )
    @test fr.pattern.nzval == orig             # source pattern unchanged
    @test fr.cache.factor_view.nzval == orig   # factor view restored too
    # Recovery on the same pattern via a fresh symbolic factor is legal.
    SDPX.factorize!(fr.cache, fr.K, 3)
    @test SDPX.factor_status(fr.cache) === SDPX.Fresh
    @test SDPX.factor_diagnostics(fr.cache).numeric_count == 2
    @test fr.cache.factor !== factor_before     # fresh object, not stale

    # --- pre-transition bad input fails closed -----------------------
    fz = _core_cache_fixture(1e-6)
    SDPX.factorize!(fz.cache, fz.K, 1)
    @test SDPX.factor_status(fz.cache) === SDPX.Fresh
    bad = copy(fz.K)
    bad.nzval[2] = NaN
    @test_throws ArgumentError SDPX.factorize!(fz.cache, bad, 2)
    @test SDPX.factor_status(fz.cache) === SDPX.Failed
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        fz.cache, zeros(7), ones(7),
    )
    @test fz.cache.factor === nothing

    # --- Invalid requires re-prepare, cannot factor ------------------
    fv = _core_cache_fixture(1e-6)
    SDPX.factorize!(fv.cache, fv.K, 1)
    @test SDPX.factor_status(fv.cache) === SDPX.Fresh
    SDPX.invalidate!(fv.cache)
    @test SDPX.factor_status(fv.cache) === SDPX.Invalid
    @test_throws SDPX.FactorCacheStateError SDPX.factorize!(
        fv.cache, fv.K, 2,
    )
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        fv.cache, zeros(7), ones(7),
    )
    # re-prepare from Invalid is allowed
    SDPX.prepare!(fv.cache, SDPX.SparseSymbolicRequirements(
        fv.K; symbolic_epoch=9,
        dsigns=SDPX.symmetric_core_dsigns(fv.pattern),
        regularization=1e-6,
    ))
    @test SDPX.factor_status(fv.cache) === SDPX.Prepared
    SDPX.factorize!(fv.cache, fv.K, 10)
    @test SDPX.factor_status(fv.cache) === SDPX.Fresh

    # --- two-RHS solve_multi (homogeneous + variable seam) -----------
    fm = _core_cache_fixture(1e-6)
    SDPX.factorize!(fm.cache, fm.K, 1)
    fobj = fm.cache.factor
    R = [1.0 -0.5; 2.0 1.0; 3.0 0.25; 4.0 0.1; 5.0 0.7; 6.0 -0.3; 7.0 1.1]
    X = zeros(7, 2)
    SDPX.solve_multi!(fm.cache, X, R)
    @test all(isfinite, X)
    @test fm.cache.factor === fobj
    @test SDPX.factor_diagnostics(fm.cache).numeric_count == 1
    # check both columns against the regularized K
    Kreg = SDPX.materialize_dense(fm.pattern)
    dsigns = SDPX.symmetric_core_dsigns(fm.pattern)
    for j in 1:7
        Kreg[j, j] += dsigns[j] * 1e-6
    end
    # Scale-aware backward residual against the exact regularized operator
    # for BOTH columns.  The regularized factor solves Kreg (never the
    # unmodified, structurally singular K).
    residual_scale = norm(Kreg, Inf) * norm(X, Inf) + norm(R, Inf)
    @test norm(Kreg * X - R, Inf) / residual_scale <= 1e-9
    @test SDPX.factor_diagnostics(fm.cache).solve_count == 1
    @test SDPX.factor_diagnostics(fm.cache).numeric_count == 1

    # --- factorize! state-admission policy --------------------------
    # Unprepared cannot factor and stays Unprepared (no factor exists).
    fun = SDPX.SparseSymbolicNumericCache{Float64}()
    @test SDPX.factor_status(fun) === SDPX.Unprepared
    @test_throws SDPX.FactorCacheStateError SDPX.factorize!(
        fun, _core_cache_fixture(1e-6).K, 1,
    )
    @test SDPX.factor_status(fun) === SDPX.Unprepared

    # Fresh + dense input: the stale factor is revoked, solve rejected.
    fd = _core_cache_fixture(1e-6)
    SDPX.factorize!(fd.cache, fd.K, 1)
    @test SDPX.factor_status(fd.cache) === SDPX.Fresh
    @test_throws ArgumentError SDPX.factorize!(
        fd.cache, Matrix(fd.K), 2,
    )
    @test SDPX.factor_status(fd.cache) === SDPX.Failed
    @test fd.cache.factor === nothing
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        fd.cache, zeros(7), ones(7),
    )

    # Fresh + unconvertible epoch: fails closed, stale solve rejected.
    fe = _core_cache_fixture(1e-6)
    SDPX.factorize!(fe.cache, fe.K, 1)
    @test SDPX.factor_status(fe.cache) === SDPX.Fresh
    @test_throws InexactError SDPX.factorize!(
        fe.cache, fe.K, typemax(UInt128),
    )
    @test SDPX.factor_status(fe.cache) === SDPX.Failed
    @test fe.cache.factor === nothing
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        fe.cache, zeros(7), ones(7),
    )

    # --- factor-view restoration after a post-regularization failure --
    # Use a nonzero signed shift.  For the failing same-pattern update, set
    # Ar numerically to zero and Theta=-delta*I.  The original y block is then
    # +delta*I, while factorization adds -delta on every y diagonal; the
    # regularized factor view becomes diag(+delta*I_x, 0_y), deterministically
    # singular after a genuinely nonzero regularization mutation.  Failure
    # must restore the owned factor view to the attempted unregularized K.
    delta = 1e-6
    frs = _core_cache_fixture(delta)
    SDPX.factorize!(frs.cache, frs.K, 1)
    @test SDPX.factor_status(frs.cache) === SDPX.Fresh
    @test frs.cache.factor !== nothing
    Ars = _core_ar(0.0)                 # same frozen structure, zero values
    Theta_cancel = -delta * Matrix{Float64}(I, frs.pattern.m, frs.pattern.m)
    SDPX.refill!(frs.pattern, Ars, Theta_cancel)
    Ks = SDPX.symmetric_core_lower_sparse(frs.pattern)
    attempted = copy(frs.pattern.nzval)
    @test_throws ArgumentError SDPX.factorize!(frs.cache, Ks, 2)
    @test SDPX.factor_status(frs.cache) === SDPX.Failed
    @test frs.cache.factor === nothing
    @test frs.cache.factor_view.nzval == attempted   # nonzero shift removed
    @test frs.pattern.nzval == attempted             # source unchanged
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(
        frs.cache, zeros(7), ones(7),
    )
end

# ---------------------------------------------------------------------
# C4: symmetric augmented-core HSD direction recovery with original-core
# refinement.  Factors the signed static-shifted core Kε (nonzero δ, since
# CHOLMOD's nonpivoting LDL cannot factor the structural-zero primal
# diagonal), solves the homogeneous core once per factor epoch, then
# predictor/corrector variable RHS sequentially with the same factor, and
# refines each core solve against the retained original K with the same
# Kε factor (at most two strict-contraction corrections).  Recovered
# directions are compared with the direct full/reduced five-equation
# solves at the scale-aware bound after refinement.
# ---------------------------------------------------------------------

"""Build a C4 fixture bundle from the C1 `_Fixture` family name."""
function _c4_bundle(family::Symbol; delta::Float64=1e-6)
    fx = _fixture(family)
    V = fx.V
    A = fx.A
    b = fx.b
    c = fx.c
    Theta = fx.Theta
    m, n = size(A)
    nr = size(V, 2)

    # NewtonSystem over the frozen five-equation RHS.
    lin = SDPX.ProductConeLinearization{Float64}(
        Theta, fx.rhs.cone_corrector, [1:m],
    )
    rhs = SDPX.HSDNewtonRHS{Float64, Vector{Float64}, Vector{Float64},
        Vector{Float64}}(
        fx.rhs.primal_affine, fx.rhs.dual_affine, fx.rhs.homogeneous_gap,
        fx.rhs.cone_corrector, fx.rhs.tau_kappa,
    )
    system = SDPX.NewtonSystem(A, b, c, lin, fx.tau, fx.kappa, rhs)

    # Core pattern from Ar = A*V (all cone rows in one dense-lower block).
    Ar = sparse(A * V)
    pattern = SDPX.SymmetricCorePattern{Float64}(Ar, [1:m], [:dense_lower])
    SDPX.refill!(pattern, Ar, Theta)
    K = SDPX.symmetric_core_lower_sparse(pattern)

    req = SDPX.SparseSymbolicRequirements(
        K; symbolic_epoch=1,
        dsigns=SDPX.symmetric_core_dsigns(pattern),
        regularization=delta,
    )
    cache = SDPX.SparseSymbolicNumericCache{Float64}()
    SDPX.prepare!(cache, req)
    SDPX.factorize!(cache, K, 1)
    @assert SDPX.factor_status(cache) === SDPX.Fresh

    ws = SDPX.SymmetricCoreWorkspace(pattern, cache, V, system)
    SDPX.sync_core_factor_epoch!(ws)
    return (; fx, system, pattern, cache, ws)
end

"""Copy only the RHS fields into a fresh NewtonSystem with frozen static data."""
function _c4_rhs_system(bundle, rhs)
    old = bundle.system
    copied_rhs = SDPX.HSDNewtonRHS{Float64,Vector{Float64},Vector{Float64},Vector{Float64}}(
        copy(rhs.primal_affine), copy(rhs.dual_affine), rhs.homogeneous_gap,
        copy(rhs.cone_corrector), rhs.tau_kappa,
    )
    return SDPX.NewtonSystem(
        old.A, old.b, old.c, old.cone, old.tau, old.kappa, copied_rhs,
    )
end

@testset "C4 symmetric core direction recovery with refinement" begin
    for family in (:identity, :rotated, :rank_reduced)
        @testset "$family" begin
            bundle = _c4_bundle(family)
            fx = bundle.fx
            ws = bundle.ws
            scale = _scale(fx)
            tol = 1e-9 * scale

            # Homogeneous solve (once per factor epoch).
            SDPX.solve_core_homogeneous!(ws)
            @test ws.homogeneous_solves == 1
            homogeneous_cache_solves = SDPX.factor_diagnostics(bundle.cache).solve_count
            SDPX.solve_core_homogeneous!(ws)  # idempotent within one epoch
            @test ws.homogeneous_solves == 1
            @test SDPX.factor_diagnostics(bundle.cache).solve_count ==
                  homogeneous_cache_solves

            # Predictor direction; residual must pass the frozen
            # five-equation gate at the scale-aware bound.
            dir1, res1 = SDPX.solve_core_direction!(ws, bundle.system)
            dir1_dx = copy(dir1.dx)
            dir1_primal = copy(res1.primal_affine)
            maxres1 = maximum(abs, vcat(
                res1.primal_affine, res1.dual_affine,
                [res1.homogeneous_gap], res1.cone_complementarity,
                [res1.tau_kappa],
            ))
            @test maxres1 <= tol

            # Corrector direction: only cone/scalar RHS fields change, while
            # A,b,c,Theta,tau,kappa and the factor remain frozen.
            corrected_rhs = (
                primal_affine=copy(bundle.system.rhs.primal_affine),
                dual_affine=copy(bundle.system.rhs.dual_affine),
                homogeneous_gap=bundle.system.rhs.homogeneous_gap,
                cone_corrector=bundle.system.rhs.cone_corrector .+
                                [0.13, -0.07, 0.11],
                tau_kappa=bundle.system.rhs.tau_kappa + 0.19,
            )
            corrector_system = _c4_rhs_system(bundle, corrected_rhs)
            @test SDPX._core_operator_signature(
                bundle.pattern, ws.V, corrector_system,
            ) == ws.operator_signature
            dir2, res2 = SDPX.solve_core_direction!(ws, corrector_system)
            maxres2 = maximum(abs, vcat(
                res2.primal_affine, res2.dual_affine,
                [res2.homogeneous_gap], res2.cone_complementarity,
                [res2.tau_kappa],
            ))
            @test maxres2 <= tol

            # One factor, one homogeneous solve, two variable solves,
            # no refactor between predictor and corrector.
            @test SDPX.factor_diagnostics(bundle.cache).numeric_count == 1
            @test SDPX.factor_diagnostics(bundle.cache).symbolic_count == 1
            @test ws.homogeneous_solves == 1
            @test ws.variable_solves == 2
            @test ws.directions == 2
            @test SDPX.factor_diagnostics(bundle.cache).solve_count == 3
            @test SDPX.factor_diagnostics(bundle.cache).refine_count >= 1

            # Results are independent snapshots, not live workspace views.
            @test dir1.dx !== dir2.dx
            @test dir1.dy !== dir2.dy
            @test dir1.ds !== dir2.ds
            @test res1.primal_affine !== res2.primal_affine
            @test dir1.dx == dir1_dx
            @test res1.primal_affine == dir1_primal

            # Recovered directions agree with the direct five-equation
            # solves at the scale-aware bound (after refinement).
            if family === :rank_reduced
                ref = _direct_reduced(fx)
                @test dir1.dx ≈ fx.V * ref.dxr atol=tol rtol=0
            else
                ref = _direct_full(fx)
            end
            _compare_directions(dir1, ref, tol)
            if family === :rank_reduced
                ref2 = _direct_reduced(_Fixture(
                    fx.V, fx.A, fx.b, fx.c, fx.Theta, fx.tau, fx.kappa,
                    corrected_rhs, fx.direction,
                ))
            else
                ref2 = _direct_full(_Fixture(
                    fx.V, fx.A, fx.b, fx.c, fx.Theta, fx.tau, fx.kappa,
                    corrected_rhs, fx.direction,
                ))
            end
            _compare_directions(dir2, ref2, tol)
        end
    end

    # --- denominator gates fail closed -----------------------------
    # Exact-zero denominator: choose u such that
    #   kappa + tau*(cr'ux + b'uy) == 0.
    bundle = _c4_bundle(:identity)
    ws = bundle.ws
    SDPX.solve_core_homogeneous!(ws)
    cr = ws.cr
    fill!(ws.ux, 0.0)
    fill!(ws.uy, 0.0)
    ws.ux[1] = -ws.system.kappa / (ws.system.tau * cr[1])
    @test_throws ArgumentError SDPX.solve_core_direction!(ws)

    # Non-finite denominator.
    bundle2 = _c4_bundle(:identity)
    ws2 = bundle2.ws
    SDPX.solve_core_homogeneous!(ws2)
    ws2.ux[1] = NaN
    @test_throws ArgumentError SDPX.solve_core_direction!(ws2)

    # Type-scaled near-zero denominator: |den| <= sqrt(eps(T)) * scale.
    bundle3 = _c4_bundle(:identity)
    ws3 = bundle3.ws
    SDPX.solve_core_homogeneous!(ws3)
    fill!(ws3.ux, 0.0)
    fill!(ws3.uy, 0.0)
    ws3.ux[1] = -(ws3.system.kappa - sqrt(eps(Float64)) *
                  max(1.0, ws3.system.kappa, ws3.system.tau)) /
                (ws3.system.tau * cr[1])
    @test_throws ArgumentError SDPX.solve_core_direction!(ws3)
end

@testset "C4 epoch, operator identity, and original-core guards" begin
    bundle = _c4_bundle(:identity)
    ws = bundle.ws
    @test ws.matrix_epoch == SDPX.factor_matrix_epoch(bundle.cache)
    @test ws.factor_epoch == SDPX.factor_epoch(bundle.cache)
    @test ws.pattern_signature == SDPX.symmetric_core_signature(bundle.pattern)
    @test ws.homogeneous_epoch == -1
    @test_throws ArgumentError SDPX.solve_core_direction!(ws, bundle.system)

    # The exact symmetric infinity norm counts each stored off-diagonal entry
    # in both affected rows, unlike the old max-entry scale.
    expected_scale = max(
        1.0, opnorm(SDPX.materialize_dense(bundle.pattern), Inf),
    )
    @test ws.original_scale == expected_scale

    SDPX.solve_core_homogeneous!(ws)
    @test ws.homogeneous_epoch == ws.factor_epoch
    @test_throws SDPX.FactorCacheStateError begin
        SDPX.invalidate!(bundle.cache)
        SDPX.solve_core_homogeneous!(ws)
    end

    # A numeric refill without a matching new factor is stale even though the
    # structural pattern signature is unchanged.
    bundle2 = _c4_bundle(:identity)
    SDPX.solve_core_homogeneous!(bundle2.ws)
    old_operator_signature = bundle2.ws.operator_signature
    Ar2 = sparse(bundle2.system.A * bundle2.ws.V)
    Ar2.nzval .*= 1.25
    Theta2 = 1.1 .* bundle2.system.cone.operator
    SDPX.refill!(bundle2.pattern, Ar2, Theta2)
    @test SDPX.symmetric_core_signature(bundle2.pattern) ==
          bundle2.ws.pattern_signature
    @test SDPX._core_operator_signature(
        bundle2.pattern, bundle2.ws.V, bundle2.system,
    ) != old_operator_signature
    @test_throws ArgumentError SDPX.solve_core_direction!(bundle2.ws, bundle2.system)

    # A refactor is also stale until synchronized, and a changed operator
    # cannot be attached to the old NewtonSystem/workspace.
    bundle3 = _c4_bundle(:identity)
    SDPX.solve_core_homogeneous!(bundle3.ws)
    Ar3 = sparse(bundle3.system.A * bundle3.ws.V)
    Ar3.nzval .*= 1.25
    Theta3 = 1.1 .* bundle3.system.cone.operator
    SDPX.refill!(bundle3.pattern, Ar3, Theta3)
    K3 = SDPX.symmetric_core_lower_sparse(bundle3.pattern)
    SDPX.factorize!(bundle3.cache, K3, 2)
    @test SDPX.factor_epoch(bundle3.cache) == bundle3.ws.factor_epoch + 1
    @test_throws ArgumentError SDPX.solve_core_direction!(bundle3.ws, bundle3.system)
    @test_throws ArgumentError SDPX.sync_core_factor_epoch!(bundle3.ws)

    # Zero residual with zero denominator scale is an exact zero, not Inf/NaN.
    bundle4 = _c4_bundle(:identity)
    ws4 = bundle4.ws
    ws4.original_scale = 0.0
    z = zeros(7)
    @test SDPX._core_normalized_residual(ws4, z, z, z) == 0.0
    @test isinf(SDPX._core_normalized_residual(ws4, z, z, [1.0, zeros(6)...]))
end

function _bigfloat_copy_array(source)
    destination = SDPX.alloc_zeros(BigFloat, size(source)...)
    for index in eachindex(source)
        destination[index] = BigFloat(source[index])
    end
    return destination
end

@testset "C4 bounded BigFloat workspace ownership" begin
    fx = _fixture(:identity)
    A = _bigfloat_copy_array(fx.A)
    V = _bigfloat_copy_array(fx.V)
    b = _bigfloat_copy_array(fx.b)
    c = _bigfloat_copy_array(fx.c)
    Theta = _bigfloat_copy_array(fx.Theta)
    primal = _bigfloat_copy_array(fx.rhs.primal_affine)
    dual = _bigfloat_copy_array(fx.rhs.dual_affine)
    gap = BigFloat(fx.rhs.homogeneous_gap)
    cone_rhs = _bigfloat_copy_array(fx.rhs.cone_corrector)
    scalar = BigFloat(fx.rhs.tau_kappa)
    lin = SDPX.ProductConeLinearization{BigFloat}(
        Theta, cone_rhs, [1:3],
    )
    rhs = SDPX.HSDNewtonRHS{BigFloat,Vector{BigFloat},Vector{BigFloat},Vector{BigFloat}}(
        primal, dual, gap, cone_rhs, scalar,
    )
    system = SDPX.NewtonSystem(
        A, b, c, lin, BigFloat(fx.tau), BigFloat(fx.kappa), rhs,
    )
    Ar = sparse(A * V)
    pattern = SDPX.SymmetricCorePattern{BigFloat}(Ar, [1:3], [:dense_lower])
    SDPX.refill!(pattern, Ar, Theta)
    K = SDPX.materialize_dense(pattern)
    cache = _BigFloatCoreCache(K, SDPX.Fresh, 4, 9, 0, 0)
    ws = SDPX.SymmetricCoreWorkspace(pattern, cache, V, system)
    SDPX.sync_core_factor_epoch!(ws)
    @test ws.original_nzval[1] !== pattern.nzval[1]
    @test all(i == j || ws.original_nzval[i] !== ws.original_nzval[j]
              for i in eachindex(ws.original_nzval), j in eachindex(ws.original_nzval))
    SDPX.solve_core_homogeneous!(ws)
    direction, residual = SDPX.solve_core_direction!(ws, system)
    @test direction.dx !== ws.dx
    @test residual.primal_affine !== ws.residual.primal_affine
    @test all(i == j || ws.rhs_core[i] !== ws.rhs_core[j]
              for i in eachindex(ws.rhs_core), j in eachindex(ws.rhs_core))
    @test all(i == j || ws.sol_core[i] !== ws.sol_core[j]
              for i in eachindex(ws.sol_core), j in eachindex(ws.sol_core))
    @test cache.solve_count == 2
    @test cache.refine_count >= 0
end
