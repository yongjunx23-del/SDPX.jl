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
        c = [0.5, -0.25, 0.75]
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
