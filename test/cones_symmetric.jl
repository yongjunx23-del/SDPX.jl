# test/cones_symmetric.jl
#
# Symmetric-cone algebra (Subagent I): mutating, zero-allocation Jordan-algebra
# kernels for the Nonnegative, SOC (Lorentz) and PSDTriangle (packed-lower PSD)
# cones, plus the extended-precision (MultiFloat / BigFloat) backends.

if !isdefined(@__MODULE__, :SDPX)
    using SDPX
    if !isdefined(SDPX, :SymmetricCones)
        @eval using SDPX
    end
end
const SC = SDPX.SymmetricCones

using Test
using LinearAlgebra
using LinearAlgebra: Diagonal, Symmetric

const _HAVE_MULTIFLOATS = try
    @eval import MultiFloats
    true
catch
    false
end

"""Packed lower triangle (column-major: col 1 full, col 2 rows 2..n, …)."""
function _pack_lower(X)
    n = size(X, 1)
    v = eltype(X)[]
    for j in 1:n, i in j:n
        push!(v, X[i, j])
    end
    v
end

# ---------------------------------------------------------------------------
# Nonnegative cone
# ---------------------------------------------------------------------------
function _sym_nn(::Type{T}) where {T}
    @testset "Nonnegative ($T)" begin
        cone = SC.NonnegativeCone(4)
        x = T[2, 3, 1, 5]
        @test SC.membership(cone, T[0, 1, 2, 3])
        @test !SC.membership(cone, T[1, -1, 2, 3])
        @test SC.dual_membership(cone, x)

        e = zeros(T, 4)
        SC.identity!(cone, e)
        @test e == ones(T, 4)
        @test SC.identity_element(cone, x) == ones(T, 4)

        z = zeros(T, 4)
        SC.jordan_product!(cone, z, x, x)
        @test z ≈ x .^ 2

        invx = zeros(T, 4)
        SC.inverse!(cone, invx, x)
        @test invx ≈ 1 ./ x
        r = zeros(T, 4)
        SC.jordan_product!(cone, r, x, invx)      # x ∘ x^{-1} = e
        @test r ≈ ones(T, 4)

        s = zeros(T, 4)
        SC.sqrt!(cone, s, x)
        ss = zeros(T, 4)
        SC.jordan_product!(cone, ss, s, s)        # sqrt(x) ∘ sqrt(x) = x
        @test ss ≈ x

        W = zeros(T, 4)
        SC.nt_scaling!(cone, W, x)                # W = x^{-1}
        y = zeros(T, 4)
        SC.scaling_apply!(cone, y, W, x)          # W·x = e
        @test y ≈ ones(T, 4)
        yi = zeros(T, 4)
        SC.scaling_inverse_apply!(cone, yi, W, x) # W^{-1}·x = x²
        @test yi ≈ x .^ 2

        g = zeros(T, 4)
        SC.barrier_gradient!(cone, g, x)
        @test g ≈ -1 ./ x
        h = zeros(T, 4)
        SC.barrier_hessian_product!(cone, h, x, T[1, 2, 3, 4])
        @test h ≈ T[1, 2, 3, 4] ./ x .^ 2

        ref = Ref(zero(T))
        a = SC.boundary_step!(cone, T[1, 2, 3, 5], ref, T[-1, 0.5, 0, 0])
        @test a ≈ T(1)
        @test ref[] ≈ T(1)

        w = zeros(T, 4)
        SC.third_order_correction!(cone, w, T[1, 2, 3, 4], T[2, 1, 0.5, 1], T[3, 4, 5, 1])
        @test w ≈ T[1, 2, 3, 4] .* T[2, 1, 0.5, 1] .* T[3, 4, 5, 1]
    end
end

# ---------------------------------------------------------------------------
# SOC cone
# ---------------------------------------------------------------------------
function _sym_soc(::Type{T}) where {T}
    @testset "SOC ($T)" begin
        cone = SC.SOCone(4)
        x = T[3, 1, 2, 0.5]
        @test SC.membership(cone, x)
        @test !SC.membership(cone, T[1, 2, 0, 0])
        @test SC.dual_membership(cone, x)

        e = zeros(T, 4)
        SC.identity!(cone, e)
        @test e == T[1, 0, 0, 0]

        invx = zeros(T, 4)
        SC.inverse!(cone, invx, x)
        r = zeros(T, 4)
        SC.jordan_product!(cone, r, x, invx)      # x ∘ x^{-1} = e
        @test r ≈ e

        s = zeros(T, 4)
        SC.sqrt!(cone, s, x)
        ss = zeros(T, 4)
        SC.jordan_product!(cone, ss, s, s)        # sqrt∘sqrt = x
        @test ss ≈ x

        W = zeros(T, 4)
        SC.nt_scaling!(cone, W, x)
        y = zeros(T, 4)
        SC.scaling_apply!(cone, y, W, x)          # W·x = e
        @test y ≈ e
        yi = zeros(T, 4)
        SC.scaling_inverse_apply!(cone, yi, W, x) # W^{-1}·x = x²
        xx = zeros(T, 4)
        SC.jordan_product!(cone, xx, x, x)
        @test yi ≈ xx

        # boundary element ‖u‖ = t: spectral decomposition with eigenvalue 0,
        # no division by zero.
        xb = T[2, 2, 0, 0]
        @test SC.membership(cone, xb)
        lam1, lam2, c1, c2 = SC.spectrum(cone, xb)
        @test lam2 ≈ zero(T)                     # eigenvalue treated as 0
        @test lam1 * c1 + lam2 * c2 ≈ xb
        # zero tail: both idempotents (1/2, 0, …)
        lam1, lam2, c1, c2 = SC.spectrum(cone, T[4, 0, 0, 0])
        @test c1 ≈ T[0.5, 0, 0, 0]
        @test c2 ≈ T[0.5, 0, 0, 0]

        # barrier gradient finite difference
        g = zeros(T, 4)
        SC.barrier_gradient!(cone, g, x)
        F = v -> -log(v[1]^2 - sum(v[2:end] .^ 2))
        hfd = T(1e-5)
        gnum = T[(F(x + hfd * onehot(T, 4, i)) - F(x - hfd * onehot(T, 4, i))) / (2hfd) for i in 1:4]
        @test g ≈ gnum rtol=1e-3 atol=1e-4

        # barrier hessian product
        d = T[0.5, -0.2, 0.1, 0.3]
        h = zeros(T, 4)
        SC.barrier_hessian_product!(cone, h, x, d)
        g_plus = zeros(T, 4)
        g_minus = zeros(T, 4)
        SC.barrier_gradient!(cone, g_plus, x + hfd * d)
        SC.barrier_gradient!(cone, g_minus, x - hfd * d)
        hnum = (g_plus - g_minus) / (2hfd)
        @test h ≈ hnum rtol=1e-3 atol=1e-4

        # third-order correction
        w_soc = zeros(T, 4)
        SC.third_order_correction!(cone, w_soc, T[1, 0.2, 0.1, 0], T[0.5, 0.1, 0, 0.2], T[1, 0, 0.1, 0.1])
        @test all(isfinite, w_soc)

        ref = Ref(zero(T))
        a = SC.boundary_step!(cone, T[2, 0, 0, 0], ref, T[-1, 0, 0, 0])
        @test a ≈ T(2)
        ref = Ref(zero(T))
        a = SC.boundary_step!(cone, T[2, 0, 0, 0], ref, T[-1, 1, 0, 0])
        @test a ≈ T(1)
    end
end

function onehot(::Type{T}, n, i) where {T}
    v = zeros(T, n)
    v[i] = one(T)
    return v
end

# ---------------------------------------------------------------------------
# PSD (packed lower triangle) cone
# ---------------------------------------------------------------------------
function _sym_psd(::Type{T}) where {T}
    @testset "PSD ($T)" begin
        n = 3
        cone = SC.PSDTriangleCone{T}(n)
        X = T[3 1 0.5; 1 2 0.2; 0.5 0.2 2]
        x = _pack_lower(X)
        @test SC.membership(cone, x)
        @test !SC.membership(cone, _pack_lower(T[1 2 0; 2 0.5 0; 0 0 2]))  # not PSD
        @test SC.dual_membership(cone, x)

        e = zeros(T, 6)
        SC.identity!(cone, e)
        @test e == _pack_lower(Matrix{T}(I, 3, 3))

        z = zeros(T, 6)
        SC.jordan_product!(cone, z, x, e)         # X ∘ I = X
        @test z ≈ x

        invx = zeros(T, 6)
        SC.inverse!(cone, invx, x)
        r = zeros(T, 6)
        SC.jordan_product!(cone, r, x, invx)      # X ∘ X^{-1} = I
        @test r ≈ e

        s = zeros(T, 6)
        SC.sqrt!(cone, s, x)
        ss = zeros(T, 6)
        SC.jordan_product!(cone, ss, s, s)        # sqrt∘sqrt = X
        @test ss ≈ x

        W = zeros(T, 6)
        SC.nt_scaling!(cone, W, x)
        y = zeros(T, 6)
        SC.scaling_apply!(cone, y, W, x)          # W·X = I
        @test y ≈ e
        yi = zeros(T, 6)
        SC.scaling_inverse_apply!(cone, yi, W, x) # W^{-1}·x = x²
        xx = zeros(T, 6)
        SC.jordan_product!(cone, xx, x, x)
        @test yi ≈ xx

        # primitive idempotents: E_k = v_k v_kᵀ are exact Jordan idempotents
        Es = SC.primitive_idempotents(cone, x)
        @test length(Es) == n
        for E in Es
            E2 = (E * E + E * E) / 2
            @test E2 ≈ E atol=100 * eps(T) rtol=100 * eps(T)
        end
        # never two identical idempotents (distinct rank-one projections)
        for i in 1:(n - 1), j in (i + 1):n
            @test norm(Es[i] - Es[j]) > 1e-6
        end

        # eigendecomposition matches LAPACK (Float64 only; a "verified" generic
        # backend that converges, with a throw-on-failure convergence check).
        if T === Float64
            w, V = SC.spectrum(cone, x)
            lw, lV = eigen(Symmetric(Matrix{Float64}(X)))
            @test sort(w) ≈ sort(lw) rtol=1e-10 atol=1e-10
        end

        # barrier gradient finite difference (packed-coordinate convention).
        # Packed col-major v = (a,b,d, c,e, f) for matrix [[a,b,d],[b,c,e],[d,e,f]].
        g = zeros(T, 6)
        SC.barrier_gradient!(cone, g, x)
        F = v -> -log(v[1]*v[4]*v[6] + 2*v[2]*v[3]*v[5] - v[1]*v[5]^2 - v[2]^2*v[6] - v[3]^2*v[4])
        hfd = T(1e-6)
        gnum = zeros(T, 6)
        for i in 1:6
            ecol = onehot(T, 6, i)
            gnum[i] = (F(x + hfd * ecol) - F(x - hfd * ecol)) / (2hfd)
        end
        @test g ≈ gnum rtol=1e-3 atol=1e-4

        # barrier hessian product h = X^{-1} d X^{-1} (off-diagonals doubled)
        D = T[1 2 0.2; 2 0.5 1; 0.2 1 3]
        d = _pack_lower(D)
        h = zeros(T, 6)
        SC.barrier_hessian_product!(cone, h, x, d)
        Xin = inv(X)
        M = Xin * D * Xin
        htrue = _pack_lower(M)
        htrue[2] *= 2; htrue[3] *= 2; htrue[5] *= 2   # off-diagonals doubled
        @test h ≈ htrue rtol=1e-5 atol=1e-6

        # boundary step
        Xb = T[4 0 0; 0 2 0; 0 0 1]
        dXb = T[-1 0 0; 0 0 0; 0 0 0]
        ref = Ref(zero(T))
        a = SC.boundary_step!(cone, _pack_lower(Xb), ref, _pack_lower(dXb))
        @test a ≈ T(4)
        @test ref[] ≈ T(4)
    end
end

# ---------------------------------------------------------------------------
# Zero-allocation warm path (Float64 is the canonical target)
# ---------------------------------------------------------------------------
function _sym_zero_alloc()
    @testset "zero-allocation hot path (Float64)" begin
        # Nonnegative
        let cone = SC.NonnegativeCone(4), x = [2.0, 3, 1, 5], z = zeros(4), W = zeros(4)
            f = () -> SC.membership(cone, x); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
            f = () -> SC.identity!(cone, z); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
            f = () -> SC.jordan_product!(cone, z, x, x); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
            f = () -> SC.inverse!(cone, z, x); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
            f = () -> SC.sqrt!(cone, z, x); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
            f = () -> SC.nt_scaling!(cone, W, x); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
            f = () -> SC.scaling_apply!(cone, z, W, x); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
            f = () -> SC.scaling_inverse_apply!(cone, z, W, x); f(); f()
            @test all(==(0), [@allocated(f()) for _ in 1:10])
        end
        # SOC
        let cone = SC.SOCone(5), x = [4.0, 1, 2, 0.5, 0.5], z = zeros(5), W = zeros(5)
            for (nm, f) in (
                ("membership", () -> SC.membership(cone, x)),
                ("identity!", () -> SC.identity!(cone, z)),
                ("jordan_product!", () -> SC.jordan_product!(cone, z, x, x)),
                ("inverse!", () -> SC.inverse!(cone, z, x)),
                ("sqrt!", () -> SC.sqrt!(cone, z, x)),
                ("nt_scaling!", () -> SC.nt_scaling!(cone, W, x)),
                ("scaling_apply!", () -> SC.scaling_apply!(cone, z, W, x)),
                ("scaling_inverse_apply!", () -> SC.scaling_inverse_apply!(cone, z, W, x)),
            )
                f(); f(); f()
                @test all(==(0), [@allocated(f()) for _ in 1:10])
            end
        end
        # PSD
        let cone = SC.PSDTriangleCone{Float64}(3),
            x = _pack_lower([3.0 1 0.5; 1 2 0.2; 0.5 0.2 2]),
            z = zeros(6), W = zeros(6)
            for f in (
                () -> SC.membership(cone, x),
                () -> SC.identity!(cone, z),
                () -> SC.jordan_product!(cone, z, x, x),
                () -> SC.inverse!(cone, z, x),
                () -> SC.sqrt!(cone, z, x),
                () -> SC.nt_scaling!(cone, W, x),
                () -> SC.scaling_apply!(cone, z, W, x),
                () -> SC.scaling_inverse_apply!(cone, z, W, x),
            )
                f(); f(); f()
                @test all(==(0), [@allocated(f()) for _ in 1:10])
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Eigen-decomposition convergence check (requirement: a convergence flag, never
# a fixed-iteration loop that silently accepts a non-converged result).
# ---------------------------------------------------------------------------
function _sym_eigen_convergence()
    @testset "PSD eigendecomposition convergence check" begin
        n = 3
        s = SC.PSDEigenScratch{Float64}(n)
        X = [3.0 1 0.5; 1 2 0.2; 0.5 0.2 2]
        x = _pack_lower(X)
        SC._unpack!(s.A, x, n)
        SC._identity!(s.V, n)
        # A single sweep is not enough to converge a dense 3×3: must throw.
        @test_throws SC._SymmetricEigenFailed SC._jacobi_eigen!(s.A, s.V, s.w; maxsweeps=1)
        # A generous budget converges.
        SC._unpack!(s.A, x, n)
        SC._identity!(s.V, n)
        SC._jacobi_eigen!(s.A, s.V, s.w; maxsweeps=50)
        @test sort(s.w) ≈ sort(eigvals(Symmetric(Matrix{Float64}(X)))) rtol=1e-10
    end
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
_sym_nn(Float64)
_sym_soc(Float64)
_sym_psd(Float64)

_sym_zero_alloc()
_sym_eigen_convergence()

@testset "SymmetricCones (BigFloat)" begin
    _sym_nn(BigFloat)
    _sym_soc(BigFloat)
    _sym_psd(BigFloat)
end

if _HAVE_MULTIFLOATS
    @testset "SymmetricCones (MultiFloat)" begin
        _sym_nn(MultiFloats.Float64x2)
        _sym_soc(MultiFloats.Float64x2)
        _sym_psd(MultiFloats.Float64x2)
    end
end
