# hsd_equations.jl — mechanical checks of the frozen HSD sign/equation conventions
# (docs/design/CANONICAL_FORM.md, the authoritative frozen spec) on hand-computed
# LP / SOC / SDP fixtures.
#
# Frozen HSD (corrected gap sign, MOSEK-style with explicit dual y ∈ K*):
#   (P)  A x + s − b τ = 0
#   (D)  A'y + c τ     = 0
#   (G)  −c'x − b'y + κ = 0          (κ = c'x + b'y = s'y ≥ 0 at feasibility)
#   s ∈ K,  y ∈ K*,  τ ≥ 0,  κ ≥ 0
#   μ = (s'y + τ κ) / (ν + 1)
#   skew matrix  Q·(x; y; τ) = (0; s; −κ),  Q = −Q'
#
#   (The sign note: the FIRST frozen convention c'x + b'y + κ = 0 forced
#    κ = −(c'x+b'y) ≤ 0 at feasible points and admitted no strictly-interior
#    central point.  It is corrected here.  The historical analysis lives in
#    docs/reviews/HSD_SIGN_HISTORY.md.)
#
# This file is intentionally self-contained (no SDPX dependency): it verifies
# the *algebra* of the formulation on small, hand-computable fixtures, so it
# never depends on the production implementation it is meant to check.

using Test
using LinearAlgebra
using SparseArrays

const TOL = 1e-9

# --- cone-membership helpers ------------------------------------------------
nonneg(v)  = all(v .>= -TOL)
function soc_member(v)                 # (t, u), t ≥ ‖u‖
    length(v) >= 1 || return false
    return v[1] >= -TOL && norm(v[2:end]) <= v[1] + TOL
end
function psd_member(v)                 # packed lower triangle (a,b,c) of [[a,b],[b,c]]
    n = round(Int, (sqrt(8 * length(v) + 1) - 1) / 2)
    n * (n + 1) ÷ 2 == length(v) || return false
    M = zeros(eltype(v), n, n)
    k = 1
    for j in 1:n, i in j:n
        M[i, j] = v[k]; M[j, i] = v[k]; k += 1
    end
    return minimum(eigvals(Symmetric(M))) >= -TOL
end

# ---------------------------------------------------------------------------
# The skew-symmetric block matrix Q of the homogeneous system.
# Unknown order (x, y, τ), block sizes (n, m, 1):
#   Q = [  0    A'   c  ]
#       [ −A    0    b  ]
#       [ −c'  −b'   0  ]
# and  Q·(x;y;τ) = (A'y + cτ ;  −A x + bτ ;  −c'x − b'y) = (0; s; −κ)
# (the third component is −κ, the frozen-convention value).
# ---------------------------------------------------------------------------
function hsd_Q(A, b, c)
    m, n = size(A)
    Q = zeros(n + m + 1, n + m + 1)     # all blocks 0 by construction
    Q[1:n, n+1:n+m]     = A'
    Q[1:n, n+m+1]       = c
    Q[n+1:n+m, 1:n]     = -A
    Q[n+1:n+m, n+m+1]   = b
    Q[n+m+1, 1:n]       = -c'
    Q[n+m+1, n+1:n+m]   = -b'
    return Q
end

# ---------------------------------------------------------------------------
# 1. Small LP  (K = R_+², ν = 2)
#    A = I₂, b = (1,1), c = (−1,−1)
# ---------------------------------------------------------------------------
@testset "LP fixture" begin
    A = Matrix{Float64}(I, 2, 2)
    b = [1.0, 1.0]
    c = [-1.0, -1.0]
    ν = 2

    # optimal point (P),(D),(G) all vanish; κ = 0  (c'x + b'y = −2 + 2 = 0)
    x = [1.0, 1.0]; y = [1.0, 1.0]; s = [0.0, 0.0]; τ = 1.0; κ = 0.0
    @test A*x + s - b*τ       ≈ zeros(2) atol = TOL
    @test A'*y + c*τ          ≈ zeros(2) atol = TOL
    @test -dot(c, x) - dot(b, y) + κ ≈ 0 atol = TOL
    @test (dot(s, y) + τ*κ) / (ν + 1) ≈ 0 atol = TOL
    @test nonneg(s) && nonneg(y)
    @test c'x + b'y ≈ s'y atol = TOL

    # strictly-interior point: s, y ∈ int K, τ, κ > 0, residuals may be nonzero.
    #   Choose x, s (s = b − A x), y, and κ = −(−c'x − b'y) = c'x + b'y = s'y.
    x = [0.4, 0.4]; y = [1.0, 1.0]; s = [0.6, 0.6]; τ = 1.0
    gap = dot(c, x) + dot(b, y)          # = s'y = 1.2 ≥ 0
    κ = gap                              # κ = c'x + b'y ≥ 0 (interior & positive)
    @test A*x + s - b*τ       ≈ zeros(2) atol = TOL    # (P): s = b − A x
    @test A'*y + c*τ          ≈ zeros(2) atol = TOL    # (D)
    @test -dot(c, x) - dot(b, y) + κ ≈ 0 atol = TOL     # (G)
    @test nonneg(s) && nonneg(y)
    @test κ > 0 && τ > 0
    μ = (dot(s, y) + τ*κ) / (ν + 1)
    @test dot(s, y) + τ*κ ≈ μ*(ν + 1) atol = TOL       # μ formula (definition)

    # skew-symmetry and action on this exact point
    Q = hsd_Q(A, b, c)
    @test Q + Q' ≈ zeros(size(Q)) atol = TOL
    @test Q[1:2, 5] ≈ c          # Q_{13} = c  (τ block = column n+m+1 = 5)
    @test vec(Q[5, 1:2]) ≈ -c    # Q_{31} = −c'
    # Q·(x;y;τ) = (0; s; −κ)
    @test Q * [x; y; τ] ≈ [zeros(2); s; -κ] atol = TOL
end

# ---------------------------------------------------------------------------
# 2. Small SOC  (K = SOC(2), self-dual, ν = 2)
#    A = (1,1)', b = (1,1), c = (−1)
# ---------------------------------------------------------------------------
@testset "SOC fixture" begin
    A = reshape([1.0, 1.0], 2, 1)
    b = [1.0, 1.0]
    c = [-1.0]
    ν = 2

    # optimal point
    x = [1.0]; y = [1.0, 0.0]; s = [0.0, 0.0]; τ = 1.0; κ = 0.0
    @test A*x + s - b*τ ≈ zeros(2) atol = TOL
    @test A'*y + c*τ    ≈ zeros(1) atol = TOL
    @test -dot(c, x) - dot(b, y) + κ ≈ 0 atol = TOL
    @test (dot(s, y) + τ*κ) / (ν + 1) ≈ 0 atol = TOL
    @test soc_member(s) && soc_member(y)

    # interior-feasible point; κ from (G) = −(−c'x − b'y) = c'x + b'y = s'y ≥ 0
    x = [0.5]; y = [0.5, 0.5]; s = [0.5, 0.5]; τ = 1.0
    gap = dot(c, x) + dot(b, y)          # = −0.5 + 1.0 = 0.5 = s'y
    κ = gap
    @test A*x + s - b*τ ≈ zeros(2) atol = TOL
    @test A'*y + c*τ    ≈ zeros(1) atol = TOL
    @test -dot(c, x) - dot(b, y) + κ ≈ 0 atol = TOL
    @test soc_member(s) && soc_member(y) && τ > 0 && κ > 0

    # skew-symmetry + action on this exact point
    Q = hsd_Q(A, b, c)
    @test Q + Q' ≈ zeros(size(Q)) atol = TOL
    @test Q * [x; y; τ] ≈ [zeros(1); s; -κ] atol = TOL
end

# ---------------------------------------------------------------------------
# 3. Small SDP  (K = PSD₂, packed lower triangle, ν = 2)
#    A = (1,1,1)' (3×1), b = (4,2,2), c = (−4), τ = 1
#    x = 0, y = (1,1,2), s = b − A x = (4,2,2), κ from (G) = c'x + b'y = 10.
#    s ↔ [[4,2],[2,2]] (det 4) PSD;  y ↔ [[1,1],[1,2]] (det 1) PSD.
# ---------------------------------------------------------------------------
@testset "SDP fixture" begin
    A = reshape([1.0, 1.0, 1.0], 3, 1)
    b = [4.0, 2.0, 2.0]
    c = [-4.0]
    ν = 2

    x = [0.0]; y = [1.0, 1.0, 2.0]; s = [4.0, 2.0, 2.0]; τ = 1.0
    gap = dot(c, x) + dot(b, y)          # = 0 + (4+2+4) = 10 = s'y
    κ = gap
    @test A*x + s - b*τ ≈ zeros(3) atol = TOL
    @test A'*y + c*τ    ≈ zeros(1) atol = TOL
    @test -dot(c, x) - dot(b, y) + κ ≈ 0 atol = TOL
    @test psd_member(s) && psd_member(y)
    @test (dot(s, y) + τ*κ) / (ν + 1) ≈ 20 / 3 atol = TOL   # s'y = 10, τκ = 10
    @test κ > 0 && τ > 0

    # skew-symmetry on the rectangular A
    Q = hsd_Q(A, b, c)
    @test size(Q) == (1 + 3 + 1, 1 + 3 + 1)
    @test Q + Q' ≈ zeros(size(Q)) atol = TOL
    @test Q * [x; y; τ] ≈ [zeros(1); s; -κ] atol = TOL
end

# ---------------------------------------------------------------------------
# 4. Rectangular m != n: the exact linear HSD identity must hold on a
#    non-square A, using only conjugate pairs (s,y) — never dot(s,x).
# ---------------------------------------------------------------------------
@testset "rectangular (m != n) linear identity" begin
    # m=3, n=2.  Synthssize a fully consistent point: pick x, s (from (P)),
    # and y, c with A'y + c = 0 (D) and y ∈ K*.  Then weak duality gives
    # c'x + b'y = s'y exactly, and κ = c'x + b'y ≥ 0.
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0]      # 3×2 (m=3, n=2)
    b = [2.0, 2.0, 3.0]
    c = [-2.0, -2.0]                     # = -A'y for y=(1,1,1)
    m, n = size(A)
    @test m != n

    x = [1.0, 1.0]
    s = [1.0, 1.0, 1.0]                  # s = b − A x
    y = [1.0, 1.0, 1.0]
    τ = 1.0
    @test A*x + s - b*τ       ≈ zeros(m) atol = TOL   # (P)
    @test A'*y + c*τ          ≈ zeros(n) atol = TOL   # (D)
    gap = dot(c, x) + dot(b, y)           # = s'y = 3 ≥ 0
    @test gap ≈ dot(s, y) atol = TOL      # weak duality identity
    κ = gap
    @test -dot(c, x) - dot(b, y) + κ ≈ 0 atol = TOL   # (G)
    @test nonneg(s) && nonneg(y)
    @test κ > 0 && τ > 0

    # Q action, dims, skew
    Q = hsd_Q(A, b, c)
    @test size(Q) == (n + m + 1, n + m + 1)
    @test Q + Q' ≈ zeros(size(Q)) atol = TOL
    @test Q * [x; y; τ] ≈ [zeros(n); s; -κ] atol = TOL

    # The forbidden pairing dot(s,x) is dimensionally impossible here.
    @test_throws DimensionMismatch dot(s, x)
end

# ---------------------------------------------------------------------------
# 5. Optimal limit: τ>0, κ→0, s'y→0 (the complementarity collapses exactly)
# ---------------------------------------------------------------------------
@testset "optimal limit fixture" begin
    A = Matrix{Float64}(I, 2, 2)
    b = [1.0, 1.0]
    c = [-1.0, -1.0]
    # x = (1,1), s = 0, y = (1,1), τ = 1, κ = 0: all residuals vanish.
    x = [1.0, 1.0]; y = [1.0, 1.0]; s = [0.0, 0.0]; τ = 1.0; κ = 0.0
    @test A*x + s - b*τ       ≈ zeros(2) atol = TOL
    @test A'*y + c*τ          ≈ zeros(2) atol = TOL
    @test -dot(c, x) - dot(b, y) + κ ≈ 0 atol = TOL
    @test dot(s, y) + τ*κ ≈ 0 atol = TOL
    @test dot(c, x) ≈ -dot(b, y) atol = TOL   # gap closes: c'x = −2, −b'y = −2
end

# ---------------------------------------------------------------------------
# 6. Certificate sign conventions (original-coordinate rays)
# ---------------------------------------------------------------------------
@testset "certificates" begin
    # (a) Primal-infeasible ray: A'y ≈ 0, y ∈ K*, b'y < 0, normalize −b'y = 1.
    #     A = 0, b = −1, K = R_+ : 0·x + s = −1, s ≥ 0 is infeasible.
    A = reshape([0.0], 1, 1)
    b = [-1.0]
    yray = [1.0]
    @test A'*yray ≈ zeros(1) atol = TOL
    @test dot(b, yray) < 0
    @test nonneg(yray)
    yray_n = yray / (-dot(b, yray))
    @test -dot(b, yray_n) ≈ 1 atol = TOL

    # (b) Dual-infeasible / primal-unbounded ray: −A x ∈ K, c'x < 0,
    #     normalize −c'x = 1.   A = (1,1)', b = (1,1), c = (1), x = −1.
    A2 = reshape([1.0, 1.0], 2, 1)
    c2 = [1.0]
    xray = [-1.0]
    @test dot(c2, xray) < 0
    @test nonneg(-A2 * xray)     # −A x = (1,1) ∈ R_+²
    xray_n = xray / (-dot(c2, xray))
    @test -dot(c2, xray_n) ≈ 1 atol = TOL
end
