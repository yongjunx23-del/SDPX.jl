# Experimental n=2 SPD-relative eigensolver route (R0-S diagnostic).
# Validates the range-safe relative gate, the bounded rotation, the refusal
# conditions, and records (not gates) the downstream rounding barrier.
using Test, LinearAlgebra, SDPX

const SE = SDPX.SymmetricCones
const δ = 2.0^-50
const M = [1.0 δ; δ 2.0*δ^2]           # dyadic: |M12|/sqrt(M11*M22) = 1/sqrt(2)
const tau_off = 10.0 * 2.0 * eps(Float64)

@testset "psd relative2 gate" begin
    # dyadic correlation is O(1) -> must rotate (fail = above tau_off)
    @test SE._relative2_offdiag_gate(1.0, δ, 2.0*δ^2, tau_off) === :fail
    # tiny correlation -> skip
    @test SE._relative2_offdiag_gate(1.0, 1e-20, 1.0, tau_off) === :pass
    # exactly-at-threshold region stays unresolved (interval straddles)
    # nonfinite / nonpositive refusal
    @test_throws ArgumentError SE._relative2_offdiag_gate(NaN, 1.0, 1.0, tau_off)
    @test_throws DomainError SE._relative2_offdiag_gate(1.0, 1.0, -1.0, tau_off)
end

@testset "psd relative2 eigensolve" begin
    A = copy(M); V = Matrix{Float64}(I, 2, 2); w = zeros(2)
    SE._relative2_jacobi_eigen!(A, V, w; tau_off)
    # rotation happened: off-diagonal zeroed, V not identity
    @test abs(A[1,2]) == 0.0 && abs(A[2,1]) == 0.0
    @test V != Matrix{Float64}(I, 2, 2)
    # relative-accuracy eigenvalues: ~1 and ~delta^2 (not 2*delta^2)
    @test w[1] ≈ 1.0 atol = 1e-12
    @test w[2] ≈ δ^2 atol = 1e-3 * δ^2      # relative accuracy of the small one
    # eigenvectors orthonormal
    @test V'V ≈ Matrix{Float64}(I, 2, 2) atol = 1e-12
    # V' A V is diagonal
    @test abs((V' * M * V)[1,2]) <= 1e-12
    # eigenvalues match the true ones
    ev = eigvals(Symmetric(M))
    @test sort(w) ≈ sort(ev) rtol = 1e-6
end

@testset "psd relative2 vs absolute route" begin
    # production route: absolute threshold skips the rotation entirely
    A = copy(M); V = Matrix{Float64}(I, 2, 2); w = zeros(2)
    SE._jacobi_eigen!(A, V, w)
    @test V == Matrix{Float64}(I, 2, 2)          # no rotation
    @test w[2] == 2.0*δ^2                         # reports 2*delta^2 (2x off)
    # relative route captures the small eigenvalue to relative accuracy
    A2 = copy(M); V2 = Matrix{Float64}(I, 2, 2); w2 = zeros(2)
    SE._relative2_jacobi_eigen!(A2, V2, w2; tau_off)
    @test w2[2] ≈ δ^2 rtol = 1e-3                 # relative-accuracy capture
end

@testset "psd relative2 refusals" begin
    # dimension != 2
    A3 = Matrix{Float64}(I, 3, 3); V3 = Matrix{Float64}(I, 3, 3); w3 = zeros(3)
    @test_throws ArgumentError SE._relative2_jacobi_eigen!(A3, V3, w3)
    # non-Float64
    Ab = Matrix{BigFloat}(I, 2, 2); Vb = Matrix{BigFloat}(I, 2, 2); wb = zeros(BigFloat, 2)
    @test_throws ArgumentError SE._relative2_jacobi_eigen!(Ab, Vb, wb)
    # nonpositive diagonal
    An = [1.0 0.1; 0.1 -1.0]; Vn = Matrix{Float64}(I, 2, 2); wn = zeros(2)
    @test_throws DomainError SE._relative2_jacobi_eigen!(An, Vn, wn)
end

@testset "psd relative2 route dispatch" begin
    # explicit route selection through the PSD NT scaling constructor of the
    # SymmetricCones module (PSDNTScaling is exported from there)
    sc = SE.PSDNTScaling{Float64}(2; eigen_route = :experimental_relative2)
    @test sc.eigen_route === :experimental_relative2
    # unsupported route still refuses
    @test_throws ArgumentError SE.PSDNTScaling{Float64}(2; eigen_route = :bogus)
    # non-2x2 through the route refuses at use time
    sc3 = SE.PSDNTScaling{Float64}(3; eigen_route = :experimental_relative2)
    A3 = Matrix{Float64}(I, 3, 3)
    @test_throws ArgumentError SE._psd_eigen_route!(A3, copy(A3), zeros(3), :experimental_relative2)
end

@testset "psd relative2 downstream rounding (recorded, not gated)" begin
    # D0 = V H^{-1/2} V' : the local spectral advantage is real, but Float64
    # root assembly at 2^50 scale destroys D0*M*D0 - I. This is the documented
    # rounding barrier the experiment measures; it is reported, not asserted.
    A = copy(M); V = Matrix{Float64}(I, 2, 2); w = zeros(2)
    SE._relative2_jacobi_eigen!(A, V, w; tau_off)
    Hinv = diagm(1.0 ./ sqrt.(w))
    D0 = V * Hinv * V'
    resid = D0 * M * D0 - Matrix{Float64}(I, 2, 2)
    println("RELATIVE2_DOWNSTREAM norm(D0*M*D0 - I) = ", norm(resid, Inf))
    # the relative contraction bound rho = 1/sqrt(2) < 1 holds at the operator
    # level: (1-rho)VHV' <= M <= (1+rho)VHV' in the relative sense
    @test w[1] > 0.0 && w[2] > 0.0
end

@testset "psd relative2 non-dyadic rounding barrier (recorded)" begin
    # Non-power-of-two small eigenvalue: the relative route still captures the
    # small eigenvalue to relative accuracy, but Float64 root assembly at the
    # 1/sqrt(lambda2) scale now leaves a real rounding residual in D0*M*D0-I.
    # This is the documented barrier the experiment measures; reported, not
    # gated.
    δn = 1.3 * 2.0^-50
    Mn = [1.0 δn; δn 2.0*δn^2]
    A = copy(Mn); V = Matrix{Float64}(I, 2, 2); w = zeros(2)
    SE._relative2_jacobi_eigen!(A, V, w; tau_off)
    @test w[2] ≈ δn^2 rtol = 1e-3
    Hinv = diagm(1.0 ./ sqrt.(w))
    D0 = V * Hinv * V'
    resid = D0 * Mn * D0 - Matrix{Float64}(I, 2, 2)
    println("RELATIVE2_NONDYADIC norm(D0*M*D0 - I) = ", norm(resid, Inf))
    # relative contraction bound rho = |b|/sqrt(a*c) < 1 holds at operator level
    rho = abs(Mn[1,2]) / sqrt(Mn[1,1]*Mn[2,2])
    @test rho < 1.0
    @test w[1] > 0.0 && w[2] > 0.0
end

@testset "psd relative2 D0 advantage vs absolute route (recorded)" begin
    # Production route (absolute-threshold Jacobi) on the dyadic case: the
    # rotation is skipped, D0 = diag(1, 1/sqrt(2*delta^2)) and the recovered
    # D0*M*D0 residual is catastrophic (~0.707 = rho, the relative
    # contraction). The experimental relative route's D0 attains ~1e-16/1e-31.
    δ = 2.0^-50
    M = [1.0 δ; δ 2.0*δ^2]
    A = copy(M); V0 = Matrix{Float64}(I, 2, 2); w0 = zeros(2)
    SE._jacobi_eigen!(A, V0, w0)
    @test V0 == Matrix{Float64}(I, 2, 2)             # rotation skipped
    Dprod = V0 * diagm(1.0 ./ sqrt.(w0)) * V0'
    resid_prod = Dprod * M * Dprod - Matrix{Float64}(I, 2, 2)
    A2 = copy(M); V2 = Matrix{Float64}(I, 2, 2); w2 = zeros(2)
    SE._relative2_jacobi_eigen!(A2, V2, w2; tau_off)
    Drel = V2 * diagm(1.0 ./ sqrt.(w2)) * V2'
    resid_rel = Drel * M * Drel - Matrix{Float64}(I, 2, 2)
    println("D0_RESID production=", norm(resid_prod, Inf), " relative=",
        norm(resid_rel, Inf))
    @test norm(resid_prod, Inf) > 0.1               # ~0.707: destroyed
    @test norm(resid_rel, Inf) < 1e-12              # relative-route D0
end
