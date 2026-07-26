#=====================================================================
    Genericity checks (§4.1): the element type must flow cleanly from
    the inputs with no hard-coded Float64/BigFloat leaking onto the
    hot path. `@inferred` on the kernel entry points catches the kind
    of latent bug the original had at `zeros(n,n)` (:78) — an
    `Int`/`Float64` literal silently promoting away the caller's type.
=====================================================================#

using SDPX
using LinearAlgebra
using Test

@testset "genericity" begin
    @testset "no type leaks in ingest — $T" for T in (Float64, BigFloat)
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        c = T[2, 3]
        B = Matrix{T}(undef, 2, 1)
        B[1, 1], B[2, 1] = 1, 0
        b = T[2]
        prob = SDPX.ingest(c, [A], [C], B, b)
        @test eltype(prob) === T
        @test eltype(prob.c) === T
        @test eltype(prob.C[1]) === T
        @test eltype(prob.B) === T
        @test eltype(prob.b) === T
        @test prob.cons isa SDPX.DenseCons{T}
        @test eltype(prob.cons.Av[1]) === T
    end

    @testset "Int/mixed inputs promote correctly, no silent Float64 leak" begin
        # mirrors the original test's c=[2,3] (Vector{Int}) pattern: only c is
        # Int, A/C/B/b are BigFloat — the result must solve at BigFloat, not
        # silently drop to Float64 (P8/N4's "zeros(n,n) is Float64" bug class).
        T = BigFloat
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        c = [2, 3]              # Int, deliberately not T
        B = Matrix{T}(undef, 2, 0)
        b = Array{T}(undef, 0)
        prob = SDPX.ingest(c, [A], [C], B, b)
        @test eltype(prob) === BigFloat
    end

    @testset "SolverOptions{T} defaults construct without cross-type promotion" for T in (Float64, BigFloat)
        opts = SDPX.SolverOptions{T}()
        @test opts.β isa T
        @test opts.ϵ_gap isa T
        @test opts.min_step isa T
    end

    @testset "kernels are type-stable on their hot-path methods" begin
        A = rand(4, 4)
        A = A + A'
        B = rand(4, 4)
        B = B + B'
        @test (@inferred SDPX.kdot(A, B)) isa Float64
        C = zeros(4, 4)
        @inferred SDPX.kmul!(C, A, B, 1.0, 0.0)
        L = Matrix(cholesky(A + 10I).L)
        X = copy(B)
        @inferred SDPX.ktrsm!(L, X)
        panel = rand(17, 6)
        gram = rand(6, 6)
        gram = gram + transpose(gram)
        gram_reference = 1.25 .* (transpose(panel) * panel) .-
                         0.5 .* gram
        @inferred SDPX.ksyrk!(gram, panel, 1.25, -0.5)
        @test gram ≈ gram_reference rtol=2e-15 atol=2e-15
        @test issymmetric(gram)
        @test (@inferred SDPX.knrmInf(A)) isa Float64
    end

    @testset "BigFloat mutating kernels preserve inputs and independent outputs" begin
        setprecision(256) do
            A = BigFloat[2 0.25; 0.25 1.5]
            B = BigFloat[1.2 -0.4; 0.3 2.1]
            A0, B0 = deepcopy(A), deepcopy(B)
            C = zeros(BigFloat, 2, 2)
            SDPX.kmul!(C, A, B)
            @test C ≈ A0 * B0 rtol=big"1e-60"
            @test A == A0
            @test B == B0
            @test !(C[1] === C[2])
        end
    end

    @testset "Ω/tolerance kwargs on legacy sdp() promote to the inferred T" begin
        T = BigFloat
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        c = T[2, 3]
        B = Matrix{T}(undef, 2, 0)
        b = Array{T}(undef, 0)
        # Ωp, ϵ_gap, etc. passed as plain Float64/Int literals (as every caller does)
        prob = SDPX.sdp(c, [A], [C], B, b; Ωp=1, Ωd=1, ϵ_gap=1e-20, verbosity=0)
        @test prob["pObj"] isa BigFloat
        @test abs(prob["pObj"] - 2 * sqrt(T(6))) < T(1e-15)
    end
end
