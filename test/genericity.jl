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
        schur_source = rand(7, 7)
        factor_buffer = fill(NaN, 7, 7)
        @inferred SDPX._copy_schur_factor_buffer!(
            factor_buffer,
            schur_source,
            true,
        )
        @test all(
            factor_buffer[row, column] ==
            schur_source[row, column]
            for column in 1:7 for row in column:7
        )
        @test all(
            isnan(factor_buffer[row, column])
            for column in 2:7 for row in 1:(column - 1)
        )
        @test (@inferred SDPX.knrmInf(A)) isa Float64
    end

    @testset "BLAS backend controller validates its public contract" begin
        @test SDPX.blas_backend() isa Symbol
        @test SDPX.blas_threads() >= 1
        @test_throws ArgumentError SDPX.set_blas_threads!(0)
    end

    @testset "exact duplicate equality columns are backend-independent" begin
        @test SDPX._has_exact_duplicate_columns([
            1.0 1.0
            -0.0 0.0
            3.0 3.0
        ])
        @test !SDPX._has_exact_duplicate_columns([
            1.0 1.0
            2.0 2.0
            3.0 nextfloat(3.0)
        ])
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

    @testset "solver options promote to the inferred T" begin
        T = BigFloat
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        c = T[2, 3]
        B = Matrix{T}(undef, 2, 0)
        b = Array{T}(undef, 0)
        # SolverOptions accepts plain Float64/Int literals and converts them
        # into the inferred arithmetic type without leaking Float64 into the
        # solve core.
        prob = SDPX.ingest(c, [A], [C], B, b; T=T, verbosity=0)
        options = SDPX.SolverOptions(
            T;
            primal_initial_scale=1,
            dual_initial_scale=1,
            gap_tolerance=1e-20,
            verbosity=0,
        )
        result = SDPX.solve!(prob, options)
        @test result.pObj isa BigFloat
        @test abs(result.pObj - 2 * sqrt(T(6))) < T(1e-15)
    end
end
