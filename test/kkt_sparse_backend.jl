using SDPX
using LinearAlgebra
using SparseArrays
using StableRNGs
using Test

@testset "sparse KKT backend with symbolic reuse" begin

    """A sparse symmetric positive definite matrix with a fixed pattern."""
    function spd(rng, n, density)
        G = sprandn(rng, n, n, density)
        return sparse(Symmetric(G * transpose(G) + n * I, :L))
    end

    @testset "arithmetic support is declared, not discovered" begin
        # CHOLMOD is Float64-only. Saying so up front beats a MethodError deep
        # inside a solve.
        @test SDPX.supports_sparse_backend(Float64)
        @test !SDPX.supports_sparse_backend(BigFloat)
        @test !SDPX.supports_sparse_backend(Float32)
    end

    @testset "pattern fingerprint depends on structure, not values" begin
        rng = StableRNG(11)
        A = spd(rng, 60, 0.08)
        B = copy(A)
        B.nzval .*= 3.7                    # same structure, different values
        @test SDPX.pattern_fingerprint(A) == SDPX.pattern_fingerprint(B)

        C = copy(A)
        C[1, 60] = 1.0                     # structural change
        C[60, 1] = 1.0
        @test SDPX.pattern_fingerprint(sparse(C)) != SDPX.pattern_fingerprint(A)
    end

    @testset "symbolic analysis is reused across value changes" begin
        rng = StableRNG(12)
        A = spd(rng, 120, 0.05)
        backend = SDPX.SparseCholeskyBackend()

        @test SDPX.factorize!(backend, A)
        first_stats = SDPX.statistics(backend)
        @test first_stats.analyses == 1
        @test first_stats.factorizations == 1
        @test first_stats.reused == 0

        # Iterations change values, never the pattern — exactly the interior
        # point case §15.2 exists for.
        for scale in (1.5, 2.0, 2.5, 3.0)
            B = copy(A)
            B.nzval .*= scale
            @test SDPX.factorize!(backend, B)
        end

        stats = SDPX.statistics(backend)
        @test stats.analyses == 1                    # analysed exactly once
        @test stats.factorizations == 5
        @test stats.reused == 4
        @test stats.symbolic_reuse_ratio ≈ 4 / 5
        @test stats.failures == 0
    end

    @testset "a changed pattern forces reanalysis" begin
        # The guard that keeps reuse from being silently wrong: refactorizing
        # into a factor analysed for a different structure is the classic way
        # symbolic reuse corrupts results.
        rng = StableRNG(13)
        A = spd(rng, 80, 0.06)
        backend = SDPX.SparseCholeskyBackend()
        @test SDPX.factorize!(backend, A)
        @test SDPX.statistics(backend).analyses == 1

        different = spd(StableRNG(99), 80, 0.12)     # different pattern
        @test SDPX.factorize!(backend, different)
        @test SDPX.statistics(backend).analyses == 2

        # And a different dimension must also reanalyse rather than error.
        smaller = spd(StableRNG(7), 40, 0.08)
        @test SDPX.factorize!(backend, smaller)
        @test SDPX.statistics(backend).analyses == 3
    end

    @testset "solves are correct, and correct after reuse" begin
        rng = StableRNG(14)
        A = spd(rng, 100, 0.06)
        backend = SDPX.SparseCholeskyBackend()
        @test SDPX.factorize!(backend, A)

        rhs = collect(range(0.5, 2.5; length=100))
        x = zeros(100)
        SDPX.solve!(x, backend, rhs)
        @test A * x ≈ rhs rtol = 1e-8

        # After a reusing refactorization the solve must correspond to the NEW
        # values, not the ones the symbolic phase was computed with.
        B = copy(A)
        B.nzval .*= 2.0
        @test SDPX.factorize!(backend, B)
        @test SDPX.statistics(backend).reused >= 1
        SDPX.solve!(x, backend, rhs)
        @test B * x ≈ rhs rtol = 1e-8
    end

    @testset "solving without a factorization is an error, not a wrong answer" begin
        backend = SDPX.SparseCholeskyBackend()
        @test_throws ErrorException SDPX.solve!(zeros(3), backend, ones(3))
    end

    @testset "backend reports that it carries a reusable analysis" begin
        prob = SDPX.ingest([1.0, 1.0],
            [reshape([1.0, 0.0], 2, 1, 1), reshape([0.0, 1.0], 2, 1, 1)],
            [fill(0.0, 1, 1), fill(1.0, 1, 1)],
            reshape([1.0, 0.0], 2, 1), [1.0]; verbosity=0)
        backend = SDPX.SparseCholeskyBackend()
        info = SDPX.analyze(backend, prob)
        @test info.backend === :sparse_cholesky
        @test info.symbolic_reuse            # unlike the dense and arrow paths
        @test SDPX.backend_name(backend) === :sparse_cholesky
    end
end

@testset "sparse LDL backend for the augmented KKT system" begin

    """Augmented KKT of the shape the LP path forms: symmetric indefinite."""
    function augmented(rng, n, m, delta)
        G = sprandn(rng, n, n, 0.08)
        H = sparse(Symmetric(G * transpose(G) + n * I, :L))
        B = sprandn(rng, n, m, 0.1)
        return SDPX.augmented_kkt(H, B, delta), H, B
    end

    @testset "assembles the same system the dense path populates" begin
        rng = StableRNG(21)
        K, H, B = augmented(rng, 40, 8, 1e-8)
        n, m = size(H, 1), size(B, 2)
        @test size(K) == (n + m, n + m)
        @test K ≈ transpose(K)
        # The LDL form is symmetric: +B above AND +B' below. This differs
        # deliberately from the dense `_lp_populate_kkt!`, which writes -B above
        # and +B' below and is therefore unsymmetric (hence its use of `lu!`).
        # The two are equivalent up to the sign of the equality multiplier.
        @test K[1:n, 1:n] ≈ H + 1e-8 * I
        @test K[1:n, (n+1):(n+m)] ≈ B
        @test K[(n+1):(n+m), 1:n] ≈ transpose(B)
        @test all(≈(-1e-8), diag(K)[(n+1):(n+m)])

        # With no equality columns it degenerates to the regularized Hessian.
        K0 = SDPX.augmented_kkt(H, spzeros(n, 0), 1e-8)
        @test size(K0) == (n, n)
        @test K0 ≈ H + 1e-8 * I
    end

    @testset "factorizes an indefinite system and reuses the analysis" begin
        rng = StableRNG(22)
        K, _, _ = augmented(rng, 60, 15, 1e-8)

        # It is genuinely indefinite, so Cholesky must not be applicable.
        eigenvalues = eigvals(Matrix(Symmetric(K)))
        @test minimum(eigenvalues) < 0 < maximum(eigenvalues)

        backend = SDPX.SparseLDLBackend()
        @test SDPX.factorize!(backend, K)
        @test SDPX.statistics(backend).analyses == 1
        @test SDPX.backend_name(backend) === :sparse_ldl

        for scale in (1.001, 1.002, 1.003)
            K2 = copy(K)
            K2.nzval .*= scale
            @test SDPX.factorize!(backend, K2)
        end
        stats = SDPX.statistics(backend)
        @test stats.analyses == 1               # analysed once for four factorizations
        @test stats.factorizations == 4
        @test stats.reused == 3
        @test stats.symbolic_reuse_ratio ≈ 3 / 4
    end

    @testset "a changed pattern forces reanalysis" begin
        rng = StableRNG(23)
        K, _, _ = augmented(rng, 50, 10, 1e-8)
        backend = SDPX.SparseLDLBackend()
        @test SDPX.factorize!(backend, K)
        other, _, _ = augmented(StableRNG(77), 50, 10, 1e-8)
        if SDPX.pattern_fingerprint(other) != SDPX.pattern_fingerprint(K)
            @test SDPX.factorize!(backend, other)
            @test SDPX.statistics(backend).analyses == 2
        end
    end

    @testset "solves the augmented system" begin
        rng = StableRNG(24)
        # A larger regularization keeps the system well enough conditioned that
        # a direct solve is meaningful; at delta=1e-8 the augmented matrix is
        # deliberately near-singular and needs the refinement of §15.4.
        K, _, _ = augmented(rng, 50, 10, 1e-4)
        backend = SDPX.SparseLDLBackend()
        @test SDPX.factorize!(backend, K)
        rhs = collect(range(0.3, 1.7; length=size(K, 1)))
        x = zeros(size(K, 1))
        SDPX.solve!(x, backend, rhs)
        @test K * x ≈ rhs rtol = 1e-6

        @test_throws ErrorException SDPX.solve!(zeros(3), SDPX.SparseLDLBackend(), ones(3))
    end
end

@testset "LP formulation selector (§12.6)" begin
    # The measured crossover is ~13 nonzeros per row of the augmented KKT
    # matrix. Choosing wrongly is costly in BOTH directions — sparse is up to
    # 133x faster below the threshold and up to 2.8x slower above it — so this
    # gate is load-bearing, not a heuristic nicety.
    sel(; dim, nz, eq=4, T=Float64) =
        SDPX.select_lp_formulation(dimension=dim, nonzeros=nz, equalities=eq, arithmetic=T)

    @testset "sparse only below the measured crossover" begin
        dim = 1200
        # nnz/row well under 13 -> sparse
        @test sel(dim=dim, nz=round(Int, 3.5 * dim)) === :sparse_ldl
        @test sel(dim=dim, nz=round(Int, 5.4 * dim)) === :sparse_ldl
        @test sel(dim=dim, nz=round(Int, 10.0 * dim)) === :sparse_ldl
        # measured dense wins at 15.9 and 32.7 -> dense
        @test sel(dim=dim, nz=round(Int, 15.9 * dim)) === :dense_lu
        @test sel(dim=dim, nz=round(Int, 32.7 * dim)) === :dense_lu
    end

    @testset "no equalities selects the positive definite path" begin
        # Without equality rows the augmented system reduces to the SPD normal
        # equations, where Cholesky beats LDL.
        @test sel(dim=1200, nz=4000, eq=0) === :sparse_normal
        @test sel(dim=1200, nz=4000, eq=7) === :sparse_ldl
    end

    @testset "small problems stay dense whatever the sparsity" begin
        # BLAS is efficient enough on small matrices that sparse bookkeeping
        # dominates, so density alone must not trigger the sparse path.
        @test sel(dim=50, nz=60) === :dense_lu
        @test sel(dim=199, nz=200) === :dense_lu
        @test sel(dim=200, nz=200) !== :dense_lu
    end

    @testset "non-Float64 arithmetic always stays dense" begin
        # CHOLMOD is Float64-only; the extended-precision types have no sparse
        # factorization available at all.
        @test sel(dim=5000, nz=5000, T=BigFloat) === :dense_lu
        @test sel(dim=5000, nz=5000, T=Float32) === :dense_lu
        @test sel(dim=5000, nz=5000, T=Float64) !== :dense_lu
    end

    @testset "each formulation maps to an implemented backend" begin
        @test SDPX.formulation_backend(:sparse_normal) isa SDPX.SparseCholeskyBackend
        @test SDPX.formulation_backend(:sparse_ldl) isa SDPX.SparseLDLBackend
        @test SDPX.formulation_backend(:dense_lu) isa SDPX.LPLUBackend
        @test_throws ArgumentError SDPX.formulation_backend(:nonsense)
    end

    @testset "degenerate dimensions do not select sparse" begin
        @test sel(dim=0, nz=0) === :dense_lu
    end
end

@testset "refinement in the original KKT equations (§12.3)" begin
    function system(n, m, delta; seed=8)
        rng = StableRNG(seed)
        G = sprandn(rng, n, n, 0.08)
        H = sparse(Symmetric(G * transpose(G) + n * I, :L))
        B = sprandn(rng, n, m, 0.1)
        return SDPX.augmented_kkt(H, B, delta)
    end

    @testset "recovers accuracy lost to regularization" begin
        # The augmented system is deliberately regularized and therefore
        # ill-conditioned; refining against the ORIGINAL equations decouples the
        # attainable residual from the regularization strength. Measured:
        # direct solve degrades from 1.8e-13 at delta=1e-2 to 2.9e-05 at
        # delta=1e-10, while the refined residual stays ~3e-14 throughout.
        n, m = 120, 30
        rhs = collect(range(-1.0, 1.0; length=n + m))
        for delta in (1e-4, 1e-8)
            K = system(n, m, delta)
            backend = SDPX.SparseLDLBackend()
            @test SDPX.factorize!(backend, K)
            x = zeros(n + m)
            SDPX.solve!(x, backend, rhs)
            direct = maximum(abs, K * x .- rhs)
            steps, refined = SDPX.refine_solution!(x, backend, K, rhs; max_steps=4)
            @test refined <= direct               # never worse
            @test refined < 1e-11                 # and genuinely accurate
            @test steps <= 4
            # `x` must be the refined solution, not left at the direct one.
            @test maximum(abs, K * x .- rhs) ≈ refined rtol = 1e-6
        end
    end

    @testset "stops early when already accurate" begin
        # A well-conditioned system needs no refinement, and the routine must
        # not burn passes proving it.
        K = system(60, 10, 1e-1)
        backend = SDPX.SparseLDLBackend()
        @test SDPX.factorize!(backend, K)
        rhs = collect(range(0.2, 1.4; length=size(K, 1)))
        x = zeros(size(K, 1))
        SDPX.solve!(x, backend, rhs)
        steps, residual = SDPX.refine_solution!(x, backend, K, rhs;
            max_steps=4, tolerance=1.0)   # tolerance already met
        @test steps == 0
        @test residual < 1.0
    end

    @testset "inertia availability is reported honestly" begin
        # §12.3 asks for inertia monitoring "where available". CHOLMOD's Julia
        # interface exposes F.D only as a FactorComponent that cannot be
        # materialized, so the sign counts are unreachable and this must report
        # false rather than substitute a proxy that detects different failures.
        @test !SDPX.inertia_available(SDPX.SparseLDLBackend())
        @test !SDPX.inertia_available(SDPX.SparseCholeskyBackend())
    end
end
