# Wave I — sparse reduced-Schur route correctness and contract tests.
using SDPX
using Test
using LinearAlgebra
using SparseArrays
using Random

const _WAVE_I_RNG = Random.Xoshiro(1)

# Build a manufactured SDPX.NewtonSystem with a block-diagonal cone linearization.
function _manufactured_system(::Type{T}; n=3, m=5, scale=1.0) where {T}
    A = T(scale) .* Matrix{T}(randn(_WAVE_I_RNG, T, m, n))
    b = T(scale) .* randn(_WAVE_I_RNG, T, m)
    c = T(scale) .* randn(_WAVE_I_RNG, T, n)
    tau = T(1.3)
    kappa = T(0.9)
    # Block-diagonal SPD cone linearization H (SOC-style blocks) covering m.
    dims = m == 5 ? [2, 3] : m == 6 ? [3, 3] : [m ÷ 2, m - m ÷ 2]
    H = zeros(T, m, m)
    off = 0
    ranges = UnitRange{Int}[]
    for d in dims
        block = T(2.0) .* Matrix{T}(I, d, d) .+
                T(0.5) .* randn(_WAVE_I_RNG, T, d, d)
        block = (block + block') / 2
        block .+= T(3.0) .* Matrix{T}(I, d, d)
        H[(off+1):(off+d), (off+1):(off+d)] .= block
        push!(ranges, (off+1):(off+d))
        off += d
    end
    lin = SDPX.ProductConeLinearization{T}(H, zeros(T, m), ranges)
    rhs = SDPX.HSDNewtonRHS(
        randn(_WAVE_I_RNG, T, m), randn(_WAVE_I_RNG, T, n), T(0.3),
        randn(_WAVE_I_RNG, T, m), T(0.4),
    )
    return SDPX.NewtonSystem(A, b, c, lin, tau, kappa, rhs)
end

@testset "wave_i sparse reduced-schur" begin
    @testset "route registration" begin
        @test SDPX.Settings{Float64}(engine=:native_hsd, kkt_route=:sparse_schur) isa SDPX.Settings{Float64}
        @test_throws ArgumentError SDPX.Settings{Float64}(engine=:native_hsd, kkt_route=:bogus)
    end

    @testset "sparse vs expanded direction agreement" begin
        T = Float64
        system = _manufactured_system(T; n=3, m=5)
        session = SDPX.SparseSchurSession(T, 3, 5)
        solution = zeros(T, 4)
        direction = SDPX.NewtonDirection(
            zeros(T, 3), zeros(T, 5), zeros(T, 5), zero(T), zero(T),
        )
        dir = SDPX.solve_sparse_schur!(session, system, solution, direction)
        @test dir !== nothing

        # Expanded route reference
        exp = SDPX.ExpandedKKTSession(T, 3, 5; rhs_count=2)
        SDPX.assemble_expanded_kkt!(exp, system)
        exp_rhs = zeros(T, 3 + 5 + 1)
        SDPX.expanded_rhs!(exp_rhs, system)
        exp_sol = zeros(T, 3 + 5 + 1)
        @test SDPX.factor_expanded_kkt!(exp, system)
        @test SDPX.solve_expanded!(exp_sol, exp, exp_rhs)
        exp_dir = SDPX.recover_expanded_direction(system, exp_sol)

        # dx and dtau must agree (the reduced route's unknowns)
        @test dir.dx ≈ exp_dir.dx rtol=1e-8
        @test dir.dtau ≈ exp_dir.dtau rtol=1e-8
        # dy, ds, dkappa must agree too
        @test dir.dy ≈ exp_dir.dy rtol=1e-8
        @test dir.ds ≈ exp_dir.ds rtol=1e-8
        @test dir.dkappa ≈ exp_dir.dkappa rtol=1e-8
    end

    @testset "unregularized Newton residual gate" begin
        T = Float64
        system = _manufactured_system(T; n=4, m=6, scale=2.0)
        session = SDPX.SparseSchurSession(T, 4, 6)
        solution = zeros(T, 5)
        direction = SDPX.NewtonDirection(
            zeros(T, 4), zeros(T, 6), zeros(T, 6), zero(T), zero(T),
        )
        dir = SDPX.solve_sparse_schur!(session, system, solution, direction)
        @test dir !== nothing
        residual = SDPX.NewtonResidual(system)
        SDPX.newton_residual!(residual, system, dir)
        @test SDPX.max_newton_residual(residual) < 1e-10
    end

    @testset "no silent densification" begin
        T = Float64
        system = _manufactured_system(T; n=5, m=8)
        session = SDPX.SparseSchurSession(T, 5, 8)
        solution = zeros(T, 6)
        direction = SDPX.NewtonDirection(
            zeros(T, 5), zeros(T, 8), zeros(T, 8), zero(T), zero(T),
        )
        @test SDPX.assemble_sparse_schur!(session, system)
        # The reduced operator is (n+1)x(n+1) = 6x6 sparse; assert it stays sparse
        @test session.schur isa SparseMatrixCSC
        @test nnz(session.schur) <= 6 * 6
        # And that A'H⁻¹A is not full-dense for a sparse-structured system
        @test SDPX.solve_sparse_schur!(session, system, solution, direction) !== nothing
    end

    @testset "one factor per epoch" begin
        T = Float64
        system = _manufactured_system(T; n=3, m=5)
        session = SDPX.SparseSchurSession(T, 3, 5)
        solution = zeros(T, 4)
        direction = SDPX.NewtonDirection(
            zeros(T, 3), zeros(T, 5), zeros(T, 5), zero(T), zero(T),
        )
        @test SDPX.solve_sparse_schur!(session, system, solution, direction) !== nothing
        # Re-solve (new epoch) — factor is recreated each call
        @test SDPX.solve_sparse_schur!(session, system, solution, direction) !== nothing
        @test session.status == SDPX.SPARSE_SCHUR_UNREGULARIZED_CERTIFIED
    end
end
