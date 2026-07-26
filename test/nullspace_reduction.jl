using LinearAlgebra
using Random
using MultiFloats: Float64x4
using SDPX
using Test

"""An SDP with `n` equalities, feasible on both sides by construction so that
`Optimal` is the only correct answer and a regression shows up as a status
change rather than a slightly different number."""
function _equality_heavy_sdp(m, n, side, blocks; seed=7)
    rng = MersenneTwister(seed)
    coefficients = [zeros(m, side, side) for _ in 1:blocks]
    for block in 1:blocks, variable in 1:m
        entry = randn(rng, side, side)
        coefficients[block][variable, :, :] = entry + transpose(entry)
    end
    interior = randn(rng, m)
    constants = Vector{Matrix{Float64}}(undef, blocks)
    objective = zeros(m)
    for block in 1:blocks
        combination = zeros(side, side)
        for variable in 1:m
            combination .+= interior[variable] .* coefficients[block][variable, :, :]
        end
        constants[block] = combination - Matrix{Float64}(1.0I, side, side)
        factor = randn(rng, side, side)
        dual = factor * transpose(factor) + side * Matrix{Float64}(1.0I, side, side)
        for variable in 1:m
            objective[variable] += dot(coefficients[block][variable, :, :], dual)
        end
    end
    equalities = randn(rng, m, n)
    righthand = vec(transpose(equalities) * interior)
    return (c=objective, A=coefficients, C=constants, B=equalities, b=righthand)
end

@testset "null-space reduction" begin
    @testset "reduced solve reproduces the range-space solve" begin
        data = _equality_heavy_sdp(120, 90, 5, 2)
        problem = SDPX.ingest(data.c, data.A, data.C, data.B, data.b;
            sparse=:auto, verbosity=0)
        reference = SDPX.solve(problem; tolerance=1e-8, verbosity=0)
        @test reference.status == SDPX.Optimal

        reduction = SDPX.nullspace_reduce(problem)
        @test reduction !== nothing
        # The equalities are independent here, so the reduction is exactly
        # `m - n` variables and the reduced problem carries none of them.
        @test reduction.problem.dims.m == 120 - 90
        @test reduction.problem.dims.n == 0

        reduced = SDPX.solve(reduction.problem; tolerance=1e-8, verbosity=0)
        @test reduced.status == SDPX.Optimal

        recovered = SDPX.nullspace_expand(reduction, reduced.x, reduced.pObj)
        # The dropped constant `cᵀx_p` is restored here; forgetting it gives a
        # plausible-looking objective that is simply wrong, so it is checked
        # against the independent range-space solve rather than against itself.
        @test isapprox(recovered.objective, reference.pObj;
            rtol=1e-6, atol=1e-6)
        # The whole point of the parameterisation: any `z` satisfies the
        # equalities exactly, so the recovered `x` must too.
        @test maximum(abs, transpose(data.B) * recovered.x - data.b) < 1e-10
    end

    @testset "equality multiplier is recovered for the original certificate" begin
        # Eliminating the equalities discards their multiplier, but the final
        # certificate is stated in the original problem and needs it back.
        data = _equality_heavy_sdp(100, 70, 5, 2)
        problem = SDPX.ingest(data.c, data.A, data.C, data.B, data.b;
            sparse=:auto, verbosity=0)
        reduction = SDPX.nullspace_reduce(problem)
        @test reduction !== nothing
        reduced = SDPX.solve(reduction.problem; tolerance=1e-8, verbosity=0)
        @test reduced.status == SDPX.Optimal

        multiplier = SDPX.recover_equality_multiplier(problem, reduced.Y)
        @test length(multiplier) == 70

        residual = copy(data.c)
        for block in 1:2, variable in 1:100
            residual[variable] -=
                dot(data.A[block][variable, :, :], reduced.Y[block])
        end
        residual .-= data.B * multiplier
        @test maximum(abs, residual) < 1e-8
    end

    @testset "reduction declines rather than returning a bad basis" begin
        data = _equality_heavy_sdp(40, 20, 4, 2)
        # No equalities: there is nothing to eliminate and no reduction to make.
        problem = SDPX.ingest(data.c, data.A, data.C,
            Matrix{Float64}(undef, 40, 0), Float64[]; sparse=:auto, verbosity=0)
        @test SDPX.nullspace_reduce(problem) === nothing

        # Inconsistent equalities: `Bᵀx = b` has no solution, so no particular
        # solution exists and proceeding would silently solve a different
        # problem on a feasible set that is empty.
        equalities = zeros(40, 2)
        equalities[:, 1] .= 1.0
        equalities[:, 2] .= 1.0
        inconsistent = SDPX.ingest(data.c, data.A, data.C, equalities, [1.0, 2.0];
            sparse=:auto, verbosity=0)
        @test SDPX.nullspace_reduce(inconsistent) === nothing
    end

    @testset "the gate reflects what the benchmarks actually look like" begin
        # Worth pinning: on the repository's real problems the reduction does
        # not apply. CSDR models have no equalities at all, and the lattice
        # benchmark has 394 of 6119 variables constrained, a ratio of 0.94
        # against a threshold of 0.5. The formulation is for equality-heavy
        # models, which these are not.
        @test !SDPX.should_use_nullspace(; variables=6119, equalities=394)
        @test !SDPX.should_use_nullspace(; variables=617, equalities=0)
        @test SDPX.should_use_nullspace(; variables=1000, equalities=800)
    end

    @testset "memory estimate is safe against rank deficiency and overflow" begin
        # A memory guard that under-reports approves exactly the allocation it
        # exists to refuse. Both failure modes below were measured on the
        # previous implementation.

        # Rank deficiency. 80 equality columns of rank 1 leave a 100x99 basis,
        # not the 100x20 that the column count suggests. The old estimate said
        # 16,000 bytes against an actual 79,200 and the gate approved a
        # 20,000-byte budget.
        rng = MersenneTwister(3)
        equalities = randn(rng, 100) * transpose(randn(rng, 80))
        righthand = zeros(80)
        basis = SDPX.build_nullspace_basis(equalities, righthand)
        @test basis.rank == 1
        @test size(basis.Z) == (100, 99)
        actual = sizeof(Float64) * length(basis.Z)

        # Without the rank, the estimate must be an upper bound, not a guess
        # derived from the column count.
        @test SDPX.nullspace_memory_bytes(100, 80, Float64) >= actual
        # With the rank, it must be exact.
        @test SDPX.nullspace_memory_bytes(100, 80, Float64; rank=1) == actual
        # And the gate must refuse the budget it previously approved.
        @test !SDPX.should_use_nullspace(; variables=100, equalities=80,
            arithmetic=Float64, memory_budget_bytes=20_000)

        # Overflow. These dimensions are never allocated; the point is that the
        # arithmetic saturates instead of wrapping to a negative number that
        # compares as smaller than every budget.
        for (variables, equality_count) in ((2_000_000_000, 1),
                                            (3_000_000_000, 1_000_000))
            estimate = SDPX.nullspace_memory_bytes(variables, equality_count, Float64)
            @test estimate > 0
            @test estimate == typemax(Int)
        end
        @test SDPX.saturating_bytes(8, 2_000_000_000, 2_000_000_000) == typemax(Int)
        @test SDPX.saturating_bytes(8, 10, 10) == 800

        # Wider arithmetic needs proportionally more, and must not overflow
        # into a smaller number than the narrow case.
        @test SDPX.nullspace_memory_bytes(1000, 500, Float64x4) >
              SDPX.nullspace_memory_bytes(1000, 500, Float64)
    end

    @testset "the basis refuses a budget it cannot fit" begin
        # Enforced inside the builder, between knowing the rank and spending
        # the memory -- the only point where both facts are available.
        rng = MersenneTwister(3)
        equalities = randn(rng, 100) * transpose(randn(rng, 80))
        righthand = zeros(80)

        refused = SDPX.build_nullspace_basis(equalities, righthand;
            memory_budget_bytes=20_000)
        @test refused.reduced_dimension == 0
        @test !refused.consistent            # callers must not proceed on this
        @test isempty(refused.Z)             # nothing was allocated

        allowed = SDPX.build_nullspace_basis(equalities, righthand;
            memory_budget_bytes=1_000_000)
        @test allowed.reduced_dimension == 99
        @test allowed.consistent

        # And the reduction refuses rather than trusting an external gate.
        data = _equality_heavy_sdp(60, 40, 4, 2)
        problem = SDPX.ingest(data.c, data.A, data.C, data.B, data.b;
            sparse=:auto, verbosity=0)
        @test SDPX.nullspace_reduce(problem; memory_budget_bytes=8) === nothing
        @test SDPX.nullspace_reduce(problem) !== nothing
    end

end
