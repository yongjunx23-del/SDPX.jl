using Test
using LinearAlgebra
using SparseArrays
using SDPX

include(joinpath(@__DIR__, "..", "benchmark", "generators", "pathological.jl"))

"Run a callback with an isolated BigFloat precision context."
function _pathological_with_precision(f, ::Type{T}) where {T}
    if T === BigFloat
        return setprecision(BigFloat, 256) do
            f()
        end
    end
    return f()
end

function _pathological_type_cases(::Type{T}) where {T}
    _pathological_with_precision(T) do
        for name in pathological_cases()
            first = build_pathological_problem(name, T)
            second = build_pathological_problem(name, T)

            # The builder contract is stable and deterministic.  In
            # particular, a high-precision objective is never narrowed to a
            # Float64 reference during construction.
            @test first.kind in (:sdp, :socp)
            @test first.case == name
            @test first.source_parameters == second.source_parameters
            @test first.expected_status == second.expected_status
            @test eltype(first.problem) == T
            if first.expected !== nothing
                @test first.expected isa T
                @test first.expected == second.expected
            end

            if first.kind === :socp
                @test first.problem.variables == length(first.problem.c)
                @test all(eltype(cone.A) == T for cone in first.problem.cones)
                @test all(eltype(cone.b) == T for cone in first.problem.cones)
            else
                @test first.problem.dims.m == length(first.problem.c)
                @test all(eltype(matrix) == T for matrix in first.problem.C)
            end
        end

        tangent = build_pathological_problem(:socp_near_tangent, T)
        eps = tangent.source_parameters.epsilon
        @test tangent.problem.cones[1].b == T[one(T), one(T), eps]
        @test tangent.expected == sqrt(one(T) + eps * eps) - one(T)

        small_eigen = build_pathological_problem(:sdp_small_eigenvalue, T)
        small_eps = small_eigen.source_parameters.epsilon
        @test small_eigen.problem.C[1][end, end] == -small_eps
        @test small_eigen.expected == -small_eps

        congruence = build_pathological_problem(:sdp_congruence_scaling, T)
        scale = congruence.source_parameters.scale
        @test congruence.problem.C[1][1, 2] == -scale
        @test congruence.problem.C[1][2, 1] == -scale
        @test congruence.problem.C[1][2, 2] == -(scale * scale)

        infeasible = build_pathological_problem(:sdp_infeasible_margin, T)
        margin = infeasible.source_parameters.epsilon
        @test infeasible.problem.C[1][2, 2] == margin
    end
end

@testset "typed pathological benchmark builders" begin
    _pathological_type_cases(Float64)

    # MultiFloats is an optional arithmetic package for downstream users.  A
    # source checkout with it installed exercises direct decimal parsing and
    # typed storage; otherwise this focused test remains a valid Float64/MPFR
    # check.
    try
        @eval import MultiFloats
        _pathological_type_cases(MultiFloats.Float64x4)
    catch exception
        @info "MultiFloats unavailable; skipping Float64x4 pathological builder test" exception
    end

    _pathological_type_cases(BigFloat)
end

@testset "pathological analytic smoke solves" begin
    lp = build_pathological_problem(:lp_degenerate_scaled, Float64)
    lp_result = SDPX.solve(
        lp.problem;
        tolerance=1.0e-8,
        maximum_iterations=120,
        threads=1,
        verbosity=0,
    )
    @test lp_result.status === SDPX.Optimal
    @test lp_result.pObj ≈ lp.expected atol=5.0e-7

    soc = build_pathological_problem(:socp_near_tangent, Float64)
    soc_result = SDPX.solve_socp(
        soc.problem;
        tolerance=1.0e-8,
        maximum_iterations=120,
        threads=1,
        verbosity=0,
    )
    @test soc_result.status === SDPX.Optimal
    @test soc_result.pObj ≈ soc.expected atol=1.0e-7

    sdp = build_pathological_problem(:sdp_small_eigenvalue, Float64)
    sdp_result = SDPX.solve(
        sdp.problem;
        tolerance=1.0e-8,
        maximum_iterations=120,
        threads=1,
        verbosity=0,
    )
    @test sdp_result.status === SDPX.Optimal
    @test sdp_result.pObj ≈ sdp.expected atol=5.0e-8
end
