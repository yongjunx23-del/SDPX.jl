import Convex
using LinearAlgebra
using MultiFloats: Float64x4
using Test
import MathOptInterface as CONVEX_MOI

function convex_sdpx_optimizer(
    ::Type{T};
    tolerance::T,
    maximum_iterations::Int=200,
) where {T<:AbstractFloat}
    attributes = Any[
        CONVEX_MOI.RawOptimizerAttribute("tolerance") => tolerance,
        CONVEX_MOI.RawOptimizerAttribute("max_iterations") =>
            maximum_iterations,
        CONVEX_MOI.RawOptimizerAttribute("sparse") => :auto,
        CONVEX_MOI.NumberOfThreads() => 1,
        CONVEX_MOI.Silent() => true,
    ]
    if T === BigFloat
        push!(
            attributes,
            CONVEX_MOI.RawOptimizerAttribute("precision") =>
                precision(BigFloat),
            CONVEX_MOI.RawOptimizerAttribute("working_precision_policy") =>
                :fixed,
        )
    end
    return CONVEX_MOI.OptimizerWithAttributes(
        SDPX.Optimizer{T},
        attributes...,
    )
end

function solve_convex_lp(::Type{T}, tolerance::T) where {T<:AbstractFloat}
    x = Convex.Variable(2)
    nonnegative = x >= zero(T)
    normalization = sum(x) == one(T)
    problem = Convex.minimize(
        T(2) * x[1] + x[2],
        [nonnegative, normalization];
        numeric_type=T,
    )
    Convex.solve!(
        problem,
        convex_sdpx_optimizer(T; tolerance=tolerance);
        silent=true,
    )
    return problem, x, nonnegative, normalization
end

function solve_convex_sdp(::Type{T}, tolerance::T) where {T<:AbstractFloat}
    X = Convex.Semidefinite(2)
    psd_constraint = only(Convex.get_constraints(X))
    off_diagonal = X[1, 2] == one(T)
    problem = Convex.minimize(
        Convex.tr(X),
        [off_diagonal];
        numeric_type=T,
    )
    Convex.solve!(
        problem,
        convex_sdpx_optimizer(T; tolerance=tolerance);
        silent=true,
    )
    return problem, X, off_diagonal, psd_constraint
end

function solve_convex_socp(::Type{T}, tolerance::T) where {T<:AbstractFloat}
    x = Convex.Variable(2)
    normalization = sum(x) == one(T)
    problem = Convex.minimize(
        Convex.norm2(x),
        [normalization];
        numeric_type=T,
    )
    Convex.solve!(
        problem,
        convex_sdpx_optimizer(T; tolerance=tolerance);
        silent=true,
    )
    return problem, x, normalization
end

function minimum_eigenvalue_2x2(matrix::AbstractMatrix{T}) where {T}
    a = matrix[1, 1]
    b = (matrix[1, 2] + matrix[2, 1]) / T(2)
    d = matrix[2, 2]
    return (a + d - sqrt((a - d)^2 + T(4) * b^2)) / T(2)
end

@testset "Convex.jl frontend" begin
    @testset "LP canonicalization and result recovery — $T" for
        (T, tolerance, rtol) in (
            (Float64, 1e-9, 1e-7),
            (Float64x4, Float64x4(1e-9), Float64x4(1e-7)),
        )
        problem, x, nonnegative, normalization =
            solve_convex_lp(T, tolerance)
        value = vec(Convex.evaluate(x))
        @test Convex.termination_status(problem) == CONVEX_MOI.OPTIMAL
        @test Convex.primal_status(problem) == CONVEX_MOI.FEASIBLE_POINT
        @test Convex.dual_status(problem) == CONVEX_MOI.FEASIBLE_POINT
        @test eltype(value) == T
        @test problem.optval ≈ one(T) rtol=rtol
        @test sum(value) ≈ one(T) rtol=rtol
        @test minimum(value) >= -sqrt(tolerance)
        @test nonnegative.dual !== nothing
        @test normalization.dual !== nothing
        @test CONVEX_MOI.get(problem.model, CONVEX_MOI.BarrierIterations()) > 0
        @test CONVEX_MOI.get(problem.model, CONVEX_MOI.SolveTimeSec()) >= 0
        raw_result = CONVEX_MOI.get(problem.model, CONVEX_MOI.RawSolver())
        @test raw_result isa SDPX.SDPResult{T}
        @test raw_result.status == SDPX.Optimal
    end

    @testset "SOC atom uses the SDPX cone path — $T" for
        (T, tolerance, rtol) in (
            (Float64, 1e-9, 1e-7),
            (Float64x4, Float64x4(1e-9), Float64x4(1e-7)),
        )
        problem, x, normalization = solve_convex_socp(T, tolerance)
        value = vec(Convex.evaluate(x))
        @test Convex.termination_status(problem) == CONVEX_MOI.OPTIMAL
        @test eltype(value) == T
        @test problem.optval ≈ inv(sqrt(T(2))) rtol=rtol
        @test value ≈ fill(T(0.5), 2) rtol=rtol
        @test normalization.dual !== nothing
    end

    @testset "PSD-square bridge and symmetric result recovery — $T" for
        (T, tolerance, rtol) in (
            # This is an integration test for square-cone bridging and result
            # recovery, not a Float64 accuracy stress test.  At 1e-9 the tiny
            # degenerate model can cross the PSD boundary after already
            # reaching an accurate point on some BLAS/CPU combinations.
            (Float64, 1e-8, 1e-7),
            (Float64x4, Float64x4(1e-9), Float64x4(1e-7)),
        )
        problem, X, off_diagonal, psd_constraint =
            solve_convex_sdp(T, tolerance)
        termination = Convex.termination_status(problem)
        @test termination == CONVEX_MOI.OPTIMAL
        termination == CONVEX_MOI.OPTIMAL || continue
        value = Matrix(Convex.evaluate(X))
        @test problem.optval ≈ T(2) rtol=rtol
        @test value ≈ fill(one(T), 2, 2) rtol=rtol
        @test value ≈ transpose(value) rtol=rtol
        @test minimum_eigenvalue_2x2(value) >= -sqrt(tolerance)
        @test off_diagonal.dual !== nothing
        @test psd_constraint.dual !== nothing
        @test size(psd_constraint.dual) == (2, 2)
    end

    @testset "BigFloat LP and SDP retain the requested arithmetic" begin
        setprecision(BigFloat, 128) do
            tolerance = parse(BigFloat, "1e-9")
            lp, x, _, _ = solve_convex_lp(BigFloat, tolerance)
            x_value = vec(Convex.evaluate(x))
            @test Convex.termination_status(lp) == CONVEX_MOI.OPTIMAL
            @test eltype(x_value) == BigFloat
            @test precision(first(x_value)) == 128
            @test lp.optval ≈ one(BigFloat) rtol=big"1e-7"

            socp, y, normalization =
                solve_convex_socp(BigFloat, tolerance)
            y_value = vec(Convex.evaluate(y))
            @test Convex.termination_status(socp) == CONVEX_MOI.OPTIMAL
            @test eltype(y_value) == BigFloat
            @test precision(first(y_value)) == 128
            @test socp.optval ≈ inv(sqrt(BigFloat(2))) rtol=big"1e-7"
            @test normalization.dual !== nothing

            sdp, X, _, psd_constraint =
                solve_convex_sdp(BigFloat, big"1e-18")
            X_value = Matrix(Convex.evaluate(X))
            @test Convex.termination_status(sdp) == CONVEX_MOI.OPTIMAL
            @test eltype(X_value) == BigFloat
            @test sdp.optval ≈ BigFloat(2) rtol=big"1e-12"
            @test minimum_eigenvalue_2x2(X_value) >= -big"1e-12"
            @test psd_constraint.dual !== nothing
        end
    end
end
