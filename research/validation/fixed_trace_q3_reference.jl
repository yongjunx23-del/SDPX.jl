using Test
using LinearAlgebra
using SparseArrays
using SDPX

function fixed_trace_problem(::Type{T}) where {T<:AbstractFloat}
    A1 = sparse(
        [2, 3, 2, 3], [1, 1, 2, 2],
        T[2, 1 // 10, 1 // 2, 3 // 2], 3, 4,
    )
    A2 = sparse(
        [2, 3, 2, 3], [3, 3, 4, 4],
        T[1, -1 // 4, 1 // 3, 2], 3, 4,
    )
    cones = [
        SDPX.SOCConstraint(A1, T[1, -1, 0]; T),
        SDPX.SOCConstraint(A2, T[2, 0, 1 // 5]; T),
    ]
    return SDPX.second_order_program(
        zeros(T, 4), cones;
        Aeq=spzeros(T, 0, 4), beq=T[], T,
    )
end

function check_fixed_trace_q3(::Type{T}) where {T<:AbstractFloat}
    problem = fixed_trace_problem(T)
    reduction = SDPX._fixed_trace_q3_reduction(problem)
    @test reduction !== nothing
    @test reduction.ownership === :owned
    @test reduction.active_ids == [1 3; 2 4]

    contribution = SDPX.FixedTraceQ3LocalElimination(reduction)
    metric = T[
        4 9;
        1 3 // 2;
        5 7
    ]
    regularization = sqrt(eps(T))
    @test SDPX.assemble_fixed_trace_q3_contribution!(
        contribution, metric, regularization,
    )
    @test all(isfinite, contribution.factors)
    @test all(>(zero(T)), contribution.inverse_pivots)
    @test contribution.regularization_scratch[2, :] == zeros(T, 2)
    @test any(!iszero, contribution.regularization_scratch[[1, 3], :])

    # A new unregularized epoch owns a fresh zero regularization record.
    @test SDPX.assemble_fixed_trace_q3_contribution!(
        contribution, metric, zero(T),
    )
    @test all(iszero, contribution.regularization_scratch)

    rhs = T[1, -2, 3, 4]
    solved = copy(rhs)
    SDPX.fixed_trace_q3_trsv_lower!(
        reduction, contribution.factors,
        contribution.inverse_pivots, solved,
    )
    SDPX.fixed_trace_q3_trsv_transpose!(
        reduction, contribution.factors,
        contribution.inverse_pivots, solved,
    )
    expected = similar(rhs)
    expected[1:2] .= Symmetric(T[4 1; 1 5]) \ rhs[1:2]
    expected[3:4] .= Symmetric(T[9 3 // 2; 3 // 2 7]) \ rhs[3:4]
    tolerance = T(256) * eps(T)
    @test maximum(abs, solved - expected) <= tolerance * max(one(T), maximum(abs, expected))

    @test_throws DimensionMismatch SDPX.assemble_fixed_trace_q3_contribution!(
        contribution, zeros(T, 2, 2), zero(T),
    )
    @test_throws ArgumentError SDPX.assemble_fixed_trace_q3_contribution!(
        contribution, metric, -one(T),
    )

    # The plan must not borrow mutable BigFloat scalars from ConicProblem.
    if T === BigFloat
        head = BigFloat(reduction.fixed_head[1])
        tail = BigFloat(reduction.tail_map[1, 1, 1])
        problem.cones[1].b[1] = BigFloat(99)
        problem.cones[1].A[2, 1] = BigFloat(77)
        @test reduction.fixed_head[1] == head
        @test reduction.tail_map[1, 1, 1] == tail
    end
end

@testset "fixed-trace Q3 Newton contribution" begin
    check_fixed_trace_q3(Float64)
    setprecision(BigFloat, 256) do
        check_fixed_trace_q3(BigFloat)
    end
end
