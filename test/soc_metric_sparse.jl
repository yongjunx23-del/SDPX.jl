using LinearAlgebra
using SparseArrays
using Test

const _SOC_METRIC_TYPES = let types = Any[Float64, BigFloat]
    try
        @eval import MultiFloats
        push!(types, MultiFloats.Float64x4)
    catch
        # Float64 and BigFloat are the required coverage.  Float64x4 is
        # exercised whenever its optional arithmetic package is installed.
    end
    types
end

function _soc_metric_options(::Type{T}) where {T}
    tolerance = T === Float64 ? T(1) / T(10)^8 : T(1) / T(10)^16
    return SDPX.SolverOptions(
        T;
        tolerance,
        maximum_iterations=40,
        parameter_policy=:fixed,
        verbosity=0,
        timing=false,
        linear_algebra_backend=:standard,
    )
end

function _soc_metric_problem(::Type{T}, A) where {T}
    variables = size(A, 2)
    b = [T(1) for _ in 1:size(A, 1)]
    # Use the public ownership seam even for the raw duplicate fixture: the
    # current sparse ingestion path preserves its CSC row/value payload, and
    # the direct accessor semantics are therefore exercised end to end.
    cone = SDPX.SOCConstraint(A, b; T=T)
    return SDPX.second_order_program(
        zeros(T, variables), [cone]; T=T,
    )
end

function _soc_metric_workspace(A, ::Type{T}) where {T}
    problem = _soc_metric_problem(T, A)
    options = _soc_metric_options(T)
    plan = SDPX.build_execution_plan(
        SDPX.AutoPlanner(), problem, options; specialization=:off,
    )
    workspace = SDPX.NativeSOCWorkspace(problem, plan.payload, options)
    # Valid Lorentz-interior NT state: w₁=2 and small typed tail entries.
    @inbounds for index in eachindex(workspace.nt_w[1])
        workspace.nt_w[1][index] = index == 1 ? T(2) : T(index) / T(100)
    end
    workspace.nt_eta_squared[1] = T(27) / T(20)
    SDPX.zero_owned!(workspace.hessian)
    return workspace, problem
end

function _soc_metric(A, ::Type{T}) where {T}
    workspace, problem = _soc_metric_workspace(A, T)
    SDPX._native_soc_add_metric!(workspace, problem.cones[1], 1)
    return workspace.hessian
end

"""Reference implementation of the frozen dense general branch."""
function _soc_metric_old_body!(workspace, cone, block)
    T = eltype(workspace.hessian)
    metric = workspace.scratch[block]
    basis = workspace.offset[block]
    variables = size(cone.A, 2)
    @inbounds for column in 1:variables
        SDPX.copy_owned!(basis, view(cone.A, :, column))
        SDPX._soc_nt_apply_hs_inverse!(
            metric,
            workspace.nt_w[block],
            workspace.nt_eta_squared[block],
            basis,
        )
        for row in column:variables
            value = zero(T)
            for coordinate in eachindex(metric)
                value += cone.A[coordinate, row] * metric[coordinate]
            end
            workspace.hessian[row, column] += value
            row == column || (workspace.hessian[column, row] += value)
        end
    end
    return workspace
end

function _soc_metric_old(A, ::Type{T}) where {T}
    workspace, problem = _soc_metric_workspace(A, T)
    _soc_metric_old_body!(workspace, problem.cones[1], 1)
    return workspace.hessian
end

function _soc_metric_pattern(::Type{T}, dimension::Int, variables::Int) where {T}
    dense = zeros(T, dimension, variables)
    active = unique(filter(1:variables) do column
        isodd(column) || column == variables
    end)
    @inbounds for column in active, coordinate in 1:dimension
        dense[coordinate, column] =
            T(coordinate + 2 * column) / T(11)
    end
    return dense
end

@testset "NativeSOC sparse general Lorentz metric" begin
    @testset "dense and canonical CSC parity across dimensions" begin
        for T in _SOC_METRIC_TYPES
            run = function ()
                for dimension in (1, 2, 3, 8)
                    dense = _soc_metric_pattern(T, dimension, 8)
                    sparse_matrix = sparse(dense)
                    @test SDPX._native_soc_sparse_has_empty_column(
                        sparse_matrix, 8,
                    )
                    dense_metric = _soc_metric(dense, T)
                    sparse_metric = _soc_metric(sparse_matrix, T)
                    @test sparse_metric == dense_metric
                    @test sparse_metric == transpose(sparse_metric)
                    for column in 1:8
                        sparse_matrix.colptr[column] ==
                        sparse_matrix.colptr[column + 1] || continue
                        @test all(iszero, sparse_metric[:, column])
                        @test all(iszero, sparse_metric[column, :])
                    end
                end
            end
            if T === BigFloat
                setprecision(BigFloat, 256) do
                    run()
                end
            else
                run()
            end
        end
    end

    @testset "empty and dense-as-CSC blocks" begin
        for T in _SOC_METRIC_TYPES
            run = function ()
                empty_dense = zeros(T, 8, 5)
                @test _soc_metric(sparse(empty_dense), T) ==
                      _soc_metric(empty_dense, T)
                dense = _soc_metric_pattern(T, 3, 4)
                for row in axes(dense, 1), column in axes(dense, 2)
                    dense[row, column] += T(1) / T(4)
                end
                dense_as_csc = sparse(dense)
                @test !SDPX._native_soc_sparse_has_empty_column(
                    dense_as_csc, 4,
                )
                @test _soc_metric(dense_as_csc, T) == _soc_metric(dense, T)
            end
            if T === BigFloat
                setprecision(BigFloat, 256) do
                    run()
                end
            else
                run()
            end
        end
    end

    @testset "duplicate and unsorted CSC follows direct accessor semantics" begin
        for T in _SOC_METRIC_TYPES
            run = function ()
                # Column 1 has rows (3, 1, 1), column 2 is empty, column 3
                # has rows (2, 1, 2), and column 4 has one row.  The raw CSC
                # therefore contains both duplicates and unsorted rowvals.
                rowval = [3, 1, 1, 2, 1, 2, 3]
                colptr = [1, 4, 4, 7, 8]
                values = T[
                    T(7) / T(10), T(1) / T(5), -T(1) / T(10),
                    T(2) / T(5), T(1) / T(10), -T(1) / T(5),
                    -T(1) / T(2),
                ]
                raw = SparseMatrixCSC{T,Int}(
                    3, 4, colptr, rowval, values,
                )
                @test raw.rowval != sort(raw.rowval)
                @test length(unique(raw.rowval)) < length(raw.rowval)
                sparse_metric = _soc_metric(raw, T)
                old_metric = _soc_metric_old(raw, T)
                @test sparse_metric == old_metric
                @test sparse_metric == transpose(sparse_metric)
                # Canonical repeated-triplet construction is a separate
                # semantic case; its sorted, duplicate-combined accessor
                # must still agree with the frozen dense body.
                canonical = sparse(
                    rowval,
                    [1, 1, 1, 3, 3, 3, 4],
                    values,
                    3,
                    4,
                )
                @test _soc_metric(canonical, T) ==
                      _soc_metric_old(canonical, T)
                @test _soc_metric(canonical, T) ==
                      _soc_metric(Matrix(canonical), T)
            end
            if T === BigFloat
                setprecision(BigFloat, 256) do
                    run()
                end
            else
                run()
            end
        end
    end

    @testset "BigFloat ownership and fixed-trace route" begin
        setprecision(BigFloat, 256) do
            dense = _soc_metric_pattern(BigFloat, 8, 5)
            workspace, problem = _soc_metric_workspace(sparse(dense), BigFloat)
            basis_id = objectid(workspace.offset[1])
            metric_id = objectid(workspace.scratch[1])
            hessian_id = objectid(workspace.hessian)
            SDPX._native_soc_add_metric!(workspace, problem.cones[1], 1)
            @test all(precision(value) == 256 for value in workspace.hessian)
            @test length(unique(objectid.(workspace.hessian))) ==
                  length(workspace.hessian)
            @test all(precision(value) == 256 for value in workspace.offset[1])
            @test all(precision(value) == 256 for value in workspace.scratch[1])
            @test length(unique(objectid.(workspace.offset[1]))) ==
                  length(workspace.offset[1])
            @test length(unique(objectid.(workspace.scratch[1]))) ==
                  length(workspace.scratch[1])
            @test objectid(workspace.offset[1]) == basis_id
            @test objectid(workspace.scratch[1]) == metric_id
            @test objectid(workspace.hessian) == hessian_id
            SDPX.zero_owned!(workspace.hessian)
            SDPX._native_soc_add_metric!(workspace, problem.cones[1], 1)
            @test objectid(workspace.offset[1]) == basis_id
            @test objectid(workspace.scratch[1]) == metric_id
            @test objectid(workspace.hessian) == hessian_id
        end
        A = zeros(3, 2)
        A[2, 1] = 1
        A[3, 2] = 1
        problem = SDPX.second_order_program(
            [-1.0, 0.0],
            [SDPX.SOCConstraint(A, [1.0, 0.0, 0.0])],
        )
        options = _soc_metric_options(Float64)
        plan = SDPX.build_execution_plan(
            SDPX.AutoPlanner(), problem, options,
        )
        @test plan.payload.cone.execution isa SDPX.FixedTraceQ3Execution
        @test plan.payload.cone.specialization === :fixed_trace_q3
    end

    @testset "native dense/sparse solve parity" begin
        # Three tiny Lorentz blocks with two unused (structurally empty)
        # variables exercise the same native route as the many-tiny-cones
        # pathological family without depending on benchmark loaders.
        dense_A = zeros(6, 8)
        for index in 1:6
            dense_A[index, index] = 1.0
        end
        sparse_A = sparse(dense_A)
        dense_Aeq = zeros(6, 8)
        for index in 1:6
            dense_Aeq[index, index] = 1.0
        end
        sparse_Aeq = sparse(dense_Aeq)
        objective = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rhs = [3.0, 1.0, 3.0, 1.0, 3.0, 1.0]
        dense_problem = SDPX.second_order_program(
            objective, dense_A, zeros(6);
            cone_dims=[2, 2, 2], Aeq=dense_Aeq, beq=rhs,
        )
        sparse_problem = SDPX.second_order_program(
            objective, sparse_A, zeros(6);
            cone_dims=[2, 2, 2], Aeq=sparse_Aeq, beq=rhs,
        )
        dense = SDPX.solve_socp(
            dense_problem;
            tolerance=1e-8,
            maximum_iterations=80,
            verbosity=0,
            timing=false,
            specialization=:off,
            presolve=false,
        )
        sparse_result = SDPX.solve_socp(
            sparse_problem;
            tolerance=1e-8,
            maximum_iterations=80,
            verbosity=0,
            timing=false,
            specialization=:off,
            presolve=false,
        )
        @test sparse_result.status === dense.status
        @test sparse_result.iterations == dense.iterations
        @test sparse_result.pObj ≈ dense.pObj atol=1e-10
        @test sparse_result.dObj ≈ dense.dObj atol=1e-10
        @test sparse_result.diagnostics.selected_algorithms.certificate.valid ==
              dense.diagnostics.selected_algorithms.certificate.valid
        @test sparse_result.diagnostics.selected_algorithms.certificate.valid
    end
end
