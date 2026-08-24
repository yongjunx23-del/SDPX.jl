#=====================================================================#
#    Pure SOC-family NativeConeProgram lowering contract tests (v0.5).
#
#    The guard bootstraps the route/lowering files when this test is run
#    standalone against a package build that has not yet wired the central
#    include.  The fixtures exercise only the native IR -> ConicProblem
#    boundary; they never invoke a planner or a second solver.
#=====================================================================#

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

if !isdefined(SDPX, :lower_soc_native)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "lower_soc.jl"))
end

using Test
using SparseArrays
using LinearAlgebra

"""Build a valid native program from ordered product/row block descriptors."""
function _soc_fixture(
    products::Tuple;
    rows::Tuple=(),
    coefficients::AbstractVector{<:AbstractVector{<:Tuple}}=Vector{Vector{Tuple{Int,Float64}}}(),
    rhs::AbstractVector{<:Real}=Float64[],
    T::Type{<:AbstractFloat}=Float64,
    sense=SDPX.Minimize(),
    objective::AbstractVector{<:Real}=Float64[],
    constant::Real=0,
    source_model::UInt64=UInt64(100),
)
    blocks = SDPX.NativeBlock[]
    offset = 1
    for (domain, shape) in products
        push!(blocks, SDPX.NativeBlock(domain, shape, offset))
        offset += SDPX.block_length(blocks[end])
    end
    variables = offset - 1
    objective_full = isempty(objective) ? zeros(T, variables) : T.(objective)
    length(objective_full) == variables || throw(ArgumentError("objective length mismatch"))

    row_blocks = SDPX.RowBlock[]
    row_offset = 1
    for (domain, source_rows) in rows
        shape = length(source_rows)
        push!(row_blocks, SDPX.RowBlock(domain, row_offset, shape, collect(source_rows)))
        row_offset += SDPX.row_block_length(row_blocks[end])
    end
    num_rows = row_offset - 1
    rhs_full = isempty(rhs) ? zeros(T, num_rows) : T.(rhs)
    length(rhs_full) == num_rows || throw(ArgumentError("rhs length mismatch"))

    matrix_rows = Int[]
    matrix_columns = Int[]
    matrix_values = T[]
    for (row, entries) in enumerate(coefficients)
        for (variable, value) in entries
            push!(matrix_rows, row)
            push!(matrix_columns, variable)
            push!(matrix_values, T(value))
        end
    end
    equality_matrix = sparse(matrix_rows, matrix_columns, matrix_values, num_rows, variables)
    primal = [SDPX.VariableRef(source_model, 1, index) for index in 1:variables]
    constraint_dual = [SDPX.ConstraintRef(source_model, 1, index) for index in 1:num_rows]
    dual_slack = copy(primal)
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(T),
        sense,
        objective_full,
        T(constant),
        equality_matrix,
        rhs_full,
        blocks,
        row_blocks,
        primal,
        constraint_dual,
        dual_slack,
        source_model,
    )
end

function _soc_caught(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

@testset "SOC lowerer preserves native block and row order" begin
    program = _soc_fixture(
        (
            (SDPX.LorentzCone(), 3),
            (SDPX.Reals(), 1),
            (SDPX.ZeroCone(), 1),
        );
        rows=(
            (SDPX.LorentzCone(), [1, 2, 3]),
            (SDPX.ZeroCone(), [4]),
        ),
        coefficients=[
            [(1, 2.0), (2, -1.0)],
            [(1, 3.0), (3, 4.0)],
            [(2, 5.0), (4, 6.0)],
            [(1, 7.0), (5, 8.0)],
        ],
        rhs=[11.0, 12.0, 13.0, 14.0],
        objective=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    lowered = SDPX.lower_soc_native(program)
    @test lowered isa SDPX.SOCLowering{Float64}
    @test lowered.core isa SDPX.ConicProblem{Float64}
    @test length(lowered.core.cones) == 2
    @test [origin.kind for origin in lowered.cone_origins] == [:product, :row]
    @test [origin.block for origin in lowered.cone_origins] == [1, 1]
    @test [origin.core_cone for origin in lowered.cone_origins] == [1, 2]

    # Product SOC is one sparse identity affine row, not scalarised rows.
    @test lowered.core.cones[1].A isa SparseMatrixCSC{Float64,Int}
    @test size(lowered.core.cones[1].A) == (3, 5)
    @test nnz(lowered.core.cones[1].A) == 3
    @test lowered.core.cones[1].A[1, 1] == 1.0
    @test lowered.core.cones[1].A[3, 3] == 1.0

    # NCP row is A*x-rhs in K, while ConicProblem stores A*x+b in Q.
    @test lowered.core.cones[2].A isa SparseMatrixCSC{Float64,Int}
    @test lowered.core.cones[2].A[1, 1] == 2.0
    @test lowered.core.cones[2].A[1, 2] == -1.0
    @test lowered.core.cones[2].A[3, 4] == 6.0
    @test lowered.core.cones[2].b == [-11.0, -12.0, -13.0]

    # Product/row ZeroCone coordinates become equalities in source order.
    @test lowered.core.Aeq isa SparseMatrixCSC{Float64,Int}
    @test size(lowered.core.Aeq) == (2, 5)
    @test lowered.core.Aeq[1, 5] == 1.0
    @test lowered.core.Aeq[2, 1] == 7.0
    @test lowered.core.Aeq[2, 5] == 8.0
    @test lowered.core.beq == [0.0, 14.0]
    @test lowered.equality_origins == [
        SDPX.SOCEqualityOrigin(:variable_dual_slack, 3, 1, 1),
        SDPX.SOCEqualityOrigin(:equality, 2, 1, 1),
    ]
    @test lowered.equality_duals == [SDPX.SOCEqualityDual(false, 1), SDPX.SOCEqualityDual(true, 1)]
end

@testset "RSOC exact Lorentz map and adjoint reconstruction" begin
    program = _soc_fixture(
        ((SDPX.RotatedLorentzCone(), 4),);
        objective=[2.0, -3.0, 4.0, 5.0],
        rhs=Float64[],
    )
    lowered = SDPX.lower_soc_native(program)
    cone = lowered.core.cones[1]
    sqrt_two = sqrt(2.0)
    @test cone.A isa SparseMatrixCSC{Float64,Int}
    @test Matrix(cone.A) == [
        1.0 1.0 0.0 0.0;
        1.0 -1.0 0.0 0.0;
        0.0 0.0 sqrt_two 0.0;
        0.0 0.0 0.0 sqrt_two;
    ]
    primal_record = only(lowered.primal_records)
    dual_record = only(lowered.dual_records)
    dual_core = [2.0, -1.0, 0.5, -3.0]
    @test SDPX.reconstruct_soc_dual(dual_record, dual_core) ≈ [
        1.0, 3.0, 0.5 * sqrt_two, -3.0 * sqrt_two,
    ]
    @test dual_record.map == transpose(SparseArrays.sparse(Matrix(cone.A)))
    @test primal_record.map * Matrix(cone.A) ≈ Matrix{Float64}(I, 4, 4)
    @test primal_record.source_indices == collect(1:4)
    @test dual_record.source_indices == collect(1:4)
end

@testset "SOC row RSOC map applies to A and -rhs" begin
    program = _soc_fixture(
        ((SDPX.Reals(), 3),);
        rows=((SDPX.RotatedLorentzCone(), [1, 2, 3]),),
        coefficients=[[(1, 1.0)], [(2, 2.0)], [(3, 3.0)]],
        rhs=[4.0, 5.0, 6.0],
    )
    lowered = SDPX.lower_soc_native(program)
    cone = only(lowered.core.cones)
    sqrt_two = sqrt(2.0)
    @test Matrix(cone.A) == [
        1.0 2.0 0.0;
        1.0 -2.0 0.0;
        0.0 0.0 3.0 * sqrt_two;
    ]
    @test cone.b ≈ [-9.0, 1.0, -6.0 * sqrt_two]
    @test only(lowered.cone_origins).kind === :row
    @test only(lowered.primal_records).source_indices == [1, 2, 3]
end

@testset "objective, sparse ownership, and explicit precision" begin
    program = _soc_fixture(
        ((SDPX.LorentzCone(), 2),);
        sense=SDPX.Maximize(),
        objective=[1.0, -2.0],
        constant=7.5,
    )
    lowered = SDPX.lower_soc_native(program)
    @test lowered.objective_sign == -1
    @test lowered.objective_constant == 7.5
    @test lowered.core.c == [-1.0, 2.0]
    @test lowered.core.cones[1].A isa SparseMatrixCSC
    @test lowered.primal_refs == program.primal_reconstruction
    @test lowered.constraint_dual_refs == program.constraint_dual_reconstruction

    setprecision(BigFloat, 64) do
        big = _soc_fixture(
            ((SDPX.RotatedLorentzCone(), 3),);
            T=BigFloat,
            objective=[BigFloat(1), BigFloat(2), BigFloat(3)],
            constant=BigFloat(4),
        )
        lowered_big = SDPX.lower_soc_native(big)
        values = nonzeros(lowered_big.core.cones[1].A)
        @test all(precision(value) >= 256 for value in values)
        @test all(precision(value) >= 256 for value in lowered_big.core.cones[1].b)
        @test all(precision(value) >= 256 for value in nonzeros(only(lowered_big.primal_records).map))
        @test all(precision(value) >= 256 for value in nonzeros(only(lowered_big.dual_records).map))
        @test lowered_big.objective_constant == 4
        # Mutating source MPFR values after lowering cannot mutate core data.
        big.objective_vector[1] = BigFloat(99)
        @test lowered_big.core.c[1] == 1
        @test lowered_big.core.cones[1].A[1, 1] == 1
    end
end

@testset "scalar SOC rays and fail-closed unsupported routes" begin
    scalar = _soc_fixture(((SDPX.LorentzCone(), 1),))
    scalar_low = SDPX.lower_soc_native(scalar)
    @test size(only(scalar_low.core.cones).A) == (1, 1)
    @test only(scalar_low.core.cones).A[1, 1] == 1

    orthant = _soc_fixture(((SDPX.Nonnegative(), 2),))
    @test (err = _soc_caught(() -> SDPX.lower_soc_native(orthant));
           err isa SDPX.SOCLoweringError && err.reason === :non_soc_route)

    mixed = _soc_fixture(((SDPX.LorentzCone(), 2), (SDPX.PSDCone(), 2)))
    @test (err = _soc_caught(() -> SDPX.lower_soc_native(mixed));
           err isa SDPX.UnsupportedNativeConeRoute)

    psd = _soc_fixture(((SDPX.PSDCone(), 2),))
    @test (err = _soc_caught(() -> SDPX.lower_soc_native(psd));
           err isa SDPX.SOCLoweringError && err.reason === :non_soc_route)

    bad_rsoc = _soc_fixture(((SDPX.RotatedLorentzCone(), 2),))
    @test (err = _soc_caught(() -> SDPX.lower_soc_native(bad_rsoc));
           err isa SDPX.SOCLoweringError && err.reason === :unsupported_dimension)

    # A source with no Lorentz block is rejected instead of synthesising a
    # dummy cone.  It reaches the route guard first and remains fail-closed.
    no_cone = _soc_fixture(((SDPX.Reals(), 1),))
    @test (err = _soc_caught(() -> SDPX.lower_soc_native(no_cone));
           err isa SDPX.SOCLoweringError && err.reason === :non_soc_route)
end

@testset "typed lowering records contain no Any fields" begin
    for type in (
        SDPX.SOCLowering,
        SDPX.SOCRecordOrigin,
        SDPX.SOCPrimalReconstruction,
        SDPX.SOCDualReconstruction,
        SDPX.SOCEqualityOrigin,
        SDPX.SOCEqualityDual,
    )
        @test all(field_type -> field_type !== Any, fieldtypes(type))
    end
    @test !(:orientation in fieldnames(SDPX.SOCLowering))
    @test !(:dual_model in fieldnames(SDPX.SOCLowering))
end
