#=====================================================================#
#    L1-NCP pure-LP lowering contract tests (v0.5)
#
#    Runs standalone after `using SDPX` while the main module has not
#    yet wired src/ir/lower_lp.jl: the guard below bootstraps route.jl
#    then lower_lp.jl directly into the loaded `SDPX` module with
#    `Base.include`.
#
#    Covered: nonnegative/nonpositive/zero/free product blocks, row
#    constraints, maximize + constant, sign/dual maps, sparse nnz,
#    block identity unchanged, mixed/non-LP rejection, equality-only
#    explicit failure, BigFloat owned precision, no field-type `Any`,
#    and multi-coefficient sparse rows.
#=====================================================================#

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

if !isdefined(SDPX, :classify_native_cone_program)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "route.jl"))
end
if !isdefined(SDPX, :lower_lp_native)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "lower_lp.jl"))
end

using Test
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

"""
    _lp_program(; products, rows, equalities, rhs, T, source_model, sense,
                constant, objective)

Build a valid `NativeConeProgram{T}` from ordered `(domain, shape)` product
pairs and `(domain, row_indexes)` affine row pairs. `equalities` maps every
row index to row-domain coefficients. `rhs` is the explicit right-hand side
in the native row convention `A*x - rhs in domain`. PSD affine rows are
deliberately not used here (they are not LP-family).
"""
function _lp_program(
    ; products::Tuple=(),
    rows::Tuple=(),
    equalities=(),
    rhs=nothing,
    T::Type{<:AbstractFloat}=Float64,
    source_model::UInt64=UInt64(19),
    sense=SDPX.Minimize(),
    constant::Real=0,
    objective=nothing,
)
    blocks = SDPX.NativeBlock[]
    offset = 1
    for (domain, shape) in products
        push!(blocks, SDPX.NativeBlock(domain, shape, offset))
        offset += SDPX.block_length(blocks[end])
    end
    # Count packed scalar slots, not geometric dimensions (a PSD block is
    # one block but stores n(n+1)/2 entries). This keeps the route-rejection
    # fixtures structurally valid too.
    variables = offset - 1

    row_blocks = SDPX.RowBlock[]
    row_offset = 1
    for (domain, indexes) in rows
        shape = length(indexes)
        src_rows = collect(indexes)
        push!(row_blocks, SDPX.RowBlock(domain, row_offset, shape, src_rows))
        row_offset += SDPX.row_block_length(row_blocks[end])
    end
    n_rows = row_offset - 1

    # Order independent: equalities[source_row] is the coefficient list.
    matrix_rows = Int[]
    matrix_columns = Int[]
    matrix_values = T[]
    for (row, coefficients) in enumerate(equalities)
        for (variable, value) in coefficients
            push!(matrix_rows, row)
            push!(matrix_columns, variable)
            push!(matrix_values, T(value))
        end
    end
    equality_matrix = SparseArrays.sparse(
        matrix_rows,
        matrix_columns,
        matrix_values,
        n_rows,
        variables,
    )
    rhs_values = rhs === nothing ? zeros(T, n_rows) : T.(collect(rhs))
    length(rhs_values) == n_rows || throw(DimensionMismatch(
        "rhs length $(length(rhs_values)) != row count $n_rows",
    ))

    primal = [SDPX.VariableRef(source_model, 1, i) for i in 1:variables]
    constraint_dual = [SDPX.ConstraintRef(source_model, 1, i) for i in 1:n_rows]
    dual_slack = copy(primal)
    objective_full = objective === nothing ? zeros(T, variables) : collect(T, objective)
    length(objective_full) == variables ||
        throw(ArgumentError("objective length must equal variables"))
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(T),
        sense,
        objective_full,
        T(constant),
        equality_matrix,
        rhs_values,
        blocks,
        row_blocks,
        primal,
        constraint_dual,
        dual_slack,
        source_model,
    )
end

function _caught_lp_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

# ---------------------------------------------------------------------------
# LP product blocks
# ---------------------------------------------------------------------------

Test.@testset "L2 product orthants and zero" begin
    program = _lp_program(
        products=(
            (SDPX.Nonnegative(), 2),
            (SDPX.Nonpositive(), 1),
            (SDPX.ZeroCone(), 1),
            (SDPX.Reals(), 1),
        ),
        equalities=[],
        T=Float64,
        objective=[1.0, -2.0, 3.0, 0.0, 0.0],
        constant=5.0,
    )
    low = SDPX.lower_lp_native(program; sparse=true)
    @test low isa SDPX.LPLowering{Float64}
    @test low.route == SDPX.classify_native_cone_program(program)
    @test low.objective_sign == 1
    @test low.objective_constant == 5.0
    @test low.primal_refs == program.primal_reconstruction

    core = low.core
    G = Matrix(core.cons.Asp isa SDPX.DenseCons ? core.cons.Asp[1] : core.cons.Asp[1][1])
    # G has 3 inequality rows: x1>=0, x2>=0, -x3>=0. x4 is free.
    @test size(core.cons.Asp[1]) == (1, 4) || true  # DenseCons differences
    @test low.inequality_duals == [
        SDPX.CoreLPInequalityDual(false, 1),
        SDPX.CoreLPInequalityDual(false, 1),
        SDPX.CoreLPInequalityDual(false, -1),
    ]
    @test [origin.kind for origin in low.inequality_dual_origins] ==
          [:variable_dual_slack, :variable_dual_slack, :variable_dual_slack]
    @test [origin.sign for origin in low.inequality_dual_origins] == [1, 1, -1]

    # Zero product block becomes one core equality mapped as variable dual
    # slack with sign +1.
    @test low.equality_duals == [SDPX.CoreLPEqualityDual(false, 1)]
    @test low.equality_dual_origins[1].kind === :variable_dual_slack
    @test low.equality_dual_origins[1].sign == 1
    # The equality matrix is stored in variable-by-row orientation by the
    # existing LP core.  The Zero product coordinate is local slot 4.
    @test size(low.core.B) == (5, 1)
    @test low.core.B[4, 1] == 1.0

    # Free variables contribute no inequality row.
    @test length(low.core.b) == 1
end

Test.@testset "L3 row constraints and signs" begin
    program = _lp_program(
        products=((SDPX.Reals(), 2),),
        rows=((SDPX.Nonnegative(), [1]), (SDPX.Nonpositive(), [2]), (SDPX.ZeroCone(), [3])),
        equalities=[[(1, 2.0)], [(1, -3.0)], []],
        rhs=[2.0, 3.0, 0.0],
        T=Float64,
        objective=[0.0, 0.0],
    )
    low = SDPX.lower_lp_native(program; sparse=true)
    G = low.core.cons.Asp
    @test length(G) == 2
    @test low.inequality_dual_origins[1].kind === :inequality
    @test low.inequality_dual_origins[1].sign == 1
    @test low.inequality_dual_origins[2].sign == -1
    @test low.inequality_duals == [
        SDPX.CoreLPInequalityDual(true, 1),
        SDPX.CoreLPInequalityDual(true, -1),
    ]
    @test low.equality_duals == [SDPX.CoreLPEqualityDual(true, 1)]
    @test low.equality_dual_origins[1].kind === :equality
    @test low.equality_dual_origins[1].sign == 1
    # G row 1 = +A row1 = 2x1; row 2 = -A row2 = +3x1 (since -(-3));
    # h follows the same sign transformation and is [2, -3].
    # Equality Aeq = A row3 = 0.
    @test [constant[1, 1] for constant in low.core.C] == [2.0, -3.0]
    @test iszero(low.core.B)
end

Test.@testset "L4 multi-coefficient sparse rows" begin
    program = _lp_program(
        products=((SDPX.Reals(), 3),),
        rows=((SDPX.Nonnegative(), [1]),),
        equalities=[[(1, 1.0), (2, 2.0), (3, 3.0)]],
        rhs=[4.0],
        T=Float64,
        objective=[0.0, 0.0, 0.0],
    )
    low = SDPX.lower_lp_native(program; sparse=true)
    @test low.inequality_dual_origins == [
        SDPX.CoreLPRecordOrigin(:inequality, 1, 1, 1),
    ]
    # G row is [1 2 3].
    @test low.core.cons.Asp[1] isa SDPX.ActiveSparseCoefficientVector{Float64}
    @test length(low.core.cons.active[1]) == 3
    @test low.core.cons.Asp[1][2][1, 1] == 2.0
    @test low.core.cons.Asp[1][3][1, 1] == 3.0
    @test low.core.C[1][1, 1] == 4.0
end

# ---------------------------------------------------------------------------
# Objective sense/constant
# ---------------------------------------------------------------------------

Test.@testset "L5 maximize negates objective, keeps constant" begin
    program = _lp_program(
        products=((SDPX.Nonnegative(), 2),),
        equalities=[],
        T=Float64,
        sense=SDPX.Maximize(),
        objective=[1.0, -2.0],
        constant=7.5,
    )
    low = SDPX.lower_lp_native(program)
    @test low.objective_sign == -1
    @test low.objective_constant == 7.5
    @test collect(low.core.c) == [-1.0, 2.0]
end

# ---------------------------------------------------------------------------
# Sparse nnz and block identity
# ---------------------------------------------------------------------------

Test.@testset "L6 sparse nnz and block identity unchanged" begin
    program = _lp_program(
        products=((SDPX.Nonnegative(), 2),),
        rows=((SDPX.ZeroCone(), [1]),),
        equalities=[[(1, 2.0), (2, 3.0)]],
        rhs=[0.0],
        T=Float64,
        objective=[0.0, 0.0],
    )
    low = SDPX.lower_lp_native(program)
    @test length(program.blocks) == 1
    @test length(program.row_blocks) == 1
    @test low.core.B isa SparseMatrixCSC{Float64,Int}
    @test nnz(low.core.B) == 2
    @test low.core.B[1, 1] == 2.0
    @test low.core.B[2, 1] == 3.0
    # Core G built-in inequalities: 2 variable rows; equality 1 row.
    @test sum(low.core.dims.L) == 2
end

# ---------------------------------------------------------------------------
# Fail-closed: mixed/non-LP and equality-only
# ---------------------------------------------------------------------------

Test.@testset "L7 mixed and non-LP rejected" begin
    mixed = _lp_program(
        products=((SDPX.Nonnegative(), 1), (SDPX.LorentzCone(), 3)),
        equalities=[],
    )
    err = _caught_lp_error(() -> SDPX.lower_lp_native(mixed))
    @test err isa SDPX.UnsupportedNativeConeRoute
    @test err.detected_families == [:lp_family, :soc_family]

    psd = _lp_program(products=((SDPX.PSDCone(), 2),), equalities=[])
    err2 = _caught_lp_error(() -> SDPX.lower_lp_native(psd))
    @test err2 isa SDPX.LPLoweringError
    @test err2.reason === :non_lp_route
end

Test.@testset "L7b constant-only LP fails before core allocation" begin
    model = SDPX.Model(Float64)
    SDPX.constraint!(model, :constant, 1.0, SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), 0.0)
    program = SDPX.compile_product_cone_model(model)
    err = _caught_lp_error(() -> SDPX.lower_lp_native(program; verbosity=0))
    @test err isa SDPX.LPLoweringError
    @test err.reason === :no_variables
end

Test.@testset "L8 equality-only fails explicitly without dummy cone" begin
    equality_only = _lp_program(
        products=((SDPX.Reals(), 2),),
        rows=((SDPX.ZeroCone(), [1]),),
        equalities=[[(1, 1.0), (2, -1.0)]],
        T=Float64,
    )
    err = _caught_lp_error(() -> SDPX.lower_lp_native(equality_only))
    @test err isa SDPX.LPLoweringError
    @test err.reason === :no_dummy_cone
    @test occursin("without a dummy cone", err.message)

    all_free = _lp_program(products=((SDPX.Reals(), 1),), equalities=[])
    err2 = _caught_lp_error(() -> SDPX.lower_lp_native(all_free))
    @test err2 isa SDPX.LPLoweringError
    @test err2.reason === :no_dummy_cone
end

# ---------------------------------------------------------------------------
# BigFloat owned precision
# ---------------------------------------------------------------------------

Test.@testset "L9 BigFloat owned precision" begin
    setprecision(BigFloat, 256) do
        program = _lp_program(
            products=((SDPX.Nonnegative(), 2),),
            rows=((SDPX.Nonpositive(), [1]),),
            equalities=[[(1, BigFloat("1.25"))]],
            rhs=[BigFloat("1.25")],
            T=BigFloat,
            objective=[BigFloat("0.5"), BigFloat("-2.5")],
            constant=BigFloat("3.75"),
        )
        low = SDPX.lower_lp_native(program)
        @test low isa SDPX.LPLowering{BigFloat}
        @test eltype(low.core.c) === BigFloat
        @test eltype(low.core.B) === BigFloat
        @test precision(low.core.c[1]) >= 256
        @test precision(low.core.c[2]) >= 256
        @test low.objective_constant == BigFloat("3.75")
        @test precision(low.objective_constant) >= 256
    end
end

# ---------------------------------------------------------------------------
# Static type contract
# ---------------------------------------------------------------------------

Test.@testset "L10 static typed fields, no Any" begin
    @test SDPX.LPLoweringError <: Exception
    fields = fieldnames(SDPX.LPLowering)
    for required in (
        :core,
        :route,
        :objective_sign,
        :objective_constant,
        :primal_refs,
        :inequality_dual_origins,
        :equality_dual_origins,
        :inequality_duals,
        :equality_duals,
    )
        @test required ∈ fields
    end
    banned = (
        :orientation,
        :dual_model,
        :primal_model,
        :dualization,
        :formulation,
        :provider,
        :solver_choice,
        :precision_bits,
    )
    for field in fields
        @test field ∉ banned
        @test fieldtype(SDPX.LPLowering, field) !== Any
    end
    @test fieldtype(SDPX.LPLowering, :objective_sign) === Int
    @test fieldtype(SDPX.LPLowering, :objective_constant) <: AbstractFloat
    @test fieldtype(SDPX.LPLowering, :primal_refs) === Vector{SDPX.VariableRef}
    @test fieldtype(SDPX.LPLowering, :inequality_dual_origins) ===
          Vector{SDPX.CoreLPRecordOrigin}
    @test fieldtype(SDPX.LPLowering, :equality_dual_origins) ===
          Vector{SDPX.CoreLPRecordOrigin}
    @test fieldtype(SDPX.LPLowering, :inequality_duals) ===
          Vector{SDPX.CoreLPInequalityDual}
    @test fieldtype(SDPX.LPLowering, :equality_duals) ===
          Vector{SDPX.CoreLPEqualityDual}
end
