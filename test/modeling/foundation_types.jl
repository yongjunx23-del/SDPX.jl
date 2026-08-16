#=====================================================================#
#    F0 foundation contract tests (v0.5)
#
#    Runs either:
#      1. after `using SDPX` once main integrates the new files
#         (include order: modeling/domains.jl, modeling/refs.jl,
#          modeling/types.jl, ir/types.jl); or
#      2. standalone, when the runner defines `Main.SDPX` as a module
#         built from those four files in that exact order (the
#         `isdefined(@__MODULE__, :SDPX)` guard below).
#
#    Covered: domain identity, ref model isolation, arithmetic /
#    precision ownership, invalid precision, native block counts /
#    dimensions / storage, PSD one-block evidence, sparse equality
#    type, and static absence of orientation / dual_model /
#    dualization fields. MultiFloat coverage is skipped honestly when
#    the optional package is not loaded.
#=====================================================================#

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays

const MULTIFLOAT_AVAILABLE = try
    @eval using MultiFloats
    true
catch
    false
end

# ---------------------------------------------------------------------------
# Local fixture builders (foundation-only; no DSL, no solver). Defined before
# the testsets below: `@testset` bodies run immediately at their include
# location, so fixtures must precede their first use.
# ---------------------------------------------------------------------------

"""
    _f0_program(; psd_dimension=2, include_soc=true, source_model=UInt64(1))

Minimal valid `NativeConeProgram{Float64}`. With the default
`include_soc=true` it has one PSD block and one SOC block (two product
blocks, one zero-cone row block); with `include_soc=false` it is a
single PSD block (the "PSD remains ONE block" evidence). Used only to
exercise the type contract.
"""
function _f0_program(; psd_dimension::Int=2, include_soc::Bool=true, source_model::UInt64=UInt64(1))
    blocks = SDPX.NativeBlock[
        SDPX.NativeBlock(SDPX.PSDCone(), psd_dimension, 1),
    ]
    if include_soc
        push!(blocks, SDPX.NativeBlock(
            SDPX.LorentzCone(), 3,
            SDPX.variable_length(SDPX.PSDCone(), psd_dimension) + 1,
        ))
    end
    num_variables = sum(SDPX.block_length, blocks)
    num_rows = 2
    equality_matrix = spzeros(Float64, num_rows, num_variables)
    rhs = zeros(Float64, num_rows)
    row_blocks = SDPX.RowBlock[SDPX.RowBlock(SDPX.ZeroCone(), 1, num_rows)]
    primal = [SDPX.VariableRef(source_model, 1, i) for i in 1:num_variables]
    constraint_dual = [SDPX.ConstraintRef(source_model, 1, i) for i in 1:num_rows]
    dual_slack = copy(primal)
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(Float64),
        SDPX.Minimize(),
        zeros(Float64, num_variables),
        0.0,
        equality_matrix,
        rhs,
        blocks,
        row_blocks,
        primal,
        constraint_dual,
        dual_slack,
        source_model,
    )
end

"""
    _f0_bad_program()

Matrix/block size mismatch that the validated constructor must reject.
"""
function _f0_bad_program()
    blocks = SDPX.NativeBlock[SDPX.NativeBlock(SDPX.PSDCone(), 2, 1)]
    equality_matrix = spzeros(Float64, 1, 1)  # wrong: one row, one column
    row_blocks = SDPX.RowBlock[SDPX.RowBlock(SDPX.ZeroCone(), 1, 1)]
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(Float64),
        SDPX.Minimize(),
        zeros(Float64, 3),
        0.0,
        equality_matrix,
        zeros(Float64, 1),
        blocks,
        row_blocks,
        [SDPX.VariableRef(UInt64(1), 1, i) for i in 1:3],
        [SDPX.ConstraintRef(UInt64(1), 1, 1)],
        [SDPX.VariableRef(UInt64(1), 1, i) for i in 1:3],
        UInt64(1),
    )
end

Test.@testset "F0 domain identity" begin
    @test SDPX.Reals() == SDPX.Reals()
    @test SDPX.Reals() != SDPX.Nonnegative()
    @test SDPX.Nonnegative() != SDPX.Nonpositive()
    @test SDPX.Nonpositive() != SDPX.ZeroCone()
    @test SDPX.ZeroCone() != SDPX.LorentzCone()
    @test SDPX.LorentzCone() != SDPX.RotatedLorentzCone()
    @test SDPX.RotatedLorentzCone() != SDPX.PSDCone()
    @test isequal(SDPX.Reals(), SDPX.Reals())
    @test isequal(SDPX.PSDCone(), SDPX.PSDCone())

    # PSD dimension belongs to the block shape, never to PSDCone.
    @test fieldcount(SDPX.PSDCone) == 0
    @test fieldcount(SDPX.LorentzCone) == 0

    # Objective sense markers.
    @test SDPX.Minimize() == SDPX.Minimize()
    @test SDPX.Maximize() == SDPX.Maximize()
    @test SDPX.Minimize() != SDPX.Maximize()

    # Domain predicates.
    @test SDPX.is_product_cone(SDPX.Reals())
    @test SDPX.is_product_cone(SDPX.PSDCone())
    @test SDPX.is_affine_cone(SDPX.PSDCone())
    @test SDPX.is_affine_cone(SDPX.ZeroCone())
    @test SDPX.is_affine_cone(SDPX.RotatedLorentzCone())

    # Shape helpers: affine dimension vs stored variable length.
    @test SDPX.affine_dimension(SDPX.PSDCone(), 3) == 6
    @test SDPX.variable_length(SDPX.PSDCone(), 3) == 6
    @test SDPX.variable_length(SDPX.LorentzCone(), 5) == 5
    @test SDPX.variable_length(SDPX.Nonnegative(), 4) == 4
end

Test.@testset "F0 affine PSD remains ONE row block" begin
    block = SDPX.RowBlock(SDPX.PSDCone(), 1, 3)
    @test SDPX.row_block_domain(block) == SDPX.PSDCone()
    @test SDPX.row_block_shape(block) == 3
    @test SDPX.row_block_length(block) == 6
    @test length(SDPX.row_block_rows(block)) == 6
    @test SDPX.row_block_psd_storage(block).matrix_dimension == 3
    @test_throws ArgumentError SDPX.RowBlock(SDPX.PSDCone(), 1, 3, collect(1:3))
end

Test.@testset "F0 reference identity and model isolation" begin
    @test isbitstype(SDPX.VariableRef)
    @test isbitstype(SDPX.ConstraintRef)

    model_a = SDPX.Model(Float64)
    model_b = SDPX.Model(Float64)
    id_a = SDPX.model_identity(model_a)
    id_b = SDPX.model_identity(model_b)
    @test id_a isa UInt64
    @test id_b isa UInt64
    @test id_a != id_b   # two live models are distinct

    # Same model, same block/index: interchangeable.
    ref_a = SDPX.VariableRef(id_a, 1, 1)
    @test ref_a == SDPX.VariableRef(id_a, 1, 1)
    @test isequal(ref_a, SDPX.VariableRef(id_a, 1, 1))
    @test hash(ref_a) == hash(SDPX.VariableRef(id_a, 1, 1))

    # Cross-model refs never compare equal even with identical block/index.
    ref_b = SDPX.VariableRef(id_b, 1, 1)
    @test !isequal(ref_a, ref_b)
    @test ref_a != ref_b

    crefa = SDPX.ConstraintRef(id_a, 2, 3)
    crefb = SDPX.ConstraintRef(id_b, 2, 3)
    @test crefa == SDPX.ConstraintRef(id_a, 2, 3)
    @test !isequal(crefa, crefb)
    @test hash(crefa) == hash(SDPX.ConstraintRef(id_a, 2, 3))

    # References carry identity fields only — no pointer to a runtime model.
    @test fieldtypes(SDPX.VariableRef) == (UInt64, Int, Int)
    @test fieldtypes(SDPX.ConstraintRef) == (UInt64, Int, Int)

    # Model identity is independent of object addresses, so stale references
    # cannot become valid if Julia later reuses storage for another model.
    stale_model_identity = id_a
    model_a = nothing
    GC.gc()
    model_c = SDPX.Model(Float64)
    @test SDPX.model_identity(model_c) != stale_model_identity
end

Test.@testset "F0 arithmetic and precision ownership" begin
    model_f64 = SDPX.Model(Float64)
    @test eltype(model_f64) === Float64
    @test SDPX.precision_bits(model_f64) == 53
    @test model_f64.arithmetic isa SDPX.ArithmeticSpec{Float64}
    @test !model_f64.arithmetic.supports_multifloat
    @test SDPX.num_variables(model_f64) == 0
    @test SDPX.num_constraints(model_f64) == 0

    model_big_256 = SDPX.Model(BigFloat)
    @test eltype(model_big_256) === BigFloat
    @test SDPX.precision_bits(model_big_256) == 256
    @test model_big_256.arithmetic.precision_bits == 256

    model_big_512 = SDPX.Model(BigFloat; precision_bits=512)
    @test eltype(model_big_512) === BigFloat
    @test SDPX.precision_bits(model_big_512) == 512

    for bad in (128, 300, 64, 1024, -1, 0)
        @test_throws ArgumentError SDPX.Model(BigFloat; precision_bits=bad)
        @test_throws ArgumentError SDPX.ArithmeticSpec(BigFloat; precision_bits=bad)
    end

    @test_throws ArgumentError SDPX.Model(Float32)
    @test_throws ArgumentError SDPX.ArithmeticSpec(Float32)
    @test_throws ArgumentError SDPX.ArithmeticSpec(Float16)
end

Test.@testset "F0 optional MultiFloat arithmetic" begin
    if MULTIFLOAT_AVAILABLE
        model_x2 = SDPX.Model(MultiFloats.Float64x2)
        @test eltype(model_x2) === MultiFloats.Float64x2
        @test SDPX.precision_bits(model_x2) == 104
        @test model_x2.arithmetic.supports_multifloat

        model_x4 = SDPX.Model(MultiFloats.Float64x4)
        @test eltype(model_x4) === MultiFloats.Float64x4
        @test SDPX.precision_bits(model_x4) == 208
        @test model_x4.arithmetic.supports_multifloat
    else
        @test_skip "MultiFloats not loaded; Float64x2/Float64x4 model construction skipped"
        @test_skip "MultiFloats not loaded; Float64x2/Float64x4 model construction skipped"
    end
end

Test.@testset "F0 native block counts, dimensions, storage" begin
    free_block = SDPX.NativeBlock(SDPX.Reals(), 2, 1)
    @test SDPX.block_cone(free_block) === :free
    @test SDPX.block_shape(free_block) == 2
    @test SDPX.block_length(free_block) == 2
    @test SDPX.block_offset(free_block) == 1
    @test SDPX.block_psd_storage(free_block) === nothing

    @test SDPX.block_cone(SDPX.NativeBlock(SDPX.Nonnegative(), 3, 3)) === :nonnegative
    @test SDPX.block_cone(SDPX.NativeBlock(SDPX.Nonpositive(), 1, 6)) === :nonpositive
    @test SDPX.block_cone(SDPX.NativeBlock(SDPX.ZeroCone(), 2, 7)) === :zero
    @test SDPX.block_cone(SDPX.NativeBlock(SDPX.LorentzCone(), 3, 9)) === :soc
    @test SDPX.block_cone(SDPX.NativeBlock(SDPX.RotatedLorentzCone(), 4, 12)) === :rsoc

    psd_block = SDPX.NativeBlock(SDPX.PSDCone(), 2, 1)
    @test SDPX.block_cone(psd_block) === :psd
    @test SDPX.block_shape(psd_block) == 2
    @test SDPX.block_length(psd_block) == 3
    @test SDPX.is_psd_block(psd_block)
    storage = SDPX.block_psd_storage(psd_block)
    @test storage !== nothing
    @test storage.side === :lower
    @test storage.order === :column_major
    @test storage.storage === :packed
    @test storage.matrix_dimension == 2
    @test storage.packed_length == 3
    @test SDPX.packed_length(storage) == 3

    # Invalid block descriptors.
    @test_throws ArgumentError SDPX.NativeBlock(:psd, SDPX.LorentzCone(), 3, 1)
    @test_throws ArgumentError SDPX.NativeBlock(:soc, SDPX.PSDCone(), 3, 1)
    @test_throws ArgumentError SDPX.NativeBlock(SDPX.Reals(), 0, 1)
    @test_throws ArgumentError SDPX.NativeBlock(SDPX.Reals(), 2, 0)
    @test_throws ArgumentError SDPX.NativeBlock(:bogus, SDPX.Reals(), 2, 1)
end

Test.@testset "F0 PSD remains ONE block" begin
    # A 3x3 PSD block stores 6 packed entries but is exactly one block.
    psd_3 = SDPX.NativeBlock(SDPX.PSDCone(), 3, 1)
    @test SDPX.block_length(psd_3) == 6
    @test SDPX.block_psd_storage(psd_3).packed_length == 6

    program = _f0_program(; psd_dimension=3, include_soc=false)
    @test SDPX.program_num_blocks(program) == 1
    @test length(SDPX.program_blocks(program)) == 1
    @test SDPX.program_num_variables(program) == 6
    @test SDPX.block_length(SDPX.program_blocks(program)[1]) == 6
    @test SDPX.is_psd_block(SDPX.program_blocks(program)[1])
end

Test.@testset "F0 NativeConeProgram contract" begin
    model = SDPX.Model(Float64)
    source_id = SDPX.model_identity(model)
    program = _f0_program(; source_model=source_id)

    @test SDPX.program_arithmetic(program) isa SDPX.ArithmeticSpec{Float64}
    @test SDPX.program_precision_bits(program) == 53
    @test SDPX.program_sense(program) == SDPX.Minimize()
    @test length(SDPX.program_objective_vector(program)) == SDPX.program_num_variables(program)
    @test SDPX.program_objective_constant(program) === 0.0
    @test SDPX.program_equality_matrix(program) isa SparseMatrixCSC{Float64,Int}
    @test size(SDPX.program_equality_matrix(program)) ==
          (SDPX.program_num_rows(program), SDPX.program_num_variables(program))
    @test length(SDPX.program_rhs(program)) == SDPX.program_num_rows(program)
    @test SDPX.program_source_model(program) == source_id

    # Reconstruction maps are present, typed, and sized.
    @test length(program.primal_reconstruction) == SDPX.program_num_variables(program)
    @test length(program.constraint_dual_reconstruction) == SDPX.program_num_rows(program)
    @test length(program.variable_dual_slack_reconstruction) == SDPX.program_num_variables(program)
    @test eltype(program.primal_reconstruction) === SDPX.VariableRef
    @test eltype(program.constraint_dual_reconstruction) === SDPX.ConstraintRef
    @test eltype(program.variable_dual_slack_reconstruction) === SDPX.VariableRef

    # Block/row ordering and offsets are enforced.
    @test SDPX.variable_block(program, 1) == 1
    @test SDPX.variable_slot(program, 1) == 1
    @test SDPX.variable_block(program, 4) == 2
    @test SDPX.variable_slot(program, 4) == 1
    @test_throws ArgumentError SDPX.variable_block(program, 0)
    @test_throws ArgumentError SDPX.variable_block(program, SDPX.program_num_variables(program) + 1)

    # Static absence of orientation / dualization metadata.
    program_fields = fieldnames(SDPX.NativeConeProgram)
    for banned in (:orientation, :dual_model, :primal_model, :dualization,
                   :dualization_metadata, :formulation, :solver_choice)
        @test banned ∉ program_fields
    end
    for required in (:primal_reconstruction, :constraint_dual_reconstruction,
                     :variable_dual_slack_reconstruction, :source_model)
        @test required ∈ program_fields
    end
    for field in program_fields
        @test !(fieldtype(SDPX.NativeConeProgram, field) <: SDPX.Model)
    end

    block_fields = fieldnames(SDPX.NativeBlock)
    @test :orientation ∉ block_fields
    @test :dualization ∉ block_fields

    # Validated construction rejects inconsistent sizes.
    @test_throws ArgumentError _f0_bad_program()
end
