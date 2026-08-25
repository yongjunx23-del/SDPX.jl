# Heterogeneous cone product layout contract (Subagent A).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

if !isdefined(SDPX, :cone_product_layout)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "layout.jl"))
end

using Test
using SparseArrays

function _layout_program(;
    products::Tuple=(),
    rows::Tuple=(),
    T::Type{<:AbstractFloat}=Float64,
    source_model::UInt64=UInt64(7),
)
    blocks = SDPX.NativeBlock[]
    offset = 1
    for (domain, shape) in products
        push!(blocks, SDPX.NativeBlock(domain, shape, offset))
        offset += SDPX.block_length(blocks[end])
    end

    row_blocks = SDPX.RowBlock[]
    row_offset = 1
    for (domain, shape) in rows
        push!(row_blocks, SDPX.RowBlock(domain, row_offset, shape))
        row_offset += SDPX.row_block_length(row_blocks[end])
    end

    variables = offset - 1
    num_rows = row_offset - 1
    primal = [SDPX.VariableRef(source_model, 1, i) for i in 1:variables]
    constraint_dual = [
        SDPX.ConstraintRef(source_model, 1, i) for i in 1:num_rows
    ]
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(T),
        SDPX.Minimize(),
        zeros(T, variables),
        zero(T),
        spzeros(T, num_rows, variables),
        zeros(T, num_rows),
        blocks,
        row_blocks,
        primal,
        constraint_dual,
        copy(primal),
        source_model,
    )
end

@testset "cone product layout builds for heterogeneous mixes" begin
    mixes = (
        ((SDPX.Nonnegative(), 2), (SDPX.LorentzCone(), 3)),
        ((SDPX.LorentzCone(), 3), (SDPX.PSDCone(), 2)),
        ((SDPX.Nonnegative(), 1), (SDPX.ExponentialCone(), 3)),
        ((SDPX.LorentzCone(), 3), (SDPX.PowerCone(0.5), 3)),
        ((SDPX.PSDCone(), 2), (SDPX.ExponentialCone(), 3)),
        ((SDPX.Nonnegative(), 1), (SDPX.LorentzCone(), 3), (SDPX.PSDCone(), 2)),
    )
    for products in mixes
        program = _layout_program(products=products)
        layout = SDPX.cone_product_layout(program)
        @test layout isa SDPX.ConeProductLayout
        @test SDPX.layout_num_blocks(layout) == length(products)
        @test SDPX.layout_dimension(layout) == SDPX.program_num_variables(program)
        # block order reproducible
        layout2 = SDPX.cone_product_layout(program)
        @test SDPX.layout_blocks(layout) == SDPX.layout_blocks(layout2)
    end
end

@testset "cone product layout coordinate mapping is exact" begin
    products = (
        (SDPX.Nonnegative(), 2),
        (SDPX.LorentzCone(), 3),
        (SDPX.PSDCone(), 2),
    )
    program = _layout_program(products=products)
    layout = SDPX.cone_product_layout(program)
    n = SDPX.layout_dimension(layout)
    for global_index in 1:n
        (block_index, within) = SDPX.global_to_block(layout, global_index)
        @test SDPX.block_to_global(layout, block_index, within) == global_index
    end
    # block 1: nonnegative dim 2 -> offsets 1,2
    @test SDPX.global_to_block(layout, 1) == (1, 1)
    @test SDPX.global_to_block(layout, 2) == (1, 2)
    # block 2: soc dim 3 -> offsets 3,4,5
    @test SDPX.global_to_block(layout, 3) == (2, 1)
    @test SDPX.global_to_block(layout, 5) == (2, 3)
    # block 3: psd 2x2 -> packed length 3 -> offsets 6,7,8
    @test SDPX.global_to_block(layout, 6) == (3, 1)
    @test SDPX.global_to_block(layout, 8) == (3, 3)
end

@testset "cone product layout block descriptors carry cone metadata" begin
    products = (
        (SDPX.Nonnegative(), 2),
        (SDPX.PSDCone(), 3),
        (SDPX.PowerCone(0.25), 3),
    )
    program = _layout_program(products=products)
    layout = SDPX.cone_product_layout(program)
    blocks = SDPX.layout_blocks(layout)
    @test SDPX.block_cone(blocks[1]) === :nonnegative
    @test SDPX.block_storage(blocks[1]) === :vector
    @test SDPX.block_cone(blocks[2]) === :psd
    @test SDPX.block_storage(blocks[2]) === :packed_lower
    @test SDPX.block_dimension(blocks[2]) == 3
    @test SDPX.block_length(blocks[2]) == 6
    @test SDPX.block_cone(blocks[3]) === :power
    @test SDPX.block_parameter(blocks[3]) == 0.25
    @test SDPX.block_dual_orientation(blocks[3]) === :primal
    @test SDPX.block_reconstruction(blocks[3]) == UInt64(3)
end

@testset "cone product layout barrier degree" begin
    @test SDPX.barrier_degree(:nonnegative, 3) == 3
    @test SDPX.barrier_degree(:soc, 3) == 2
    @test SDPX.barrier_degree(:psd, 3) == 3
    @test SDPX.barrier_degree(:exp, 3) == 3
    @test SDPX.barrier_degree(:power, 3) == 2
    @test SDPX.barrier_degree(:free, 3) == 0
    @test SDPX.barrier_degree(:zero, 3) == 0
    program = _layout_program(products=(
        (SDPX.Nonnegative(), 2),
        (SDPX.LorentzCone(), 3),
        (SDPX.PSDCone(), 2),
    ))
    layout = SDPX.cone_product_layout(program)
    @test SDPX.layout_barrier_degree(layout) == 2 + 2 + 2
end

@testset "cone product layout is arithmetic independent" begin
    products = ((SDPX.Nonnegative(), 1), (SDPX.LorentzCone(), 3), (SDPX.PSDCone(), 2))
    f64 = SDPX.cone_product_layout(_layout_program(products=products, T=Float64))
    bf = SDPX.cone_product_layout(_layout_program(products=products, T=BigFloat))
    @test SDPX.layout_blocks(f64) == SDPX.layout_blocks(bf)
    @test SDPX.layout_dimension(f64) == SDPX.layout_dimension(bf)
    @test SDPX.layout_barrier_degree(f64) == SDPX.layout_barrier_degree(bf)
end
