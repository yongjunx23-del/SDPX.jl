# Canonical slack-cone product layout contract (Subagent C).
#
# The layout describes the block partition of the canonical SLACK `s`
# (and dual `y`) — the rows of `A` / entries of `b` — never the original
# product-variable blocks. Blocks are native canonical cones.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using SparseArrays

# Build a canonical slack layout from an explicit list of
# `(cone, dimension)` blocks (native canonical cones only).
function _layout(descs::Tuple; T::Type{<:AbstractFloat}=Float64)
    descriptors = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for (cone, dimension) in descs
        push!(descriptors, SDPX.ConeBlockDescriptor(
            T, cone, dimension;
            offset=offset,
            parameter=cone === :power ? T(0.5) : zero(T),
            reconstruction=SDPX.CanonicalBlockMap(:constraint, 1, 1, 1),
        ))
        offset += SDPX.block_length(descriptors[end])
    end
    return SDPX.canonical_layout(descriptors)
end

@testset "canonical slack layout builds for heterogeneous mixes" begin
    mixes = (
        ((:nonnegative, 2), (:soc, 3)),
        ((:soc, 3), (:psd, 2)),
        ((:nonnegative, 1), (:exp, 3)),
        ((:soc, 3), (:power, 3)),
        ((:psd, 2), (:exp, 3)),
        ((:nonnegative, 1), (:soc, 3), (:psd, 2)),
        ((:zero, 2), (:nonnegative, 1)),   # equality + orthant mix
    )
    for desc in mixes
        layout = _layout(desc)
        @test layout isa SDPX.ConeProductLayout
        @test SDPX.layout_num_blocks(layout) == length(desc)
        @test SDPX.layout_dimension(layout) == sum(
            SDPX.block_length, SDPX.layout_blocks(layout))
        layout2 = _layout(desc)
        @test SDPX.layout_blocks(layout) == SDPX.layout_blocks(layout2)
    end
end

@testset "canonical layout coordinate mapping is exact" begin
    desc = ((:nonnegative, 2), (:soc, 3), (:psd, 2))
    layout = _layout(desc)
    n = SDPX.layout_dimension(layout)
    for global_index in 1:n
        (block_index, within) = SDPX.global_to_block(layout, global_index)
        @test SDPX.block_to_global(layout, block_index, within) == global_index
    end
    @test SDPX.global_to_block(layout, 1) == (1, 1)
    @test SDPX.global_to_block(layout, 2) == (1, 2)
    @test SDPX.global_to_block(layout, 3) == (2, 1)
    @test SDPX.global_to_block(layout, 5) == (2, 3)
    @test SDPX.global_to_block(layout, 6) == (3, 1)
    @test SDPX.global_to_block(layout, 8) == (3, 3)
end

@testset "canonical block descriptors carry cone and reconstruction metadata" begin
    layout = _layout(((:nonnegative, 2), (:psd, 3), (:power, 3)))
    blocks = SDPX.layout_blocks(layout)
    @test SDPX.block_cone(blocks[1]) === :nonnegative
    @test SDPX.block_storage(blocks[1]) === :vector
    @test SDPX.block_cone(blocks[2]) === :psd
    @test SDPX.block_storage(blocks[2]) === :packed_lower
    @test SDPX.block_dimension(blocks[2]) == 3
    @test SDPX.block_length(blocks[2]) == 6
    @test SDPX.block_cone(blocks[3]) === :power
    @test SDPX.block_parameter(blocks[3]) == 0.5
    @test SDPX.block_reconstruction(blocks[3]).source === :constraint
end

@testset "canonical layout barrier degree is the sum of block degrees" begin
    @test SDPX.barrier_degree(:nonnegative, 3) == 3
    @test SDPX.barrier_degree(:soc, 3) == 2
    @test SDPX.barrier_degree(:psd, 3) == 3
    @test SDPX.barrier_degree(:exp, 3) == 3
    @test SDPX.barrier_degree(:power, 3) == 2
    @test SDPX.barrier_degree(:zero, 3) == 0
    @test SDPX.barrier_degree(:free, 3) == 0
    layout = _layout(((:nonnegative, 2), (:soc, 3), (:psd, 2)))
    @test SDPX.layout_barrier_degree(layout) == 2 + 2 + 2
    @test SDPX.layout_barrier_degree(_layout(((:zero, 2),))) == 0
end

@testset "canonical layout preserves the power-cone parameter precision" begin
    f64 = _layout(((:power, 3),))
    @test SDPX.block_parameter(SDPX.layout_blocks(f64)[1]) === Float64(0.5)
    big = SDPX.canonical_layout(
        [SDPX.ConeBlockDescriptor(
            BigFloat, :power, 3; offset=1,
            parameter=BigFloat(0.5; precision=256),
            reconstruction=SDPX.CanonicalBlockMap(:constraint, 1, 1, 1),
        )],
    )
    @test SDPX.block_parameter(SDPX.layout_blocks(big)[1]) isa BigFloat
end

@testset "canonical layout rejects non-native cones" begin
    for cone in (:nonpositive, :rsoc, :interval)
        @test_throws ArgumentError SDPX.ConeBlockDescriptor(
            Float64, cone, 3; offset=1,
        )
    end
end
