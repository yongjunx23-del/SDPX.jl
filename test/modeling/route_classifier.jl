# Native cone family routing contract.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

if !isdefined(SDPX, :classify_native_cone_program)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "route.jl"))
end

using Test
using SparseArrays

function _route_program(;
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

function _route_error(f)
    try
        f()
        return nothing
    catch error_value
        return error_value
    end
end

@testset "native cone routes select one family" begin
    cases = (
        (
            :lp_family,
            ((SDPX.Nonnegative(), 2), (SDPX.Nonpositive(), 1), (SDPX.Reals(), 1)),
            ((SDPX.ZeroCone(), 1),),
        ),
        (
            :soc_family,
            ((SDPX.LorentzCone(), 3), (SDPX.RotatedLorentzCone(), 4)),
            ((SDPX.ZeroCone(), 1),),
        ),
        (
            :sdp_family,
            ((SDPX.PSDCone(), 3), (SDPX.Reals(), 1)),
            ((SDPX.PSDCone(), 2),),
        ),
        (:lp_family, (), ()),
        (:lp_family, ((SDPX.Reals(), 2),), ((SDPX.ZeroCone(), 1),)),
    )
    for (expected, products, rows) in cases
        route = SDPX.classify_native_cone_program(
            _route_program(products=products, rows=rows),
        )
        @test route isa SDPX.NativeConeRoute
        @test route.route === expected
        @test fieldnames(typeof(route)) == (:route,)
    end

    @test SDPX.NativeConeRoute(:lp_family) == SDPX.NativeConeRoute(:lp_family)
    @test SDPX.NativeConeRoute(:lp_family) != SDPX.NativeConeRoute(:sdp_family)
end

@testset "mixed native cone families return one clear error" begin
    cases = (
        (
            ((SDPX.Nonnegative(), 2), (SDPX.LorentzCone(), 3)),
            [:lp_family, :soc_family],
        ),
        (
            ((SDPX.Nonpositive(), 1), (SDPX.PSDCone(), 2)),
            [:lp_family, :sdp_family],
        ),
        (
            ((SDPX.PSDCone(), 2), (SDPX.LorentzCone(), 3)),
            [:soc_family, :sdp_family],
        ),
        (
            (
                (SDPX.Nonnegative(), 1),
                (SDPX.LorentzCone(), 3),
                (SDPX.PSDCone(), 2),
            ),
            [:lp_family, :soc_family, :sdp_family],
        ),
    )

    for (products, expected) in cases
        error_value = _route_error(() -> SDPX.classify_native_cone_program(
            _route_program(products=products),
        ))
        @test error_value isa SDPX.UnsupportedNativeConeRoute
        @test error_value.detected_families == expected
        @test sprint(showerror, error_value) ==
              "model combines unsupported cone families $expected"
    end

    cross = _route_program(
        products=((SDPX.Nonnegative(), 1),),
        rows=((SDPX.LorentzCone(), 3), (SDPX.PSDCone(), 2)),
    )
    cross_error = _route_error(
        () -> SDPX.classify_native_cone_program(cross),
    )
    @test cross_error.detected_families ==
          [:lp_family, :soc_family, :sdp_family]

    for products in (
        ((SDPX.Nonnegative(), 1), (SDPX.LorentzCone(), 3)),
        ((SDPX.LorentzCone(), 3), (SDPX.Nonnegative(), 1)),
    )
        error_value = _route_error(() -> SDPX.classify_native_cone_program(
            _route_program(products=products),
        ))
        @test error_value.detected_families == [:lp_family, :soc_family]
    end
end

@testset "route classification is arithmetic independent" begin
    specs = (
        (
            products=((SDPX.Nonnegative(), 2), (SDPX.Reals(), 1)),
            rows=((SDPX.ZeroCone(), 2),),
        ),
        (
            products=((SDPX.LorentzCone(), 3),),
            rows=((SDPX.RotatedLorentzCone(), 4),),
        ),
        (products=((SDPX.PSDCone(), 3),), rows=((SDPX.ZeroCone(), 1),)),
        (products=(), rows=()),
    )
    for spec in specs
        float_route = SDPX.classify_native_cone_program(
            _route_program(; spec..., T=Float64),
        )
        big_route = SDPX.classify_native_cone_program(
            _route_program(; spec..., T=BigFloat),
        )
        @test big_route == float_route
    end
end

@testset "route classification revalidates mutable program structure" begin
    program = _route_program(
        products=((SDPX.Nonnegative(), 2),),
        rows=((SDPX.ZeroCone(), 1),),
    )
    @test SDPX.classify_native_cone_program(program).route === :lp_family

    bad_blocks = _route_program(products=((SDPX.Nonnegative(), 2),))
    offset = 1 + sum(SDPX.block_length, bad_blocks.blocks)
    push!(bad_blocks.blocks, SDPX.NativeBlock(SDPX.ZeroCone(), 1, offset))
    block_error = _route_error(
        () -> SDPX.classify_native_cone_program(bad_blocks),
    )
    @test block_error isa ArgumentError
    @test occursin("block variables total", sprint(showerror, block_error))

    bad_rows = _route_program(rows=((SDPX.ZeroCone(), 1),))
    offset = 1 + sum(SDPX.row_block_length, bad_rows.row_blocks)
    push!(bad_rows.row_blocks, SDPX.RowBlock(SDPX.ZeroCone(), offset, 2))
    row_error = _route_error(
        () -> SDPX.classify_native_cone_program(bad_rows),
    )
    @test row_error isa ArgumentError
    @test occursin("row block rows total", sprint(showerror, row_error))
end

@testset "route metadata stays minimal and typed" begin
    @test isconcretetype(SDPX.NativeConeRoute)
    @test isconcretetype(SDPX.UnsupportedNativeConeRoute)
    @test fieldnames(SDPX.NativeConeRoute) == (:route,)
    @test fieldtype(SDPX.NativeConeRoute, :route) === Symbol
    @test fieldnames(SDPX.UnsupportedNativeConeRoute) == (:detected_families,)
    @test fieldtype(
        SDPX.UnsupportedNativeConeRoute,
        :detected_families,
    ) === Vector{Symbol}
end
