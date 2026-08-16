#=====================================================================#
#    R1 route classifier contract tests (v0.5)
#
#    Runs either:
#      1. after `using SDPX` once the main module wires
#         src/ir/route.jl; or
#      2. standalone after `using SDPX` while the central include is
#         not yet wired: the guard below bootstraps route.jl directly
#         into the loaded `SDPX` module with `Base.include`.
#
#    Covered: pure LP/SOC/SDP family classification, affine PSD
#    counting as PSD, empty / all-free / equality-only programs, block
#    permutations (including mixed-route permutations), exact
#    `UnsupportedNativeConeRoute` fields and message, BigFloat
#    classification invariance, structural-count validation, order
#    metadata preservation, and static banned-field / no-`Any` checks
#    on both new types.
#=====================================================================#

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

if !isdefined(SDPX, :classify_native_cone_program)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "route.jl"))
end

using Test
using SparseArrays

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

"""
    _route_program(; products, rows, T, source_model)

Build a valid `NativeConeProgram{T}` from ordered `(domain, shape)`
pairs. `products` and `rows` default to empty; one offset-contiguous
`NativeBlock` / `RowBlock` is appended per pair in order.
"""
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

    num_variables = offset - 1
    num_rows = row_offset - 1
    primal = [SDPX.VariableRef(source_model, 1, i) for i in 1:num_variables]
    constraint_dual = [SDPX.ConstraintRef(source_model, 1, i) for i in 1:num_rows]
    dual_slack = copy(primal)
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(T),
        SDPX.Minimize(),
        zeros(T, num_variables),
        zero(T),
        spzeros(T, num_rows, num_variables),
        zeros(T, num_rows),
        blocks,
        row_blocks,
        primal,
        constraint_dual,
        dual_slack,
        source_model,
    )
end

function _caught_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

# ---------------------------------------------------------------------------
# Pure LP family
# ---------------------------------------------------------------------------

Test.@testset "R1 pure LP family" begin
    # Nonnegative product + zero row.
    route = SDPX.classify_native_cone_program(
        _route_program(products=((SDPX.Nonnegative(), 2),), rows=((SDPX.ZeroCone(), 2),)),
    )
    @test route isa SDPX.NativeConeRoute
    @test route.route === :lp_family
    @test route.product_families == [:lp_family]
    @test route.row_families == [:zero]
    @test route.num_product_blocks == 1
    @test route.num_row_blocks == 1
    @test route.product_dimensions == [2]
    @test route.row_dimensions == [2]
    @test route.variables == 2
    @test route.rows == 2

    # Both orthants + zero + free remain LP.
    route2 = SDPX.classify_native_cone_program(
        _route_program(
            products=((SDPX.Nonpositive(), 1), (SDPX.Reals(), 3)),
            rows=((SDPX.Nonnegative(), 2), (SDPX.ZeroCone(), 1)),
        ),
    )
    @test route2.route === :lp_family
    @test route2.product_families == [:lp_family, :free]
    @test route2.row_families == [:lp_family, :zero]
    @test route2.variables == 4
    @test route2.rows == 3

    # Nonnegative + Nonpositive together are still one LP family.
    route3 = SDPX.classify_native_cone_program(
        _route_program(products=((SDPX.Nonnegative(), 1), (SDPX.Nonpositive(), 1))),
    )
    @test route3.route === :lp_family
    @test route3.product_families == [:lp_family, :lp_family]

    # Orthant row block alone classifies LP.
    route4 = SDPX.classify_native_cone_program(
        _route_program(rows=((SDPX.Nonpositive(), 2),)),
    )
    @test route4.route === :lp_family
    @test route4.row_families == [:lp_family]

    # Empty program classifies LP.
    empty_route = SDPX.classify_native_cone_program(_route_program())
    @test empty_route.route === :lp_family
    @test empty_route.num_product_blocks == 0
    @test empty_route.num_row_blocks == 0
    @test empty_route.variables == 0
    @test empty_route.rows == 0

    # All-free and equality-only programs classify LP.
    free_route = SDPX.classify_native_cone_program(
        _route_program(products=((SDPX.Reals(), 2), (SDPX.Reals(), 1))),
    )
    @test free_route.route === :lp_family
    @test free_route.product_families == [:free, :free]

    equality_route = SDPX.classify_native_cone_program(
        _route_program(
            products=((SDPX.ZeroCone(), 1),),
            rows=((SDPX.ZeroCone(), 1), (SDPX.Reals(), 2)),
        ),
    )
    @test equality_route.route === :lp_family
    @test equality_route.product_families == [:zero]
    @test equality_route.row_families == [:zero, :free]
end

# ---------------------------------------------------------------------------
# Pure SOC family
# ---------------------------------------------------------------------------

Test.@testset "R2 pure SOC family" begin
    # Lorentz product + zero row.
    route = SDPX.classify_native_cone_program(
        _route_program(products=((SDPX.LorentzCone(), 3),), rows=((SDPX.ZeroCone(), 1),)),
    )
    @test route.route === :soc_family
    @test route.product_families == [:soc_family]
    @test route.row_families == [:zero]
    @test route.product_dimensions == [3]
    @test route.variables == 3
    @test route.rows == 1

    # RSOC only.
    rsoc_route = SDPX.classify_native_cone_program(
        _route_program(products=((SDPX.RotatedLorentzCone(), 4),)),
    )
    @test rsoc_route.route === :soc_family
    @test rsoc_route.product_families == [:soc_family]

    # SOC + RSOC together remain one family; Reals and Zero are neutral.
    mixed_soc = SDPX.classify_native_cone_program(
        _route_program(
            products=((SDPX.LorentzCone(), 3), (SDPX.RotatedLorentzCone(), 4), (SDPX.Reals(), 2)),
            rows=((SDPX.ZeroCone(), 2),),
        ),
    )
    @test mixed_soc.route === :soc_family
    @test mixed_soc.product_families == [:soc_family, :soc_family, :free]
    @test mixed_soc.row_families == [:zero]
    @test mixed_soc.variables == 9
    @test mixed_soc.rows == 2

    # SOC as an affine row block.
    soc_row = SDPX.classify_native_cone_program(
        _route_program(products=((SDPX.ZeroCone(), 1),), rows=((SDPX.LorentzCone(), 3),)),
    )
    @test soc_row.route === :soc_family
    @test soc_row.row_families == [:soc_family]
    @test soc_row.row_dimensions == [3]
    @test soc_row.rows == 3
end

# ---------------------------------------------------------------------------
# Pure SDP family, including affine PSD row blocks
# ---------------------------------------------------------------------------

Test.@testset "R3 pure SDP family and affine PSD rows" begin
    # PSD product block only; packed length 6 for a 3x3 block.
    route = SDPX.classify_native_cone_program(
        _route_program(products=((SDPX.PSDCone(), 3),)),
    )
    @test route.route === :sdp_family
    @test route.product_families == [:sdp_family]
    @test route.product_dimensions == [3]
    @test route.variables == 6
    @test route.rows == 0

    # Affine PSD RowBlock counts as PSD (domain and shape).
    affine_psd = SDPX.classify_native_cone_program(
        _route_program(
            products=((SDPX.ZeroCone(), 1),),
            rows=((SDPX.PSDCone(), 2), (SDPX.Reals(), 1)),
        ),
    )
    @test affine_psd.route === :sdp_family
    @test affine_psd.product_families == [:zero]
    @test affine_psd.row_families == [:sdp_family, :free]
    @test affine_psd.row_dimensions == [2, 1]
    @test affine_psd.rows == 3 + 1
    @test affine_psd.variables == 1

    # PSD product + PSD row + zero stays one family.
    all_psd = SDPX.classify_native_cone_program(
        _route_program(
            products=((SDPX.PSDCone(), 2), (SDPX.ZeroCone(), 1)),
            rows=((SDPX.PSDCone(), 2),),
        ),
    )
    @test all_psd.route === :sdp_family
    @test all_psd.product_families == [:sdp_family, :zero]
    @test all_psd.row_families == [:sdp_family]
    @test all_psd.variables == 4
    @test all_psd.rows == 3
end

# ---------------------------------------------------------------------------
# Mixed routes fail closed
# ---------------------------------------------------------------------------

Test.@testset "R4 mixed routes rejected with exact fields and message" begin
    cases = (
        (
            "orthant + SOC",
            ((SDPX.Nonnegative(), 2), (SDPX.LorentzCone(), 3)),
            [:lp_family, :soc_family],
            "unsupported mixed native cone route: detected families [:lp_family, :soc_family]",
        ),
        (
            "orthant + PSD",
            ((SDPX.Nonpositive(), 1), (SDPX.PSDCone(), 2)),
            [:lp_family, :sdp_family],
            "unsupported mixed native cone route: detected families [:lp_family, :sdp_family]",
        ),
        (
            "PSD + SOC",
            ((SDPX.PSDCone(), 2), (SDPX.LorentzCone(), 3)),
            [:soc_family, :sdp_family],
            "unsupported mixed native cone route: detected families [:soc_family, :sdp_family]",
        ),
        (
            "orthant + SOC + PSD",
            ((SDPX.Nonnegative(), 1), (SDPX.LorentzCone(), 3), (SDPX.PSDCone(), 2)),
            [:lp_family, :soc_family, :sdp_family],
            "unsupported mixed native cone route: detected families [:lp_family, :soc_family, :sdp_family]",
        ),
    )

    for (name, products, expected_families, expected_message) in cases
        program = _route_program(products=products)
        err = _caught_error(() -> SDPX.classify_native_cone_program(program))
        @test err isa SDPX.UnsupportedNativeConeRoute
        @test err.detected_families == expected_families
        @test err.message == expected_message
        @test sprint(showerror, err) == expected_message
        # Fail closed: no partial route was returned and memory metadata is untouched.
        @test length(program.blocks) == length(products)
    end

    # Cross placement: orthant product with SOC row and PSD row.
    mixed_rows = _route_program(
        products=((SDPX.Nonnegative(), 1),),
        rows=((SDPX.LorentzCone(), 3), (SDPX.PSDCone(), 2)),
    )
    err_rows = _caught_error(() -> SDPX.classify_native_cone_program(mixed_rows))
    @test err_rows isa SDPX.UnsupportedNativeConeRoute
    @test err_rows.detected_families == [:lp_family, :soc_family, :sdp_family]

    # Orthant rows + SOC product also reject.
    cross = _route_program(
        products=((SDPX.LorentzCone(), 3),),
        rows=((SDPX.Nonnegative(), 2),),
    )
    cross_err = _caught_error(() -> SDPX.classify_native_cone_program(cross))
    @test cross_err isa SDPX.UnsupportedNativeConeRoute
    @test cross_err.detected_families == [:lp_family, :soc_family]

    # RSOC conflicts with orthant exactly like SOC.
    rsoc_mix = _route_program(
        products=((SDPX.RotatedLorentzCone(), 4), (SDPX.Nonnegative(), 1)),
    )
    rsoc_err = _caught_error(() -> SDPX.classify_native_cone_program(rsoc_mix))
    @test rsoc_err.detected_families == [:lp_family, :soc_family]
end

# ---------------------------------------------------------------------------
# Permutations preserve family and order metadata
# ---------------------------------------------------------------------------

Test.@testset "R5 block permutations" begin
    lp_a = SDPX.classify_native_cone_program(_route_program(
        products=((SDPX.Nonnegative(), 2), (SDPX.ZeroCone(), 1)),
        rows=((SDPX.ZeroCone(), 1),),
    ))
    lp_b = SDPX.classify_native_cone_program(_route_program(
        products=((SDPX.ZeroCone(), 1), (SDPX.Nonnegative(), 2)),
        rows=(),
    ))
    @test lp_a.route == lp_b.route
    @test lp_a.route === :lp_family
    @test lp_a.product_families == [:lp_family, :zero]
    @test lp_b.product_families == [:zero, :lp_family]
    @test lp_a.product_dimensions == [2, 1]
    @test lp_b.product_dimensions == [1, 2]

    soc_a = SDPX.classify_native_cone_program(_route_program(
        products=((SDPX.LorentzCone(), 3), (SDPX.RotatedLorentzCone(), 4), (SDPX.ZeroCone(), 1)),
    ))
    soc_b = SDPX.classify_native_cone_program(_route_program(
        products=((SDPX.ZeroCone(), 1), (SDPX.LorentzCone(), 3)),
        rows=((SDPX.RotatedLorentzCone(), 4),),
    ))
    @test soc_a.route === :soc_family
    @test soc_b.route === :soc_family
    @test soc_a.product_families == [:soc_family, :soc_family, :zero]
    @test soc_b.product_families == [:zero, :soc_family]
    @test soc_b.row_families == [:soc_family]

    sdp_a = SDPX.classify_native_cone_program(_route_program(
        products=((SDPX.PSDCone(), 2), (SDPX.Reals(), 2)),
        rows=((SDPX.PSDCone(), 2),),
    ))
    sdp_b = SDPX.classify_native_cone_program(_route_program(
        products=((SDPX.Reals(), 2), (SDPX.PSDCone(), 2)),
        rows=((SDPX.ZeroCone(), 1),),
    ))
    @test sdp_a.route === :sdp_family
    @test sdp_b.route === :sdp_family
    @test sdp_a.product_families == [:sdp_family, :free]
    @test sdp_b.product_families == [:free, :sdp_family]

    # Mixed permutations throw the same exact error regardless of order.
    for products in (
        ((SDPX.Nonnegative(), 2), (SDPX.LorentzCone(), 3)),
        ((SDPX.LorentzCone(), 3), (SDPX.Nonnegative(), 2)),
    )
        err = _caught_error(() -> SDPX.classify_native_cone_program(_route_program(products=products)))
        @test err isa SDPX.UnsupportedNativeConeRoute
        @test err.detected_families == [:lp_family, :soc_family]
        @test err.message ==
              "unsupported mixed native cone route: detected families [:lp_family, :soc_family]"
    end

    for products in (
        ((SDPX.PSDCone(), 2), (SDPX.LorentzCone(), 3)),
        ((SDPX.LorentzCone(), 3), (SDPX.PSDCone(), 2)),
    )
        err = _caught_error(() -> SDPX.classify_native_cone_program(_route_program(products=products)))
        @test err isa SDPX.UnsupportedNativeConeRoute
        @test err.detected_families == [:soc_family, :sdp_family]
    end
end

# ---------------------------------------------------------------------------
# BigFloat classification invariance
# ---------------------------------------------------------------------------

Test.@testset "R6 BigFloat classification matches Float64" begin
    specs = (
        (products=((SDPX.Nonnegative(), 2), (SDPX.Reals(), 1)), rows=((SDPX.ZeroCone(), 2),)),
        (products=((SDPX.LorentzCone(), 3), (SDPX.RotatedLorentzCone(), 4)), rows=((SDPX.Reals(), 1),)),
        (products=((SDPX.PSDCone(), 3),), rows=((SDPX.ZeroCone(), 1),)),
        (products=(), rows=()),
    )
    for spec in specs
        f64_route = SDPX.classify_native_cone_program(_route_program(; spec..., T=Float64))
        big_route = SDPX.classify_native_cone_program(_route_program(; spec..., T=BigFloat))
        @test big_route isa SDPX.NativeConeRoute
        @test big_route.route == f64_route.route
        @test big_route.product_families == f64_route.product_families
        @test big_route.row_families == f64_route.row_families
        @test big_route.product_dimensions == f64_route.product_dimensions
        @test big_route.row_dimensions == f64_route.row_dimensions
        @test big_route.variables == f64_route.variables
        @test big_route.rows == f64_route.rows
    end

    # Mixed error is T-independent too.
    mixed = (products=((SDPX.Nonnegative(), 1), (SDPX.LorentzCone(), 3)),)
    f64_err = _caught_error(() -> SDPX.classify_native_cone_program(_route_program(; mixed..., T=Float64)))
    big_err = _caught_error(() -> SDPX.classify_native_cone_program(_route_program(; mixed..., T=BigFloat)))
    @test f64_err isa SDPX.UnsupportedNativeConeRoute
    @test big_err isa SDPX.UnsupportedNativeConeRoute
    @test big_err.detected_families == f64_err.detected_families
    @test big_err.message == f64_err.message
end

# ---------------------------------------------------------------------------
# Structural-count validation
# ---------------------------------------------------------------------------

Test.@testset "R7 structural counts validated fail-closed" begin
    program = _route_program(products=((SDPX.Nonnegative(), 2),), rows=((SDPX.ZeroCone(), 1),))
    @test SDPX.classify_native_cone_program(program).route === :lp_family

    # Mutate a product block into `blocks` after construction: totals no
    # longer match the program's declared variable count.
    bad_blocks = _route_program(products=((SDPX.Nonnegative(), 2),))
    offset = 1 + sum(SDPX.block_length, bad_blocks.blocks)
    push!(bad_blocks.blocks, SDPX.NativeBlock(SDPX.ZeroCone(), 1, offset))
    err = _caught_error(() -> SDPX.classify_native_cone_program(bad_blocks))
    @test err isa ArgumentError
    @test occursin("block variables total", sprint(showerror, err))

    # Mutate a row block similarly.
    bad_rows = _route_program(rows=((SDPX.ZeroCone(), 1),))
    row_offset = 1 + sum(SDPX.row_block_length, bad_rows.row_blocks)
    push!(bad_rows.row_blocks, SDPX.RowBlock(SDPX.ZeroCone(), row_offset, 2))
    err_rows = _caught_error(() -> SDPX.classify_native_cone_program(bad_rows))
    @test err_rows isa ArgumentError
    @test occursin("row block rows total", sprint(showerror, err_rows))
end

# ---------------------------------------------------------------------------
# Static type contract: required fields, banned fields, no Any
# ---------------------------------------------------------------------------

Test.@testset "R8 static fields, banned names, no Any" begin
    @test SDPX.UnsupportedNativeConeRoute <: Exception
    @test isconcretetype(SDPX.NativeConeRoute)
    @test isconcretetype(SDPX.UnsupportedNativeConeRoute)

    route_fields = fieldnames(SDPX.NativeConeRoute)
    for required in (
        :route,
        :product_families,
        :row_families,
        :num_product_blocks,
        :num_row_blocks,
        :product_dimensions,
        :row_dimensions,
        :variables,
        :rows,
    )
        @test required ∈ route_fields
    end

    error_fields = fieldnames(SDPX.UnsupportedNativeConeRoute)
    @test :detected_families ∈ error_fields
    @test :message ∈ error_fields

    banned = (
        :orientation,
        :dual_model,
        :primal_model,
        :dualization,
        :dualization_metadata,
        :formulation,
        :formulation_choice,
        :provider,
        :solver_choice,
        :precision_bits,
        :arithmetic,
        :source_model,
    )
    for field in route_fields
        @test field ∉ banned
        @test fieldtype(SDPX.NativeConeRoute, field) !== Any
    end
    for field in error_fields
        @test field ∉ banned
        @test fieldtype(SDPX.UnsupportedNativeConeRoute, field) !== Any
    end

    @test fieldtype(SDPX.NativeConeRoute, :route) === Symbol
    @test fieldtype(SDPX.NativeConeRoute, :product_families) === Vector{Symbol}
    @test fieldtype(SDPX.NativeConeRoute, :row_families) === Vector{Symbol}
    @test fieldtype(SDPX.NativeConeRoute, :variables) === Int
    @test fieldtype(SDPX.NativeConeRoute, :rows) === Int
    @test fieldtype(SDPX.UnsupportedNativeConeRoute, :detected_families) === Vector{Symbol}
    @test fieldtype(SDPX.UnsupportedNativeConeRoute, :message) === String
end
