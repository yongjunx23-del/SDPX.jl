#=====================================================================#
#    Native cone route classifier (v0.5).
#
#    Pure, typed dispatch metadata for a `NativeConeProgram`: it
#    decides whether an ordered program belongs to the LP, SOC or SDP
#    route family and, when nonfree cone families are mixed, it fails
#    closed with `UnsupportedNativeConeRoute` instead of guessing.
#
#    This file contains NO numerical lowering, NO canonicalization, NO
#    split/lift/scalarization, and NO solver/provider/formulation or
#    precision decisions. It records only mathematical route families
#    and structural block metadata in program order.
#
#    DELIBERATE ABSENCES (fixed by the SOL contract):
#    - no `orientation` field of any kind;
#    - no primal/dual model labels or dualization metadata;
#    - no model names, dimension fingerprints, provider choices or
#      arithmetic/precision decisions in fields or API.
#
#    Include order: after src/modeling/domains.jl, refs.jl,
#    types.jl and src/ir/types.jl (uses `NativeConeProgram`,
#    `NativeBlock` and `RowBlock`).
#=====================================================================#

# ---------------------------------------------------------------------------
# Route classification record
# ---------------------------------------------------------------------------

"""
    SDPX.NativeConeRoute

Immutable, fully typed route classification of a
[`NativeConeProgram`](@ref). One program maps to exactly one route
family; the record preserves product/row block order.

Fields
- `route::Symbol` — `:lp_family`, `:soc_family` or `:sdp_family`.
- `product_families::Vector{Symbol}` — one family label per product
  block, in program order: `:free`, `:zero`, `:lp_family`,
  `:soc_family` or `:sdp_family`.
- `row_families::Vector{Symbol}` — one family label per affine row
  block, in program order (same label set; an affine PSD `RowBlock`
  is `:sdp_family`).
- `num_product_blocks::Int`, `num_row_blocks::Int` — native block
  counts.
- `product_dimensions::Vector{Int}`, `row_dimensions::Vector{Int}` —
  native block dimensions in program order (matrix dimension for PSD
  blocks).
- `variables::Int` — total stored scalar variables (concatenated
  product-block lengths).
- `rows::Int` — total stored rows (concatenated row-block lengths).

This is dispatch metadata only. It carries no orientation, no
primal/dual labels, no dualization metadata, no provider or
formulation choice, and no arithmetic/precision decision. It never
scalarizes, lifts or splits a block.
"""
struct NativeConeRoute
    route::Symbol
    product_families::Vector{Symbol}
    row_families::Vector{Symbol}
    num_product_blocks::Int
    num_row_blocks::Int
    product_dimensions::Vector{Int}
    row_dimensions::Vector{Int}
    variables::Int
    rows::Int
end

# Route records are value metadata.  The default `==` for a struct with
# mutable vector fields compares those vectors by identity in Julia, which
# makes two independent classifications of the same program appear unequal.
# Define the intended structural comparison explicitly; route classification
# itself remains allocation-free on the mixed-family rejection path.
Base.:(==)(left::NativeConeRoute, right::NativeConeRoute) =
    left.route == right.route &&
    left.product_families == right.product_families &&
    left.row_families == right.row_families &&
    left.num_product_blocks == right.num_product_blocks &&
    left.num_row_blocks == right.num_row_blocks &&
    left.product_dimensions == right.product_dimensions &&
    left.row_dimensions == right.row_dimensions &&
    left.variables == right.variables &&
    left.rows == right.rows

Base.isequal(left::NativeConeRoute, right::NativeConeRoute) = left == right

"""
    SDPX.UnsupportedNativeConeRoute <: Exception

Fail-closed error thrown by [`classify_native_cone_program`](@ref)
when the nonfree cone domains of a program span more than one route
family (for example orthant + SOC, orthant + PSD, or PSD + SOC).

Fields
- `detected_families::Vector{Symbol}` — the conflicting route
  families, deterministically ordered `:lp_family`, `:soc_family`,
  `:sdp_family`.
- `message::String` — fixed-format diagnostic message.
"""
struct UnsupportedNativeConeRoute <: Exception
    detected_families::Vector{Symbol}
    message::String
end

function UnsupportedNativeConeRoute(detected_families::Vector{Symbol})
    # Canonical family order: LP, SOC, SDP (lexicographic Symbol order
    # would place `:sdp_family` before `:soc_family`, which is neither
    # documented nor stable under implementation details).
    families = Symbol[
        family for family in (:lp_family, :soc_family, :sdp_family)
        if family in detected_families
    ]
    length(families) >= 2 ||
        throw(ArgumentError("a mixed-route error needs at least two detected families, got $families"))
    message = "unsupported mixed native cone route: detected families $families"
    return UnsupportedNativeConeRoute(families, message)
end

Base.showerror(io::IO, err::UnsupportedNativeConeRoute) = print(io, err.message)

# ---------------------------------------------------------------------------
# Family labels (pure mathematical mapping; never a formulation choice)
# ---------------------------------------------------------------------------

_route_family(::Reals) = :free
_route_family(::Nonnegative) = :lp_family
_route_family(::Nonpositive) = :lp_family
_route_family(::ZeroCone) = :zero
_route_family(::LorentzCone) = :soc_family
_route_family(::RotatedLorentzCone) = :soc_family
_route_family(::PSDCone) = :sdp_family
_route_family(domain) =
    throw(ArgumentError("unsupported native cone domain $domain in route classification"))

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

"""
    classify_native_cone_program(program) -> NativeConeRoute

Classify an ordered native program into exactly one route family:

- `:lp_family` — every nonfree cone is Nonnegative/Nonpositive/Zero
  (`Reals` is allowed; an empty, all-free or equality-only program
  classifies as LP).
- `:soc_family` — every nonfree cone is SOC/RSOC/Zero (`Reals` is
  allowed).
- `:sdp_family` — every nonfree cone is PSD/Zero (`Reals` is
  allowed).
- otherwise it throws [`UnsupportedNativeConeRoute`](@ref) (mixed
  families) before constructing any route result.

Affine PSD [`RowBlock`](@ref)s count as PSD for family detection. The
classifier validates structural counts — product-block variables,
row-block rows and equality-matrix size must match the program — and
preserves block order in every per-block vector. It never lowers,
lifts, splits or scalarizes the program, and it makes no provider,
formulation, orientation or precision decision.
"""
function classify_native_cone_program(program::NativeConeProgram{T}) where {T<:AbstractFloat}
    blocks = program.blocks
    row_blocks = program.row_blocks

    # First pass: structural counts + family detection. Allocation-free,
    # so mixed programs fail closed before any route record is built.
    total_variables = 0
    total_rows = 0
    saw_lp = false
    saw_soc = false
    saw_sdp = false
    for block in blocks
        total_variables += block.length
        family = _route_family(block.domain)
        saw_lp |= family === :lp_family
        saw_soc |= family === :soc_family
        saw_sdp |= family === :sdp_family
    end
    for row_block in row_blocks
        total_rows += row_block.length
        family = _route_family(row_block.domain)
        saw_lp |= family === :lp_family
        saw_soc |= family === :soc_family
        saw_sdp |= family === :sdp_family
    end

    total_variables == program_num_variables(program) ||
        throw(ArgumentError(
            "block variables total $total_variables != program variables $(program_num_variables(program))",
        ))
    total_rows == program_num_rows(program) ||
        throw(ArgumentError(
            "row block rows total $total_rows != program rows $(program_num_rows(program))",
        ))
    size(program.equality_matrix) == (total_rows, total_variables) ||
        throw(ArgumentError(
            "equality matrix size $(size(program.equality_matrix)) != ($total_rows, $total_variables)",
        ))

    detected_count = (saw_lp ? 1 : 0) + (saw_soc ? 1 : 0) + (saw_sdp ? 1 : 0)
    if detected_count > 1
        families = Symbol[]
        saw_lp && push!(families, :lp_family)
        saw_soc && push!(families, :soc_family)
        saw_sdp && push!(families, :sdp_family)
        throw(UnsupportedNativeConeRoute(families))
    end
    route = saw_lp ? :lp_family :
            saw_soc ? :soc_family :
            saw_sdp ? :sdp_family :
            :lp_family

    # Second pass: ordered, typed metadata for the accepted route.
    product_families = Vector{Symbol}(undef, length(blocks))
    row_families = Vector{Symbol}(undef, length(row_blocks))
    product_dimensions = Vector{Int}(undef, length(blocks))
    row_dimensions = Vector{Int}(undef, length(row_blocks))
    for (i, block) in enumerate(blocks)
        product_families[i] = _route_family(block.domain)
        product_dimensions[i] = block.shape
    end
    for (i, row_block) in enumerate(row_blocks)
        row_families[i] = _route_family(row_block.domain)
        row_dimensions[i] = row_block.shape
    end
    return NativeConeRoute(
        route,
        product_families,
        row_families,
        length(blocks),
        length(row_blocks),
        product_dimensions,
        row_dimensions,
        total_variables,
        total_rows,
    )
end
