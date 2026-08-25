#=====================================================================#
#    Native cone route classifier (v0.5).
#
#    Pure, typed dispatch metadata for a `NativeConeProgram`: it
#    decides whether an ordered program belongs to the LP, SOC or SDP
#    route family. Only the selected family is retained; block order and
#    dimensions already live in `NativeConeProgram` and are not duplicated.
#
#    This file contains NO numerical lowering, NO canonicalization, NO
#    split/lift/scalarization, and NO solver/provider/formulation or
#    precision decisions. It records only the mathematical route family.
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
[`NativeConeProgram`](@ref). One program maps to exactly one route family.

Fields
- `route::Symbol` — `:lp_family`, `:soc_family`, `:sdp_family` or
  `:exp_family`.

This is dispatch metadata only. It carries no orientation, no
primal/dual labels, no dualization metadata, no provider or
formulation choice, and no arithmetic/precision decision. It never
scalarizes, lifts or splits a block.
"""
struct NativeConeRoute
    route::Symbol
end

const NATIVE_ROUTE_FAMILY_ORDER =
    (:lp_family, :soc_family, :sdp_family, :exp_family, :power_family)

"""
    SDPX.UnsupportedNativeConeRoute <: Exception

Fail-closed error thrown by [`classify_native_cone_program`](@ref)
when the nonfree cone domains of a program span more than one route
family (for example orthant + SOC, orthant + PSD, PSD + SOC, or
orthant + exponential).

The detected families are stored in deterministic LP, SOC, SDP, EXP
order.
"""
struct UnsupportedNativeConeRoute <: Exception
    detected_families::Vector{Symbol}
    function UnsupportedNativeConeRoute(detected_families::Vector{Symbol})
        # Canonical family order: LP, SOC, SDP, EXP. Lexicographic Symbol
        # order would put SDP before SOC and is not part of the model contract.
        families = Symbol[
            family for family in
            NATIVE_ROUTE_FAMILY_ORDER
            if family in detected_families
        ]
        length(families) >= 2 || throw(ArgumentError(
            "a mixed-route error needs at least two detected families, got $families",
        ))
        return new(families)
    end
end

Base.showerror(io::IO, err::UnsupportedNativeConeRoute) =
    print(io, "model combines unsupported cone families ", err.detected_families)

# ---------------------------------------------------------------------------
# Family labels (pure mathematical mapping; never a formulation choice)
# ---------------------------------------------------------------------------

function _route_family(kind::Symbol)
    kind === :free && return :free
    kind === :zero && return :zero
    kind in (:nonnegative, :nonpositive, :interval) && return :lp_family
    kind in (:soc, :rsoc) && return :soc_family
    kind in (:psd, :psd_scaled) && return :sdp_family
    kind === :exp && return :exp_family
    kind === :power && return :power_family
    throw(ArgumentError("unsupported cone kind $kind in route classification"))
end

_route_family(domain::ProductConeDomain) = _route_family(_domain_cone(domain))
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
- `:exp_family` — every nonfree cone is ExponentialCone/Zero
  (`Reals` is allowed). The exponential solver root lands with the
  Phase B native kernels; until then classification succeeds while
  any lowering attempt fails closed.
- `:power_family` — every nonfree cone is PowerCone/Zero (`Reals`
  is allowed).
- `:mixed_family` — the program spans more than one family. A
  heterogeneous cone product is a first-class layout (see
  [`cone_product_layout`](@ref)), not an error: the single-family
  lowering paths fail closed at lowering time, while the unified HSD
  path consumes the layout.

Affine PSD [`RowBlock`](@ref)s count as PSD for family detection. The
classifier validates structural counts — product-block variables,
row-block rows and equality-matrix size must match the program — and
does not duplicate block metadata already owned by the program. It never
lowers, lifts, splits or scalarizes the program, and it makes no provider,
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
    saw_exp = false
    saw_power = false
    for block in blocks
        total_variables += block.length
        family = _route_family(block.domain)
        saw_lp |= family === :lp_family
        saw_soc |= family === :soc_family
        saw_sdp |= family === :sdp_family
        saw_exp |= family === :exp_family
        saw_power |= family === :power_family
    end
    for row_block in row_blocks
        total_rows += row_block.length
        family = _route_family(row_block.domain)
        saw_lp |= family === :lp_family
        saw_soc |= family === :soc_family
        saw_sdp |= family === :sdp_family
        saw_exp |= family === :exp_family
        saw_power |= family === :power_family
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

    detected_count = (saw_lp ? 1 : 0) + (saw_soc ? 1 : 0) + (saw_sdp ? 1 : 0) +
                     (saw_exp ? 1 : 0) + (saw_power ? 1 : 0)
    if detected_count > 1
        # A heterogeneous cone product is a first-class layout, not an
        # error. The single-family lowering paths still fail closed at
        # lowering time; the unified HSD path consumes the layout.
        return NativeConeRoute(:mixed_family)
    end
    route = saw_lp ? :lp_family :
            saw_soc ? :soc_family :
            saw_sdp ? :sdp_family :
            saw_exp ? :exp_family :
            saw_power ? :power_family :
            :lp_family

    return NativeConeRoute(route)
end
