# ===========================================================================
# src/session/update.jl
#
# S07 step 1 — the EXPLICIT prepared-update invalidation table.
#
# WHAT THIS FILE IS.  When a prepared problem changes, SDPX must decide, per
# change, which of four *different* things survives:
#
#     symbolic structure   |  the numeric factor  |  the rank authority
#     the certificate      |  the warm start      |  the workspace
#
# A single "the matrix changed, so start over" rule is wrong in both
# directions: it throws away a factor that is provably still valid (a `c` or
# `b` change does not touch `A` at all), and it *keeps* authority that is no
# longer valid (a same-pattern value change can change the rank).  The card
# names the second failure exactly: 不能因 same pattern 误保留 rank authority.
#
# So the decision here is a table, not a predicate, and the table is audited by
# `update_table_audit()` — a runnable object, not a comment.
#
# THREE AXES, NOT ONE SWITCH.  warm start, workspace reuse and numeric factor
# reuse are separate fields of `UpdateEffect` and the table is required to
# contain a row where they disagree; `update_table_audit()` reports those rows
# as witnesses.  Collapsing them into one `reuse::Bool` is the mistake the card
# forbids.
#
# WHAT THIS FILE DOES NOT DO.  It does not factorize, does not solve, does not
# decide tolerances for the caller, and does not introduce a second HSD loop.
# It reads and mutates the ADR-002 §4 logical lease through the existing
# `revoke!` / `FactorHandle` machinery in `src/la/factor_lease.jl`; it never
# reads provider status to justify a reuse.  Nothing here reads or writes the
# rounding mode of any provider — B01's `src/mpfr.jl` proposal is NOT applied,
# and this file does not depend on it.
#
# INCLUDE ORDER.  `replay.jl` uses `SessionTolerance`, `SessionScalarPayload`,
# `session_scalar_payload` and `ProblemFingerprint`, so this file must be
# included BEFORE `replay.jl`.  That order is measured, not assumed: the driver
# includes the pair in both orders and records which one compiles.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. what can change about a prepared problem
# ---------------------------------------------------------------------------

"""
    ProblemChange

One reason a prepared problem is not the same prepared problem any more.
The set is closed on purpose: an update that cannot name its change cannot be
planned, and a planner with a catch-all `ChangeOther` would silently acquire the
"reuse everything" default this table exists to prevent.
"""
@enum ProblemChange::UInt8 begin
    ChangeNone               # the "no change" row; never folded with a real one
    ChangeObjective          # c
    ChangeRHS                # b
    ChangeOperatorValues     # A values, SAME structural pattern
    ChangeOperatorPattern    # A structure/pattern, or its dimensions
    ChangeConeParameters     # cone-member parameters (RSOC, PSD block sizes, ...)
    ChangePrecision          # arithmetic family or requested bit width
    ChangeRounding           # rounding mode of the arithmetic contract
    ChangeOrdering           # provider fill-reducing ordering
    ChangeRankTransform      # rank kind / rank transform applied to the operator
end

const SESSION_PROBLEM_CHANGES = (
    ChangeObjective, ChangeRHS, ChangeOperatorValues, ChangeOperatorPattern,
    ChangeConeParameters, ChangePrecision, ChangeRounding, ChangeOrdering,
    ChangeRankTransform,
)

# Canonical ordering for printing and for folding, so two callers that detect
# the same change set in different orders get the same plan.
const SESSION_CHANGE_ORDER = Dict{ProblemChange,Int}(
    c => i for (i, c) in enumerate(SESSION_PROBLEM_CHANGES)
)

change_label(c::ProblemChange) = Symbol(lowercase(string(c)))

"""
    SESSION_RANK_REVOKING_CHANGES

Changes after which the *previous* factor's rank must not be quoted again.
This tuple is the load-bearing constant of the whole card:

  * `ChangeOperatorValues` is in it although the pattern is unchanged.  Rank is
    a property of the VALUES, not of the pattern, so "same pattern" is not a
    licence to keep the rank.  A rank-revealing factor of a full-rank matrix
    does not reveal the rank of a rank-deficient matrix with the same sparsity.
  * `ChangeRankTransform` is in it because it changes what "rank" is even
    being asked for.
  * `ChangePrecision` / `ChangeRounding` are in it because a rank authority
    obtained in one arithmetic is not an authority in another.

`ChangeObjective`, `ChangeRHS` and `ChangeConeParameters` are NOT in it: none
of them is an input to the factor, and the operator's rank is untouched by
definition.  `update_table_audit()` checks that claim against the table row by
row rather than trusting this comment.
"""
const SESSION_RANK_REVOKING_CHANGES = (
    ChangeOperatorValues, ChangeOperatorPattern, ChangePrecision,
    ChangeRounding, ChangeOrdering, ChangeRankTransform,
)

"""
    RankAuthority

Where a rank answer comes from, and whether it may be quoted.
`RankAuthorityRetained` is only legal while the operator bytes that produced the
factor are unchanged; see [`SESSION_RANK_REVOKING_CHANGES`](@ref).
"""
@enum RankAuthority::UInt8 begin
    RankAuthorityNone        # never had one
    RankAuthorityRetained    # the operator is unchanged, so the old rank stands
    RankAuthorityFromNewFactor  # a new numeric factor produced a new rank
    RankAuthorityRevoked     # invalidated; may NOT be quoted, must be recomputed
end

rank_authority_quotable(a::RankAuthority) =
    a === RankAuthorityRetained || a === RankAuthorityFromNewFactor

"""
    RecomputeLevel

The strongest work an update forces, ordered.  `max` of two levels is the fold
rule for a multi-change update: a single `ChangePrecision` inside a set of
otherwise cheap changes must not be averaged away.
"""
@enum RecomputeLevel::UInt8 begin
    RecomputeNothing     # no re-factorization, no re-symbolization
    RecomputeFactorOnly  # numeric refactor against the existing symbolic factor
    RecomputeSymbolic    # symbolic structure must be rebuilt (prepare again)
    RecomputeFull        # symbolic + admission + certificate + re-verification
end

# ---------------------------------------------------------------------------
# 2. the rounding label
#
# This is a *label the update table compares*, not a rounding authority.  It is
# mapped from Julia's own `RoundingMode` so that the replay envelope records the
# same object the arithmetic was performed under, and `RoundingUnspecified` is a
# legal, honestly-labelled state rather than a silent default.
# ---------------------------------------------------------------------------

@enum SessionRounding::UInt8 begin
    RoundingUnspecified
    RoundingNearestEven
    RoundingUp
    RoundingDown
    RoundingToZero
    RoundingAwayFromZero
end

function session_rounding(mode::Base.Rounding.RoundingMode)
    mode === Base.Rounding.RoundNearest && return RoundingNearestEven
    mode === Base.Rounding.RoundUp && return RoundingUp
    mode === Base.Rounding.RoundDown && return RoundingDown
    mode === Base.Rounding.RoundToZero && return RoundingToZero
    if isdefined(Base.Rounding, :RoundFromZero) && mode === Base.Rounding.RoundFromZero
        return RoundingAwayFromZero
    end
    return RoundingUnspecified
end

"""
    session_rounding_supported(r) -> Bool

Whether this build can map `r` back to a live Julia `RoundingMode`.  Used by the
driver to avoid asserting a mapping the running Julia does not define.
"""
function session_rounding_supported(r::SessionRounding)
    r === RoundingUnspecified && return false
    r === RoundingNearestEven && return true
    r === RoundingUp && return true
    r === RoundingDown && return true
    r === RoundingToZero && return true
    return isdefined(Base.Rounding, :RoundFromZero)
end

# ---------------------------------------------------------------------------
# 3. exact scalar payloads — the "no implicit Float64" boundary
#
# The fingerprint of the operator VALUES and the replay file must be computed
# from the same exact bytes, or a replay can reproduce a problem that is not the
# one that ran.  So there is ONE codec here, and `replay.jl` reuses it.
#
# A value that this codec cannot encode EXACTLY is REFUSED.  It is never
# converted through Float64 to make it fit: a silent `Float64(x)` is exactly the
# "隐式降精度" the card forbids, and it is undetectable downstream because the
# resulting number is a perfectly ordinary float.
# ---------------------------------------------------------------------------

"""
    UpdateScalarError

Thrown when an exact arithmetic payload cannot be produced or consumed.  Codes:
`:unsupported_scalar_type`, `:unknown_type`, `:width_mismatch`, `:bad_number`,
`:not_round_trippable`.
"""
struct UpdateScalarError <: Exception
    code::Symbol
    detail::String
end

function Base.showerror(io::IO, e::UpdateScalarError)
    print(io, "UpdateScalarError[", e.code, "]: ", e.detail)
end

"""
    SessionScalarPayload

The exact serialized form of one scalar.  `hex` is load-bearing; `text` is a
human-readable rendering and is NEVER used to reconstruct the value for the
`:ieee_bits` and `:isbits_bytes` kinds.

`kind`:
  * `:ieee_bits`      — the raw IEEE bit pattern, hex.  No float parsing, so
                        `-0.0`, `NaN` payloads and subnormals survive exactly.
  * `:bigfloat_string`— `string(x)` at the value's own precision.  MPFR prints
                        enough digits to round-trip; the decoder re-checks that
                        property and refuses the payload if it fails.
  * `:isbits_bytes`   — the raw bytes of an isbits scalar (the MF-limb path).
                        Decoding requires the recorded type to resolve and to
                        have the recorded width.
"""
struct SessionScalarPayload
    kind::Symbol
    type_name::String
    type_bits::Int
    hex::String
    text::String
end

_bytes_to_hex(bytes) = join(string(b, base=16, pad=2) for b in bytes)
_hex_to_bytes(s::AbstractString) = UInt8[
    parse(UInt8, s[i:i + 1], base=16) for i in 1:2:lastindex(s)
]

"""
    _ieee_uint_type(T) -> Type{<:Unsigned}

The unsigned integer of the same width as an IEEE float.  Explicit rather than
`unsigned(T)`: in Julia 1.12 `unsigned(::Type{Float64})` is NOT defined (the
generic `unsigned(x) = x % unsigned(typeof(x))` ends in
`rem(::Type{Float64}, ::Type{UInt64})`), and a width that has to be *added*
deliberately is better than one that is inferred.
"""
_ieee_uint_type(::Type{Float16}) = UInt16
_ieee_uint_type(::Type{Float32}) = UInt32
_ieee_uint_type(::Type{Float64}) = UInt64

"""
    session_scalar_payload(x) -> SessionScalarPayload

Exact encoding of one arithmetic input scalar.
"""
function session_scalar_payload(x::T) where {T<:Base.IEEEFloat}
    U = _ieee_uint_type(T)
    bits = reinterpret(U, x)
    SessionScalarPayload(:ieee_bits, string(T), 8 * sizeof(T),
                         string(bits, base=16, pad=2 * sizeof(T)), string(x))
end

function session_scalar_payload(x::BigFloat)
    s = string(x)
    SessionScalarPayload(:bigfloat_string, "BigFloat", precision(x),
                         _bytes_to_hex(codeunits(s)), s)
end

function session_scalar_payload(x::T) where {T}
    if isbitstype(T)
        bytes = collect(reinterpret(UInt8, [x]))
        return SessionScalarPayload(:isbits_bytes, string(T), 8 * sizeof(T),
                                    _bytes_to_hex(bytes), _isbits_text(x))
    end
    throw(UpdateScalarError(:unsupported_scalar_type,
        "no exact payload encoding for $(T); refusing rather than converting through Float64"))
end

# The readable label for the opaque-bytes kinds.  Deliberately ALLOWED to throw:
# a label is not part of the contract, and a type without a `show` method must
# not make the encoding fail.
function _isbits_text(x)
    try
        return string(x)
    catch
        return ""
    end
end

"""
    session_scalar_payload(x, ::Type{T}; module) — decode helper for the driver.

    session_resolve_type(mod, name, bits) -> Type

Resolve a recorded type name in `mod` (Base first), requiring it to be an isbits
type of exactly `bits` bits.  Anything else is an explicit `UpdateScalarError`,
which is how a version mismatch between the writer and the reader surfaces.
"""
function session_resolve_type(mod::Module, name::AbstractString, bits::Integer)
    ex = try
        Meta.parse(name)
    catch err
        throw(UpdateScalarError(:unknown_type, "type name $(repr(name)) does not parse: $(err)"))
    end
    if ex isa Symbol
        isdefined(Base, ex) && (base_t = getfield(Base, ex); base_t isa Type &&
            return _check_type_width(base_t, name, bits))
    end
    T = try
        Core.eval(mod, ex)
    catch err
        throw(UpdateScalarError(:unknown_type,
            "type $(repr(name)) does not resolve in $(mod): $(err)"))
    end
    T isa Type || throw(UpdateScalarError(:unknown_type, "$(repr(name)) is not a type"))
    return _check_type_width(T, name, bits)
end

function _check_type_width(::Type{T}, name, bits) where {T}
    isbitstype(T) || throw(UpdateScalarError(:unknown_type,
        "recorded type $(name) is not isbits in this build; refusing to reinterpret"))
    sizeof(T) * 8 == bits || throw(UpdateScalarError(:width_mismatch,
        "recorded type $(name) is $(8 * sizeof(T)) bits here, payload claims $(bits)"))
    T
end

"""
    session_payload_scalar(p; mod=Main) -> scalar

Exact inverse of [`session_scalar_payload`](@ref).
"""
function session_payload_scalar(p::SessionScalarPayload; mod::Module=Main)
    if p.kind === :ieee_bits
        T = session_resolve_type(mod, p.type_name, p.type_bits)
        T <: Base.IEEEFloat || throw(UpdateScalarError(:bad_number,
            "payload $(p.type_name) is not an IEEE float"))
        U = _ieee_uint_type(T)
        expected = 2 * sizeof(T)
        length(p.hex) == expected || throw(UpdateScalarError(:bad_number,
            "ieee payload for $(p.type_name) has $(length(p.hex)) hex digits, want $(expected)"))
        v = try
            parse(U, p.hex, base=16)
        catch err
            throw(UpdateScalarError(:bad_number, "payload $(repr(p.hex)) is not hex: $(err)"))
        end
        return reinterpret(T, v)
    elseif p.kind === :bigfloat_string
        text = String(_hex_to_bytes(p.hex))
        y = setprecision(BigFloat, p.type_bits) do
            parse(BigFloat, text)
        end
        precision(y) == p.type_bits || throw(UpdateScalarError(:width_mismatch,
            "decoded BigFloat has $(precision(y)) bits, payload claims $(p.type_bits)"))
        # The round-trip property is CHECKED, not assumed: a payload whose text
        # does not re-print identically is refused rather than trusted.
        string(y) == text || throw(UpdateScalarError(:not_round_trippable,
            "BigFloat payload $(repr(text)) does not round-trip at $(p.type_bits) bits"))
        return y
    elseif p.kind === :isbits_bytes
        T = session_resolve_type(mod, p.type_name, p.type_bits)
        bytes = _hex_to_bytes(p.hex)
        length(bytes) == sizeof(T) || throw(UpdateScalarError(:width_mismatch,
            "payload has $(length(bytes)) bytes, $(p.type_name) is $(sizeof(T))"))
        return only(reinterpret(T, bytes))
    end
    throw(UpdateScalarError(:unsupported_scalar_type, "unknown payload kind $(p.kind)"))
end

# ---------------------------------------------------------------------------
# 4. digests — pattern and values are different facts
# ---------------------------------------------------------------------------

const SESSION_DIGEST_SEED = 0xcbf29ce484222325 % UInt64
const SESSION_DIGEST_PRIME = 0x100000001b3 % UInt64

"""
    session_digest_mix(h, bytes) -> UInt64

FNV-1a over an explicit byte string.  A digest is a *mismatch detector here, not
a security primitive*, and it is used only to decide "same or not same".
"""
function session_digest_mix(h::UInt64, bytes)
    for b in bytes
        h = (h ⊻ UInt64(b)) * SESSION_DIGEST_PRIME % UInt64
    end
    h
end

session_digest_mix(h::UInt64, x::Integer) = session_digest_mix(h, codeunits(string(x)))
session_digest_mix(h::UInt64, s::Symbol) = session_digest_mix(h, codeunits(String(s)))
session_digest_mix(h::UInt64, s::AbstractString) = session_digest_mix(h, codeunits(s))

function session_digest(parts...)
    h = SESSION_DIGEST_SEED
    for p in parts
        h = session_digest_mix(h, p)
    end
    h
end

"""
    session_values_digest(A) -> UInt64

Digest of the EXACT arithmetic values of `A`, column-major, through
[`session_scalar_payload`](@ref).  Two matrices with the same digest are the
same values in the same arithmetic; `0.0` and `-0.0` are different inputs and
get different digests.
"""
function session_values_digest(A::AbstractMatrix)
    h = SESSION_DIGEST_SEED
    h = session_digest_mix(h, :values)
    h = session_digest_mix(h, size(A, 1))
    h = session_digest_mix(h, size(A, 2))
    for j in axes(A, 2), i in axes(A, 1)
        p = session_scalar_payload(A[i, j])
        h = session_digest_mix(h, p.kind === :ieee_bits ? :ieee_bits : Symbol(p.type_name))
        h = session_digest_mix(h, p.hex)
    end
    h
end

function session_values_digest(v::AbstractVector)
    h = SESSION_DIGEST_SEED
    h = session_digest_mix(h, :values_vector)
    h = session_digest_mix(h, length(v))
    for i in eachindex(v)
        p = session_scalar_payload(v[i])
        h = session_digest_mix(h, p.hex)
    end
    h
end

"""
    session_pattern_digest(A) -> UInt64

The STRUCTURAL pattern, which is a different fact from the values.

  * A dense `AbstractMatrix` has no structural zeros to report — every cell is
    present in storage — so its pattern is its dimensions and an explicit
    `:dense_all_present` tag.  Two dense matrices of the same shape therefore
    have the same pattern digest *by definition*, which is exactly why a
    value change on a dense operator must still revoke rank authority.
  * A `SparseMatrixCSC` reports `colptr`/`rowval`, which is its structure and
    not its values.
"""
function session_pattern_digest(A::AbstractMatrix)
    h = SESSION_DIGEST_SEED
    h = session_digest_mix(h, :pattern_dense)
    h = session_digest_mix(h, size(A, 1))
    session_digest_mix(h, size(A, 2))
end

function session_pattern_digest(A::SparseArrays.AbstractSparseMatrixCSC)
    h = SESSION_DIGEST_SEED
    h = session_digest_mix(h, :pattern_csc)
    h = session_digest_mix(h, size(A, 1))
    h = session_digest_mix(h, size(A, 2))
    for c in A.colptr
        h = session_digest_mix(h, c)
    end
    for r in A.rowval
        h = session_digest_mix(h, r)
    end
    h
end

# ---------------------------------------------------------------------------
# 5. the tolerance object
#
# "同容差" is enforced structurally: a tolerance is an immutable object that a
# caller passes to BOTH the fresh and the updated comparison, and
# `session_tolerance_strictest` can only ever make a bound TIGHTER.  There is
# deliberately no `relax`, and no tolerance field on `UpdateEffect` or
# `UpdatePlan`, so the update path has nowhere to store a private looser bound.
# ---------------------------------------------------------------------------

"""
    SessionTolerance

A named (absolute, relative) bound together with the exact textual literal it
was written as.  `atol`/`rtol` are Float64 because a tolerance IS a Float64
threshold; the literals are carried so a replay can prove it compared with the
same bound and not merely a numerically similar one.
"""
struct SessionTolerance
    name::Symbol
    atol::Float64
    rtol::Float64
    atol_literal::String
    rtol_literal::String
end

SessionTolerance(name::Symbol, atol::Real, rtol::Real) =
    SessionTolerance(name, Float64(atol), Float64(rtol), string(atol), string(rtol))

"""
    session_tolerance_strictest(a, b) -> SessionTolerance

The tighter of two bounds, elementwise.  Never looser than either input.
"""
function session_tolerance_strictest(a::SessionTolerance, b::SessionTolerance)
    SessionTolerance(Symbol(a.name, :_, b.name), min(a.atol, b.atol), min(a.rtol, b.rtol),
                     string(min(a.atol, b.atol)), string(min(a.rtol, b.rtol)))
end

session_tolerance_matches(t::SessionTolerance, atol::Real, rtol::Real) =
    t.atol == Float64(atol) && t.rtol == Float64(rtol)

session_tolerance_admits(t::SessionTolerance, abs_err::Real, rel_err::Real) =
    abs_err <= t.atol || rel_err <= t.rtol

"""
    session_error_measures(observed, reference) -> NamedTuple

The ONE comparison instrument used for both the updated and the fresh result.
Both paths are measured by this function with the same tolerance object, so the
comparison cannot be stricter for one path than the other by accident.
"""
function session_error_measures(observed::AbstractArray, reference::AbstractArray)
    length(observed) == length(reference) || throw(ArgumentError(
        "error measures compare equal shapes; got $(size(observed)) and $(size(reference))"))
    max_abs = 0.0
    max_rel = 0.0
    for (o, r) in zip(observed, reference)
        d = abs(Float64(o) - Float64(r))
        max_abs = max(max_abs, d)
        denom = abs(Float64(r))
        max_rel = max(max_rel, denom == 0.0 ? (d == 0.0 ? 0.0 : Inf) : d / denom)
    end
    (max_abs=max_abs, max_rel=max_rel, n=length(observed))
end

# ---------------------------------------------------------------------------
# 6. the fingerprint
# ---------------------------------------------------------------------------

"""
    ProblemFingerprint

Everything the update table is allowed to look at.  `values` and `pattern` are
separate digests on purpose: the whole point of the table is that "same pattern"
and "same values" are different questions.
"""
struct ProblemFingerprint
    n::Int
    m::Int
    pattern::UInt64
    values::UInt64
    objective::UInt64
    rhs::UInt64
    cone_parameters::UInt64
    arithmetic::ArithmeticFamily
    precision_bits::Int
    rounding::SessionRounding
    ordering::Symbol
    rank_transform::Symbol
end

"""
    session_problem_fingerprint(; A, b, c, cone_parameters, arithmetic, ...)

Build a fingerprint.  `nothing` is a *different* fact from "all zeros", and the
digest domains say so: an absent `b` and a zero `b` must not compare equal.
"""
function session_problem_fingerprint(;
        A::Union{Nothing,AbstractMatrix}=nothing,
        b::Union{Nothing,AbstractVector}=nothing,
        c::Union{Nothing,AbstractVector}=nothing,
        cone_parameters::Union{Nothing,AbstractVector}=nothing,
        arithmetic::ArithmeticFamily=ArithFloat,
        precision_bits::Integer=64,
        rounding::SessionRounding=RoundingNearestEven,
        ordering::Symbol=:natural,
        rank_transform::Symbol=:full,
        n::Integer=A === nothing ? 0 : size(A, 1),
        m::Integer=A === nothing ? 0 : size(A, 2))
    ProblemFingerprint(
        Int(n), Int(m),
        A === nothing ? session_digest(:absent_operator) : session_pattern_digest(A),
        A === nothing ? session_digest(:absent_operator) : session_values_digest(A),
        c === nothing ? session_digest(:absent_objective) : session_values_digest(c),
        b === nothing ? session_digest(:absent_rhs) : session_values_digest(b),
        cone_parameters === nothing ? session_digest(:absent_cone_parameters) :
            session_values_digest(cone_parameters),
        arithmetic, Int(precision_bits), rounding, ordering, rank_transform,
    )
end

"""
    session_classify_changes(before, after) -> Vector{ProblemChange}

Exactly the changes between two fingerprints, in canonical order.  An empty
vector means the two fingerprints were identical, and the caller gets
`[ChangeNone]` so that "nothing changed" is a named row of the table rather than
a silently skipped call.
"""
function session_classify_changes(before::ProblemFingerprint, after::ProblemFingerprint)
    out = ProblemChange[]
    before.pattern == after.pattern || push!(out, ChangeOperatorPattern)
    before.values == after.values || push!(out, ChangeOperatorValues)
    before.n == after.n && before.m == after.m || push!(out, ChangeOperatorPattern)
    before.objective == after.objective || push!(out, ChangeObjective)
    before.rhs == after.rhs || push!(out, ChangeRHS)
    before.cone_parameters == after.cone_parameters || push!(out, ChangeConeParameters)
    before.arithmetic == after.arithmetic || push!(out, ChangePrecision)
    before.precision_bits == after.precision_bits || push!(out, ChangePrecision)
    before.rounding == after.rounding || push!(out, ChangeRounding)
    before.ordering == after.ordering || push!(out, ChangeOrdering)
    before.rank_transform == after.rank_transform || push!(out, ChangeRankTransform)
    unique!(out)
    sort!(out; by=c -> SESSION_CHANGE_ORDER[c])
    isempty(out) && return ProblemChange[ChangeNone]
    out
end

# ---------------------------------------------------------------------------
# 7. the table
# ---------------------------------------------------------------------------

"""
    UpdateEffect

One row of the invalidation table, folded over a change set.

The six booleans and the authority label are INDEPENDENT fields because they are
independent facts.  In particular:

  * `numeric_factor_reusable` — the physical factor still answers the same
    question.  False for any operator change.
  * `warm_start_allowed` — the *iterate* may seed the next solve.  This is a
    solver-level decision that the table does not forbid; it says nothing about
    the factor.
  * `workspace_reusable` — the direction/line-search workspace still has the
    right shape and the right cone scaling.  A cone-parameter change keeps the
    factor and destroys the workspace.
"""
struct UpdateEffect
    changes::Vector{ProblemChange}
    symbolic_reusable::Bool
    numeric_factor_reusable::Bool
    refactor_required::Bool
    rank_authority::RankAuthority
    certificate_valid::Bool
    readmission_required::Bool
    warm_start_allowed::Bool
    workspace_reusable::Bool
    recompute::RecomputeLevel
end

"""
    update_effect_for_change(c) -> UpdateEffect

The single-change rows.  This is the table; `update_table_audit()` reads it.
"""
function update_effect_for_change(c::ProblemChange)
    if c === ChangeNone
        return UpdateEffect([ChangeNone], true, true, false, RankAuthorityRetained,
                            true, false, true, true, RecomputeNothing)
    elseif c === ChangeObjective
        # `c` is not an input to the factor.  The factor stands; the certificate
        # does not, because the duality gap is measured against `c`.
        return UpdateEffect([c], true, true, false, RankAuthorityRetained,
                            false, false, true, true, RecomputeNothing)
    elseif c === ChangeRHS
        # `b` is not an input to the factor either.  A solve under a new `b` is
        # not a reuse of the old *solution*, which is why warm start is a
        # separate — and here still permitted — decision.
        return UpdateEffect([c], true, true, false, RankAuthorityRetained,
                            false, false, true, true, RecomputeNothing)
    elseif c === ChangeConeParameters
        # The operator is untouched, so the factor and its rank stand.  The cone
        # scaling and every workspace sized from it do not.
        return UpdateEffect([c], true, true, false, RankAuthorityRetained,
                            false, false, true, false, RecomputeNothing)
    elseif c === ChangeOperatorValues
        # SAME PATTERN, DIFFERENT NUMBERS.  Symbolic structure survives; the
        # numeric factor and the rank authority do not.
        return UpdateEffect([c], true, false, true, RankAuthorityRevoked,
                            false, false, true, true, RecomputeFactorOnly)
    elseif c === ChangeOperatorPattern
        return UpdateEffect([c], false, false, true, RankAuthorityRevoked,
                            false, true, true, false, RecomputeSymbolic)
    elseif c === ChangePrecision
        return UpdateEffect([c], false, false, true, RankAuthorityRevoked,
                            false, true, true, false, RecomputeFull)
    elseif c === ChangeRounding
        # The sparsity pattern is a property of the indices, so the symbolic
        # structure *could* survive; the arithmetic contract the factor was
        # produced under did not, so the numeric factor and the rank do not.
        return UpdateEffect([c], true, false, true, RankAuthorityRevoked,
                            false, true, true, true, RecomputeFactorOnly)
    elseif c === ChangeOrdering
        # A fill-reducing ordering IS the symbolic factorization.
        return UpdateEffect([c], false, false, true, RankAuthorityRevoked,
                            false, true, true, false, RecomputeSymbolic)
    elseif c === ChangeRankTransform
        return UpdateEffect([c], false, false, true, RankAuthorityRevoked,
                            false, true, true, false, RecomputeSymbolic)
    end
    throw(ArgumentError("no table row for $(c)"))
end

"""
    update_effect(changes) -> UpdateEffect

Fold a change set.  The fold rules are the conservative ones: a requirement
imposed by ANY change in the set is imposed on the whole update, so a cheap
change can never average away an expensive one.
"""
function update_effect(changes::AbstractVector{ProblemChange})
    rows = [update_effect_for_change(c) for c in
            (isempty(changes) ? ProblemChange[ChangeNone] : changes)]
    length(rows) == 1 && return rows[1]
    UpdateEffect(
        sort!(unique!(reduce(vcat, [r.changes for r in rows]));
              by=c -> SESSION_CHANGE_ORDER[c]),
        all(r -> r.symbolic_reusable, rows),
        all(r -> r.numeric_factor_reusable, rows),
        any(r -> r.refactor_required, rows),
        all(r -> rank_authority_quotable(r.rank_authority), rows) ?
            RankAuthorityRetained : RankAuthorityRevoked,
        all(r -> r.certificate_valid, rows),
        any(r -> r.readmission_required, rows),
        all(r -> r.warm_start_allowed, rows),
        all(r -> r.workspace_reusable, rows),
        maximum(r.recompute for r in rows),
    )
end

update_effect(before::ProblemFingerprint, after::ProblemFingerprint) =
    update_effect(session_classify_changes(before, after))

"""
    update_rank_rule_holds(e) -> Bool

The card's rule, as a function that can fail: a change that can change the rank
must not leave the previous rank authorised.
"""
function update_rank_rule_holds(e::UpdateEffect)
    any(c -> c in SESSION_RANK_REVOKING_CHANGES, e.changes) || return true
    !rank_authority_quotable(e.rank_authority)
end

"""
    update_table_audit() -> NamedTuple

A runnable audit of the table: one row per change with its effects, plus the
invariant verdicts and the witnesses that the three reuse axes really are three
axes.  A table is only "explicit" if it can be printed and checked.
"""
function update_table_audit()
    rows = NamedTuple[]
    for c in (ChangeNone, SESSION_PROBLEM_CHANGES...)
        e = update_effect_for_change(c)
        push!(rows, (change=change_label(c),
                     symbolic_reusable=e.symbolic_reusable,
                     numeric_factor_reusable=e.numeric_factor_reusable,
                     refactor_required=e.refactor_required,
                     rank_authority=Symbol(lowercase(string(e.rank_authority))),
                     certificate_valid=e.certificate_valid,
                     readmission_required=e.readmission_required,
                     warm_start_allowed=e.warm_start_allowed,
                     workspace_reusable=e.workspace_reusable,
                     recompute=Symbol(lowercase(string(e.recompute))),
                     rank_rule_holds=update_rank_rule_holds(e)))
    end
    # Witnesses for "three axes, not one switch".
    witnesses = [r.change for r in rows
                 if !(r.numeric_factor_reusable == r.workspace_reusable ==
                      r.warm_start_allowed)]
    (rows=rows, n_rows=length(rows),
     all_rank_rules_hold=all(r -> r.rank_rule_holds, rows),
     n_independent_axis_witnesses=length(witnesses),
     axis_witnesses=witnesses)
end

# ---------------------------------------------------------------------------
# 8. the plan, and applying it to the logical lease
# ---------------------------------------------------------------------------

"""
    UpdatePlan

A classified update with the action it licenses.  Note what is NOT a field: a
tolerance, a warm-start vector, and a factor.  There is nowhere here to store a
private looser bound or a borrowed factor.
"""
struct UpdatePlan
    before::ProblemFingerprint
    after::ProblemFingerprint
    changes::Vector{ProblemChange}
    effect::UpdateEffect
    revoke_before_refactor::Bool
    detail::String
end

"""
    update_plan(before, after) -> UpdatePlan

`revoke_before_refactor` is true whenever the update touches authority at all.
It is not merely a decoration: `refactor_numeric!` returns `revoked=false` when
its own admission refuses the (possibly mutated) request, so a caller that
waits for the refactor's verdict to decide whether to revoke can be left holding
a bound lease.  The driver measures that path with a control; see the S07 log
section "lease hazard".
"""
function update_plan(before::ProblemFingerprint, after::ProblemFingerprint)
    changes = session_classify_changes(before, after)
    effect = update_effect(changes)
    revoke = effect.refactor_required || effect.readmission_required ||
             !effect.numeric_factor_reusable || !effect.symbolic_reusable
    UpdatePlan(before, after, changes, effect, revoke,
               join(string.(change_label.(changes)), ","))
end

"""
    UpdateOutcome

What an applied plan actually did, including the case where it refused to do
anything.  `applied` is not `ok`: a plan that only had to revoke authority is
"applied" with `ok = false` and no refactor.
"""
struct UpdateOutcome
    changes::Vector{ProblemChange}
    effect::UpdateEffect
    revoked::Bool
    lease_state_after::LeaseState
    rank_authority::RankAuthority
    refactor_required::Bool
    applied::Bool
    detail::String
end

"""
    apply_update!(plan, handle::FactorHandle) -> UpdateOutcome

Mutate the ADR-002 §4 logical lease for a planned update, BEFORE any provider
interaction.  Ordering is the point:

  1. if the plan touches authority, `revoke!` runs FIRST;
  2. the cached summary is invalidated, so no caller can quote a rank or a
     status that belongs to the previous operator;
  3. no provider status is read at all.  ADR-002 §4: the provider's retained
     success flag is evidence, never authorization.

This is the HANDLE-level primitive: it knows nothing about a
[`PreparedUpdateState`](@ref) that may be pointing at the same handle.  Use
[`apply_update!(plan, state)`](@ref) when you hold the state — the driver
measures why (a handle-only call leaves the state's own authority flag set,
which is a real gap that `session_rank` closes by consulting the lease).
"""
function apply_update!(plan::UpdatePlan, handle::FactorHandle)
    if plan.revoke_before_refactor
        revoke!(handle, "prepared update: $(plan.detail)")
    end
    handle.summary_valid = false
    UpdateOutcome(plan.changes, plan.effect, plan.revoke_before_refactor,
                  handle.lease.state, plan.effect.rank_authority,
                  plan.effect.refactor_required, true,
                  plan.revoke_before_refactor ?
                      "lease revoked before any provider interaction" :
                      "operator untouched; factor and rank authority retained")
end

"""
    apply_update!(plan, state::PreparedUpdateState) -> UpdateOutcome

The state-level entry point: revoke the lease AND drop the state's rank
authority, so the two objects that together decide "may I quote a rank" cannot
disagree.  Defined after `PreparedUpdateState` (a method signature is evaluated
at definition time, so it cannot be written above the struct).
"""

# ---------------------------------------------------------------------------
# 9. the prepared update state
# ---------------------------------------------------------------------------

"""
    PreparedUpdateState

Ties a `FactorHandle` to the fingerprint of the problem it was prepared for, and
keeps the rank authority explicit.  `rank_answer` is only ever produced through
[`session_rank`](@ref), which refuses to answer while the authority is revoked or
while the summary belongs to a previous provider generation.
"""
mutable struct PreparedUpdateState
    handle::FactorHandle
    fingerprint::ProblemFingerprint
    rank_authority::RankAuthority
    rank_generation::UInt64
    n_plans::Int
    n_refactors::Int
    n_failed_refactors::Int
    last_detail::String
end

function session_prepared_state(handle::FactorHandle, fp::ProblemFingerprint)
    PreparedUpdateState(handle, fp, RankAuthorityNone, UInt64(0), 0, 0, 0, "")
end

"""
    apply_update!(plan, state::PreparedUpdateState) -> UpdateOutcome

The state-level entry point: revoke the lease AND drop the state's rank
authority, so the two objects that together decide "may I quote a rank" cannot
disagree.  Defined here rather than beside the handle method because a method
signature is evaluated at definition time and `PreparedUpdateState` does not
exist yet at that point in the file.
"""
function apply_update!(plan::UpdatePlan, state::PreparedUpdateState)
    outcome = apply_update!(plan, state.handle)
    state.rank_authority = RankAuthorityRevoked
    state.last_detail = plan.detail
    outcome
end

"""
    session_rank(state) -> NamedTuple

`available = false` with a reason is a complete answer.  `:stale_generation`
means the provider has advanced its factor since the summary was taken, which is
the second way a rank can be out of date even when the authority was never
formally revoked.
"""
function session_rank(state::PreparedUpdateState)
    handle = state.handle
    a = state.rank_authority
    rank_authority_quotable(a) || return (available=false, rank=nothing,
                                          source=Symbol(lowercase(string(a))))
    # The lease is the only authorization predicate in the system (ADR-002 §4),
    # so a rank answer must not survive a revoked lease even when this state's
    # own flag was not cleared — e.g. when `apply_update!` was called on the
    # handle rather than on the state. Found by this task's own driver.
    is_valid(handle.lease) || return (available=false, rank=nothing,
                                      source=:lease_not_bound)
    s = factor_summary(handle)
    s.generation == handle.provider_generation ||
        return (available=false, rank=nothing, source=:stale_generation)
    s.numeric_epoch == state.rank_generation ||
        return (available=false, rank=nothing, source=:stale_generation)
    (available=true, rank=s.rank, source=Symbol(lowercase(string(a))))
end

"""
    session_update!(state, after, values; request=nothing) -> NamedTuple

The prepared update path.

  * `request` is REQUIRED when the plan demands re-admission (precision,
    rounding, ordering, rank transform, pattern).  There is no default: the
    handle's stored request is the OLD request, and silently continuing with it
    is the "second authority" this table exists to prevent.  A plan that needs a
    new request and does not get one is refused with `stage = :request_required`.
  * On any refactor failure the lease stays revoked and the rank authority
    stays revoked.  Nothing in this function consults provider status.
"""
function session_update!(state::PreparedUpdateState, after::ProblemFingerprint, values;
                         request::Union{Nothing,FactorRequest}=nothing)
    handle = state.handle
    plan = update_plan(state.fingerprint, after)
    state.n_plans += 1
    outcome = apply_update!(plan, handle)
    state.rank_authority = RankAuthorityRevoked
    state.last_detail = plan.detail

    if plan.effect.readmission_required && request === nothing
        state.last_detail = "plan requires re-admission but no request was supplied"
        return (ok=false, stage=:request_required, outcome=outcome, plan=plan,
                status=nothing, generation=handle.provider_generation, detail=state.last_detail)
    end
    if request !== nothing
        # The fingerprint after the update claims a different request shape; the
        # handle must carry exactly that request, not the old one.
        handle.request = request
        prep = prepare_factor!(handle)
        if !prep.allowed
            state.last_detail = "re-admission refused: $(prep.detail)"
            return (ok=false, stage=:not_admitted, outcome=outcome, plan=plan,
                    admission=prep, status=nothing,
                    generation=handle.provider_generation, detail=state.last_detail)
        end
    end

    if !plan.effect.refactor_required
        # Nothing to refactor: the factor still answers the same operator.
        state.rank_authority = plan.effect.rank_authority
        state.rank_generation = handle.provider_generation
        state.fingerprint = after
        return (ok=true, stage=:reused, outcome=outcome, plan=plan, status=StatusOk,
                generation=handle.provider_generation, detail=state.last_detail)
    end

    r = refactor_numeric!(handle, values)
    state.n_refactors += 1
    if !r.ok
        # Belt and braces: `refactor_numeric!` already revokes on a commit
        # failure, but a non-admitted request returns `revoked = false` while a
        # previous factor may still be bound (measured; see the S07 log). The
        # update path therefore does not depend on that return value.
        is_valid(handle.lease) && revoke!(handle, "prepared update refactor failed: $(r.detail)")
        state.n_failed_refactors += 1
        state.rank_authority = RankAuthorityRevoked
        state.last_detail = "refactor failed: $(r.detail)"
        return (ok=false, stage=:refactor_failed, outcome=outcome, plan=plan,
                status=r.status, admission=r.admission,
                generation=handle.provider_generation, detail=state.last_detail)
    end
    state.fingerprint = after
    state.rank_authority = RankAuthorityFromNewFactor
    state.rank_generation = handle.provider_generation
    (ok=true, stage=:refactored, outcome=outcome, plan=plan, status=r.status,
     admission=r.admission, generation=handle.provider_generation, detail="")
end

"""
    session_prepare!(state, request, values) -> NamedTuple

The fresh-setup path through the same objects, so the driver can compare an
updated run against a fresh one with the same instrument and the same tolerance.
"""
function session_prepare!(state::PreparedUpdateState, request::FactorRequest, values)
    handle = state.handle
    handle.request = request
    prep = prepare_factor!(handle)
    prep.allowed || return (ok=false, stage=:not_admitted, admission=prep,
                            status=nothing, generation=handle.provider_generation,
                            detail=prep.detail)
    r = refactor_numeric!(handle, values)
    state.n_refactors += 1
    if !r.ok
        state.n_failed_refactors += 1
        state.rank_authority = RankAuthorityRevoked
        is_valid(handle.lease) && revoke!(handle, "fresh prepare refactor failed: $(r.detail)")
        return (ok=false, stage=:refactor_failed, admission=r.admission, status=r.status,
                generation=handle.provider_generation, detail=r.detail)
    end
    state.rank_authority = RankAuthorityFromNewFactor
    state.rank_generation = handle.provider_generation
    (ok=true, stage=:refactored, admission=r.admission, status=r.status,
     generation=handle.provider_generation, detail="")
end
