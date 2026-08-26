"""
Build-first matrix-bootstrap models.

This file deliberately has no dependency on the SDP solver.  It constructs a
conservative, exact-rational affine SDP artifact from the low-order O(2) word
algebra used by Lin--Zheng.  `to_sdp_problem` is an optional adapter for callers
that have already loaded SDPX.

The primary model is the D=2 bosonic model in arXiv:2507.21007.  The complete
production quotient (all indexed irreps, momentum commutator reductions, and
the authors' free-variable basis) is not published as a machine-readable data
set.  Consequently this generator is an independently checkable low-order
relaxation and accepts only `reference_status = :build_only`.  It never claims
to reproduce the published level census, objective bounds, or numerical
reference values.
"""
module MatrixBootstrap

using LinearAlgebra
using SparseArrays
using SHA

export Rat, Word, LETTER_NAMES, LETTER_CHARGE, LETTER_LEVEL
export word, word_string, word_charge, word_level, word_dagger
export canonical_trace_word, canonical_moment, affine_moment_reduction, enumerate_words
export AffineMomentReduction
export AffineTerm, AffineEntry, AffineVariable, SparseAffineRow, PSDBlock
export MatrixBootstrapSpec, SDPSliceArtifact
export build_matrix_bootstrap, build_lin_zheng, build_kazakov_zheng_lambda4
export build_kz_lambda4, to_sdp_problem, affine_matrix, evaluate_affine
export moment_value, eom_terms, eom_residual, published_counts, published_intervals
export validate_artifact

const Rat = Rational{BigInt}
const Word = Tuple{Vararg{UInt8}}

"Letters are ordered as in the O(2) complex basis used by the paper."
const LETTER_NAMES = (:Z, :Zbar, :Pi, :Pibar)
const LETTER_CHARGE = (1, -1, 1, -1)
const LETTER_LEVEL = (1, 1, 2, 2)
const LETTER_DAGGER = UInt8[2, 1, 4, 3]
const LETTER_MOMENTUM = (false, false, true, true)

const _LIN_SOURCE = :lin_zheng_o2
const _LIN_VERSION = "arXiv:2507.21007v3"
const _LIN_STATUS = :build_only
const _KZ_SOURCE = :kazakov_zheng_lambda4
const _KZ_VERSION = "arXiv:2108.04830 Appendix E (Lambda=4 relaxation)"
const _KZ_STATUS = :cross_check_only
const _SUPPORTED_LEVEL_SCOPES = Dict(4 => :tiny, 6 => :small, 8 => :medium)

const _PUBLISHED_FREE = Dict(
    4 => 3, 6 => 8, 8 => 22, 10 => 77, 12 => 326, 14 => 1569,
)
const _PUBLISHED_ALL = Dict(
    4 => 14, 6 => 94, 8 => 614, 10 => 4086, 12 => 27830,
    14 => 192374,
)
const _PUBLISHED_INTERVALS = Dict(
    (:massless, :energy) => (Rat(707832, 1_000_000), Rat(707868, 1_000_000)),
    (:massless, :x2) => (Rat(1_15420, 100_000), Rat(1_15460, 100_000)),
    (:massless, :z2barz2) => (Rat(37055, 100_000), Rat(37085, 100_000)),
    (:massive, :energy) => (Rat(1_172098376, 1_000_000_000), Rat(1_172098408, 1_000_000_000)),
    (:massive, :x2) => (Rat(77800898, 100_000_000), Rat(77800934, 100_000_000)),
    (:massive, :z2barz2) => (Rat(15850588, 100_000_000), Rat(15850607, 100_000_000)),
)

published_counts() = (free=copy(_PUBLISHED_FREE), all=copy(_PUBLISHED_ALL))
published_intervals() = copy(_PUBLISHED_INTERVALS)

_rat(x::Rat) = x
_rat(x::Integer) = Rat(BigInt(x), BigInt(1))
_rat(x::Rational) = Rat(BigInt(numerator(x)), BigInt(denominator(x)))
_rat(x::AbstractFloat) = rationalize(BigInt, x)
_rat(x) = throw(ArgumentError("expected an exact integer/rational scan parameter, got $(typeof(x))"))

_word(x::Word) = x
_word(x::Tuple) = Tuple(UInt8.(x))
_word(x::AbstractVector{<:Integer}) = Tuple(UInt8.(x))

function _concat_words(ws::Word...)
    out = UInt8[]
    for w in ws
        append!(out, w)
    end
    return Tuple(out)
end

"Construct a word from letter names or integer letter IDs."
function word(xs...)
    isempty(xs) && return Word(())
    return Tuple(UInt8[_letter_id(x) for x in xs])
end

function _letter_id(x::Integer)
    1 <= x <= length(LETTER_NAMES) || throw(ArgumentError("letter ID $x is outside 1:4"))
    return Int(x)
end
function _letter_id(x::UInt8)
    1 <= x <= length(LETTER_NAMES) || throw(ArgumentError("letter ID $x is outside 1:4"))
    return Int(x)
end
function _letter_id(x::Symbol)
    i = findfirst(==(x), LETTER_NAMES)
    i === nothing && throw(ArgumentError("unknown O(2) letter $x"))
    return i
end
function _letter_id(x::AbstractString)
    return _letter_id(Symbol(x))
end

word_string(w::Word) = isempty(w) ? "1" : join((string(LETTER_NAMES[Int(x)]) for x in w), " ")
word_string(w::Tuple) = word_string(_word(w))

function word_charge(w::Word)
    q = 0
    @inbounds for x in w
        q += LETTER_CHARGE[Int(x)]
    end
    return q
end
word_charge(w::Tuple) = word_charge(_word(w))

function word_level(w::Word)
    ell = 0
    @inbounds for x in w
        ell += LETTER_LEVEL[Int(x)]
    end
    return ell
end
word_level(w::Tuple) = word_level(_word(w))

function _word_reverse(w::Word)
    return Tuple(w[i] for i in length(w):-1:1)
end

"The complex Hermitian adjoint, including the sign from Pi=-iP."
function word_dagger(w::Word)
    isempty(w) && return (Word(()), Rat(1))
    out = Vector{UInt8}(undef, length(w))
    s = Rat(1)
    @inbounds for i in eachindex(w)
        x = w[length(w) - i + 1]
        out[i] = LETTER_DAGGER[Int(x)]
        LETTER_MOMENTUM[Int(x)] && (s = -s)
    end
    return (Tuple(out), s)
end
word_dagger(w::Tuple) = word_dagger(_word(w))
word_dagger(w::AbstractVector{<:Integer}) = word_dagger(_word(w))

function _reflection(w::Word)
    return Tuple(LETTER_DAGGER[Int(x)] for x in w)
end

function _rotations(w::Word)
    n = length(w)
    n == 0 && return Word[Word(())]
    out = Word[]
    sizehint!(out, n)
    for i in 1:n
        push!(out, _concat_words(Tuple(w[i:end]), Tuple(w[1:i-1])))
    end
    return out
end

"Return the smallest formal cyclic/reversal/O(2) word representative."
function canonical_trace_word(w::Word; reflection::Bool=true, reversal::Bool=true)
    candidates = _rotations(w)
    reflection && append!(candidates, _rotations(_reflection(w)))
    if reversal
        reversed = _word_reverse(w)
        append!(candidates, _rotations(reversed))
        reflection && append!(candidates, _rotations(_reflection(reversed)))
    end
    sort!(candidates)
    return first(candidates)
end
canonical_trace_word(w::Tuple; kwargs...) = canonical_trace_word(_word(w); kwargs...)
canonical_trace_word(w::AbstractVector{<:Integer}; kwargs...) = canonical_trace_word(_word(w); kwargs...)

"""
    canonical_moment(w; reflection=true) -> (representative, sign)

Canonicalize a physical single-trace moment.  Coordinate-only words use trace
cyclicity, O(2) reflection, and reversal.  A word containing `Pi` or `Pibar`
is *not* cyclically rotated: matrix entries are quantum operators and moving a
momentum through the trace produces canonical-commutator terms.  Such words
use only O(2) reflection and the signed reversal implied by Hermiticity and
time reversal.  A zero sign is returned only when the same ordered orbit is
genuinely self-reversing with both signs.
"""
function canonical_moment(w::Word; reflection::Bool=true)
    isempty(w) && return (Word(()), Rat(1))
    nmom = count(x -> LETTER_MOMENTUM[Int(x)], w)
    nmom == 0 && return (
        canonical_trace_word(w; reflection=reflection, reversal=true), Rat(1))

    revsign = isodd(nmom) ? Rat(-1) : Rat(1)
    candidates = Tuple{Word,Rat}[]
    push!(candidates, (w, Rat(1)))
    reflection && push!(candidates, (_reflection(w), Rat(1)))
    reversed = _word_reverse(w)
    push!(candidates, (reversed, revsign))
    reflection && push!(candidates, (_reflection(reversed), revsign))
    sort!(candidates, by=first)
    representative = first(candidates)[1]
    # If the same orbit representative is reached with both signs, the
    # Hermiticity/time-reversal constraints force that real correlator to be
    # exactly zero.  Do not silently choose whichever candidate happened to
    # sort first; that would make odd-momentum loop identities unsound.
    signs = Rat[candidate[2] for candidate in candidates if candidate[1] == representative]
    first_sign = first(signs)
    any(sign != first_sign for sign in signs) && return (representative, Rat(0))
    return (representative, first_sign)
end
canonical_moment(w::Tuple; kwargs...) = canonical_moment(_word(w); kwargs...)
canonical_moment(w::AbstractVector{<:Integer}; kwargs...) = canonical_moment(_word(w); kwargs...)

"An exact affine reduction `constant + coefficient * moment[representative]`."
struct AffineMomentReduction
    constant::Rat
    representative::Union{Nothing,Word}
    coefficient::Rat
    relation::Symbol
end

"""
    affine_moment_reduction(w) -> AffineMomentReduction

Apply the low-order affine kinematic quotient.  Besides normalization and
signed ordered-word canonicalization, this implements Appendix-E E34 in the
D=2 complex basis.  From `[Z,Pibar]=1` and time reversal,
`<tr Z Pibar>=1/2` and `<tr Pibar Z>=-1/2` (and likewise for the reflected
pair).  Longer momentum commutators generally generate factorized multi-trace
terms and are intentionally left as distinct ordered moments.
"""
function affine_moment_reduction(w::Word)
    representative, sign = canonical_moment(w)
    iszero(sign) && return AffineMomentReduction(Rat(0), nothing, Rat(0), :time_reversal_zero)
    representative == Word(()) &&
        return AffineMomentReduction(sign, nothing, Rat(0), :normalization)
    if representative == word(:Z, :Pibar)
        return AffineMomentReduction(sign * Rat(1, 2), nothing, Rat(0),
                                     :canonical_commutator_E34)
    end
    return AffineMomentReduction(Rat(0), representative, sign, :ordered_moment)
end
affine_moment_reduction(w::Tuple) = affine_moment_reduction(_word(w))
affine_moment_reduction(w::AbstractVector{<:Integer}) = affine_moment_reduction(_word(w))

"Enumerate deterministic raw or trace words up to a weighted level."
function enumerate_words(max_level::Integer; charge=nothing, trace::Bool=true,
                         reflection::Bool=true, include_empty::Bool=true)
    max_level >= 0 || throw(ArgumentError("max_level must be nonnegative"))
    raw = Word[]
    include_empty && push!(raw, Word(()))
    function visit!(w::Word, ell::Int)
        for x in UInt8(1):UInt8(4)
            nextell = ell + LETTER_LEVEL[Int(x)]
            nextell > max_level && continue
            nw = _concat_words(w, (x,))
            if charge === nothing || word_charge(nw) == Int(charge)
                push!(raw, nw)
            end
            visit!(nw, nextell)
        end
    end
    visit!(Word(()), 0)
    if trace
        out = Word[]
        seen = Set{Word}()
        for w in raw
            charge !== nothing && word_charge(w) != Int(charge) && continue
            k, sign = canonical_moment(w; reflection=reflection)
            if !iszero(sign) && !(k in seen)
                push!(out, k)
                push!(seen, k)
            end
        end
    else
        out = unique(raw)
    end
    sort!(out, by=w -> (word_level(w), w))
    return out
end

struct AffineTerm
    variable::Int
    coefficient::Rat
end

struct AffineEntry
    constant::Rat
    terms::Vector{AffineTerm}
    function AffineEntry(constant::Rat=Rat(0), terms::AbstractVector{AffineTerm}=AffineTerm[])
        merged = Dict{Int,Rat}()
        for term in terms
            term.variable > 0 || throw(ArgumentError("affine variable IDs must be positive"))
            merged[term.variable] = get(merged, term.variable, Rat(0)) + term.coefficient
        end
        cleaned = AffineTerm[AffineTerm(i, merged[i]) for i in sort!(collect(keys(merged))) if !iszero(merged[i])]
        return new(constant, cleaned)
    end
end
AffineEntry(constant::Integer) = AffineEntry(_rat(constant), AffineTerm[])

struct AffineVariable
    index::Int
    name::String
    kind::Symbol
    word::Union{Nothing,Word}
    pair::Union{Nothing,Tuple{Int,Int}}
end

struct SparseAffineRow
    label::Symbol
    rhs::Rat
    columns::Vector{Int}
    values::Vector{Rat}
end

struct PSDBlock
    name::Symbol
    kind::Symbol
    basis::Vector{Word}
    entries::Matrix{AffineEntry}
    metadata::NamedTuple
end

Base.size(block::PSDBlock) = size(block.entries)
Base.size(block::PSDBlock, d::Integer) = size(block.entries, d)
block_dimension(block::PSDBlock) = size(block.entries, 1)

Base.@kwdef struct MatrixBootstrapSpec
    source::Symbol = _LIN_SOURCE
    source_version::String = _LIN_VERSION
    D::Int = 2
    mass2::Rat = Rat(1)
    level::Int = 4
    scan_observable::Symbol = :x2
    scan_value::Union{Nothing,Rat} = nothing
    reference_status::Symbol = _LIN_STATUS
    relaxation::Bool = true
end

struct SDPSliceArtifact
    spec::MatrixBootstrapSpec
    source::Symbol
    source_version::String
    model::Symbol
    level::Int
    parameters::NamedTuple
    words::Vector{Word}
    variables::Vector{AffineVariable}
    moment_indices::Dict{Word,Int}
    q_indices::Matrix{Int}
    equalities::SparseMatrixCSC{Rat,Int}
    rhs::Vector{Rat}
    equality_labels::Vector{Symbol}
    psd_blocks::Vector{PSDBlock}
    reference_status::Symbol
    objective::Nothing
    reference_objective::Nothing
    metadata::NamedTuple
    fingerprint::String
end

Base.eltype(::SDPSliceArtifact) = Rat
Base.size(a::SDPSliceArtifact) = (length(a.variables), length(a.rhs), length(a.psd_blocks))
Base.show(io::IO, a::SDPSliceArtifact) = print(io, "SDPSliceArtifact($(a.source), level=$(a.level), variables=$(length(a.variables)), equalities=$(length(a.rhs)), blocks=$(length(a.psd_blocks)), reference_status=$(a.reference_status), fingerprint=$(a.fingerprint[1:12])…)")

function _entry_moment(key::Word, coefficient::Rat, indices::Dict{Word,Int})
    reduction = affine_moment_reduction(key)
    constant = coefficient * reduction.constant
    value = coefficient * reduction.coefficient
    iszero(value) && return AffineEntry(constant)
    representative = reduction.representative
    representative === nothing && throw(ArgumentError("nonzero affine coefficient has no moment representative"))
    id = get(indices, representative, 0)
    id > 0 || throw(KeyError("moment $(word_string(representative)) is absent from the variable table"))
    return AffineEntry(constant, [AffineTerm(id, value)])
end

function _entry_q(id::Int, coefficient::Rat=Rat(1))
    return AffineEntry(Rat(0), [AffineTerm(id, coefficient)])
end

function _entry_sub(a::AffineEntry, b::AffineEntry)
    terms = vcat(a.terms, [AffineTerm(t.variable, -t.coefficient) for t in b.terms])
    return AffineEntry(a.constant - b.constant, terms)
end

function _entry_add(a::AffineEntry, b::AffineEntry)
    return AffineEntry(a.constant + b.constant, vcat(a.terms, b.terms))
end

function _moment_entry(w1::Word, w2::Word, indices::Dict{Word,Int})
    d, ds = word_dagger(w1)
    return _entry_moment(_concat_words(d, w2), ds, indices)
end

function affine_matrix(block::PSDBlock, ::Type{T}=Rat; values=nothing) where {T}
    n, m = size(block.entries)
    n == m || throw(ArgumentError("PSD affine block must be square"))
    maxvar = maximum((t.variable for e in block.entries for t in e.terms), init=0)
    values === nothing && (values = zeros(T, maxvar))
    out = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        e = block.entries[i, j]
        v = T(e.constant)
        for t in e.terms
            t.variable <= length(values) || throw(BoundsError(values, t.variable))
            v += T(t.coefficient) * values[t.variable]
        end
        out[i, j] = v
    end
    return out
end

function evaluate_affine(a::SDPSliceArtifact, values; T=eltype(values))
    length(values) == length(a.variables) || throw(DimensionMismatch("values has length $(length(values)); expected $(length(a.variables))"))
    return (equalities=a.equalities * values,
            rhs=convert(Vector{T}, a.rhs),
            psd=[affine_matrix(block, T; values=values) for block in a.psd_blocks])
end

function moment_value(values, a::SDPSliceArtifact, w::Word)
    reduction = affine_moment_reduction(w)
    value = eltype(values)(reduction.constant)
    representative = reduction.representative
    if representative !== nothing && !iszero(reduction.coefficient)
        id = get(a.moment_indices, representative, 0)
        id > 0 || throw(KeyError("moment $(word_string(representative)) is not represented"))
        value += eltype(values)(reduction.coefficient) * values[id]
    end
    return value
end
moment_value(values, a::SDPSliceArtifact, w::Tuple) = moment_value(values, a, _word(w))

"Evaluate a stationary EOM residual against an artifact's moment vector."
function eom_residual(values, a::SDPSliceArtifact, w::Word)
    terms = eom_terms(w, a.spec.mass2)
    residual = zero(eltype(values))
    for (key, coefficient) in terms
        if key == Word(())
            residual += eltype(values)(coefficient)
        else
            id = get(a.moment_indices, key, 0)
            id > 0 || throw(KeyError("moment $(word_string(key)) is not represented"))
            residual += eltype(values)(coefficient) * values[id]
        end
    end
    return residual
end
eom_residual(values, a::SDPSliceArtifact, w::Tuple) = eom_residual(values, a, _word(w))
eom_residual(a::SDPSliceArtifact, values, w) = eom_residual(values, a, w)

function _add_term!(d::Dict{Word,Rat}, w::Word, c::Rat)
    iszero(c) && return d
    reduction = affine_moment_reduction(w)
    if !iszero(reduction.constant)
        key = Word(())
        d[key] = get(d, key, Rat(0)) + c * reduction.constant
        iszero(d[key]) && delete!(d, key)
    end
    if reduction.representative !== nothing && !iszero(reduction.coefficient)
        key = reduction.representative
        d[key] = get(d, key, Rat(0)) + c * reduction.coefficient
        iszero(d[key]) && delete!(d, key)
    end
    return d
end

"The real formal Heisenberg derivation used by the minimal O(2) model."
function _letter_derivative(x::UInt8, mass2::Rat)
    if x == UInt8(1) # [H,Z] in the Pi=-iP convention, up to common scaling
        return [(Rat(1), word(:Pi))]
    elseif x == UInt8(2)
        return [(Rat(1), word(:Pibar))]
    elseif x == UInt8(3) # mass + [Z,[Zbar,Z]], charge +1
        # The first and last cubic terms are identical; together they are
        # the exact coefficient 2 of Z*Zbar*Z.
        return [
            (mass2, word(:Z)),
            (Rat(1), word(:Z, :Zbar, :Z)),
            (Rat(-1), word(:Z, :Z, :Zbar)),
            (Rat(-1), word(:Zbar, :Z, :Z)),
            (Rat(1), word(:Z, :Zbar, :Z)),
        ]
    elseif x == UInt8(4) # reflection partner
        return [
            (mass2, word(:Zbar)),
            (Rat(1), word(:Zbar, :Z, :Zbar)),
            (Rat(-1), word(:Zbar, :Zbar, :Z)),
            (Rat(-1), word(:Z, :Zbar, :Zbar)),
            (Rat(1), word(:Zbar, :Z, :Zbar)),
        ]
    end
    throw(ArgumentError("invalid letter ID $x"))
end

"Return the exact moment coefficients in the stationary EOM residual for a word."
function eom_terms(w::Word, mass2=Rat(1))
    m2 = _rat(mass2)
    terms = Dict{Word,Rat}()
    for i in eachindex(w)
        left = i == 1 ? Word(()) : Tuple(w[1:i-1])
        right = i == length(w) ? Word(()) : Tuple(w[i+1:end])
        for (coefficient, replacement) in _letter_derivative(w[i], m2)
            _add_term!(terms, _concat_words(left, replacement, right), coefficient)
        end
    end
    return terms
end
eom_terms(w::Tuple, mass2=Rat(1)) = eom_terms(_word(w), mass2)

function _new_variable!(variables::Vector{AffineVariable}, name::String, kind::Symbol;
                        word::Union{Nothing,Word}=nothing, pair=nothing)
    i = length(variables) + 1
    p = pair === nothing ? nothing : (Int(pair[1]), Int(pair[2]))
    push!(variables, AffineVariable(i, name, kind, word, p))
    return i
end

function _sorted_words(words)
    return sort!(collect(words), by=w -> (word_level(w), w))
end

function _scan_entry(observable::Symbol, mass2::Rat, moment_indices)
    if observable == :x2
        return _entry_moment(word(:Z, :Zbar), Rat(1), moment_indices)
    elseif observable == :z2barz2
        return _entry_moment(word(:Z, :Z, :Zbar, :Zbar), Rat(1), moment_indices)
    elseif observable == :energy
        # Appendix E40, specialized to D=2 and Pi=-iP:
        # E = -3/2 <tr Pi Pibar> + M^2/2 <tr Z Zbar>.
        kinetic = _entry_moment(word(:Pi, :Pibar), Rat(-3, 2), moment_indices)
        mass = _entry_moment(word(:Z, :Zbar), mass2 / Rat(2), moment_indices)
        return _entry_add(kinetic, mass)
    end
    throw(ArgumentError("unsupported scan observable $observable"))
end

function _make_rows(eom_sources, mass2, moment_indices, scan_observable, scan_value)
    rows = SparseAffineRow[]
    for source_word in eom_sources
        terms = eom_terms(source_word, mass2)
        ids = Dict{Int,Rat}()
        constant = get(terms, Word(()), Rat(0))
        for (key, coefficient) in terms
            key == Word(()) && continue
            id = get(moment_indices, key, 0)
            id > 0 || throw(KeyError("EOM introduced unrepresented moment $(word_string(key))"))
            ids[id] = get(ids, id, Rat(0)) + coefficient
        end
        ids = Dict(i => c for (i, c) in ids if !iszero(c))
        isempty(ids) && iszero(constant) && continue
        columns = sort!(collect(keys(ids)))
        push!(rows, SparseAffineRow(:eom, -constant, columns, Rat[ids[i] for i in columns]))
    end
    if scan_value !== nothing
        entry = _scan_entry(scan_observable, mass2, moment_indices)
        columns = Int[term.variable for term in entry.terms]
        values = Rat[term.coefficient for term in entry.terms]
        isempty(columns) && throw(ArgumentError("scan observable $scan_observable is constant at this level"))
        push!(rows, SparseAffineRow(:fixed_scan, scan_value - entry.constant,
                                    columns, values))
    end
    return rows
end

function _rows_sparse(rows::Vector{SparseAffineRow}, nvariables::Int)
    I = Int[]; J = Int[]; V = Rat[]; rhs = Rat[]; labels = Symbol[]
    for (r, row) in enumerate(rows)
        push!(rhs, row.rhs); push!(labels, row.label)
        for (j, value) in zip(row.columns, row.values)
            iszero(value) && continue
            push!(I, r); push!(J, j); push!(V, value)
        end
    end
    return sparse(I, J, V, length(rows), nvariables), rhs, labels
end

function _word_set_for_moments(level, eom_sources, basis_words, x_words)
    allwords = Set{Word}([Word(())])
    function add_reduced!(w::Word)
        reduction = affine_moment_reduction(w)
        representative = reduction.representative
        representative === nothing && return
        word_level(representative) <= level && push!(allwords, representative)
        return
    end
    for w in enumerate_words(level; charge=0, trace=true, reflection=true)
        add_reduced!(w)
    end
    for w in eom_sources, (k, _) in eom_terms(w, Rat(1))
        k == Word(()) || add_reduced!(k)
    end
    for w in basis_words, x in basis_words
        word_charge(w) == word_charge(x) || continue
        d, _ = word_dagger(w)
        add_reduced!(_concat_words(d, x))
    end
    for w in x_words, x in x_words
        d, _ = word_dagger(w)
        add_reduced!(_concat_words(d, x))
    end
    # E40/E42 require these two level-four/two moments independently of which
    # low-order operator basis first introduced them.
    add_reduced!(word(:Pi, :Pibar))
    add_reduced!(word(:Z, :Zbar))
    return _sorted_words(allwords)
end

function _block(name, kind, basis, entries; metadata=(;))
    size(entries, 1) == size(entries, 2) || throw(ArgumentError("PSD block $name is not square"))
    return PSDBlock(name, kind, Word[basis...], entries, metadata)
end

function _fingerprint_payload(a::SDPSliceArtifact)
    io = IOBuffer()
    print(io, "spec=", repr(a.spec), "\nsource=", a.source,
          "\nversion=", a.source_version, "\nmodel=", a.model,
          "\nlevel=", a.level, "\n")
    print(io, "params=", repr(a.parameters), "\nstatus=", a.reference_status,
          "\nobjective=", repr(a.objective),
          "\nreference_objective=", repr(a.reference_objective),
          "\nmetadata=", repr(a.metadata), "\n")
    for w in a.words
        print(io, "w:", join(Int.(w), ","), "\n")
    end
    for v in a.variables
        print(io, "v:", v.index, ":", v.kind, ":", v.name, ":")
        v.word === nothing || print(io, join(Int.(v.word), ","))
        v.pair === nothing || print(io, ":", v.pair[1], ",", v.pair[2])
        print(io, "\n")
    end
    for (w, i) in sort!(collect(a.moment_indices), by=first)
        print(io, "moment_index:", join(Int.(w), ","), "=", i, "\n")
    end
    print(io, "q_indices:", join(vec(a.q_indices), ","), "\n")
    rows = rowvals(a.equalities)
    vals = nonzeros(a.equalities)
    for r in 1:size(a.equalities, 1)
        print(io, "eq:", a.equality_labels[r], ":", a.rhs[r], ":")
        for c in 1:size(a.equalities, 2)
            for p in a.equalities.colptr[c]:(a.equalities.colptr[c + 1] - 1)
                rows[p] == r && print(io, c, "=", vals[p], ",")
            end
        end
        print(io, "\n")
    end
    for block in a.psd_blocks
        print(io, "block:", block.name, ":", block.kind, ":", repr(block.metadata), "\n")
        for w in block.basis
            print(io, "basis:", join(Int.(w), ","), "\n")
        end
        for i in axes(block.entries, 1), j in axes(block.entries, 2)
            e = block.entries[i, j]
            print(io, "e:", i, ",", j, ":", e.constant, ":")
            for t in e.terms
                print(io, t.variable, "=", t.coefficient, ",")
            end
            print(io, "\n")
        end
    end
    return take!(io)
end

function _with_fingerprint(a::SDPSliceArtifact)
    fp = bytes2hex(SHA.sha256(_fingerprint_payload(a)))
    return SDPSliceArtifact(a.spec, a.source, a.source_version, a.model, a.level,
        a.parameters, a.words, a.variables, a.moment_indices, a.q_indices,
        a.equalities, a.rhs, a.equality_labels, a.psd_blocks,
        a.reference_status, a.objective, a.reference_objective, a.metadata, fp)
end

function _validate_spec(spec::MatrixBootstrapSpec)
    if spec.source == _LIN_SOURCE
        spec.source_version == _LIN_VERSION ||
            throw(ArgumentError("Lin--Zheng source_version must be exactly $_LIN_VERSION"))
        spec.reference_status == _LIN_STATUS ||
            throw(ArgumentError("Lin--Zheng artifacts are build-only; reference_status must be :build_only"))
        spec.D == 2 || throw(ArgumentError("the low-order Lin--Zheng generator supports D=2 only"))
        haskey(_SUPPORTED_LEVEL_SCOPES, spec.level) ||
            throw(ArgumentError("supported build-first scopes are level 4 (:tiny), 6 (:small), and 8 (:medium); higher levels require the unpublished full quotient"))
        spec.scan_observable in (:x2, :energy, :z2barz2) ||
            throw(ArgumentError("unsupported scan observable $(spec.scan_observable)"))
    elseif spec.source == _KZ_SOURCE
        spec.source_version == _KZ_VERSION ||
            throw(ArgumentError("Kazakov--Zheng source_version must be exactly $_KZ_VERSION"))
        spec.reference_status == _KZ_STATUS ||
            throw(ArgumentError("Kazakov--Zheng fixture reference_status must be :cross_check_only"))
        spec.level == 4 || throw(ArgumentError("the Kazakov--Zheng fixture is fixed at Lambda=4"))
    else
        throw(ArgumentError("unsupported matrix-bootstrap source $(spec.source); allowed sources are $_LIN_SOURCE and $_KZ_SOURCE"))
    end
    return nothing
end

"""
    validate_artifact(artifact; rebuild=true) -> true

Fail closed if an artifact is internally inconsistent, has been mutated after
fingerprinting, or no longer matches the deterministic artifact rebuilt from
its frozen specification.  This validates benchmark semantics; it does not
claim that a build-only relaxation is paper-equivalent.
"""
function validate_artifact(a::SDPSliceArtifact; rebuild::Bool=true)
    _validate_spec(a.spec)
    a.source == a.spec.source || throw(ArgumentError("artifact source differs from its specification"))
    a.source_version == a.spec.source_version || throw(ArgumentError("artifact source version differs from its specification"))
    a.level == a.spec.level || throw(ArgumentError("artifact level differs from its specification"))
    a.reference_status == a.spec.reference_status || throw(ArgumentError("artifact status differs from its specification"))
    size(a.equalities, 1) == length(a.rhs) == length(a.equality_labels) ||
        throw(ArgumentError("artifact equality dimensions are inconsistent"))
    size(a.equalities, 2) == length(a.variables) ||
        throw(ArgumentError("artifact variable dimension is inconsistent"))
    all(v.index == i for (i, v) in enumerate(a.variables)) ||
        throw(ArgumentError("artifact variable indices are not canonical"))
    all(size(block, 1) == size(block, 2) for block in a.psd_blocks) ||
        throw(ArgumentError("artifact contains a nonsquare PSD block"))
    if a.source == _LIN_SOURCE
        get(a.metadata, :paper_equivalent, nothing) === false ||
            throw(ArgumentError("Lin--Zheng low-order artifacts must remain paper_equivalent=false"))
        get(a.metadata, :publication_claim, nothing) == :none ||
            throw(ArgumentError("Lin--Zheng low-order artifacts cannot carry a publication claim"))
    end

    expected = bytes2hex(SHA.sha256(_fingerprint_payload(a)))
    a.fingerprint == expected ||
        throw(ArgumentError("artifact semantic fingerprint mismatch"))

    if rebuild
        s = a.spec
        reference = build_matrix_bootstrap(
            source=s.source, source_version=s.source_version, D=s.D,
            mass2=s.mass2, level=s.level,
            scan_observable=s.scan_observable, scan_value=s.scan_value,
            reference_status=s.reference_status, relaxation=s.relaxation)
        a.fingerprint == reference.fingerprint ||
            throw(ArgumentError("artifact differs from the deterministic rebuild of its specification"))
    end
    return true
end

"""
    build_matrix_bootstrap(; kwargs...) -> SDPSliceArtifact

Build an exact-rational affine slice of the conservative low-order O(2)
hierarchy.  A `scan_value` adds one fixed-slice equality.  The nonlinear scan
itself is not an objective: `objective` and `reference_objective` are always
`nothing`.
"""
function build_matrix_bootstrap(; source::Symbol=_LIN_SOURCE,
                                source_version::AbstractString=(source == _LIN_SOURCE ? _LIN_VERSION : _KZ_VERSION),
                                D::Integer=2, mass2=(source == _LIN_SOURCE ? 1 : 0), level::Integer=4,
                                scan_observable::Symbol=:x2,
                                scan_value=nothing,
                                reference_status::Symbol=(source == _LIN_SOURCE ? _LIN_STATUS : _KZ_STATUS),
                                relaxation::Bool=true)
    source in (_LIN_SOURCE, _KZ_SOURCE) ||
        throw(ArgumentError("unsupported matrix-bootstrap source $source"))
    if source == _KZ_SOURCE
        D == 2 || throw(ArgumentError("the independent Kazakov--Zheng fixture uses its fixed two-matrix alphabet"))
        level == 4 || throw(ArgumentError("the independent Kazakov--Zheng fixture is fixed at Lambda=4"))
        _rat(mass2) == Rat(0) || throw(ArgumentError("KZ Lambda=4 uses the source's g=h=1 two-matrix integral"))
        scan_value === nothing || throw(ArgumentError("KZ Lambda=4 has no fixed scan parameter in this build-only fixture"))
        scan_observable == :x2 || throw(ArgumentError("KZ Lambda=4 exposes no O(2) scan observable"))
        relaxation || throw(ArgumentError("KZ Lambda=4 requires its Appendix-E relaxation block"))
        return _build_kazakov_zheng_lambda4(
            source_version=String(source_version), reference_status=reference_status)
    end
    m2 = _rat(mass2)
    sv = scan_value === nothing ? nothing : _rat(scan_value)
    spec = MatrixBootstrapSpec(source=source, source_version=String(source_version),
        D=Int(D), mass2=m2, level=Int(level), scan_observable=scan_observable,
        scan_value=sv, reference_status=reference_status, relaxation=relaxation)
    _validate_spec(spec)

    # Positivity uses operators of half the hierarchy level.  The supported
    # tiny/small/medium scopes remain build-first and intentionally stop before
    # the unpublished high-order momentum/gauge quotient becomes necessary.
    half_level = level ÷ 2
    open_words = enumerate_words(half_level; trace=false, include_empty=true)
    singlet_words = enumerate_words(half_level; charge=0, trace=true, include_empty=false)
    eom_sources = enumerate_words(level - 1; charge=0, trace=true, include_empty=false)
    moment_words = _word_set_for_moments(Int(level), eom_sources, open_words, singlet_words)

    # Empty is the normalized moment 1, not a decision variable.
    variable_words = [w for w in moment_words if w != Word(())]
    sort!(variable_words, by=w -> (word_level(w), w))
    variables = AffineVariable[]
    moment_indices = Dict{Word,Int}()
    for w in variable_words
        id = _new_variable!(variables, "mu[" * word_string(w) * "]", :moment; word=w)
        moment_indices[w] = id
    end

    q_indices = zeros(Int, length(singlet_words), length(singlet_words))
    if relaxation
        for i in eachindex(singlet_words), j in i:length(singlet_words)
            qid = _new_variable!(variables, "Q[" * string(i) * "," * string(j) * "]", :quadratic;
                                 pair=(i, j))
            q_indices[i, j] = qid
            q_indices[j, i] = qid
        end
    end

    rows = _make_rows(eom_sources, m2, moment_indices, scan_observable, sv)
    B, b, labels = _rows_sparse(rows, length(variables))

    blocks = PSDBlock[]
    # U(1)/O(2) charge blocks. Reflection pairs q and -q; one representative
    # per nonnegative charge is sufficient, with the partner recorded in metadata.
    charges = sort!(unique(word_charge(w) for w in open_words))
    for q in charges
        q < 0 && continue
        basis = Word[w for w in open_words if word_charge(w) == q]
        isempty(basis) && continue
        entries = Matrix{AffineEntry}(undef, length(basis), length(basis))
        for i in eachindex(basis), j in eachindex(basis)
            entries[i, j] = _moment_entry(basis[i], basis[j], moment_indices)
        end
        push!(blocks, _block(Symbol("M_charge_", q), :moment, basis, entries;
            metadata=(charge=q, reflection_partner=(q == 0 ? 0 : -q),
                      symmetry=:O2_charge, source_equation=:main_eq_5)))
    end

    # Main Eq. (6), in the explicit low-order form of Appendix E42.  For D=2,
    # x=<tr Z Zbar> and p=<tr Pi Pibar>, while
    # <tr X_I X_I>=2x and <tr P_I P_I>=-2p.
    ground = Matrix{AffineEntry}(undef, 2, 2)
    ground[1, 1] = AffineEntry(Rat(1))
    ground[1, 2] = ground[2, 1] =
        _entry_moment(word(:Pi, :Pibar), Rat(-2), moment_indices)
    ground[2, 2] = _entry_add(
        AffineEntry(m2),
        _entry_moment(word(:Z, :Zbar), Rat(2), moment_indices),
    )
    push!(blocks, _block(:ground_state_E42, :ground_state_positivity,
        Word[word(:Z), word(:Pi)], ground;
        metadata=(source_equations=(:main_eq_6, :appendix_E42),
                  basis_semantics=(:X_I, :P_I), D=2,
                  equation_exact=true, hierarchy_complete=false)))

    # A scalar singlet moment block is used for the nonlinear relaxation.  It
    # is intentionally explicit: M-Q is affine only after Q is introduced.
    if relaxation && !isempty(singlet_words)
        n = length(singlet_words)
        mentries = Matrix{AffineEntry}(undef, n, n)
        lentries = Matrix{AffineEntry}(undef, n, n)
        for i in 1:n, j in 1:n
            mentries[i, j] = _moment_entry(singlet_words[i], singlet_words[j], moment_indices)
            lentries[i, j] = _entry_sub(mentries[i, j], _entry_q(q_indices[i, j]))
        end
        push!(blocks, _block(:M_singlet, :moment_singlet, singlet_words, mentries;
            metadata=(charge=0, symmetry=:O2_singlet, source_equation=:main_eq_5)))
        push!(blocks, _block(:M_minus_Q, :relaxation_covariance, singlet_words, lentries;
            metadata=(charge=0, symmetry=:O2_singlet, source_equation=:relax2)))

        # [1 x'; x Q] >= 0.  The first row/column uses x_i moments and the
        # lower-right block uses the symmetric Q variables.
        liftbasis = vcat(Word[()], singlet_words)
        lift = Matrix{AffineEntry}(undef, n + 1, n + 1)
        lift[1, 1] = AffineEntry(Rat(1))
        for i in 1:n
            lift[1, i + 1] = _entry_moment(singlet_words[i], Rat(1), moment_indices)
            lift[i + 1, 1] = lift[1, i + 1]
        end
        for i in 1:n, j in 1:n
            lift[i + 1, j + 1] = _entry_q(q_indices[i, j])
        end
        push!(blocks, _block(:Q_lift, :nonlinear_relaxation, liftbasis, lift;
            metadata=(symmetry=:O2_singlet, source_equation=:relax1,
                      fixed_scan=(sv !== nothing))))
    end

    scan_expression = scan_observable == :energy ?
        (PiPibar=Rat(-3, 2), ZZbar=m2 / Rat(2), source_equation=:appendix_E40) :
        (observable=scan_observable, coefficient=Rat(1))
    params = (D=spec.D, mass2=spec.mass2, level=spec.level,
              scan_observable=spec.scan_observable, scan_value=spec.scan_value,
              relaxation=spec.relaxation,
              scope=_SUPPORTED_LEVEL_SCOPES[Int(level)],
              paper_source=true, paper_equivalent=false,
              independent_oracle=false)
    metadata = (is_affine=true,
        scan_is_slice=(sv !== nothing),
        nonlinear_scan_requires_rebuild=true,
        fixed_scan_semantics=:one_slice_one_affine_sdp,
        no_objective_oracle=true,
        normalization=(Word(()) => Rat(1)),
        charge_map=(Z=1, Zbar=-1, Pi=1, Pibar=-1),
        level_weights=(Z=1, Zbar=1, Pi=2, Pibar=2),
        published_free_variables=copy(_PUBLISHED_FREE),
        published_all_variables=copy(_PUBLISHED_ALL),
        published_intervals=copy(_PUBLISHED_INTERVALS),
        published_census_status=:metadata_only_fail_closed,
        scan_expression=scan_expression,
        construction=:ordered_momentum_low_order_relaxation,
        paper_equivalent=false,
        publication_claim=:none,
        supported_scope=_SUPPORTED_LEVEL_SCOPES[Int(level)],
        kinematic_constraints=(
            charge_neutrality=:eliminated,
            O2_reflection=:quotiented,
            hermiticity_time_reversal=:signed_ordered_quotient,
            canonical_commutator=:appendix_E34_affine_reduction,
            gauge_singlet=:vacuum_generator_normalization_only,
        ),
        supported_paper_constraints=(:main_eq_4_stationarity_subset,
            :main_eq_5_moment_positivity, :main_eq_6_low_order_E42,
            :appendix_E34_low_order, :appendix_E40_energy),
        omitted_constraints=(:full_momentum_trace_commutator_quotient,
            :full_gauge_Ward_tower, :full_O2_indexed_irrep_quotient,
            :full_ground_state_positivity_hierarchy,
            :published_Table_1_2_reproduction),
        production_quotient_available=false,
        raw_moment_count=length(moment_words),
        decision_variable_count=length(variables),
        eom_equation_count=count(label -> label == :eom, labels),
        equality_count=length(labels),
        psd_block_sizes=Tuple(size(block, 1) for block in blocks),
        build_only_level=true,
        source_scope=:lin_zheng_low_order_relaxation)
    artifact = SDPSliceArtifact(spec, source, String(source_version),
        :lin_zheng_o2_low_order_relaxation,
        Int(level), params, moment_words, variables, moment_indices, q_indices,
        B, b, labels, blocks, reference_status, nothing, nothing, metadata, "")
    return _with_fingerprint(artifact)
end

build_matrix_bootstrap(level::Integer; kwargs...) = build_matrix_bootstrap(; level=level, kwargs...)
build_lin_zheng(; kwargs...) = build_matrix_bootstrap(; source=:lin_zheng_o2, kwargs...)
build_lin_zheng(level::Integer; kwargs...) = build_lin_zheng(; level=level, kwargs...)

const _KZ_WORD_STRINGS = (
    "A2", "A4", "A2B2", "ABAB", "A6", "A4B2", "A3BAB", "A2BA2B",
    "A8", "A6B2", "A5BAB", "A4BA2B", "A4B4", "A3BA3B", "A3BAB3",
    "A3B2AB2", "A2BABAB2", "A2BAB2AB", "A2B2A2B2", "ABABABAB",
)

function _kz_word(s::AbstractString)
    out = UInt8[]
    for c in s
        c == 'A' && push!(out, UInt8(1))
        c == 'B' && push!(out, UInt8(2))
        isdigit(c) || continue
        n = Int(c - '0')
        # The compact names above use decimal exponents only as notation;
        # expand them into the corresponding binary word.
        last = out[end]
        for _ in 2:n
            push!(out, last)
        end
    end
    return Tuple(out)
end

function _kz_swap(w::Word)
    return Tuple(x == UInt8(1) ? UInt8(2) : UInt8(1) for x in w)
end

function _kz_canonical(w::Word)
    candidates = Word[]
    append!(candidates, _rotations(w))
    append!(candidates, _rotations(_word_reverse(w)))
    append!(candidates, _rotations(_kz_swap(w)))
    append!(candidates, _rotations(_kz_swap(_word_reverse(w))))
    sort!(candidates)
    return first(candidates)
end

function _kz_entry_moment(w1::Word, w2::Word, indices::Dict{Word,Int})
    key = _kz_canonical(_concat_words(_word_reverse(w1), w2))
    key == Word(()) && return AffineEntry(Rat(1))
    id = get(indices, key, 0)
    id > 0 || throw(KeyError("KZ Lambda=4 moment $(key) is absent"))
    return AffineEntry(Rat(0), [AffineTerm(id, Rat(1))])
end

function _kz_add!(d::Dict{Int,Rat}, id::Int, c)
    d[id] = get(d, id, Rat(0)) + _rat(c)
    iszero(d[id]) && delete!(d, id)
    return d
end

function _build_kazakov_zheng_lambda4(; source_version=_KZ_VERSION,
                                      reference_status::Symbol=_KZ_STATUS)
    words = Word[_kz_canonical(_kz_word(s)) for s in _KZ_WORD_STRINGS]
    words = unique(words)
    length(words) == 20 || throw(ArgumentError("KZ Lambda=4 word list lost an orbit"))
    variables = AffineVariable[]
    indices = Dict{Word,Int}()
    for (i, w) in enumerate(words)
        indices[w] = _new_variable!(variables, "mu[" * _KZ_WORD_STRINGS[i] * "]", :moment; word=w)
    end
    beta = _new_variable!(variables, "beta", :quadratic)
    rows = Dict{Int,Rat}[]
    rhsv = Rat[]
    function row(terms; rhs=0)
        d = Dict{Int,Rat}()
        for (s, c) in terms
            if s === :beta
                _kz_add!(d, beta, c)
            else
                _kz_add!(d, indices[_kz_canonical(_kz_word(s))], c)
            end
        end
        push!(rows, d); push!(rhsv, _rat(rhs))
    end
    # Appendix-E Eq. (14) at g=h=1, with every quadratic product represented
    # by beta.  These are exact loop equations, not numerically fitted rows.
    row(("A2"=>1, "A4"=>1, "A2B2"=>2, "ABAB"=>-2); rhs=1)
    row(("A2"=>-2, "A4"=>1, "A3BAB"=>-2, "A4B2"=>2, "A6"=>1))
    row(("A2"=>-1, "A2B2"=>1, "A2BA2B"=>1, "A3BAB"=>-2, "A4B2"=>2))
    row(("A2BA2B"=>-2, "A3BAB"=>3, "ABAB"=>1))
    row((:beta=>1, "A4"=>2, "A6"=>-1, "A5BAB"=>2, "A6B2"=>-2, "A8"=>-1))
    row((:beta=>1, "A2B2"=>1, "A4B2"=>-1, "A3B2AB2"=>-1, "A3BAB3"=>2, "A4B4"=>-1, "A6B2"=>-1))
    row(("A2B2"=>-2, "A2B2A2B2"=>1, "A2BABAB2"=>-2, "A3B2AB2"=>1, "A4B2"=>1, "A6B2"=>1))
    row(("A4"=>-1, "A4B2"=>1, "A4B4"=>1, "A4BA2B"=>1, "A5BAB"=>-2, "A6B2"=>1))
    row(("A3BAB"=>1, "A2BAB2AB"=>-2, "A2BABAB2"=>1, "A3BAB3"=>1, "A5BAB"=>1, "ABAB"=>-1))
    row(("A3BAB"=>1, "A5BAB"=>1, "ABAB"=>-2, "A2BABAB2"=>2, "ABABABAB"=>-2))
    row(("A3BAB"=>1, "A3BAB3"=>1, "A3BA3B"=>1, "A4BA2B"=>-2, "A5BAB"=>1))
    row(("A3BA3B"=>1, "A3BAB"=>1, "A3B2AB2"=>-2, "A3BAB3"=>2))
    row(("A2B2"=>-1, "A2BA2B"=>1, "A2BAB2AB"=>1, "A2BABAB2"=>-2, "A3B2AB2"=>1, "A4BA2B"=>1))
    row((:beta=>1, "A2BA2B"=>-1, "A3B2AB2"=>-1, "A3BA3B"=>2, "A4BA2B"=>-2))

    I = Int[]; J = Int[]; V = Rat[]
    for (r, d) in enumerate(rows), c in sort!(collect(keys(d)))
        push!(I, r); push!(J, c); push!(V, d[c])
    end
    B = sparse(I, J, V, length(rows), length(variables))
    labels = fill(:loop_equation, length(rows))
    basis_even = [_kz_word(s) for s in ("", "AA", "BB", "AAAA", "AABB", "ABAB", "ABBA", "BAAB", "BABA", "BBAA", "BBBB")]
    basis_odd = [_kz_word(s) for s in ("AB", "BA", "AAAB", "AABA", "ABAA", "ABBB", "BAAA", "BABB", "BBAB", "BBBA")]
    basis_mixed = [_kz_word(s) for s in ("B", "AAB", "ABA", "BAA", "BBB")]
    blocks = PSDBlock[]
    for (name, basis, kind) in ((:M_even_even, basis_even, :correlation_even),
                                (:M_odd_odd, basis_odd, :correlation_odd),
                                (:M_even_odd, basis_mixed, :correlation_mixed))
        n = length(basis)
        entries = Matrix{AffineEntry}(undef, n, n)
        for i in 1:n, j in 1:n
            entries[i, j] = _kz_entry_moment(basis[i], basis[j], indices)
        end
        push!(blocks, _block(name, kind, basis, entries;
            metadata=(alphabet=(:A, :B), symmetry=:Z2_cubed, cutoff=4,
                      source_equation=:correlation_positivity)))
    end
    liftbasis = Word[Word(()), _kz_word("AA")]
    lift = Matrix{AffineEntry}(undef, 2, 2)
    lift[1, 1] = AffineEntry(Rat(1))
    lift[1, 2] = lift[2, 1] = AffineEntry(Rat(0), [AffineTerm(indices[_kz_canonical(_kz_word("AA"))], Rat(1))])
    lift[2, 2] = _entry_q(beta)
    push!(blocks, _block(:Q_lift, :nonlinear_relaxation, liftbasis, lift;
        metadata=(alphabet=(:A, :B), symmetry=:Z2_cubed, cutoff=4,
                  source_equation=:relaxation_matrix)))
    spec = MatrixBootstrapSpec(source=:kazakov_zheng_lambda4,
        source_version=String(source_version), D=2, mass2=Rat(0), level=4,
        scan_observable=:x2, scan_value=nothing,
        reference_status=reference_status, relaxation=true)
    _validate_spec(spec)
    params = (alphabet=(:A, :B), g=Rat(1), h=Rat(1), cutoff=4,
              symmetry=:Z2_cubed, beta_definition=:(TrA2^2))
    metadata = (is_affine=true, scan_is_slice=false,
        fixed_scan_semantics=:one_slice_one_affine_sdp,
        no_objective_oracle=true, construction=:kazakov_zheng_lambda4,
        alphabet=(:A, :B), symmetry=:Z2_cubed, couplings=(g=Rat(1), h=Rat(1)),
        operator_count=20, loop_equation_count=14, quadratic_variables=1,
        paper_interval=(Rat(393566, 1_000_000), Rat(431148, 1_000_000)),
        production_quotient_available=true, build_only_level=false,
        source_scope=:kazakov_zheng_relaxation)
    artifact = SDPSliceArtifact(spec, :kazakov_zheng_lambda4, String(source_version),
        :kazakov_zheng_lambda4, 4, params, words, variables, indices,
        reshape([beta], 1, 1), B, rhsv, labels, blocks, reference_status,
        nothing, nothing, metadata, "")
    return _with_fingerprint(artifact)
end

"Independent small cross-check fixture; it is not the 2025 Lin--Zheng model."
function build_kazakov_zheng_lambda4(; level::Integer=4, mass2=0,
                                      reference_status::Symbol=_KZ_STATUS,
                                      kwargs...)
    level == 4 || throw(ArgumentError("the independent Kazakov--Zheng fixture is fixed at Lambda=4"))
    isempty(kwargs) || throw(ArgumentError("unsupported KZ Lambda=4 options: $(collect(keys(kwargs)))"))
    _rat(mass2) == Rat(0) || throw(ArgumentError("KZ Lambda=4 fixture uses the source's g=h=1 two-matrix integral, not mass2"))
    return _build_kazakov_zheng_lambda4(reference_status=reference_status)
end
build_kz_lambda4(; kwargs...) = build_kazakov_zheng_lambda4(; kwargs...)

"Optional conversion to SDPX's user-facing SDPProblem representation."
function to_sdp_problem(a::SDPSliceArtifact, ::Type{T}=Float64) where {T}
    # Resolve SDPX lazily: the benchmark module remains usable in a clean
    # Julia process that only wants canonical words and exact artifacts.
    isdefined(Main, :SDPX) || throw(ArgumentError("load SDPX before calling to_sdp_problem"))
    sdpx = getfield(Main, :SDPX)
    n = length(a.variables)
    L = length(a.psd_blocks)
    A = Vector{Array{T,3}}(undef, L)
    C = Vector{Matrix{T}}(undef, L)
    for (ell, block) in enumerate(a.psd_blocks)
        k = size(block, 1)
        A[ell] = zeros(T, n, k, k)
        C[ell] = zeros(T, k, k)
        for i in 1:k, j in 1:k
            entry = block.entries[i, j]
            # SDPX represents a PSD block as A(x) - C >= 0, whereas the
            # artifact stores F(x)=F0+A(x) >= 0.  Therefore C=-F0.
            C[ell][i, j] = -T(entry.constant)
            for term in entry.terms
                A[ell][term.variable, i, j] += T(term.coefficient)
            end
        end
    end
    # SDPX's user-facing B is variable-by-equality (m×n), whereas the
    # artifact stores the conventional sparse row system (n×m).
    B = SparseMatrixCSC{T,Int}(transpose(a.equalities))
    b = T.(a.rhs)
    c = zeros(T, n)
    return sdpx.ingest(c, A, C, B, b; T=T, sparse=:auto, validate=true,
                       symmetrize=false, verbosity=0)
end

end # module MatrixBootstrap
