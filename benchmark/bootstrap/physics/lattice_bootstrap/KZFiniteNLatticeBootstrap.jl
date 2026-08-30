module KZFiniteNLatticeBootstrap

using LinearAlgebra
using SHA
using SparseArrays
import SDPX

export LinkWord, LatticeBenchmarkSpec, LatticeBootstrapArtifact
export AffineEquation, AffineEntry, AffinePSDMatrix
export reduce_word, inverse_word, canonical_loop, is_closed, is_edge_simple
export enumerate_based_wilson_lines, build_lattice_bootstrap
export build_sdpx_problem, validate_artifact, canonical_text, stable_fingerprint
export exact_su2_plaquette, lattice_benchmark_specs, paper_lambda3_census
export assert_paper_lambda3_reproduced!

const ARTIFACT_SCHEMA_VERSION = 1
const PRIMARY_ARXIV = "2404.16925v4"
const PRIMARY_DOI = "10.1007/JHEP03(2025)099"
const VALID_STEPS = (Int8(-2), Int8(-1), Int8(1), Int8(2))

"""A word in oriented square-lattice links: ±1 is x and ±2 is y."""
struct LinkWord
    steps::Tuple{Vararg{Int8}}
end

function LinkWord(raw::AbstractVector{<:Integer})
    steps = Tuple(Int8(step) for step in raw)
    all(step -> step in VALID_STEPS, steps) || throw(ArgumentError(
        "lattice steps must be one of -2, -1, 1, 2",
    ))
    return LinkWord(steps)
end

LinkWord() = LinkWord(())
Base.length(word::LinkWord) = length(word.steps)
Base.isempty(word::LinkWord) = isempty(word.steps)
Base.getindex(word::LinkWord, index::Int) = word.steps[index]
Base.iterate(word::LinkWord, state...) = iterate(word.steps, state...)
Base.:(==)(left::LinkWord, right::LinkWord) = left.steps == right.steps
Base.hash(word::LinkWord, seed::UInt) = hash(word.steps, seed)
Base.copy(word::LinkWord) = word
Base.show(io::IO, word::LinkWord) = print(io, "LinkWord(", collect(word.steps), ")")

@inline function _delta(step::Int8)
    step == 1 && return (1, 0)
    step == -1 && return (-1, 0)
    step == 2 && return (0, 1)
    return (0, -1)
end

function is_closed(word::LinkWord)
    x = 0
    y = 0
    for step in word
        dx, dy = _delta(step)
        x += dx
        y += dy
    end
    return iszero(x) && iszero(y)
end

"""Cancel exact adjacent link/inverse-link pairs.

With `cyclic=true`, also cancel across the trace boundary. The latter is valid
for Wilson-loop traces, but is deliberately not used when enumerating the
open-index, origin-based operators entering the Gram matrix.
"""
function reduce_word(raw; cyclic::Bool=false)
    stack = Int8[]
    sizehint!(stack, length(raw))
    for item in raw
        step = Int8(item)
        step in VALID_STEPS || throw(ArgumentError(
            "lattice steps must be one of -2, -1, 1, 2",
        ))
        if !isempty(stack) && stack[end] == -step
            pop!(stack)
        else
            push!(stack, step)
        end
    end
    if cyclic
        while length(stack) >= 2 && stack[1] == -stack[end]
            pop!(stack)
            popfirst!(stack)
        end
    end
    return LinkWord(stack)
end

inverse_word(word::LinkWord) = LinkWord(Int8[-step for step in Iterators.reverse(word.steps)])

# Images of (+x,+y) for all eight signed permutation matrices B2 ≃ D4.
const _D4_MAPS = (
    (Int8(1), Int8(2)), (Int8(2), Int8(-1)),
    (Int8(-1), Int8(-2)), (Int8(-2), Int8(1)),
    (Int8(1), Int8(-2)), (Int8(-1), Int8(2)),
    (Int8(2), Int8(1)), (Int8(-2), Int8(-1)),
)

@inline function _map_step(step::Int8, map::Tuple{Int8,Int8})
    image = abs(step) == 1 ? map[1] : map[2]
    return step > 0 ? image : -image
end

_word_key(word::LinkWord) = isempty(word) ? "I" : join(string.(word.steps), ',')

@inline function _rotation_isless(steps::Vector{Int8}, left::Int, right::Int)
    count = length(steps)
    for offset in 0:(count - 1)
        left_value = steps[mod1(left + offset, count)]
        right_value = steps[mod1(right + offset, count)]
        left_value == right_value && continue
        return left_value < right_value
    end
    return false
end

function _minimum_rotation(steps::Vector{Int8})
    isempty(steps) && return ()
    offset = 1
    for candidate in 2:length(steps)
        _rotation_isless(steps, candidate, offset) && (offset = candidate)
    end
    count = length(steps)
    return ntuple(index -> steps[mod1(offset + index - 1, count)], count)
end

"""Canonicalize a closed traced loop under unitarity, cyclicity, B2, and C.

Implemented identities are exactly: adjacent/cyclic backtrack cancellation,
cyclic trace rotation, the eight square-lattice signed axis permutations
`B2 ≃ D4`, and charge conjugation (reverse order and flip every direction).
No reflection-positivity relation or finite-N trace identity is hidden here.
"""
function canonical_loop(raw)
    reduced = reduce_word(raw; cyclic=true)
    is_closed(reduced) || throw(ArgumentError("Wilson-loop word must be closed"))
    isempty(reduced) && return reduced
    best = nothing
    source = collect(reduced.steps)
    for map in _D4_MAPS
        transformed = Int8[_map_step(step, map) for step in source]
        reversed = Int8[-step for step in Iterators.reverse(transformed)]
        for orientation in (transformed, reversed)
            candidate = _minimum_rotation(orientation)
            if best === nothing || isless(candidate, best)
                best = candidate
            end
        end
    end
    return LinkWord(best::Tuple{Vararg{Int8}})
end

canonical_loop(word::LinkWord) = canonical_loop(word.steps)

"""Return true when no unoriented geometric link is traversed twice."""
function is_edge_simple(word::LinkWord)
    is_closed(word) || return false
    x = 0
    y = 0
    seen = Set{Tuple{Int,Int,Int}}()
    for step in word
        axis = abs(Int(step))
        if step > 0
            edge = (x, y, axis)
        else
            dx, dy = _delta(step)
            edge = (x + dx, y + dy, axis)
        end
        edge in seen && return false
        push!(seen, edge)
        dx, dy = _delta(step)
        x += dx
        y += dy
    end
    return true
end

function _enumerate_exact_based_length(length_cutoff::Int)
    length_cutoff >= 0 || throw(ArgumentError("length must be nonnegative"))
    iseven(length_cutoff) || return LinkWord[]
    paths = LinkWord[]
    current = Int8[]
    function visit!(depth::Int, x::Int, y::Int, previous::Int8)
        remaining = length_cutoff - depth
        distance = abs(x) + abs(y)
        (distance > remaining || isodd(remaining - distance)) && return
        if depth == length_cutoff
            iszero(x) && iszero(y) && push!(paths, LinkWord(copy(current)))
            return
        end
        for step in VALID_STEPS
            depth > 0 && step == -previous && continue
            dx, dy = _delta(step)
            push!(current, step)
            visit!(depth + 1, x + dx, y + dy, step)
            pop!(current)
        end
    end
    visit!(0, 0, 0, Int8(0))
    sort!(paths; by=_word_key)
    return paths
end

"""Enumerate reduced, origin-based closed Wilson-line operators.

These are operator words, so no cyclic or D4 quotient is applied to the
basis. Moment entries are canonicalized only after forming `Oi† Oj`.
"""
function enumerate_based_wilson_lines(max_length::Int)
    max_length >= 4 || throw(ArgumentError("operator cutoff must be at least 4"))
    iseven(max_length) || throw(ArgumentError("operator cutoff must be even"))
    result = LinkWord[LinkWord()]
    for length_cutoff in 4:2:max_length
        append!(result, _enumerate_exact_based_length(length_cutoff))
    end
    return result
end

Base.@kwdef struct LatticeBenchmarkSpec{T}
    id::String
    scale::Symbol
    coupling::T
    dimension::Int = 2
    gauge_group::Symbol = :SU2
    operator_max_length::Int
    equation_max_length::Int
    hierarchy::Symbol = :based_length
    scope::Symbol = :based_length_edge_simple_subset
    reference_status::Symbol = :build_only
    paper_equivalent::Bool = false
    publication_claim::Symbol = :none
end

"""A scalar affine expression used in a Gram entry.

Each Hermitian-conjugation Gram entry is one Wilson moment (or the identity),
so the compact representation needs at most one variable.
"""
struct AffineEntry{T}
    constant::T
    variable::Int
    coefficient::T
end

Base.:(==)(left::AffineEntry, right::AffineEntry) =
    left.constant == right.constant &&
    left.variable == right.variable &&
    left.coefficient == right.coefficient
Base.hash(entry::AffineEntry, seed::UInt) =
    hash((entry.constant, entry.variable, entry.coefficient), seed)

struct AffineEquation{T}
    constant::T
    indices::Vector{Int}
    coefficients::Vector{T}
    source::String
end

Base.:(==)(left::AffineEquation, right::AffineEquation) =
    left.constant == right.constant &&
    left.indices == right.indices &&
    left.coefficients == right.coefficients &&
    left.source == right.source

struct AffinePSDMatrix{T}
    name::Symbol
    entries::Matrix{AffineEntry{T}}
    source::String
end

Base.:(==)(left::AffinePSDMatrix, right::AffinePSDMatrix) =
    left.name == right.name &&
    left.entries == right.entries &&
    left.source == right.source

struct LatticeBootstrapArtifact{T}
    schema_version::Int
    spec::LatticeBenchmarkSpec{T}
    variables::Vector{LinkWord}
    equations::Vector{AffineEquation{T}}
    gram_blocks::Vector{AffinePSDMatrix{T}}
    objective::AffineEntry{T}
    operator_basis::Vector{LinkWord}
    oracle::NamedTuple
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

function _validate_spec(spec::LatticeBenchmarkSpec)
    spec.dimension == 2 || throw(ArgumentError(
        "only the primary paper's D=2 validation sector is implemented",
    ))
    spec.gauge_group === :SU2 || throw(ArgumentError(
        "only the finite-N SU(2) single-trace closure is implemented",
    ))
    spec.hierarchy === :based_length || throw(ArgumentError(
        "paper Λ hierarchy is not reconstructed; use hierarchy=:based_length",
    ))
    spec.scope === :based_length_edge_simple_subset || throw(ArgumentError(
        "only the based-length edge-simple subset is implemented",
    ))
    spec.reference_status === :build_only || throw(ArgumentError(
        "lattice artifacts are build-only; reference_status must be :build_only",
    ))
    spec.paper_equivalent === false || throw(ArgumentError(
        "the based-length edge-simple subset is not paper-equivalent",
    ))
    spec.publication_claim === :none || throw(ArgumentError(
        "the based-length edge-simple subset cannot carry a publication claim",
    ))
    isfinite(spec.coupling) && spec.coupling > zero(spec.coupling) ||
        throw(ArgumentError("coupling λ must be finite and strictly positive"))
    spec.operator_max_length in (4, 6, 8) || throw(ArgumentError(
        "operator_max_length must be one of 4, 6, 8 for the audited catalog",
    ))
    spec.equation_max_length >= 4 && iseven(spec.equation_max_length) ||
        throw(ArgumentError("equation_max_length must be even and at least 4"))
    spec.equation_max_length <= 2 * spec.operator_max_length || throw(ArgumentError(
        "equation cutoff exceeds the moments supplied by this Gram truncation",
    ))
    return nothing
end

function lattice_benchmark_specs(::Type{T}=Float64) where {T}
    lambda = T(2)
    return (
        tiny=LatticeBenchmarkSpec{T}(
            id="kz25/su2_d2_based_l4_lambda2",
            scale=:tiny,
            coupling=lambda,
            operator_max_length=4,
            equation_max_length=4,
        ),
        small=LatticeBenchmarkSpec{T}(
            id="kz25/su2_d2_based_l6_lambda2",
            scale=:small,
            coupling=lambda,
            operator_max_length=6,
            equation_max_length=6,
        ),
        medium=LatticeBenchmarkSpec{T}(
            id="kz25/su2_d2_based_l8_lambda2",
            scale=:medium,
            coupling=lambda,
            operator_max_length=8,
            equation_max_length=8,
        ),
    )
end

function _besseli01(z::T) where {T}
    z >= zero(T) || throw(ArgumentError("Bessel series expects z ≥ 0"))
    quarter_z2 = (z * z) / T(4)
    term0 = one(T)
    sum0 = one(T)
    term1 = z / T(2)
    sum1 = term1
    tolerance = eps(T)
    for k in 1:100_000
        term0 *= quarter_z2 / T(k * k)
        term1 *= quarter_z2 / T(k * (k + 1))
        next0 = sum0 + term0
        next1 = sum1 + term1
        if abs(term0) <= tolerance * max(one(T), abs(next0)) &&
           abs(term1) <= tolerance * max(one(T), abs(next1))
            return next0, next1
        end
        sum0 = next0
        sum1 = next1
    end
    throw(ErrorException("modified-Bessel series did not converge"))
end

"""Exact D=2 SU(2) plaquette average at the paper's coupling convention.

From Appendix A, `Z = λ I1(8/λ)/4` and `Tr = tr/2`, hence
`uP = I0(8/λ)/I1(8/λ) - λ/4`.
"""
function exact_su2_plaquette(lambda::T) where {T}
    isfinite(lambda) && lambda > zero(T) || throw(ArgumentError(
        "coupling λ must be finite and strictly positive",
    ))
    z = T(8) / lambda
    i0, i1 = _besseli01(z)
    return i0 / i1 - lambda / T(4)
end

function _canonical_trace_loops(max_length::Int)
    moments = Dict{String,LinkWord}()
    for length_cutoff in 4:2:max_length
        for word in _enumerate_exact_based_length(length_cutoff)
            loop = canonical_loop(word)
            moments[_word_key(loop)] = loop
        end
    end
    return sort!(collect(values(moments)); by=_word_key)
end

function _add_term!(row::Dict{LinkWord,T}, word::LinkWord, coefficient::T) where {T}
    iszero(coefficient) && return row
    row[word] = get(row, word, zero(T)) + coefficient
    iszero(row[word]) && delete!(row, word)
    return row
end

function _rotate_steps(word::LinkWord, selected::Int)
    steps = collect(word.steps)
    return vcat(steps[selected:end], steps[1:(selected - 1)])
end

"""One edge-simple specialization of KZ eqs. (2.31)--(2.33) and (2.36).

The row contains `Aid + Avar = 0` for `N=2`, `m=0`: `Aid` and `Avar` are
the paper's eqs. (2.32) and (2.33), while eq. (2.36) is the compact SU(2)
form. Repeated geometric links would require the split/rejoin terms of
eqs. (2.34)--(2.35) and are rejected rather than silently emitting an
incomplete equation.
"""
function _edge_simple_loop_equation(word::LinkWord, selected::Int, lambda::T) where {T}
    is_edge_simple(word) || throw(ArgumentError(
        "edge-simple equation generator rejects repeated geometric links",
    ))
    1 <= selected <= length(word) || throw(BoundsError(word.steps, selected))
    contour = _rotate_steps(word, selected)
    direction = contour[1]
    perpendicular_axis = abs(direction) == 1 ? Int8(2) : Int8(1)
    row = Dict{LinkWord,T}()
    for eta in (perpendicular_axis, -perpendicular_axis)
        plaquette = Int8[direction, eta, -direction, -eta]
        adjoint_plaquette = Int8[eta, direction, -eta, -direction]
        positive = canonical_loop(vcat(plaquette, contour))
        negative = canonical_loop(vcat(adjoint_plaquette, contour))
        _add_term!(row, positive, inv(lambda))
        _add_term!(row, negative, -inv(lambda))
    end
    # Paper eq. (2.32), with sG=1, N=2 and no additional link overlap.
    _add_term!(row, canonical_loop(contour), T(3) / T(4))
    return row
end

function _row_signature(row::Dict{LinkWord,T}) where {T}
    pairs = sort!(collect(row); by=pair -> _word_key(first(pair)))
    return join((string(_word_key(word), ':', coefficient) for (word, coefficient) in pairs), ';')
end

function _temporary_equations(spec::LatticeBenchmarkSpec{T}) where {T}
    rows = Dict{String,Tuple{Dict{LinkWord,T},String}}()
    for loop in _canonical_trace_loops(spec.equation_max_length)
        is_edge_simple(loop) || continue
        for selected in 1:length(loop)
            row = _edge_simple_loop_equation(loop, selected, spec.coupling)
            isempty(row) && continue
            signature = _row_signature(row)
            source = (
                "KZ25 Eqs. (2.31)-(2.33), compact SU(2) Eq. (2.36), " *
                "m=0 edge-simple Aid+Avar; selected=$selected"
            )
            length(loop) == 4 && (source *= "; lowest single-trace row Eq. (2.37)")
            rows[signature] = (
                row,
                source,
            )
        end
    end
    return [rows[key] for key in sort!(collect(keys(rows)))]
end

function _gram_words(basis::Vector{LinkWord})
    side = length(basis)
    words = Matrix{LinkWord}(undef, side, side)
    cache = Dict{Tuple{Vararg{Int8}},LinkWord}()
    for column in 1:side, row in 1:column
        product = vcat(
            collect(inverse_word(basis[row]).steps),
            collect(basis[column].steps),
        )
        reduced = reduce_word(product; cyclic=true)
        moment = get!(cache, reduced.steps) do
            canonical_loop(reduced)
        end
        words[row, column] = moment
        words[column, row] = moment
    end
    return words
end

function _provenance()
    return (
        title="Bootstrap for finite N lattice Yang-Mills theory",
        authors=("Vladimir Kazakov", "Zechuan Zheng"),
        arxiv=PRIMARY_ARXIV,
        doi=PRIMARY_DOI,
        reference_status=:build_only,
        paper_equivalent=false,
        publication_claim=:none,
        scope=:based_length_edge_simple_subset,
        convention="Tr=tr/N; lambda=2N^2/beta; N=2; D=2",
        derivation_context=(
            "2.26 (action variation)",
            "2.28 (A tensor)",
            "2.29 (traceless-generator contraction)",
            "2.30 (s_G definition)",
        ),
        implemented_equations=(
            "2.31 (master loop identity, edge-simple single-trace restriction)",
            "2.32 (Aid)",
            "2.33 (Avar)",
            "2.36 (compact SU(2), edge-simple specialization)",
            "2.37 (lowest single-trace row at the tiny scale)",
            "3.2 (Hermitian-conjugation positivity)",
            "3.3 (Gram entries)",
            "3.7 (9x9 length-four example structure)",
            "A.1-A.4 (D=2 SU(2) plaquette oracle)",
        ),
        implemented_scope=(
            "Hermitian-conjugation Gram matrix, without irrep block reduction",
            "edge-simple single-trace Aid+Avar equations; repeated-link split terms fail closed",
            "B2=D4 and charge-conjugation moment canonicalization",
        ),
        excluded_claims=(
            "reflection positivity is not generated",
            "paper recursive Lambda hierarchy is not generated",
            "Table 2 Lambda=3 census is not claimed reproduced",
            "the based-length edge-simple subset is not paper-equivalent",
        ),
    )
end

function _entry_for_word(word::LinkWord, index::Dict{LinkWord,Int}, ::Type{T}) where {T}
    if isempty(word)
        return AffineEntry{T}(one(T), 0, zero(T))
    end
    return AffineEntry{T}(zero(T), index[word], one(T))
end

function build_lattice_bootstrap(spec::LatticeBenchmarkSpec{T}) where {T}
    _validate_spec(spec)
    basis = enumerate_based_wilson_lines(spec.operator_max_length)
    gram_words = _gram_words(basis)
    temporary_equations = _temporary_equations(spec)

    moments = Dict{String,LinkWord}()
    plaquette = canonical_loop(Int8[1, 2, -1, -2])
    moments[_word_key(plaquette)] = plaquette
    gram_moments = Set{LinkWord}()
    for word in gram_words
        isempty(word) && continue
        moments[_word_key(word)] = word
        push!(gram_moments, word)
    end
    for (row, _) in temporary_equations, word in keys(row)
        isempty(word) && continue
        moments[_word_key(word)] = word
    end
    variables = sort!(collect(values(moments)); by=_word_key)
    index = Dict(word => position for (position, word) in pairs(variables))

    equations = AffineEquation{T}[]
    for (raw, source) in temporary_equations
        constant = get(raw, LinkWord(), zero(T))
        terms = sort!(
            [(index[word], coefficient) for (word, coefficient) in raw if !isempty(word)];
            by=first,
        )
        push!(equations, AffineEquation{T}(
            constant,
            first.(terms),
            last.(terms),
            source,
        ))
    end

    side = length(basis)
    entries = Matrix{AffineEntry{T}}(undef, side, side)
    for column in 1:side, row in 1:side
        entries[row, column] = _entry_for_word(gram_words[row, column], index, T)
    end
    blocks = [AffinePSDMatrix{T}(
        :hermitian_origin_based,
        entries,
        "KZ25 eqs. (3.2), (3.3), and the length-4 example (3.7)",
    )]
    objective = _entry_for_word(plaquette, index, T)
    oracle = (
        observable=:plaquette,
        exact_value=exact_su2_plaquette(spec.coupling),
        source="KZ25 Appendix A, eqs. (A.1)-(A.4), differentiated at N=2",
        use="validation reference only; a finite-truncation SDP bound need not equal it",
    )
    counts = (
        variables=length(variables),
        equations=length(equations),
        psd_blocks=length(blocks),
        block_sizes=(side,),
        operator_basis=length(basis),
        gram_moments=length(gram_moments),
        maximum_moment_length=maximum(length, variables; init=0),
        equation_scope=:edge_simple_Aid_Avar,
        hierarchy=:based_length_not_paper_Lambda,
        scope=:based_length_edge_simple_subset_not_paper_lambda,
    )
    provisional = LatticeBootstrapArtifact{T}(
        ARTIFACT_SCHEMA_VERSION,
        spec,
        variables,
        equations,
        blocks,
        objective,
        basis,
        oracle,
        _provenance(),
        counts,
        "",
    )
    fingerprint = stable_fingerprint(provisional)
    return LatticeBootstrapArtifact{T}(
        provisional.schema_version,
        provisional.spec,
        provisional.variables,
        provisional.equations,
        provisional.gram_blocks,
        provisional.objective,
        provisional.operator_basis,
        provisional.oracle,
        provisional.provenance,
        provisional.counts,
        fingerprint,
    )
end

build_lattice_bootstrap(scale::Symbol, ::Type{T}=Float64) where {T} =
    build_lattice_bootstrap(getproperty(lattice_benchmark_specs(T), scale))

_number_token(value) = string(value)

function _write_canonical(io::IO, artifact::LatticeBootstrapArtifact)
    spec = artifact.spec
    println(io, "kz-finite-n-lattice-bootstrap-schema=", artifact.schema_version)
    println(io, "id=", spec.id)
    println(io, "scale=", spec.scale)
    println(io, "lambda=", _number_token(spec.coupling))
    println(io, "dimension=", spec.dimension)
    println(io, "group=", spec.gauge_group)
    println(io, "operator-max-length=", spec.operator_max_length)
    println(io, "equation-max-length=", spec.equation_max_length)
    println(io, "hierarchy=", spec.hierarchy)
    println(io, "scope=", spec.scope)
    println(io, "reference-status=", spec.reference_status)
    println(io, "paper-equivalent=", spec.paper_equivalent)
    println(io, "publication-claim=", spec.publication_claim)
    for (name, value) in pairs(artifact.counts)
        println(io, "count.", name, '=', repr(value))
    end
    for (name, value) in pairs(artifact.provenance)
        println(io, "provenance.", name, '=', repr(value))
    end
    for (index, word) in pairs(artifact.operator_basis)
        println(io, "operator[", index, "]=", _word_key(word))
    end
    for (index, word) in pairs(artifact.variables)
        println(io, "var[", index, "]=", _word_key(word))
    end
    for (row_index, row) in pairs(artifact.equations)
        print(io, "eq[", row_index, "]=", _number_token(row.constant))
        for position in eachindex(row.indices)
            print(io, ';', row.indices[position], ':', _number_token(row.coefficients[position]))
        end
        println(io, "|", row.source)
    end
    for (block_index, block) in pairs(artifact.gram_blocks)
        println(io, "psd[", block_index, "]=", block.name, ':', size(block.entries, 1),
            '|', block.source)
        for column in axes(block.entries, 2), row in axes(block.entries, 1)
            entry = block.entries[row, column]
            println(io, "g[", row, ',', column, "]=",
                _number_token(entry.constant), ';', entry.variable, ':',
                _number_token(entry.coefficient))
        end
    end
    println(io, "objective=", _number_token(artifact.objective.constant), ';',
        artifact.objective.variable, ':', _number_token(artifact.objective.coefficient))
    for (name, value) in pairs(artifact.oracle)
        println(io, "oracle.", name, '=', repr(value))
    end
    return io
end

function canonical_text(artifact::LatticeBootstrapArtifact)
    io = IOBuffer()
    _write_canonical(io, artifact)
    return String(take!(io))
end

# Write-only IO adapter that streams the exact canonical bytes into SHA-256.
# `unsafe_wrap(...; own=false)` creates only a view of each print chunk; no
# artifact-sized String or byte copy is materialized.
mutable struct _SHA256WriteIO <: IO
    context::SHA.SHA2_256_CTX
    byte::Vector{UInt8}
end
Base.isopen(::_SHA256WriteIO) = true
Base.iswritable(::_SHA256WriteIO) = true
function Base.write(io::_SHA256WriteIO, byte::UInt8)
    io.byte[1] = byte
    SHA.update!(io.context, io.byte)
    return 1
end
function Base.unsafe_write(
    io::_SHA256WriteIO, pointer::Ptr{UInt8}, count::UInt,
)
    bytes = unsafe_wrap(Vector{UInt8}, pointer, Int(count); own=false)
    SHA.update!(io.context, bytes)
    return count
end

function stable_fingerprint(artifact::LatticeBootstrapArtifact)
    io = _SHA256WriteIO(SHA.SHA2_256_CTX(), Vector{UInt8}(undef, 1))
    _write_canonical(io, artifact)
    return bytes2hex(SHA.digest!(io.context))
end

function validate_artifact(artifact::LatticeBootstrapArtifact)
    failures = String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures, "schema_version")
    valid_spec = true
    try
        _validate_spec(artifact.spec)
    catch
        valid_spec = false
        push!(failures, "spec")
    end
    artifact.spec.reference_status === :build_only || push!(failures, "reference_status")
    artifact.spec.paper_equivalent === false || push!(failures, "paper_equivalent")
    artifact.spec.publication_claim === :none || push!(failures, "publication_claim")
    get(artifact.provenance, :reference_status, nothing) === :build_only ||
        push!(failures, "provenance_reference_status")
    get(artifact.provenance, :paper_equivalent, nothing) === false ||
        push!(failures, "provenance_paper_equivalent")
    get(artifact.provenance, :publication_claim, nothing) === :none ||
        push!(failures, "provenance_publication_claim")
    get(artifact.provenance, :scope, nothing) === :based_length_edge_simple_subset ||
        push!(failures, "provenance_scope")
    variable_count = length(artifact.variables)
    issorted(_word_key.(artifact.variables)) || push!(failures, "variable_order")
    length(unique(artifact.variables)) == variable_count || push!(failures, "duplicate_variables")
    all(word -> is_closed(word) && canonical_loop(word) == word, artifact.variables) ||
        push!(failures, "noncanonical_variable")
    for equation in artifact.equations
        length(equation.indices) == length(equation.coefficients) ||
            push!(failures, "equation_dimension")
        issorted(equation.indices) || push!(failures, "equation_order")
        length(unique(equation.indices)) == length(equation.indices) ||
            push!(failures, "equation_duplicate_index")
        all(index -> 1 <= index <= variable_count, equation.indices) ||
            push!(failures, "equation_index")
        isfinite(equation.constant) && all(isfinite, equation.coefficients) ||
            push!(failures, "equation_nonfinite")
    end
    for block in artifact.gram_blocks
        size(block.entries, 1) == size(block.entries, 2) || push!(failures, "psd_nonsquare")
        for column in axes(block.entries, 2), row in axes(block.entries, 1)
            entry = block.entries[row, column]
            mirror = block.entries[column, row]
            entry == mirror || push!(failures, "psd_nonsymmetric")
            0 <= entry.variable <= variable_count || push!(failures, "psd_index")
            isfinite(entry.constant) && isfinite(entry.coefficient) ||
                push!(failures, "psd_nonfinite")
            (entry.variable == 0) == iszero(entry.coefficient) ||
                push!(failures, "psd_affine_encoding")
        end
    end
    isfinite(artifact.oracle.exact_value) || push!(failures, "oracle_nonfinite")

    # Rebuild the complete deterministic semantics from the declared spec.
    # This prevents a self-consistent re-fingerprinting of altered counts,
    # provenance, basis words, equations, Gram entries, or objective.
    if valid_spec
        try
            expected = build_lattice_bootstrap(artifact.spec)
            artifact.variables == expected.variables ||
                push!(failures, "variable_semantics")
            artifact.equations == expected.equations ||
                push!(failures, "equation_semantics")
            artifact.gram_blocks == expected.gram_blocks ||
                push!(failures, "psd_semantics")
            artifact.objective == expected.objective ||
                push!(failures, "objective_semantics")
            artifact.operator_basis == expected.operator_basis ||
                push!(failures, "operator_basis_semantics")
            artifact.oracle == expected.oracle ||
                push!(failures, "oracle_semantics")
            artifact.provenance == expected.provenance ||
                push!(failures, "provenance_semantics")
            artifact.counts == expected.counts ||
                push!(failures, "counts_semantics")
        catch
            push!(failures, "semantic_rebuild")
        end
    end
    stable_fingerprint(artifact) == artifact.fingerprint || push!(failures, "fingerprint")
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

"""Lower the build artifact to SDPX's ordinary affine SDP representation."""
function build_sdpx_problem(artifact::LatticeBootstrapArtifact{T}) where {T}
    verdict = validate_artifact(artifact)
    verdict.valid || throw(ArgumentError(
        "invalid lattice artifact: $(join(verdict.failures, ", "))",
    ))
    variable_count = length(artifact.variables)
    coefficients = T[zero(T) for _ in 1:variable_count]
    objective = artifact.objective
    objective.variable > 0 && (coefficients[objective.variable] += objective.coefficient)

    psd_coefficients = Vector{Vector{SparseMatrixCSC{T,Int}}}()
    psd_constants = Matrix{T}[]
    for block in artifact.gram_blocks
        side = size(block.entries, 1)
        row_indices = [Int[] for _ in 1:variable_count]
        column_indices = [Int[] for _ in 1:variable_count]
        values = [T[] for _ in 1:variable_count]
        constant = zeros(T, side, side)
        for column in 1:side, row in 1:side
            entry = block.entries[row, column]
            constant[row, column] = -entry.constant
            if entry.variable > 0 && !iszero(entry.coefficient)
                push!(row_indices[entry.variable], row)
                push!(column_indices[entry.variable], column)
                push!(values[entry.variable], entry.coefficient)
            end
        end
        matrices = [
            sparse(row_indices[index], column_indices[index], values[index], side, side)
            for index in 1:variable_count
        ]
        push!(psd_coefficients, matrices)
        push!(psd_constants, constant)
    end

    equality_count = length(artifact.equations)
    eq_rows = Int[]
    eq_columns = Int[]
    eq_values = T[]
    rhs = zeros(T, equality_count)
    for (column, equation) in pairs(artifact.equations)
        rhs[column] = -equation.constant
        for position in eachindex(equation.indices)
            push!(eq_rows, equation.indices[position])
            push!(eq_columns, column)
            push!(eq_values, equation.coefficients[position])
        end
    end
    equality_matrix = sparse(
        eq_rows, eq_columns, eq_values, variable_count, equality_count,
    )
    return SDPX.ingest(
        coefficients,
        psd_coefficients,
        psd_constants,
        equality_matrix,
        rhs;
        T,
        sparse=true,
        validate=true,
        symmetrize=false,
        verbosity=0,
    )
end

"""Primary-source Table 2 / eq. (4.5) values, not generated counts."""
paper_lambda3_census() = (
    dimension=2,
    paper_hierarchy_level=3,
    maximum_loop_length=24,
    loops=8335,
    equations=14591,
    free_variables=1044,
    block_sizes=(54, 52, 46, 45, 98, 45, 46, 52, 53, 98, 30, 27, 21, 24, 16, 11, 8, 12),
    source="KZ25 section 4.2, eq. (4.5), and Table 2",
    independently_reproduced=false,
    reason=(
        "The paper defines its hierarchy recursively through diagrammatic Schwinger-Dyson variations; " *
        "the released article does not provide a machine-readable operator census. " *
        "This implementation uses the explicitly labeled based-length hierarchy and refuses equivalence."
    ),
)

function assert_paper_lambda3_reproduced!()
    census = paper_lambda3_census()
    census.independently_reproduced && return census
    throw(ArgumentError(
        "paper Lambda=3 census is reference-only and has not been independently reproduced: " *
        census.reason,
    ))
end

end # module
