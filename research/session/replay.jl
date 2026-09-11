# ===========================================================================
# src/session/replay.jl
#
# S07 step 2 — the replayable result format.
#
# WHAT THIS FILE IS.  A prepared update is only reproducible if the *raw
# arithmetic input* is reproducible.  This file defines an envelope that carries
#
#     * the full source SHA (40 hex characters, enforced complete),
#     * the exact arithmetic inputs, as exact scalar payloads,
#     * the arithmetic family, the working precision and the rounding label,
#   * every tolerance the run was judged by, with its exact literal,
#
# and a decoder that fails EXPLICITLY — with a named code — on a corrupt body, a
# truncated body, a digest mismatch, a type that does not resolve in this build,
# a width mismatch, or an arithmetic/version mismatch against the caller's
# expectations.  A replay that silently succeeds against the wrong build is
# worse than one that refuses.
#
# NO Float64 SIDEWAYS DOOR.  Every scalar goes through the ONE codec in
# `update.jl` (`session_scalar_payload` / `session_payload_scalar`): IEEE floats
# as raw bit patterns, BigFloat as an exactly-round-tripping string at its own
# precision, other isbits scalars (the MF-limb path) as raw bytes.  A value the
# codec cannot represent exactly is REFUSED; it is never narrowed to Float64.
#
# WHAT THIS FILE DOES NOT DO.  It does not factorize, solve, or certify.  It
# does not serialize a solver's `Fresh`/`Optimal` state — see
# `replay_state_is_not_evidence` at the bottom, which is the executable form of
# "a restored checkpoint must be re-admitted and re-certified".
#
# INCLUDE ORDER.  Requires `update.jl` (tolerances, payload codec, fingerprints)
# to have been included first.  Measured in the driver, both orders.
# ===========================================================================

const REPLAY_SCHEMA = "sdpx-replay/1"
const REPLAY_SCHEMA_VERSION = 1

"""
    ReplayIntegrityError

Why a replay refused.  Codes: `:bad_schema`, `:bad_sha`, `:truncated`,
`:missing_field`, `:duplicate_field`, `:bad_digest`, `:bad_number`,
`:bad_field_line`, `:mixed_arithmetic`, `:version_mismatch`,
`:tolerance_mismatch`, `:fingerprint_mismatch`, `:not_evidence`.
"""
struct ReplayIntegrityError <: Exception
    code::Symbol
    detail::String
end

function Base.showerror(io::IO, e::ReplayIntegrityError)
    print(io, "ReplayIntegrityError[", e.code, "]: ", e.detail)
end

# ---------------------------------------------------------------------------
# 1. one field of raw arithmetic input
# ---------------------------------------------------------------------------

"""
    ReplayField

A named block of raw arithmetic input: the operator, the objective, the
right-hand side, the cone parameters.  The scalars are stored in column-major
order, which is also the order they are digested in, so the digest of the
envelope and the digest of the live matrix agree.
"""
struct ReplayField
    name::Symbol
    rows::Int
    cols::Int
    kind::Symbol              # :matrix or :vector — recorded, not inferred
    payloads::Vector{SessionScalarPayload}
end

function ReplayField(name::Symbol, A::AbstractMatrix)
    payloads = SessionScalarPayload[]
    for j in axes(A, 2), i in axes(A, 1)
        push!(payloads, session_scalar_payload(A[i, j]))
    end
    ReplayField(name, size(A, 1), size(A, 2), :matrix, payloads)
end

function ReplayField(name::Symbol, v::AbstractVector)
    payloads = [session_scalar_payload(x) for x in v]
    ReplayField(name, length(v), 1, :vector, payloads)
end

ReplayField(name::Symbol, x::Number) = ReplayField(name, fill(x, 1, 1))

# ---------------------------------------------------------------------------
# 2. the envelope
# ---------------------------------------------------------------------------

"""
    ReplayEnvelope

A complete, self-describing replay record.  `body_digest` covers every byte of
the encoded body except itself, so a single altered digit anywhere — including
in the SHA line — is a detected corruption rather than a plausible replay.
"""
struct ReplayEnvelope
    schema::String
    source_sha::String
    provider_revision::String
    arithmetic::ArithmeticFamily
    precision_bits::Int
    rounding::SessionRounding
    ordering::Symbol
    rank_transform::Symbol
    tolerances::Vector{SessionTolerance}
    fields::Vector{ReplayField}
    body_digest::UInt64
end

"""
    replay_envelope(; source_sha, fields, tolerances, ...) -> ReplayEnvelope

Build an envelope.  The SHA is validated as COMPLETE here, at construction, so a
record can never be written with a 7-character abbreviation that later has to be
trusted.
"""
function replay_envelope(; source_sha::AbstractString,
                         fields::Vector{ReplayField},
                         tolerances::Vector{SessionTolerance},
                         provider_revision::AbstractString="unspecified",
                         arithmetic::ArithmeticFamily=ArithFloat,
                         precision_bits::Integer=64,
                         rounding::SessionRounding=RoundingNearestEven,
                         ordering::Symbol=:natural,
                         rank_transform::Symbol=:full)
    replay_require_complete_sha(source_sha)
    env = ReplayEnvelope(REPLAY_SCHEMA, String(source_sha), String(provider_revision),
                         arithmetic, Int(precision_bits), rounding, ordering,
                         rank_transform, tolerances, fields, UInt64(0))
    ReplayEnvelope(env.schema, env.source_sha, env.provider_revision, env.arithmetic,
                   env.precision_bits, env.rounding, env.ordering, env.rank_transform,
                   env.tolerances, env.fields, replay_body_digest(env))
end

"""
    replay_require_complete_sha(sha)

完整SHA means complete.  A short prefix is rejected with its own code rather
than being accepted as "close enough": an abbreviated SHA cannot identify a
revision, and a replay file that claims to is lying about what it pins.
"""
function replay_require_complete_sha(sha::AbstractString)
    occursin(r"^[0-9a-f]{40}$", sha) ||
        throw(ReplayIntegrityError(:bad_sha,
            "source SHA must be 40 lowercase hex characters; got $(repr(sha)) " *
            "($(length(sha)) characters)"))
    true
end

session_field(env::ReplayEnvelope, name::Symbol) = begin
    hits = [f for f in env.fields if f.name === name]
    isempty(hits) && throw(ReplayIntegrityError(:missing_field, "no field named $(name)"))
    length(hits) > 1 && throw(ReplayIntegrityError(:duplicate_field,
        "$(length(hits)) fields named $(name)"))
    hits[1]
end

# ---------------------------------------------------------------------------
# 3. encoding — a line format, so that "no Float64 sideways door" is auditable
# ---------------------------------------------------------------------------

const _REPLAY_HEADER_KEYS = ("source_sha", "provider_revision", "arithmetic",
                             "precision_bits", "rounding", "ordering", "rank_transform")

# `text` is a human-readable label.  It is escaped so that it can never break the
# line format, and it is NEVER the decode source — see the `text` corruption
# control in the driver.
function _replay_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c === '%'
            print(io, "%25")
        elseif c in (' ', '\t', '\n', '\r')
            print(io, '%', string(UInt8(c), base=16, pad=2))
        else
            print(io, c)
        end
    end
    String(take!(io))
end

function _replay_unescape(s::AbstractString)
    io = IOBuffer()
    i = firstindex(s)
    while i <= lastindex(s)
        if s[i] == '%' && i + 2 <= lastindex(s)
            print(io, Char(parse(UInt8, s[i + 1:i + 2], base=16)))
            i += 3
        else
            print(io, s[i])
            i = nextind(s, i)
        end
    end
    String(take!(io))
end

_enum_token(x) = Symbol(lowercase(string(x)))

function _payload_line(p::SessionScalarPayload)
    string("p ", _replay_escape(String(p.kind)), " ", _replay_escape(p.type_name), " ",
           p.type_bits, " ", p.hex, " ", _replay_escape(p.text))
end

"""
    replay_body_lines(env) -> Vector{String}

The canonical body, WITHOUT the digest line.  Deterministic: the same envelope
encodes to the same bytes, so the digest is stable across runs.
"""
function replay_body_lines(env::ReplayEnvelope)
    lines = String[env.schema,
                   "source_sha $(env.source_sha)",
                   "provider_revision $(_replay_escape(env.provider_revision))",
                   "arithmetic $(_enum_token(env.arithmetic))",
                   "precision_bits $(env.precision_bits)",
                   "rounding $(_enum_token(env.rounding))",
                   "ordering $(_replay_escape(String(env.ordering)))",
                   "rank_transform $(_replay_escape(String(env.rank_transform)))"]
    for t in env.tolerances
        push!(lines, string("tolerance ", t.name, " ", t.atol_literal, " ", t.rtol_literal))
    end
    for f in env.fields
        push!(lines, string("field ", f.name, " ", f.kind, " ", f.rows, " ", f.cols, " ",
                            length(f.payloads)))
        for p in f.payloads
            push!(lines, _payload_line(p))
        end
    end
    lines
end

"""
    replay_body_digest(env) -> UInt64

Digest over the canonical body.  A one-character change anywhere in the body —
payload, SHA, tolerance literal — changes it.
"""
function replay_body_digest(env::ReplayEnvelope)
    h = SESSION_DIGEST_SEED
    for line in replay_body_lines(env)
        h = session_digest_mix(h, line)
        h = session_digest_mix(h, UInt8(0x0a))   # separator: line boundaries matter
    end
    h
end

function replay_encode(env::ReplayEnvelope)
    io = IOBuffer()
    for line in replay_body_lines(env)
        println(io, line)
    end
    println(io, "digest ", string(env.body_digest, base=16, pad=16))
    println(io, "end")
    String(take!(io))
end

# ---------------------------------------------------------------------------
# 4. decoding — every failure mode named
# ---------------------------------------------------------------------------

function _replay_parse_enum(::Type{T}, token::AbstractString, what) where {T}
    for v in instances(T)
        _enum_token(v) === Symbol(token) && return v
    end
    throw(ReplayIntegrityError(:bad_number,
        "$(what) $(repr(token)) is not one of $(join(string.(_enum_token.(instances(T))), ","))"))
end

function _replay_split_payload(line::AbstractString)
    # p <kind> <type_name> <type_bits> <hex> <text...>
    parts = split(line; limit=6)
    length(parts) == 6 && parts[1] == "p" ||
        throw(ReplayIntegrityError(:bad_field_line, "malformed payload line $(repr(line))"))
    kind = Symbol(_replay_unescape(parts[2]))
    type_name = _replay_unescape(parts[3])
    bits = try
        parse(Int, parts[4])
    catch err
        throw(ReplayIntegrityError(:bad_number, "payload width $(repr(parts[4])): $(err)"))
    end
    SessionScalarPayload(kind, type_name, bits, parts[5], _replay_unescape(parts[6]))
end

"""
    replay_decode(text) -> ReplayEnvelope

Inverse of [`replay_encode`](@ref).  Refuses — never guesses — on a foreign
schema, a truncated body, a missing `end`, a digest mismatch, or a malformed
line.  The digest is checked BEFORE the payload is interpreted, so a corrupted
body is reported as corruption rather than as a hundred unrelated parse errors.
"""
function replay_decode(text::AbstractString)
    lines = split(chomp(String(text)), '\n')
    isempty(lines) && throw(ReplayIntegrityError(:truncated, "empty replay body"))
    lines[1] == REPLAY_SCHEMA || throw(ReplayIntegrityError(:bad_schema,
        "expected schema $(REPLAY_SCHEMA), got $(repr(lines[1]))"))
    lines[end] == "end" || throw(ReplayIntegrityError(:truncated,
        "replay body does not end with `end`; last line is $(repr(lines[end]))"))

    digest_pos = findlast(l -> startswith(l, "digest "), lines)
    digest_pos === nothing && throw(ReplayIntegrityError(:truncated, "no digest line"))
    digest_pos == length(lines) - 1 || throw(ReplayIntegrityError(:truncated,
        "digest line is not the second-to-last line"))
    recorded = try
        parse(UInt64, split(lines[digest_pos])[2], base=16)
    catch err
        throw(ReplayIntegrityError(:bad_number, "digest line: $(err)"))
    end

    body = lines[1:digest_pos - 1]
    h = SESSION_DIGEST_SEED
    for line in body
        h = session_digest_mix(h, line)
        h = session_digest_mix(h, UInt8(0x0a))
    end
    h == recorded || throw(ReplayIntegrityError(:bad_digest,
        "body digest is $(string(h, base=16, pad=16)), header claims " *
        "$(string(recorded, base=16, pad=16))"))

    header = Dict{String,String}()
    tolerances = SessionTolerance[]
    fields = ReplayField[]
    current = nothing
    expected_payloads = 0
    for line in body[2:end]
        startswith(line, "tolerance ") && begin
            parts = split(line)
            length(parts) == 4 || throw(ReplayIntegrityError(:bad_field_line,
                "malformed tolerance line $(repr(line))"))
            atol = tryparse(Float64, parts[3])
            rtol = tryparse(Float64, parts[4])
            (atol === nothing || rtol === nothing) && throw(ReplayIntegrityError(:bad_number,
                "tolerance literals $(repr(parts[3])) / $(repr(parts[4])) are not numbers"))
            push!(tolerances, SessionTolerance(Symbol(parts[2]), atol, rtol, parts[3], parts[4]))
            continue
        end
        startswith(line, "field ") && begin
            parts = split(line)
            length(parts) == 6 || throw(ReplayIntegrityError(:bad_field_line,
                "malformed field line $(repr(line))"))
            kind = Symbol(parts[3])
            kind in (:matrix, :vector) || throw(ReplayIntegrityError(:bad_field_line,
                "field kind $(repr(parts[3])) is neither matrix nor vector"))
            n = try
                (parse(Int, parts[4]), parse(Int, parts[5]), parse(Int, parts[6]))
            catch err
                throw(ReplayIntegrityError(:bad_number, "field line $(repr(line)): $(err)"))
            end
            current = ReplayField(Symbol(parts[2]), n[1], n[2], kind, SessionScalarPayload[])
            expected_payloads = n[3]
            haskey(header, "field:" * parts[2]) && throw(ReplayIntegrityError(:duplicate_field,
                "field $(parts[2]) appears twice"))
            header["field:" * parts[2]] = "1"
            push!(fields, current)
            continue
        end
        if startswith(line, "p ")
            current === nothing && throw(ReplayIntegrityError(:bad_field_line,
                "payload line before any field line"))
            push!(current.payloads, _replay_split_payload(line))
            continue
        end
        parts = split(line; limit=2)
        length(parts) == 2 || throw(ReplayIntegrityError(:bad_field_line,
            "malformed header line $(repr(line))"))
        haskey(header, parts[1]) && throw(ReplayIntegrityError(:duplicate_field,
            "header key $(parts[1]) appears twice"))
        header[parts[1]] = parts[2]
    end
    for key in _REPLAY_HEADER_KEYS
        haskey(header, key) || throw(ReplayIntegrityError(:missing_field,
            "replay header has no $(key)"))
    end
    for f in fields
        length(f.payloads) == (f.kind === :vector ? f.rows : f.rows * f.cols) ||
            throw(ReplayIntegrityError(:truncated, "field $(f.name) declares " *
                "$(f.kind === :vector ? f.rows : f.rows * f.cols) scalars and carries " *
                "$(length(f.payloads))"))
    end

    ReplayEnvelope(REPLAY_SCHEMA, header["source_sha"], _replay_unescape(header["provider_revision"]),
                   _replay_parse_enum(ArithmeticFamily, header["arithmetic"], "arithmetic"),
                   parse(Int, header["precision_bits"]),
                   _replay_parse_enum(SessionRounding, header["rounding"], "rounding"),
                   Symbol(_replay_unescape(header["ordering"])),
                   Symbol(_replay_unescape(header["rank_transform"])),
                   tolerances, fields, recorded)
end

# ---------------------------------------------------------------------------
# 5. verification — the caller states what it expects
# ---------------------------------------------------------------------------

"""
    ReplayVerification

`failures` is a list of NAMED mismatches, empty iff `ok`.  A verification that
returned a bare `false` would force the caller to re-derive why, which is how a
version mismatch gets mistaken for a corrupt file.
"""
struct ReplayVerification
    ok::Bool
    failures::Vector{Symbol}
    detail::String
end

"""
    replay_verify(env; source_sha, arithmetic, precision_bits, rounding, ordering,
                  rank_transform, tolerances, require_fields, fingerprint)

Verify an envelope against what the caller is about to do with it.  Every named
expectation is optional; supplying none checks only the envelope's own internal
consistency (which `replay_decode` already did).
"""
function replay_verify(env::ReplayEnvelope;
                       source_sha::Union{Nothing,AbstractString}=nothing,
                       arithmetic::Union{Nothing,ArithmeticFamily}=nothing,
                       precision_bits::Union{Nothing,Integer}=nothing,
                       rounding::Union{Nothing,SessionRounding}=nothing,
                       ordering::Union{Nothing,Symbol}=nothing,
                       rank_transform::Union{Nothing,Symbol}=nothing,
                       tolerances::Union{Nothing,Vector{SessionTolerance}}=nothing,
                       require_fields::Union{Nothing,Vector{Symbol}}=nothing,
                       fingerprint::Union{Nothing,ProblemFingerprint}=nothing)
    failures = Symbol[]
    detail = String[]
    if source_sha !== nothing
        replay_require_complete_sha(source_sha)
        env.source_sha == source_sha || (push!(failures, :version_mismatch);
            push!(detail, "source SHA $(env.source_sha) != expected $(source_sha)"))
    end
    if arithmetic !== nothing && env.arithmetic !== arithmetic
        push!(failures, :version_mismatch)
        push!(detail, "arithmetic $(env.arithmetic) != expected $(arithmetic)")
    end
    if precision_bits !== nothing && env.precision_bits != precision_bits
        push!(failures, :version_mismatch)
        push!(detail, "precision $(env.precision_bits) != expected $(precision_bits)")
    end
    if rounding !== nothing && env.rounding !== rounding
        push!(failures, :version_mismatch)
        push!(detail, "rounding $(env.rounding) != expected $(rounding)")
    end
    if ordering !== nothing && env.ordering !== ordering
        push!(failures, :version_mismatch)
        push!(detail, "ordering $(env.ordering) != expected $(ordering)")
    end
    if rank_transform !== nothing && env.rank_transform !== rank_transform
        push!(failures, :version_mismatch)
        push!(detail, "rank transform $(env.rank_transform) != expected $(rank_transform)")
    end
    if tolerances !== nothing
        # Compared on the EXACT (atol, rtol) pair and on the recorded literal:
        # a replay that was judged by a different bound is not a replay of the
        # same experiment, even if the numbers look similar.
        for t in tolerances
            hit = findfirst(x -> x.name === t.name, env.tolerances)
            hit === nothing && (push!(failures, :tolerance_mismatch);
                push!(detail, "no recorded tolerance named $(t.name)"); continue)
            rec = env.tolerances[hit]
            (rec.atol == t.atol && rec.rtol == t.rtol &&
             rec.atol_literal == t.atol_literal && rec.rtol_literal == t.rtol_literal) ||
                (push!(failures, :tolerance_mismatch);
                 push!(detail, "tolerance $(t.name): recorded atol=$(rec.atol_literal) " *
                               "rtol=$(rec.rtol_literal), expected atol=$(t.atol_literal) " *
                               "rtol=$(t.rtol_literal)"))
        end
    end
    if require_fields !== nothing
        for name in require_fields
            any(f -> f.name === name, env.fields) ||
                (push!(failures, :missing_field); push!(detail, "no field $(name)"))
        end
    end
    if fingerprint !== nothing
        got = replay_fingerprint(env)
        got == fingerprint || (push!(failures, :fingerprint_mismatch);
            push!(detail, "replayed fingerprint $(got) != expected $(fingerprint)"))
    end
    ReplayVerification(isempty(failures), failures, join(detail, "; "))
end

# ---------------------------------------------------------------------------
# 6. reconstructing the raw arithmetic inputs
# ---------------------------------------------------------------------------

"""
    replay_field_scalars(f; mod=Main) -> Vector

Decode a field's scalars through the exact codec.  Mixed scalar types in one
field are an explicit refusal: a matrix whose elements came from two arithmetics
is not the matrix that ran.
"""
function replay_field_scalars(f::ReplayField; mod::Module=Main)
    isempty(f.payloads) && return Any[]
    first_type = f.payloads[1].type_name
    out = Any[]
    for p in f.payloads
        (p.type_name == first_type && p.kind === f.payloads[1].kind) ||
            throw(ReplayIntegrityError(:mixed_arithmetic,
                "field $(f.name) mixes $(f.payloads[1].kind)/$(first_type) with " *
                "$(p.kind)/$(p.type_name)"))
        push!(out, session_payload_scalar(p; mod=mod))
    end
    out
end

function replay_field_vector(env::ReplayEnvelope, name::Symbol; mod::Module=Main)
    f = session_field(env, name)
    f.kind === :vector || throw(ReplayIntegrityError(:bad_field_line,
        "field $(name) is $(f.kind), not a vector"))
    v = replay_field_scalars(f; mod=mod)
    isempty(v) && return Float64[]
    T = typeof(v[1])
    all(x -> typeof(x) === T, v) || throw(ReplayIntegrityError(:mixed_arithmetic,
        "field $(name) does not decode to one scalar type"))
    return T[x for x in v]
end

function replay_field_matrix(env::ReplayEnvelope, name::Symbol; mod::Module=Main)
    f = session_field(env, name)
    f.kind === :matrix || throw(ReplayIntegrityError(:bad_field_line,
        "field $(name) is $(f.kind), not a matrix"))
    v = replay_field_scalars(f; mod=mod)
    isempty(v) && return Matrix{Float64}(undef, 0, 0)
    T = typeof(v[1])
    all(x -> typeof(x) === T, v) || throw(ReplayIntegrityError(:mixed_arithmetic,
        "field $(name) does not decode to one scalar type"))
    A = Matrix{T}(undef, f.rows, f.cols)
    k = 0
    for j in 1:f.cols, i in 1:f.rows
        A[i, j] = v[k += 1]
    end
    A
end

"""
    replay_fingerprint(env; mod=Main) -> ProblemFingerprint

Rebuild the update-table fingerprint from the replayed raw inputs.  This is the
bridge between the two halves of S07: a `ProblemFingerprint` reconstructed from a
file is comparable, with `==`, against one taken from a live problem, so "the
replay describes the same problem" is a checkable statement and not a hope.
"""
function replay_fingerprint(env::ReplayEnvelope; mod::Module=Main)
    A = any(f -> f.name === :operator, env.fields) ?
        replay_field_matrix(env, :operator; mod=mod) : nothing
    b = any(f -> f.name === :rhs, env.fields) ?
        replay_field_vector(env, :rhs; mod=mod) : nothing
    c = any(f -> f.name === :objective, env.fields) ?
        replay_field_vector(env, :objective; mod=mod) : nothing
    cp = any(f -> f.name === :cone_parameters, env.fields) ?
        replay_field_vector(env, :cone_parameters; mod=mod) : nothing
    session_problem_fingerprint(A=A, b=b, c=c, cone_parameters=cp,
                                arithmetic=env.arithmetic,
                                precision_bits=env.precision_bits,
                                rounding=env.rounding, ordering=env.ordering,
                                rank_transform=env.rank_transform)
end

# ---------------------------------------------------------------------------
# 7. a serialized verdict is not evidence
# ---------------------------------------------------------------------------

"""
    replay_state_is_not_evidence(claim) -> Bool

The executable form of the card's second acceptance item: a checkpoint's
serialized solver state — including one that says `optimal` — is a CLAIM, and a
restore must re-admit the problem and recompute the certificate before that claim
means anything.  This predicate is `false` for every claim, by construction, so a
caller cannot accidentally use it as a shortcut.  It exists so that the refusal is
a named object in the code rather than a convention.
"""
replay_state_is_not_evidence(claim::Symbol) = false
replay_state_is_not_evidence(claim) = false

"""
    replay_restore_requirements(claim) -> NamedTuple

What a checkpoint restore actually needs, printed rather than assumed.
"""
replay_restore_requirements(claim::Symbol) = (
    claimed_state=claim,
    claim_is_evidence=false,
    readmission_required=true,
    certificate_recompute_required=true,
    rank_authority=RankAuthorityRevoked,
    reason="a serialized state is a claim about a run; admission and the " *
           "certificate are properties of the arithmetic as executed, and are " *
           "recomputed from the replayed raw inputs",
)
