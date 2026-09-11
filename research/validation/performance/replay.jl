# Replay of accepted points, Newton epochs, certificates, first divergence.
#
# Purpose (P0-02): given two runs (e.g. serial reference vs threaded run, or
# baseline vs candidate commit), report the FIRST iteration whose recorded
# fields diverge and WHICH field diverged, so a reviewer can see exactly
# where determinism broke instead of diffing raw traces.
#
# Per iteration record: accepted point payload (digest always; full bytes
# only when the recorder is built with `keep_full=true`), alpha, sigma,
# direction payload-or-digest, actual RHS payload-or-digest, gate results
# (as plain booleans/strings), and the policy id string.
#
# Memory discipline: the rolling SHA-256 digest is updated incrementally per
# iteration, so the default recorder is O(1) in the problem size beyond the
# current iterate. Full per-iteration payloads are retained ONLY on demand
# (`keep_full=true`).
#
# Witnessed errors (iteration records whose gate results contain a failure)
# are reported sorted by original iteration index, so scheduling cannot
# change which failure is presented first.

using SHA: sha256, SHA2_256_CTX, update!, digest!

if !isdefined(@__MODULE__, :__PERF_REPLAY_LOADED__)
@eval(@__MODULE__, const __PERF_REPLAY_LOADED__ = true)
end

# Field names compared by `compare_replays`, in comparison order. The order
# is fixed so the reported "first differing field" is schedule-independent.
const REPLAY_COMPARE_FIELDS = (
    :point_digest,
    :alpha,
    :sigma,
    :direction_digest,
    :rhs_digest,
    :gates,
    :policy,
)

struct ReplayRecorder
    policy::String
    keep_full::Bool
    ctx::SHA2_256_CTX
    count::Ref{Int}
    entries::Vector{NamedTuple}
end

"""
    ReplayRecorder(policy; keep_full=false) -> ReplayRecorder

Create a recorder. `policy` is a free-form policy id string (solver
settings digest, thread budget, commit SHA, ...). With `keep_full=false`
(the default) only digests plus scalar fields are retained per iteration;
with `keep_full=true` the full payload byte vectors are kept as well.
"""
function ReplayRecorder(policy::AbstractString; keep_full::Bool=false)
    return ReplayRecorder(
        String(policy), keep_full, SHA2_256_CTX(), Ref(0), NamedTuple[],
    )
end

_digest_of_bytes(b::Vector{UInt8}) = bytes2hex(sha256(b))

"""
    record_iteration!(rec, iter, point_bytes, alpha, sigma, direction_bytes,
                      rhs_bytes, gates, policy) -> entry NamedTuple

Append one iteration record and fold it into the rolling digest. `gates` is
any collection of gate outcomes; it is normalized to a sorted
`Vector{String}` (`"name=true|false"`) so record comparison is order-stable.
"""
function record_iteration!(
    rec::ReplayRecorder,
    iter::Integer,
    point_bytes::Vector{UInt8},
    alpha::Real,
    sigma::Real,
    direction_bytes::Vector{UInt8},
    rhs_bytes::Vector{UInt8},
    gates,
    policy::AbstractString,
)
    gate_strings = sort!(String[
        string(name, "=", value === true ? "true" :
            value === false ? "false" : string(value))
        for (name, value) in gates
    ])
    point_digest = _digest_of_bytes(point_bytes)
    direction_digest = _digest_of_bytes(direction_bytes)
    rhs_digest = _digest_of_bytes(rhs_bytes)
    # Rolling digest folds every compared field in a fixed order.
    update!(rec.ctx, Vector{UInt8}(point_digest))
    update!(rec.ctx, Vector{UInt8}(string(Float64(alpha), "/", Float64(sigma))))
    update!(rec.ctx, Vector{UInt8}(direction_digest))
    update!(rec.ctx, Vector{UInt8}(rhs_digest))
    for g in gate_strings
        update!(rec.ctx, Vector{UInt8}(g))
    end
    update!(rec.ctx, Vector{UInt8}(String(policy)))
    rec.count[] += 1
    entry = (
        iter=Int(iter),
        point_digest=point_digest,
        alpha=Float64(alpha),
        sigma=Float64(sigma),
        direction_digest=direction_digest,
        rhs_digest=rhs_digest,
        gates=gate_strings,
        policy=String(policy),
        point_bytes=rec.keep_full ? copy(point_bytes) : UInt8[],
        direction_bytes=rec.keep_full ? copy(direction_bytes) : UInt8[],
        rhs_bytes=rec.keep_full ? copy(rhs_bytes) : UInt8[],
    )
    push!(rec.entries, entry)
    return entry
end

"""
    finalize_replay(rec) -> NamedTuple

Close the recorder: `(policy, count, rolling_digest, entries)` where
`entries` is only populated when `keep_full=true` (otherwise it is the
empty vector and the rolling digest is the evidence).
"""
function finalize_replay(rec::ReplayRecorder)
    snapshot = rec.ctx
    rolling = bytes2hex(digest!(copy(snapshot)))
    return (
        policy=rec.policy,
        count=rec.count[],
        rolling_digest=rolling,
        entries=rec.keep_full ? copy(rec.entries) : NamedTuple[],
        keep_full=rec.keep_full,
    )
end

# Compare two gate vectors (already normalized/sorted at record time).
_gates_equal(a::Vector{String}, b::Vector{String}) = a == b

"""
    compare_replays(a, b) -> NamedTuple

Compare two finalized replays (or two entry vectors). Returns
`(equal::Bool, first_iter::Union{Nothing,Int}, field::Union{Nothing,Symbol},
detail::String)`: the first iteration index (original `iter`, not position)
at which a compared field differs, and which field it was. Counts that
differ are reported with `field=:count`.
"""
function compare_replays(a::NamedTuple, b::NamedTuple)
    ea = a.entries
    eb = b.entries
    if isempty(ea) || isempty(eb)
        # Digest-only comparison: rolling digests decide, no field detail.
        if a.rolling_digest == b.rolling_digest && a.count == b.count
            return (equal=true, first_iter=nothing, field=nothing,
                detail="rolling digests and counts agree")
        end
        return (equal=false, first_iter=nothing,
            field=(a.count != b.count ? :count : :rolling_digest),
            detail="digest-only comparison: counts $(a.count) vs $(b.count)")
    end
    n = min(length(ea), length(eb))
    for k in 1:n
        ra = ea[k]
        rb = eb[k]
        for field in REPLAY_COMPARE_FIELDS
            va = getfield(ra, field)
            vb = getfield(rb, field)
            equal = field === :gates ? _gates_equal(va, vb) : isequal(va, vb)
            if !equal
                return (equal=false, first_iter=ra.iter, field=field,
                    detail="iteration $(ra.iter): field $field diverged " *
                           "($(va) vs $(vb))")
            end
        end
    end
    if length(ea) != length(eb)
        return (equal=false,
            first_iter=(length(ea) > length(eb) ? ea[n + 1].iter : eb[n + 1].iter),
            field=:count,
            detail="lengths differ: $(length(ea)) vs $(length(eb))")
    end
    return (equal=true, first_iter=nothing, field=nothing,
        detail="all $(length(ea)) iterations agree")
end

"""
    witnessed_failures(entries) -> Vector{NamedTuple}

Return the entries whose normalized gate list contains a `false` outcome,
sorted by original iteration index so scheduling cannot change which
failure is reported first.
"""
function witnessed_failures(entries::Vector)
    bad = filter(
        e -> any(g -> endswith(g, "=false"), e.gates),
        entries,
    )
    return sort!(copy(bad); by=e -> e.iter)
end
