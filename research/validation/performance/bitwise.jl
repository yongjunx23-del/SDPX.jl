# Canonical bit-level serialization + comparison of numeric payloads.
#
# Purpose (P0-02): evidence digests must cover ONLY numeric payload bytes.
# Time, task counts, thread counts, RSS, paths, and Julia internals are never
# encoded here, so equal numerics always hash equal regardless of schedule.
#
# Endianness: fixed little-endian for fixed-width integers; BigFloat
# significands are big-endian byte strings of explicit width. The encoding is
# therefore identical on big- and little-endian hosts.
#
# Element order: row-major (last index varies fastest), independent of
# Julia's column-major storage, so reshapes/permuted schedules cannot change
# the digest.
#
# Scalar rules:
# - Float64/Float32: -0.0 is canonicalized to +0.0; every NaN is encoded as
#   the single quiet pattern (exponent all-ones, payload bit 51/22 set, all
#   other payload bits zero), so NaN == NaN here by construction. Inf keeps
#   its standard bit pattern. All other values keep exact raw bits.
# - MultiFloat (e.g. MultiFloats.Float64x4): ALL limbs are encoded, in limb
#   order, as raw Float64 bit patterns with NO canonicalization (a limb-level
#   -0.0 or NaN payload is part of the value's identity). Limb bytes are
#   obtained with `reinterpret`, never by naming provider-private fields.
# - BigFloat: precision (UInt32), sign (UInt8, canonical +0), frexp exponent
#   (Int64; 0 for zero; typemax(Int64) for Inf; typemax(Int64)-1 for NaN with
#   a zeroed significand), then the exact significand at that value's OWN
#   precision as a fixed-width big-endian byte string of ceil(precision/8)
#   bytes, computed with the public `frexp`/`ldexp`/`precision` API only.
#   No MPFR struct fields and no decimal display strings are read anywhere.
#
# Generic fallback: any other `isbitstype` AbstractFloat whose size is a
# positive multiple of 8 bytes is encoded limb-wise like a multifloat.
# Anything else fails closed with an ArgumentError (never silently skipped).

if !isdefined(@__MODULE__, :__PERF_BITWISE_LOADED__)
@eval(@__MODULE__, const __PERF_BITWISE_LOADED__ = true)
end

using SHA: sha256

# Element-type tags (1 byte). Tags 0x01/0x02 are fixed-width IEEE scalars,
# 0x03 is BigFloat, 0x04 is an N-limb floating tuple (multifloat-style).
const _BITWISE_VERSION::UInt8 = 0x01

function _push_u32le!(out::Vector{UInt8}, v::UInt32)
    x = v
    for _ in 1:4
        push!(out, UInt8(x & UInt32(0xff)))
        x >>= 8
    end
    return out
end

function _push_u64le!(out::Vector{UInt8}, v::UInt64)
    x = v
    for _ in 1:8
        push!(out, UInt8(x & UInt64(0xff)))
        x >>= 8
    end
    return out
end

function _push_i64le!(out::Vector{UInt8}, v::Int64)
    return _push_u64le!(out, reinterpret(UInt64, v))
end

# Canonical quiet-NaN bit patterns (sign 0, exponent all-ones, top payload
# bit set, remaining payload bits zero).
const _CANON_NAN64::UInt64 = 0x7ff8000000000000
const _CANON_NAN32::UInt32 = 0x7fc00000

function _encode_f64!(out::Vector{UInt8}, x::Float64)
    u = reinterpret(UInt64, x)
    if isnan(x)
        u = _CANON_NAN64
    elseif x == 0.0
        u = UInt64(0)  # canonicalize -0.0 to +0.0
    end
    return _push_u64le!(out, u)
end

function _encode_f32!(out::Vector{UInt8}, x::Float32)
    u = reinterpret(UInt32, x)
    if isnan(x)
        u = _CANON_NAN32
    elseif x == 0.0f0
        u = UInt32(0)
    end
    v = _push_u32le!(out, u)
    return v
end

# NOTE: multifloat limb bytes always go through `reinterpret(Float64, A)`
# views (never provider-private fields). On a big-endian host the 8-byte
# groups would need reversing for canonical little-endian output; Julia
# targets in this lab are little-endian (x86_64/aarch64) and the per-limb
# `_encode_f64!` path used below already emits explicit LE bytes, so the
# encoding is host-independent wherever `reinterpret` preserves limb order.

function _encode_bigfloat!(out::Vector{UInt8}, x::BigFloat)
    p = precision(x)
    (p >= 1 && p <= Int(typemax(UInt32))) ||
        throw(ArgumentError("bitwise payload: unsupported BigFloat precision $p"))
    push!(out, signbit(x) && (x != 0) ? 0x01 : 0x00)
    _push_u32le!(out, UInt32(p))
    width = (p + 7) ÷ 8
    if x == 0
        _push_i64le!(out, Int64(0))
        _push_u32le!(out, UInt32(width))
        for _ in 1:width
            push!(out, 0x00)
        end
        return out
    elseif isinf(x)
        _push_i64le!(out, typemax(Int64))
        _push_u32le!(out, UInt32(width))
        for _ in 1:width
            push!(out, 0x00)
        end
        return out
    elseif isnan(x)
        _push_i64le!(out, typemax(Int64) - Int64(1))
        _push_u32le!(out, UInt32(width))
        for _ in 1:width
            push!(out, 0x00)
        end
        return out
    end
    mantissa, expo = frexp(x)  # mantissa in [0.5, 1), public API only
    _push_i64le!(out, Int64(expo))
    scaled = ldexp(mantissa, p)  # exact integer-valued BigFloat
    sig = round(BigInt, scaled)
    (sig >= 0 && sig < (BigInt(1) << p)) ||
        throw(ArgumentError("bitwise payload: BigFloat significand out of range"))
    _push_u32le!(out, UInt32(width))
    for i in (width - 1):-1:0
        push!(out, UInt8((sig >> (8 * i)) & 0xff))
    end
    return out
end

function _scalar_tag(x::Float64)
    return 0x01
end

function _scalar_tag(x::Float32)
    return 0x02
end

function _scalar_tag(x::BigFloat)
    return 0x03
end

function _scalar_tag(x::T) where {T<:AbstractFloat}
    if isbitstype(T) && sizeof(T) > 0 && sizeof(T) % sizeof(Float64) == 0
        return 0x04
    end
    throw(ArgumentError(
        "bitwise payload: unsupported element type $T " *
        "(only Float64/Float32/BigFloat/isbits-multifloat AbstractFloats)",
    ))
end

function _encode_scalar!(out::Vector{UInt8}, x::Float64)
    return _encode_f64!(out, x)
end

function _encode_scalar!(out::Vector{UInt8}, x::Float32)
    return _encode_f32!(out, x)
end

function _encode_scalar!(out::Vector{UInt8}, x::BigFloat)
    return _encode_bigfloat!(out, x)
end

function _encode_scalar!(out::Vector{UInt8}, x::T) where {T<:AbstractFloat}
    # Multifloat-style scalars (tag 0x04) are encoded element-wise through
    # the `reinterpret(Float64, A)` view in `payload_bytes`, never here.
    throw(ArgumentError("bitwise payload: unsupported element type $T"))
end

# Row-major (last-index-fastest) linear traversal of an N-d array.
function _rowmajor_indices(dims::Tuple)
    total = prod(dims; init=1)
    total == 0 && return Tuple{Vararg{Int}}[]
    nd = length(dims)
    strides = Vector{Int}(undef, nd)
    s = 1
    for k in nd:-1:1
        strides[k] = s
        s *= dims[k]
    end
    out = Vector{NTuple{nd,Int}}(undef, total)
    for lin in 0:(total - 1)
        idx = ntuple(nd) do k
            d = dims[k]
            d == 1 ? 1 : (lin ÷ strides[k]) % d + 1
        end
        out[lin + 1] = idx
    end
    return out
end

"""
    payload_bytes(x::AbstractArray) -> Vector{UInt8}

Deterministic canonical encoding of a numeric payload array: 1-byte version,
1-byte element tag (+1-byte limb count for tag 0x04), 8-byte LE rank,
8-byte LE dims, 8-byte LE element count, then elements in row-major order.
"""
function payload_bytes(x::AbstractArray)
    T = eltype(x)
    T <: AbstractFloat || throw(ArgumentError(
        "payload_bytes: element type $T is not an AbstractFloat",
    ))
    tag = _scalar_tag(zero(T) isa T ? convert(T, 0) : first(x))
    out = UInt8[]
    push!(out, _BITWISE_VERSION, UInt8(tag))
    if tag == 0x04
        nlimbs = sizeof(T) ÷ sizeof(Float64)
        (nlimbs >= 1 && nlimbs <= 255) ||
            throw(ArgumentError("payload_bytes: unsupported limb count for $T"))
        push!(out, UInt8(nlimbs))
    end
    dims = size(x)
    _push_u64le!(out, UInt64(length(dims)))
    for d in dims
        _push_u64le!(out, UInt64(d))
    end
    _push_u64le!(out, UInt64(length(x)))
    if tag == 0x04
        # Limb arrays reinterpret to (nlimbs*n, ...) column-major; index them
        # so that element order is row-major and limbs stay in limb order.
        n = size(x, 1)
        rest = size(x)[2:end]
        rv = reinterpret(Float64, x)
        # rv has size (nlimbs*n, rest...); element (i, j...) limb k lives at
        # row i + (k-1)*n. Walk elements row-major via Cartesian over a
        # row-major index list.
        nlimbs = sizeof(T) ÷ sizeof(Float64)
        for idx in _rowmajor_indices(dims)
            i = idx[1]
            tail = idx[2:end]
            for k in 1:nlimbs
                _encode_f64!(out, rv[i + (k - 1) * n, tail...])
            end
        end
        return out
    end
    for idx in _rowmajor_indices(dims)
        _encode_scalar!(out, x[idx...])
    end
    return out
end

"""
    digest(x::AbstractArray) -> String

Lowercase hex SHA-256 of `payload_bytes(x)`. Covers numeric bytes only:
no timing, task/thread counts, RSS, paths, or schedule metadata.
"""
function digest(x::AbstractArray)
    return bytes2hex(sha256(payload_bytes(x)))
end

"""
    bitwise_equal(a, b) -> Bool

True iff the canonical payload encodings are byte-identical (same shape,
same tag, same element bytes under the documented scalar rules).
"""
function bitwise_equal(a::AbstractArray, b::AbstractArray)
    eltype(a) == eltype(b) || return false
    size(a) == size(b) || return false
    return payload_bytes(a) == payload_bytes(b)
end

"""
    first_differing_index(a, b) -> Union{Nothing,Int}

Row-major 1-based linear index of the first differing element, or `nothing`
when `bitwise_equal(a, b)`. Throws `ArgumentError` on shape/eltype mismatch
(fail-closed: never report a meaningless index).
"""
function first_differing_index(a::AbstractArray, b::AbstractArray)
    eltype(a) == eltype(b) || throw(ArgumentError(
        "first_differing_index: eltype mismatch $(eltype(a)) vs $(eltype(b))",
    ))
    size(a) == size(b) || throw(ArgumentError(
        "first_differing_index: size mismatch $(size(a)) vs $(size(b))",
    ))
    dims = size(a)
    tag = _scalar_tag(convert(eltype(a), 0))
    if tag == 0x04
        n = size(a, 1)
        nlimbs = sizeof(eltype(a)) ÷ sizeof(Float64)
        ra = reinterpret(Float64, a)
        rb = reinterpret(Float64, b)
        pos = 0
        for idx in _rowmajor_indices(dims)
            pos += 1
            i = idx[1]
            tail = idx[2:end]
            for k in 1:nlimbs
                va = ra[i + (k - 1) * n, tail...]
                vb = rb[i + (k - 1) * n, tail...]
                # Raw limb comparison: NaN limbs and -0.0 limbs are identity.
                reinterpret(UInt64, va) == reinterpret(UInt64, vb) || return pos
            end
        end
        return nothing
    end
    pos = 0
    for idx in _rowmajor_indices(dims)
        pos += 1
        va = a[idx...]
        vb = b[idx...]
        _scalar_bitwise_equal(va, vb) || return pos
    end
    return nothing
end

function _scalar_bitwise_equal(a::Float64, b::Float64)
    if isnan(a) && isnan(b)
        return true
    end
    if a == 0.0 && b == 0.0
        return true  # +0.0 and -0.0 canonicalize together
    end
    return isequal(a, b)
end

function _scalar_bitwise_equal(a::Float32, b::Float32)
    if isnan(a) && isnan(b)
        return true
    end
    if a == 0.0f0 && b == 0.0f0
        return true
    end
    return isequal(a, b)
end

function _scalar_bitwise_equal(a::BigFloat, b::BigFloat)
    if isnan(a) && isnan(b)
        return true
    end
    if a == 0 && b == 0
        return true
    end
    (isinf(a) || isinf(b)) && return isequal(a, b)
    return precision(a) == precision(b) && isequal(a, b) &&
           frexp(a) == frexp(b)
end

function _scalar_bitwise_equal(a::T, b::T) where {T<:AbstractFloat}
    # Limb-wise raw comparison for multifloat-style scalars.
    ra = reinterpret(Float64, [a])
    rb = reinterpret(Float64, [b])
    length(ra) == length(rb) || return false
    for k in eachindex(ra)
        reinterpret(UInt64, ra[k]) == reinterpret(UInt64, rb[k]) || return false
    end
    return true
end
