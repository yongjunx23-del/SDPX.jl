# Verification-only predicate for EXTRA Float64 inverse-publication candidates.
# No floating-point value, factor, solve, precision or rounding context is changed.
# This is NOT a whole-solver storage bound or a certificate for the legacy midpoint.
const _NS_FLOAT64_SPD_MAX_BITS = 6400
const _NS_FLOAT64_FRACTION_MASK = UInt64(0x000fffffffffffff)
const _NS_FLOAT64_MAGNITUDE_MASK = UInt64(0x7fffffffffffffff)

@inline function _ns_float64_dyadic_parts(word::UInt64)
    exponent = Int((word >> 52) & UInt64(0x7ff))
    exponent == 0x7ff && throw(ArgumentError("nonfinite Float64 has no finite dyadic encoding"))
    fraction = word & _NS_FLOAT64_FRACTION_MASK
    significand = iszero(exponent) ? fraction : fraction | (UInt64(1) << 52)
    shift = iszero(exponent) ? 0 : exponent - 1
    return significand, shift
end

"""Exact integer 2^1074*x from a finite Float64 word, not floating rescaling."""
function _ns_float64_dyadic1074_word(word::UInt64)
    significand, shift = _ns_float64_dyadic_parts(word)
    value = BigInt(significand) << shift
    return iszero(word >> 63) ? value : -value
end

@inline function _ns_float64_dyadic_width(word::UInt64)
    significand, shift = _ns_float64_dyadic_parts(word)
    return iszero(significand) ? 0 : 64 - leading_zeros(significand) + shift
end

@inline _ns_float64_real_bits(word::UInt64) =
    iszero(word & _NS_FLOAT64_MAGNITUDE_MASK) ? zero(UInt64) : word

"""
    _ns_float64_exact_spd3_veto(B; max_bits=6400) -> Bool

Exact Sylvester-sign veto for a stored Matrix{Float64} of size (3,3). Only
new Float64 symmetric selections use this additional check; it cannot override
any native rejection, generate entries, or certify unchanged midpoint/BigFloat
paths. Signed zeros denote the same real value; every other finite Float64 has
unique real-value bits. Nonfinite, asymmetric, unsupported or over-budget inputs
reject. Solver-owned B must not be concurrently mutated.

Finite encoded entries have at most 2098 magnitude bits. With largest width w,
the fixed expression below has intermediates bounded by 3w+3 bits (at most6297):
products of two entries need at most2w, differences at most2w+1, and the three
cubic terms plus their sums at most3w+3. This conservative preflight runs BEFORE
BigInt construction. At most11 BigInt multiplications, with no adaptive search.
The cap bounds exact integer magnitudes/operation count, not allocator/GMP scratch.
"""
function _ns_float64_exact_spd3_veto(B; max_bits::Integer=_NS_FLOAT64_SPD_MAX_BITS)
    B isa Matrix{Float64} && size(B) == (3,3) || return false
    0 < max_bits <= _NS_FLOAT64_SPD_MAX_BITS || return false
    words = ntuple(i -> reinterpret(UInt64, B[i]), 9)
    any(w -> ((w >> 52) & UInt64(0x7ff)) == UInt64(0x7ff), words) && return false
    # Column-major symmetric entries: compare bits, not rounded arithmetic.
    for (i,j) in ((2,4),(3,7),(6,8))
        _ns_float64_real_bits(words[i]) == _ns_float64_real_bits(words[j]) || return false
    end
    selected = (words[1],words[4],words[7],words[5],words[8],words[9])
    width = maximum(_ns_float64_dyadic_width, selected)
    3width + 3 <= max_bits || return false
    a,b,c,d,e,f = map(_ns_float64_dyadic1074_word, selected)
    a > 0 || return false
    a*d - b*b > 0 || return false
    return a*(d*f-e*e) - b*(b*f-c*e) + c*(b*e-c*d) > 0
end
