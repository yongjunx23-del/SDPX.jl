# SOC metric as a diagonal plus rank-2 update (plan PR-02 / finding F03).
#
# The frozen symmetric core stores the cone metric `Theta` as a dense lower
# triangle per cone block, so a block of size `k` costs `k(k+1)/2` slots. For a
# single large second-order cone that dominates the KKT storage.
#
# A second-order cone metric has a much smaller representation:
#
#     Theta = D + u*u' - v*v'
#
# with `D` diagonal. Embedding the two rank-2 terms as two auxiliary KKT
# unknowns replaces the `k(k+1)/2` dense triangle with about `3k+2` cone-local
# slots, and `validation/clarabel_borrowing/soc_rank2_gate.jl` proves that
# eliminating those auxiliaries reproduces the dense block exactly -- a
# representation change of the linear system, not an approximation of the cone.
#
# ## Why the parameters here are NOT Clarabel's formulas
#
# The plan warns explicitly: establish the correspondence between SDPX's Theta
# and this H first, and "do not take a similarly-named `w` from the SDPX NT
# state and apply the formula directly." That warning is correct, and this file
# is the result of heeding it.
#
# Measured from the executable kernel `SymmetricCones.quadratic_apply!(cone,
# out, w, z)` -- which is exactly what `theta_apply!` calls -- SDPX's operator is
#
#     alpha = w0^2 + ww,   beta = w0^2 - ww,   ww = sum(w[i]^2, i >= 2)
#
#     Theta = [ alpha          2*w0*w_tail'                ]
#             [ 2*w0*w_tail    beta*I + 2*w_tail*w_tail'   ]
#
# whereas Clarabel's `Q_w = 2*w*w' - J` has `2*w0^2 - 1` in the (1,1) entry.
# The two differ **only** in that entry, by `2*ww`. SDPX's `nt_scaling!`
# therefore does not produce Clarabel's normalized `w`: verified numerically,
# `w0^2 - ww` is not 1 for SDPX scaling points, and Clarabel's `(D,u,v)` formulas
# reproduce SDPX's `Theta` only to ~1e-1, not to working precision.
#
# The parameters below are derived from SDPX's own structure instead. Writing
# the target as `D + u*u' - v*v'` with
#
#     u = [u0; u1*w_tail],   v = [0; v1*w_tail],   D = [d0; beta*ones]
#
# the requirement that `u*u' - v*v'` contribute `2*w0*w_tail` to the first
# column and `2*w_tail*w_tail'` to the tail block gives `u0*u1 = 2*w0` and
# `u1^2 - v1^2 = 2`. Choosing `u0 = sqrt(alpha/2)` fixes `u1 = 2*w0/u0` and
# `v1 = sqrt(u1^2 - 2)`, and then `d0 = alpha - u0^2`. This is an exact identity,
# verified to 2.2e-16 against `theta_apply!` for k in 3..512.
#
# This file owns only that mapping. It does not change any existing route, and
# wiring it into `SymmetricCorePattern` is a separate step that must keep the
# dense path for small blocks (measured crossover: k = 6).

# `BigFloat` is mutable, so storing one scalar into another slot aliases the
# MPFR object. This file is included before `kkt/symmetric_core.jl` (which owns
# the same idea as `_core_store_owned!`), so it carries its own copy to avoid an
# include-order dependency. Only cold setup paths call it.
@inline _soc_rank2_owned(value) = value
@inline _soc_rank2_owned(value::BigFloat) = MutableArithmetics.mutable_copy(value)

@inline function _soc_rank2_store!(destination::AbstractArray, index::Int, value)
    destination[index] = _soc_rank2_owned(value)
    return destination
end

"""
    soc_rank2_parameters(w) -> (D, u, v)

Map an SDPX SOC scaling point to the diagonal-plus-rank-2 form of the metric
that `theta_apply!` actually applies, so that

    D + u*u' - v*v' == Theta

holds to working precision.

`w` is `SOCNTScaling.w` as produced by `nt_scaling!`; the header above documents
why Clarabel's formulas do **not** apply to it unchanged. Returns owned vectors;
`D` is the diagonal.

Throws `ArgumentError` if the scaling point does not yield a usable metric
(non-finite or degenerate `w`, or a non-positive diagonal), so a caller fails
closed rather than building a silently wrong operator.
"""
function soc_rank2_parameters(w::AbstractVector{T}) where {T<:AbstractFloat}
    k = length(w)
    k >= 2 || throw(ArgumentError(
        "SOC rank-2 parameters require a cone dimension of at least 2",
    ))
    w0 = w[1]
    ww = zero(T)
    @inbounds for i in 2:k
        ww += w[i] * w[i]
    end
    isfinite(w0) && isfinite(ww) || throw(ArgumentError(
        "SOC scaling point contains non-finite data",
    ))
    alpha = w0 * w0 + ww
    beta = w0 * w0 - ww
    # The tail diagonal of Theta is `beta`; the operator is only a valid cone
    # metric when that is positive, which is strict interiority in disguise.
    beta > zero(T) || throw(ArgumentError(
        "SOC scaling point is not strictly interior (beta <= 0)",
    ))
    alpha > zero(T) || throw(ArgumentError(
        "SOC scaling point yields a non-positive leading diagonal",
    ))

    # u0 = sqrt(alpha/2) makes u0*u1 = 2*w0 with u1 = 2*w0/u0.
    u0 = sqrt(alpha / T(2))
    isfinite(u0) && u0 > zero(T) || throw(ArgumentError(
        "SOC rank-2 positive term is not resolvable",
    ))
    u1 = (T(2) * w0) / u0
    # u1^2 - v1^2 must equal 2; alpha > 0 and beta > 0 guarantee u1^2 > 2.
    v1_squared = u1 * u1 - T(2)
    v1_squared >= zero(T) || throw(ArgumentError(
        "SOC rank-2 negative term is not resolvable",
    ))
    v1 = sqrt(v1_squared)

    D = _cone_zeros(T, k)
    u = _cone_zeros(T, k)
    v = _cone_zeros(T, k)

    # Leading diagonal absorbs the part of u*u' the tail does not carry.
    _soc_rank2_store!(D, 1, alpha - u0 * u0)
    @inbounds for i in 2:k
        _soc_rank2_store!(D, i, beta)
    end
    _soc_rank2_store!(u, 1, u0)
    @inbounds for i in 2:k
        _soc_rank2_store!(u, i, u1 * w[i])
    end
    # v has a structurally zero leading entry, which is what keeps the first
    # column at exactly 2*w0*w_tail.
    _soc_rank2_store!(v, 1, zero(T))
    @inbounds for i in 2:k
        _soc_rank2_store!(v, i, v1 * w[i])
    end

    return D, u, v
end

"""
    soc_rank2_cone_slots(k) -> Int

Cone-local numerical slots the expanded representation needs for a
`k`-dimensional SOC block: `k` diagonal entries, `2k` coupling-column entries,
and the two auxiliary diagonals.

The measured crossover against the packed dense lower triangle (`k(k+1)/2`) is
`k = 6`; below that the dense block is smaller and must be kept. Established by
`validation/clarabel_borrowing/soc_rank2_gate.jl`.
"""
soc_rank2_cone_slots(k::Integer) = 3 * Int(k) + 2

"""
    soc_rank2_prefer_expanded(k) -> Bool

Whether the expanded representation is smaller than the packed dense lower
triangle for a `k`-dimensional SOC block. True from `k = 6` upward.
"""
soc_rank2_prefer_expanded(k::Integer) =
    soc_rank2_cone_slots(k) < div(Int(k) * (Int(k) + 1), 2)
