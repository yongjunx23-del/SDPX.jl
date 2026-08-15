# ============================================================================
# Provider-neutral cone cold-start math helpers (Phase 2)
#
# These helpers compute the arithmetic-only shifts used to pull a cold-start
# point strictly into the interior of each cone family (orthant / LP, Lorentz,
# PSD), the aggregate identity-mass floor applied after those strict shifts
# and before the shared cross-centering, and the shared pre-centering scalar
# formula.  They never call a provider, touch a workspace, or read private
# problem fields: everything operates on plain `AbstractVector` /
# `AbstractMatrix` values in a single arithmetic type `T`.
#
# Strict safety: a point is certified strictly interior when its cone margin
# is greater than
#
#     sqrt(eps(T)) * max(one(T), scale)
#
# where `scale` is the max-norm of the point.  Every shift is applied in place
# (trailing `!`) and re-verified after application, so a returned
# `ok == true` always means the reported `margin` is finite and strictly above
# the safety threshold computed from the reported `scale`.
# ============================================================================

"""Strict cold-start safety threshold `sqrt(eps(T)) * max(one(T), scale)`."""
@inline function _cold_start_safety(::Type{T}, scale::T) where {T}
    return sqrt(eps(T)) * max(one(T), scale)
end

# BigFloat arithmetic allocates its result at ambient task precision. Enter
# the precision owned by the scaled problem value explicitly so a direct
# helper call cannot narrow an authoritative cold-start decision.
@inline function _cold_start_safety(
    ::Type{BigFloat},
    scale::BigFloat,
)
    return setprecision(BigFloat, precision(scale)) do
        sqrt(eps(BigFloat)) * max(one(BigFloat), scale)
    end
end

"""
    _cold_start_rounding_slack(::Type{T}, scale) -> T

Few-ULP additive push applied on top of the nominal shift so the certified
margin is *strictly* above the safety threshold after in-place rounding, even
when the nominal target would round to exactly the threshold.
"""
@inline function _cold_start_rounding_slack(::Type{T}, scale::T) where {T}
    return T(8) * eps(T) * max(one(T), scale)
end

@inline function _cold_start_rounding_slack(
    ::Type{BigFloat},
    scale::BigFloat,
)
    return setprecision(BigFloat, precision(scale)) do
        BigFloat(8) * eps(BigFloat) * max(one(BigFloat), scale)
    end
end

"""
    _cold_start_add_vector_identity!(vector, shift) -> vector

Add `shift * e`, with `e` the all-ones orthant identity, to every coordinate
of `vector` in place and return `vector`.
"""
function _cold_start_add_vector_identity!(
    vector::AbstractVector{T},
    shift::T,
) where {T}
    @inbounds for index in eachindex(vector)
        vector[index] += shift
    end
    return vector
end

"""
    _cold_start_add_lorentz_identity!(vector, shift) -> vector

Add `shift * e`, with `e = (1, 0, …, 0)` the Lorentz identity, to the head
coordinate of `vector` in place and return `vector`.
"""
function _cold_start_add_lorentz_identity!(
    vector::AbstractVector{T},
    shift::T,
) where {T}
    isempty(vector) && throw(DimensionMismatch(
        "cold-start Lorentz identity add requires a nonempty vector",
    ))
    vector[firstindex(vector)] += shift
    return vector
end

"""
    _cold_start_add_psd_identity!(matrix, shift) -> matrix

Add `shift * I` to the diagonal of the square `matrix` in place and return
`matrix`.
"""
function _cold_start_add_psd_identity!(
    matrix::AbstractMatrix{T},
    shift::T,
) where {T}
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension ||
        throw(DimensionMismatch(
            "cold-start PSD identity add requires a square matrix",
        ))
    @inbounds for index in 1:dimension
        matrix[index, index] += shift
    end
    return matrix
end

"""Finite max-norm scale and componentwise margin of an orthant vector."""
function _cold_start_positive_stats(vector::AbstractVector{T}) where {T}
    isempty(vector) && return false, zero(T), zero(T)
    scale = zero(T)
    margin = vector[firstindex(vector)]
    @inbounds for value in vector
        isfinite(value) || return false, scale, margin
        scale = max(scale, abs(value))
        margin = min(margin, value)
    end
    return true, scale, margin
end

"""Finite max-norm scale and stable `_soc_margin` of a Lorentz vector."""
function _cold_start_lorentz_stats(vector::AbstractVector{T}) where {T}
    isempty(vector) && return false, zero(T), zero(T)
    scale = zero(T)
    @inbounds for value in vector
        isfinite(value) || return false, scale, zero(T)
        scale = max(scale, abs(value))
    end
    return true, scale, _soc_margin(vector)
end

"""
    _cold_start_psd_lower_bound(matrix, dimension) -> T

Certified lower bound on the minimum eigenvalue of a square symmetric block,
using the lower triangle as authoritative: the exact eigenvalue for 1×1 and
2×2 blocks, and the symmetric Gershgorin
bound

    λ_min ≥ min_i (a_ii - Σ_{j≠i} |a_max(i,j),min(i,j)|)

for larger blocks. The inactive upper triangle may be stale or poisoned;
automatic initialization therefore follows the same lower-authoritative
storage contract as the SDP factorization path.
"""
function _cold_start_psd_lower_bound(
    matrix::AbstractMatrix{T},
    dimension::Int,
) where {T}
    if dimension == 1
        return matrix[1, 1]
    elseif dimension == 2
        two = one(T) + one(T)
        first = matrix[1, 1]
        second = matrix[2, 2]
        off_diagonal = matrix[2, 1]
        center = (first + second) / two
        radius = hypot((first - second) / two, off_diagonal)
        return center - radius
    end
    bound = zero(T)
    first_row = true
    @inbounds for row in 1:dimension
        radius = zero(T)
        for column in 1:dimension
            column == row && continue
            lower_row = max(row, column)
            lower_column = min(row, column)
            radius += abs(matrix[lower_row, lower_column])
        end
        candidate = matrix[row, row] - radius
        if first_row
            bound = candidate
            first_row = false
        else
            bound = min(bound, candidate)
        end
    end
    return bound
end

"""Finite max-norm scale and certified PSD eigenvalue lower bound."""
function _cold_start_psd_stats(
    matrix::AbstractMatrix{T},
    dimension::Int,
) where {T}
    isempty(matrix) && return false, zero(T), zero(T)
    scale = zero(T)
    @inbounds for column in 1:dimension, row in column:dimension
        value = matrix[row, column]
        isfinite(value) || return false, scale, zero(T)
        scale = max(scale, abs(value))
    end
    return true, scale, _cold_start_psd_lower_bound(matrix, dimension)
end

"""
    _cold_start_positive_shift!(vector) -> (ok, shift, margin, scale)

Lift an orthant / LP-product vector strictly into the positive orthant by
adding one global scalar to every coordinate (the LP identity direction).

Returns:

* `ok` — the shifted vector is certified: `margin` is finite and
  `margin > sqrt(eps(T)) * max(one(T), scale)`;
* `shift` — the total global scalar added (`zero(T)` when no shift was
  needed);
* `margin` — the componentwise minimum of the shifted vector;
* `scale` — the max-norm of the shifted vector.

Non-finite inputs return `ok == false`; empty vectors are rejected.
"""
function _cold_start_positive_shift!(vector::AbstractVector{T}) where {T}
    isempty(vector) && throw(DimensionMismatch(
        "cold-start positive vector must be nonempty",
    ))
    finite, scale, margin = _cold_start_positive_stats(vector)
    finite || return false, zero(T), margin, scale
    safety = _cold_start_safety(T, scale)
    slack = _cold_start_rounding_slack(T, scale)
    margin > safety && return true, zero(T), margin, scale
    applied = zero(T)
    for _ in 1:16
        shift = safety - margin + slack
        isfinite(shift) || return false, applied, margin, scale
        _cold_start_add_vector_identity!(vector, shift)
        applied += shift
        finite, scale, margin = _cold_start_positive_stats(vector)
        finite || return false, applied, margin, scale
        safety = _cold_start_safety(T, scale)
        margin > safety && return true, applied, margin, scale
    end
    return false, applied, margin, scale
end

"""
    _cold_start_lorentz_shift!(vector) -> (ok, shift, margin, scale)

Lift a Lorentz-cone vector strictly into the cone interior by adding a scalar
to the head coordinate only (the Lorentz identity direction).  The margin is
the stable `_soc_margin(vector) = vector[1] - norm(vector[2:end])`.

Returns the same `(ok, shift, margin, scale)` tuple as
`_cold_start_positive_shift!`.
"""
function _cold_start_lorentz_shift!(vector::AbstractVector{T}) where {T}
    isempty(vector) && throw(DimensionMismatch(
        "cold-start Lorentz vector must be nonempty",
    ))
    finite, scale, margin = _cold_start_lorentz_stats(vector)
    finite || return false, zero(T), margin, scale
    safety = _cold_start_safety(T, scale)
    slack = _cold_start_rounding_slack(T, scale)
    margin > safety && return true, zero(T), margin, scale
    applied = zero(T)
    for _ in 1:16
        shift = safety - margin + slack
        isfinite(shift) || return false, applied, margin, scale
        _cold_start_add_lorentz_identity!(vector, shift)
        applied += shift
        finite, scale, margin = _cold_start_lorentz_stats(vector)
        finite || return false, applied, margin, scale
        safety = _cold_start_safety(T, scale)
        margin > safety && return true, applied, margin, scale
    end
    return false, applied, margin, scale
end

"""
    _cold_start_psd_shift!(matrix) -> (ok, shift, margin, scale)

Lift a square PSD block strictly into the positive-definite interior by adding
`shift * I` (the PSD identity direction).  The certified margin is the exact
minimum eigenvalue for 1×1 and 2×2 blocks and the symmetric Gershgorin lower
bound for larger blocks, and it is re-verified after the shift is applied.

Returns the same `(ok, shift, margin, scale)` tuple as
`_cold_start_positive_shift!`.  A `0×0` block is trivially certified with
zero shift.
"""
function _cold_start_psd_shift!(matrix::AbstractMatrix{T}) where {T}
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension ||
        throw(DimensionMismatch(
            "cold-start PSD shift requires a square matrix",
        ))
    dimension == 0 && return true, zero(T), zero(T), one(T)
    finite, scale, margin = _cold_start_psd_stats(matrix, dimension)
    finite || return false, zero(T), margin, scale
    safety = _cold_start_safety(T, scale)
    slack = _cold_start_rounding_slack(T, scale)
    margin > safety && return true, zero(T), margin, scale
    applied = zero(T)
    for _ in 1:16
        shift = safety - margin + slack
        isfinite(shift) || return false, applied, margin, scale
        _cold_start_add_psd_identity!(matrix, shift)
        applied += shift
        finite, scale, margin = _cold_start_psd_stats(matrix, dimension)
        finite || return false, applied, margin, scale
        safety = _cold_start_safety(T, scale)
        margin > safety && return true, applied, margin, scale
    end
    return false, applied, margin, scale
end

"""
    _cold_start_identity_mass_shifts(
        primal_identity_dot,
        dual_identity_dot,
        identity_norm_squared,
    ) -> (ok, primal_shift, dual_shift)

Return the smallest nonnegative identity shifts that raise each aggregate
identity mass to the mass of the unit cone point.  Writing `e` for the
product-cone identity and `rho = <e,e>`, adding `delta*e` changes `<e,w>` by
`delta*rho`, so the two shifts are

    delta_p = max(0, (rho - <e,s>) / rho)
    delta_d = max(0, (rho - <e,z>) / rho).

This is a scale-free guard against a KKT-derived point landing arbitrarily
close to the cone vertex.  It uses only the identity geometry of the working
cone after equilibration: orthant `rho` is the number of inequalities, PSD
`rho` is the sum of block dimensions, and a Lorentz product contributes one
per block.  Barrier degree is deliberately separate; it continues to
normalize complementarity, while this helper reproduces the unit identity
mass without changing a FixedTraceQ3 head that is already one.

Non-finite masses or a nonpositive identity norm fail closed and return zero
shifts.  Callers apply these shifts before the complementarity cross-centering
rule and recompute the masses and complementarity afterwards.
"""
function _cold_start_identity_mass_shifts(
    primal_identity_dot::T,
    dual_identity_dot::T,
    identity_norm_squared::Integer,
) where {T}
    valid = identity_norm_squared > 0 &&
        isfinite(primal_identity_dot) &&
        isfinite(dual_identity_dot)
    valid || return false, zero(T), zero(T)
    target = T(identity_norm_squared)
    isfinite(target) && target > zero(T) ||
        return false, zero(T), zero(T)
    primal_shift = max(
        zero(T),
        (target - primal_identity_dot) / target,
    )
    dual_shift = max(
        zero(T),
        (target - dual_identity_dot) / target,
    )
    valid = isfinite(primal_shift) && isfinite(dual_shift)
    return valid, valid ? primal_shift : zero(T), valid ? dual_shift : zero(T)
end

function _cold_start_identity_mass_shifts(
    primal_identity_dot::BigFloat,
    dual_identity_dot::BigFloat,
    identity_norm_squared::Integer,
)
    bits = min(
        precision(primal_identity_dot),
        precision(dual_identity_dot),
    )
    return setprecision(BigFloat, bits) do
        identity_norm_squared > 0 ||
            return false, zero(BigFloat), zero(BigFloat)
        isfinite(primal_identity_dot) && isfinite(dual_identity_dot) ||
            return false, zero(BigFloat), zero(BigFloat)
        target = BigFloat(identity_norm_squared)
        primal_shift = max(
            zero(BigFloat),
            (target - primal_identity_dot) / target,
        )
        dual_shift = max(
            zero(BigFloat),
            (target - dual_identity_dot) / target,
        )
        valid = isfinite(primal_shift) && isfinite(dual_shift)
        return valid,
               valid ? primal_shift : zero(BigFloat),
               valid ? dual_shift : zero(BigFloat)
    end
end

"""
    _cold_start_centering_shifts(kappa, primal_identity_dot, dual_identity_dot)
        -> (ok, primal_shift, dual_shift)

Standard cold pre-centering shifts for barrier target `kappa`:

    primal_shift = kappa / (2 * <e, z>),    dual_shift = kappa / (2 * <e, s>)

The caller supplies the cone identity inner products `<e, s>` and `<e, z>`;
for a product cone these are summed over every block (LP: `sum(s)`,
Lorentz: head coordinate, PSD: trace).  `ok` is `false` unless `kappa` and
both inner products are finite and strictly positive and both resulting
shifts are finite; failed calls return zero shifts.
"""
function _cold_start_centering_shifts(
    kappa::T,
    primal_identity_dot::T,
    dual_identity_dot::T,
) where {T}
    ok = isfinite(kappa) && kappa > zero(kappa)
    ok = ok && isfinite(primal_identity_dot) &&
        primal_identity_dot > zero(primal_identity_dot)
    ok = ok && isfinite(dual_identity_dot) &&
        dual_identity_dot > zero(dual_identity_dot)
    two = one(T) + one(T)
    # This is the standard primal-dual cold-start pre-centering cross rule:
    # the primal identity shift is normalized by the dual identity mass and
    # vice versa.  Each linear cross term therefore contributes kappa/2 to
    # the post-shift complementarity.
    primal_shift = ok ? kappa / (two * dual_identity_dot) : zero(T)
    dual_shift = ok ? kappa / (two * primal_identity_dot) : zero(T)
    ok = ok && isfinite(primal_shift) && isfinite(dual_shift)
    return ok, primal_shift, dual_shift
end

"""
    _cold_start_centering_shifts(kappa, primal, dual, identity)
        -> (ok, primal_shift, dual_shift)

Single-cone convenience form of `_cold_start_centering_shifts` that computes
`<e, s>` and `<e, z>` with `LinearAlgebra.dot` against the cone identity
vector (all-ones for the orthant, `(1, 0, …, 0)` for Lorentz, and a matrix
identity for PSD blocks is handled by passing the identity inner products
directly).
"""
function _cold_start_centering_shifts(
    kappa::T,
    primal::AbstractVector{T},
    dual::AbstractVector{T},
    identity::AbstractVector{T},
) where {T}
    length(primal) == length(dual) ||
        throw(DimensionMismatch(
            "cold-start centering vectors must have equal length",
        ))
    length(primal) == length(identity) ||
        throw(DimensionMismatch(
            "cold-start centering identity must match vector length",
        ))
    return _cold_start_centering_shifts(
        kappa,
        dot(identity, primal),
        dot(identity, dual),
    )
end
