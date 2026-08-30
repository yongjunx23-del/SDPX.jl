#=====================================================================#
#    Compatible singular scalar closure classification.
#
#    The homogeneous self-dual scalar recovery solves
#
#        D * d_tau = N
#        D = kappa + tau * eta_u
#        N = rhs.tau_kappa - tau * (rhs.homogeneous_gap + eta_w)
#
#    D=0 with N=0 describes a compatible rank-zero scalar closure: d_tau is
#    a gauge coordinate and the five-equation system has a family of valid
#    directions.  D=0 with N != 0 is an incompatible rank-deficient closure.
#    Near-zero classification must use independent accumulated-work bounds
#    (never the public optimization tolerance, a detached epsilon multiplier,
#    or max(1, scale)) so a genuine compatible singular system is not
#    rejected and a marginally incompatible one is not silently accepted.
#=====================================================================#

"""
    classify_scalar_closure(denominator, numerator;
                            denominator_work, numerator_work,
                            kind=Val(:multifloat)) -> Symbol

Classify the scalar closure with independent arithmetic-work bounds.

Returns `:regular`, `:compatible_singular_gauge`, or `:incompatible_singular`.
The caller decides `d_tau = 0` for the compatible gauge and must then pass
the reconstructed direction through the unchanged five-equation residual gate.
"""
@inline function classify_scalar_closure(
    denominator::T, numerator::T;
    denominator_work::T, numerator_work::T,
) where {T<:AbstractFloat}
    isfinite(denominator) && isfinite(numerator) &&
        isfinite(denominator_work) && isfinite(numerator_work) ||
        return :insufficient_precision
    eps_scale = sqrt(eps(T))
    d_floor = eps_scale * denominator_work
    if abs(denominator) > d_floor
        # Resolvable nonzero denominator.  The division is only entered when
        # the value carries real information above its own accumulation error.
        return :regular
    end
    # Denominator is unresolved against its work bound.  The closure is
    # compatible only when N is independently compatible with zero.
    n_floor = eps_scale * numerator_work
    if abs(numerator) <= n_floor
        return :compatible_singular_gauge
    end
    return :incompatible_singular
end

"""
    scalar_closure_resolution(classification, denominator, numerator) -> T

Map a classification to the deterministic `d_tau` value: regular divides,
compatible singular uses the exact rank-zero gauge `d_tau = 0`, and an
incompatible closure is the caller's responsibility to reject before this
helper is reached.
"""
@inline function scalar_closure_resolution(
    classification::Symbol, denominator::T, numerator::T,
) where {T<:AbstractFloat}
    if classification === :regular
        return numerator / denominator
    elseif classification === :compatible_singular_gauge
        return zero(T)
    end
    throw(ArgumentError(
        "scalar closure classification $classification has no resolution",
    ))
end
