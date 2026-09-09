# R0-P4 backend selector admission (plan-time slice).
#
# Reference design: docs/design/R0P4_FACTOR_PAIR_BOUNDARY.md
#
# The experimental half-Power factor-pair backend is selected by
# `Settings.nonsymmetric_backend`.  This file implements the *typed, fail-closed*
# admission boundary only: an explicit experimental request either passes the
# declared capability checks or refuses with a typed error BEFORE any numerical
# setup.  It never silently runs the historical dense-metric path, and a default
# solve never attempts the experimental backend.
#
# The execution fork itself (steps 4+) is not implemented yet, so a request that
# passes every declared capability check still refuses with
# `reason=:not_implemented`.  That refusal is intentional: it keeps the selector
# honest until the factor-pair epoch is actually admitted, and it is the single
# place to flip once the ordered implementation plan reaches step 9+.

"""
    FactorPairAdmission

Plan-time admission decision for the opt-in factor-pair nonsymmetric backend.

`admitted=true` means the request is within the declared experimental scope and
the backend implementation is available.  Every other outcome carries a
`stage`/`reason` pair describing the typed refusal; `detail` is a human-readable
sentence and `checks` records the evaluated capability facts.
"""
struct FactorPairAdmission
    admitted::Bool
    stage::Symbol
    reason::Symbol
    backend::NonsymmetricBackendChoice
    detail::String
    checks::NamedTuple
end

"""
    UnsupportedBackendError

Typed refusal for a valid-but-unsupported backend request.  Surfaced before
iteration; it is not a numerical failure and must not be caught as one.
"""
struct UnsupportedBackendError <: Exception
    stage::Symbol
    reason::Symbol
    backend::NonsymmetricBackendChoice
    detail::String
end

function Base.showerror(io::IO, err::UnsupportedBackendError)
    print(
        io,
        "UnsupportedBackendError(stage=", err.stage,
        ", reason=", err.reason,
        ", backend=", err.backend,
        "): ", err.detail,
    )
end

"""Cone symbols the first experimental admission accepts (orthant + Power).

`:zero` is deliberately excluded: the factor state has no ZeroCone
representation, so equality rows must be handled by the ordinary
canonical/equality-reduction lineage and admitted explicitly, never by
pretending a zero row is an orthant row.
"""
const FACTOR_PAIR_ADMITTED_CONES = (:nonnegative, :power)

"""
    factor_pair_cones_admitted(cones) -> Bool

Post-reduction cone-composition admission: orthant/zero blocks plus at least one
`:power` block, and nothing else.  SOC, PSD and Exp products are out of the
first admission scope and must refuse rather than take a mixed-cone hybrid.
"""
function factor_pair_cones_admitted(cones)
    cones === nothing && return true
    seen = Symbol[]
    for cone in cones
        cone in FACTOR_PAIR_ADMITTED_CONES || return false
        push!(seen, cone)
    end
    return :power in seen
end

_factor_pair_knobs_default(settings::Settings) = (
    settings.iteration_knobs.sigma === nothing &&
    settings.iteration_knobs.beta === nothing &&
    settings.iteration_knobs.gamma === nothing &&
    settings.iteration_knobs.predictor === :classic
)

"""
    factor_pair_admission(settings; cones=nothing) -> FactorPairAdmission

Evaluate the declared R0-P4 capability scope for `settings`.  The default
`NativeNonsymmetricBackend` is always admitted and reports
`reason=:native_default` (the historical path is unchanged).  An explicit
experimental request is checked against the frozen first-admission scope and
refused with a typed `reason` otherwise.
"""
function factor_pair_admission(
    settings::Settings{T};
    cones=nothing,
) where {T<:AbstractFloat}
    backend = settings.nonsymmetric_backend
    checks = (
        arithmetic=T,
        engine=settings.engine,
        scaling=settings.scaling,
        formulation=settings.formulation,
        kkt_route=settings.kkt_route,
        provider=settings.provider,
        sparse=settings.sparse,
        threads=settings.limits.threads,
        iteration_policy=settings.iteration_knobs.predictor,
        cones=cones,
    )
    backend === NativeNonsymmetricBackend && return FactorPairAdmission(
        true, :plan, :native_default, backend,
        "default dense-metric backend; historical behavior unchanged",
        checks,
    )

    if T !== Float64
        return FactorPairAdmission(
            false, :plan, :arithmetic, backend,
            "experimental factor-pair backend admits Float64 only, got $T",
            checks,
        )
    end
    if !(settings.engine in (:auto, :native_hsd))
        return FactorPairAdmission(
            false, :plan, :engine, backend,
            "experimental factor-pair backend requires engine=:auto or :native_hsd, got $(settings.engine)",
            checks,
        )
    end
    if settings.kkt_route !== :bordered
        return FactorPairAdmission(
            false, :plan, :kkt_route, backend,
            "experimental factor-pair backend requires kkt_route=:bordered, got $(settings.kkt_route)",
            checks,
        )
    end
    if settings.provider !== :auto
        return FactorPairAdmission(
            false, :plan, :provider, backend,
            "experimental factor-pair backend resolves its own provider and rejects provider=$(settings.provider)",
            checks,
        )
    end
    if settings.formulation !== :auto
        return FactorPairAdmission(
            false, :plan, :formulation, backend,
            "experimental factor-pair backend requires formulation=:auto, got $(settings.formulation)",
            checks,
        )
    end
    if settings.sparse !== :auto
        return FactorPairAdmission(
            false, :plan, :sparse, backend,
            "experimental factor-pair backend owns its storage plan and rejects sparse=$(settings.sparse)",
            checks,
        )
    end
    if settings.scaling === :equilibrate
        return FactorPairAdmission(
            false, :plan, :scaling, backend,
            "experimental factor-pair backend rejects equilibration (scaling=$(settings.scaling))",
            checks,
        )
    end
    if settings.limits.threads != 1
        return FactorPairAdmission(
            false, :plan, :threads, backend,
            "experimental factor-pair backend requires single-thread execution, got threads=$(settings.limits.threads)",
            checks,
        )
    end
    if !_factor_pair_knobs_default(settings)
        return FactorPairAdmission(
            false, :plan, :iteration_policy, backend,
            "experimental factor-pair backend requires the classic/default iteration knobs, got $(settings.iteration_knobs)",
            checks,
        )
    end
    if !factor_pair_cones_admitted(cones)
        return FactorPairAdmission(
            false, :plan, :cones, backend,
            "experimental factor-pair backend admits orthant plus at least one half-Power block only, got cones=$(cones)",
            checks,
        )
    end
    # Every declared capability check passed, but the execution fork is not
    # implemented yet.  Fail closed instead of running the legacy backend.
    return FactorPairAdmission(
        false, :admission, :not_implemented, backend,
        "experimental half-Power factor-pair backend is not yet admitted to execution (R0-P4 implementation in progress); no fallback to the default backend",
        checks,
    )
end

"""
    enforce_factor_pair_admission!(settings; cones=nothing)

Throw `UnsupportedBackendError` when the selected backend is not admitted.
"""
function enforce_factor_pair_admission!(
    settings::Settings{T};
    cones=nothing,
) where {T<:AbstractFloat}
    decision = factor_pair_admission(settings; cones=cones)
    decision.admitted && return decision
    throw(UnsupportedBackendError(
        decision.stage, decision.reason, decision.backend, decision.detail,
    ))
end
