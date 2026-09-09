#=====================================================================#
#    R1-A unified AccuracyContract (read-only, no routing authority).
#
#    This file owns exactly one immutable record plus its derivation and
#    classification helpers:
#      - `AccuracyContract{T}` — storage type / effective bits, construction
#        / working / verification precision bits, rounding mode, arithmetic
#        domain flags (finite/subnormal/overflow policy), requested and
#        allowed error, and the actually-loaded provider/backend identity.
#      - `AccuracyClass` — exactly four classes: verified, unsupported,
#        numerical failure, infrastructure failure.
#      - `accuracy_class(status)` — documented mapping from the existing
#        `SolveStatus` values.  It never relabels any public status; the
#        public `status(::Result)` symbols are untouched.
#      - `accuracy_contract(model, settings)` / `(model, result)` —
#        read-only derivation at the public Settings / result-diagnostics
#        seam.  Derivation never mutates the model, the settings, the
#        result, the ambient precision, or any solver routing decision.
#
#    Unsupported contexts (a rounding mode the backend does not implement,
#    a non-finite/subnormal policy SDPX does not support, an unmaintained
#    arithmetic type, ...) throw `UnsupportedAccuracyContext` instead of
#    silently degrading.  There is no fallback, no tolerance widening, no
#    hidden precision promotion, and no default numerical behavior change.
#=====================================================================#

"""Typed refusal for an accuracy context SDPX does not implement.

`context` names the refused dimension (`:rounding`, `:finite_domain`,
`:subnormal_policy`, `:arithmetic`, `:precision_bits`,
`:verification_precision`); `message` is self-contained.
"""
struct UnsupportedAccuracyContext <: Exception
    context::Symbol
    message::String
end

Base.showerror(io::IO, error::UnsupportedAccuracyContext) =
    print(io, "UnsupportedAccuracyContext(", error.context, "): ", error.message)

"""
    AccuracyClass

R1-A four-class outcome classification.  Exactly these four values exist:

- `AccuracyVerified` — an independently checked certificate was produced
  (requested tolerance met and verified in the original coordinates, or a
  certified infeasibility finding).
- `AccuracyUnsupported` — the requested arithmetic context is not
  implemented by the loaded backend.  This class never arises from mapping
  a `SolveStatus`; it arises from `accuracy_contract` refusing to build a
  contract (throwing `UnsupportedAccuracyContext`) instead of silently
  degrading.
- `AccuracyNumericalFailure` — the solver ran but did not produce a
  verified certificate (stall, budget exhaustion, breakdown, relaxed-only
  `AlmostOptimal`, precision floor, ...).
- `AccuracyInfrastructureFailure` — the solver never ran to a numerical
  verdict (never started, externally stopped).

See `accuracy_class` for the exact `SolveStatus` mapping.
"""
@enum AccuracyClass begin
    AccuracyVerified
    AccuracyUnsupported
    AccuracyNumericalFailure
    AccuracyInfrastructureFailure
end

"""
    accuracy_class(status::SolveStatus) -> AccuracyClass

Documented total mapping from every existing solver outcome to the R1-A
four-class view.  This is a separate read-only projection: no public
`status(::Result)` symbol is relabeled and no `SolveStatus` value changes.

Mapping:
- verified: `Optimal`, `FeasibleCert`, `InfeasibleCert`,
  `PrimalInfeasible`, `DualInfeasible` — every independently checked
  certificate outcome.
- infrastructure failure: `NotStarted` (never executed), `UserStopped`
  (externally interrupted before any numerical verdict).
- numerical failure: everything else — `Stalled`, `IterLimit`,
  `TimeLimit`, `NumericalBreakdown`, `MaxRestartsExceeded`,
  `AlmostOptimal` (relaxed tolerance only, requested accuracy not met),
  `InsufficientPrecision` (working-precision floor), `NumericalFailure`.
- unsupported: not produced by this mapping.  `AccuracyUnsupported`
  describes a refused *context* (see `UnsupportedAccuracyContext`), never
  a relabeled solver outcome.
"""
function accuracy_class(status::SolveStatus)
    status in (
        Optimal,
        FeasibleCert,
        InfeasibleCert,
        PrimalInfeasible,
        DualInfeasible,
    ) && return AccuracyVerified
    status in (NotStarted, UserStopped) &&
        return AccuracyInfrastructureFailure
    status in (
        Stalled,
        IterLimit,
        TimeLimit,
        NumericalBreakdown,
        MaxRestartsExceeded,
        AlmostOptimal,
        InsufficientPrecision,
        NumericalFailure,
    ) && return AccuracyNumericalFailure
    throw(ArgumentError("unknown solve status $(repr(status))"))
end

"""Project a public result to its R1-A accuracy class (read-only)."""
accuracy_class(result::Result) = accuracy_class(result.status)

"""
    AccuracyContract{T<:AbstractFloat}

Immutable unified accuracy record for one solve arithmetic.  All fields are
concrete facts captured at derivation time; the struct is immutable so a
later mutation of the source model or result can never change an existing
contract.

Fields:
- `storage_type` / `storage_name` — the Julia arithmetic type holding model
  data and its stable `_arithmetic_symbol` tag.
- `effective_bits` — effective significand bits of the storage type:
  `53` for `Float64`, the model `precision_bits` for `BigFloat`, and the
  live `precision(T)` probe for MultiFloat types, which equals the
  `53N-(N-1)` formula (`105`/`157`/`209` for x2/x3/x4).
- `construction_precision_bits` — the model-declared `precision_bits`
  (what the stored numbers were built at).
- `working_precision_bits` — the ambient working precision observed at
  derivation (`Base.precision(BigFloat)` for `BigFloat`, `effective_bits`
  otherwise).
- `verification_precision_bits` — precision the certificate is stated at
  (defaults to the working precision; an explicit override must be `>= 1`
  and is recorded, never silently promoted).
- `rounding` — the observed rounding mode (`rounding(BigFloat)` for
  `BigFloat`, `RoundNearest` otherwise).  SDPX only executes `RoundNearest`;
  any other request is refused.
- `finite_required` — always `true`: SDPX's finite gates reject non-finite
  data before any tolerance comparison.  Requesting `false` is refused.
- `subnormal_policy` — `:ieee_preserved` for `Float64`/`BigFloat`
  (subnormals are preserved, never flushed); `:not_guaranteed` for
  MultiFloat arithmetic (no subnormal contract).  `:flush_to_zero` and
  `:ieee_preserved`-on-MultiFloat are refused.
- `overflow_policy` — `:to_infinity` for `Float64` (IEEE infinity),
  `:extended_exponent` for `BigFloat` (enormous MPFR exponent range),
  `:to_nan` for MultiFloat (finite range inherited from `Float64`;
  infinities collapse to `NaN`, hence the non-finite-iterate guard).
- `requested_error` / `allowed_error` — strictest requested stopping target
  as `Float64`, and the allowance actually granted.  SDPX never widens a
  tolerance: `allowed_error == requested_error` always.  A relaxed-only
  outcome stays `AlmostOptimal` and maps to `AccuracyNumericalFailure`
  rather than becoming a widened pass.
- `provider_requested` — the `Settings.provider` policy (`:auto`, ...).
- `provider_loaded` — the actually-loaded LA provider name reported by the
  `la_provider_descriptor(T)` seam (`:none` when no optional provider is
  loaded).  Read-only: no provider is instantiated or probed numerically.
- `provider_implementation` — backend implementation label carried by the
  execution plan when derived from a `Result`
  (`:none` for the pre-solve Settings seam, which plans no backend).
- `julia_version` — `VERSION` observed at derivation.
- `multifloat_available` — whether the extension seam recognizes `T` as
  MultiFloat arithmetic.
"""
struct AccuracyContract{T<:AbstractFloat}
    storage_type::Type{T}
    storage_name::Symbol
    effective_bits::Int
    construction_precision_bits::Int
    working_precision_bits::Int
    verification_precision_bits::Int
    rounding::RoundingMode
    finite_required::Bool
    subnormal_policy::Symbol
    overflow_policy::Symbol
    requested_error::Float64
    allowed_error::Float64
    provider_requested::Symbol
    provider_loaded::Symbol
    provider_implementation::Symbol
    julia_version::VersionNumber
    multifloat_available::Bool
end

function Base.show(io::IO, contract::AccuracyContract{T}) where {T}
    print(
        io,
        "AccuracyContract{", T, "}(",
        "effective_bits=", contract.effective_bits,
        ", construction=", contract.construction_precision_bits,
        ", working=", contract.working_precision_bits,
        ", verification=", contract.verification_precision_bits,
        ", rounding=", contract.rounding,
        ", requested_error=", contract.requested_error,
        ", allowed_error=", contract.allowed_error,
        ", provider_loaded=", contract.provider_loaded,
        ")",
    )
end

# --- internal helpers (read-only probes, no mutation) ------------------------

@inline function _accuracy_storage_name(::Type{T}) where {T}
    return _arithmetic_symbol(T)
end

"""Live effective significand bits for `T` (the `53N-(N-1)` formula for xN).

`Float64` is exactly 53.  `BigFloat` has no fixed width: the caller passes
the model-declared bits.  Every other supported type must be a loaded
MultiFloat width, probed live via `precision(T)` so the record matches the
loaded runtime rather than a hardcoded table.
"""
function _accuracy_effective_bits(::Type{Float64}, ::Int)
    precision(Float64) == 53 || throw(UnsupportedAccuracyContext(
        :arithmetic,
        "Float64 effective bits probe returned $(precision(Float64)), expected 53",
    ))
    return 53
end

function _accuracy_effective_bits(::Type{BigFloat}, construction_bits::Int)
    construction_bits >= 2 || throw(UnsupportedAccuracyContext(
        :precision_bits,
        "BigFloat construction precision must be >= 2, got $construction_bits",
    ))
    return construction_bits
end

function _accuracy_effective_bits(::Type{T}, ::Int) where {T}
    is_multifloat_arithmetic(T) || is_supported_arithmetic(T) || throw(
        UnsupportedAccuracyContext(
            :arithmetic,
            "unsupported storage type $T; use Float64, BigFloat, or a loaded MultiFloat width",
        ),
    )
    live = try
        precision(T)
    catch exception
        _recoverable(exception) || rethrow()
        throw(UnsupportedAccuracyContext(
            :arithmetic,
            "storage type $T exposes no live precision probe",
        ))
    end
    live >= 2 || throw(UnsupportedAccuracyContext(
        :arithmetic,
        "storage type $T reports effective bits $live, expected >= 2",
    ))
    return Int(live)
end

@inline function _accuracy_working_bits(::Type{BigFloat}, ::Int)
    return Int(Base.precision(BigFloat))
end

@inline function _accuracy_working_bits(::Type{T}, effective_bits::Int) where {T}
    return effective_bits
end

function _accuracy_rounding(::Type{BigFloat})
    return rounding(BigFloat)
end

function _accuracy_rounding(::Type{T}) where {T}
    return RoundNearest
end

function _accuracy_require_rounding(::Type{T}, requested::RoundingMode) where {T}
    observed = _accuracy_rounding(T)
    requested == observed || throw(UnsupportedAccuracyContext(
        :rounding,
        "rounding $requested is not implemented by the $T backend (observed $observed)",
    ))
    T === BigFloat && observed != RoundNearest && throw(
        UnsupportedAccuracyContext(
            :rounding,
            "BigFloat rounding $observed is outside the verified RoundNearest contract",
        ),
    )
    requested != RoundNearest && throw(UnsupportedAccuracyContext(
        :rounding,
        "rounding $requested is outside the verified RoundNearest contract",
    ))
    return observed
end

function _accuracy_subnormal_policy(::Type{T}) where {T}
    T === Float64 && return :ieee_preserved
    T === BigFloat && return :ieee_preserved
    return :not_guaranteed
end

function _accuracy_require_subnormal(::Type{T}, requested::Symbol) where {T}
    truth = _accuracy_subnormal_policy(T)
    requested === :default && return truth
    requested === truth && return truth
    throw(UnsupportedAccuracyContext(
        :subnormal_policy,
        "subnormal policy $(repr(requested)) is not supported for $T (backend truth $(repr(truth)))",
    ))
end

function _accuracy_overflow_policy(::Type{T}) where {T}
    T === Float64 && return :to_infinity
    T === BigFloat && return :extended_exponent
    return :to_nan
end

function _accuracy_requested_error(::Type{T}, construction_bits::Int, settings::Settings{T}) where {T}
    automatic = try
        Float64(auto_tolerance(T, construction_bits))
    catch exception
        _recoverable(exception) || rethrow()
        throw(UnsupportedAccuracyContext(
            :precision_bits,
            "cannot resolve automatic tolerance for $T at $construction_bits bits",
        ))
    end
    isfinite(automatic) && automatic > 0.0 || throw(UnsupportedAccuracyContext(
        :precision_bits,
        "automatic tolerance for $T at $construction_bits bits is not a positive finite value",
    ))
    targets = Float64[]
    settings.tolerances.primal === nothing || push!(targets, Float64(settings.tolerances.primal))
    settings.tolerances.dual === nothing || push!(targets, Float64(settings.tolerances.dual))
    settings.tolerances.gap === nothing || push!(targets, Float64(settings.tolerances.gap))
    strictest = isempty(targets) ? automatic : minimum(targets)
    isfinite(strictest) && strictest > 0.0 || throw(UnsupportedAccuracyContext(
        :precision_bits,
        "requested tolerance is not a positive finite value",
    ))
    return strictest
end

function _accuracy_provider_loaded(::Type{T}) where {T}
    descriptor = try
        la_provider_descriptor(T, 1)
    catch exception
        _recoverable(exception) || rethrow()
        return :unknown
    end
    provider = try
        Symbol(descriptor.provider)
    catch exception
        _recoverable(exception) || rethrow()
        return :unknown
    end
    return provider
end

function _accuracy_build(
    ::Type{T},
    construction_bits::Int,
    working_bits::Int,
    verification_bits::Int,
    rounding_mode::RoundingMode,
    subnormal::Symbol,
    requested_error::Float64,
    provider_requested::Symbol,
) where {T<:AbstractFloat}
    verification_bits >= 1 || throw(UnsupportedAccuracyContext(
        :verification_precision,
        "verification precision must be >= 1, got $verification_bits",
    ))
    effective = _accuracy_effective_bits(T, construction_bits)
    return AccuracyContract{T}(
        T,
        _accuracy_storage_name(T),
        effective,
        construction_bits,
        working_bits,
        verification_bits,
        rounding_mode,
        true,
        subnormal,
        _accuracy_overflow_policy(T),
        requested_error,
        requested_error,
        provider_requested,
        _accuracy_provider_loaded(T),
        :none,
        VERSION,
        is_multifloat_arithmetic(T),
    )
end

"""
    accuracy_contract(model::Model{T}, settings::Settings{T}=Settings{T}();
                      verification_precision_bits=nothing,
                      rounding=nothing, require_finite=true,
                      subnormal_policy=:default) -> AccuracyContract{T}

Derive the read-only pre-solve accuracy contract at the public Settings
seam.  Nothing is mutated (model, settings, ambient precision, routing);
every field is a live-observed fact.  Unsupported contexts throw
`UnsupportedAccuracyContext` instead of silently degrading.

Keywords:
- `verification_precision_bits` — stated verification precision
  (default: the observed working precision).
- `rounding` — required rounding mode (default: the observed mode).
  Anything but the observed `RoundNearest` is refused.
- `require_finite` — must stay `true` (SDPX finite gates are mandatory).
- `subnormal_policy` — `:default` records the backend truth; any other
  value must equal that truth or it is refused (`:flush_to_zero` is never
  supported).
"""
function accuracy_contract(
    model::Model{T},
    settings::Settings{T}=Settings{T}();
    verification_precision_bits::Union{Nothing,Integer}=nothing,
    rounding::Union{Nothing,RoundingMode}=nothing,
    require_finite::Bool=true,
    subnormal_policy::Symbol=:default,
) where {T<:AbstractFloat}
    construction_bits = precision_bits(model)
    working_bits = _accuracy_working_bits(T, _accuracy_effective_bits(T, construction_bits))
    verification = verification_precision_bits === nothing ?
                   working_bits : Int(verification_precision_bits)
    observed_rounding = _accuracy_rounding(T)
    rounding_mode = rounding === nothing ? observed_rounding :
                    _accuracy_require_rounding(T, rounding)
    rounding_mode == observed_rounding || throw(UnsupportedAccuracyContext(
        :rounding,
        "rounding $rounding_mode disagrees with the observed $T mode $observed_rounding",
    ))
    require_finite || throw(UnsupportedAccuracyContext(
        :finite_domain,
        "non-finite arithmetic is not supported: SDPX finite gates require finite data",
    ))
    subnormal = _accuracy_require_subnormal(T, subnormal_policy)
    requested = _accuracy_requested_error(T, construction_bits, settings)
    return _accuracy_build(
        T,
        construction_bits,
        working_bits,
        verification,
        rounding_mode,
        subnormal,
        requested,
        settings.provider,
    )
end

"""
    accuracy_contract(model::Model{T}, result::Result{T};
                      verification_precision_bits=nothing,
                      rounding=nothing, require_finite=true,
                      subnormal_policy=:default) -> AccuracyContract{T}

Derive the read-only post-solve accuracy contract at the result-diagnostics
seam.  The storage/effective/construction/working facts come from the source
`model` (the live model is only read); the requested/allowed error comes
from the retained `result.certificate` limits; the provider implementation
label comes from the retained `result.execution_plan`.  The result itself is
never mutated.  Refusal semantics are identical to the Settings seam.
"""
function accuracy_contract(
    model::Model{T},
    result::Result{T};
    verification_precision_bits::Union{Nothing,Integer}=nothing,
    rounding::Union{Nothing,RoundingMode}=nothing,
    require_finite::Bool=true,
    subnormal_policy::Symbol=:default,
) where {T<:AbstractFloat}
    construction_bits = precision_bits(model)
    working_bits = _accuracy_working_bits(T, _accuracy_effective_bits(T, construction_bits))
    verification = verification_precision_bits === nothing ?
                   working_bits : Int(verification_precision_bits)
    observed_rounding = _accuracy_rounding(T)
    rounding_mode = rounding === nothing ? observed_rounding :
                    _accuracy_require_rounding(T, rounding)
    rounding_mode == observed_rounding || throw(UnsupportedAccuracyContext(
        :rounding,
        "rounding $rounding_mode disagrees with the observed $T mode $observed_rounding",
    ))
    require_finite || throw(UnsupportedAccuracyContext(
        :finite_domain,
        "non-finite arithmetic is not supported: SDPX finite gates require finite data",
    ))
    subnormal = _accuracy_require_subnormal(T, subnormal_policy)
    certificate = result.certificate
    strictest = minimum(Float64.(
        (certificate.primal_limit, certificate.dual_limit, certificate.gap_limit),
    ))
    isfinite(strictest) && strictest > 0.0 || throw(UnsupportedAccuracyContext(
        :precision_bits,
        "retained certificate limits are not positive finite values",
    ))
    implementation = try
        Symbol(result.execution_plan.la_config.provider_implementation)
    catch exception
        _recoverable(exception) || rethrow()
        :unknown
    end
    base = _accuracy_build(
        T,
        construction_bits,
        working_bits,
        verification,
        rounding_mode,
        subnormal,
        strictest,
        :from_result,
    )
    return AccuracyContract{T}(
        base.storage_type,
        base.storage_name,
        base.effective_bits,
        base.construction_precision_bits,
        base.working_precision_bits,
        base.verification_precision_bits,
        base.rounding,
        base.finite_required,
        base.subnormal_policy,
        base.overflow_policy,
        base.requested_error,
        base.allowed_error,
        base.provider_requested,
        base.provider_loaded,
        implementation,
        base.julia_version,
        base.multifloat_available,
    )
end
