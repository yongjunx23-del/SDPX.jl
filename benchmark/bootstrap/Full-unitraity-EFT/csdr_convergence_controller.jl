"""
    CSDRConvergence

Small, solver-independent validation and refinement controller for the CSDR
parameter sweep.  The controller consumes one TOML report per point and never
looks at solver internals; this makes it safe to run after a batch job and
keeps the convergence decision independent of the driver implementation.

The default sweep is the production Float64x2/MFLA sweep:

``J = (40, 80, 160, 320)``, ``Nₐ = 3J/8``,
``Nμ = (400, 800, 1600, 3200)``, and alpha counts
``(2, 3, 5, 9, 17, 33)``.

The public entry points are [`read_point_toml`](@ref),
[`validate_point`](@ref), [`axis_diagnostics`](@ref), and
[`adaptive_manifest`](@ref).  All invalid input is represented as a failed
record; no missing or stopped solve is silently converted into a bound.
"""
module CSDRConvergence

using TOML

export SweepSpec,
    ValidationPolicy,
    PointKey,
    PointResult,
    RefinementDelta,
    AxisSummary,
    FinalCornerSummary,
    AdaptiveAction,
    AdaptiveManifest,
    default_sweep,
    alpha_level,
    canonical_alpha_labels,
    na_for_j,
    read_point_toml,
    validate_point,
    validate_points,
    relative_endpoint_delta,
    three_point_spread,
    axis_diagnostics,
    adaptive_manifest,
    manifest_dict,
    write_manifest,
    resource_frontier_point

const _DEFAULT_J = (40, 80, 160, 320)
const _DEFAULT_NMU = (400, 800, 1600, 3200)
const _DEFAULT_ALPHA = (2, 3, 5, 9, 17, 33)

"""Description of the allowed CSDR refinement lattice."""
struct SweepSpec
    nx::Int
    Js::Vector{Int}
    Nmus::Vector{Int}
    alpha_counts::Vector{Int}
    tolerance::Float64
    relative_tolerance::Float64
    precompute_precision_bits::Int
    solve_arithmetic::String
    la_provider::String
    expected_identity::Dict{String,String}
end

function SweepSpec(
    ;
    nx::Integer=1,
    Js=_DEFAULT_J,
    Nmus=_DEFAULT_NMU,
    alpha_counts=_DEFAULT_ALPHA,
    tolerance::Real=1e-6,
    relative_tolerance::Real=1e-4,
    precompute_precision_bits::Integer=256,
    solve_arithmetic="Float64x2",
    la_provider="multifloat_linear_algebra",
    expected_identity=Dict{String,String}(),
)
    j_values = sort!(unique!(Int.(collect(Js))))
    mu_values = sort!(unique!(Int.(collect(Nmus))))
    alpha_values = sort!(unique!(Int.(collect(alpha_counts))))
    isempty(j_values) && throw(ArgumentError("Js must not be empty"))
    isempty(mu_values) && throw(ArgumentError("Nmus must not be empty"))
    isempty(alpha_values) && throw(ArgumentError("alpha_counts must not be empty"))
    all(>(0), j_values) || throw(ArgumentError("J values must be positive"))
    all(>(0), mu_values) || throw(ArgumentError("Nmu values must be positive"))
    all(>=(2), alpha_values) || throw(ArgumentError("alpha counts must be at least 2"))
    isfinite(Float64(tolerance)) && Float64(tolerance) > 0 ||
        throw(ArgumentError("tolerance must be positive and finite"))
    isfinite(Float64(relative_tolerance)) && Float64(relative_tolerance) > 0 ||
        throw(ArgumentError("relative_tolerance must be positive and finite"))
    precompute_precision_bits >= 64 ||
        throw(ArgumentError("precompute precision must be at least 64 bits"))
    ids = Dict{String,String}()
    for (key, value) in pairs(expected_identity)
        ids[string(key)] = string(value)
    end
    return SweepSpec(
        Int(nx),
        j_values,
        mu_values,
        alpha_values,
        Float64(tolerance),
        Float64(relative_tolerance),
        Int(precompute_precision_bits),
        string(solve_arithmetic),
        string(la_provider),
        ids,
    )
end

"""Default requested CSDR sweep."""
default_sweep() = SweepSpec()

"""Validation switches for one result row."""
struct ValidationPolicy
    tolerance::Float64
    relative_tolerance::Float64
    orientation_tolerance::Float64
    repeat_tolerance::Float64
    require_optimal::Bool
    require_certificate::Bool
    require_provider::Bool
    require_no_fallback::Bool
    require_tolerance_fields::Bool
    require_alpha_set::Bool
    require_numerical_gate::Bool
    max_retries::Int
end

function ValidationPolicy(
    ;
    tolerance::Real=1e-6,
    relative_tolerance::Real=1e-4,
    orientation_tolerance::Real=1e-6,
    repeat_tolerance::Real=0.25e-6,
    require_optimal::Bool=true,
    require_certificate::Bool=true,
    require_provider::Bool=true,
    require_no_fallback::Bool=true,
    require_tolerance_fields::Bool=true,
    require_alpha_set::Bool=true,
    require_numerical_gate::Bool=true,
    max_retries::Integer=1,
)
    for (name, value) in (
        (:tolerance, tolerance),
        (:relative_tolerance, relative_tolerance),
        (:orientation_tolerance, orientation_tolerance),
        (:repeat_tolerance, repeat_tolerance),
    )
        isfinite(Float64(value)) && Float64(value) >= 0 ||
            throw(ArgumentError("$name must be finite and nonnegative"))
    end
    max_retries >= 0 || throw(ArgumentError("max_retries must be nonnegative"))
    return ValidationPolicy(
        Float64(tolerance),
        Float64(relative_tolerance),
        Float64(orientation_tolerance),
        Float64(repeat_tolerance),
        require_optimal,
        require_certificate,
        require_provider,
        require_no_fallback,
        require_tolerance_fields,
        require_alpha_set,
        require_numerical_gate,
        Int(max_retries),
    )
end

"""Coordinate identifying one point of the sweep."""
struct PointKey
    J::Int
    Nmu::Int
    alpha_count::Int
end

Base.:(==)(a::PointKey, b::PointKey) =
    a.J == b.J && a.Nmu == b.Nmu && a.alpha_count == b.alpha_count
Base.hash(key::PointKey, h::UInt) = hash((key.J, key.Nmu, key.alpha_count), h)
Base.show(io::IO, key::PointKey) = print(
    io,
    "PointKey(J=", key.J,
    ", Nmu=", key.Nmu,
    ", alpha_count=", key.alpha_count,
    ")",
)

"""Validated or rejected result row loaded from a TOML report."""
struct PointResult
    path::String
    raw::Dict{String,Any}
    key::Union{Nothing,PointKey}
    valid::Bool
    reasons::Vector{String}
    status::String
    certificate_available::Bool
    certificate_valid::Bool
    planned_provider::String
    provider::String
    fallback_reasons::Vector{String}
    lower::Union{Nothing,BigFloat}
    upper::Union{Nothing,BigFloat}
    midpoint::Union{Nothing,BigFloat}
    width::Union{Nothing,BigFloat}
    scale::Union{Nothing,BigFloat}
    orientation_repaired::Bool
    resource_frontier::Bool
end

"""Endpoint change between two valid rows."""
struct RefinementDelta
    axis::Symbol
    previous::Union{Nothing,PointKey}
    current::Union{Nothing,PointKey}
    lower_absolute::Union{Nothing,BigFloat}
    upper_absolute::Union{Nothing,BigFloat}
    midpoint_absolute::Union{Nothing,BigFloat}
    endpoint_relative::Union{Nothing,BigFloat}
    midpoint_relative::Union{Nothing,BigFloat}
    width_relative::Union{Nothing,BigFloat}
    eligible::Bool
    pass::Bool
    reason::String
end

"""Summary of one fixed-anchor refinement axis."""
struct AxisSummary
    axis::Symbol
    anchor::PointKey
    levels::Vector{Int}
    checked_levels::Vector{Int}
    valid_count::Int
    invalid_count::Int
    deltas::Vector{RefinementDelta}
    spread::Union{Nothing,BigFloat}
    monotonic_ok::Bool
    monotonic_violations::Vector{String}
    status::Symbol
    converged::Bool
    closed_on_ladder::Bool
    selected_level::Int
    next_level::Union{Nothing,Int}
    next_key::Union{Nothing,PointKey}
    reason::String
end

"""One deterministic action requested by [`adaptive_manifest`](@ref)."""
struct AdaptiveAction
    action::Symbol
    axis::Symbol
    key::Union{Nothing,PointKey}
    reason::String
end

"""Audit record for one axis of the final-corner cross.

The controller keeps this record even when the cross is incomplete.  In
particular, missing and resource-frontier rows are represented by the
``status``/``reason`` fields rather than being silently dropped from the
manifest.  Optional numerical fields are populated only when the requisite
valid rows are available; ``manifest_dict`` omits absent values so that the
result remains TOML-safe.
"""
struct FinalCornerSummary
    axis::Symbol
    candidate::PointKey
    predecessor_keys::Vector{PointKey}
    delta1::Union{Nothing,RefinementDelta}
    delta2::Union{Nothing,RefinementDelta}
    spread::Union{Nothing,BigFloat}
    pass::Bool
    status::Symbol
    reason::String
end

"""Adaptive sweep state and its next actions."""
struct AdaptiveManifest
    actions::Vector{AdaptiveAction}
    axis_reports::Vector{AxisSummary}
    selected_key::PointKey
    status::Symbol
    reason::String
    attempted_count::Int
    omitted_count::Int
    final_corner::Vector{FinalCornerSummary}
end

# Preserve the original seven-argument constructor for callers that only
# need the staged axis reports.  New manifests use the explicit final-corner
# audit vector populated by `adaptive_manifest`.
AdaptiveManifest(
    actions::Vector{AdaptiveAction},
    axis_reports::Vector{AxisSummary},
    selected_key::PointKey,
    status::Symbol,
    reason::String,
    attempted_count::Int,
    omitted_count::Int,
) = AdaptiveManifest(
    actions,
    axis_reports,
    selected_key,
    status,
    reason,
    attempted_count,
    omitted_count,
    FinalCornerSummary[],
)

_canonical_name(value) = lowercase(replace(strip(string(value)), r"[^a-zA-Z0-9]" => ""))

function _lookup(raw::AbstractDict, names::AbstractVector{<:AbstractString})
    for name in names
        haskey(raw, name) && return (true, raw[name])
        symbol_name = Symbol(name)
        haskey(raw, symbol_name) && return (true, raw[symbol_name])
    end
    return (false, nothing)
end

function _all_alias_values(raw::AbstractDict, names::AbstractVector{<:AbstractString})
    values = Pair{String,Any}[]
    for name in names
        if haskey(raw, name)
            push!(values, name => raw[name])
        elseif haskey(raw, Symbol(name))
            push!(values, name => raw[Symbol(name)])
        end
    end
    return values
end

function _to_bigfloat(value)
    value isa Bool && throw(ArgumentError("boolean is not numeric"))
    if value isa BigFloat
        return value
    elseif value isa AbstractString
        return parse(BigFloat, strip(value))
    elseif value isa Real
        return BigFloat(value)
    else
        throw(ArgumentError("not numeric"))
    end
end

function _to_int(value)
    if value isa Integer
        return Int(value)
    end
    number = _to_bigfloat(value)
    isfinite(number) && isinteger(number) || throw(ArgumentError("not integral"))
    return Int(number)
end

function _to_bool(value)
    value isa Bool && return value
    name = _canonical_name(value)
    name in ("true", "yes", "on", "1") && return true
    name in ("false", "no", "off", "0") && return false
    throw(ArgumentError("not boolean"))
end

function _read_string(raw::AbstractDict, names)
    found, value = _lookup(raw, names)
    found || return nothing
    return string(value)
end

function _read_number(raw::AbstractDict, names)
    found, value = _lookup(raw, names)
    found || return nothing
    try
        return _to_bigfloat(value)
    catch
        return nothing
    end
end

function _read_bool(raw::AbstractDict, names)
    found, value = _lookup(raw, names)
    found || return nothing
    try
        return _to_bool(value)
    catch
        return nothing
    end
end

function _read_integer(raw::AbstractDict, names)
    found, value = _lookup(raw, names)
    found || return nothing
    try
        return _to_int(value)
    catch
        return nothing
    end
end

"""Return the dyadic alpha level corresponding to a canonical point count."""
function alpha_level(count::Integer)
    count >= 2 || throw(ArgumentError("alpha count must be at least 2"))
    n = Int(count) - 1
    ispow2 = (n & (n - 1)) == 0
    ispow2 || throw(ArgumentError("alpha count $(count) is not dyadic"))
    return trailing_zeros(n) + 1
end

"""Return the exact canonical alpha labels for a dyadic point count."""
function canonical_alpha_labels(count::Integer)
    level = alpha_level(count)
    denominator = 1 << level
    last_index = 1 << (level - 1)
    labels = String["0"]
    for index in 1:last_index
        divisor = gcd(index, denominator)
        push!(labels, string(-(index ÷ divisor), "/", denominator ÷ divisor))
    end
    return labels
end

"""The required matching-row quadrature size ``Nₐ = 3J/8``."""
function na_for_j(J::Integer)
    J > 0 || throw(ArgumentError("J must be positive"))
    3 * J % 8 == 0 || throw(ArgumentError("J=$(J) does not give integral N_a=3J/8"))
    return (3 * J) ÷ 8
end

function _expected_alpha_count(spec::SweepSpec, count::Int)
    return count in spec.alpha_counts
end

function _alpha_values(raw::AbstractDict)
    found, value = _lookup(raw, ["alpha_set"])
    found || return nothing
    if value isa AbstractVector
        return string.(value)
    end
    text = strip(string(value))
    isempty(text) && return String[]
    return strip.(split(text, ','))
end

function _identity_mismatch(raw, spec::SweepSpec, reasons)
    for (name, expected) in sort!(collect(spec.expected_identity); by=first)
        aliases = if name in ("sdpx_commit", "source_commit")
            ["sdpx_commit", "source_commit"]
        elseif name in ("mfla_commit", "multifloat_linear_algebra_commit")
            ["mfla_commit", "multifloat_linear_algebra_commit", "multi_float_linear_algebra_commit"]
        else
            [name]
        end
        found, value = _lookup(raw, aliases)
        if !found
            push!(reasons, "missing_identity_$(name)")
        elseif string(value) != expected
            push!(reasons, "identity_mismatch_$(name)")
        end
    end
end

function _require_hex_identity!(raw, reasons, label, aliases, width)
    found, value = _lookup(raw, aliases)
    if !found
        _record_reason!(reasons, "missing_identity_$(label)")
        return nothing
    end
    text = strip(string(value))
    occursin(Regex("^[0-9a-fA-F]{$(width)}$"), text) ||
        _record_reason!(reasons, "invalid_identity_$(label)")
    return text
end

function _record_reason!(reasons::Vector{String}, reason::String)
    reason in reasons || push!(reasons, reason)
    return nothing
end

"""Validate one already-parsed TOML report."""
function validate_point(
    raw_input::AbstractDict;
    path::AbstractString="",
    spec::SweepSpec=default_sweep(),
    policy::ValidationPolicy=ValidationPolicy(
        tolerance=spec.tolerance,
        relative_tolerance=spec.relative_tolerance,
    ),
)
    raw = Dict{String,Any}(string(key) => value for (key, value) in pairs(raw_input))
    reasons = String[]

    j = _read_integer(raw, ["J", "j", "l_max", "L_max"])
    nmu = _read_integer(raw, ["N_mu", "Nmu", "n_mu", "nmu"])
    na = _read_integer(raw, ["N_a", "Na", "n_a", "na"])
    nx = _read_integer(raw, ["N_x", "Nx", "n_x", "nx"])
    alpha_count = _read_integer(raw, ["alpha_count", "alpha_points", "alpha_n"])
    arithmetic = _read_string(raw, ["solve_arithmetic", "arithmetic"])
    precision_bits = _read_integer(raw, [
        "precompute_precision_bits", "precision_bits", "precompute_bits",
    ])
    memory_estimate_gate = _read_bool(raw, ["memory_estimate_gate_valid"])

    key = if j !== nothing && nmu !== nothing && alpha_count !== nothing
        PointKey(j, nmu, alpha_count)
    else
        nothing
    end

    j === nothing && _record_reason!(reasons, "missing_or_invalid_J")
    nmu === nothing && _record_reason!(reasons, "missing_or_invalid_Nmu")
    na === nothing && _record_reason!(reasons, "missing_or_invalid_Na")
    nx === nothing && _record_reason!(reasons, "missing_or_invalid_Nx")
    alpha_count === nothing && _record_reason!(reasons, "missing_or_invalid_alpha_count")
    arithmetic === nothing && _record_reason!(reasons, "missing_solve_arithmetic")
    precision_bits === nothing && _record_reason!(reasons, "missing_precompute_precision_bits")
    memory_estimate_gate === nothing &&
        _record_reason!(reasons, "missing_memory_estimate_gate_valid")
    memory_estimate_gate === false &&
        _record_reason!(reasons, "memory_estimate_gate_invalid")

    arithmetic !== nothing &&
        _canonical_name(arithmetic) != _canonical_name(spec.solve_arithmetic) &&
        _record_reason!(reasons, "solve_arithmetic_mismatch")
    precision_bits !== nothing && precision_bits != spec.precompute_precision_bits &&
        _record_reason!(reasons, "precompute_precision_mismatch")

    if j !== nothing
        if !(j in spec.Js)
            _record_reason!(reasons, "J_not_in_sweep")
        else
            try
                na == na_for_j(j) || _record_reason!(reasons, "Na_rule_violation")
            catch
                _record_reason!(reasons, "Na_rule_violation")
            end
        end
    end
    nmu !== nothing && !(nmu in spec.Nmus) &&
        _record_reason!(reasons, "Nmu_not_in_sweep")
    nx !== nothing && nx != spec.nx && _record_reason!(reasons, "Nx_mismatch")
    if alpha_count !== nothing
        _expected_alpha_count(spec, alpha_count) ||
            _record_reason!(reasons, "alpha_count_not_in_sweep")
        if policy.require_alpha_set
            labels = _alpha_values(raw)
            labels === nothing && _record_reason!(reasons, "missing_alpha_set")
            if labels !== nothing && _expected_alpha_count(spec, alpha_count)
                labels == canonical_alpha_labels(alpha_count) ||
                    _record_reason!(reasons, "alpha_set_not_canonical")
            end
        end
    end

    _identity_mismatch(raw, spec, reasons)

    # Every accepted campaign row must carry immutable provenance.  The
    # names below are the explicit campaign contract emitted by the g0 driver.
    _require_hex_identity!(raw, reasons, "sdpx_commit",
                           ["sdpx_commit"], 40)
    _require_hex_identity!(raw, reasons, "mfla_commit",
                           ["mfla_commit"], 40)
    # The CSDR source tree is not itself a Git checkout.  Require its
    # aggregate tree hash; an optional source commit is only recorded when a
    # producer happens to provide one.
    _require_hex_identity!(raw, reasons, "csdr_source_tree_sha256",
                           ["csdr_source_tree_sha256"], 64)
    _require_hex_identity!(raw, reasons, "driver_sha256",
                           ["driver_sha256", "driver_hash"], 64)
    _require_hex_identity!(raw, reasons, "cache_sha256",
                           ["cache_sha256", "cache_hash"], 64)

    status = something(_read_string(raw, ["status", "solver_status"]), "")
    status_name = _canonical_name(status)
    policy.require_optimal && status_name != "optimal" &&
        _record_reason!(reasons, "status_not_optimal")

    certificate_available = something(
        _read_bool(raw, ["certificate_available", "cert_available"]),
        false,
    )
    certificate_valid = something(
        _read_bool(raw, ["certificate_valid", "cert_valid"]),
        false,
    )
    if policy.require_certificate
        !certificate_available && _record_reason!(reasons, "certificate_unavailable")
        !certificate_valid && _record_reason!(reasons, "certificate_invalid")
    end

    # The top-level residuals are the preferred fields; certificate residuals
    # are accepted for reports that expose only the independent certificate.
    p_res = _read_number(raw, ["primal_residual", "certificate_primal_residual"])
    d_res = _read_number(raw, ["dual_residual", "certificate_dual_residual"])
    gap = _read_number(raw, ["relative_gap", "certificate_relative_gap"])
    for (name, value) in (("primal_residual", p_res), ("dual_residual", d_res), ("relative_gap", gap))
        value === nothing && _record_reason!(reasons, "missing_$(name)")
        value !== nothing && !isfinite(value) && _record_reason!(reasons, "nonfinite_$(name)")
    end
    if p_res !== nothing && isfinite(p_res) && abs(p_res) > BigFloat(policy.tolerance)
        _record_reason!(reasons, "primal_residual_above_tolerance")
    end
    if d_res !== nothing && isfinite(d_res) && abs(d_res) > BigFloat(policy.tolerance)
        _record_reason!(reasons, "dual_residual_above_tolerance")
    end
    if gap !== nothing && isfinite(gap) && abs(gap) > BigFloat(policy.tolerance)
        _record_reason!(reasons, "relative_gap_above_tolerance")
    end

    if policy.require_tolerance_fields
        tolerance_names = (
            ("primal", ["tolerance_primal", "primal_tolerance"]),
            ("dual", ["tolerance_dual", "dual_tolerance"]),
            ("relative_gap", ["tolerance_relative_gap", "relative_gap_tolerance"]),
        )
        for (label, names) in tolerance_names
            value = _read_number(raw, names)
            if value === nothing
                _record_reason!(reasons, "missing_tolerance_$(label)")
            elseif !isfinite(value) || abs(value - BigFloat(policy.tolerance)) > BigFloat(policy.tolerance) * BigFloat("1e-12")
                _record_reason!(reasons, "tolerance_mismatch_$(label)")
            end
        end
    end

    lower = _read_number(raw, ["physical_g0_lower_bound", "bound_lower", "lower_bound"])
    upper = _read_number(raw, ["physical_g0_upper_bound", "bound_upper", "upper_bound"])
    objective = _read_number(raw, ["physical_g0_max", "physical_objective", "objective"])
    for (name, value) in (("bound_lower", lower), ("bound_upper", upper), ("physical_objective", objective))
        value === nothing && _record_reason!(reasons, "missing_$(name)")
        value !== nothing && !isfinite(value) && _record_reason!(reasons, "nonfinite_$(name)")
    end

    midpoint = nothing
    width = nothing
    scale = nothing
    orientation_repaired = false
    if lower !== nothing && upper !== nothing && isfinite(lower) && isfinite(upper)
        scale = max(BigFloat(1), abs(lower), abs(upper))
        if upper < lower
            if lower - upper <= BigFloat(policy.orientation_tolerance) * scale
                # Preserve the fact that a tiny reversal happened in the raw
                # report, but use the repaired interval for diagnostics.
                lower, upper = upper, lower
                orientation_repaired = true
                _record_reason!(reasons, "bound_orientation_repaired")
            else
                _record_reason!(reasons, "bound_orientation_invalid")
            end
        end
        midpoint = (lower + upper) / 2
        width = upper - lower
        if objective !== nothing && isfinite(objective)
            pad = BigFloat(policy.orientation_tolerance) * scale
            objective >= lower - pad && objective <= upper + pad ||
                _record_reason!(reasons, "physical_objective_outside_interval")
        end
    end

    planned_provider = something(
        _read_string(raw, [
            "la_planned_provider", "planned_la_provider", "planned_provider",
            "la_provider",
        ]),
        "",
    )
    provider = something(
        _read_string(raw, [
            "la_executed_provider", "executed_la_provider", "executed_provider",
        ]),
        "",
    )
    if policy.require_provider
        isempty(planned_provider) && _record_reason!(reasons, "missing_planned_la_provider")
        isempty(provider) && _record_reason!(reasons, "missing_executed_la_provider")
        _canonical_name(planned_provider) == _canonical_name(spec.la_provider) ||
            _record_reason!(reasons, "planned_la_provider_mismatch")
        _canonical_name(provider) == _canonical_name(spec.la_provider) ||
            _record_reason!(reasons, "executed_la_provider_mismatch")
    end

    fallback_values = String[]
    for names in (
        ["la_fallback_reason", "linear_algebra_fallback_reason"],
        ["fallback_reason", "backend_fallback_reason"],
    )
        value = _read_string(raw, names)
        if value === nothing
            policy.require_no_fallback && _record_reason!(reasons, "missing_fallback_reason")
        else
            push!(fallback_values, value)
            policy.require_no_fallback &&
                _canonical_name(value) != "none" &&
                _record_reason!(reasons, "fallback_detected")
        end
    end

    if policy.require_numerical_gate
        numerical_gate = _read_bool(raw, ["numerical_gate_valid", "full_numerical_gate"])
        numerical_gate === true || _record_reason!(reasons, "numerical_gate_false")
    end

    # A repaired orientation is a diagnostic, not a failure.  Remove it from
    # the validity reasons while retaining it in the raw-derived fields.
    filter!(reason -> reason != "bound_orientation_repaired", reasons)
    valid = isempty(reasons)
    return PointResult(
        string(path),
        raw,
        key,
        valid,
        reasons,
        status,
        certificate_available,
        certificate_valid,
        planned_provider,
        provider,
        fallback_values,
        lower,
        upper,
        midpoint,
        width,
        scale,
        orientation_repaired,
        false,
    )
end

"""Read and validate one per-point TOML report."""
function read_point_toml(
    path::AbstractString;
    spec::SweepSpec=default_sweep(),
    policy::ValidationPolicy=ValidationPolicy(
        tolerance=spec.tolerance,
        relative_tolerance=spec.relative_tolerance,
    ),
)
    if !isfile(path)
        return PointResult(
            string(path), Dict{String,Any}(), nothing, false,
            ["missing_file"], "", false, false, "", "", String[],
            nothing, nothing, nothing, nothing, nothing,
            false, false,
        )
    end
    try
        return validate_point(TOML.parsefile(path); path=path, spec=spec, policy=policy)
    catch error
        return PointResult(
            string(path), Dict{String,Any}(), nothing, false,
            ["toml_parse_error:" * sprint(showerror, error)], "",
            false, false, "", "", String[], nothing, nothing, nothing, nothing, nothing,
            false, false,
        )
    end
end

"""Create an explicit blocked point from a scheduler/resource frontier.

The marker is deliberately not a valid optimization row.  It carries the
mathematical `PointKey` so the adaptive controller can distinguish a known
resource boundary from an unattempted point and return a blocked frontier
action instead of silently treating the point as ordinary pending work.
"""
function resource_frontier_point(
    key::PointKey;
    reason::AbstractString="resource_frontier",
    path::AbstractString="",
    raw::AbstractDict=Dict{String,Any}(),
)
    values = Dict{String,Any}(string(name) => value for (name, value) in pairs(raw))
    values["J"] = key.J
    values["N_mu"] = key.Nmu
    values["N_a"] = try
        na_for_j(key.J)
    catch
        0
    end
    values["alpha_count"] = key.alpha_count
    values["resource_frontier"] = true
    text = isempty(strip(reason)) ? "resource_frontier" : strip(reason)
    return PointResult(
        string(path),
        values,
        key,
        false,
        ["resource_frontier:" * text],
        "ResourceFrontier",
        false,
        false,
        "",
        "",
        String[],
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        false,
        true,
    )
end

"""Read and validate a collection of per-point TOML reports."""
function validate_points(
    paths;
    spec::SweepSpec=default_sweep(),
    policy::ValidationPolicy=ValidationPolicy(
        tolerance=spec.tolerance,
        relative_tolerance=spec.relative_tolerance,
    ),
)
    return [read_point_toml(path; spec=spec, policy=policy) for path in paths]
end

function _delta_invalid(axis, previous, current, reason)
    return RefinementDelta(
        axis, previous === nothing ? nothing : previous.key,
        current === nothing ? nothing : current.key,
        nothing, nothing, nothing, nothing, nothing, nothing,
        false, false, reason,
    )
end

"""Compute interval endpoint and midpoint changes for two rows."""
function relative_endpoint_delta(
    previous::PointResult,
    current::PointResult;
    axis::Symbol=:unknown,
    threshold::Real=1e-4,
)
    if !previous.valid || !current.valid ||
       previous.lower === nothing || previous.upper === nothing ||
       current.lower === nothing || current.upper === nothing
        return _delta_invalid(axis, previous, current, "invalid_row")
    end
    lower_absolute = abs(current.lower - previous.lower)
    upper_absolute = abs(current.upper - previous.upper)
    midpoint_absolute = abs(current.midpoint - previous.midpoint)
    scale = max(
        BigFloat(1),
        abs(previous.lower), abs(previous.upper),
        abs(current.lower), abs(current.upper),
    )
    endpoint_relative = max(lower_absolute, upper_absolute) / scale
    midpoint_relative = midpoint_absolute / scale
    width_relative = abs(current.width - previous.width) / scale
    pass = endpoint_relative <= BigFloat(threshold)
    return RefinementDelta(
        axis,
        previous.key,
        current.key,
        lower_absolute,
        upper_absolute,
        midpoint_absolute,
        endpoint_relative,
        midpoint_relative,
        width_relative,
        true,
        pass,
        pass ? "within_relative_tolerance" : "relative_change_above_tolerance",
    )
end

"""Return the robust endpoint spread over up to three valid rows."""
function three_point_spread(rows::AbstractVector{<:PointResult})
    length(rows) >= 3 || return nothing
    tail = rows[(end - 2):end]
    all(row -> row.valid && row.lower !== nothing && row.upper !== nothing, tail) ||
        return nothing
    lower_values = BigFloat[row.lower for row in tail]
    upper_values = BigFloat[row.upper for row in tail]
    scale = maximum(vcat(BigFloat[1], abs.(lower_values), abs.(upper_values)))
    return max(maximum(lower_values) - minimum(lower_values),
               maximum(upper_values) - minimum(upper_values)) / scale
end

function _axis_values(spec::SweepSpec, axis::Symbol)
    axis === :J && return spec.Js
    axis === :Nmu && return spec.Nmus
    axis === :alpha && return spec.alpha_counts
    throw(ArgumentError("axis must be :alpha, :J, or :Nmu"))
end

function _replace_key(key::PointKey, axis::Symbol, value::Int)
    axis === :J && return PointKey(value, key.Nmu, key.alpha_count)
    axis === :Nmu && return PointKey(key.J, value, key.alpha_count)
    axis === :alpha && return PointKey(key.J, key.Nmu, value)
    throw(ArgumentError("axis must be :alpha, :J, or :Nmu"))
end

function _key_description(key::PointKey)
    return "J=$(key.J),Nmu=$(key.Nmu),alpha=$(key.alpha_count)"
end

function _raw_text(row::PointResult, names)
    found, value = _lookup(row.raw, names)
    return found ? string(value) : ""
end

function _valid_rows_conflict(rows::AbstractVector{<:PointResult})
    valid_rows = [row for row in rows if row.valid]
    length(valid_rows) <= 1 && return false
    reference = first(valid_rows)
    identity_names = (
        ["sdpx_commit"], ["mfla_commit"], ["driver_sha256"],
        ["cache_sha256"], ["csdr_source_tree_sha256"],
        ["la_planned_provider"], ["la_executed_provider"],
        ["fallback_reason"], ["la_fallback_reason"],
        ["physical_g0_max"],
    )
    for row in Iterators.drop(valid_rows, 1)
        reference.lower == row.lower && reference.upper == row.upper || return true
        for names in identity_names
            _raw_text(reference, names) == _raw_text(row, names) || return true
        end
    end
    return false
end

function _conflict_result(row::PointResult, reason::AbstractString)
    return PointResult(
        row.path,
        row.raw,
        row.key,
        false,
        unique(vcat(row.reasons, [reason])),
        row.status,
        row.certificate_available,
        row.certificate_valid,
        row.planned_provider,
        row.provider,
        row.fallback_reasons,
        row.lower,
        row.upper,
        row.midpoint,
        row.width,
        row.scale,
        row.orientation_repaired,
        false,
    )
end

"""Aggregate cumulative rows by mathematical point, fail-closed on conflict."""
function _result_map(
    results;
    policy::ValidationPolicy=ValidationPolicy(),
)
    groups = Dict{PointKey,Vector{PointResult}}()
    attempts = Dict{PointKey,Int}()
    for result in results
        result.key === nothing && continue
        key = result.key
        normalized = if result.valid &&
            _read_bool(result.raw, ["numerical_gate_valid", "full_numerical_gate"]) !== true
            _conflict_result(result, "numerical_gate_false")
        else
            result
        end
        push!(get!(groups, key, PointResult[]), normalized)
        attempts[key] = get(attempts, key, 0) + 1
    end
    map = Dict{PointKey,PointResult}()
    conflicts = Dict{PointKey,String}()
    for (key, rows) in groups
        valid_rows = [row for row in rows if row.valid]
        frontier_rows = [row for row in rows if row.resource_frontier]
        if _valid_rows_conflict(rows)
            map[key] = _conflict_result(first(valid_rows), "conflicting_valid_results")
            conflicts[key] = "conflicting_valid_results"
        elseif !isempty(valid_rows) && !isempty(frontier_rows)
            map[key] = _conflict_result(first(valid_rows), "valid_resource_frontier_conflict")
            conflicts[key] = "valid_resource_frontier_conflict"
        elseif !isempty(valid_rows)
            map[key] = first(valid_rows)
        elseif !isempty(frontier_rows)
            map[key] = first(frontier_rows)
        else
            map[key] = last(rows)
        end
    end
    return map, attempts, conflicts
end

function _resource_reason(result::PointResult)
    result.resource_frontier && return true
    for reason in vcat(result.reasons, result.fallback_reasons)
        name = _canonical_name(reason)
        if occursin("resource", name) || occursin("oom", name) ||
           occursin("timeout", name) || occursin("scheduler", name)
            return true
        end
    end
    return false
end

function _alpha_monotonic(rows, tolerance)
    violations = String[]
    for index in 2:length(rows)
        previous, current = rows[index - 1], rows[index]
        scale = max(
            BigFloat(1), abs(previous.lower), abs(previous.upper),
            abs(current.lower), abs(current.upper),
        )
        pad = BigFloat(tolerance) * scale
        # Alpha refinement is a nested tightening for this maximization, so
        # the upper endpoint should not increase beyond uncertainty padding.
        current.upper <= previous.upper + pad || push!(
            violations,
            "upper_increase:" * _key_description(previous.key) *
            "->" * _key_description(current.key),
        )
    end
    return isempty(violations), violations
end

"""Inspect one axis at a fixed anchor and select its next level."""
function axis_diagnostics(
    results::AbstractVector{<:PointResult},
    axis::Symbol;
    anchor::PointKey,
    spec::SweepSpec=default_sweep(),
    policy::ValidationPolicy=ValidationPolicy(
        tolerance=spec.tolerance,
        relative_tolerance=spec.relative_tolerance,
    ),
)
    levels = _axis_values(spec, axis)
    result_map, attempts, _ = _result_map(results; policy=policy)
    rows = PointResult[]
    checked_levels = Int[]
    invalid_count = 0
    next_level = nothing
    next_key = nothing
    reason = ""
    frontier_hit = false

    for level in levels
        key = _replace_key(anchor, axis, level)
        if !haskey(result_map, key)
            next_level = level
            next_key = key
            reason = "missing_report"
            break
        end
        row = result_map[key]
        push!(checked_levels, level)
        if !row.valid
            invalid_count += 1
            next_level = level
            next_key = key
            if row.resource_frontier
                frontier_hit = true
                reason = isempty(row.reasons) ?
                    "resource_frontier" : first(row.reasons)
            elseif any(occursin("conflicting_valid_results", text) for text in row.reasons)
                reason = "conflicting_valid_results"
            elseif attempts[key] <= policy.max_retries
                reason = _resource_reason(row) ? "resource_retry" : "invalid_retry"
            else
                reason = _resource_reason(row) ? "resource_frontier" : "repeated_invalid"
            end
            break
        end
        push!(rows, row)
    end

    deltas = RefinementDelta[]
    for index in 2:length(rows)
        push!(deltas, relative_endpoint_delta(
            rows[index - 1], rows[index]; axis=axis,
            threshold=policy.relative_tolerance,
        ))
    end
    spread = three_point_spread(rows)
    monotonic_ok, monotonic_violations = axis === :alpha ?
        _alpha_monotonic(rows, policy.relative_tolerance) : (true, String[])
    two_passes = length(deltas) >= 2 &&
        deltas[end - 1].pass && deltas[end].pass
    spread_pass = spread !== nothing && spread <= BigFloat(policy.relative_tolerance)
    converged = two_passes && spread_pass && monotonic_ok
    selected_level = isempty(rows) ? first(levels) : rows[end].key === nothing ? first(levels) :
        (axis === :J ? rows[end].key.J : axis === :Nmu ? rows[end].key.Nmu : rows[end].key.alpha_count)
    closed_on_ladder = false
    status = :pending

    if converged
        if selected_level == last(levels)
            status = :closed_on_ladder
            closed_on_ladder = true
            next_level = nothing
            next_key = nothing
            reason = "two_consecutive_steps_and_three_point_spread"
        else
            status = :converged_local
            next_level = nothing
            next_key = nothing
            reason = "two_consecutive_steps_and_three_point_spread"
        end
    elseif !isempty(rows) && next_level === nothing
        if !monotonic_ok
            status = :nonmonotone
            reason = "alpha_monotonicity_violation"
        elseif length(rows) >= length(levels)
            status = :unresolved
            reason = "requested_ladder_exhausted_without_convergence"
        else
            status = :pending
            reason = "insufficient_adjacent_levels"
        end
    elseif !isempty(rows) && next_level !== nothing
        if frontier_hit || startswith(reason, "resource_frontier")
            status = :unresolved_at_resource_frontier
        elseif reason in ("repeated_invalid", "conflicting_valid_results",
                          "valid_resource_frontier_conflict")
            status = :unresolved
        elseif reason == "invalid_retry"
            status = :pending
        else
            status = :pending
        end
    elseif isempty(rows)
        status = next_level === nothing ? :unresolved : :pending
        reason = isempty(reason) ? "no_valid_rows" : reason
    end

    return AxisSummary(
        axis,
        anchor,
        copy(levels),
        checked_levels,
        length(rows),
        invalid_count,
        deltas,
        spread,
        monotonic_ok,
        monotonic_violations,
        status,
        converged,
        closed_on_ladder,
        selected_level,
        next_level,
        next_key,
        reason,
    )
end

function _action_for_summary(summary::AxisSummary)
    summary.next_key === nothing && return nothing
    summary.status in (:unresolved, :nonmonotone) && return AdaptiveAction(
        :stop, summary.axis, nothing, summary.reason,
    )
    action = summary.status === :unresolved_at_resource_frontier ? :blocked :
        summary.status === :pending && summary.reason == "invalid_retry" ? :retry : :run
    return AdaptiveAction(action, summary.axis, summary.next_key, summary.reason)
end

function _final_audit(
    axis::Symbol,
    candidate::PointKey,
    predecessor_keys::Vector{PointKey};
    delta1=nothing,
    delta2=nothing,
    spread=nothing,
    pass::Bool=false,
    status::Symbol=:pending,
    reason::AbstractString="",
)
    return FinalCornerSummary(
        axis,
        candidate,
        copy(predecessor_keys),
        delta1,
        delta2,
        spread,
        pass,
        status,
        string(reason),
    )
end

"""Evaluate one final-corner axis and optionally promote its candidate level.

The returned tuple is `(actions, promotion, audit)`.  A non-`nothing`
promotion is already present and valid at the final cross; the caller must
restart the alpha/J/Nmu pass so the other two axes are checked at the new
cross.  The audit is retained in the manifest even when actions are returned
for missing, invalid, or resource-frontier rows.
"""
function _final_axis_step(
    result_map,
    attempts,
    axis,
    current::PointKey,
    spec::SweepSpec,
    policy::ValidationPolicy,
)
    values = _axis_values(spec, axis)
    current_value = axis === :alpha ? current.alpha_count :
        axis === :J ? current.J : current.Nmu
    index = findfirst(==(current_value), values)
    index === nothing && return (
        AdaptiveAction[AdaptiveAction(
            :stop, axis, nothing, "final_corner_anchor_not_on_ladder",
        )], nothing,
        _final_audit(axis, current, PointKey[]; status=:stop,
                     reason="final_corner_anchor_not_on_ladder"),
    )
    predecessor_indices = max(1, index - 2):(index - 1)
    predecessor_keys = PointKey[
        _replace_key(current, axis, values[predecessor_index])
        for predecessor_index in predecessor_indices
    ]
    isempty(predecessor_indices) && return (
        AdaptiveAction[AdaptiveAction(
            :stop, axis, nothing, "final_corner_insufficient_predecessors",
        )], nothing,
        _final_audit(axis, current, predecessor_keys; status=:stop,
                     reason="final_corner_insufficient_predecessors"),
    )
    actions = AdaptiveAction[]
    rows = PointResult[]
    audit_status = :pending
    audit_reason = ""
    for key in predecessor_keys
        if !haskey(result_map, key)
            push!(actions, AdaptiveAction(
                :fence, axis, key, "missing_final_corner_predecessor",
            ))
            isempty(audit_reason) && (audit_reason = "missing_final_corner_predecessor")
            continue
        end
        row = result_map[key]
        if !row.valid
            action = row.resource_frontier ? :blocked :
                attempts[key] <= policy.max_retries ? :retry : :stop
            push!(actions, AdaptiveAction(
                action, axis, key, row.resource_frontier ?
                    "resource_frontier_final_corner_predecessor" :
                    "invalid_final_corner_predecessor",
            ))
            if row.resource_frontier
                audit_status = :blocked
                audit_reason = "resource_frontier_final_corner_predecessor"
            elseif action === :stop
                audit_status = :invalid
                isempty(audit_reason) &&
                    (audit_reason = "invalid_final_corner_predecessor")
            else
                isempty(audit_reason) &&
                    (audit_reason = "invalid_final_corner_predecessor")
            end
            continue
        end
        push!(rows, row)
    end
    current_row = get(result_map, current, nothing)
    if !(current_row isa PointResult) || !current_row.valid
        current_action = if current_row isa PointResult && current_row.resource_frontier
            :blocked
        elseif current_row isa PointResult
            attempts[current] <= policy.max_retries ? :retry : :stop
        else
            :fence
        end
        current_reason = if current_row isa PointResult && current_row.resource_frontier
            "resource_frontier_final_corner_anchor"
        elseif current_row isa PointResult
            "invalid_final_corner_anchor"
        else
            "missing_final_corner_anchor"
        end
        push!(actions, AdaptiveAction(
            current_action,
            axis,
            current,
            current_reason,
        ))
        if current_row isa PointResult && current_row.resource_frontier
            audit_status = :blocked
            audit_reason = "resource_frontier_final_corner_anchor"
        elseif current_action === :stop
            audit_status = :invalid
            isempty(audit_reason) && (audit_reason = "invalid_final_corner_anchor")
        elseif current_row isa PointResult
            isempty(audit_reason) && (audit_reason = "invalid_final_corner_anchor")
        else
            isempty(audit_reason) && (audit_reason = "missing_final_corner_anchor")
        end
        return (
            actions,
            nothing,
            _final_audit(axis, current, predecessor_keys;
                         pass=false, status=audit_status,
                         reason=audit_reason),
        )
    end
    if !isempty(actions)
        # Preserve a blocked/invalid status when one was observed; otherwise
        # this is an ordinary pending fence/retry for missing data.
        isempty(audit_reason) && (audit_reason = "missing_final_corner_predecessor")
        audit_status in (:blocked, :invalid) || (audit_status = :pending)
        return (
            actions,
            nothing,
            _final_audit(axis, current, predecessor_keys;
                         pass=false, status=audit_status,
                         reason=audit_reason),
        )
    end
    push!(rows, current_row)
    if length(rows) < 3
        return (
            AdaptiveAction[AdaptiveAction(
                :stop, axis, nothing, "final_corner_insufficient_predecessors",
            )], nothing,
            _final_audit(axis, current, predecessor_keys; status=:stop,
                         reason="final_corner_insufficient_predecessors"),
        )
    end
    first_delta = relative_endpoint_delta(rows[1], rows[2]; axis=axis,
                                          threshold=policy.relative_tolerance)
    second_delta = relative_endpoint_delta(rows[2], rows[3]; axis=axis,
                                           threshold=policy.relative_tolerance)
    spread = three_point_spread(rows)
    if first_delta.pass && second_delta.pass && spread !== nothing &&
       spread <= BigFloat(policy.relative_tolerance)
        return (
            AdaptiveAction[],
            nothing,
            _final_audit(axis, current, predecessor_keys;
                         delta1=first_delta, delta2=second_delta,
                         spread=spread, pass=true, status=:pass,
                         reason="two_delta_and_three_point_spread"),
        )
    end
    if index < length(values)
        next_key = _replace_key(current, axis, values[index + 1])
        if haskey(result_map, next_key)
            next_row = result_map[next_key]
            if next_row.resource_frontier
                return (
                    AdaptiveAction[AdaptiveAction(
                        :blocked, axis, next_key,
                        "resource_frontier_final_corner_candidate",
                    )], nothing,
                    _final_audit(axis, current, predecessor_keys;
                                 delta1=first_delta, delta2=second_delta,
                                 spread=spread, status=:blocked,
                                 reason="resource_frontier_final_corner_candidate"),
                )
            elseif !next_row.valid
                action = attempts[next_key] <= policy.max_retries ? :retry : :stop
                return (
                    AdaptiveAction[AdaptiveAction(
                        action, axis, next_key,
                        "invalid_final_corner_candidate",
                    )], nothing,
                    _final_audit(axis, current, predecessor_keys;
                                 delta1=first_delta, delta2=second_delta,
                                 spread=spread,
                                 status=action === :stop ? :invalid : :pending,
                                 reason="invalid_final_corner_candidate"),
                )
            end
            # The next level is already present at this cross.  Promote it
            # and let the outer fixed-point loop re-check all three axes.
            return (
                AdaptiveAction[],
                next_key,
                _final_audit(axis, current, predecessor_keys;
                             delta1=first_delta, delta2=second_delta,
                             spread=spread, status=:promote,
                             reason="final_corner_delta_or_spread_above_tolerance"),
            )
        end
        return (
            AdaptiveAction[AdaptiveAction(
                :run, axis, next_key,
                "final_corner_delta_or_spread_above_tolerance",
            )], nothing,
            _final_audit(axis, current, predecessor_keys;
                         delta1=first_delta, delta2=second_delta,
                         spread=spread, status=:pending,
                         reason="final_corner_delta_or_spread_above_tolerance"),
        )
    end
    return (
        AdaptiveAction[AdaptiveAction(
            :stop, axis, nothing, "final_corner_delta_or_spread_above_tolerance",
        )], nothing,
        _final_audit(axis, current, predecessor_keys;
                     delta1=first_delta, delta2=second_delta,
                     spread=spread, status=:stop,
                     reason="final_corner_delta_or_spread_above_tolerance"),
    )
end

"""Run the final-corner cross to a fixed point without external state."""
function _final_corner_fixed_point(
    results::AbstractVector{<:PointResult},
    initial::PointKey;
    spec::SweepSpec=default_sweep(),
    policy::ValidationPolicy=ValidationPolicy(
        tolerance=spec.tolerance,
        relative_tolerance=spec.relative_tolerance,
    ),
)
    result_map, attempts, _ = _result_map(results; policy=policy)
    candidate = initial
    max_promotions = length(spec.alpha_counts) + length(spec.Js) + length(spec.Nmus)
    for _ in 1:(max_promotions + 1)
        promoted = false
        pending_actions = AdaptiveAction[]
        audits = FinalCornerSummary[]
        for axis in (:alpha, :J, :Nmu)
            axis_actions, promotion, audit = _final_axis_step(
                result_map,
                attempts,
                axis,
                candidate,
                spec,
                policy,
            )
            push!(audits, audit)
            if promotion !== nothing
                candidate = promotion
                promoted = true
                break
            end
            append!(pending_actions, axis_actions)
            # A blocked frontier or a repeated invalid/stop result is already
            # a terminal decision for this candidate.  Missing/fence/run/
            # retry actions remain collectable across all three axes so PBS
            # can launch them in parallel.
            if any(action -> action.action in (:blocked, :stop), axis_actions)
                return candidate, pending_actions, audits
            end
        end
        if promoted
            # A promotion changes the cross.  Discard audits/actions from the
            # previous candidate and restart all axes at the promoted key.
            continue
        end
        return candidate, pending_actions, audits
    end
    return candidate,
        AdaptiveAction[AdaptiveAction(
            :stop, :fence, nothing, "final_corner_fixed_point_limit",
        )],
        FinalCornerSummary[]
end

function _previous_level(values, current)
    index = findfirst(==(current), values)
    index === nothing || index == 1 ? nothing : values[index - 1]
end

"""Build the deterministic alpha → J → Nmu → final-corner action manifest."""
function adaptive_manifest(
    results::AbstractVector{<:PointResult};
    spec::SweepSpec=default_sweep(),
    policy::ValidationPolicy=ValidationPolicy(
        tolerance=spec.tolerance,
        relative_tolerance=spec.relative_tolerance,
    ),
)
    result_map, attempts, _ = _result_map(results; policy=policy)
    base = PointKey(first(spec.Js), first(spec.Nmus), first(spec.alpha_counts))
    selected = base
    reports = AxisSummary[]
    actions = AdaptiveAction[]

    # Keep this order explicit: alpha is the only structurally nested axis;
    # Nmu is deliberately postponed until J and Na have been selected.
    for axis in (:alpha, :J, :Nmu)
        summary = axis_diagnostics(
            results,
            axis;
            anchor=selected,
            spec=spec,
            policy=policy,
        )
        push!(reports, summary)
        action = _action_for_summary(summary)
        if action !== nothing
            push!(actions, action)
            break
        elseif summary.status in (:nonmonotone, :unresolved,
                                  :unresolved_at_resource_frontier)
            push!(actions, AdaptiveAction(:stop, axis, nothing, summary.reason))
            break
        end
        selected = _replace_key(selected, axis, summary.selected_level)
    end

    final_corner = FinalCornerSummary[]
    if isempty(actions) && length(reports) == 3
        # The final cross is a fixed-point inference: a valid finer point
        # already present at the cross promotes the candidate, then all three
        # axes are rechecked at the new cross.  Missing finer points become
        # actions; no external selected-level state is required.
        selected, actions, final_corner = _final_corner_fixed_point(
            results,
            selected;
            spec=spec,
            policy=policy,
        )
    end

    status = if !isempty(actions)
        any(action -> action.action === :blocked, actions) ?
            :unresolved_at_resource_frontier :
            any(action -> action.action === :stop, actions) ? :unresolved : :pending
    elseif length(reports) < 3
        :pending
    elseif all(report -> report.converged, reports)
        :converged
    else
        :unresolved
    end
    reason = if isempty(actions)
        status === :converged ? "all_axes_converged_and_final_fence_passed" :
            "no_safe_next_action"
    elseif status === :unresolved_at_resource_frontier
        first(action for action in actions if action.action === :blocked).reason
    else
        first(actions).reason
    end
    total = length(spec.Js) * length(spec.Nmus) * length(spec.alpha_counts)
    attempted_keys = PointKey[
        result.key for result in results if result.key !== nothing
    ]
    attempted = length(unique(attempted_keys))
    omitted = max(total - attempted, 0)
    return AdaptiveManifest(
        actions,
        reports,
        selected,
        status,
        reason,
        attempted,
        omitted,
        final_corner,
    )
end

function _key_dict(key::PointKey)
    return Dict("J" => key.J, "N_mu" => key.Nmu, "alpha_count" => key.alpha_count,
                "N_a" => na_for_j(key.J))
end

function _delta_dict(delta::RefinementDelta)
    result = Dict{String,Any}(
        "axis" => string(delta.axis),
        "eligible" => delta.eligible,
        "pass" => delta.pass,
        "reason" => delta.reason,
    )
    delta.endpoint_relative === nothing ||
        (result["endpoint_relative"] = string(delta.endpoint_relative))
    delta.midpoint_relative === nothing ||
        (result["midpoint_relative"] = string(delta.midpoint_relative))
    delta.width_relative === nothing ||
        (result["width_relative"] = string(delta.width_relative))
    return result
end

function _summary_dict(summary::AxisSummary)
    result = Dict{String,Any}(
        "axis" => string(summary.axis),
        "anchor" => _key_dict(summary.anchor),
        "levels" => summary.levels,
        "checked_levels" => summary.checked_levels,
        "valid_count" => summary.valid_count,
        "invalid_count" => summary.invalid_count,
        "deltas" => [_delta_dict(delta) for delta in summary.deltas],
        "monotonic_ok" => summary.monotonic_ok,
        "monotonic_violations" => summary.monotonic_violations,
        "status" => string(summary.status),
        "converged" => summary.converged,
        "closed_on_ladder" => summary.closed_on_ladder,
        "selected_level" => summary.selected_level,
        "reason" => summary.reason,
    )
    summary.spread === nothing ||
        (result["three_point_spread"] = string(summary.spread))
    summary.next_level === nothing || (result["next_level"] = summary.next_level)
    summary.next_key === nothing || (result["next_key"] = _key_dict(summary.next_key))
    return result
end

function _final_corner_dict(summary::FinalCornerSummary)
    predecessors = [_key_dict(key) for key in summary.predecessor_keys]
    result = Dict{String,Any}(
        "axis" => string(summary.axis),
        "candidate" => _key_dict(summary.candidate),
        "predecessors" => predecessors,
        # Keep an explicit alias for consumers that use the terminology from
        # the convergence specification.
        "predecessor_keys" => predecessors,
        "pass" => summary.pass,
        "status" => string(summary.status),
        "reason" => summary.reason,
    )
    if summary.delta1 !== nothing
        result["delta1"] = _delta_dict(summary.delta1)
        summary.delta1.endpoint_relative === nothing ||
            (result["delta1_endpoint_relative"] =
                string(summary.delta1.endpoint_relative))
    end
    if summary.delta2 !== nothing
        result["delta2"] = _delta_dict(summary.delta2)
        summary.delta2.endpoint_relative === nothing ||
            (result["delta2_endpoint_relative"] =
                string(summary.delta2.endpoint_relative))
    end
    summary.spread === nothing ||
        (result["three_point_spread"] = string(summary.spread))
    return result
end

"""Convert a manifest to a TOML-friendly dictionary."""
function manifest_dict(manifest::AdaptiveManifest)
    actions = Dict{String,Any}[]
    for action in manifest.actions
        entry = Dict{String,Any}(
            "action" => string(action.action),
            "axis" => string(action.axis),
            "reason" => action.reason,
        )
        action.key === nothing || (entry["key"] = _key_dict(action.key))
        push!(actions, entry)
    end
    stage = if isempty(manifest.actions)
        manifest.status === :converged ? "complete" : "unresolved"
    else
        action = first(manifest.actions)
        occursin("final_corner", action.reason) || action.action === :fence ?
            "fence" : string(action.axis)
    end
    return Dict(
        "status" => string(manifest.status),
        "stage" => stage,
        "reason" => manifest.reason,
        "selected" => _key_dict(manifest.selected_key),
        "attempted_count" => manifest.attempted_count,
        "omitted_count" => manifest.omitted_count,
        "actions" => actions,
        "axes" => [_summary_dict(summary) for summary in manifest.axis_reports],
        "final_corner" => [_final_corner_dict(summary)
                           for summary in manifest.final_corner],
    )
end

"""Write a TOML action manifest and return its path."""
function write_manifest(path::AbstractString, manifest::AdaptiveManifest)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, manifest_dict(manifest); sorted=true)
    end
    return string(path)
end

end # module CSDRConvergence
