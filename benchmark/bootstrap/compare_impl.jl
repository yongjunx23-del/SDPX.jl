function _result_key(row)
    return (
        string(get(row, "problem_id", "")),
        string(get(row, "arithmetic", "")),
        string(get(row, "requested_provider", "")),
    )
end

function _index_unique_results(rows, label)
    indexed = Dict{Tuple{String,String,String},Any}()
    for row in rows
        key = _result_key(row)
        haskey(indexed, key) && throw(ArgumentError(
            "$label contains duplicate result key $(repr(key))",
        ))
        indexed[key] = row
    end
    return indexed
end

_text(value) = value === missing || value === nothing ? "" : strip(string(value))

const _COMPARISON_EXECUTION_MODES = (:build, :solve, :profile)
const _COMPARISON_REQUESTED_ENGINES = (
    :auto, :legacy, :sdpx_legacy, :catalog_contract,
    :native, :native_hsd, :product_hsd,
)
const _COMPARISON_TIMING_FIELDS = (
    "seconds_per_iteration", "total_seconds_iqr", "gc_seconds",
    "setup_seconds", "frontend_seconds", "preprocess_seconds",
    "presolve_seconds", "core_seconds", "certification_seconds",
    "sample_median_seconds", "sample_min_seconds", "sample_max_seconds",
    "sample_mad_seconds", "sample_spread_seconds", "assembly_seconds",
    "factor_seconds", "solve_seconds", "refinement_seconds",
    "local_metric_seconds", "local_factor_seconds",
    "panel_transform_seconds", "equality_gram_seconds",
    "equality_factor_seconds", "predictor_rhs_seconds",
    "corrector_rhs_seconds", "block_residual_seconds",
    "block_recovery_seconds",
)

function _route_token(value)
    text = lowercase(replace(_text(value), '-' => '_', ' ' => '_'))
    isempty(text) && return nothing
    return Symbol(text)
end

function _canonical_requested_route(value)
    token = _route_token(value)
    token === :auto && return :auto
    token in (:legacy, :sdpx_legacy) && return :legacy
    token === :catalog_contract && return :contract
    token in (:native, :native_hsd, :product_hsd) && return :native
    return nothing
end

function _canonical_executed_route(value)
    token = _route_token(value)
    token === :none && return :none
    token in (:legacy, :sdpx_legacy) && return :legacy
    token === :catalog_contract && return :contract
    return nothing
end

"""Validate the serialized route matrix and return canonical route labels."""
function _validate_route(row; label="row")
    mode = _route_token(get(row, "execution_mode", missing))
    requested_raw = _route_token(get(row, "requested_engine", missing))
    executed_raw = _route_token(get(row, "executed_engine", missing))
    failures = String[]
    mode in _COMPARISON_EXECUTION_MODES || push!(failures, "$(label)_execution_mode_invalid")
    requested = _canonical_requested_route(requested_raw)
    requested === nothing && push!(failures, "$(label)_requested_engine_invalid")
    executed = _canonical_executed_route(executed_raw)
    executed === nothing && push!(failures, "$(label)_executed_engine_invalid")
    if mode === :build
        executed === :none || push!(failures, "$(label)_build_executed_engine_not_none")
    elseif mode in (:solve, :profile)
        executed in (:legacy, :contract) ||
            push!(failures, "$(label)_solve_executed_engine_invalid")
        requested === :native &&
            push!(failures, "$(label)_native_requested_engine_invalid")
        requested === :legacy && executed !== :legacy &&
            push!(failures, "$(label)_legacy_route_mismatch")
        requested === :contract && executed !== :contract &&
            push!(failures, "$(label)_contract_route_mismatch")
    end
    return (valid=isempty(failures), failures=failures, mode=mode,
            requested=requested, executed=executed)
end

function _timing_failures(row, label)
    failures = String[]
    total = _parsed_float(row, "total_seconds")
    (total !== nothing && total > 0) ||
        push!(failures, "$(label)_total_seconds_invalid")
    for field in _COMPARISON_TIMING_FIELDS
        haskey(row, field) || continue
        text = _number_text(row, field)
        isempty(something(text, "")) && continue
        value = _parsed_float(row, field)
        (value !== nothing && value >= 0) ||
            push!(failures, "$(label)_$(field)_invalid")
    end
    return failures
end

function _float_close(a, b)
    isfinite(a) && isfinite(b) || return false
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= 64 * eps(Float64) * scale
end

function _split_serialized_array(text)
    inner = strip(text[2:end-1])
    isempty(inner) && return String[]
    pieces = String[]
    start = firstindex(inner)
    in_quote = false
    escaped = false
    for index in eachindex(inner)
        character = inner[index]
        if in_quote
            if escaped
                escaped = false
            elseif character == '\\'
                escaped = true
            elseif character == '"'
                in_quote = false
            end
        elseif character == '"'
            in_quote = true
        elseif character == ','
            push!(pieces, strip(inner[start:prevind(inner, index)]))
            start = nextind(inner, index)
        end
    end
    in_quote && return nothing
    push!(pieces, strip(inner[start:end]))
    return pieces
end

function _serialized_token(token)
    isempty(token) && return nothing
    lowercase(token) == "null" && return nothing
    lowercase(token) == "true" && return true
    lowercase(token) == "false" && return false
    if startswith(token, "\"") && endswith(token, "\"") && length(token) >= 2
        body = token[nextind(token, firstindex(token)):prevind(token, lastindex(token))]
        return replace(replace(replace(replace(body,
            "\\\\" => "\\"), "\\\"" => "\""),
            "\\n" => "\n"), "\\r" => "\r")
    end
    return try
        parse(Float64, token)
    catch
        nothing
    end
end

function _serialized_array(row, field)
    raw = get(row, field, missing)
    (raw === missing || raw === nothing) && return nothing
    raw isa AbstractVector && return collect(raw)
    text = _text(raw)
    (length(text) >= 2 && startswith(text, "[") && endswith(text, "]")) ||
        return nothing
    tokens = _split_serialized_array(text)
    tokens === nothing && return nothing
    return [_serialized_token(token) for token in tokens]
end

function _sample_number(value)
    value isa Real && isfinite(value) && return Float64(value)
    value isa AbstractString || return nothing
    return try
        parsed = parse(Float64, strip(value))
        isfinite(parsed) ? parsed : nothing
    catch
        nothing
    end
end

function _sample_integer(value)
    value isa Integer && return BigInt(value)
    text = _text(value)
    isempty(text) && return nothing
    return try
        BigInt(text)
    catch
        parsed = _sample_number(value)
        parsed !== nothing && isinteger(parsed) ? BigInt(parsed) : nothing
    end
end

function _canonical_sample_route(row, value)
    parts = split(_text(value), '|'; keepempty=true)
    length(parts) == 16 || return nothing
    mode = _route_token(parts[1])
    requested = _canonical_requested_route(parts[2])
    executed = _canonical_executed_route(parts[3])
    (mode in _COMPARISON_EXECUTION_MODES && requested !== nothing &&
     executed !== nothing) || return nothing
    parts[1] = string(mode)
    parts[2] = string(requested)
    parts[3] = string(executed)
    return join(parts, "|")
end

function _sample_bool(value)
    value === true && return true
    value === false && return false
    text = lowercase(_text(value))
    text == "true" && return true
    text == "false" && return false
    return nothing
end

function _sample_objective_values(values, row)
    strings = [_text(value) for value in values]
    all(isempty, strings) && return nothing
    any(isempty, strings) && return nothing
    absolute = _number_text(row, "reference_absolute_tolerance")
    relative = _number_text(row, "reference_relative_tolerance")
    (absolute === nothing || relative === nothing) && return nothing
    bits = max(256, 4 * maximum(length.(strings)))
    try
        return setprecision(BigFloat, bits) do
            parsed = [parse(BigFloat, value) for value in strings]
            atol = parse(BigFloat, absolute)
            rtol = parse(BigFloat, relative)
            all(isfinite, parsed) && isfinite(atol) && atol >= 0 &&
                isfinite(rtol) && rtol >= 0 || return nothing
            reference = first(parsed)
            allowed = atol + rtol * max(one(BigFloat), abs(reference))
            all(value -> abs(value - reference) <= allowed, parsed) ? parsed : nothing
        end
    catch
        return nothing
    end
end

function _sample_validation(row, label)
    failures = String[]
    count = _parsed_integer(row, "sample_count")
    count isa Integer && count >= 3 && count <= typemax(Int) || return (
        valid=false, failures=["$(label)_sample_count_invalid"],
        seconds=Float64[], statuses=String[], iterations=BigInt[],
        objectives=nothing, certificates=Any[], routes=String[],
        semantic=Bool[], count=count,
    )
    n = Int(count)
    fields = Dict(field => _serialized_array(row, field) for field in (
        "sample_seconds", "sample_semantic_pass", "sample_status",
        "sample_iterations", "sample_objective", "sample_certificate_valid",
        "sample_route",
    ))
    for (field, values) in fields
        values === nothing && push!(failures, "$(label)_$(field)_missing")
        values !== nothing && length(values) == n ||
            (values !== nothing && push!(failures, "$(label)_$(field)_length"))
    end
    seconds = Float64[]
    if fields["sample_seconds"] !== nothing && length(fields["sample_seconds"]) == n
        for value in fields["sample_seconds"]
            parsed = _sample_number(value)
            (parsed !== nothing && parsed > 0) || push!(failures, "$(label)_sample_seconds_invalid")
            parsed === nothing || push!(seconds, parsed)
        end
    end
    semantic = Bool[]
    for value in something(fields["sample_semantic_pass"], Any[])
        parsed = _sample_bool(value)
        parsed === nothing && push!(failures, "$(label)_sample_semantic_pass_invalid")
        parsed === nothing || push!(semantic, parsed)
    end
    statuses = [_text(value) for value in something(fields["sample_status"], Any[])]
    any(isempty, statuses) && push!(failures, "$(label)_sample_status_empty")
    build_mode = _route_token(get(row, "execution_mode", missing)) === :build
    iterations = BigInt[]
    for value in something(fields["sample_iterations"], Any[])
        parsed = _sample_integer(value)
        if parsed === nothing && !(build_mode && isempty(_text(value)))
            push!(failures, "$(label)_sample_iterations_invalid")
        else
            parsed === nothing || push!(iterations, parsed)
        end
    end
    certificates = Any[]
    for value in something(fields["sample_certificate_valid"], Any[])
        parsed = _sample_bool(value)
        if parsed === nothing &&
           !(build_mode && (value === nothing || isempty(_text(value))))
            push!(failures, "$(label)_sample_certificate_invalid")
        else
            push!(certificates, parsed)
        end
    end
    routes = String[]
    for value in something(fields["sample_route"], Any[])
        route = _canonical_sample_route(row, value)
        route === nothing && push!(failures, "$(label)_sample_route_invalid")
        route === nothing || push!(routes, route)
    end
    any(isempty, routes) && push!(failures, "$(label)_sample_route_empty")
    objectives = if fields["sample_objective"] === nothing
        push!(failures, "$(label)_sample_objective_missing")
        nothing
    elseif all(value -> value === nothing || isempty(_text(value)),
               fields["sample_objective"]) &&
           isempty(_text(get(row, "objective", "")))
        nothing
    else
        _sample_objective_values(fields["sample_objective"], row)
    end
    !build_mode && objectives === nothing &&
        !any(occursin("sample_objective", failure) for failure in failures) &&
        push!(failures, "$(label)_sample_objective_invalid")
    isempty(semantic) || all(semantic) || push!(failures, "$(label)_sample_semantic_pass")
    isempty(statuses) || length(unique(statuses)) == 1 || push!(failures, "$(label)_sample_status_parity")
    isempty(iterations) || length(unique(iterations)) == 1 || push!(failures, "$(label)_sample_iterations_parity")
    if !isempty(certificates)
        certificate_values = unique(certificates)
        if build_mode
            length(certificate_values) == 1 && first(certificate_values) === nothing ||
                push!(failures, "$(label)_sample_certificate_parity")
        else
            length(certificate_values) == 1 && first(certificate_values) === true ||
                push!(failures, "$(label)_sample_certificate_parity")
        end
    end
    isempty(routes) || length(unique(routes)) == 1 || push!(failures, "$(label)_sample_route_parity")
    route_info = _validate_route(row; label=label)
    expected_route = join((
        string(route_info.mode), string(route_info.requested),
        string(route_info.executed),
        (_text(get(row, field, "")) for field in (
            "scaling", "layout", "conic_formulation", "planned_formulation",
            "executed_formulation", "planned_backend", "executed_backend",
            "planned_provider", "executed_provider", "executed_specialization",
            "psd_lift_used", "fallback_reason", "la_fallback_reason",
        ))...
    ), "|")
    isempty(routes) || all(route -> route == expected_route, routes) ||
        push!(failures, "$(label)_sample_route_mismatch")
    expected_status = _text(get(row, "status", ""))
    isempty(statuses) || all(status -> status == expected_status, statuses) ||
        push!(failures, "$(label)_sample_status_mismatch")
    expected_iterations = _sample_integer(get(row, "iterations", ""))
    expected_iterations === nothing || isempty(iterations) ||
        all(iteration -> iteration == expected_iterations, iterations) ||
        push!(failures, "$(label)_sample_iterations_mismatch")
    expected_certificate = _sample_bool(get(row, "certificate_valid", missing))
    if build_mode
        expected_certificate === nothing || isempty(certificates) ||
            all(certificate -> certificate === nothing, certificates) ||
            push!(failures, "$(label)_sample_certificate_mismatch")
    else
        expected_certificate === true ||
            push!(failures, "$(label)_sample_certificate_mismatch")
    end
    declared_parity = _sample_bool(get(row, "sample_semantic_parity", missing))
    declared_parity === false && push!(failures, "$(label)_sample_semantic_parity_false")
    declared_failures = _text(get(row, "sample_parity_failures", ""))
    isempty(declared_failures) || push!(failures, "$(label)_sample_parity_failures_nonempty")
    if length(seconds) == n
        ordered = sort(seconds)
        center = isodd(n) ? ordered[(n + 1) ÷ 2] :
                 (ordered[n ÷ 2] + ordered[n ÷ 2 + 1]) / 2
        deviations = sort(abs.(ordered .- center))
        summary = (
            total_seconds=center,
            sample_median_seconds=center,
            sample_min_seconds=first(ordered),
            sample_max_seconds=last(ordered),
            sample_mad_seconds=(isodd(n) ? deviations[(n + 1) ÷ 2] :
                (deviations[n ÷ 2] + deviations[n ÷ 2 + 1]) / 2),
            sample_spread_seconds=last(ordered) - first(ordered),
            total_seconds_iqr=quantile(ordered, 0.75) - quantile(ordered, 0.25),
        )
        for (field, expected) in pairs(summary)
            actual = _parsed_float(row, string(field))
            (actual !== nothing && actual >= 0 && _float_close(actual, expected)) ||
                push!(failures, "$(label)_$(field)_mismatch")
        end
    end
    return (valid=isempty(failures), failures=failures, seconds=seconds,
            statuses, iterations, objectives, certificates, routes, semantic, count)
end

function _sample_semantic_match(before, after, baseline, candidate)
    before.statuses == after.statuses || return false
    before.iterations == after.iterations || return false
    before.certificates == after.certificates || return false
    before.routes == after.routes || return false
    before.semantic == after.semantic || return false
    if before.objectives === nothing || after.objectives === nothing
        return before.objectives === nothing && after.objectives === nothing &&
               isempty(_text(get(baseline, "objective", ""))) &&
               isempty(_text(get(candidate, "objective", "")))
    end
    absolute = _number_text(baseline, "reference_absolute_tolerance")
    relative = _number_text(baseline, "reference_relative_tolerance")
    if absolute === nothing || relative === nothing
        return false
    end
    bits = max(256, 4 * maximum(length.([_text(x) for x in before.objectives]); init=0))
    return try
        setprecision(BigFloat, bits) do
            atol = parse(BigFloat, absolute)
            rtol = parse(BigFloat, relative)
            all(index -> begin
                a = parse(BigFloat, _text(before.objectives[index]))
                b = parse(BigFloat, _text(after.objectives[index]))
                abs(a - b) <= atol + rtol * max(one(BigFloat), abs(a))
            end, eachindex(before.objectives))
        end
    catch
        false
    end
end

"""Return a non-empty textual representation of a scalar result field."""
function _number_text(row, field)
    value = get(row, field, "")
    (value === missing || value === nothing) && return nothing
    text = strip(string(value))
    isempty(text) && return nothing
    return text
end

const _SHA256_HEX64_RE = r"^[0-9a-f]{64}$"

"""Return whether a serialized solver-source hash is a canonical SHA-256."""
function _valid_solver_source_sha256(value)
    (value === missing || value === nothing) && return false
    text = strip(string(value))
    isempty(text) && return false
    return occursin(_SHA256_HEX64_RE, text)
end

"""Parse a finite timing-like scalar without throwing on malformed TOML."""
function _parsed_float(row, field)
    text = _number_text(row, field)
    text === nothing && return nothing
    value = try
        parse(Float64, text)
    catch
        nothing
    end
    value isa Real && isfinite(value) || return nothing
    return Float64(value)
end

# Keep the historical private helper available to downstream benchmark
# scripts; comparison internals now use the more explicit name above.
_parsed_number(row, field) = _parsed_float(row, field)

# result_schema.jl has been v9 since the fresh-process identity/route
# contract landed. Keep comparison strict: v8 is legacy, never current.
const _CURRENT_RESULT_SCHEMA_VERSION = 9

const _COMPARISON_PAIRING_FIELDS = (
    # A baseline and candidate must exercise the same semantic route and
    # shard topology.  Run-specific identifiers are deliberately excluded so
    # an otherwise valid comparison can cross PBS jobs.
    "execution_mode", "requested_engine", "executed_engine",
    "scaling", "layout", "conic_formulation",
    "planned_formulation", "executed_formulation",
    "planned_backend", "executed_backend",
    "planned_provider", "executed_provider",
    "executed_specialization", "psd_lift_used",
    "fallback_reason", "la_fallback_reason",
)

const _AUDIT_IDENTITY_FIELDS = (
    # These fields are required to be present in current rows, but their
    # values may differ between independent baseline/candidate campaigns.
    "campaign_id", "shard_id", "pbs_job_id", "pbs_array_index",
    "pbs_queue", "pbs_node",
)

const _REQUIRED_NONEMPTY_FIELDS = (
    "catalog_name", "catalog_version", "source",
    "execution_mode", "requested_engine", "executed_engine",
    "scaling", "layout", "campaign_id", "shard_id",
    "shard_index", "shard_count", "input_fingerprint",
    "external_checksum", "project_sha256", "manifest_sha256",
    "benchmark_driver_sha256", "solver_source_sha256",
    "catalog_source_sha256", "harness_source_sha256",
    "schema_source_sha256", "contract_fingerprint",
)

const _REQUIRED_SHA256_FIELDS = (
    "project_sha256", "manifest_sha256", "benchmark_driver_sha256",
    "solver_source_sha256", "catalog_source_sha256",
    "harness_source_sha256", "schema_source_sha256",
    "contract_fingerprint",
)

function _canonical_shard(row, field)
    value = _parsed_integer(row, field)
    value !== nothing && value > 0 || return nothing
    return value
end

function _schema_status(version)
    version === _CURRENT_RESULT_SCHEMA_VERSION && return :current
    (version === nothing || version === missing) && return :legacy
    version isa Integer && version < _CURRENT_RESULT_SCHEMA_VERSION && return :legacy
    return :unsupported
end

function _field_presence_failures(row, fields, prefix)
    failures = String[]
    for field in fields
        haskey(row, field) || push!(failures, "$(prefix)$(field)_missing")
    end
    return failures
end

"""Parse an integer counter, accepting integer-valued decimal strings."""
function _parsed_integer(row, field)
    value = get(row, field, "")
    (value === missing || value === nothing) && return nothing
    if value isa Integer
        return BigInt(value)
    elseif value isa Real
        isfinite(value) && isinteger(value) || return nothing
        return try
            BigInt(value)
        catch
            nothing
        end
    end
    text = strip(string(value))
    isempty(text) && return nothing
    try
        return BigInt(text)
    catch
        # TOML fixtures produced by older runners occasionally contain 8.0
        # for a counter.  Parse that form only when it is finite and exact.
        precision_bits = max(256, 8 * max(length(text), 1) + 128)
        return try
            setprecision(BigFloat, precision_bits) do
                parsed = parse(BigFloat, text)
                isfinite(parsed) && isinteger(parsed) ? BigInt(parsed) : nothing
            end
        catch
            nothing
        end
    end
end

function _ratio(candidate, baseline, field)
    numerator = _parsed_float(candidate, field)
    denominator = _parsed_float(baseline, field)
    if numerator === nothing || denominator === nothing ||
       numerator <= 0 || denominator <= 0
        return missing
    end
    return numerator / denominator
end

function _integer_delta(candidate, baseline, field; require_positive=false)
    after = _parsed_integer(candidate, field)
    before = _parsed_integer(baseline, field)
    if after === nothing || before === nothing ||
       (require_positive && (after <= 0 || before <= 0))
        return missing
    end
    delta = after - before
    return typemin(Int) <= delta <= typemax(Int) ? Int(delta) : delta
end

function _decimal_precision_bits(texts)
    # The mantissa length determines the significant decimal digits needed for
    # subtraction.  The guard bits cover cancellation and conversion to a
    # printable decimal.  A local setprecision scope below leaves the caller's
    # BigFloat precision unchanged.
    digits = maximum((count(isdigit, text) for text in texts); init=0)
    return max(256, ceil(Int, digits * log2(10.0)) + 128)
end

function _objective_delta(candidate, baseline)
    after = _number_text(candidate, "objective")
    before = _number_text(baseline, "objective")
    (after === nothing || before === nothing) && return missing
    precision_bits = _decimal_precision_bits((after, before))
    return try
        setprecision(BigFloat, precision_bits) do
            a = parse(BigFloat, after)
            b = parse(BigFloat, before)
            isfinite(a) && isfinite(b) ? a - b : missing
        end
    catch
        missing
    end
end

function _comparison_evidence(before, after, status_match,
                              semantic_pass_match, certificate_match,
                              sample_parity_match, before_samples_valid,
                              after_samples_valid, schema_status,
                              solver_source_hash_valid,
                              pairing_failures=String[],
                              audit_failures=String[])
    evidence = String[]
    status_match || push!(evidence, "status_mismatch")
    semantic_pass_match || push!(evidence, "semantic_pass_mismatch")
    certificate_match || push!(evidence, "certificate_mismatch")
    before_semantic = get(before, "semantic_pass", missing)
    after_semantic = get(after, "semantic_pass", missing)
    before_semantic === true || push!(evidence, "baseline_semantic_not_passed")
    after_semantic === true || push!(evidence, "candidate_semantic_not_passed")
    build_mode = get(before, "execution_mode", "") == "build" &&
                 get(after, "execution_mode", "") == "build"
    before_certificate = get(before, "certificate_valid", missing)
    after_certificate = get(after, "certificate_valid", missing)
    if !build_mode
        before_certificate === true ||
            push!(evidence, "baseline_certificate_not_valid")
        after_certificate === true ||
            push!(evidence, "candidate_certificate_not_valid")
    else
        get(before, "catalog_validation_pass", missing) === true ||
            push!(evidence, "baseline_catalog_validation_not_passed")
        get(after, "catalog_validation_pass", missing) === true ||
            push!(evidence, "candidate_catalog_validation_not_passed")
    end
    sample_parity_match === false && push!(evidence, "sample_parity_mismatch")
    before_samples_valid || push!(evidence, "baseline_samples_not_valid")
    after_samples_valid || push!(evidence, "candidate_samples_not_valid")
    if schema_status === :legacy
        push!(evidence, "legacy_schema_version")
    elseif schema_status === :unsupported
        push!(evidence, "unsupported_schema_version")
    end
    solver_source_hash_valid ||
        push!(evidence, "solver_source_sha256_invalid")
    append!(evidence, pairing_failures)
    append!(evidence, audit_failures)
    isempty(evidence) ? "" : join(evidence, ",")
end

function compare_result_files(
    baseline_path::AbstractString,
    candidate_path::AbstractString;
    output::Union{Nothing,AbstractString}=nothing,
    allow_dirty::Bool=false,
)
    baseline_document = TOML.parsefile(baseline_path)
    candidate_document = TOML.parsefile(candidate_path)
    get(baseline_document, "schema_version", nothing) ==
        get(candidate_document, "schema_version", nothing) ||
        throw(ArgumentError("result schema versions differ"))
    document_schema_version = get(baseline_document, "schema_version", nothing)
    schema_status = _schema_status(document_schema_version)
    document_schema_v8 = schema_status === :current
    baseline = get(baseline_document, "result", nothing)
    candidate = get(candidate_document, "result", nothing)
    baseline isa AbstractVector && !isempty(baseline) || throw(ArgumentError(
        "baseline result document is empty or missing",
    ))
    candidate isa AbstractVector && !isempty(candidate) || throw(ArgumentError(
        "candidate result document is empty or missing",
    ))
    baseline_by_key = _index_unique_results(baseline, "baseline")
    candidate_by_key = _index_unique_results(candidate, "candidate")
    keys(baseline_by_key) == keys(candidate_by_key) || throw(ArgumentError(
        "baseline and candidate selections differ",
    ))
    rows = NamedTuple[]
    for key in sort!(collect(keys(baseline_by_key)))
        before = baseline_by_key[key]
        after = candidate_by_key[key]
        before_row_schema = _schema_status(
            get(before, "schema_version", missing),
        )
        after_row_schema = _schema_status(
            get(after, "schema_version", missing),
        )
        row_schema_status = if before_row_schema === :legacy ||
                               after_row_schema === :legacy
            :legacy
        elseif before_row_schema === :current &&
               after_row_schema === :current
            :current
        else
            :unsupported
        end
        schema_status === :current || (row_schema_status = schema_status)
        schema_v8 = document_schema_v8 && row_schema_status === :current
        fingerprint = get(before, "input_fingerprint", "")
        isempty(string(fingerprint)) && throw(ArgumentError(
            "input fingerprint is missing for $(key[1])",
        ))
        fingerprint == get(after, "input_fingerprint", "") ||
            throw(ArgumentError("input fingerprint differs for $(key[1])"))
        for field in ("catalog_name", "catalog_version")
            isempty(strip(string(get(before, field, "")))) && throw(ArgumentError(
                "$field is missing for $(key[1])",
            ))
        end
        for field in (
            "suite", "catalog_name", "catalog_version", "name",
            "family", "problem_type", "source", "purpose", "seed",
            "julia_version", "os", "cpu_name", "julia_threads",
            "blas_threads", "conic_formulation", "precision_bits",
            "reference_status", "reference_objective",
            "reference_absolute_tolerance", "reference_relative_tolerance",
            "external_checksum", "project_sha256", "manifest_sha256",
            "benchmark_driver_sha256", "mfla_commit", "bfla_commit",
            "catalog_source_sha256", "harness_source_sha256",
            "schema_source_sha256", "contract_fingerprint",
            "executed_specialization", "psd_lift_used", "benchmark_scale",
            "input_generation_precision_bits", "original_equalities",
            "source_parameters",
            "objective_interval_lower", "objective_interval_upper",
        )
            isequal(get(before, field, ""), get(after, field, "")) || throw(ArgumentError(
                "$field differs for $(key[1]); timing/semantic comparison is not paired",
            ))
        end
        pairing_failures = String[]
        audit_failures = String[]
        before_route = _validate_route(before; label="baseline")
        after_route = _validate_route(after; label="candidate")
        append!(audit_failures, before_route.failures)
        append!(audit_failures, after_route.failures)
        append!(audit_failures, _timing_failures(before, "baseline"))
        append!(audit_failures, _timing_failures(after, "candidate"))
        if schema_v8
            append!(audit_failures, _field_presence_failures(
                before, _AUDIT_IDENTITY_FIELDS, "baseline_",
            ))
            append!(audit_failures, _field_presence_failures(
                after, _AUDIT_IDENTITY_FIELDS, "candidate_",
            ))
            for field in _REQUIRED_NONEMPTY_FIELDS
                isempty(_text(get(before, field, ""))) && push!(
                    audit_failures, "baseline_$(field)_empty",
                )
                isempty(_text(get(after, field, ""))) && push!(
                    audit_failures, "candidate_$(field)_empty",
                )
            end
            for field in _REQUIRED_SHA256_FIELDS
                _valid_solver_source_sha256(get(before, field, missing)) ||
                    push!(audit_failures, "baseline_$(field)_invalid")
                _valid_solver_source_sha256(get(after, field, missing)) ||
                    push!(audit_failures, "candidate_$(field)_invalid")
            end
            for (label, row) in (("baseline", before), ("candidate", after))
                index = _canonical_shard(row, "shard_index")
                count = _canonical_shard(row, "shard_count")
                (index !== nothing && count !== nothing && index <= count) ||
                    push!(audit_failures, "$(label)_shard_topology_invalid")
                if _text(get(row, "execution_mode", "")) != "build"
                    for field in (
                        "conic_formulation", "planned_formulation",
                        "executed_formulation", "planned_backend",
                        "executed_backend", "planned_provider",
                        "executed_provider", "fallback_reason",
                        "la_fallback_reason",
                    )
                        isempty(_text(get(row, field, ""))) && push!(
                            audit_failures, "$(label)_$(field)_empty",
                        )
                    end
                end
            end
            for field in _COMPARISON_PAIRING_FIELDS
                before_value = get(before, field, missing)
                after_value = get(after, field, missing)
                if field == "execution_mode"
                    before_route.mode == after_route.mode || push!(
                        pairing_failures, "pairing:$(field)",
                    )
                elseif field == "requested_engine"
                    before_route.requested == after_route.requested || push!(
                        pairing_failures, "pairing:$(field)",
                    )
                elseif field == "executed_engine"
                    before_route.executed == after_route.executed || push!(
                        pairing_failures, "pairing:$(field)",
                    )
                else
                    isequal(before_value, after_value) || push!(
                        pairing_failures, "pairing:$(field)",
                    )
                end
            end
            for field in ("shard_index", "shard_count")
                _canonical_shard(before, field) == _canonical_shard(after, field) ||
                    push!(pairing_failures, "pairing:$(field)")
            end
        end
        if !allow_dirty && (get(before, "source_dirty", true) ||
                            get(after, "source_dirty", true))
            throw(ArgumentError(
                "dirty source tree for $(key[1]); pass allow_dirty=true only for local diagnostics",
            ))
        end
        status_match = get(before, "status", "") == get(after, "status", "")
        semantic_pass_match = get(before, "semantic_pass", "") ==
                              get(after, "semantic_pass", "")
        certificate_match = get(before, "certificate_valid", "") ==
                            get(after, "certificate_valid", "")
        build_mode = get(before, "execution_mode", "") == "build" &&
                     get(after, "execution_mode", "") == "build"
        semantic_passed = get(before, "semantic_pass", missing) === true &&
                          get(after, "semantic_pass", missing) === true
        certificates_available =
            get(before, "certificate_valid", missing) === true &&
            get(after, "certificate_valid", missing) === true
        catalog_validation_passed =
            get(before, "catalog_validation_pass", missing) === true &&
            get(after, "catalog_validation_pass", missing) === true
        result_gate_passed = build_mode ? catalog_validation_passed :
                             certificates_available
        before_count = _parsed_integer(before, "sample_count")
        after_count = _parsed_integer(after, "sample_count")
        before_sample_validation = before_count isa Integer && before_count >= 3 ?
            _sample_validation(before, "baseline") : nothing
        after_sample_validation = after_count isa Integer && after_count >= 3 ?
            _sample_validation(after, "candidate") : nothing
        if before_sample_validation !== nothing
            append!(audit_failures, before_sample_validation.failures)
        end
        if after_sample_validation !== nothing
            append!(audit_failures, after_sample_validation.failures)
        end
        sample_parity_match = if before_count === nothing || after_count === nothing
            false
        elseif before_count != after_count
            false
        elseif before_count == 1
            true
        elseif before_sample_validation === nothing ||
               after_sample_validation === nothing ||
               !before_sample_validation.valid ||
               !after_sample_validation.valid
            false
        else
            _sample_semantic_match(
                before_sample_validation,
                after_sample_validation,
                before,
                after,
            )
        end
        before_samples_valid = before_count == 1 ||
            (before_sample_validation !== nothing && before_sample_validation.valid)
        after_samples_valid = after_count == 1 ||
            (after_sample_validation !== nothing && after_sample_validation.valid)
        baseline_solver_source_hash_valid =
            _valid_solver_source_sha256(
                get(before, "solver_source_sha256", missing),
            )
        candidate_solver_source_hash_valid =
            _valid_solver_source_sha256(
                get(after, "solver_source_sha256", missing),
            )
        solver_source_hash_valid = schema_v8 &&
            baseline_solver_source_hash_valid &&
            candidate_solver_source_hash_valid
        evidence = _comparison_evidence(
            before,
            after,
            status_match,
            semantic_pass_match,
            certificate_match,
            sample_parity_match,
            before_samples_valid,
            after_samples_valid,
            schema_status === :current && schema_v8 ? :current : row_schema_status,
            solver_source_hash_valid,
            pairing_failures,
            audit_failures,
        )
        push!(rows, (
            problem_id=key[1],
            arithmetic=key[2],
            provider=key[3],
            baseline_source_commit=get(before, "source_commit", ""),
            candidate_source_commit=get(after, "source_commit", ""),
            baseline_solver_source_sha256=
                get(before, "solver_source_sha256", ""),
            candidate_solver_source_sha256=
                get(after, "solver_source_sha256", ""),
            baseline_solver_source_sha256_valid=baseline_solver_source_hash_valid,
            candidate_solver_source_sha256_valid=candidate_solver_source_hash_valid,
            solver_source_sha256_valid=solver_source_hash_valid,
            baseline_catalog_source_sha256=get(before, "catalog_source_sha256", ""),
            candidate_catalog_source_sha256=get(after, "catalog_source_sha256", ""),
            baseline_harness_source_sha256=get(before, "harness_source_sha256", ""),
            candidate_harness_source_sha256=get(after, "harness_source_sha256", ""),
            baseline_schema_source_sha256=get(before, "schema_source_sha256", ""),
            candidate_schema_source_sha256=get(after, "schema_source_sha256", ""),
            baseline_contract_fingerprint=get(before, "contract_fingerprint", ""),
            candidate_contract_fingerprint=get(after, "contract_fingerprint", ""),
            baseline_status=get(before, "status", ""),
            candidate_status=get(after, "status", ""),
            status_match,
            baseline_execution_mode=get(before, "execution_mode", ""),
            candidate_execution_mode=get(after, "execution_mode", ""),
            execution_mode_match=before_route.mode == after_route.mode,
            baseline_requested_engine=get(before, "requested_engine", ""),
            candidate_requested_engine=get(after, "requested_engine", ""),
            requested_engine_match=before_route.requested == after_route.requested,
            baseline_executed_engine=get(before, "executed_engine", ""),
            candidate_executed_engine=get(after, "executed_engine", ""),
            executed_engine_match=before_route.executed == after_route.executed,
            baseline_route_valid=before_route.valid,
            candidate_route_valid=after_route.valid,
            route_valid=before_route.valid && after_route.valid,
            baseline_scaling=get(before, "scaling", ""),
            candidate_scaling=get(after, "scaling", ""),
            scaling_match=get(before, "scaling", "") ==
                          get(after, "scaling", ""),
            baseline_layout=get(before, "layout", ""),
            candidate_layout=get(after, "layout", ""),
            layout_match=get(before, "layout", "") == get(after, "layout", ""),
            baseline_shard_index=_canonical_shard(before, "shard_index"),
            candidate_shard_index=_canonical_shard(after, "shard_index"),
            shard_index_match=_canonical_shard(before, "shard_index") ==
                              _canonical_shard(after, "shard_index"),
            baseline_shard_count=_canonical_shard(before, "shard_count"),
            candidate_shard_count=_canonical_shard(after, "shard_count"),
            shard_count_match=_canonical_shard(before, "shard_count") ==
                              _canonical_shard(after, "shard_count"),
            shard_topology_match=begin
                before_index = _canonical_shard(before, "shard_index")
                after_index = _canonical_shard(after, "shard_index")
                before_shards = _canonical_shard(before, "shard_count")
                after_shards = _canonical_shard(after, "shard_count")
                before_index !== nothing && after_index !== nothing &&
                    before_shards !== nothing && after_shards !== nothing &&
                    before_index <= before_shards && after_index <= after_shards &&
                    before_index == after_index && before_shards == after_shards
            end,
            baseline_campaign_id=get(before, "campaign_id", ""),
            candidate_campaign_id=get(after, "campaign_id", ""),
            campaign_id_recorded=haskey(before, "campaign_id") &&
                                 haskey(after, "campaign_id"),
            baseline_shard_id=get(before, "shard_id", ""),
            candidate_shard_id=get(after, "shard_id", ""),
            shard_id_recorded=haskey(before, "shard_id") &&
                              haskey(after, "shard_id"),
            baseline_pbs_job_id=get(before, "pbs_job_id", ""),
            candidate_pbs_job_id=get(after, "pbs_job_id", ""),
            pbs_job_id_recorded=haskey(before, "pbs_job_id") &&
                                haskey(after, "pbs_job_id"),
            baseline_pbs_array_index=get(before, "pbs_array_index", ""),
            candidate_pbs_array_index=get(after, "pbs_array_index", ""),
            baseline_pbs_queue=get(before, "pbs_queue", ""),
            candidate_pbs_queue=get(after, "pbs_queue", ""),
            baseline_pbs_node=get(before, "pbs_node", ""),
            candidate_pbs_node=get(after, "pbs_node", ""),
            audit_identity_recorded=isempty(audit_failures),
            baseline_semantic_pass=get(before, "semantic_pass", ""),
            candidate_semantic_pass=get(after, "semantic_pass", ""),
            semantic_pass_match,
            samples_parity_match=sample_parity_match,
            conic_formulation_match=get(before, "conic_formulation", "") ==
                                    get(after, "conic_formulation", ""),
            planned_backend_match=get(before, "planned_backend", "") ==
                                  get(after, "planned_backend", ""),
            executed_backend_match=get(before, "executed_backend", "") ==
                                   get(after, "executed_backend", ""),
            provider_match=get(before, "executed_provider", "") ==
                           get(after, "executed_provider", ""),
            specialization_match=get(before, "executed_specialization", "") ==
                                 get(after, "executed_specialization", ""),
            psd_lift_match=get(before, "psd_lift_used", "") ==
                           get(after, "psd_lift_used", ""),
            fallback_match=get(before, "fallback_reason", "") ==
                           get(after, "fallback_reason", "") &&
                           get(before, "la_fallback_reason", "") ==
                           get(after, "la_fallback_reason", ""),
            certificate_match,
            comparison_valid= status_match && semantic_pass_match &&
                              certificate_match && semantic_passed &&
                              result_gate_passed &&
                              before_route.valid && after_route.valid &&
                              solver_source_hash_valid &&
                              isempty(_timing_failures(before, "baseline")) &&
                              isempty(_timing_failures(after, "candidate")) &&
                              sample_parity_match !== false &&
                              before_samples_valid && after_samples_valid &&
                              isempty(pairing_failures) &&
                              isempty(audit_failures) && schema_v8,
            comparison_evidence=evidence,
            objective_delta=_objective_delta(after, before),
            iteration_delta=_integer_delta(after, before, "iterations"),
            total_seconds_ratio=_ratio(after, before, "total_seconds"),
            setup_seconds_ratio=_ratio(after, before, "setup_seconds"),
            frontend_seconds_ratio=_ratio(after, before, "frontend_seconds"),
            preprocess_seconds_ratio=_ratio(after, before, "preprocess_seconds"),
            presolve_seconds_ratio=_ratio(after, before, "presolve_seconds"),
            core_seconds_ratio=_ratio(after, before, "core_seconds"),
            certification_seconds_ratio=_ratio(
                after, before, "certification_seconds",
            ),
            sample_median_seconds_ratio=begin
                if before_count isa Integer && after_count isa Integer &&
                   before_count >= 3 && after_count >= 3
                    _ratio(after, before, "sample_median_seconds")
                else
                    missing
                end
            end,
            factor_seconds_ratio=_ratio(after, before, "factor_seconds"),
            allocated_bytes_delta=_integer_delta(
                after, before, "allocated_bytes"; require_positive=true,
            ),
            allocated_bytes_ratio=_ratio(after, before, "allocated_bytes"),
            process_peak_rss_bytes_delta=_integer_delta(
                after, before, "process_peak_rss_bytes"; require_positive=true,
            ),
            process_peak_rss_bytes_ratio=_ratio(
                after, before, "process_peak_rss_bytes",
            ),
            workspace_bytes_delta=_integer_delta(
                after, before, "workspace_bytes"; require_positive=true,
            ),
            workspace_bytes_ratio=_ratio(after, before, "workspace_bytes"),
        ))
    end
    if output !== nothing
        fields = propertynames(first(rows))
        open(output, "w") do io
            println(io, join(string.(fields), '\t'))
            for row in rows
                println(io, join((_cell(getproperty(row, field)) for field in fields), '\t'))
            end
        end
    end
    return rows
end
