function _result_key(row)
    return (
        string(get(row, "problem_id", "")),
        string(get(row, "arithmetic", "")),
        string(get(row, "requested_provider", "")),
    )
end

"""Return a non-empty textual representation of a scalar result field."""
function _number_text(row, field)
    value = get(row, field, "")
    (value === missing || value === nothing) && return nothing
    text = strip(string(value))
    isempty(text) && return nothing
    return text
end

const _SHA256_HEX64_RE = r"^[0-9a-fA-F]{64}$"

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
                              after_samples_valid, schema_v6,
                              solver_source_hash_valid)
    evidence = String[]
    status_match || push!(evidence, "status_mismatch")
    semantic_pass_match || push!(evidence, "semantic_pass_mismatch")
    certificate_match || push!(evidence, "certificate_mismatch")
    before_semantic = get(before, "semantic_pass", missing)
    after_semantic = get(after, "semantic_pass", missing)
    before_semantic === true || push!(evidence, "baseline_semantic_not_passed")
    after_semantic === true || push!(evidence, "candidate_semantic_not_passed")
    before_certificate = get(before, "certificate_valid", missing)
    after_certificate = get(after, "certificate_valid", missing)
    before_certificate === true ||
        push!(evidence, "baseline_certificate_not_valid")
    after_certificate === true ||
        push!(evidence, "candidate_certificate_not_valid")
    sample_parity_match === false && push!(evidence, "sample_parity_mismatch")
    before_samples_valid || push!(evidence, "baseline_samples_not_valid")
    after_samples_valid || push!(evidence, "candidate_samples_not_valid")
    schema_v6 || push!(evidence, "legacy_schema_version")
    solver_source_hash_valid ||
        push!(evidence, "solver_source_sha256_invalid")
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
    document_schema_v6 =
        get(baseline_document, "schema_version", nothing) === 6
    baseline = baseline_document["result"]
    candidate = candidate_document["result"]
    baseline_by_key = Dict(_result_key(row) => row for row in baseline)
    candidate_by_key = Dict(_result_key(row) => row for row in candidate)
    keys(baseline_by_key) == keys(candidate_by_key) || throw(ArgumentError(
        "baseline and candidate selections differ",
    ))
    rows = NamedTuple[]
    for key in sort!(collect(keys(baseline_by_key)))
        before = baseline_by_key[key]
        after = candidate_by_key[key]
        schema_v6 = document_schema_v6 &&
            get(before, "schema_version", missing) === 6 &&
            get(after, "schema_version", missing) === 6
        fingerprint = get(before, "input_fingerprint", "")
        isempty(string(fingerprint)) && throw(ArgumentError(
            "input fingerprint is missing for $(key[1])",
        ))
        fingerprint == get(after, "input_fingerprint", "") ||
            throw(ArgumentError("input fingerprint differs for $(key[1])"))
        for field in (
            "suite", "julia_version", "os", "cpu_name", "julia_threads",
            "blas_threads", "conic_formulation", "precision_bits",
            "reference_status", "reference_objective",
            "reference_absolute_tolerance", "reference_relative_tolerance",
            "external_checksum", "project_sha256", "manifest_sha256",
            "benchmark_driver_sha256", "mfla_commit", "bfla_commit",
            "executed_specialization", "psd_lift_used", "benchmark_scale",
            "input_generation_precision_bits", "original_equalities",
            "source_parameters",
            "objective_interval_lower", "objective_interval_upper",
        )
            get(before, field, "") == get(after, field, "") || throw(ArgumentError(
                "$field differs for $(key[1]); timing/semantic comparison is not paired",
            ))
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
        semantic_passed = get(before, "semantic_pass", missing) === true &&
                          get(after, "semantic_pass", missing) === true
        certificates_available =
            get(before, "certificate_valid", missing) === true &&
            get(after, "certificate_valid", missing) === true
        before_count = _parsed_integer(before, "sample_count")
        after_count = _parsed_integer(after, "sample_count")
        sample_fields = (
            "sample_semantic_pass",
            "sample_status",
            "sample_iterations",
            "sample_objective",
            "sample_certificate_valid",
            "sample_route",
            "sample_semantic_parity",
            "sample_parity_failures",
        )
        sample_parity_match = if before_count === nothing || after_count === nothing
            missing
        elseif before_count != after_count
            false
        else
            all(isequal(
                    get(before, field, missing),
                    get(after, field, missing),
                )
                for field in sample_fields)
        end
        before_samples_valid = before_count !== nothing &&
            (before_count < 3 ||
             get(before, "sample_semantic_parity", missing) === true)
        after_samples_valid = after_count !== nothing &&
            (after_count < 3 ||
             get(after, "sample_semantic_parity", missing) === true)
        baseline_solver_source_hash_valid =
            _valid_solver_source_sha256(
                get(before, "solver_source_sha256", missing),
            )
        candidate_solver_source_hash_valid =
            _valid_solver_source_sha256(
                get(after, "solver_source_sha256", missing),
            )
        solver_source_hash_valid = schema_v6 &&
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
            schema_v6,
            solver_source_hash_valid,
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
            baseline_status=get(before, "status", ""),
            candidate_status=get(after, "status", ""),
            status_match,
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
                              certificates_available &&
                              solver_source_hash_valid &&
                              sample_parity_match !== false &&
                              before_samples_valid && after_samples_valid,
            comparison_evidence=evidence,
            objective_delta=_objective_delta(after, before),
            iteration_delta=_integer_delta(after, before, "iterations"),
            total_seconds_ratio=_ratio(after, before, "total_seconds"),
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
