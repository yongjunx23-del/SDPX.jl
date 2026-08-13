function _result_key(row)
    return (
        string(get(row, "problem_id", "")),
        string(get(row, "arithmetic", "")),
        string(get(row, "requested_provider", "")),
    )
end

function _parsed_number(row, field)
    value = get(row, field, "")
    value isa Real && return Float64(value)
    isempty(string(value)) && return nothing
    return try
        parse(Float64, string(value))
    catch
        nothing
    end
end

function _ratio(candidate, baseline, field)
    numerator = _parsed_number(candidate, field)
    denominator = _parsed_number(baseline, field)
    if numerator === nothing || denominator === nothing || denominator <= 0
        return missing
    end
    return numerator / denominator
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
            "external_checksum",
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
        push!(rows, (
            problem_id=key[1],
            arithmetic=key[2],
            provider=key[3],
            baseline_source_commit=get(before, "source_commit", ""),
            candidate_source_commit=get(after, "source_commit", ""),
            baseline_status=get(before, "status", ""),
            candidate_status=get(after, "status", ""),
            status_match=get(before, "status", "") == get(after, "status", ""),
            baseline_semantic_pass=get(before, "semantic_pass", ""),
            candidate_semantic_pass=get(after, "semantic_pass", ""),
            semantic_pass_match=get(before, "semantic_pass", "") ==
                                get(after, "semantic_pass", ""),
            conic_formulation_match=get(before, "conic_formulation", "") ==
                                    get(after, "conic_formulation", ""),
            planned_backend_match=get(before, "planned_backend", "") ==
                                  get(after, "planned_backend", ""),
            executed_backend_match=get(before, "executed_backend", "") ==
                                   get(after, "executed_backend", ""),
            provider_match=get(before, "executed_provider", "") ==
                           get(after, "executed_provider", ""),
            fallback_match=get(before, "fallback_reason", "") ==
                           get(after, "fallback_reason", "") &&
                           get(before, "la_fallback_reason", "") ==
                           get(after, "la_fallback_reason", ""),
            certificate_match=get(before, "certificate_valid", "") ==
                              get(after, "certificate_valid", ""),
            objective_delta=begin
                a = _parsed_number(after, "objective")
                b = _parsed_number(before, "objective")
                a === nothing || b === nothing ? missing : a - b
            end,
            iteration_delta=begin
                a = _parsed_number(after, "iterations")
                b = _parsed_number(before, "iterations")
                a === nothing || b === nothing ? missing : Int(a - b)
            end,
            total_seconds_ratio=_ratio(after, before, "total_seconds"),
            factor_seconds_ratio=_ratio(after, before, "factor_seconds"),
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
