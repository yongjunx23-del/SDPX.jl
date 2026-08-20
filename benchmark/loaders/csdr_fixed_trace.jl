using Serialization

function _owned_csdr_payload(payload::NamedTuple, ::Type{T}) where {T}
    required = (
        :schema, :reduced_c, :reduced_B, :reduced_b,
        :coefficient_constant, :coefficient_from_spectrum,
        :coefficient_labels, :objective, :fixed_coefficients,
        :source_model_sha256,
    )
    for field in required
        hasproperty(payload, field) || throw(ArgumentError(
            "CSDR neutral payload is missing field $field",
        ))
    end
    payload.schema === :csdr_fixed_trace_reduced_v1 || throw(ArgumentError(
        "unsupported CSDR payload schema $(repr(payload.schema))",
    ))
    source_model_sha256 = lowercase(String(payload.source_model_sha256))
    occursin(r"^[0-9a-f]{64}$", source_model_sha256) || throw(ArgumentError(
        "CSDR source_model_sha256 must be one lowercase SHA-256 digest",
    ))
    return (
        schema=:csdr_fixed_trace_reduced_v1,
        source_model_sha256,
        reduced_c=SDPX._owned_array_copy(T, payload.reduced_c),
        reduced_B=SDPX._owned_array_copy(T, sparse(payload.reduced_B)),
        reduced_b=SDPX._owned_array_copy(T, payload.reduced_b),
        coefficient_constant=SDPX._owned_array_copy(
            T, payload.coefficient_constant,
        ),
        coefficient_from_spectrum=SDPX._owned_array_copy(
            T, payload.coefficient_from_spectrum,
        ),
        coefficient_labels=String.(payload.coefficient_labels),
        objective=Dict(String(key) => String(value)
                       for (key, value) in pairs(payload.objective)),
        fixed_coefficients=Dict(String(key) => String(value)
                                for (key, value) in pairs(payload.fixed_coefficients)),
    )
end

function _csdr_fixed_trace_problem(payload::NamedTuple, ::Type{T}) where {T}
    variables = length(payload.reduced_c)
    iseven(variables) || throw(ArgumentError(
        "fixed-trace reduced variable count must be even",
    ))
    blocks = variables ÷ 2
    size(payload.reduced_B, 1) == variables || throw(DimensionMismatch(
        "CSDR reduced_B must have one row per reduced variable",
    ))
    length(payload.reduced_b) == size(payload.reduced_B, 2) ||
        throw(DimensionMismatch(
            "CSDR reduced_b length must match reduced_B columns",
        ))
    cones = Vector{SDPX.SOCConstraint{T}}(undef, blocks)
    @inbounds for block in 1:blocks
        r_index = 2block - 1
        q_index = 2block
        affine = sparse(
            [2, 3], [q_index, r_index], T[one(T), one(T)], 3, variables,
        )
        cones[block] = SDPX.SOCConstraint(
            affine, T[one(T), -one(T), zero(T)]; T,
        )
    end
    return SDPX.second_order_program(
        payload.reduced_c,
        cones;
        Aeq=sparse(transpose(payload.reduced_B)),
        beq=payload.reduced_b,
        T,
    )
end

function _csdr_physical_objective(payload::NamedTuple, x)
    T = eltype(x)
    coefficients = payload.coefficient_constant +
                   payload.coefficient_from_spectrum * x
    full_x = vcat(coefficients, x)
    label_index = Dict(
        label => index for (index, label) in enumerate(payload.coefficient_labels)
    )
    value = zero(T)
    for (label, coefficient) in payload.objective
        beta = parse(T, coefficient)
        if haskey(label_index, label)
            value += beta * full_x[label_index[label]]
        else
            haskey(payload.fixed_coefficients, label) || throw(ArgumentError(
                "objective label $label is absent from CSDR coefficients",
            ))
            value += beta * parse(T, payload.fixed_coefficients[label])
        end
    end
    return value
end

function _build_csdr_fixed_trace_problem(
    spec::BenchmarkSpec,
    ::Type{T},
    path::AbstractString,
    checksum::AbstractString,
) where {T}
    spec.loader === :csdr_fixed_trace_reduced_v1 || throw(ArgumentError(
        "unsupported external benchmark loader $(spec.loader)",
    ))
    actual = _sha256_file(path)
    actual == checksum || throw(ArgumentError(
        "external checksum changed between cache validation and load",
    ))
    spec.external.sha256 === nothing || actual == spec.external.sha256 ||
        throw(ArgumentError("CSDR payload checksum does not match registry"))
    raw = open(deserialize, path)
    raw isa NamedTuple || throw(ArgumentError(
        "CSDR neutral payload must be a NamedTuple",
    ))
    payload = _owned_csdr_payload(raw, T)
    expected_source_sha = lowercase(String(
        spec.parameters.source_model_sha256,
    ))
    payload.source_model_sha256 == expected_source_sha || throw(ArgumentError(
        "CSDR source-model checksum does not match registry",
    ))
    problem = _csdr_fixed_trace_problem(payload, T)
    expected_size = spec.size
    hasproperty(expected_size, :variables) &&
        problem.variables != expected_size.variables && throw(ArgumentError(
            "CSDR variable count does not match registry",
        ))
    hasproperty(expected_size, :soc_blocks) &&
        length(problem.cones) != expected_size.soc_blocks && throw(ArgumentError(
            "CSDR cone count does not match registry",
        ))
    hasproperty(expected_size, :equalities) &&
        size(problem.Aeq, 1) != expected_size.equalities && throw(ArgumentError(
            "CSDR equality count does not match registry",
        ))
    parameters = spec.parameters
    return (
        kind=:socp,
        problem,
        expected=nothing,
        physical_objective=x -> _csdr_physical_objective(payload, x),
        objective_interval=parameters.objective_interval,
        solve_settings=parameters.solve_settings,
        required_specialization=:fixed_trace_q3,
        forbid_psd_lift=true,
        require_no_fallback=true,
        maximum_relative_gap="1e-12",
        external_checksum=actual,
        source_model_sha256=payload.source_model_sha256,
        benchmark_scale=parameters.benchmark_scale,
        input_generation_precision_bits=parameters.input_generation_precision_bits,
        original_equalities=parameters.original_equalities,
        source_parameters=parameters.source_parameters,
    )
end
