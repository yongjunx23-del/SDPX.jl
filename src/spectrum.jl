#=====================================================================
    Optional spectrum reconstruction and export
=====================================================================#

const _SPECTRUM_PRECISIONS = (:native, :float64)
const _SPECTRUM_NONFINITE_POLICIES = (:null, :string, :error)

"""
    SpectrumResult <: AbstractVector

Spectrum records with solve-wide metadata stored exactly once. The wrapper
behaves as an ordinary vector for iteration, indexing, and broadcasting while
avoiding quadratic metadata growth on models with thousands of cone blocks.
"""
struct SpectrumResult{M,R<:AbstractVector{NamedTuple}} <:
       AbstractVector{NamedTuple}
    metadata::M
    records::R
end

Base.IndexStyle(::Type{<:SpectrumResult}) = IndexLinear()
Base.size(result::SpectrumResult) = size(result.records)
Base.axes(result::SpectrumResult) = axes(result.records)
Base.length(result::SpectrumResult) = length(result.records)
Base.getindex(result::SpectrumResult, index::Int) = result.records[index]
Base.iterate(result::SpectrumResult, state...) =
    iterate(result.records, state...)
Base.similar(
    ::SpectrumResult,
    ::Type{T},
    dimensions::Dims,
) where {T} = Array{T}(undef, dimensions)

function _spectrum_is_certified(result::SDPResult)
    return result.status in (Optimal, FeasibleCert)
end

function _spectrum_warnings(
    result::SDPResult{T},
    precision::Symbol,
    allow_uncertified::Bool,
) where {T}
    warnings = result.diagnostics === nothing ?
               String[] :
               copy(result.diagnostics.warnings)
    if !_spectrum_is_certified(result)
        allow_uncertified || throw(
            ArgumentError(
                "spectrum extraction requires an Optimal or FeasibleCert " *
                "result; got $(result.status). Pass allow_uncertified=true " *
                "to inspect this iterate explicitly.",
            ),
        )
        push!(
            warnings,
            "Spectrum extracted from uncertified solve status $(result.status).",
        )
    end
    if precision === :float64 && T !== Float64
        push!(
            warnings,
            "Spectrum matrices were explicitly converted from $T to Float64; " *
            "reported eigenvalues are not in the result's native arithmetic.",
        )
    end
    return unique!(warnings)
end

function _native_eigenvalues(matrix::AbstractMatrix{T}) where {T}
    try
        return eigvals(Symmetric(matrix))
    catch error
        if error isa MethodError
            throw(
                ArgumentError(
                    "no native symmetric eigensolver is available for $T. " *
                    "Load a compatible generic eigensolver package, or pass " *
                    "precision=:float64 to request an explicit projection.",
                ),
            )
        end
        rethrow()
    end
end

function _spectrum_eigenvalues(
    matrix::AbstractMatrix,
    precision::Symbol,
)
    rows, columns = size(matrix)
    rows == columns ||
        throw(DimensionMismatch("spectrum blocks must be square"))
    if rows == 1
        return precision === :native ?
               [matrix[1, 1]] :
               Float64[Float64(matrix[1, 1])]
    end
    if precision === :float64
        return eigvals(Symmetric(Float64.(matrix)))
    end
    return _native_eigenvalues(matrix)
end

"""
    reconstruct_spectrum(
        result;
        source=:primal,
        precision=:native,
        allow_uncertified=false,
    ) -> SpectrumResult

Compute eigenvalues of each primal (`source=:primal`) or dual
(`source=:dual`) cone block. `precision=:native` never narrows the result
arithmetic. If no native eigensolver exists, the function reports that
limitation and suggests the explicit `precision=:float64` projection.

Only certified successful results are accepted by default. Set
`allow_uncertified=true` to inspect a stopped or failed iterate. Solve-wide
status, residuals, block dimensions, arithmetic information, and warnings are
available in `result.metadata`; iterating `result` yields only compact
per-eigenvalue records. The operation remains separate from `solve`, so
eigendecomposition is not included in solve timings.
"""
function reconstruct_spectrum(
    result::SDPResult{T};
    source::Symbol=:primal,
    precision::Symbol=:native,
    allow_uncertified::Bool=false,
) where {T}
    source in (:primal, :dual) ||
        throw(ArgumentError("source must be :primal or :dual"))
    precision in _SPECTRUM_PRECISIONS ||
        throw(ArgumentError("precision must be :native or :float64"))
    warnings = _spectrum_warnings(result, precision, allow_uncertified)
    matrices = source === :primal ? result.X : result.Y
    block_dimensions = Tuple(size(matrix, 1) for matrix in matrices)
    projected = precision === :float64 && T !== Float64
    certified = _spectrum_is_certified(result)
    records = NamedTuple[]
    for (block, matrix) in pairs(matrices)
        values = _spectrum_eigenvalues(matrix, precision)
        for (index, value) in pairs(values)
            push!(
                records,
                (
                    source=source,
                    block=block,
                    eigenvalue_index=index,
                    eigenvalue=value,
                ),
            )
        end
    end
    eigenvalue_arithmetic = isempty(records) ?
                            (precision === :native ? string(T) : "Float64") :
                            string(typeof(first(records).eigenvalue))
    metadata = (
        source=source,
        block_dimensions=block_dimensions,
        solve_status=string(result.status),
        solve_message=result.message,
        certified=certified,
        primal_objective=result.pObj,
        dual_objective=result.dObj,
        relative_gap=result.gap_rel,
        primal_residual=result.p_res,
        dual_residual=result.d_res,
        result_arithmetic=string(T),
        requested_precision=precision,
        eigenvalue_arithmetic=eigenvalue_arithmetic,
        projected=projected,
        warnings=Tuple(warnings),
    )
    return SpectrumResult(metadata, records)
end

function _validate_nonfinite_policy(policy::Symbol)
    policy in _SPECTRUM_NONFINITE_POLICIES ||
        throw(
            ArgumentError(
                "nonfinite must be :null, :string, or :error",
            ),
        )
    return policy
end

function _write_json_string(io::IO, value::AbstractString)
    print(io, '"')
    for character in value
        if character == '"'
            print(io, "\\\"")
        elseif character == '\\'
            print(io, "\\\\")
        elseif character == '\b'
            print(io, "\\b")
        elseif character == '\f'
            print(io, "\\f")
        elseif character == '\n'
            print(io, "\\n")
        elseif character == '\r'
            print(io, "\\r")
        elseif character == '\t'
            print(io, "\\t")
        elseif Int(character) < 0x20
            code = uppercase(string(Int(character); base=16, pad=4))
            print(io, "\\u", code)
        else
            print(io, character)
        end
    end
    print(io, '"')
    return nothing
end

function _write_nonfinite_json(io::IO, value, policy::Symbol)
    if policy === :null
        print(io, "null")
    elseif policy === :string
        label = isnan(value) ? "NaN" : signbit(value) ? "-Infinity" : "Infinity"
        _write_json_string(io, label)
    else
        throw(DomainError(value, "JSON cannot represent non-finite numbers"))
    end
    return nothing
end

function _write_json(io::IO, value, nonfinite::Symbol)
    if value === nothing || value === missing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa AbstractString
        _write_json_string(io, value)
    elseif value isa Symbol || value isa Enum
        _write_json_string(io, string(value))
    elseif value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        if isfinite(value)
            representation = string(value)
            occursin(
                r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$",
                representation,
            ) || throw(
                ArgumentError(
                    "cannot encode $value ($(typeof(value))) as a JSON number",
                ),
            )
            print(io, representation)
        else
            _write_nonfinite_json(io, value, nonfinite)
        end
    elseif value isa NamedTuple
        print(io, '{')
        for (index, name) in enumerate(keys(value))
            index > 1 && print(io, ',')
            _write_json_string(io, string(name))
            print(io, ':')
            _write_json(io, getfield(value, name), nonfinite)
        end
        print(io, '}')
    elseif value isa AbstractVector || value isa Tuple
        print(io, '[')
        for (index, item) in enumerate(value)
            index > 1 && print(io, ',')
            _write_json(io, item, nonfinite)
        end
        print(io, ']')
    else
        throw(
            ArgumentError(
                "cannot encode $(typeof(value)) in spectrum JSON output",
            ),
        )
    end
    return nothing
end

function _csv_nonfinite(value, policy::Symbol)
    policy === :null && return ""
    policy === :string &&
        return isnan(value) ? "NaN" : signbit(value) ? "-Infinity" : "Infinity"
    throw(DomainError(value, "CSV non-finite policy is :error"))
end

function _csv_scalar(value, nonfinite::Symbol)
    text = if value isa AbstractFloat && !isfinite(value)
        _csv_nonfinite(value, nonfinite)
    elseif value isa Tuple || value isa AbstractVector
        join((_csv_scalar(item, nonfinite) for item in value), "; ")
    else
        string(value)
    end
    if any(character -> character in (',', '"', '\n', '\r'), text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function _write_spectrum_csv(
    io::IO,
    spectrum::SpectrumResult,
    nonfinite::Symbol,
)
    columns = (:source, :block, :eigenvalue_index, :eigenvalue)
    println(io, join(string.(columns), ","))
    print(io, "# metadata=")
    _write_json(io, spectrum.metadata, nonfinite)
    println(io)
    for record in spectrum
        println(
            io,
            join(
                (
                    _csv_scalar(getfield(record, column), nonfinite)
                    for column in columns
                ),
                ",",
            ),
        )
    end
    return nothing
end

function _atomic_replace(writer, path::AbstractString)
    destination = abspath(path)
    directory = dirname(destination)
    temporary, io = mktemp(directory; cleanup=false)
    close(io)
    committed = false
    try
        writer(temporary)
        Base.Filesystem.rename(temporary, destination)
        committed = true
    finally
        !committed && ispath(temporary) && rm(temporary; force=true)
    end
    return path
end

"""Extension hook implemented when JLD2 is loaded."""
function save_spectrum_jld2 end

function _spectrum_format(path::AbstractString, format::Symbol)
    if format === :auto
        extension = lowercase(splitext(path)[2])
        extension == ".csv" && return :csv
        extension == ".json" && return :json
        extension == ".jld2" && return :jld2
        extension in (".jls", ".bin") && return :jls
        throw(
            ArgumentError(
                "cannot infer spectrum format from extension $extension",
            ),
        )
    end
    format in (:csv, :json, :jld2, :jls) ||
        throw(
            ArgumentError(
                "format must be :auto, :csv, :json, :jld2, or :jls",
            ),
        )
    return format
end

"""
    export_spectrum(
        path,
        result;
        source=:primal,
        format=:auto,
        precision=:native,
        allow_uncertified=false,
        nonfinite=:null,
    )

Export an optionally reconstructed spectrum as CSV, standards-compliant JSON,
JLD2, or Julia's Serialization format (`:jls`). Exports include solve status,
objectives, residuals, block dimensions, arithmetic metadata, and warnings.

JSON and CSV encode non-finite values as `null`/an empty cell by default.
Choose `nonfinite=:string` for explicit `NaN`/`Infinity` strings or
`nonfinite=:error` to reject them. Files are written beside the destination
and atomically renamed only after a complete successful write.
"""
function export_spectrum(
    path::AbstractString,
    result::SDPResult;
    source::Symbol=:primal,
    format::Symbol=:auto,
    precision::Symbol=:native,
    allow_uncertified::Bool=false,
    nonfinite::Symbol=:null,
)
    selected_format = _spectrum_format(path, format)
    _validate_nonfinite_policy(nonfinite)
    records = reconstruct_spectrum(
        result;
        source=source,
        precision=precision,
        allow_uncertified=allow_uncertified,
    )
    if selected_format === :jld2 &&
       !applicable(save_spectrum_jld2, path, records)
        throw(
            ArgumentError(
                "JLD2 spectrum export is not loaded. Run `using JLD2` before " *
                "calling export_spectrum(...; format=:jld2).",
            ),
        )
    end
    return _atomic_replace(path) do temporary
        if selected_format === :csv
            open(temporary, "w") do io
                _write_spectrum_csv(io, records, nonfinite)
            end
        elseif selected_format === :json
            open(temporary, "w") do io
                # The one-element compatibility envelope keeps historical JSON
                # consumers that identify spectrum output as an array working,
                # while metadata and records now form one normalized payload.
                payload = (metadata=records.metadata, records=records.records)
                _write_json(io, (payload,), nonfinite)
                println(io)
            end
        elseif selected_format === :jls
            open(temporary, "w") do io
                Serialization.serialize(io, records)
            end
        else
            save_spectrum_jld2(temporary, records)
        end
    end
end
