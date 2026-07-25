#=====================================================================
    Optional spectrum reconstruction and export
=====================================================================#

"""
    reconstruct_spectrum(result; source=:primal) -> Vector{NamedTuple}

Compute eigenvalues of each primal (`source=:primal`) or dual
(`source=:dual`) cone block. The operation is intentionally separate from
`solve` because eigenvalue decomposition can be substantial post-processing
for large dense PSD blocks.
"""
function reconstruct_spectrum(result::SDPResult; source::Symbol=:primal)
    source in (:primal, :dual) ||
        throw(ArgumentError("source must be :primal or :dual"))
    matrices = source === :primal ? result.X : result.Y
    records = NamedTuple[]
    for (block, matrix) in pairs(matrices)
        values = if size(matrix, 1) == 1
            [matrix[1, 1]]
        elseif eltype(matrix) <: LinearAlgebra.BlasFloat
            eigvals(Symmetric(matrix))
        else
            # Julia's standard library does not provide a generic Hermitian
            # eigensolver for every extended scalar. The Float64 projection is
            # explicit in the output type and is only used on request.
            eigvals(Symmetric(Float64.(matrix)))
        end
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
    return records
end

function _json_scalar(value)
    value isa Symbol && return "\"$(String(value))\""
    value isa AbstractString &&
        return "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
    return string(value)
end

"""Extension hook implemented when JLD2 is loaded."""
function save_spectrum_jld2 end

"""
    export_spectrum(path, result; source=:primal, format=:auto)

Export an optionally reconstructed spectrum as CSV, JSON, or Julia's
Serialization format (`:jls`). JLD2 users can store the returned records
directly; spectrum extraction intentionally does not make JLD2 a hard
dependency.
"""
function export_spectrum(
    path::AbstractString,
    result::SDPResult;
    source::Symbol=:primal,
    format::Symbol=:auto,
)
    selected_format = if format === :auto
        extension = lowercase(splitext(path)[2])
        extension == ".csv" ? :csv :
        extension == ".json" ? :json :
        extension == ".jld2" ? :jld2 :
        extension in (".jls", ".bin") ? :jls :
        throw(ArgumentError("cannot infer spectrum format from extension $extension"))
    else
        format
    end
    records = reconstruct_spectrum(result; source=source)
    if selected_format === :csv
        open(path, "w") do io
            println(io, "source,block,eigenvalue_index,eigenvalue")
            for record in records
                println(
                    io,
                    "$(record.source),$(record.block),$(record.eigenvalue_index),$(record.eigenvalue)",
                )
            end
        end
    elseif selected_format === :json
        open(path, "w") do io
            println(io, "[")
            for (index, record) in pairs(records)
                suffix = index == length(records) ? "" : ","
                println(
                    io,
                    "  {\"source\":$(_json_scalar(record.source))," *
                    "\"block\":$(record.block)," *
                    "\"eigenvalue_index\":$(record.eigenvalue_index)," *
                    "\"eigenvalue\":$(_json_scalar(record.eigenvalue))}$suffix",
                )
            end
            println(io, "]")
        end
    elseif selected_format === :jls
        open(path, "w") do io
            Serialization.serialize(io, records)
        end
    elseif selected_format === :jld2
        save_spectrum_jld2(path, records)
    else
        throw(ArgumentError(
            "format must be :auto, :csv, :json, :jld2, or :jls",
        ))
    end
    return path
end
