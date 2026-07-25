#=====================================================================
    JLD2-backed checkpoints (§5.5) — an optional, more portable
    alternative to the core Serialization-based checkpoint (which is
    always available with no extra dependency, but ties the file to
    the writing Julia version). Same atomic tmp+rename discipline.
=====================================================================#
module SDPXJLD2Ext

using SDPX
using JLD2

function SDPX.save_checkpoint_jld2(path::AbstractString, ::Type{T}, x, X, y, Y, μ, iter, restarts, dims) where {T}
    isempty(path) && return nothing
    cp = SDPX.Checkpoint{T}(SDPX.CHECKPOINT_FORMAT_VERSION, x, X, y, Y, μ, iter, restarts, dims)
    tmp = path * ".tmp"
    JLD2.jldsave(tmp; checkpoint=cp)
    mv(tmp, path; force=true)
    return nothing
end

function SDPX.load_checkpoint_jld2(path::AbstractString, ::Type{T}) where {T}
    cp = JLD2.load(path, "checkpoint")
    cp isa SDPX.Checkpoint{T} || throw(ArgumentError("checkpoint at $path is not a Checkpoint{$T} (got $(typeof(cp)))"))
    cp.format_version == SDPX.CHECKPOINT_FORMAT_VERSION ||
        throw(ArgumentError("checkpoint format version $(cp.format_version) unsupported"))
    return cp
end

function SDPX.save_spectrum_jld2(
    path::AbstractString,
    spectrum::SDPX.SpectrumResult,
)
    # Store an explicit, versioned payload. SpectrumResult is an AbstractVector
    # for user convenience, and JLD2 otherwise serializes it as only its array
    # elements, silently dropping solve-wide metadata.
    payload = (
        format_version=1,
        metadata=spectrum.metadata,
        records=spectrum.records,
    )
    JLD2.jldsave(path; spectrum=payload)
    return path
end

end
