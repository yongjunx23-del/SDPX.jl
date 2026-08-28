#=====================================================================
    SDP checkpointing (§5.5). Serialization-based default I/O with an
    atomic tmp+rename discipline; the JLD2-backed variant lives in the
    `SDPXJLD2Ext` package extension and binds to these stubs.
=====================================================================#

function save_checkpoint(path::AbstractString, ::Type{T}, x, X, y, Y, μ, iter, restarts, dims) where {T}
    isempty(path) && return
    cp = Checkpoint{T}(CHECKPOINT_FORMAT_VERSION, x, X, y, Y, μ, iter, restarts, dims)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        Serialization.serialize(io, cp)
    end
    mv(tmp, path; force=true)
    return nothing
end

"""
    save_checkpoint_jld2(path, T, x, X, y, Y, μ, iter, restarts, dims)
    load_checkpoint_jld2(path, T)

JLD2-backed checkpoint I/O (§5.5) — only available once the `JLD2`
package extension is loaded (`using JLD2`); more portable across
Julia versions than the default Serialization-based
[`save_checkpoint`](@ref)/[`load_checkpoint`](@ref).
"""
function save_checkpoint_jld2 end
function load_checkpoint_jld2 end

function load_checkpoint(path::AbstractString, ::Type{T}) where {T}
    cp = open(path, "r") do io
        Serialization.deserialize(io)
    end
    cp isa Checkpoint{T} || throw(ArgumentError("checkpoint at $path is not a Checkpoint{$T} (got $(typeof(cp)))"))
    cp.format_version == CHECKPOINT_FORMAT_VERSION ||
        throw(ArgumentError("checkpoint format version $(cp.format_version) unsupported (expected $CHECKPOINT_FORMAT_VERSION)"))
    return cp
end

