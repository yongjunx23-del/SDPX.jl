"""Source files that define one external benchmark loader contract.

The runner folds these files into the problem fingerprint.  A parser or sign
convention change must therefore invalidate old input identities even when the
downloaded bytes are unchanged.
"""
function external_loader_source_files(loader::Symbol)
    common = joinpath(ROOT, "loaders", "external.jl")
    specific = if loader === :csdr_fixed_trace_reduced_v1
        joinpath(ROOT, "loaders", "csdr_fixed_trace.jl")
    elseif loader in _NETLIB_MPS_LOADERS
        joinpath(ROOT, "loaders", "netlib_mps.jl")
    elseif loader in _SDPA_DIMACS_LOADERS ||
           loader in _SDPPACK_COMPACT_LOADERS
        joinpath(ROOT, "loaders", "sdpa_sparse.jl")
    elseif loader in _CBF_LOADERS
        joinpath(ROOT, "loaders", "cbf.jl")
    else
        throw(ArgumentError(
            "unsupported external benchmark loader $(repr(loader))",
        ))
    end
    return (common, specific)
end

"""Dispatch one checksum-validated external input to its explicit loader."""
function build_external_problem(
    spec::BenchmarkSpec,
    ::Type{T},
    path::AbstractString,
    checksum::AbstractString,
) where {T}
    if spec.loader === :csdr_fixed_trace_reduced_v1
        return _build_csdr_fixed_trace_problem(spec, T, path, checksum)
    elseif spec.loader in _NETLIB_MPS_LOADERS
        return _build_netlib_mps_problem(spec, T, path, checksum)
    elseif spec.loader in _SDPA_DIMACS_LOADERS
        return _build_sdpa_sparse_problem(spec, T, path, checksum)
    elseif spec.loader in _SDPPACK_COMPACT_LOADERS
        return _build_sdppack_compact_problem(spec, T, path, checksum)
    elseif spec.loader in _CBF_LOADERS
        return _build_cbf_problem(spec, T, path, checksum)
    end
    throw(ArgumentError(
        "unsupported external benchmark loader $(repr(spec.loader))",
    ))
end
