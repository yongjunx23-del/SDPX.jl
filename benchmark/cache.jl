using Downloads

const DEFAULT_CACHE = joinpath(ROOT, "data", "cache")
const SUPPORTED_EXTERNAL_LOADERS = Set{Symbol}((
    :csdr_fixed_trace_reduced_v1,
    :external_netlib_compressed_mps,
    :external_sdpa_sparse_gzip,
    :external_sdppack_compact_gzip,
    :external_cbf_gzip,
))

function _cached_path(spec::BenchmarkSpec; cache_dir=DEFAULT_CACHE)
    spec.external === nothing && return nothing
    return joinpath(cache_dir, string(spec.source), spec.external.filename)
end

function _sha256_file(path)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function external_cache_status(spec::BenchmarkSpec; cache_dir=DEFAULT_CACHE)
    spec.external === nothing && return (
        available=true,
        loadable=true,
        reason=:generated,
        path=nothing,
        checksum=nothing,
    )
    path = _cached_path(spec; cache_dir=cache_dir)
    isfile(path) || return (
        available=false,
        loadable=false,
        reason=:not_cached,
        path=path,
        checksum=nothing,
    )
    checksum = _sha256_file(path)
    expected = spec.external.sha256
    expected !== nothing && checksum != expected && return (
        available=false,
        loadable=false,
        reason=:checksum_mismatch,
        path=path,
        checksum=checksum,
    )
    return (
        available=true,
        loadable=spec.loader in SUPPORTED_EXTERNAL_LOADERS,
        reason=spec.loader in SUPPORTED_EXTERNAL_LOADERS ? :ready : :loader_unavailable,
        path=path,
        checksum=checksum,
    )
end

"""Return a unique same-directory download path on every supported Julia."""
function _cache_part_path(path::AbstractString)
    # Julia 1.10 supports `cleanup` but not the later `suffix` keyword on
    # `tempname`. Appending the marker preserves the observable `.part`
    # contract while keeping the final rename on the destination filesystem.
    return tempname(dirname(path); cleanup=false) * ".part"
end

"""
    _prepare_external_spec!(spec; cache_dir, verbose, downloader)

Prepare one external benchmark artifact.  Downloads are written to a unique
temporary path in the target directory, checked before publication, and then
atomically renamed over the canonical cache path.  Keeping the temporary file
in the destination directory makes the final `mv` a same-filesystem rename;
the `finally` block removes it on checksum failure, downloader errors, or
interrupts without touching an existing cache entry.
"""
function _prepare_external_spec!(
    spec::BenchmarkSpec;
    cache_dir=DEFAULT_CACHE,
    verbose=true,
    downloader=Downloads.download,
)
    source = spec.external
    source === nothing && return (
        id=spec.id,
        status=:generated,
        path=nothing,
        checksum=nothing,
    )
    isempty(source.authoritative_url) && return (
        id=spec.id,
        status=:no_url,
        path=nothing,
        checksum=nothing,
    )

    path = _cached_path(spec; cache_dir=cache_dir)
    mkpath(dirname(path))
    expected = source.sha256

    # A present artifact with no declared digest retains the historical
    # semantics: it is trusted and its actual digest is reported.  A declared
    # digest mismatch is explicitly repairable by --prepare.
    if isfile(path)
        checksum = _sha256_file(path)
        if expected === nothing || checksum == expected
            return (id=spec.id, status=:cached, path=path, checksum=checksum)
        end
    end

    verbose && println("download ", spec.id, " <- ", source.authoritative_url)
    part_path = _cache_part_path(path)
    try
        downloader(source.authoritative_url, part_path)
        isfile(part_path) || throw(ArgumentError(
            "downloader did not create a cache artifact for $(spec.id)",
        ))
        checksum = _sha256_file(part_path)
        if expected !== nothing && checksum != expected
            throw(ArgumentError(
                "checksum mismatch for $(spec.id): expected $expected, got $checksum",
            ))
        end
        # `path` and `part_path` share a directory, so `mv` is an atomic
        # replacement on the supported filesystems.  The old file remains
        # untouched if download or validation fails above.
        mv(part_path, path; force=true)
        part_path = nothing
    finally
        part_path === nothing || rm(part_path; force=true)
    end
    return (id=spec.id, status=:cached, path=path, checksum=checksum)
end

"""Explicitly download selected external cases. Never called by tests/solves."""
function prepare_external!(
    ids::AbstractVector{<:AbstractString};
    cache_dir=DEFAULT_CACHE,
    verbose=true,
    downloader=Downloads.download,
)
    rows = NamedTuple[]
    for id in ids
        spec = benchmark_spec(id)
        push!(rows, _prepare_external_spec!(
            spec;
            cache_dir=cache_dir,
            verbose=verbose,
            downloader=downloader,
        ))
    end
    return rows
end
