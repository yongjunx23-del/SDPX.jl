using Downloads

const DEFAULT_CACHE = joinpath(ROOT, "data", "cache")
const SUPPORTED_EXTERNAL_LOADERS = Set{Symbol}()

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

"""Explicitly download selected external cases. Never called by tests/solves."""
function prepare_external!(
    ids::AbstractVector{<:AbstractString};
    cache_dir=DEFAULT_CACHE,
    verbose=true,
)
    rows = NamedTuple[]
    for id in ids
        spec = benchmark_spec(id)
        source = spec.external
        source === nothing && begin
            push!(rows, (id=spec.id, status=:generated, path=nothing,
                         checksum=nothing))
            continue
        end
        isempty(source.authoritative_url) && begin
            push!(rows, (id=spec.id, status=:no_url, path=nothing,
                         checksum=nothing))
            continue
        end
        path = _cached_path(spec; cache_dir=cache_dir)
        mkpath(dirname(path))
        if !isfile(path)
            verbose && println("download ", spec.id, " <- ", source.authoritative_url)
            Downloads.download(source.authoritative_url, path)
        end
        checksum = _sha256_file(path)
        if source.sha256 !== nothing && checksum != source.sha256
            throw(ArgumentError(
                "checksum mismatch for $(spec.id): expected $(source.sha256), got $checksum",
            ))
        end
        push!(rows, (id=spec.id, status=:cached, path=path,
                     checksum=checksum))
    end
    return rows
end
