#!/usr/bin/env julia
using Downloads
using SHA
using TOML
using Dates

const ROOT = normpath(joinpath(@__DIR__, ".."))
const MANIFEST = TOML.parsefile(joinpath(ROOT, "manifests", "downloads.toml"))

function parse_args(args)
    suite = "all"
    tier = "core"
    limit = typemax(Int)
    for a in args
        startswith(a, "--suite=") && (suite = split(a, "="; limit=2)[2])
        startswith(a, "--tier=") && (tier = split(a, "="; limit=2)[2])
        startswith(a, "--limit=") && (limit = parse(Int, split(a, "="; limit=2)[2]))
    end
    return suite, tier, limit
end

sha256_file(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function eligible(table, suite, tier)
    suite == "all" || table == suite || return false
    return true
end

function main(args)
    suite, tier, limit = parse_args(args)
    targets = NamedTuple[]
    tables = suite == "all" ? ("sdp", "socp", "lp_netlib") : (suite,)
    for table in tables
        haskey(MANIFEST, table) || continue
        for item in MANIFEST[table]
            item_tier = get(item, "tier", "core")
            tier == "full" || item_tier == "core" || continue
            isempty(get(item, "url", "")) && continue
            push!(targets, (suite=table, item=item))
        end
    end
    targets = first(targets, min(length(targets), limit))
    isempty(targets) && error("No downloadable entries matched suite=$suite tier=$tier")

    lock = Dict{String,Any}(
        "generated_at_utc" => string(Dates.now(Dates.UTC)),
        "entries" => Any[],
    )
    for target in targets
        item = target.item
        destdir = joinpath(ROOT, "data", "external", target.suite)
        mkpath(destdir)
        filename = get(item, "filename", item["name"])
        dest = joinpath(destdir, filename)
        if !isfile(dest)
            @info "Downloading" name=item["name"] url=item["url"] dest
            Downloads.download(item["url"], dest)
        else
            @info "Using cached file" dest
        end
        digest = sha256_file(dest)
        push!(lock["entries"], Dict(
            "suite" => target.suite,
            "name" => item["name"],
            "url" => item["url"],
            "path" => relpath(dest, ROOT),
            "bytes" => filesize(dest),
            "sha256" => digest,
        ))
    end
    open(joinpath(ROOT, "data", "download_lock.toml"), "w") do io
        TOML.print(io, lock)
    end
    println("Pinned $(length(lock["entries"])) downloads in data/download_lock.toml")
end

main(ARGS)
