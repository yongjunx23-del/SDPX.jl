#!/usr/bin/env julia
# Deliberately guarded: this is the only checked-in production entry point.
using SHA
include(joinpath(@__DIR__, "MasslessEFT.jl"))
using .MasslessEFT

if length(ARGS) != 1 || ARGS[1] != "--confirm-production" ||
   get(ENV, "SDPX_MASSLESS_EFT_PRODUCTION", "") != "1"
    println(stderr, "refusing production build; pass --confirm-production and set SDPX_MASSLESS_EFT_PRODUCTION=1")
    exit(2)
end
artifact = build_production_massless_eft(BigFloat; confirm=true)
println("status=build_only")
println("schema_version=$(artifact.schema_version)")
println("external_checksum=$(artifact.fingerprint)")
println("manifest_sha256=$(manifest_sha256())")
