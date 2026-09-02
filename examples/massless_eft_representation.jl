#!/usr/bin/env julia
# Small build/parity probe; it intentionally does not solve or generate N14.
using SDPX
include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "physics", "massless_eft", "MasslessEFT.jl"))
using .MasslessEFT

representation = length(ARGS) == 1 ? Symbol(ARGS[1]) : :soc
representation in (:soc, :sdp) || error("choose representation soc or sdp")
artifact = build_massless_eft(:smoke)
proof = prove_representation_parity(artifact)
proof.valid || error("representation parity certificate failed")
model = representation === :soc ? build_model(artifact; objective=:none) : build_sdp_model(artifact; objective=:none)
compiled = SDPX.compile_product_cone_model(model)
compiled === nothing && error("model compilation returned no model")
println("status=build_only")
println("representation=$(representation)")
println("certificate_valid=$(proof.valid)")
println("determinant_parity=$(proof.determinant_parity)")
println("constraint_blocks=$(length(model.constraint_blocks))")
