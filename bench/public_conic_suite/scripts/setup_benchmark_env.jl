#!/usr/bin/env julia
using Pkg

suite_root = normpath(joinpath(@__DIR__, ".."))
repo_root = normpath(joinpath(suite_root, "..", ".."))
Pkg.activate(suite_root)

if isfile(joinpath(repo_root, "Project.toml")) &&
   occursin("name = \"SDPX\"", read(joinpath(repo_root, "Project.toml"), String))
    @info "Developing local SDPX" repo_root
    Pkg.develop(path=repo_root)
else
    @warn "SDPX repository root was not detected two levels above the suite. Move this folder under SDPX.jl/bench/ or run Pkg.develop(path=\"/path/to/SDPX.jl\") manually."
end

for pkg in ("JuMP", "MultiFloats")
    try
        Pkg.add(pkg)
    catch err
        @warn "Could not add benchmark dependency" pkg exception=(err, catch_backtrace())
    end
end
Pkg.instantiate()
