# Build an incremental SDPX performance sysimage for the current Julia build.
# Usage:
#   julia --project=<env-with-PackageCompiler> \
#       scripts/build_performance_sysimage.jl [output.so]
# Run solves with: julia -J output.so --project=<same-env> ...

using PackageCompiler

output = isempty(ARGS) ? joinpath(pwd(), "sdpx-performance-sysimage.so") : abspath(ARGS[1])
workload = joinpath(@__DIR__, "precompile_performance_workload.jl")
packages = ["SDPX"]
for package in ("MultiFloats", "MultiFloatLinearAlgebra", "BigFloatLinearAlgebra")
    Base.find_package(package) === nothing || push!(packages, package)
end

PackageCompiler.create_sysimage(
    packages;
    sysimage_path=output,
    precompile_execution_file=workload,
    cpu_target=get(ENV, "SDPX_SYSIMAGE_CPU_TARGET", "native"),
    incremental=true,
)
println("SDPX performance sysimage: ", output)
