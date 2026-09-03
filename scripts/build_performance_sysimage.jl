# Build an incremental SDPX performance sysimage for the current Julia build.
# Usage:
#   julia --project=<env-with-PackageCompiler> \
#       scripts/build_performance_sysimage.jl [output.so]
# Run solves with: julia -J output.so --project=<same-env> ...

using PackageCompiler

output = isempty(ARGS) ? joinpath(pwd(), "sdpx-performance-sysimage.so") : abspath(ARGS[1])
profile = Symbol(get(ENV, "SDPX_SYSIMAGE_PROFILE", "safe_float64"))
packages = ["SDPX"]
profile === :safe_float64 || error(
    "unsupported SDPX_SYSIMAGE_PROFILE=$profile; only safe_float64 may be serialized",
)
# Never serialize mutable MPFR state or provider/thread payloads into the
# release image. Extended-precision methods JIT normally in each consumer.
workload = joinpath(@__DIR__, "precompile_safe_float64_workload.jl")
cpu_target = get(ENV, "SDPX_SYSIMAGE_CPU_TARGET", "generic")
max_bytes = parse(Int, get(ENV, "SDPX_SYSIMAGE_MAX_BYTES", string(1024^3)))

PackageCompiler.create_sysimage(
    packages;
    sysimage_path=output,
    precompile_execution_file=workload,
    cpu_target=cpu_target,
    incremental=true,
)
size_bytes = filesize(output)
if size_bytes > max_bytes
    rm(output; force=true)
    error("sysimage exceeded size bound: $size_bytes > $max_bytes bytes")
end
println(
    "SDPX performance sysimage: ", output,
    " profile=", profile,
    " cpu_target=", cpu_target,
    " bytes=", size_bytes,
)
