# Bounded PackageCompiler execution workload.
#
# Keep this file intentionally Float64-only.  MultiFloats and BigFloat carry
# provider state and mutable precision-dependent storage that must not be
# serialized into the release sysimage.  Extended-precision solver methods
# compile normally in the consuming process against this base image.

using SDPX

model = SDPX.Model(Float64; name="sysimage_safe_soc")
x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
SDPX.constraint!(
    model, :unit, Any[1.0, x[1] - 1.0, x[2]], SDPX.LorentzCone(),
)
SDPX.objective!(model, SDPX.Maximize(), x[1] + 0.5)
settings = SDPX.Settings(
    Float64;
    limits=SDPX.Limits(iterations=80, time=30.0, threads=1),
    verbosity=0,
    certification=true,
)
outputs = SDPX.Outputs(
    :all, :all, :all;
    objectives=true,
    certificate=:summary,
    diagnostics=:summary,
    history=false,
    trace=false,
)
result = SDPX.optimize!(model; settings, outputs)
SDPX.status(result) === :optimal || error(
    "safe sysimage workload failed with status $(SDPX.status(result))",
)
SDPX.certificate(result).valid || error(
    "safe sysimage workload did not produce a valid certificate",
)
