# Stable performance precompile workload for PackageCompiler.
#
# Covers the complete public Float64 E2E matrix plus small Float64x4 and
# BigFloat256 fixed-trace Q3 solves so first real solves do not pay 100+ seconds
# of MultiFloat/BigFloat method compilation.  This file is execution-only: it
# changes no solver setting, default, tolerance, or certificate policy.

using SDPX

include(joinpath(@__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl"))
using .GenericConicBenchmark

for id in (
    :lp_afiro_style,
    :lp_infeasible,
    :lp_unbounded,
    :socp_portfolio_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
    :power_epigraph_small,
)
    spec = only(filter(s -> s.id === id, inventory(tier=:small)))
    run_one(spec, Float64)
end

function _precompile_fixed_trace(::Type{T}; iterations::Int=2) where {T<:AbstractFloat}
    cells = 4
    variables = 2cells
    model = SDPX.Model(T; name="precompile_fixed_trace_$(T)")
    x = SDPX.variable!(model, :spectral, variables; domain=SDPX.Reals())
    equality = zero(T)
    for j in 1:variables
        equality += x[j]
    end
    SDPX.constraint!(model, :sum_rule, equality, SDPX.ZeroCone())
    for cell in 1:cells
        r = x[2cell - 1]
        q = x[2cell]
        SDPX.constraint!(
            model, Symbol(:unitarity_, cell),
            Any[one(T), q - one(T), r], SDPX.LorentzCone(),
        )
    end
    objective = zero(T)
    for j in 1:variables
        objective += T(j) * x[j]
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    settings = SDPX.Settings{T}(
        tolerances=SDPX.Tolerances{T}(
            primal=T(1e-8), dual=T(1e-8), gap=T(1e-8),
        ),
        limits=SDPX.Limits(iterations=iterations, time=120.0, threads=1),
        kkt_route=:bordered,
        verbosity=0,
    )
    outputs = SDPX.Outputs(
        :all, :all, :all;
        objectives=true, certificate=:summary,
        diagnostics=:full, history=false, trace=false,
    )
    SDPX.optimize!(model; settings, outputs)
    return nothing
end

try
    @eval using MultiFloats: Float64x4
    @eval using MultiFloatLinearAlgebra
    ENV["SDPX_MEMORY_RSS_OVERRIDE_MB"] = "512"
    _precompile_fixed_trace(Float64x4)
catch exception
    @warn "MultiFloat fixed-trace precompile workload unavailable" exception
end

try
    @eval using BigFloatLinearAlgebra
    setprecision(BigFloat, 256) do
        ENV["SDPX_MEMORY_RSS_OVERRIDE_MB"] = "512"
        _precompile_fixed_trace(BigFloat; iterations=1)
    end
catch exception
    @warn "BigFloat fixed-trace precompile workload unavailable" exception
end
