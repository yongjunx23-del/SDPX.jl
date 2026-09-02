# Stable performance precompile workload for PackageCompiler.
#
# Covers the complete public Float64 E2E matrix, pure-SOCP/pure-2x2-SDP/mixed
# generic Float64x4 symmetric-core paths, and retained Float64x4/BigFloat256
# fixed-trace Q3 paths. This file is execution-only: it changes no solver
# setting, default, tolerance, or certificate policy.

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

function _precompile_generic_symmetric_core(
    ::Type{T}, family::Symbol; iterations::Int=3,
) where {T<:AbstractFloat}
    family in (:soc, :sdp, :mixed) || throw(ArgumentError(
        "generic precompile family must be :soc, :sdp, or :mixed",
    ))
    model = SDPX.Model(T; name="precompile_generic_$(family)_$(T)")
    x = SDPX.variable!(model, :shared, 3; domain=SDPX.Reals())
    # Every cone shares the same variables. These fixtures compile the generic
    # symmetric augmented core and cannot satisfy disjoint-Q3 applicability.
    if family in (:soc, :mixed)
        SDPX.constraint!(model, :soc_1,
            Any[one(T), x[1] + (T(1) / T(4)) * x[2], x[3]],
            SDPX.LorentzCone())
        SDPX.constraint!(model, :soc_2,
            Any[one(T), x[1] - (T(1) / T(3)) * x[2],
                x[2] + (T(1) / T(5)) * x[3]],
            SDPX.LorentzCone())
    end
    if family in (:sdp, :mixed)
        SDPX.constraint!(model, :psd_1,
            Any[one(T) + x[1] x[2]; x[2] one(T) - x[1]], SDPX.PSDCone())
        SDPX.constraint!(model, :psd_2,
            Any[one(T) + x[2] x[3]; x[3] one(T) - x[2]], SDPX.PSDCone())
    end
    SDPX.objective!(model, SDPX.Minimize(),
        -x[1] + (T(1) / T(7)) * x[2])
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
    result = SDPX.optimize!(model; settings, outputs)
    diagnostics = SDPX.diagnostics(result)
    selected = diagnostics.selected_algorithms
    selected.executed_kkt_formulation === :symmetric_augmented_hsd_core ||
        error("$family workload missed the generic symmetric core")
    selected.executed_backend === :symmetric_augmented_core ||
        error("$family workload selected backend $(selected.executed_backend)")
    selected.la_executed_provider === :multifloat_linear_algebra ||
        error("$family workload missed MFLA")
    selected.executed_factorization_kernel === :mfla_pivoted_ldlt ||
        error("$family workload missed MFLA pivoted LDLT")
    result.iterations >= 1 || error("$family workload performed no iteration")
    diagnostics.memory.symmetric_core_actual_factor_epoch >= 1 ||
        error("$family workload performed no symmetric-core factorization")
    return result
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

if Base.find_package("MultiFloats") !== nothing &&
   Base.find_package("MultiFloatLinearAlgebra") !== nothing
    @eval using MultiFloats: Float64x4
    @eval using MultiFloatLinearAlgebra
    old_override = get(ENV, "SDPX_MEMORY_RSS_OVERRIDE_MB", nothing)
    try
        ENV["SDPX_MEMORY_RSS_OVERRIDE_MB"] = "512"
        for family in (:soc, :sdp, :mixed)
            _precompile_generic_symmetric_core(Float64x4, family)
        end
        _precompile_fixed_trace(Float64x4)
    finally
        if old_override === nothing
            pop!(ENV, "SDPX_MEMORY_RSS_OVERRIDE_MB", nothing)
        else
            ENV["SDPX_MEMORY_RSS_OVERRIDE_MB"] = old_override
        end
    end
else
    @info "skipping optional Float64x4 workloads: MultiFloats/MFLA unavailable"
end

if Base.find_package("BigFloatLinearAlgebra") !== nothing
    @eval using BigFloatLinearAlgebra
    old_override = get(ENV, "SDPX_MEMORY_RSS_OVERRIDE_MB", nothing)
    try
        ENV["SDPX_MEMORY_RSS_OVERRIDE_MB"] = "512"
        setprecision(BigFloat, 256) do
            _precompile_fixed_trace(BigFloat; iterations=1)
        end
    finally
        if old_override === nothing
            pop!(ENV, "SDPX_MEMORY_RSS_OVERRIDE_MB", nothing)
        else
            ENV["SDPX_MEMORY_RSS_OVERRIDE_MB"] = old_override
        end
    end
else
    @info "skipping optional BigFloat workload: BFLA unavailable"
end
