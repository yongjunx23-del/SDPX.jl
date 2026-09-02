#!/usr/bin/env julia
# Provider-backed qualification runner for the certified V2 optimal catalog.
# Run in a fresh process with --gcthreads=1. The runner deliberately skips
# ray cases and enumerates every live certified optimal-path catalog row.
using Dates
using Printf
using MultiFloats
using MultiFloatLinearAlgebra
using BigFloatLinearAlgebra
using SDPX

include(joinpath(@__DIR__, "GeneralBenchmarkV2.jl"))
using .GeneralBenchmarkV2

const OUT = get(ENV, "V2_PRECISION_OUT",
    joinpath(dirname(dirname(@__DIR__)), "..", "docs", "reviews",
             "V2_PRECISION_QUALIFICATION.md"))
const CATALOGS = [
    lp_tranche_catalog(), ill_conditioned_tranche_catalog(),
    socp_tranche_catalog(), rsoc_tranche_catalog(), sdp_tranche_catalog(),
    exp_tranche_catalog(), power_tranche_catalog(), mixed_tranche_catalog(),
]
const CASES = [i for c in CATALOGS for i in c.instances
               if i.reference.status === :optimal]

function _specs()
    return [
        V2Precision(:Float64x2, Float64x2, 104, "1e-15", "5e-13", :multifloat_linear_algebra),
        V2Precision(:Float64x4, Float64x4, 208, "1e-28", "5e-22", :multifloat_linear_algebra),
        V2Precision(:BigFloat256, BigFloat, 256, "1e-32", "5e-28", :bigfloat_linear_algebra),
    ]
end

function _exact_objective(instance)
    hasproperty(instance.payload, :objective) || return nothing
    BigFloat(getproperty(instance.payload, :objective))
end

function _row(instance, precision)
    started = time()
    exact = _exact_objective(instance)
    try
        result = run_instance(instance_catalog(instance), instance, precision)
        value = setprecision(BigFloat, max(256, 2 * precision.bits)) do
            BigFloat(result.objective)
        end
        err = exact === nothing ? nothing : value - exact
        qualified = result.status === :optimal && result.certificate_valid &&
                    result.validation.reference && isempty(result.validation.failures)
        reason = qualified ? "" : join(string.(result.validation.failures), ",")
        return (; id=instance.id, precision=precision.name, bits=precision.bits,
            status=string(result.status), certificate=result.certificate_valid,
            objective=result.objective, error=err === nothing ? "n/a" : string(err),
            iterations=result.iterations, core=result.core_seconds,
            qualification=qualified ? "qualified" : "not-qualified",
            reason=isempty(reason) ? "status/certificate/oracle gate" : reason,
            elapsed=time() - started)
    catch err
        return (; id=instance.id, precision=precision.name, bits=precision.bits,
            status="exception", certificate=false, objective="n/a", error="n/a",
            iterations="n/a", core="n/a", qualification="not-qualified",
            reason=string(typeof(err), ": ", sprint(showerror, err)),
            elapsed=time() - started)
    end
end

# V2 run_instance receives a catalog to retain family-specific builders.
const INSTANCE_CATALOG = Dict{Symbol,Any}(i.id => c for c in CATALOGS for i in c.instances)
instance_catalog(instance) = INSTANCE_CATALOG[instance.id]

function main()
    requested = isempty(ARGS) ? nothing : Symbol(ARGS[1])
    requested_case = get(ENV, "V2_CASE_ID", "")
    specs = requested === nothing ? _specs() : filter(p -> p.name === requested, _specs())
    isempty(specs) && error("unknown qualification precision $(requested)")
    cases = isempty(requested_case) ? CASES : filter(i -> string(i.id) == requested_case, CASES)
    isempty(cases) && error("unknown qualification case $(requested_case)")
    rows = NamedTuple[]
    for precision in specs
        for instance in cases
            push!(rows, _row(instance, precision))
            row = rows[end]
            @printf("%s %s %s status=%s cert=%s obj=%s iters=%s core=%s %s\n",
                row.precision, row.id, row.qualification, row.status,
                row.certificate, row.objective, row.iterations, row.core,
                row.reason)
        end
    end
    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, "# V2 provider-backed precision qualification")
        println(io)
        println(io, "Generated in a fresh Julia process on ", Dates.now(),
                ". All ", length(CASES), " live certified optimal-path cases are run; " *
                "the two ray cases are intentionally skipped by this report.")
        println(io)
        println(io, "Provider status: `MultiFloats`, `MultiFloatLinearAlgebra`, and " *
                "`BigFloatLinearAlgebra` loaded in this process. BigFloat runs " *
                "are inside `setprecision(BigFloat, bits)` and all results are " *
                "qualified only through the existing V2 certificate/oracle gates.")
        println(io)
        println(io, "## Qualification matrix")
        println(io)
        println(io, "| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |")
        println(io, "|---|---:|---|---:|---:|---:|---:|---:|---|---|")
        for row in rows
            println(io, "| `", row.id, "` | `", row.precision, "` | `", row.status,
                "` | ", row.certificate, " | `", row.objective, "` | `", row.error,
                "` | `", row.iterations, "` | `", row.core, "` | **", row.qualification,
                "** | ", replace(row.reason, "|" => "\\|"), " |")
        end
        println(io)
        println(io, "## Contract")
        println(io)
        println(io, "A row is **qualified** only when the requested arithmetic provider " *
                "is loaded, status is `:optimal`, the public certificate is valid, " *
                "the original-coordinate certificate gate passes, the independent " *
                "oracle passes at `max(256,2*bits)`, and the objective is within the " *
                "reviewed allowance (Float64x2 `5e-13`, Float64x4 `5e-22`, " *
                "BigFloat256 `5e-28`). No solver tolerance or formulation is changed " *
                "by this report.")
    end
    qualified = count(r -> r.qualification == "qualified", rows)
    println("WROTE ", OUT, " rows=", length(rows), " qualified=", qualified,
        " not_qualified=", length(rows) - qualified)
end

main()
