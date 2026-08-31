#!/usr/bin/env julia

# One arithmetic per fresh process (provider isolation is mandatory):
#   SDPX_PRECISION=Float64x4 julia --project=benchmark/general/precision_env \
#       benchmark/general/precision_matrix.jl
# Filters: SDPX_PRECISION_IDS=lp_afiro_style,rsoc_epigraph_small

using SDPX
using Printf

const PRECISION_NAME=Symbol(get(ENV,"SDPX_PRECISION","Float64"))
const ARITHMETIC = if PRECISION_NAME===:Float64
    Float64
elseif PRECISION_NAME in (:Float64x2,:Float64x3,:Float64x4)
    @eval using MultiFloats
    @eval using MultiFloatLinearAlgebra
    getfield(MultiFloats,PRECISION_NAME)
elseif PRECISION_NAME in (:BigFloat256,:BigFloat512,:BigFloat1024)
    @eval using BigFloatLinearAlgebra
    BigFloat
else
    error("unknown SDPX_PRECISION=$PRECISION_NAME")
end

include(joinpath(@__DIR__,"GenericConicBenchmark.jl"))
using .GenericConicBenchmark

const DEFAULT_IDS=(
    :lp_afiro_style,
    :socp_portfolio_small,
    :socp_ill_scaled_small,
    :rsoc_epigraph_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
    :power_epigraph_small,
    :mixed_orthant_exp_small,
)

function selected_ids()
    value=get(ENV,"SDPX_PRECISION_IDS","")
    isempty(value) && return collect(DEFAULT_IDS)
    return Symbol.(strip.(split(value,',')))
end

function selected_precision_spec()
    rows = if PRECISION_NAME===:Float64
        precision_specs(Float64,Float64,Float64)
    elseif PRECISION_NAME in (:Float64x2,:Float64x3,:Float64x4)
        precision_specs(
            MultiFloats.Float64x2,
            MultiFloats.Float64x3,
            MultiFloats.Float64x4,
        )
    else
        # MultiFloat types are intentionally not loaded into BigFloat workers.
        bits=PRECISION_NAME===:BigFloat256 ? 256 :
            PRECISION_NAME===:BigFloat512 ? 512 : 1024
        tolerance=bits==256 ? "1e-32" : bits==512 ? "1e-50" : "1e-80"
        limit=bits==256 ? "5e-28" : bits==512 ? "5e-46" : "5e-74"
        return PrecisionSpec(PRECISION_NAME,BigFloat,bits,tolerance,limit,
            :bigfloat_linear_algebra)
    end
    return only(filter(row->row.name===PRECISION_NAME,rows))
end

function main()
    threads=parse(Int,get(ENV,"SDPX_PRECISION_THREADS","1"))
    SDPX.set_blas_threads!(1)
    precision_spec=selected_precision_spec()
    rows=PrecisionBenchmarkResult[]
    for id in selected_ids()
        matches=filter(spec->spec.id===id,inventory())
        length(matches)==1 || error("unknown or duplicate precision case $id")
        row=run_precision_case(precision_spec,only(matches);threads)
        push!(rows,row)
        @printf("PRECISION id=%s arithmetic=%s bits=%d status=%s cert=%s iter=%d time_s=%.6f bytes=%d pass=%s obj=%s expected=%s rp=%s rd=%s gap=%s\n",
            String(row.id),String(row.arithmetic),row.bits,String(row.status),
            string(row.certificate_valid),row.iterations,row.seconds,row.bytes,
            string(row.passed),row.objective,row.expected_objective,
            row.primal_residual,row.dual_residual,row.relative_gap)
        flush(stdout)
    end
    failures=filter(row->!row.passed,rows)
    isempty(failures) || error("precision matrix failures: " *
        join(("$(row.arithmetic)/$(row.id):$(row.status)" for row in failures),","))
    return rows
end

main()
