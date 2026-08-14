using Serialization
using SparseArrays

const DEFAULT_STUDY_ROOT = "/Users/xuyongjun/Desktop/project/massless eft scalar/benchmarks/c00_nx3_primal_full_study/mac_bf256_zero_c00_j40_na15_nmu200_nx2_nalpha2_v040_worktree"
const STUDY_ROOT = get(ENV, "CSDR_STUDY_ROOT", DEFAULT_STUDY_ROOT)
const CSDR_SOURCE = joinpath(STUDY_ROOT, "source", "src", "CSDRBootstrap.jl")
isfile(CSDR_SOURCE) || error("CSDRBootstrap source not found at $CSDR_SOURCE")
isdefined(Main, :CSDRBootstrap) || Base.include(Main, CSDR_SOURCE)

function main(args=ARGS)
    model_path = isempty(args) ?
        joinpath(STUDY_ROOT, "results", "generate", "model.bin") :
        abspath(args[1])
    output_path = length(args) >= 2 ? abspath(args[2]) :
        joinpath(@__DIR__, "reduced-neutral.bin")
    payload = open(deserialize, model_path)
    elimination = Main.CSDRBootstrap._eliminate_low_energy_variables(payload)
    reduced = elimination.problem
    neutral = (
        schema=:csdr_fixed_trace_reduced_v1,
        reduced_c=copy(reduced.c),
        reduced_B=sparse(reduced.B),
        reduced_b=copy(reduced.b),
        coefficient_constant=copy(elimination.coefficient_constant),
        coefficient_from_spectrum=copy(elimination.coefficient_from_spectrum),
        coefficient_labels=copy(payload.coefficient_labels),
        objective=Dict(payload.config.objective),
        fixed_coefficients=Dict(payload.config.fixed_coefficients),
        source_model_path=model_path,
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        serialize(io, neutral)
    end
    println("Reduced neutral CSDR payload written to $output_path")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
