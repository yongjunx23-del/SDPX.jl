# Build the documentation site:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl
#
using Documenter
using SDPX

makedocs(;
    sitename="SDPX.jl",
    modules=[SDPX],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => [
            "Quick start" => "quickstart.md",
            "Command line" => "cli.md",
            "Precision" => "precision.md",
            "JuMP and MOI" => "jump.md",
            "Convex.jl" => "convex.md",
            "SOC and fixed trace" => "soc-fixed-trace.md",
        ],
        "Solver workflow" => [
            "Automatic pipeline" => "pipeline.md",
            "Parameters" => "parameters.md",
            "Diagnostics and certificates" => "diagnostics.md",
        ],
        "API reference" => "api.md",
        "Experimental internals" => "internals.md",
        "Formulation planner" => "formulation-planner.md",
        "Dense augmented KKT" => "augmented-kkt.md",
        "Round 3 augmented A/B" => "round3-augmented-kkt-results.md",
        "Round 4 formulation planner" => "round4-formulation-planner-results.md",
        "Architecture v0.4.1 -> v0.5" => "architecture-v041.md",
        "Development" => "development.md",
    ],
    checkdocs=:exports,
    checkdocs_ignored_modules=[SDPX.ExtendedPrecisionBLAS],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(
        ;
        repo="github.com/yongjunx23-del/SDPX.jl.git",
        devbranch="main",
        push_preview=true,
    )
end
