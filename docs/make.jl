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
            "SOCP and fixed trace" => "socp.md",
        ],
        "Solver workflow" => [
            "Automatic pipeline" => "pipeline.md",
            "Preprocessing" => "preprocessing.md",
            "Sparse execution" => "sparse-execution.md",
            "Parameters" => "parameters.md",
            "Diagnostics and certificates" => "diagnostics.md",
        ],
        "Architecture" => [
            "Architecture" => "architecture.md",
            "Linear-algebra providers" => "providers.md",
        ],
        "Topic guides" => [
            "Threading" => "threading.md",
            "Cluster workflow" => "cluster-workflow.md",
            "Bridge schema" => "bridge-schema.md",
            "Adaptive parameter policy" => "adaptive-parameter-policy.md",
            "Adaptive dense/sparse optimization" =>
                "adaptive-dense-sparse-optimization.md",
        ],
        "Benchmarks and evidence" => "benchmarks.md",
        "API reference" => "api.md",
        "Experimental internals" => "internals.md",
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
