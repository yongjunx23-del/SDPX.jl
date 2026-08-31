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
            "Adaptive predictor-corrector policy" =>
                "adaptive-parameter-policy.md",
        ],
        "Benchmarks and evidence" => "benchmarks.md",
        "API reference" => "api.md",
        "Experimental internals" => "internals.md",
        "Development" => "development.md",
    ],
    checkdocs=:exports,
    # ExtendedPrecisionBLAS and SymmetricCones are explicitly internal
    # implementation modules, not part of SDPX's frozen public export surface.
    checkdocs_ignored_modules=[SDPX.ExtendedPrecisionBLAS, SDPX.SymmetricCones],
)

# Pull requests must build the same site as main without attempting a deploy.
# The workflow opts into deployment explicitly only for trusted push runs.
if get(ENV, "SDPX_DEPLOY_DOCS", "false") == "true"
    deploydocs(
        ;
        repo="github.com/yongjunx23-del/SDPX.jl.git",
        devbranch="main",
        push_preview=true,
    )
end
