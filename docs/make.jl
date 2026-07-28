# Build the documentation site:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl
#
# CI builds this without deploying; `deploydocs` activates once the repository
# has a DOCUMENTER_KEY secret (see docs/package-status.md, next steps).
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
        "API reference" => "api.md",
    ],
    # The wider docstrings are written for the source; only exported names are
    # required on pages.
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references],
)

if get(ENV, "CI", "false") == "true" && haskey(ENV, "DOCUMENTER_KEY")
    deploydocs(; repo="github.com/yongjunx23-del/SDPX.jl", devbranch="main")
end
