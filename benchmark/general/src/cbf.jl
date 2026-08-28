struct CBFData
    version::Int
    objective_sense::Symbol
    sections::Dict{String,Vector{String}}
end

const _CBF_SUPPORTED_VERSION = 2
const _CBF_KEYS = Set([
    "VER", "OBJSENSE", "PSDVAR", "VAR", "INT", "PSDCON", "CON",
    "OBJFCOORD", "OBJACOORD", "OBJBCOORD", "OBJHCOORD", "FCOORD",
    "ACOORD", "BCOORD", "HCOORD", "DCOORD",
])

"Read continuous CBF 2 sections, preserving sparse coordinate records verbatim."
function read_cbf(path::AbstractString)
    isfile(path) || throw(ArgumentError("missing CBF file $path; run benchmark/general/scripts/fetch_generic_benchmarks.sh"))
    io = endswith(path, ".gz") ? open(`gzip -cd $path`) : open(path)
    sections = Dict{String,Vector{String}}()
    current = ""
    try
        for raw in eachline(io)
            line = strip(first(split(raw, '#'; limit=2)))
            isempty(line) && continue
            if line in _CBF_KEYS
                line == "INT" && throw(ArgumentError(
                    "CBF integer declarations are unsupported; refusing to relax a MIP"))
                haskey(sections, line) && throw(ArgumentError(
                    "duplicate CBF section $line in $path"))
                current = line
                sections[current] = String[]
            else
                isempty(current) && throw(ArgumentError("CBF data before section header in $path"))
                push!(sections[current], line)
            end
        end
    finally
        close(io)
    end
    haskey(sections, "VER") || throw(ArgumentError("CBF file has no VER section"))
    haskey(sections, "OBJSENSE") || throw(ArgumentError("CBF file has no OBJSENSE section"))
    length(sections["VER"]) == 1 || throw(ArgumentError("CBF VER must contain exactly one integer"))
    version = try
        parse(Int, sections["VER"][1])
    catch error
        throw(ArgumentError("malformed CBF version $(repr(sections["VER"][1])): $error"))
    end
    version == _CBF_SUPPORTED_VERSION || throw(ArgumentError(
        "unsupported CBF version $version; only version $(_CBF_SUPPORTED_VERSION) is supported"))
    length(sections["OBJSENSE"]) == 1 || throw(ArgumentError(
        "CBF OBJSENSE must contain exactly one token"))
    objective_sense = sections["OBJSENSE"][1]
    objective_sense in ("MIN", "MAX") || throw(ArgumentError(
        "invalid CBF OBJSENSE $(repr(objective_sense)); expected exactly MIN or MAX"))
    sense = objective_sense == "MIN" ? :minimize : :maximize
    return CBFData(version, sense, sections)
end
