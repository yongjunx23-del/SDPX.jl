struct CBFData
    version::Int
    objective_sense::Symbol
    sections::Dict{String,Vector{String}}
end

const _CBF_KEYS = Set([
    "VER", "OBJSENSE", "PSDVAR", "VAR", "INT", "PSDCON", "CON",
    "OBJFCOORD", "OBJACOORD", "OBJBCOORD", "OBJHCOORD", "FCOORD",
    "ACOORD", "BCOORD", "HCOORD", "DCOORD",
])

"Read CBF 2.x sections, preserving sparse coordinate records verbatim."
function read_cbf(path::AbstractString)
    isfile(path) || throw(ArgumentError("missing CBF file $path; run benchmark/generic/scripts/fetch_generic_benchmarks.sh"))
    io = endswith(path, ".gz") ? open(`gzip -cd $path`) : open(path)
    sections = Dict{String,Vector{String}}()
    current = ""
    try
        for raw in eachline(io)
            line = strip(first(split(raw, '#'; limit=2)))
            isempty(line) && continue
            if line in _CBF_KEYS
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
    version = parse(Int, sections["VER"][1])
    sense = uppercase(sections["OBJSENSE"][1]) == "MIN" ? :minimize : :maximize
    return CBFData(version, sense, sections)
end
