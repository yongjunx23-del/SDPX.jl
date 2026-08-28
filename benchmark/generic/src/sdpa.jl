struct SDPAEntry
    matrix::Int
    block::Int
    row::Int
    column::Int
    value::Float64
end

struct SDPAData
    constraints::Int
    block_sizes::Vector{Int}
    rhs::Vector{Float64}
    entries::Vector{SDPAEntry}
end

_sdpa_numbers(line) = split(replace(replace(strip(line), '{'=>' '), '}'=>' '), r"[\s,]+"; keepempty=false)

"Read SDPLIB's sparse SDPA format without materializing dense block matrices."
function read_sdpa(path::AbstractString)
    isfile(path) || throw(ArgumentError("missing SDPA file $path; run benchmark/generic/scripts/fetch_generic_benchmarks.sh"))
    lines = String[]
    for raw in eachline(path)
        line = strip(first(split(raw, r"[\*\"]"; limit=2)))
        isempty(line) || push!(lines, line)
    end
    length(lines) >= 4 || throw(ArgumentError("truncated SDPA file $path"))
    constraints = parse(Int, only(_sdpa_numbers(lines[1])))
    blocks = parse(Int, only(_sdpa_numbers(lines[2])))
    block_sizes = parse.(Int, _sdpa_numbers(lines[3]))
    length(block_sizes) == blocks || throw(DimensionMismatch("SDPA block count mismatch in $path"))
    rhs = parse.(Float64, replace.(_sdpa_numbers(lines[4]), 'D'=>'E'))
    length(rhs) == constraints || throw(DimensionMismatch("SDPA RHS length mismatch in $path"))
    entries = SDPAEntry[]
    for line in lines[5:end]
        tokens = _sdpa_numbers(line)
        length(tokens) == 5 || throw(ArgumentError("invalid SDPA sparse entry: $line"))
        push!(entries, SDPAEntry(parse(Int, tokens[1]), parse(Int, tokens[2]),
            parse(Int, tokens[3]), parse(Int, tokens[4]),
            parse(Float64, replace(tokens[5], 'D'=>'E'))))
    end
    return SDPAData(constraints, block_sizes, rhs, entries)
end
