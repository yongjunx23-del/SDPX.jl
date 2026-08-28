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

@inline function _sdpa_finite(token, context)
    value = parse(Float64, replace(token, 'D'=>'E'))
    isfinite(value) || throw(ArgumentError("SDPA $context must be finite, got $token"))
    return value
end

"Read and validate SDPLIB's sparse SDPA format without dense block matrices."
function read_sdpa(path::AbstractString)
    isfile(path) || throw(ArgumentError("missing SDPA file $path; run benchmark/generic/scripts/fetch_generic_benchmarks.sh"))
    lines = String[]
    for raw in eachline(path)
        line = strip(first(split(raw, r"[\*\"]"; limit=2)))
        isempty(line) || push!(lines, line)
    end
    length(lines) >= 4 || throw(ArgumentError("truncated SDPA file $path"))
    constraints = parse(Int, only(_sdpa_numbers(lines[1])))
    constraints >= 1 || throw(ArgumentError("SDPA constraint count must be positive"))
    blocks = parse(Int, only(_sdpa_numbers(lines[2])))
    blocks >= 1 || throw(ArgumentError("SDPA block count must be positive"))
    block_sizes = parse.(Int, _sdpa_numbers(lines[3]))
    length(block_sizes) == blocks || throw(DimensionMismatch("SDPA block count mismatch in $path"))
    all(!iszero, block_sizes) || throw(ArgumentError("SDPA block sizes must be nonzero"))
    rhs_tokens = _sdpa_numbers(lines[4])
    length(rhs_tokens) == constraints || throw(DimensionMismatch("SDPA RHS length mismatch in $path"))
    rhs = [_sdpa_finite(token, "RHS value") for token in rhs_tokens]
    entries = SDPAEntry[]
    coordinates = Set{NTuple{4,Int}}()
    for line in lines[5:end]
        tokens = _sdpa_numbers(line)
        length(tokens) == 5 || throw(ArgumentError("invalid SDPA sparse entry: $line"))
        matrix = parse(Int, tokens[1])
        block = parse(Int, tokens[2])
        row = parse(Int, tokens[3])
        column = parse(Int, tokens[4])
        value = _sdpa_finite(tokens[5], matrix == 0 ? "objective value" : "constraint value")
        0 <= matrix <= constraints || throw(ArgumentError(
            "SDPA matrix index $matrix is outside 0:$constraints"))
        1 <= block <= blocks || throw(ArgumentError(
            "SDPA block index $block is outside 1:$blocks"))
        dimension = abs(block_sizes[block])
        1 <= row <= dimension || throw(ArgumentError(
            "SDPA row $row is outside block $block dimension $dimension"))
        1 <= column <= dimension || throw(ArgumentError(
            "SDPA column $column is outside block $block dimension $dimension"))
        row <= column || throw(ArgumentError(
            "SDPA sparse symmetric entries must use upper-triangle indices (row <= column)"))
        block_sizes[block] < 0 && row != column && throw(ArgumentError(
            "SDPA diagonal block $block cannot contain off-diagonal entry ($row,$column)"))
        coordinate = (matrix, block, row, column)
        coordinate in coordinates && throw(ArgumentError(
            "duplicate SDPA sparse coordinate $coordinate"))
        push!(coordinates, coordinate)
        push!(entries, SDPAEntry(matrix, block, row, column, value))
    end
    isempty(entries) && throw(ArgumentError("SDPA file $path has no sparse matrix entries"))
    return SDPAData(constraints, block_sizes, rhs, entries)
end
