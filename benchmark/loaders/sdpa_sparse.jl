"""
    _build_sdpa_sparse_problem(spec, ::Type{T}, path, checksum)

Load a sparse SDPA problem into SDPX's canonical `SDPProblem{T}` layout.

The sparse SDPA data format stores one constant matrix `F₀` (matrix number
zero), one matrix `Fᵢ` for every scalar variable, and the objective vector
`c`.  Its standard primal convention is

```
    min c'x  subject to  Σᵢ Fᵢ xᵢ - F₀ ⪰ 0.
```

This is exactly SDPX's convention (`C = F₀`, `Aᵢ = Fᵢ`) for DIMACS
`.dat-s` files, whose five-field records are consumed directly.  SDPLIB's
historical `.sdp.gz` files use SDPpack's compact one-number-per-line encoding
instead.  SDPpack stores the primal `min F₀•X` with `Fᵢ•X=bᵢ`; converting its
dual to SDPX's `min c'x` / `Σ Aᵢxᵢ−C ⪰ 0` convention therefore maps
`c=-b`, `Aᵢ=-Fᵢ`, and `C=-F₀`.  The compact parser reconstructs those matrices
from SDPpack's per-block storage flags and lower-triangle payload.
A negative SDPA block size denotes a diagonal/linear cone block; it is
expanded into one positive `1×1` PSD block per scalar coordinate before
ingestion.

The helper intentionally has a loader-specific name.  The benchmark registry
may dispatch to it from its generic external-loader entry point without
introducing another `build_external_problem` method with the same signature
as an existing loader.
"""

using SparseArrays
using SHA

const _SDPA_SPARSE_LOADERS = Set{Symbol}((
    :sdpa_sparse,
    :sdpa_sparse_gzip,
    :external_sdpa_sparse,
    :external_sdpa_sparse_gzip,
    # SDPLIB's `.sdp.gz` files are SDPpack/Nemirovskii compact streams, not
    # DIMACS `.dat-s` streams.  Keep the format symbols explicit so a caller
    # cannot accidentally parse one with the other parser.
    :sdppack_compact,
    :sdppack_compact_gzip,
    :external_sdppack_compact,
    :external_sdppack_compact_gzip,
))

const _SDPPACK_COMPACT_LOADERS = Set{Symbol}((
    :sdppack_compact,
    :sdppack_compact_gzip,
    :external_sdppack_compact,
    :external_sdppack_compact_gzip,
))

const _SDPA_DIMACS_LOADERS = Set{Symbol}((
    :sdpa_sparse,
    :sdpa_sparse_gzip,
    :external_sdpa_sparse,
    :external_sdpa_sparse_gzip,
))

"""Read an SDPA file, using a direct argv-style gzip command for `.gz` input."""
function _sdpa_read_text(path::AbstractString)
    isfile(path) || throw(ArgumentError("SDPA input does not exist: $path"))
    filename = String(path)
    if endswith(lowercase(filename), ".gz")
        # `Cmd` receives the path as one argument; no shell interpolation is
        # involved, so spaces and metacharacters in a cache path are harmless.
        return read(Cmd(["gzip", "-dc", "--", filename]), String)
    end
    return read(filename, String)
end

"""Compute the lowercase SHA-256 digest used by the benchmark cache contract."""
function _sdpa_sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"""
    _sdpa_tokens(text) -> Vector{String}

Tokenise the permissive textual part of the SDPA sparse format.  SDPA files
commonly contain `*` comment lines and braces/commas around the header; a few
DIMACS mirrors also carry shell-style inline comments.  We remove only those
known comment forms and punctuation, leaving unknown text to the typed parser
so malformed input is rejected rather than silently ignored.
"""
function _sdpa_tokens(text::AbstractString)
    tokens = String[]
    for original_line in split(String(text), '\n'; keepempty=true)
        line = original_line
        stripped = lstrip(line)
        isempty(stripped) && continue
        first_char = first(stripped)
        first_char in ('*', '#', '%', '!', ';') && continue

        # Strip inline comments before replacing punctuation.  The order is
        # significant for `//`; the individual slash is not a valid token and
        # will consequently be rejected if a future format uses it otherwise.
        for marker in ("//", "#", "%", "!", "*")
            position = findfirst(marker, line)
            position === nothing && continue
            marker_start = first(position)
            line = marker_start == firstindex(line) ?
                   "" : line[firstindex(line):prevind(line, marker_start)]
        end
        isempty(strip(line)) && continue

        line = replace(
            line,
            ',' => ' ',
            '{' => ' ',
            '}' => ' ',
            '(' => ' ',
            ')' => ' ',
            '[' => ' ',
            ']' => ' ',
            '=' => ' ',
            ';' => ' ',
        )
        append!(tokens, split(line))
    end
    return tokens
end

function _sdpa_take_token(tokens::Vector{String}, cursor::Base.RefValue{Int}, role)
    index = cursor[]
    index <= length(tokens) || throw(ArgumentError(
        "truncated SDPA input while reading $role",
    ))
    cursor[] = index + 1
    return tokens[index]
end

function _sdpa_parse_int(
    tokens::Vector{String},
    cursor::Base.RefValue{Int},
    role,
)
    token = _sdpa_take_token(tokens, cursor, role)
    try
        return parse(Int, token)
    catch
        throw(ArgumentError("invalid integer token $(repr(token)) for $role"))
    end
end

"""Parse one finite scalar directly in the requested arithmetic type."""
function _sdpa_parse_scalar(
    tokens::Vector{String},
    cursor::Base.RefValue{Int},
    ::Type{T},
    role,
) where {T}
    source = _sdpa_take_token(tokens, cursor, role)
    # Julia's BigFloat and MultiFloats parsers accept `E` but not Fortran's
    # `D`; replacing both cases keeps all significant digits in the target
    # type and avoids a Float64 staging conversion.
    normalized = replace(replace(source, 'D' => 'E'), 'd' => 'e')
    value = try
        parse(T, normalized)
    catch
        throw(ArgumentError("invalid numeric token $(repr(source)) for $role"))
    end
    isfinite(value) || throw(ArgumentError(
        "non-finite numeric token $(repr(source)) for $role",
    ))
    return value
end

"""Insert a value into a canonical symmetric sparse entry, summing repeats."""
function _sdpa_add_entry!(
    groups::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},T}},
    matrix_index::Int,
    block_index::Int,
    row::Int,
    column::Int,
    value::T,
) where {T}
    pair = (min(row, column), max(row, column))
    group = get!(groups, (matrix_index, block_index)) do
        Dict{Tuple{Int,Int},T}()
    end
    group[pair] = get(group, pair, zero(T)) + value
    return nothing
end

"""
Parse the header and sparse records.  `source` is deliberately explicit: the
two public collections use different header orders even though both use SDPA
record syntax.
"""
function _parse_sdpa_sparse(path::AbstractString, source::Symbol, ::Type{T}) where {T}
    source === :dimacs || throw(ArgumentError(
        "DIMACS SDPA sparse loader expects source=:dimacs, got $source; " *
        "use _parse_sdppack_compact for SDPLIB SDPpack streams",
    ))
    tokens = _sdpa_tokens(_sdpa_read_text(path))
    isempty(tokens) && throw(ArgumentError("SDPA input is empty"))
    cursor = Ref(1)

    m = _sdpa_parse_int(tokens, cursor, "variable count m")
    m > 0 || throw(ArgumentError("SDPA variable count m must be positive"))

    c = T[]
    nblocks = 0
    block_sizes = Int[]
    nblocks = _sdpa_parse_int(tokens, cursor, "block count")
    nblocks > 0 || throw(ArgumentError("SDPA block count must be positive"))
    sizehint!(block_sizes, nblocks)
    for block in 1:nblocks
        dimension = _sdpa_parse_int(tokens, cursor, "block_sizes[$block]")
        dimension != 0 || throw(ArgumentError(
            "SDPA block_sizes[$block] may not be zero",
        ))
        push!(block_sizes, dimension)
    end
    sizehint!(c, m)
    for variable in 1:m
        push!(c, _sdpa_parse_scalar(tokens, cursor, T, "objective[$variable]"))
    end

    # Each dictionary is keyed by (matrix number, block number), and each
    # nested key is an unordered matrix coordinate.  Thus repeated records
    # and mirrored records are summed into one canonical entry before the
    # symmetric matrix is materialised.
    groups = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},T}}()
    while cursor[] <= length(tokens)
        matrix_index = _sdpa_parse_int(tokens, cursor, "matrix index")
        block_index = _sdpa_parse_int(tokens, cursor, "block index")
        row = _sdpa_parse_int(tokens, cursor, "row index")
        column = _sdpa_parse_int(tokens, cursor, "column index")
        value = _sdpa_parse_scalar(tokens, cursor, T, "sparse coefficient")

        0 <= matrix_index <= m || throw(ArgumentError(
            "matrix index $matrix_index is outside 0:$m",
        ))
        1 <= block_index <= nblocks || throw(ArgumentError(
            "block index $block_index is outside 1:$nblocks",
        ))
        dimension = abs(block_sizes[block_index])
        1 <= row <= dimension || throw(ArgumentError(
            "row index $row is outside 1:$dimension for block $block_index",
        ))
        1 <= column <= dimension || throw(ArgumentError(
            "column index $column is outside 1:$dimension for block $block_index",
        ))
        block_sizes[block_index] < 0 && row != column && throw(ArgumentError(
            "negative/linear block $block_index only permits diagonal entries; " *
            "got ($row,$column)",
        ))
        _sdpa_add_entry!(groups, matrix_index, block_index, row, column, value)
    end

    return (m=m, c=c, block_sizes=block_sizes, groups=groups)
end

function _sdpa_dense_symmetric(
    ::Type{T},
    dimension::Int,
    group::Union{Nothing,Dict{Tuple{Int,Int},T}},
    scale::T=one(T),
) where {T}
    matrix = zeros(T, dimension, dimension)
    group === nothing && return matrix
    for ((row, column), value) in group
        iszero(value) && continue
        scaled = scale * value
        matrix[row, column] = scaled
        row == column || (matrix[column, row] = scaled)
    end
    return matrix
end

function _sdpa_sparse_matrix(
    ::Type{T},
    dimension::Int,
    group::Union{Nothing,Dict{Tuple{Int,Int},T}},
    scale::T=one(T),
) where {T}
    group === nothing && return spzeros(T, dimension, dimension)
    rows = Int[]
    columns = Int[]
    values = T[]
    for ((row, column), value) in group
        iszero(value) && continue
        push!(rows, row)
        push!(columns, column)
        push!(values, scale * value)
        if row != column
            push!(rows, column)
            push!(columns, row)
            push!(values, scale * value)
        end
    end
    isempty(values) && return spzeros(T, dimension, dimension)
    return sparse(rows, columns, values, dimension, dimension)
end

"""Read one SDPpack compact C/A block into the common sparse-entry groups."""
function _sdppack_read_block!(
    tokens::Vector{String},
    cursor::Base.RefValue{Int},
    ::Type{T},
    groups::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},T}},
    matrix_index::Int,
    block_index::Int,
    dimension::Int,
    expected_storage::Union{Nothing,Int}=nothing,
) where {T}
    sparse_flag = _sdpa_parse_int(tokens, cursor, "SDPpack block storage flag")
    sparse_flag in (0, 1) || throw(ArgumentError(
        "SDPpack block storage flag must be 0 (dense) or 1 (sparse), " *
        "got $sparse_flag",
    ))
    expected_storage === nothing || sparse_flag == expected_storage || throw(ArgumentError(
        "SDPpack mixes dense and sparse blocks: expected storage flag " *
        "$expected_storage but block $block_index has $sparse_flag",
    ))
    if sparse_flag == 1
        nonzeros = _sdpa_parse_int(tokens, cursor, "SDPpack block nonzero count")
        nonzeros >= 0 || throw(ArgumentError(
            "SDPpack block nonzero count must be nonnegative",
        ))
        for entry in 1:nonzeros
            row = _sdpa_parse_int(tokens, cursor, "SDPpack row[$entry]")
            column = _sdpa_parse_int(tokens, cursor, "SDPpack column[$entry]")
            1 <= row <= dimension || throw(ArgumentError(
                "SDPpack row $row is outside 1:$dimension for block $block_index",
            ))
            1 <= column <= dimension || throw(ArgumentError(
                "SDPpack column $column is outside 1:$dimension for block $block_index",
            ))
            value = _sdpa_parse_scalar(
                tokens,
                cursor,
                T,
                "SDPpack coefficient[$entry]",
            )
            _sdpa_add_entry!(
                groups,
                matrix_index,
                block_index,
                row,
                column,
                value,
            )
        end
        return nothing
    end

    # SDPpack's dense payload is the lower triangle, column by column:
    # (1,1), (2,1), ..., (n,1), (2,2), ..., (n,n).
    for column in 1:dimension
        for row in column:dimension
            value = _sdpa_parse_scalar(
                tokens,
                cursor,
                T,
                "SDPpack dense coefficient[$row,$column]",
            )
            _sdpa_add_entry!(
                groups,
                matrix_index,
                block_index,
                row,
                column,
                value,
            )
        end
    end
    return nothing
end

"""Read C.s and A.s blocks in the pinned SDPLIB export layout.

SDPpack 0.9's historical `export.m` (the layout used by the public SDPLIB
`.sdp.gz` files) writes one `spblk` flag before every C block and every Aᵢ
block, with no global marker.  Its companion `import.m` contains an upstream
inconsistency: it attempts to consume global `sparseblocks` markers that
`export.m` never writes.  The benchmark contract is intentionally pinned to
the observed/export layout; any marker-bearing variant fails closed rather
than being guessed through backtracking.
"""
function _sdppack_read_sections(
    tokens::Vector{String},
    start_cursor::Int,
    ::Type{T},
    m::Int,
    block_sizes::Vector{Int},
) where {T}
    cursor = Ref(start_cursor)
    groups = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},T}}()
    nblocks = length(block_sizes)
    for block in 1:nblocks
        _sdppack_read_block!(
            tokens,
            cursor,
            T,
            groups,
            0,
            block,
            block_sizes[block],
        )
    end
    for matrix_index in 1:m
        for block in 1:nblocks
            _sdppack_read_block!(
                tokens,
                cursor,
                T,
                groups,
                matrix_index,
                block,
                block_sizes[block],
            )
        end
    end
    return groups, cursor[]
end

"""Consume SDPpack's unsupported q/l side channels and exact end-of-input."""
function _sdppack_read_tail(
    tokens::Vector{String},
    start_cursor::Int,
)
    cursor = Ref(start_cursor)
    nblocks_q = _sdpa_parse_int(tokens, cursor, "SDPpack quadratic block count")
    nblocks_q == 0 || throw(ArgumentError(
        "SDPLIB SDPpack quadratic-cone blocks are unsupported (count=$nblocks_q)",
    ))
    linear_dimension = _sdpa_parse_int(tokens, cursor, "SDPpack LP dimension")
    linear_dimension == 0 || throw(ArgumentError(
        "SDPLIB SDPpack linear side-channel is unsupported (dimension=$linear_dimension)",
    ))
    cursor[] > length(tokens) || throw(ArgumentError(
        "trailing tokens after SDPpack compact input",
    ))
    return cursor[]
end

"""Parse SDPLIB's historical SDPpack/Nemirovskii compact `.sdp` stream."""
function _parse_sdppack_compact(path::AbstractString, ::Type{T}) where {T}
    tokens = _sdpa_tokens(_sdpa_read_text(path))
    isempty(tokens) && throw(ArgumentError("SDPLIB SDPpack input is empty"))
    cursor = Ref(1)

    # SDPpack calls the primal equality count m and stores its right-hand side
    # b immediately after it.  The SDPX conversion applies the sign change in
    # `_sdpa_ingest(...; sign=-one(T))`, after all values are parsed exactly at
    # the requested precision.
    m = _sdpa_parse_int(tokens, cursor, "SDPpack constraint count m")
    m > 0 || throw(ArgumentError("SDPpack constraint count m must be positive"))
    c = T[]
    sizehint!(c, m)
    for variable in 1:m
        push!(c, _sdpa_parse_scalar(tokens, cursor, T, "SDPpack rhs[$variable]"))
    end

    nblocks = _sdpa_parse_int(tokens, cursor, "SDPpack PSD block count")
    nblocks > 0 || throw(ArgumentError(
        "SDPpack PSD block count must be positive for the SDPLIB loader",
    ))
    block_sizes = Int[]
    sizehint!(block_sizes, nblocks)
    for block in 1:nblocks
        dimension = _sdpa_parse_int(tokens, cursor, "SDPpack block_sizes[$block]")
        dimension > 0 || throw(ArgumentError(
            "SDPpack semidefinite block_sizes[$block] must be positive",
        ))
        push!(block_sizes, dimension)
    end
    payload_start = cursor[]

    groups, payload_end = _sdppack_read_sections(
        tokens,
        payload_start,
        T,
        m,
        block_sizes,
    )
    _sdppack_read_tail(tokens, payload_end)
    return (
        m=m,
        c=c,
        block_sizes=block_sizes,
        groups=groups,
        storage_layout=(global_c=false, global_a=false),
    )
end

"""Build expanded positive PSD blocks from a parsed sparse SDPA instance."""
function _sdpa_ingest(
    parsed,
    ::Type{T},
    sign::T=one(T),
) where {T}
    m = parsed.m
    block_sizes = parsed.block_sizes
    groups = parsed.groups

    C = Matrix{T}[]
    # The abstract element type accepts both ordinary sparse coefficient
    # vectors and SDPX's compact active-variable representation.  The latter
    # avoids an m-entry vector of empty matrices for large public instances.
    A = Vector{AbstractVector{<:AbstractMatrix}}()

    for (source_block, signed_dimension) in pairs(block_sizes)
        dimension = abs(signed_dimension)
        if signed_dimension > 0
            push!(C, _sdpa_dense_symmetric(
                T,
                dimension,
                get(groups, (0, source_block), nothing),
                sign,
            ))
            active_variables = Int[]
            coefficients = SparseMatrixCSC{T,Int}[]
            for variable in 1:m
                group = get(groups, (variable, source_block), nothing)
                group === nothing && continue
                matrix = _sdpa_sparse_matrix(T, dimension, group, sign)
                nnz(matrix) == 0 && continue
                push!(active_variables, variable)
                push!(coefficients, matrix)
            end
            push!(A, SDPX.ActiveSparseCoefficientVector(
                T,
                m,
                active_variables,
                coefficients,
                dimension,
            ))
        else
            # A negative block is an LP/diagonal cone.  SDPA uses one
            # diagonal coordinate per scalar inequality; preserving the
            # original order while splitting it gives an equivalent product
            # of 1×1 PSD cones in SDPX.
            for coordinate in 1:dimension
                scalar_group = function (matrix_index)
                    group = get(groups, (matrix_index, source_block), nothing)
                    group === nothing && return nothing
                    value = get(group, (coordinate, coordinate), zero(T))
                    return iszero(value) ? nothing : Dict((1, 1) => value)
                end
                push!(C, _sdpa_dense_symmetric(T, 1, scalar_group(0), sign))
                active_variables = Int[]
                coefficients = SparseMatrixCSC{T,Int}[]
                for variable in 1:m
                    group = scalar_group(variable)
                    group === nothing && continue
                    push!(active_variables, variable)
                    push!(coefficients, _sdpa_sparse_matrix(T, 1, group, sign))
                end
                push!(A, SDPX.ActiveSparseCoefficientVector(
                    T,
                    m,
                    active_variables,
                    coefficients,
                    1,
                ))
            end
        end
    end

    # There are no equality constraints in sparse SDPA files.  A sparse
    # m×0 matrix keeps the canonical B layout while avoiding a dense zero
    # allocation for large m.
    B = spzeros(T, m, 0)
    b = T[]
    return SDPX.ingest(
        [sign * value for value in parsed.c],
        A,
        C,
        B,
        b;
        T=T,
        sparse=:sparse,
        validate=true,
        symmetrize=false,
        verbosity=0,
    )
end

"""Validate a benchmark external path and return its content digest."""
function _sdpa_validate_external(
    spec,
    path::AbstractString,
    checksum::AbstractString,
    accepted_loaders::Set{Symbol},
    expected_source::Symbol,
)
    loader = hasproperty(spec, :loader) ? spec.loader : nothing
    loader in accepted_loaders || throw(ArgumentError(
        "unsupported SDPA loader $(repr(loader)); expected one of " *
        repr(sort!(collect(accepted_loaders))),
    ))
    source = hasproperty(spec, :source) ? spec.source : nothing
    source === expected_source || throw(ArgumentError(
        "SDPA source must be :$expected_source for loader $(repr(loader)), " *
        "got $(repr(source))",
    ))

    actual = _sdpa_sha256_file(path)
    lowercase(String(checksum)) == actual || throw(ArgumentError(
        "external checksum changed between cache validation and load",
    ))
    if hasproperty(spec, :external) && spec.external !== nothing
        expected_checksum = spec.external.sha256
        expected_checksum === nothing || lowercase(String(expected_checksum)) == actual ||
            throw(ArgumentError("sparse SDPA checksum does not match registry"))
    end
    return actual
end

function _sdpa_reference_objective(spec, ::Type{T}) where {T}
    hasproperty(spec, :reference) || return nothing
    hasproperty(spec.reference, :objective) || return nothing
    objective = spec.reference.objective
    objective === nothing && return nothing
    normalized = replace(replace(string(objective), 'D' => 'E'), 'd' => 'e')
    return try
        parse(T, normalized)
    catch
        throw(ArgumentError(
            "invalid SDPA reference objective $(repr(objective)) for $T",
        ))
    end
end

"""Return the canonical benchmark tuple for a DIMACS `.dat-s` input."""
function _build_sdpa_sparse_problem(
    spec,
    ::Type{T},
    path::AbstractString,
    checksum::AbstractString,
) where {T}
    actual = _sdpa_validate_external(
        spec,
        path,
        checksum,
        _SDPA_DIMACS_LOADERS,
        :dimacs,
    )
    parsed = _parse_sdpa_sparse(path, :dimacs, T)
    problem = _sdpa_ingest(parsed, T, one(T))
    expected = _sdpa_reference_objective(spec, T)
    return (
        kind=:sdp,
        problem,
        expected=expected,
        external_checksum=actual,
    )
end

"""Return the canonical benchmark tuple for an SDPLIB SDPpack `.sdp` input."""
function _build_sdppack_compact_problem(
    spec,
    ::Type{T},
    path::AbstractString,
    checksum::AbstractString,
) where {T}
    actual = _sdpa_validate_external(
        spec,
        path,
        checksum,
        _SDPPACK_COMPACT_LOADERS,
        :sdplib,
    )
    parsed = _parse_sdppack_compact(path, T)
    # SDPpack's primal/dual pair is
    #   min F₀•X  s.t. Fᵢ•X=bᵢ, X⪰0
    #   max bᵀy  s.t. Σ yᵢFᵢ + Z = F₀, Z⪰0.
    # SDPX uses min cᵀx s.t. Σ Aᵢxᵢ − C⪰0, hence c=-b, Aᵢ=-Fᵢ,
    # C=-F₀.  Applying one exact-precision sign at ingest keeps the
    # formulation invariant for Float64, BigFloat, and MultiFloats.
    problem = _sdpa_ingest(parsed, T, -one(T))
    expected = _sdpa_reference_objective(spec, T)
    return (
        kind=:sdp,
        problem,
        expected=expected,
        external_checksum=actual,
    )
end
