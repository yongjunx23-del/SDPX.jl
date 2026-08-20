"""
Pure Julia reader for the compressed MPS files distributed by Netlib.

Netlib's small files are not gzip archives: they are the output of David
M. Gay's ``emps`` compressor.  The format is documented by the source of the
authoritative decoder (``https://www.netlib.org/lp/data/emps.c``).  This file
translates the two compact coding routines (`exindx` and `exform`) directly;
no executable decoder, shell, Python, or external solver is needed at run
time.  The resulting text is ordinary free-format MPS and is then parsed into
SDPX's native LP representation.

The parser deliberately fails closed.  Netlib's integer-marker extensions,
multiple RHS/range/bound sets, duplicate coefficients, malformed fields, and
unsupported row/bound types are rejected instead of being silently guessed.
"""

using SHA
using SparseArrays

const _NETLIB_MPS_LOADERS = Set{Symbol}((
    :external_netlib_compressed_mps,
    :netlib_compressed_mps,
    :external_netlib_mps,
    :netlib_mps,
))

const _NETLIB_TR = collect(codeunits(
    "!\"#\$%&'()*+,-./0123456789;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~",
))
const _NETLIB_INV_TR = let table = fill(92, 256)
    for (index, value) in enumerate(_NETLIB_TR)
        table[Int(value) + 1] = index - 1
    end
    table
end

_netlib_tr(value::UInt8) = begin
    index = Int(value) + 1
    index <= length(_NETLIB_INV_TR) || throw(ArgumentError(
        "Netlib compressed stream contains a non-ASCII byte",
    ))
    result = _NETLIB_INV_TR[index]
    result == 92 && throw(ArgumentError(
        "Netlib compressed stream contains an invalid code byte $(value)",
    ))
    result
end

mutable struct _NetlibEncodedCursor
    lines::Vector{String}
    next_line::Int
    current::String
    position::Int
    skip_checksums::Bool
end

function _netlib_cursor(text::AbstractString)
    return _NetlibEncodedCursor(
        split(String(text), '\n'; keepempty=true), 1, "", 1, false,
    )
end

function _netlib_fetch_line!(cursor::_NetlibEncodedCursor)
    while cursor.next_line <= length(cursor.lines)
        line = cursor.lines[cursor.next_line]
        cursor.next_line += 1
        # ``emps`` calls these mystery lines and skips them unless -m is set.
        startswith(line, ":") && continue
        # Every 72 input records `emps.c` emits a one-line checksum.  The
        # external SHA-256 check protects the cache itself; after the compact
        # header has been consumed these fixed-width checksum records are not
        # MPS data and must be skipped before decoding the next index.
        # The final checksum before ENDATA is shorter than 72 characters, so
        # use the compressor's invariant (encoded data never begins with a
        # literal space) rather than only the common fixed width.
        cursor.skip_checksums && startswith(line, " ") && continue
        cursor.current = line
        cursor.position = firstindex(line)
        return line
    end
    throw(ArgumentError("truncated Netlib compressed stream"))
end

function _netlib_ensure_char!(cursor::_NetlibEncodedCursor)
    while cursor.position > lastindex(cursor.current)
        _netlib_fetch_line!(cursor)
    end
    return nothing
end

function _netlib_take_char!(cursor::_NetlibEncodedCursor)
    _netlib_ensure_char!(cursor)
    character = codeunit(cursor.current, cursor.position)
    cursor.position = nextind(cursor.current, cursor.position)
    return character
end

function _netlib_take_line!(cursor::_NetlibEncodedCursor)
    # The caller has already consumed the current encoded line.  Do not fetch
    # here: the next decoder invocation must see the first line of the next
    # section, exactly as `emps.c`'s `rdline` call does.
    if cursor.position <= lastindex(cursor.current)
        throw(ArgumentError(
            "Netlib compressed stream has an unexpected trailing encoded field",
        ))
    end
    return nothing
end

function _netlib_remainder!(cursor::_NetlibEncodedCursor)
    # An empty remainder is meaningful in compressed RHS/RANGES records: a
    # marker line containing only the zero index denotes the default/unnamed
    # set.  Do not advance to the next line in that case.
    cursor.position > lastindex(cursor.current) && return ""
    start = cursor.position
    cursor.position = nextind(cursor.current, lastindex(cursor.current))
    # `nextind(s,lastindex(s))` is one past the last character for ordinary
    # strings.  Preserve an empty remainder without indexing it below.
    return start > lastindex(cursor.current) ? "" : cursor.current[start:end]
end

function _netlib_index_from_code!(cursor::_NetlibEncodedCursor, code::Int)
    code >= 23 && return code - 23
    value = code
    while true
        next = _netlib_tr(_netlib_take_char!(cursor))
        value = value * 46 + next
        next >= 46 && return value - 46
    end
end

"""Translate `emps.c`'s supersparse index routine."""
function _netlib_exindx!(cursor::_NetlibEncodedCursor)
    code = _netlib_tr(_netlib_take_char!(cursor))
    return _netlib_index_from_code!(cursor, code)
end

function _netlib_decimal_from_mantissa(mantissa::BigInt, exponent::Int)
    mantissa == 0 && return "0"
    sign = mantissa < 0 ? "-" : ""
    digits = string(abs(mantissa))
    if exponent >= 0
        return sign * digits * repeat("0", exponent)
    end
    split_at = length(digits) + exponent
    if split_at > 0
        return sign * digits[1:split_at] * "." * digits[(split_at + 1):end]
    elseif split_at == 0
        return sign * "0." * digits
    end
    return sign * "0." * repeat("0", -split_at) * digits
end

"""
Translate `emps.c`'s `exform` routine to a decimal token.

The returned token is intentionally a string.  `_parse_mps` parses it directly
into its requested arithmetic type, so BigFloat and MultiFloat inputs never
pass through Float64 (or a lower precision intermediate).
"""
function _netlib_exform!(
    cursor::_NetlibEncodedCursor,
    number_table::Vector{String},
)
    code = _netlib_tr(_netlib_take_char!(cursor))
    if code < 46
        index = _netlib_index_from_code!(cursor, code)
        1 <= index <= length(number_table) || throw(ArgumentError(
            "Netlib number-table reference $index is outside 1:$(length(number_table))",
        ))
        return number_table[index]
    end

    code -= 46
    negative = false
    nelim = 12
    if code >= 23
        negative = true
        code -= 23
        nelim = 11
    end

    # Integer floating-point branch from emps.c.  The source writes a trailing
    # decimal point; a plain decimal token is equivalent and easier to audit.
    if code >= 11
        code -= 11
        magnitude = if code >= 6
            BigInt(code - 6)
        else
            value = code
            while true
                next = _netlib_tr(_netlib_take_char!(cursor))
                value = value * 46 + next
                next >= 46 && break
            end
            BigInt(value - 46)
        end
        token = string(magnitude)
        return negative ? "-" * token : token
    end

    # General floating-point branch.  `emps.c` uses a machine `long` and a
    # spill variable to avoid overflow while accumulating base-92 digits.  A
    # BigInt is exact for the same operation and keeps all source digits.
    exponent = _netlib_tr(_netlib_take_char!(cursor)) - 50
    magnitude = BigInt(_netlib_tr(_netlib_take_char!(cursor)))
    remaining = code
    while remaining > 0
        magnitude = magnitude * 92 + _netlib_tr(_netlib_take_char!(cursor))
        remaining -= 1
    end
    token = _netlib_decimal_from_mantissa(magnitude, exponent)
    negative && !startswith(token, "-") && (token = "-" * token)
    return token
end

function _netlib_parse_stats(line::AbstractString, count::Int, role::AbstractString)
    tokens = split(strip(line))
    length(tokens) == count || throw(ArgumentError(
        "Netlib $role statistics require $count integers, got $(length(tokens))",
    ))
    values = Int[]
    for token in tokens
        value = tryparse(Int, token)
        value === nothing && throw(ArgumentError(
            "Netlib $role statistics contain invalid integer $(repr(token))",
        ))
        value >= 0 || throw(ArgumentError(
            "Netlib $role statistics cannot be negative",
        ))
        push!(values, value)
    end
    return values
end

function _netlib_emit(io::IO, fields...)
    println(io, "    ", join(string.(fields), "  "))
end

function _netlib_next_semantic_line!(cursor::_NetlibEncodedCursor)
    line = if cursor.position <= lastindex(cursor.current)
        throw(ArgumentError("Netlib parser attempted to skip a partial line"))
    else
        _netlib_fetch_line!(cursor)
    end
    while isempty(strip(line)) || startswith(line, ":")
        line = _netlib_fetch_line!(cursor)
    end
    # Semantic records are line-oriented; leave the encoded cursor positioned
    # at the end so the following call advances to the next record.
    cursor.position = nextind(line, lastindex(line))
    return line
end

"""Decode one official Netlib compressed-MPS text into ordinary free MPS."""
function _decode_netlib_compressed_mps_text(text::AbstractString)
    cursor = _netlib_cursor(text)
    first_line = _netlib_next_semantic_line!(cursor)
    startswith(first_line, "NAME") || throw(ArgumentError(
        "Netlib compressed input must begin with a NAME record",
    ))
    stats = _netlib_parse_stats(
        _netlib_next_semantic_line!(cursor), 8, "primary",
    )
    bounds_stats = _netlib_parse_stats(
        _netlib_next_semantic_line!(cursor), 3, "bound",
    )
    cursor.skip_checksums = true
    nrow, ncol, _, nz, nrhs, rhsnz, nran, ranz = stats
    nbd, bdnz, number_count = bounds_stats
    nrow >= 1 && ncol >= 1 || throw(ArgumentError(
        "Netlib compressed dimensions must be positive",
    ))

    table = String[]
    sizehint!(table, number_count)
    for _ in 1:number_count
        push!(table, _netlib_exform!(cursor, table))
    end

    row_types = Vector{Char}(undef, nrow)
    row_names = Vector{String}(undef, nrow)
    row_seen = Set{String}()
    for index in 1:nrow
        line = _netlib_next_semantic_line!(cursor)
        length(line) >= 2 || throw(ArgumentError("malformed Netlib ROWS record"))
        row_type = first(strip(line))
        row_name = strip(line[nextind(line, firstindex(line)):end])
        row_type in ('N', 'E', 'L', 'G') || throw(ArgumentError(
            "unsupported Netlib row type $(repr(row_type))",
        ))
        !isempty(row_name) || throw(ArgumentError("empty Netlib row name"))
        row_name in row_seen && throw(ArgumentError("duplicate Netlib row name $row_name"))
        push!(row_seen, row_name)
        row_types[index] = row_type
        row_names[index] = row_name
    end
    row_index = Dict(name => index for (index, name) in enumerate(row_names))

    columns = Tuple{String,String,String}[]
    rhs = Tuple{String,String,String}[]
    ranges = Tuple{String,String,String}[]
    bounds = Tuple{Int,String,String,Union{Nothing,String}}[]
    column_names = String[]
    column_seen = Set{String}()
    function decode_pairs!(target, count::Int, what::Int)
        current_column = ""
        for _ in 1:count
            index = _netlib_exindx!(cursor)
            if what == 4
                # A zero marker carries the bound-set name.  The next
                # nonzero code is the type index (1=UP,...,6=PL).
                while index == 0
                    current_column = strip(_netlib_remainder!(cursor))
                    isempty(current_column) && throw(ArgumentError(
                        "Netlib compressed bound-set name is empty",
                    ))
                    index = _netlib_exindx!(cursor)
                end
            else
                # Column starts are zero indices and do not count toward the
                # compressed nonzero count.  `emps.c` consumes them in an
                # inner loop before processing the next row/value pair.
                while index == 0
                    current_column = strip(_netlib_remainder!(cursor))
                    what == 1 && isempty(current_column) && throw(ArgumentError(
                        "Netlib compressed column name is empty",
                    ))
                    if what == 1 && !(current_column in column_seen)
                        push!(column_seen, current_column)
                        push!(column_names, current_column)
                    elseif what == 1 && current_column in column_seen
                        throw(ArgumentError("duplicate Netlib column start $current_column"))
                    end
                    index = _netlib_exindx!(cursor)
                end
            end
            if what == 4
                1 <= index <= 6 || throw(ArgumentError("invalid Netlib bound type index $index"))
                col_index = _netlib_exindx!(cursor)
                1 <= col_index <= ncol || throw(ArgumentError(
                    "Netlib bound column index $col_index is outside 1:$ncol",
                ))
                # The historical decoder omits values only for MI and PL
                # (encoded indices 5 and 6); FR may carry a legacy value.
                value = index <= 4 ? _netlib_exform!(cursor, table) : nothing
                push!(target, (index, string(col_index), current_column, value))
                continue
            end
            index <= nrow || throw(ArgumentError(
                "Netlib row index $index is outside 1:$nrow",
            ))
            what == 1 && isempty(current_column) && throw(ArgumentError(
                "Netlib coefficient appears before a column name",
            ))
            value = _netlib_exform!(cursor, table)
            row = row_names[index]
            if what == 1
                push!(target, (current_column, row, value))
            elseif what == 2
                push!(target, ("", row, value))
            else
                push!(target, ("", row, value))
            end
        end
    end

    decode_pairs!(columns, nz, 1)
    _netlib_take_line!(cursor) # section boundary after COLUMNS
    decode_pairs!(rhs, rhsnz, 2)
    nrhs == 0 && nothing
    _netlib_take_line!(cursor)
    decode_pairs!(ranges, ranz, 3)
    _netlib_take_line!(cursor)
    # Bound records carry a type index, column index, and (for UP/LO/FX) value.
    decode_pairs!(bounds, bdnz, 4)

    io = IOBuffer()
    println(io, first_line)
    println(io, "ROWS")
    for (row_type, row_name) in zip(row_types, row_names)
        _netlib_emit(io, row_type, row_name)
    end
    println(io, "COLUMNS")
    for (column, row, value) in columns
        _netlib_emit(io, column, row, value)
    end
    println(io, "RHS")
    for (_, row, value) in rhs
        _netlib_emit(io, "RHS1", row, value)
    end
    println(io, "RANGES")
    for (_, row, value) in ranges
        _netlib_emit(io, "RNG1", row, value)
    end
    println(io, "BOUNDS")
    bound_types = ("UP", "LO", "FX", "FR", "MI", "PL")
    length(column_names) == ncol || throw(ArgumentError(
        "Netlib compressed column count $(length(column_names)) does not match header $ncol",
    ))
    for (type_index, column_index, bound_set_name, value) in bounds
        type_index <= length(bound_types) || throw(ArgumentError(
            "unsupported Netlib bound type index $type_index",
        ))
        type = bound_types[type_index]
        column = column_names[parse(Int, column_index)]
        if value === nothing
            _netlib_emit(io, type, bound_set_name, column)
        else
            _netlib_emit(io, type, bound_set_name, column, value)
        end
    end
    println(io, "ENDATA")
    return String(take!(io))
end

"""Decode a path (or an already loaded compressed text) from Netlib."""
function _decode_netlib_compressed_mps(source::AbstractString)
    text = isfile(source) ? read(source, String) : String(source)
    # A plain MPS fixture is already decoded; accepting it makes the loader
    # useful for local tests and for verified mirrors that publish MPS directly.
    first_line = begin
        lines = split(text, '\n'; keepempty=true)
        index = findfirst(line -> !isempty(strip(line)) && !startswith(strip(line), "*"), lines)
        index === nothing ? "" : strip(lines[index])
    end
    startswith(first_line, "NAME") || throw(ArgumentError("Netlib input has no NAME record"))
    plain_rows = any(strip(line) == "ROWS" for line in split(text, '\n'; keepempty=true))
    return plain_rows ? String(text) : _decode_netlib_compressed_mps_text(text)
end

function _netlib_parse_scalar(::Type{T}, source::AbstractString, role::AbstractString) where {T}
    normalized = replace(replace(strip(source), 'D' => 'E'), 'd' => 'e')
    isempty(normalized) && throw(ArgumentError("empty Netlib numeric token for $role"))
    value = try
        parse(T, normalized)
    catch error
        throw(ArgumentError("invalid Netlib numeric token $(repr(source)) for $role: $error"))
    end
    isfinite(value) || throw(ArgumentError("non-finite Netlib numeric token $(repr(source))"))
    return value
end

function _mps_fields(line::AbstractString)
    stripped = strip(line)
    isempty(stripped) && return String[]
    startswith(stripped, "*") && return String[]
    return split(stripped)
end

"""Parse ordinary MPS text directly into typed sparse LP arrays."""
function _parse_mps(text::AbstractString, ::Type{T}) where {T}
    section = :none
    row_types = Dict{String,Char}()
    row_order = String[]
    objective_row = nothing
    column_entries = Tuple{String,String,T}[]
    rhs_entries = Tuple{String,String,T}[]
    range_entries = Tuple{String,String,T}[]
    bound_entries = Tuple{String,String,String,Union{Nothing,T}}[]
    column_order = String[]
    column_seen = Set{String}()
    seen_coefficients = Set{Tuple{String,String}}()
    rhs_set = nothing
    range_set = nothing
    bound_set = nothing
    seen_bounds = Set{Tuple{String,String}}()
    ended = false

    for raw_line in split(String(text), '\n'; keepempty=true)
        fields = _mps_fields(raw_line)
        isempty(fields) && continue
        keyword = uppercase(fields[1])
        if keyword in ("NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA")
            if keyword == "ENDATA"
                ended = true
                section = :ended
                continue
            end
            ended && throw(ArgumentError("MPS record appears after ENDATA"))
            if keyword == "NAME"
                section = :name
            elseif keyword == "ROWS"
                section = :rows
            elseif keyword == "COLUMNS"
                section = :columns
            elseif keyword == "RHS"
                section = :rhs
            elseif keyword == "RANGES"
                section = :ranges
            elseif keyword == "BOUNDS"
                section = :bounds
            end
            continue
        end
        ended && throw(ArgumentError("MPS record appears after ENDATA"))
        section == :none && throw(ArgumentError("MPS data record appears before a section header"))
        if section == :name
            continue
        elseif section == :rows
            length(fields) == 2 || throw(ArgumentError("malformed MPS ROWS record"))
            type = uppercase(fields[1])
            type in ("N", "E", "L", "G") || throw(ArgumentError("unsupported MPS row type $(fields[1])"))
            name = fields[2]
            haskey(row_types, name) && throw(ArgumentError("duplicate MPS row $name"))
            row_types[name] = first(type)
            push!(row_order, name)
            if first(type) == 'N'
                objective_row === nothing || throw(ArgumentError("multiple MPS objective rows"))
                objective_row = name
            end
        elseif section == :columns
            any(token -> uppercase(token) in ("MARKER", "'MARKER'", "INTORG", "INTEND"), fields) &&
                throw(ArgumentError("MPS integer markers are unsupported"))
            length(fields) in (3, 5) || throw(ArgumentError("malformed MPS COLUMNS record"))
            column = fields[1]
            column in column_seen || (push!(column_seen, column); push!(column_order, column))
            pairs = length(fields) == 3 ? (2,) : (2, 4)
            for start in pairs
                row = fields[start]
                haskey(row_types, row) || throw(ArgumentError("MPS COLUMNS references unknown row $row"))
                key = (column, row)
                key in seen_coefficients && throw(ArgumentError("duplicate MPS coefficient $column/$row"))
                push!(seen_coefficients, key)
                value = _netlib_parse_scalar(T, fields[start + 1], "COLUMNS $column/$row")
                push!(column_entries, (column, row, value))
            end
        elseif section == :rhs || section == :ranges
            # `emps.c` preserves an empty RHS/RANGES set name as a blank
            # fixed-format field, which appears as 2/4 tokens after `split`.
            # Accept both that representation and the usual named 3/5-token
            # free-format form.
            length(fields) in (2, 3, 4, 5) || throw(ArgumentError("malformed MPS $(section) record"))
            named_set = length(fields) in (3, 5)
            set_name = named_set ? fields[1] : ""
            target = section == :rhs ? rhs_set : range_set
            if target === nothing
                section == :rhs ? (rhs_set = set_name) : (range_set = set_name)
            elseif target != set_name
                throw(ArgumentError("multiple conflicting MPS $(section) sets"))
            end
            destination = section == :rhs ? rhs_entries : range_entries
            seen = Set{String}(entry[2] for entry in destination)
            pairs = named_set ? (2, 4)[1:(length(fields) == 3 ? 1 : 2)] :
                     (1, 3)[1:(length(fields) == 2 ? 1 : 2)]
            for start in pairs
                row = fields[start]
                haskey(row_types, row) || throw(ArgumentError("MPS $(section) references unknown row $row"))
                row in seen && throw(ArgumentError("duplicate MPS $(section) entry for $row"))
                push!(seen, row)
                value = _netlib_parse_scalar(T, fields[start + 1], "$(section) $row")
                push!(destination, (set_name, row, value))
            end
        elseif section == :bounds
            length(fields) in (3, 4) || throw(ArgumentError("malformed MPS BOUNDS record"))
            type = uppercase(fields[1])
            type in ("LO", "UP", "FX", "FR", "MI", "PL") || throw(ArgumentError("unsupported MPS bound type $type"))
            set_name = fields[2]
            if bound_set === nothing
                bound_set = set_name
            elseif bound_set != set_name
                throw(ArgumentError("multiple conflicting MPS BOUNDS sets"))
            end
            column = fields[3]
            column in column_seen || (push!(column_seen, column); push!(column_order, column))
            bound_key = (type, column)
            bound_key in seen_bounds && throw(ArgumentError(
                "duplicate MPS $type bound for $column",
            ))
            push!(seen_bounds, bound_key)
            has_value = length(fields) == 4
            type in ("LO", "UP", "FX") && !has_value && throw(ArgumentError("MPS $type bound requires a value"))
            type in ("MI", "PL") && has_value && throw(ArgumentError("MPS $type bound must not carry a value"))
            # `emps.c` historically emits a numeric field for FR records;
            # accept and ignore that harmless value while retaining strict
            # arity for MI/PL.
            value = has_value ? _netlib_parse_scalar(T, fields[4], "bound $column") : nothing
            push!(bound_entries, (type, set_name, column, value))
        else
            throw(ArgumentError("MPS data record appears outside a known section"))
        end
    end
    ended || throw(ArgumentError("MPS input is missing ENDATA"))
    objective_row === nothing && throw(ArgumentError("MPS input has no objective row"))
    isempty(column_order) && throw(ArgumentError("MPS input has no columns"))

    n = length(column_order)
    variable_index = Dict(name => index for (index, name) in enumerate(column_order))
    c = zeros(T, n)
    coefficient_maps = [Dict{Int,T}() for _ in row_order]
    row_index = Dict(name => index for (index, name) in enumerate(row_order))
    for (column, row, value) in column_entries
        j = variable_index[column]
        if row == objective_row
            c[j] += value
        else
            i = row_index[row]
            haskey(coefficient_maps[i], j) && throw(ArgumentError("duplicate MPS coefficient"))
            coefficient_maps[i][j] = value
        end
    end

    rhs_map = Dict{String,T}(entry[2] => entry[3] for entry in rhs_entries)
    range_map = Dict{String,T}(entry[2] => entry[3] for entry in range_entries)
    haskey(range_map, objective_row) && throw(ArgumentError(
        "MPS objective row cannot carry a RANGES entry",
    ))
    objective_constant = get(rhs_map, objective_row, zero(T))
    lower = fill(zero(T), n)
    upper = fill(T(Inf), n)
    for (type, _, column, value) in bound_entries
        j = variable_index[column]
        if type == "LO"
            value === nothing && throw(ArgumentError("LO bound missing value")); lower[j] = value
        elseif type == "UP"
            value === nothing && throw(ArgumentError("UP bound missing value")); upper[j] = value
        elseif type == "FX"
            value === nothing && throw(ArgumentError("FX bound missing value")); lower[j] = value; upper[j] = value
        elseif type == "FR"
            lower[j] = T(-Inf); upper[j] = T(Inf)
        elseif type == "MI"
            lower[j] = T(-Inf)
        elseif type == "PL"
            upper[j] = T(Inf)
        end
        lower[j] <= upper[j] || throw(ArgumentError("inconsistent MPS bounds for $(column)"))
    end

    equalities = Tuple{Int,Dict{Int,T},T}[]
    inequalities = Tuple{Dict{Int,T},T}[]
    function add_lower!(map, bound)
        push!(inequalities, (map, bound))
    end
    for (i, row) in enumerate(row_order)
        row == objective_row && continue
        type = row_types[row]
        b = get(rhs_map, row, zero(T))
        has_range = haskey(range_map, row)
        if type == 'E' && !has_range
            push!(equalities, (i, coefficient_maps[i], b))
        else
            low = nothing
            high = nothing
            if type == 'E'
                r = get(range_map, row, zero(T)); low = r >= 0 ? b : b + r; high = r >= 0 ? b + r : b
            elseif type == 'L'
                if has_range
                    r = abs(range_map[row]); low = b - r; high = b
                else
                    high = b
                end
            elseif type == 'G'
                if has_range
                    r = abs(range_map[row]); low = b; high = b + r
                else
                    low = b
                end
            end
            low !== nothing && add_lower!(coefficient_maps[i], low)
            high !== nothing && add_lower!(Dict(j => -v for (j, v) in coefficient_maps[i]), -high)
        end
    end
    for j in 1:n
        isfinite(lower[j]) && add_lower!(Dict(j => one(T)), lower[j])
        isfinite(upper[j]) && add_lower!(Dict(j => -one(T)), -upper[j])
    end
    isempty(inequalities) && add_lower!(Dict{Int,T}(), -one(T))

    rows = length(inequalities)
    gi = Int[]; gj = Int[]; gv = T[]; h = zeros(T, rows)
    for (row, (map, bound)) in enumerate(inequalities)
        h[row] = bound
        for (j, value) in map
            iszero(value) && continue
            push!(gi, row); push!(gj, j); push!(gv, value)
        end
    end
    G = sparse(gi, gj, gv, rows, n)
    ei = Int[]; ej = Int[]; ev = T[]; beq = zeros(T, length(equalities))
    for (row, (i, map, bound)) in enumerate(equalities)
        beq[row] = bound
        for (j, value) in map
            iszero(value) && continue
            push!(ei, row); push!(ej, j); push!(ev, value)
        end
    end
    Aeq = sparse(ei, ej, ev, length(equalities), n)
    return (c=c, G=G, h=h, Aeq=Aeq, beq=beq, objective_constant=objective_constant,
            variables=n, rows=length(row_order), nonzeros=length(column_entries),
            objective_row=objective_row)
end

"""Build a typed SDPX LP from one checked Netlib cache entry."""
function _build_netlib_mps_problem(
    spec,
    ::Type{T},
    path::AbstractString,
    checksum::AbstractString,
) where {T}
    hasproperty(spec, :loader) && spec.loader in _NETLIB_MPS_LOADERS ||
        throw(ArgumentError("unsupported Netlib loader $(repr(getproperty(spec, :loader)))"))
    actual = open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
    lowercase(String(checksum)) == actual || throw(ArgumentError("external checksum changed between cache validation and load"))
    if hasproperty(spec, :external) && spec.external !== nothing && spec.external.sha256 !== nothing
        lowercase(String(spec.external.sha256)) == actual || throw(ArgumentError("Netlib checksum does not match registry"))
    end
    parsed = _parse_mps(_decode_netlib_compressed_mps(path), T)
    expected = hasproperty(spec, :reference) && spec.reference.objective === nothing ? nothing :
               _netlib_parse_scalar(T, string(spec.reference.objective), "reference objective")
    problem = SDPX.linear_program(
        parsed.c, parsed.G, parsed.h;
        Aeq=parsed.Aeq, beq=parsed.beq, T=T, sparse=true, verbosity=0,
    )
    return (
        kind=:sdp,
        problem=problem,
        expected=expected,
        physical_objective=x -> dot(parsed.c, x) + parsed.objective_constant,
        objective_constant=parsed.objective_constant,
        external_checksum=actual,
        source_parameters=(rows=parsed.rows, columns=parsed.variables,
                           nonzeros=parsed.nonzeros),
    )
end
