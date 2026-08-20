"""
Pure Julia reader for the scalar part of the Conic Benchmark Format (CBF).

CBF is an affine conic format.  SDPX's native SOCP frontend stores free
variables, so variable domains in `VAR` are represented as additional native
Lorentz/equality constraints.  This is an exact reformulation for the
supported CBF cones (`F`, `L+`, `L-`, `L=`, and `Q`); all other cones and every
PSD-variable/PSD-constraint construct are rejected explicitly.  Numeric
tokens are parsed directly as the requested `T`, including BigFloat and
MultiFloats, and are never staged through Float64.
"""

using SparseArrays
using SHA
using LinearAlgebra

const _CBF_LOADERS = Set{Symbol}((
    :cbf,
    :cbf_gzip,
    :external_cbf,
    :external_cbf_gzip,
))

const _CBF_LINEAR_SDPX_CONES = Set{String}(("F", "L+", "L-", "L="))

"""Read a CBF file as UTF-8 text; compressed input is decoded by gzip."""
function _cbf_read_text(path::AbstractString)
    isfile(path) || throw(ArgumentError("CBF input does not exist: $path"))
    filename = String(path)
    if endswith(lowercase(filename), ".gz")
        return read(Cmd(["gzip", "-dc", "--", filename]), String)
    end
    return read(filename, String)
end

_cbf_sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

"""
Retain full-line CBF comments as block separators.

CBF comments are only permitted when an information item is not being read;
in particular, an inline `#` is not a comment delimiter.  Keeping comment
lines here lets the top-level item loop skip them while `_cbf_expect_tokens`
can reject a comment encountered inside a keyword body.
"""
function _cbf_lines(text::AbstractString)
    lines = String[]
    for raw in split(String(text), '\n'; keepempty=true)
        isempty(strip(raw)) && continue
        # The CBF grammar permits a comment only when `#` is the first byte
        # of the physical line.  Leading whitespace before `#` is therefore
        # rejected as an inline/commented data line below.
        if startswith(raw, "#")
            push!(lines, strip(raw))
            continue
        end
        occursin('#', raw) && throw(ArgumentError(
            "inline CBF comments are not permitted; comments must occupy " *
            "an entire line between information items",
        ))
        line = strip(raw)
        push!(lines, line)
    end
    return lines
end

function _cbf_expect_tokens(
    lines::Vector{String},
    cursor::Base.RefValue{Int},
    count::Int,
    role::AbstractString,
)
    cursor[] <= length(lines) || throw(ArgumentError(
        "truncated CBF input while reading $role",
    ))
    line = lines[cursor[]]
    startswith(line, "#") && throw(ArgumentError(
        "CBF comment encountered inside information item $role; " *
        "comments are only permitted between information items",
    ))
    fields = split(line)
    cursor[] += 1
    length(fields) == count || throw(ArgumentError(
        "CBF $role expects $count fields, got $(length(fields)) on line " *
        repr(line),
    ))
    return fields
end

function _cbf_int(
    lines::Vector{String},
    cursor::Base.RefValue{Int},
    role::AbstractString,
)
    fields = _cbf_expect_tokens(lines, cursor, 1, role)
    try
        return parse(Int, fields[1])
    catch
        throw(ArgumentError("invalid integer token $(repr(fields[1])) for CBF $role"))
    end
end

function _cbf_header_ints(
    lines::Vector{String},
    cursor::Base.RefValue{Int},
    count::Int,
    role::AbstractString,
)
    fields = _cbf_expect_tokens(lines, cursor, count, role)
    values = Int[]
    for (position, token) in pairs(fields)
        try
            push!(values, parse(Int, token))
        catch
            throw(ArgumentError(
                "invalid integer token $(repr(token)) for CBF $role[$position]",
            ))
        end
    end
    return values
end

function _cbf_scalar(
    token::AbstractString,
    ::Type{T},
    role::AbstractString,
) where {T}
    normalized = replace(replace(String(token), 'D' => 'E'), 'd' => 'e')
    value = try
        parse(T, normalized)
    catch
        throw(ArgumentError(
            "invalid numeric token $(repr(token)) for CBF $role",
        ))
    end
    isfinite(value) || throw(ArgumentError(
        "non-finite numeric token $(repr(token)) for CBF $role",
    ))
    return value
end

function _cbf_coordinate_index(token::AbstractString, upper::Int, role::AbstractString)
    index = try
        parse(Int, token)
    catch
        throw(ArgumentError("invalid index token $(repr(token)) for CBF $role"))
    end
    0 <= index < upper || throw(ArgumentError(
        "CBF $role index $index is outside 0:$(upper - 1)",
    ))
    return index + 1
end

function _cbf_validate_domain(domain::AbstractString, dimension::Int, role)
    dimension > 0 || throw(ArgumentError(
        "CBF $role cone dimension must be positive, got $dimension",
    ))
    domain in _CBF_LINEAR_SDPX_CONES || domain == "Q" || throw(ArgumentError(
        "unsupported CBF $role cone $(repr(domain)); supported cones are F, L+, L-, L=, Q",
    ))
    if domain == "Q"
        # Q¹ is the nonnegative ray and is valid in the CBF specification.
        dimension >= 1 || throw(ArgumentError("CBF Q cone dimension must be positive"))
    end
    return nothing
end

"""
    _parse_cbf(path, T) -> NamedTuple

Parse CBF v1--v4 scalar structure and data.  The reader accepts the older
scalar syntax because CBLIB's canonical continuous SOCP collection predates
CBF v4; v4 syntax is identical for the supported cones.  Newer versions and
unknown keywords fail closed.
"""
function _parse_cbf(path::AbstractString, ::Type{T}) where {T}
    lines = _cbf_lines(_cbf_read_text(path))
    isempty(lines) && throw(ArgumentError("CBF input is empty"))
    cursor = Ref(1)
    while cursor[] <= length(lines) && startswith(lines[cursor[]], "#")
        cursor[] += 1
    end
    first_line = _cbf_expect_tokens(lines, cursor, 1, "first keyword")
    first_line[1] == "VER" || throw(ArgumentError(
        "CBF VER must be the first keyword, got $(repr(first_line[1]))",
    ))
    version = _cbf_int(lines, cursor, "version")
    1 <= version <= 4 || throw(ArgumentError(
        "unsupported CBF version $version; expected a version in 1:4",
    ))

    seen = Set{String}(("VER",))
    objective_sense = nothing
    nvars = nothing
    variable_domains = Tuple{String,Int}[]
    integer_indices = Int[]
    ncons = nothing
    constraint_domains = Tuple{String,Int}[]
    objective = T[]
    objective_constant = zero(T)
    objective_coords = Dict{Int,T}()
    objective_constant_seen = false
    Acoords = Dict{Tuple{Int,Int},T}()
    bcoords = Dict{Int,T}()
    power_sections_seen = Set{String}()

    while cursor[] <= length(lines)
        while cursor[] <= length(lines) && startswith(lines[cursor[]], "#")
            cursor[] += 1
        end
        cursor[] > length(lines) && break
        keyword_fields = _cbf_expect_tokens(lines, cursor, 1, "keyword")
        keyword = keyword_fields[1]
        keyword in seen && throw(ArgumentError("duplicate CBF keyword $keyword"))

        if keyword == "POWCONES" || keyword == "POW*CONES"
            push!(seen, keyword)
            header = _cbf_header_ints(lines, cursor, 2, keyword * " header")
            count, total = header
            count >= 0 && total >= 0 || throw(ArgumentError(
                "CBF $keyword header must be nonnegative",
            ))
            # Even an empty section is harmless; any non-empty table would
            # imply power-cone variables/constraints unsupported by SDPX.
            count == 0 && total == 0 || throw(ArgumentError(
                "CBF power cones are unsupported by the native SDPX loader",
            ))
            push!(power_sections_seen, keyword)
            continue
        elseif keyword == "OBJSENSE"
            push!(seen, keyword)
            fields = _cbf_expect_tokens(lines, cursor, 1, "OBJSENSE")
            fields[1] in ("MIN", "MAX") || throw(ArgumentError(
                "CBF OBJSENSE must be MIN or MAX, got $(repr(fields[1]))",
            ))
            objective_sense = Symbol(lowercase(fields[1]))
            continue
        elseif keyword == "PSDVAR"
            push!(seen, keyword)
            count = _cbf_int(lines, cursor, "PSDVAR count")
            count >= 0 || throw(ArgumentError("CBF PSDVAR count must be nonnegative"))
            count == 0 || throw(ArgumentError(
                "PSD variables are unsupported by the native SDPX CBF loader",
            ))
            continue
        elseif keyword == "VAR"
            push!(seen, keyword)
            nvars === nothing || throw(ArgumentError("duplicate CBF VAR"))
            header = _cbf_header_ints(lines, cursor, 2, "VAR header")
            n, sections = header
            n > 0 || throw(ArgumentError("CBF VAR variable count must be positive"))
            sections >= 0 || throw(ArgumentError("CBF VAR section count must be nonnegative"))
            used = 0
            for section in 1:sections
                fields = _cbf_expect_tokens(lines, cursor, 2, "VAR[$section]")
                dimension = try
                    parse(Int, fields[2])
                catch
                    throw(ArgumentError("invalid CBF VAR dimension $(repr(fields[2]))"))
                end
                _cbf_validate_domain(fields[1], dimension, "VAR[$section]")
                used += dimension
                used <= n || throw(ArgumentError(
                    "CBF VAR cone dimensions exceed declared variable count $n",
                ))
                push!(variable_domains, (fields[1], dimension))
            end
            used == n || throw(ArgumentError(
                "CBF VAR cone dimensions sum to $used, expected $n",
            ))
            nvars = n
            objective = zeros(T, n)
            continue
        elseif keyword == "INT"
            push!(seen, keyword)
            nvars === nothing && throw(ArgumentError("CBF INT must follow VAR"))
            count = _cbf_int(lines, cursor, "INT count")
            count >= 0 || throw(ArgumentError("CBF INT count must be nonnegative"))
            for position in 1:count
                fields = _cbf_expect_tokens(lines, cursor, 1, "INT[$position]")
                index = _cbf_coordinate_index(fields[1], nvars, "INT")
                index in integer_indices && throw(ArgumentError(
                    "duplicate CBF INT index $(index - 1)",
                ))
                push!(integer_indices, index)
            end
            count == 0 || throw(ArgumentError(
                "integer variables are unsupported by the native SDPX CBF loader",
            ))
            continue
        elseif keyword == "PSDCON"
            push!(seen, keyword)
            count = _cbf_int(lines, cursor, "PSDCON count")
            count >= 0 || throw(ArgumentError("CBF PSDCON count must be nonnegative"))
            for position in 1:count
                dimension = _cbf_int(lines, cursor, "PSDCON[$position] dimension")
                dimension > 0 || throw(ArgumentError("CBF PSDCON dimensions must be positive"))
            end
            count == 0 || throw(ArgumentError(
                "PSD constraints are unsupported by the native SDPX CBF loader",
            ))
            continue
        elseif keyword == "CON"
            push!(seen, keyword)
            nvars === nothing && throw(ArgumentError("CBF CON must follow VAR"))
            ncons === nothing || throw(ArgumentError("duplicate CBF CON"))
            header = _cbf_header_ints(lines, cursor, 2, "CON header")
            m, sections = header
            m >= 0 || throw(ArgumentError("CBF CON row count must be nonnegative"))
            sections >= 0 || throw(ArgumentError("CBF CON section count must be nonnegative"))
            used = 0
            for section in 1:sections
                fields = _cbf_expect_tokens(lines, cursor, 2, "CON[$section]")
                dimension = try
                    parse(Int, fields[2])
                catch
                    throw(ArgumentError("invalid CBF CON dimension $(repr(fields[2]))"))
                end
                _cbf_validate_domain(fields[1], dimension, "CON[$section]")
                used += dimension
                used <= m || throw(ArgumentError(
                    "CBF CON cone dimensions exceed declared row count $m",
                ))
                push!(constraint_domains, (fields[1], dimension))
            end
            used == m || throw(ArgumentError(
                "CBF CON cone dimensions sum to $used, expected $m",
            ))
            ncons = m
            continue
        elseif keyword == "OBJFCOORD"
            push!(seen, keyword)
            count = _cbf_int(lines, cursor, "OBJFCOORD count")
            count >= 0 || throw(ArgumentError("CBF OBJFCOORD count must be nonnegative"))
            for position in 1:count
                _cbf_expect_tokens(lines, cursor, 4, "OBJFCOORD[$position]")
            end
            count == 0 || throw(ArgumentError(
                "PSD objective coordinates are unsupported by the native SDPX CBF loader",
            ))
            continue
        elseif keyword == "OBJACOORD"
            push!(seen, keyword)
            nvars === nothing && throw(ArgumentError("CBF OBJACOORD must follow VAR"))
            count = _cbf_int(lines, cursor, "OBJACOORD count")
            count >= 0 || throw(ArgumentError("CBF OBJACOORD count must be nonnegative"))
            for position in 1:count
                fields = _cbf_expect_tokens(lines, cursor, 2, "OBJACOORD[$position]")
                index = _cbf_coordinate_index(fields[1], nvars, "OBJACOORD")
                haskey(objective_coords, index) && throw(ArgumentError(
                    "duplicate CBF objective coordinate $(index - 1)",
                ))
                objective_coords[index] = _cbf_scalar(
                    fields[2], T, "OBJACOORD[$position]",
                )
            end
            continue
        elseif keyword == "OBJBCOORD"
            push!(seen, keyword)
            fields = _cbf_expect_tokens(lines, cursor, 1, "OBJBCOORD")
            objective_constant = _cbf_scalar(fields[1], T, "OBJBCOORD")
            objective_constant_seen = true
            continue
        elseif keyword == "FCOORD"
            push!(seen, keyword)
            count = _cbf_int(lines, cursor, "FCOORD count")
            count >= 0 || throw(ArgumentError("CBF FCOORD count must be nonnegative"))
            for position in 1:count
                _cbf_expect_tokens(lines, cursor, 5, "FCOORD[$position]")
            end
            count == 0 || throw(ArgumentError(
                "PSD scalar-constraint coordinates are unsupported by the native SDPX CBF loader",
            ))
            continue
        elseif keyword == "ACOORD"
            push!(seen, keyword)
            nvars === nothing && throw(ArgumentError("CBF ACOORD must follow VAR"))
            ncons === nothing && throw(ArgumentError("CBF ACOORD must follow CON"))
            count = _cbf_int(lines, cursor, "ACOORD count")
            count >= 0 || throw(ArgumentError("CBF ACOORD count must be nonnegative"))
            for position in 1:count
                fields = _cbf_expect_tokens(lines, cursor, 3, "ACOORD[$position]")
                row = _cbf_coordinate_index(fields[1], ncons, "ACOORD row")
                col = _cbf_coordinate_index(fields[2], nvars, "ACOORD column")
                key = (row, col)
                haskey(Acoords, key) && throw(ArgumentError(
                    "duplicate CBF ACOORD coordinate ($(row - 1),$(col - 1))",
                ))
                Acoords[key] = _cbf_scalar(fields[3], T, "ACOORD[$position]")
            end
            continue
        elseif keyword == "BCOORD"
            push!(seen, keyword)
            ncons === nothing && throw(ArgumentError("CBF BCOORD must follow CON"))
            count = _cbf_int(lines, cursor, "BCOORD count")
            count >= 0 || throw(ArgumentError("CBF BCOORD count must be nonnegative"))
            for position in 1:count
                fields = _cbf_expect_tokens(lines, cursor, 2, "BCOORD[$position]")
                row = _cbf_coordinate_index(fields[1], ncons, "BCOORD row")
                haskey(bcoords, row) && throw(ArgumentError(
                    "duplicate CBF BCOORD row $(row - 1)",
                ))
                bcoords[row] = _cbf_scalar(fields[2], T, "BCOORD[$position]")
            end
            continue
        elseif keyword == "HCOORD" || keyword == "DCOORD"
            push!(seen, keyword)
            count = _cbf_int(lines, cursor, "$keyword count")
            count >= 0 || throw(ArgumentError("CBF $keyword count must be nonnegative"))
            for position in 1:count
                _cbf_expect_tokens(lines, cursor, keyword == "HCOORD" ? 5 : 4, "$keyword[$position]")
            end
            count == 0 || throw(ArgumentError(
                "PSD-constraint coordinates are unsupported by the native SDPX CBF loader",
            ))
            continue
        else
            throw(ArgumentError("unsupported or unknown CBF keyword $(repr(keyword))"))
        end
    end

    objective_sense === nothing && throw(ArgumentError("CBF OBJSENSE is required"))
    nvars === nothing && throw(ArgumentError("CBF VAR is required"))
    ncons === nothing && throw(ArgumentError("CBF CON is required for native SDPX ingestion"))
    isempty(integer_indices) || throw(ArgumentError("integer variables are unsupported by the native SDPX CBF loader"))

    for (index, value) in objective_coords
        objective[index] = value
    end
    A = spzeros(T, ncons, nvars)
    for ((row, col), value) in Acoords
        A[row, col] = value
    end
    b = zeros(T, ncons)
    for (row, value) in bcoords
        b[row] = value
    end

    return (
        version=version,
        objective_sense=objective_sense,
        c=objective,
        objective_constant=objective_constant,
        objective_constant_seen=objective_constant_seen,
        variable_domains=variable_domains,
        constraint_domains=constraint_domains,
        integer_indices=integer_indices,
        A=A,
        b=b,
        nvars=nvars,
        ncons=ncons,
    )
end

function _cbf_row_matrix(A::SparseMatrixCSC{T,Int}, rows::UnitRange{Int}, nvars::Int) where {T}
    isempty(rows) && return spzeros(T, 0, nvars)
    return SparseMatrixCSC{T,Int}(A[rows, :])
end

function _cbf_variable_domain_constraints(
    cones::Vector{SDPX.SOCConstraint{T}},
    equality_row_indices::Vector{Int},
    equality_col_indices::Vector{Int},
    equality_values::Vector{T},
    equality_rhs::Vector{T},
    nvars::Int,
    domains,
) where {T}
    offset = 0
    for (domain, dimension) in domains
        indices = offset .+ (1:dimension)
        if domain == "F"
            # Free variables need no explicit constraint.
        elseif domain == "Q"
            rows = sparse(collect(1:dimension), collect(indices), ones(T, dimension), dimension, nvars)
            push!(cones, SDPX.SOCConstraint(rows, zeros(T, dimension); T=T))
        elseif domain == "L+" || domain == "L-"
            sign = domain == "L+" ? one(T) : -one(T)
            for index in indices
                row = spzeros(T, 1, nvars)
                row[1, index] = sign
                push!(cones, SDPX.SOCConstraint(row, T[zero(T)]; T=T))
            end
        elseif domain == "L="
            for index in indices
                equality_row = length(equality_rhs) + 1
                push!(equality_row_indices, equality_row)
                push!(equality_col_indices, index)
                push!(equality_values, one(T))
                push!(equality_rhs, zero(T))
            end
        else
            throw(ArgumentError("unsupported CBF variable cone $(repr(domain))"))
        end
        offset += dimension
    end
    offset == nvars || throw(ArgumentError("CBF variable domain dimensions do not match nvars"))
    return nothing
end

function _build_cbf_native_problem(parsed, ::Type{T}) where {T}
    nvars = parsed.nvars
    cones = SDPX.SOCConstraint{T}[]
    # Equality rows are accumulated directly as sparse triplets.  Keeping a
    # dense nvars-wide row here would make an L= block in nql30 materialise
    # roughly 2780×4501 BigFloat/Float64 entries before CSC conversion.
    equality_row_indices = Int[]
    equality_col_indices = Int[]
    equality_values = T[]
    equality_rhs = T[]
    _cbf_variable_domain_constraints(
        cones,
        equality_row_indices,
        equality_col_indices,
        equality_values,
        equality_rhs,
        nvars,
        parsed.variable_domains,
    )

    row_offset = 0
    for (domain, dimension) in parsed.constraint_domains
        rows = row_offset .+ (1:dimension)
        affine = _cbf_row_matrix(parsed.A, rows, nvars)
        offset = parsed.b[rows]
        if domain == "F"
            # An affine expression constrained to the free domain imposes no
            # restriction, but its coordinates are still validated above.
        elseif domain == "Q"
            push!(cones, SDPX.SOCConstraint(affine, Vector{T}(offset); T=T))
        elseif domain == "L+" || domain == "L-"
            sign = domain == "L+" ? one(T) : -one(T)
            for coordinate in 1:dimension
                row = sparse(reshape(sign .* affine[coordinate, :], 1, nvars))
                push!(cones, SDPX.SOCConstraint(row, T[sign * offset[coordinate]]; T=T))
            end
        elseif domain == "L="
            for coordinate in 1:dimension
                equality_row = length(equality_rhs) + 1
                columns, values = findnz(affine[coordinate, :])
                append!(equality_row_indices, fill(equality_row, length(columns)))
                append!(equality_col_indices, columns)
                append!(equality_values, values)
                push!(equality_rhs, -offset[coordinate])
            end
        else
            throw(ArgumentError("unsupported CBF constraint cone $(repr(domain))"))
        end
        row_offset += dimension
    end
    row_offset == parsed.ncons || throw(ArgumentError("CBF constraint domain dimensions do not match ncons"))

    # A model containing only free variables and no affine restrictions is
    # still representable.  NativeSOC requires at least one cone, so add a
    # tautological Q¹ expression (1 ∈ Q¹), which leaves the feasible set and
    # objective unchanged.
    isempty(cones) && push!(cones, SDPX.SOCConstraint(
        spzeros(T, 1, nvars), T[one(T)]; T=T,
    ))

    equality = isempty(equality_rhs) ?
        spzeros(T, 0, nvars) :
        sparse(
            equality_row_indices,
            equality_col_indices,
            equality_values,
            length(equality_rhs),
            nvars,
        )
    rhs = copy(equality_rhs)

    original_c = parsed.c
    transformed_c = parsed.objective_sense === :max ? -original_c : original_c
    problem = SDPX.second_order_program(
        transformed_c,
        cones;
        Aeq=equality,
        beq=rhs,
        T=T,
    )
    return problem, original_c
end

"""
    _cbf_reference_objective(spec, T)

Parse the optional registry/reference objective directly in the arithmetic
type used by the model.  CBF references are external metadata, so accepting a
malformed value as an untyped string would make benchmark comparisons
silently meaningless; parsing therefore fails closed.  In particular, this
path never stages a BigFloat/MultiFloat value through Float64.
"""
function _cbf_reference_objective(spec, ::Type{T}) where {T}
    hasproperty(spec, :reference) || return nothing
    reference = spec.reference
    hasproperty(reference, :objective) || return nothing
    raw = reference.objective
    raw === nothing && return nothing
    raw isa Missing && throw(ArgumentError(
        "CBF reference objective is missing",
    ))
    # `_cbf_scalar` normalizes CBF's Fortran `D` exponent and validates
    # finiteness while calling `parse(T, ...)` directly.
    return _cbf_scalar(string(raw), T, "reference objective")
end

function _cbf_solve_settings(spec)
    reference = hasproperty(spec, :reference) ? spec.reference : nothing
    absolute = reference !== nothing &&
               hasproperty(reference, :absolute_tolerance) ?
               Float64(reference.absolute_tolerance) : 1.0e-7
    relative = reference !== nothing &&
               hasproperty(reference, :relative_tolerance) ?
               Float64(reference.relative_tolerance) : 1.0e-7
    reference_gate = min(absolute, relative)
    tolerance = isfinite(reference_gate) && reference_gate > 0.0 ?
                reference_gate / 10.0 : 1.0e-8
    # Public CBF rows must be bounded even when invoked outside a scheduler.
    # One order tighter than the published objective gate keeps semantic and
    # certificate checks meaningful without turning a rounded public reference
    # into an unnecessarily expensive 1e-8 solve (nql30 is registered at 1e-5).
    return (
        tolerance=string(tolerance),
        maximum_iterations=100,
        max_time=60.0,
    )
end

"""
    _build_cbf_problem(spec, T, path, checksum)

Build a native SDPX SOCP model after revalidating the cache and registry
checksums.  The returned tuple follows the benchmark runner's external-loader
contract and exposes the original CBF objective through `physical_objective`.
"""
function _build_cbf_problem(
    spec,
    ::Type{T},
    path::AbstractString,
    checksum::AbstractString,
) where {T}
    loader = hasproperty(spec, :loader) ? spec.loader : nothing
    loader in _CBF_LOADERS || throw(ArgumentError(
        "unsupported CBF loader $(repr(loader))",
    ))
    actual = _cbf_sha256_file(path)
    lowercase(String(checksum)) == actual || throw(ArgumentError(
        "external CBF checksum changed between cache validation and load",
    ))
    if hasproperty(spec, :external) && spec.external !== nothing
        expected = spec.external.sha256
        expected === nothing || lowercase(String(expected)) == actual || throw(ArgumentError(
            "CBF checksum does not match registry",
        ))
    end
    parsed = _parse_cbf(path, T)
    problem, original_c = _build_cbf_native_problem(parsed, T)
    physical = x -> dot(original_c, x) + parsed.objective_constant
    expected = _cbf_reference_objective(spec, T)
    return (
        kind=:socp,
        problem,
        expected=expected,
        physical_objective=physical,
        objective_constant=parsed.objective_constant,
        objective_sense=parsed.objective_sense,
        objective_reference_form=:cbf,
        external_checksum=actual,
        source_format=:cbf,
        cbf_version=parsed.version,
        variable_domains=parsed.variable_domains,
        constraint_domains=parsed.constraint_domains,
        solve_settings=_cbf_solve_settings(spec),
    )
end
