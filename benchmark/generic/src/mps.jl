"Minimal continuous-MPS representation used by the curated NETLIB subset."
struct MPSData
    name::String
    objective_row::String
    row_sense::Dict{String,Char}
    columns::Dict{String,Dict{String,Float64}}
    rhs::Dict{String,Float64}
    lower::Dict{String,Float64}
    upper::Dict{String,Float64}
end

const _MPS_SECTIONS = ("NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA")
const _MPS_UNSUPPORTED_SECTIONS = (
    "OBJSENSE", "OBJNAME", "QSECTION", "QUADOBJ", "QMATRIX",
    "CSECTION", "SOS", "INDICATORS", "GENCONS",
)
const _MPS_CONTINUOUS_BOUNDS = ("LO", "UP", "FX", "FR", "MI", "PL")
const _MPS_INTEGER_BOUNDS = ("BV", "LI", "UI", "SC", "SI")

@inline function _mps_number(token, context)
    value = parse(Float64, replace(token, 'D'=>'E'))
    isfinite(value) || throw(ArgumentError("MPS $context must be finite, got $token"))
    return value
end

"Read fixed- or free-field continuous MPS (ROWS/COLUMNS/RHS/BOUNDS)."
function read_mps(path::AbstractString)
    isfile(path) || throw(ArgumentError("missing MPS file $path; run benchmark/generic/scripts/fetch_generic_benchmarks.sh"))
    section = ""
    name = basename(path)
    objective_row = ""
    senses = Dict{String,Char}()
    columns = Dict{String,Dict{String,Float64}}()
    rhs = Dict{String,Float64}()
    lower = Dict{String,Float64}()
    upper = Dict{String,Float64}()
    rhs_name = nothing
    saw_endata = false
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "*")) && continue
        tokens = split(line)
        key = uppercase(tokens[1])
        key in _MPS_UNSUPPORTED_SECTIONS && throw(ArgumentError(
            "unsupported MPS section $key in $path"))
        if key in _MPS_SECTIONS
            key == "RANGES" && throw(ArgumentError(
                "MPS RANGES semantics are unsupported; refusing to relax $path"))
            section = key
            key == "NAME" && length(tokens) > 1 && (name = tokens[2])
            if key == "ENDATA"
                saw_endata = true
                break
            end
            continue
        end
        isempty(section) && throw(ArgumentError("MPS data before section header in $path"))
        if section == "ROWS"
            length(tokens) == 2 || throw(ArgumentError("invalid MPS ROWS record: $line"))
            length(tokens[1]) == 1 || throw(ArgumentError("invalid MPS row sense $(tokens[1])"))
            sense, row = only(tokens[1]), tokens[2]
            sense in ('N', 'E', 'L', 'G') || throw(ArgumentError("unsupported MPS row sense $sense"))
            haskey(senses, row) && throw(ArgumentError("duplicate MPS row $row"))
            senses[row] = sense
            if sense == 'N'
                isempty(objective_row) || throw(ArgumentError("multiple MPS objective rows are unsupported"))
                objective_row = row
            end
        elseif section == "COLUMNS"
            marker_record = uppercase(raw)
            any(token -> occursin(token, marker_record), ("MARKER", "INTORG", "INTEND")) &&
                throw(ArgumentError(
                    "MPS integer MARKER/INTORG/INTEND declarations are unsupported"))
            length(tokens) in (3, 5) || throw(ArgumentError("invalid MPS COLUMNS record: $line"))
            variable = tokens[1]
            entries = get!(columns, variable, Dict{String,Float64}())
            for index in 2:2:(length(tokens) - 1)
                row = tokens[index]
                haskey(senses, row) || throw(ArgumentError("MPS column references unknown row $row"))
                value = _mps_number(tokens[index + 1], "column coefficient")
                entries[row] = get(entries, row, 0.0) + value
            end
        elseif section == "RHS"
            length(tokens) in (3, 5) || throw(ArgumentError("invalid MPS RHS record: $line"))
            rhs_name === nothing ? (rhs_name = tokens[1]) :
                rhs_name == tokens[1] || throw(ArgumentError("multiple MPS RHS vectors are unsupported"))
            for index in 2:2:(length(tokens) - 1)
                row = tokens[index]
                haskey(senses, row) || throw(ArgumentError("MPS RHS references unknown row $row"))
                rhs[row] = _mps_number(tokens[index + 1], "RHS value")
            end
        elseif section == "BOUNDS"
            length(tokens) in (3, 4) || throw(ArgumentError("invalid MPS BOUNDS record: $line"))
            kind = uppercase(tokens[1]); variable = tokens[3]
            kind in _MPS_INTEGER_BOUNDS && throw(ArgumentError(
                "integer-specific MPS bound $kind is unsupported"))
            kind in _MPS_CONTINUOUS_BOUNDS || throw(ArgumentError(
                "unsupported MPS bound type $kind"))
            value_required = kind in ("LO", "UP", "FX")
            value_required == (length(tokens) == 4) || throw(ArgumentError(
                "MPS bound $kind has the wrong number of fields"))
            value = value_required ? _mps_number(tokens[4], "bound value") : 0.0
            if kind == "LO"; lower[variable] = value
            elseif kind == "UP"; upper[variable] = value
            elseif kind == "FX"; lower[variable] = value; upper[variable] = value
            elseif kind == "FR"; lower[variable] = -Inf; upper[variable] = Inf
            elseif kind == "MI"; lower[variable] = -Inf
            elseif kind == "PL"; upper[variable] = Inf
            end
        elseif section == "NAME"
            throw(ArgumentError("unexpected MPS data after NAME in $path"))
        else
            throw(ArgumentError("unsupported or malformed MPS section $section"))
        end
    end
    saw_endata || throw(ArgumentError("MPS $path has no ENDATA terminator"))
    isempty(objective_row) && throw(ArgumentError("MPS $path has no N objective row"))
    isempty(columns) && throw(ArgumentError("MPS $path has no columns"))
    for variable in union(keys(lower), keys(upper))
        haskey(columns, variable) || throw(ArgumentError(
            "MPS bound references unknown variable $variable"))
    end
    for variable in keys(columns)
        get!(lower, variable, 0.0); get!(upper, variable, Inf)
    end
    return MPSData(name, objective_row, senses, columns, rhs, lower, upper)
end

"Lower a continuous MPS instance to the public SDPX Model API."
function mps_model(data::MPSData, ::Type{T}=Float64) where {T<:AbstractFloat}
    variables = sort!(collect(keys(data.columns)))
    all(variable -> data.lower[variable] == 0.0 && isinf(data.upper[variable]), variables) ||
        throw(ArgumentError("the curated native MPS lowerer currently requires standard nonnegative variables"))
    model = SDPX.Model(T; name=data.name)
    x = SDPX.variable!(model, :mps_variables, length(variables); domain=SDPX.Nonnegative())
    index = Dict(variable => position for (position, variable) in enumerate(variables))
    rows = sort!(collect(keys(data.row_sense)))
    for row in rows
        sense = data.row_sense[row]
        sense == 'N' && continue
        expression = zero(T)
        for variable in variables
            coefficient = get(data.columns[variable], row, 0.0)
            iszero(coefficient) || (expression += T(coefficient) * x[index[variable]])
        end
        right = T(get(data.rhs, row, 0.0))
        if sense == 'E'
            SDPX.constraint!(model, Symbol(:mps_, row), expression - right, SDPX.ZeroCone())
        elseif sense == 'L'
            SDPX.constraint!(model, Symbol(:mps_, row), right - expression, SDPX.Nonnegative())
        elseif sense == 'G'
            SDPX.constraint!(model, Symbol(:mps_, row), expression - right, SDPX.Nonnegative())
        else
            throw(ArgumentError("unsupported MPS row sense $sense"))
        end
    end
    objective = zero(T)
    for variable in variables
        coefficient = get(data.columns[variable], data.objective_row, 0.0)
        iszero(coefficient) || (objective += T(coefficient) * x[index[variable]])
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    return model
end
