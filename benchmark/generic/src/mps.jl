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
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "*")) && continue
        tokens = split(line)
        key = uppercase(tokens[1])
        if key in ("NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA")
            section = key
            key == "NAME" && length(tokens) > 1 && (name = tokens[2])
            key == "ENDATA" && break
            continue
        end
        if section == "ROWS"
            sense, row = only(tokens[1]), tokens[2]
            senses[row] = sense
            sense == 'N' && isempty(objective_row) && (objective_row = row)
        elseif section == "COLUMNS"
            length(tokens) >= 3 || continue
            occursin("MARKER", raw) && continue
            variable = tokens[1]
            entries = get!(columns, variable, Dict{String,Float64}())
            for index in 2:2:(length(tokens) - 1)
                entries[tokens[index]] = get(entries, tokens[index], 0.0) + parse(Float64, replace(tokens[index + 1], 'D'=>'E'))
            end
        elseif section == "RHS"
            for index in 2:2:(length(tokens) - 1)
                rhs[tokens[index]] = parse(Float64, replace(tokens[index + 1], 'D'=>'E'))
            end
        elseif section == "BOUNDS"
            kind = uppercase(tokens[1]); variable = tokens[3]
            value = length(tokens) >= 4 ? parse(Float64, replace(tokens[4], 'D'=>'E')) : 0.0
            if kind in ("LO", "LI"); lower[variable] = value
            elseif kind in ("UP", "UI"); upper[variable] = value
            elseif kind == "FX"; lower[variable] = value; upper[variable] = value
            elseif kind == "FR"; lower[variable] = -Inf; upper[variable] = Inf
            elseif kind == "MI"; lower[variable] = -Inf
            elseif kind == "PL"; upper[variable] = Inf
            elseif kind == "BV"; lower[variable] = 0.0; upper[variable] = 1.0
            end
        end
    end
    isempty(objective_row) && throw(ArgumentError("MPS $path has no N objective row"))
    for variable in keys(columns)
        get!(lower, variable, 0.0); get!(upper, variable, Inf)
    end
    return MPSData(name, objective_row, senses, columns, rhs, lower, upper)
end
