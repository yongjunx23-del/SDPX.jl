#=====================================================================
    MathOptInterface wrapper.

    SDPX uses geometric SDP form directly:

        min c'x
        s.t. sum_i x_i A_i^(l) - C^(l) >= 0
             B'x = b.

    The wrapper is intentionally non-incremental. JuMP/MOI builds a
    cached model, then `copy_to` finalizes all PSD incidence data in one
    pass and calls the same `ingest`/`solve!` core as the native API.
=====================================================================#

const MOI = MathOptInterface

abstract type AbstractMOIConstraintInfo end

struct MOIPSDConstraintInfo <: AbstractMOIConstraintInfo
    block::Int
    scaled::Bool
end

struct MOIEqualityConstraintInfo{T} <: AbstractMOIConstraintInfo
    column::Int
    constant::T
end

struct MOIScalarInequalityConstraintInfo{T} <: AbstractMOIConstraintInfo
    block::Int
    bound::T
    direction::T
end

struct MOIScalarIntervalConstraintInfo{T} <: AbstractMOIConstraintInfo
    lower_block::Int
    upper_block::Int
    lower::T
    upper::T
end

struct MOISOCConstraintInfo <: AbstractMOIConstraintInfo
    block::Int
    dimension::Int
    representation::Symbol
end

"""
    Optimizer{T}(; kwargs...)
    Optimizer(; kwargs...)

Create SDPX's non-incremental MathOptInterface optimizer. The untyped
constructor uses `Float64`; select `Optimizer{Float64x4}` or
`Optimizer{BigFloat}` for extended precision. JuMP and Convex.jl normally
wrap this optimizer in an MOI cache and copy the completed model into SDPX in
one pass.

Common raw keywords include `tolerance`, `max_iterations`, `time_limit`,
`threads`, `precision`, `verbosity`, and `sparse`. For Convex.jl,
[`convex_optimizer`](@ref) provides a more explicit typed factory.
"""
mutable struct Optimizer{T<:AbstractFloat} <: MOI.AbstractOptimizer
    options::SolverOptions{T}
    problem::Union{Nothing,SDPProblem{T},ConicProblem{T}}
    result::Union{Nothing,SDPResult{T},ConicResult{T}}
    num_variables::Int
    sense::MOI.OptimizationSense
    objective_constant::T
    constraint_info::Dict{Any,AbstractMOIConstraintInfo}
    solve_time::Float64
    requested_threads::Union{Nothing,Int}

    function Optimizer{T}(; kwargs...) where {T<:AbstractFloat}
        optimizer = new{T}(
            SolverOptions{T}(sparse=:auto),
            nothing,
            nothing,
            0,
            MOI.MIN_SENSE,
            zero(T),
            Dict{Any,AbstractMOIConstraintInfo}(),
            0.0,
            nothing,
        )
        for (name, value) in kwargs
            _set_raw_option!(optimizer, String(name), value)
        end
        return optimizer
    end
end

Optimizer(; kwargs...) = Optimizer{Float64}(; kwargs...)

const _MOI_OPTION_ALIASES = Dict(
    "beta" => :β,
    "gamma" => :γ,
    "omega_p" => :Ωp,
    "omega_d" => :Ωd,
    "tol_gap" => :ϵ_gap,
    "tol_primal" => :ϵ_primal,
    "tol_dual" => :ϵ_dual,
    "max_iter" => :iter_max,
    "time_limit" => :max_time,
    "verbose" => :verbosity,
    "max_iterations" => :iter_max,
    "precision" => :precision_bits,
    "num_threads" => :threads,
)

function _replace_option(options::SolverOptions{T}, name::Symbol, value) where {T}
    names = fieldnames(typeof(options))
    index = findfirst(==(name), names)
    index === nothing && throw(ArgumentError("unknown SDPX option: $name"))
    field_type = fieldtype(typeof(options), index)
    converted = if name === :verbosity && value isa Bool
        value ? 1 : 0
    elseif field_type === Any
        value
    else
        convert(field_type, value)
    end
    values = NamedTuple{names}(Tuple(getfield(options, field) for field in names))
    replacement = NamedTuple{(name,)}((converted,))
    return SolverOptions{T}(; merge(values, replacement)...)
end

function _option_symbol(options::SolverOptions, name::String)
    symbol = get(_MOI_OPTION_ALIASES, name, Symbol(name))
    symbol in fieldnames(typeof(options)) ||
        throw(MOI.UnsupportedAttribute(MOI.RawOptimizerAttribute(name)))
    return symbol
end

function _set_raw_option!(optimizer::Optimizer, name::String, value)
    if name == "tolerance"
        for field in (:ϵ_gap, :ϵ_primal, :ϵ_dual)
            optimizer.options = _replace_option(optimizer.options, field, value)
        end
        return nothing
    end
    symbol = _option_symbol(optimizer.options, name)
    if symbol === :threads
        value isa Integer && value > 0 ||
            throw(ArgumentError("threads must be a positive integer"))
        optimizer.requested_threads = Int(value)
    end
    optimizer.options = _replace_option(optimizer.options, symbol, value)
    return nothing
end

# ---- model lifecycle ----

MOI.supports_incremental_interface(::Optimizer) = false

function MOI.empty!(optimizer::Optimizer{T}) where {T}
    optimizer.problem = nothing
    optimizer.result = nothing
    optimizer.num_variables = 0
    optimizer.sense = MOI.MIN_SENSE
    optimizer.objective_constant = zero(T)
    empty!(optimizer.constraint_info)
    optimizer.solve_time = 0.0
    return nothing
end

MOI.is_empty(optimizer::Optimizer) = optimizer.problem === nothing

function Base.show(io::IO, optimizer::Optimizer{T}) where {T}
    if optimizer.result === nothing
        print(io, "SDPX.Optimizer{$T} (not solved)")
    else
        print(
            io,
            "SDPX.Optimizer{$T} ($(optimizer.result.status), " *
            "$(optimizer.result.iterations) iterations)",
        )
    end
end

# ---- supported model forms ----

const MOIPSDSet = Union{
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle},
}

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VectorAffineFunction{T}},
    ::Type{S},
) where {T,S<:MOIPSDSet}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VectorAffineFunction{T}},
    ::Type{MOI.SecondOrderCone},
) where {T}
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.SecondOrderCone},
)
    return true
end

const MOIScalarInequalitySet{T} = Union{MOI.GreaterThan{T},MOI.LessThan{T}}
const MOIScalarBoundSet{T} = Union{
    MOI.GreaterThan{T},
    MOI.LessThan{T},
    MOI.Interval{T},
}

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{S},
) where {T,S<:MOIScalarInequalitySet{T}}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{MOI.Interval{T}},
) where {T}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VariableIndex},
    ::Type{MOI.Interval{T}},
) where {T}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VariableIndex},
    ::Type{S},
) where {T,S<:MOIScalarInequalitySet{T}}
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{S},
) where {S<:MOIPSDSet}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{MOI.EqualTo{T}},
) where {T}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VariableIndex},
    ::Type{MOI.EqualTo{T}},
) where {T}
    return true
end

function MOI.supports(
    ::Optimizer{T},
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
) where {T}
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.VariableIndex}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

# ---- source-model conversion ----

_is_scaled_psd(::MOI.PositiveSemidefiniteConeTriangle) = false
_is_scaled_psd(::MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}) = true

function _triangle_coordinates(side::Int)
    coordinates = Tuple{Int,Int}[]
    sizehint!(coordinates, side * (side + 1) ÷ 2)
    for column in 1:side, row in 1:column
        push!(coordinates, (row, column))
    end
    return coordinates
end

function _canonical_empty_coefficient!(
    cache::Dict{Int,SparseMatrixCSC{T,Int}},
    side::Int,
) where {T}
    return get!(cache, side) do
        spzeros(T, side, side)
    end
end

function _empty_coefficient_vector(
    cache::Dict{Int,SparseMatrixCSC{T,Int}},
    side::Int,
    variables::Int,
) where {T}
    # Coefficient matrices are read-only after copy-in. Sharing one canonical
    # empty CSC object avoids three heap arrays for every inactive variable.
    return fill(_canonical_empty_coefficient!(cache, side), variables)
end

function _new_constraint_index!(
    counts::Dict{Any,Int},
    ::Type{F},
    ::Type{S},
) where {F,S}
    key = (F, S)
    value = get(counts, key, 0) + 1
    counts[key] = value
    return MOI.ConstraintIndex{F,S}(value)
end

function _append_psd_constraint!(
    A::Vector{SparseCoefficientVector{T}},
    C::Vector{Matrix{T}},
    empty_cache::Dict{Int,SparseMatrixCSC{T,Int}},
    optimizer::Optimizer{T},
    source,
    index_map,
    counts,
    source_index::MOI.ConstraintIndex{F,S},
) where {T,F,S<:MOIPSDSet}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    side = MOI.side_dimension(set)
    coordinates = _triangle_coordinates(side)
    dimension = length(coordinates)
    MOI.output_dimension(function_value) == dimension ||
        throw(DimensionMismatch("PSD function dimension does not match side dimension $side"))
    block_C = zeros(T, side, side)
    builders = Dict{
        Int,
        Tuple{Vector{Int},Vector{Int},Vector{T}},
    }()
    scaled = _is_scaled_psd(set)
    sqrt_two = sqrt(T(2))

    function append_coefficient!(
        variable::Int,
        row::Int,
        column::Int,
        coefficient::T,
    )
        rows, columns, values = get!(builders, variable) do
            (Int[], Int[], T[])
        end
        push!(rows, row)
        push!(columns, column)
        push!(values, coefficient)
        if row != column
            push!(rows, column)
            push!(columns, row)
            push!(values, coefficient)
        end
        return nothing
    end

    constants = function_value isa MOI.VectorAffineFunction{T} ?
                function_value.constants : zeros(T, dimension)
    @inbounds for output in 1:dimension
        row, column = coordinates[output]
        scale = scaled && row != column ? sqrt_two : one(T)
        value = constants[output] / scale
        block_C[row, column] = -value
        row != column && (block_C[column, row] = -value)
    end

    if function_value isa MOI.VectorAffineFunction{T}
        for term in function_value.terms
            output = term.output_index
            row, column = coordinates[output]
            scale = scaled && row != column ? sqrt_two : one(T)
            variable = index_map[term.scalar_term.variable].value
            coefficient = term.scalar_term.coefficient / scale
            append_coefficient!(variable, row, column, coefficient)
        end
    else
        for (output, variable_index) in pairs(function_value.variables)
            row, column = coordinates[output]
            scale = scaled && row != column ? sqrt_two : one(T)
            variable = index_map[variable_index].value
            coefficient = inv(scale)
            append_coefficient!(variable, row, column, coefficient)
        end
    end

    block_A = _empty_coefficient_vector(
        empty_cache,
        side,
        optimizer.num_variables,
    )
    for (variable, (rows, columns, values)) in builders
        matrix = sparse(rows, columns, values, side, side)
        dropzeros!(matrix)
        nnz(matrix) > 0 && (block_A[variable] = matrix)
    end
    push!(A, block_A)
    push!(C, block_C)
    destination_index = _new_constraint_index!(counts, F, S)
    index_map[source_index] = destination_index
    optimizer.constraint_info[destination_index] =
        MOIPSDConstraintInfo(length(A), scaled)
    return nothing
end

function _append_equality_constraint!(
    matrix_rows::Vector{Int},
    matrix_columns::Vector{Int},
    matrix_values::Vector{T},
    rhs::Vector{T},
    optimizer::Optimizer{T},
    source,
    index_map,
    counts,
    source_index::MOI.ConstraintIndex{F,MOI.EqualTo{T}},
) where {T,F}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    column = length(rhs) + 1
    constant = zero(T)
    if function_value isa MOI.ScalarAffineFunction{T}
        constant = function_value.constant
        for term in function_value.terms
            iszero(term.coefficient) && continue
            push!(matrix_rows, index_map[term.variable].value)
            push!(matrix_columns, column)
            push!(matrix_values, term.coefficient)
        end
    else
        push!(matrix_rows, index_map[function_value].value)
        push!(matrix_columns, column)
        push!(matrix_values, one(T))
    end
    push!(rhs, set.value - constant)
    destination_index = _new_constraint_index!(counts, F, MOI.EqualTo{T})
    index_map[source_index] = destination_index
    optimizer.constraint_info[destination_index] =
        MOIEqualityConstraintInfo{T}(
            length(rhs), _ingest_owned_scalar(T, constant),
        )
    return nothing
end

function _scalar_coefficient_vector(
    ::Type{T},
    variables::Int,
    coefficients::Dict{Int,T},
    direction::T,
) where {T}
    active = Tuple{Int,T}[]
    sizehint!(active, length(coefficients))
    for (variable, coefficient) in coefficients
        value = direction * coefficient
        iszero(value) || push!(active, (variable, value))
    end
    sort!(active; by=first)
    if length(active) == 1
        variable, value = only(active)
        return CompactScalarCoefficientVector(T, variables, variable, value)
    end
    active_variables = Vector{Int}(undef, length(active))
    active_coefficients = Vector{SparseMatrixCSC{T,Int}}(undef, length(active))
    @inbounds for position in eachindex(active)
        variable, value = active[position]
        active_variables[position] = variable
        active_coefficients[position] = sparse([1], [1], T[value], 1, 1)
    end
    return ActiveSparseCoefficientVector(
        T,
        variables,
        active_variables,
        active_coefficients,
        1,
    )
end

function _append_scalar_inequality!(
    A::Vector{SparseCoefficientVector{T}},
    C::Vector{Matrix{T}},
    empty_cache::Dict{Int,SparseMatrixCSC{T,Int}},
    optimizer::Optimizer{T},
    source,
    index_map,
    counts,
    source_index::MOI.ConstraintIndex{F,S},
) where {T,F,S<:MOIScalarInequalitySet{T}}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    coefficients = Dict{Int,T}()
    constant = zero(T)
    if function_value isa MOI.ScalarAffineFunction{T}
        constant = function_value.constant
        for term in function_value.terms
            variable = index_map[term.variable].value
            coefficients[variable] =
                get(coefficients, variable, zero(T)) + term.coefficient
        end
    else
        coefficients[index_map[function_value].value] = one(T)
    end
    if set isa MOI.GreaterThan{T}
        bound = set.lower
        direction = one(T)
        block_C = reshape(T[bound - constant], 1, 1)
    else
        bound = set.upper
        direction = -one(T)
        block_C = reshape(T[constant - bound], 1, 1)
    end
    block_A = _scalar_coefficient_vector(
        T,
        optimizer.num_variables,
        coefficients,
        direction,
    )
    push!(A, block_A)
    push!(C, block_C)
    destination_index = _new_constraint_index!(counts, F, S)
    index_map[source_index] = destination_index
    optimizer.constraint_info[destination_index] =
        MOIScalarInequalityConstraintInfo{T}(
            length(A),
            _ingest_owned_scalar(T, bound),
            _ingest_owned_scalar(T, direction),
        )
    return nothing
end

function _append_scalar_interval!(
    A::Vector{SparseCoefficientVector{T}},
    C::Vector{Matrix{T}},
    empty_cache::Dict{Int,SparseMatrixCSC{T,Int}},
    optimizer::Optimizer{T},
    source,
    index_map,
    counts,
    source_index::MOI.ConstraintIndex{F,MOI.Interval{T}},
) where {T,F}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    coefficients = Dict{Int,T}()
    constant = zero(T)
    if function_value isa MOI.ScalarAffineFunction{T}
        constant = function_value.constant
        for term in function_value.terms
            variable = index_map[term.variable].value
            coefficients[variable] =
                get(coefficients, variable, zero(T)) + term.coefficient
        end
    else
        coefficients[index_map[function_value].value] = one(T)
    end

    function append_block!(direction::T, block_constant::T)
        block_A = _scalar_coefficient_vector(
            T,
            optimizer.num_variables,
            coefficients,
            direction,
        )
        push!(A, block_A)
        push!(C, reshape(T[block_constant], 1, 1))
        return length(A)
    end

    lower_block = append_block!(one(T), set.lower - constant)
    upper_block = append_block!(-one(T), constant - set.upper)
    destination_index =
        _new_constraint_index!(counts, F, MOI.Interval{T})
    index_map[source_index] = destination_index
    optimizer.constraint_info[destination_index] =
        MOIScalarIntervalConstraintInfo{T}(
            lower_block,
            upper_block,
            _ingest_owned_scalar(T, set.lower),
            _ingest_owned_scalar(T, set.upper),
        )
    return nothing
end

function _moi_scalar_affine_data(
    ::Type{T},
    optimizer::Optimizer{T},
    function_value,
    index_map,
) where {T}
    coefficients = zeros(T, optimizer.num_variables)
    constant = zero(T)
    if function_value isa MOI.ScalarAffineFunction{T}
        constant = function_value.constant
        @inbounds for term in function_value.terms
            coefficients[index_map[term.variable].value] += term.coefficient
        end
    else
        coefficients[index_map[function_value].value] = one(T)
    end
    return coefficients, constant
end

function _append_native_soc_constraint!(
    cones::Vector{SOCConstraint{T}},
    optimizer::Optimizer{T},
    source,
    index_map,
    counts,
    source_index::MOI.ConstraintIndex{F,MOI.SecondOrderCone},
) where {T,F}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    dimension = set.dimension
    MOI.output_dimension(function_value) == dimension ||
        throw(DimensionMismatch("SOC function dimension does not match its set"))
    matrix = zeros(T, dimension, optimizer.num_variables)
    offset = function_value isa MOI.VectorAffineFunction{T} ?
             Vector{T}(function_value.constants) : zeros(T, dimension)
    if function_value isa MOI.VectorAffineFunction{T}
        @inbounds for term in function_value.terms
            matrix[
                term.output_index,
                index_map[term.scalar_term.variable].value,
            ] += term.scalar_term.coefficient
        end
    else
        @inbounds for (output, variable) in pairs(function_value.variables)
            matrix[output, index_map[variable].value] += one(T)
        end
    end
    push!(cones, SOCConstraint(matrix, offset; T=T))
    destination_index = _new_constraint_index!(counts, F, MOI.SecondOrderCone)
    index_map[source_index] = destination_index
    optimizer.constraint_info[destination_index] =
        MOISOCConstraintInfo(length(cones), dimension, :native_lorentz)
    return nothing
end

function _append_native_scalar_inequality!(
    cones::Vector{SOCConstraint{T}},
    optimizer::Optimizer{T},
    source,
    index_map,
    counts,
    source_index::MOI.ConstraintIndex{F,S},
) where {T,F,S<:MOIScalarInequalitySet{T}}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    coefficients, constant = _moi_scalar_affine_data(
        T, optimizer, function_value, index_map,
    )
    if set isa MOI.GreaterThan{T}
        bound = set.lower
        direction = one(T)
        offset = constant - bound
    else
        bound = set.upper
        direction = -one(T)
        offset = bound - constant
    end
    matrix = reshape(direction .* coefficients, 1, optimizer.num_variables)
    push!(cones, SOCConstraint(matrix, T[offset]; T=T))
    destination_index = _new_constraint_index!(counts, F, S)
    index_map[source_index] = destination_index
    optimizer.constraint_info[destination_index] =
        MOIScalarInequalityConstraintInfo{T}(
            length(cones),
            _ingest_owned_scalar(T, bound),
            _ingest_owned_scalar(T, direction),
        )
    return nothing
end

function _append_native_scalar_interval!(
    cones::Vector{SOCConstraint{T}},
    optimizer::Optimizer{T},
    source,
    index_map,
    counts,
    source_index::MOI.ConstraintIndex{F,MOI.Interval{T}},
) where {T,F}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    coefficients, constant = _moi_scalar_affine_data(
        T, optimizer, function_value, index_map,
    )
    push!(cones, SOCConstraint(
        reshape(coefficients, 1, optimizer.num_variables),
        T[constant - set.lower];
        T=T,
    ))
    lower_block = length(cones)
    push!(cones, SOCConstraint(
        reshape(-coefficients, 1, optimizer.num_variables),
        T[set.upper - constant];
        T=T,
    ))
    upper_block = length(cones)
    destination_index = _new_constraint_index!(counts, F, MOI.Interval{T})
    index_map[source_index] = destination_index
    optimizer.constraint_info[destination_index] =
        MOIScalarIntervalConstraintInfo{T}(
            lower_block,
            upper_block,
            _ingest_owned_scalar(T, set.lower),
            _ingest_owned_scalar(T, set.upper),
        )
    return nothing
end

function _objective_data(
    optimizer::Optimizer{T},
    source,
    index_map,
) where {T}
    c = zeros(T, optimizer.num_variables)
    constant = zero(T)
    sense = MOI.get(source, MOI.ObjectiveSense())
    if sense != MOI.FEASIBILITY_SENSE
        function_type = MOI.get(source, MOI.ObjectiveFunctionType())
        if function_type == MOI.ScalarAffineFunction{T}
            function_value = MOI.get(
                source,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            )
            constant = function_value.constant
            for term in function_value.terms
                c[index_map[term.variable].value] += term.coefficient
            end
        elseif function_type == MOI.VariableIndex
            variable = MOI.get(source, MOI.ObjectiveFunction{MOI.VariableIndex}())
            c[index_map[variable].value] = one(T)
        else
            throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{function_type}()))
        end
    end
    sense == MOI.MAX_SENSE && (c .*= -one(T))
    return c, constant, sense
end

function _check_copy_attributes(optimizer::Optimizer, source)
    for attribute in MOI.get(source, MOI.ListOfModelAttributesSet())
        if attribute == MOI.Name() ||
           attribute == MOI.ObjectiveSense() ||
           attribute isa MOI.ObjectiveFunction
            continue
        end
        throw(MOI.UnsupportedAttribute(attribute))
    end
    for attribute in MOI.get(source, MOI.ListOfVariableAttributesSet())
        attribute == MOI.VariableName() && continue
        throw(MOI.UnsupportedAttribute(attribute))
    end
    for (F, S) in MOI.get(source, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(optimizer, F, S) ||
            throw(MOI.UnsupportedConstraint{F,S}())
        for attribute in MOI.get(source, MOI.ListOfConstraintAttributesSet{F,S}())
            attribute == MOI.ConstraintName() && continue
            throw(MOI.UnsupportedAttribute(attribute))
        end
    end
    return nothing
end

function MOI.copy_to(optimizer::Optimizer{T}, source::MOI.ModelLike) where {T}
    _check_copy_attributes(optimizer, source)
    MOI.empty!(optimizer)
    index_map = MOI.Utilities.IndexMap()
    source_variables = MOI.get(source, MOI.ListOfVariableIndices())
    optimizer.num_variables = length(source_variables)
    for (i, variable) in pairs(source_variables)
        index_map[variable] = MOI.VariableIndex(i)
    end

    constraint_types = MOI.get(source, MOI.ListOfConstraintTypesPresent())
    has_soc = any(entry -> last(entry) == MOI.SecondOrderCone, constraint_types)
    has_psd = any(entry -> last(entry) <: MOIPSDSet, constraint_types)
    has_soc && has_psd && throw(ArgumentError(
        "mixed PSD and SOC constraints are not supported by one production " *
        "route; use the native SOCP API or an all-PSD model explicitly",
    ))

    A = SparseCoefficientVector{T}[]
    C = Matrix{T}[]
    native_cones = SOCConstraint{T}[]
    equality_rows = Int[]
    equality_columns = Int[]
    equality_values = T[]
    equality_rhs = T[]
    counts = Dict{Any,Int}()
    empty_cache = Dict{Int,SparseMatrixCSC{T,Int}}()

    for (F, S) in constraint_types
        for source_index in MOI.get(source, MOI.ListOfConstraintIndices{F,S}())
            if has_soc && S == MOI.SecondOrderCone
                _append_native_soc_constraint!(
                    native_cones,
                    optimizer,
                    source,
                    index_map,
                    counts,
                    source_index,
                )
            elseif has_soc && S <: MOIScalarInequalitySet{T}
                _append_native_scalar_inequality!(
                    native_cones,
                    optimizer,
                    source,
                    index_map,
                    counts,
                    source_index,
                )
            elseif has_soc && S <: MOI.Interval{T}
                _append_native_scalar_interval!(
                    native_cones,
                    optimizer,
                    source,
                    index_map,
                    counts,
                    source_index,
                )
            elseif S <: MOIPSDSet
                _append_psd_constraint!(
                    A,
                    C,
                    empty_cache,
                    optimizer,
                    source,
                    index_map,
                    counts,
                    source_index,
                )
            elseif S <: MOI.EqualTo{T}
                _append_equality_constraint!(
                    equality_rows,
                    equality_columns,
                    equality_values,
                    equality_rhs,
                    optimizer,
                    source,
                    index_map,
                    counts,
                    source_index,
                )
            elseif S <: MOIScalarInequalitySet{T}
                _append_scalar_inequality!(
                    A,
                    C,
                    empty_cache,
                    optimizer,
                    source,
                    index_map,
                    counts,
                    source_index,
                )
            elseif S <: MOI.Interval{T}
                _append_scalar_interval!(
                    A,
                    C,
                    empty_cache,
                    optimizer,
                    source,
                    index_map,
                    counts,
                    source_index,
                )
            else
                throw(MOI.UnsupportedConstraint{F,S}())
            end
        end
    end
    if !has_soc && isempty(A)
        # The core representation intentionally keeps L >= 1. A pure-equality
        # or unconstrained LP therefore receives one internal, always-satisfied
        # scalar cone row:
        #
        #     0*x - (-1) >= 0.
        #
        # LP presolve removes this row and reaches its equality-only path. No
        # MOI constraint index or constraint_info entry is created, so the
        # implementation detail is invisible through the public model.
        push!(
            A,
            _empty_coefficient_vector(
                empty_cache,
                1,
                optimizer.num_variables,
            ),
        )
        push!(C, reshape(T[-one(T)], 1, 1))
    end

    equality_count = length(equality_rhs)
    sparse_B = sparse(
        equality_rows,
        equality_columns,
        equality_values,
        optimizer.num_variables,
        equality_count,
    )
    equality_slots = optimizer.num_variables * equality_count
    B = equality_slots > 0 && nnz(sparse_B) * 4 > equality_slots ?
        Matrix(sparse_B) : sparse_B
    c, objective_constant, sense = _objective_data(optimizer, source, index_map)
    optimizer.sense = sense
    optimizer.objective_constant = _ingest_owned_scalar(T, objective_constant)
    optimizer.problem = if has_soc
        equality_matrix = B isa SparseMatrixCSC ?
                          sparse(transpose(B)) : transpose(B)
        second_order_program(
            c,
            native_cones;
            Aeq=equality_matrix,
            beq=equality_rhs,
            T=T,
        )
    else
        ingest(
            c,
            A,
            C,
            B,
            equality_rhs;
            sparse=optimizer.options.sparse,
            verbosity=optimizer.options.verbosity,
        )
    end
    optimizer.result = nothing
    return index_map
end

# ---- solve and result attributes ----

function MOI.optimize!(optimizer::Optimizer)
    optimizer.problem === nothing &&
        error("no model has been copied to SDPX")
    optimizer.solve_time = @elapsed begin
        if optimizer.problem isa ConicProblem
            optimizer.result = _run_native_soc_frontend(
                optimizer.problem,
                optimizer.options,
                :auto,
            )
        else
            optimizer.result = solve!(optimizer.problem, optimizer.options)
        end
    end
    return nothing
end

MOI.get(::Optimizer, ::MOI.SolverName) = "SDPX"
function MOI.get(::Optimizer, ::MOI.SolverVersion)
    version = Base.pkgversion(@__MODULE__)
    return version === nothing ? "unknown" : string(version)
end
MOI.get(optimizer::Optimizer, ::MOI.RawSolver) = optimizer.result
# A CachingOptimizer maps model-attribute values back through its source index
# map. `SDPResult` contains no MOI variable or constraint indices, so it must
# pass through unchanged. Convex.jl automatically inserts this cache because
# SDPX is non-incremental; without the attribute-specific method,
# `MOI.get(problem.model, MOI.RawSolver())` fails after an otherwise successful
# Convex solve.
MOI.Utilities.map_indices(::Any, ::MOI.RawSolver, result::SDPResult) = result
MOI.Utilities.map_indices(::Any, ::MOI.RawSolver, result::ConicResult) = result
MOI.get(optimizer::Optimizer, ::MOI.NumberOfVariables) = optimizer.num_variables
MOI.get(optimizer::Optimizer, ::MOI.ObjectiveSense) = optimizer.sense
MOI.set(optimizer::Optimizer, ::MOI.ObjectiveSense, sense) =
    (optimizer.sense = sense)
MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec) = optimizer.solve_time
MOI.get(optimizer::Optimizer, ::MOI.BarrierIterations) =
    optimizer.result === nothing ? 0 : optimizer.result.iterations

function _moi_has_iterate(result::Union{SDPResult,ConicResult})
    return result.status in (
        Optimal,
        FeasibleCert,
        IterLimit,
        TimeLimit,
        Stalled,
        MaxRestartsExceeded,
        UserStopped,
        PrimalInfeasible,
        DualInfeasible,
    )
end

function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    optimizer.result === nothing && return 0
    return _moi_has_iterate(optimizer.result) ? 1 : 0
end

MOI.get(optimizer::Optimizer, ::MOI.RawStatusString) =
    optimizer.result === nothing ? "optimize! not called" : optimizer.result.message

MOI.supports(::Optimizer, ::MOI.TerminationStatus) = true
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    optimizer.result === nothing && return MOI.OPTIMIZE_NOT_CALLED
    status = optimizer.result.status
    status == Optimal && return MOI.OPTIMAL
    # In FEASIBILITY mode every feasible point is globally optimal for the
    # zero objective represented by MOI.FEASIBILITY_SENSE.
    status == FeasibleCert && return MOI.OPTIMAL
    status == InfeasibleCert && return MOI.INFEASIBLE
    status == PrimalInfeasible && return MOI.INFEASIBLE
    status == DualInfeasible && return MOI.DUAL_INFEASIBLE
    status == IterLimit && return MOI.ITERATION_LIMIT
    status == TimeLimit && return MOI.TIME_LIMIT
    status == UserStopped && return MOI.INTERRUPTED
    status == Stalled && return MOI.SLOW_PROGRESS
    status == MaxRestartsExceeded && return MOI.SLOW_PROGRESS
    # A point that met a relaxed multiple of the requested tolerance but not the
    # tolerance itself. MOI has an exact term for this, and using it keeps
    # `MOI.OPTIMAL` meaning "the requested tolerance was met".
    status == AlmostOptimal && return MOI.ALMOST_OPTIMAL
    # The arithmetic width, not the algorithm, was the binding constraint.
    # `SLOW_PROGRESS` is the closest honest MOI code: the solve did not fail, it
    # ran out of precision, and the remedy is a wider type.
    status == InsufficientPrecision && return MOI.SLOW_PROGRESS
    status == NumericalFailure && return MOI.NUMERICAL_ERROR
    return MOI.NUMERICAL_ERROR
end

MOI.supports(::Optimizer, ::MOI.PrimalStatus) = true
function MOI.get(optimizer::Optimizer, attribute::MOI.PrimalStatus)
    optimizer.result === nothing && return MOI.NO_SOLUTION
    1 <= attribute.result_index <= MOI.get(optimizer, MOI.ResultCount()) ||
        return MOI.NO_SOLUTION
    status = optimizer.result.status
    status in (Optimal, FeasibleCert) && return MOI.FEASIBLE_POINT
    status == DualInfeasible && return MOI.INFEASIBILITY_CERTIFICATE
    status == PrimalInfeasible && return MOI.NO_SOLUTION
    # Report the weaker-but-honest point status rather than claiming a feasible
    # point for an iterate that only met a relaxed tolerance.
    status == AlmostOptimal && return MOI.NEARLY_FEASIBLE_POINT
    return MOI.UNKNOWN_RESULT_STATUS
end

MOI.supports(::Optimizer, ::MOI.DualStatus) = true
function MOI.get(optimizer::Optimizer, attribute::MOI.DualStatus)
    optimizer.result === nothing && return MOI.NO_SOLUTION
    1 <= attribute.result_index <= MOI.get(optimizer, MOI.ResultCount()) ||
        return MOI.NO_SOLUTION
    status = optimizer.result.status
    status == Optimal && return MOI.FEASIBLE_POINT
    status == PrimalInfeasible && return MOI.INFEASIBILITY_CERTIFICATE
    status == DualInfeasible && return MOI.NO_SOLUTION
    status == AlmostOptimal && return MOI.NEARLY_FEASIBLE_POINT
    return MOI.UNKNOWN_RESULT_STATUS
end

function _check_result(optimizer::Optimizer, attribute)
    MOI.check_result_index_bounds(optimizer, attribute)
    return optimizer.result
end

MOI.supports(::Optimizer, ::MOI.ObjectiveValue) = true
MOI.supports(::Optimizer, ::MOI.DualObjectiveValue) = true
MOI.supports(::Optimizer, ::MOI.RelativeGap) = true
MOI.supports(::Optimizer, ::MOI.VariablePrimal) = true
MOI.supports(::Optimizer, ::MOI.ConstraintPrimal) = true
MOI.supports(::Optimizer, ::MOI.ConstraintDual) = true

function MOI.get(optimizer::Optimizer, attribute::MOI.ObjectiveValue)
    result = _check_result(optimizer, attribute)
    result.status === PrimalInfeasible && return NaN
    value = optimizer.sense == MOI.MAX_SENSE ? -result.pObj : result.pObj
    return value + optimizer.objective_constant
end

function MOI.get(optimizer::Optimizer, attribute::MOI.DualObjectiveValue)
    result = _check_result(optimizer, attribute)
    result.status === DualInfeasible && return NaN
    value = optimizer.sense == MOI.MAX_SENSE ? -result.dObj : result.dObj
    return value + optimizer.objective_constant
end

function MOI.get(optimizer::Optimizer, attribute::MOI.RelativeGap)
    result = _check_result(optimizer, attribute)
    return result.gap_rel
end

function MOI.get(
    optimizer::Optimizer,
    attribute::MOI.VariablePrimal,
    variable::MOI.VariableIndex,
)
    result = _check_result(optimizer, attribute)
    return result.x[variable.value]
end

function _pack_psd_matrix(matrix::AbstractMatrix{T}, scaled::Bool) where {T}
    side = size(matrix, 1)
    packed = Vector{T}(undef, side * (side + 1) ÷ 2)
    sqrt_two = sqrt(T(2))
    output = 0
    @inbounds for column in 1:side, row in 1:column
        output += 1
        scale = scaled && row != column ? sqrt_two : one(T)
        packed[output] = scale * matrix[row, column]
    end
    return packed
end

function MOI.get(
    optimizer::Optimizer,
    attribute::MOI.ConstraintPrimal,
    index::MOI.ConstraintIndex,
)
    result = _check_result(optimizer, attribute)
    info = optimizer.constraint_info[index]
    if result isa ConicResult
        if info isa MOIScalarInequalityConstraintInfo
            return info.bound + info.direction * result.slack[info.block][1]
        end
        if info isa MOIScalarIntervalConstraintInfo
            return info.lower + result.slack[info.lower_block][1]
        end
        if info isa MOISOCConstraintInfo
            return copy(result.slack[info.block])
        end
        equality = info::MOIEqualityConstraintInfo
        return dot(
            view(optimizer.problem.Aeq, equality.column, :),
            result.x,
        ) + equality.constant
    end
    if info isa MOIPSDConstraintInfo
        return _pack_psd_matrix(result.X[info.block], info.scaled)
    end
    if info isa MOIScalarInequalityConstraintInfo
        return info.bound + info.direction * result.X[info.block][1, 1]
    end
    if info isa MOIScalarIntervalConstraintInfo
        return info.lower + result.X[info.lower_block][1, 1]
    end
    equality = info::MOIEqualityConstraintInfo
    return dot(view(optimizer.problem.B, :, equality.column), result.x) +
           equality.constant
end

function MOI.get(
    optimizer::Optimizer,
    attribute::MOI.ConstraintDual,
    index::MOI.ConstraintIndex,
)
    result = _check_result(optimizer, attribute)
    info = optimizer.constraint_info[index]
    if result isa ConicResult
        if info isa MOIScalarInequalityConstraintInfo
            return info.direction * result.dual[info.block][1]
        end
        if info isa MOIScalarIntervalConstraintInfo
            return result.dual[info.lower_block][1] -
                   result.dual[info.upper_block][1]
        end
        if info isa MOISOCConstraintInfo
            return copy(result.dual[info.block])
        end
        equality = info::MOIEqualityConstraintInfo
        return result.equality_dual[equality.column]
    end
    if info isa MOIPSDConstraintInfo
        return _pack_psd_matrix(result.Y[info.block], info.scaled)
    end
    if info isa MOIScalarInequalityConstraintInfo
        return info.direction * result.Y[info.block][1, 1]
    end
    if info isa MOIScalarIntervalConstraintInfo
        return result.Y[info.lower_block][1, 1] -
               result.Y[info.upper_block][1, 1]
    end
    equality = info::MOIEqualityConstraintInfo
    return result.y[equality.column]
end

# ---- optimizer attributes ----

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.options.verbosity == 0
function MOI.set(optimizer::Optimizer, ::MOI.Silent, silent::Bool)
    verbosity = silent ? 0 : max(optimizer.options.verbosity, 1)
    optimizer.options = _replace_option(optimizer.options, :verbosity, verbosity)
    return nothing
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec) = optimizer.options.max_time
function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, value::Real)
    optimizer.options = _replace_option(optimizer.options, :max_time, value)
    return nothing
end
function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    optimizer.options = _replace_option(optimizer.options, :max_time, Inf)
    return nothing
end

MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = true
MOI.get(optimizer::Optimizer, ::MOI.NumberOfThreads) =
    optimizer.requested_threads
function MOI.set(
    optimizer::Optimizer,
    ::MOI.NumberOfThreads,
    value::Int,
)
    value > 0 || throw(ArgumentError("number of threads must be positive"))
    optimizer.requested_threads = value
    optimizer.options = _replace_option(optimizer.options, :threads, value)
    return nothing
end
function MOI.set(optimizer::Optimizer, ::MOI.NumberOfThreads, ::Nothing)
    optimizer.requested_threads = nothing
    optimizer.options = _replace_option(
        optimizer.options,
        :threads,
        Base.Threads.nthreads(),
    )
    return nothing
end

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true
function MOI.get(optimizer::Optimizer, attribute::MOI.RawOptimizerAttribute)
    if attribute.name == "tolerance"
        gap = optimizer.options.ϵ_gap
        primal = optimizer.options.ϵ_primal
        dual = optimizer.options.ϵ_dual
        return gap == primal == dual ?
               gap :
               (gap=gap, primal=primal, dual=dual)
    end
    symbol = _option_symbol(optimizer.options, attribute.name)
    return getfield(optimizer.options, symbol)
end
function MOI.set(
    optimizer::Optimizer,
    attribute::MOI.RawOptimizerAttribute,
    value,
)
    _set_raw_option!(optimizer, attribute.name, value)
    return nothing
end
