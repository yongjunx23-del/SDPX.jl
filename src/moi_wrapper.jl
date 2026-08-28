#=====================================================================
    MathOptInterface wrapper.

    SDPX uses geometric SDP form directly:

        min c'x
        s.t. sum_i x_i A_i^(l) - C^(l) >= 0
             B'x = b.

    The wrapper is intentionally non-incremental. JuMP/MOI builds a
    cached model, then `copy_to` finalizes all PSD incidence data in one
    pass and calls the public `Model`/`Result` solve seam exactly once.
=====================================================================#

const MOI = MathOptInterface

"""Typed coordinate metadata reserved for the future MOI asymmetric bridge.

The R1 adapter remains fail-closed for Exp/Power sets; this identity metadata
is retained on existing records only so the later mapping can be added
without changing their storage shape.
"""
struct MOIAsymmetricMap{T<:AbstractFloat}
    kind::Symbol
    alpha::T

    function MOIAsymmetricMap{T}(kind::Symbol, alpha::T) where {T<:AbstractFloat}
        kind in (:identity, :dual_exp, :dual_power) ||
            throw(ArgumentError("unknown MOI asymmetric map kind $kind"))
        return new{T}(kind, alpha)
    end
end

@inline _moi_identity_map(::Type{T}) where {T<:AbstractFloat} =
    MOIAsymmetricMap{T}(:identity, zero(T))

# The v0.5 MOI bridge stores source constraints as identities in the public
# `Model` builder.  These records deliberately contain only typed Model refs
# and original affine expressions; no second SDP/SOC/LP canonicalizer is
# involved.  `kind=:variable` records a VectorOfVariables product block,
# while all other kinds retain one or two affine Model constraint blocks.
struct MOIModelConstraintInfo{T<:AbstractFloat}
    kind::Symbol
    set_kind::Symbol
    expressions::Vector{ScalarAffine{T}}
    refs::Vector{ConstraintRef}
    aux_refs::Vector{ConstraintRef}
    entries::Vector{VariableEntry{T}}
    variable_block::Union{Nothing,VariableBlockRef{T}}
    scaled::Bool
    # Reserved typed metadata for a future asymmetric MOI bridge.  R1 does
    # not expose Exp/Power constraints and therefore never consumes a map.
    asym_map::MOIAsymmetricMap{T}
end

struct MOIAdapterError <: Exception
    reason::Symbol
    message::String
end

Base.showerror(io::IO, error::MOIAdapterError) = print(io, error.message)

const MOIModelConstraintKey = Tuple{DataType,Int}
@inline _moi_constraint_key(index::MOI.ConstraintIndex) =
    (typeof(index), index.value)

"""
    Optimizer{T}(; kwargs...)
    Optimizer(; kwargs...)

Create SDPX's non-incremental MathOptInterface optimizer. The untyped
constructor uses `Float64`; select `Optimizer{Float64x4}` or
`Optimizer{BigFloat}` for extended precision. JuMP normally wraps this
optimizer in an MOI cache and copies the completed model into SDPX in one
pass.

Common raw keywords include `tolerance`, `max_iterations`, `time_limit`,
`threads`, `precision`, `verbosity`, and `sparse`.

Phase-9 engine policy: the wrapper is one-shot and non-incremental
(`MOI.supports_incremental_interface` is `false`), and native product HSD is
the only engine.  The raw `"engine"` attribute accepts only `:auto` (the
default) or `:native_hsd`; the historical `:legacy` selector is rejected with
a migration error.  The raw `"algorithm"` attribute accepts only `:auto`;
family selectors (`:lp`, `:socp`, `:sdp`) are deprecated and rejected.
Exponential- and power-cone constraints stay fail-closed: they are not part
of the supported MOI function/set surface and fail during discovery or copy.
"""
mutable struct Optimizer{T<:AbstractFloat} <: MOI.AbstractOptimizer
    options::SolverOptions{T}
    # Public engine selector.  This is deliberately separate from the
    # historical SolverOptions record: SolverOptions predates the direct HSD
    # route, while Settings owns the authoritative `:auto/:native_hsd`
    # policy consumed by public optimize!.  Phase 9 removed the `:legacy`
    # engine selector from the public surface.
    engine::Symbol
    # v0.5 authoritative builder/result seam.  The adapter is included after
    # public/result.jl, so this field is the concrete public Result boundary.
    model::Union{Nothing,Model{T}}
    public_result::Union{Nothing,Result{T}}
    model_variables::Vector{VariableEntry{T}}
    model_constraint_records::Dict{MOIModelConstraintKey,MOIModelConstraintInfo{T}}
    model_constraint_starts::Dict{MOIModelConstraintKey,Vector{T}}
    start_error::Union{Nothing,Tuple{Symbol,String}}
    num_variables::Int
    sense::MOI.OptimizationSense
    objective_constant::T
    solve_time::Float64
    requested_threads::Union{Nothing,Int}

    function Optimizer{T}(; kwargs...) where {T<:AbstractFloat}
        _require_supported_arithmetic_type(T)
        optimizer = new{T}(
            SolverOptions{T}(sparse=:auto),
            :auto,
            nothing,
            nothing,
            VariableEntry{T}[],
            Dict{MOIModelConstraintKey,MOIModelConstraintInfo{T}}(),
            Dict{MOIModelConstraintKey,Vector{T}}(),
            nothing,
            0,
            MOI.FEASIBILITY_SENSE,
            zero(T),
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

# Only options with a lossless representation in the public Settings bridge
# may enter the non-incremental MOI solve path.  SolverOptions intentionally
# contains many expert interior-point controls, but _moi_settings does not
# carry those fields through Settings -> SolveOptions.  Accepting them here
# would make a subsequent solve silently use a default value.  BigFloat's
# precision_bits is the one exception: it is consumed while MOI copies the
# source model and therefore has a real adapter-side effect before solve.
const _MOI_LOSSLESS_OPTION_FIELDS = (
    :ϵ_gap,
    :ϵ_primal,
    :ϵ_dual,
    :iter_max,
    :max_time,
    :threads,
    :precision_bits,
    :scaling,
    :formulation,
    :linear_algebra_backend,
    :presolve,
    :algorithm,
    :sparse,
    :equality_solver,
    :working_precision_policy,
    :diagnostics,
    :verbosity,
    :timing,
    :certification,
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
    if !(symbol in fieldnames(typeof(options)) &&
         symbol in _MOI_LOSSLESS_OPTION_FIELDS)
        throw(MOI.UnsupportedAttribute(MOI.RawOptimizerAttribute(name)))
    end
    return symbol
end

function _set_raw_option!(optimizer::Optimizer, name::String, value)
    if name == "engine"
        value isa Symbol || throw(ArgumentError(
            "MOI engine must be a Symbol (:auto or :native_hsd)",
        ))
        if value == :legacy
            throw(ArgumentError(
                "engine=:legacy is deprecated and no longer selectable: " *
                "native product HSD is the only public engine.  Use " *
                "engine=:auto or engine=:native_hsd.  There is no hidden " *
                "legacy fallback on the MOI surface.",
            ))
        end
        value in (:auto, :native_hsd) || throw(ArgumentError(
            "MOI engine must be :auto or :native_hsd, got $(repr(value))",
        ))
        optimizer.engine = value
        return nothing
    end
    if name == "tolerance"
        for field in (:ϵ_gap, :ϵ_primal, :ϵ_dual)
            optimizer.options = _replace_option(optimizer.options, field, value)
        end
        return nothing
    end
    symbol = _option_symbol(optimizer.options, name)
    if symbol === :algorithm && value !== :auto
        throw(ArgumentError(
            "algorithm=$(repr(value)) is deprecated: algorithm-family " *
            "selection no longer exists because every MOI solve executes " *
            "the native product-HSD engine.  Use algorithm=:auto (the only " *
            "accepted value).",
        ))
    end
    if symbol === :threads
        value isa Integer && value > 0 ||
            throw(ArgumentError("threads must be a positive integer"))
        optimizer.requested_threads = Int(value)
    end
    if symbol === :iter_max && value isa Integer && value == 0
        # The MOI path funnels expert options through the public Settings
        # surface, where an iteration count of 0 is the *automatic*
        # sentinel and would silently resolve to the 200-iteration
        # default. Fail closed instead of misreporting the request.
        throw(ArgumentError(
            "max_iterations must be at least 1 on the MOI surface; " *
            "0 is the Settings automatic sentinel and cannot request a " *
            "zero-iteration solve here",
        ))
    end
    if symbol === :formulation &&
       !(value isa Symbol && value in (:auto, :normal_equations, :augmented))
        # :primal is a historical SolverOptions orientation flag.  It has no
        # one-to-one Settings representation and previously fell through to
        # :auto in _moi_settings, which made the requested option untruthful.
        throw(ArgumentError(
            "MOI formulation must be :auto, :normal_equations, or :augmented; " *
            "historical orientation values are not supported",
        ))
    end
    optimizer.options = _replace_option(optimizer.options, symbol, value)
    return nothing
end

# ---- model lifecycle ----

MOI.supports_incremental_interface(::Optimizer) = false

function MOI.empty!(optimizer::Optimizer{T}) where {T}
    optimizer.model = nothing
    optimizer.public_result = nothing
    empty!(optimizer.model_variables)
    empty!(optimizer.model_constraint_records)
    empty!(optimizer.model_constraint_starts)
    optimizer.start_error = nothing
    optimizer.num_variables = 0
    optimizer.sense = MOI.FEASIBILITY_SENSE
    optimizer.objective_constant = zero(T)
    optimizer.solve_time = 0.0
    return nothing
end

MOI.is_empty(optimizer::Optimizer) = optimizer.model === nothing

function Base.show(io::IO, optimizer::Optimizer{T}) where {T}
    if optimizer.public_result !== nothing
        result = optimizer.public_result
        print(io, "SDPX.Optimizer{$T} (", status(result), ")")
    else
        print(io, "SDPX.Optimizer{$T} (not solved)")
    end
end

# ---- supported model forms ----

const MOIPSDSet = Union{
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle},
}

const MOIVectorConicSet = Union{
    MOI.Nonnegatives,
    MOI.Nonpositives,
    MOI.Zeros,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    # ExponentialCone / PowerCone remain deliberately absent from the MOI
    # capability claim.  The public Model route now executes primal
    # Exp/Power blocks through native HSD, but the MOI adapter still stays
    # fail-closed until its standard MOI.Test coverage is green.
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
    ::Type{S},
) where {T,S<:MOIVectorConicSet}
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{S},
) where {S<:MOIVectorConicSet}
    return true
end

MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.Reals},
) = true

MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VectorAffineFunction{T}},
    ::Type{MOI.Reals},
) where {T} = true

const MOIScalarInequalitySet{T} = Union{MOI.GreaterThan{T},MOI.LessThan{T}}

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

# Unscaled product PSD variables are bridged by `_moi_vector_variable_groups`.
# The scaled product form is not; keep that capability false so MOI clients
# fail during discovery/copy rather than entering the unsupported branch
# (VectorAffineFunction PSD constraints remain supported above).
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{S},
) where {S<:MOIPSDSet}
    return true
end

MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}},
) = false

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

function _triangle_coordinates(side::Int)
    coordinates = Tuple{Int,Int}[]
    sizehint!(coordinates, psd_packed_length(side))
    for column in 1:side, row in 1:column
        push!(coordinates, (row, column))
    end
    return coordinates
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

function _new_constraint_index!(
    counts::Dict{Tuple{DataType,DataType},Int},
    ::Type{F},
    ::Type{S},
) where {F,S}
    key = (F, S)
    value = get(counts, key, 0) + 1
    counts[key] = value
    return MOI.ConstraintIndex{F,S}(value)
end

#=====================================================================#
# v0.5 Model-backed MOI bridge
#
# This section is the sole MOI adapter.  `copy_to` builds one authoritative
# `Model`, and `optimize!` invokes the public `SDPX.optimize!` seam exactly
# once. The retired draft conversion/solve path has been removed.
#=====================================================================#

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
        (attribute == MOI.VariableName() ||
         attribute == MOI.VariablePrimalStart()) && continue
        throw(MOI.UnsupportedAttribute(attribute))
    end
    for (F, S) in MOI.get(source, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(optimizer, F, S) ||
            throw(MOI.UnsupportedConstraint{F,S}())
        for attribute in MOI.get(source, MOI.ListOfConstraintAttributesSet{F,S}())
            (attribute == MOI.ConstraintName() ||
             attribute == MOI.ConstraintDualStart()) && continue
            throw(MOI.UnsupportedAttribute(attribute))
        end
    end
    return nothing
end

struct MOIVariableGroup{T<:AbstractFloat}
    source_constraint::MOI.ConstraintIndex
    set_kind::Symbol
    source_variables::Vector{MOI.VariableIndex}
    entries::Vector{VariableEntry{T}}
    block::VariableBlockRef{T}
    scaled::Bool
end

_moi_set_kind(::Type{<:MOI.Nonnegatives}) = :nonnegative
_moi_set_kind(::Type{<:MOI.Nonpositives}) = :nonpositive
_moi_set_kind(::Type{<:MOI.Zeros}) = :zero
_moi_set_kind(::Type{<:MOI.SecondOrderCone}) = :soc
_moi_set_kind(::Type{<:MOI.RotatedSecondOrderCone}) = :rsoc
_moi_set_kind(::Type{<:MOI.Reals}) = :free
_moi_set_kind(::Type{<:MOI.PositiveSemidefiniteConeTriangle}) = :psd
_moi_set_kind(::Type{<:MOI.Scaled{MOI.PositiveSemidefiniteConeTriangle}}) = :psd_scaled
_moi_set_kind(::Type{<:MOI.ExponentialCone}) = :exp
_moi_set_kind(::Type{<:MOI.GreaterThan}) = :nonnegative
_moi_set_kind(::Type{<:MOI.LessThan}) = :nonpositive
_moi_set_kind(::Type{<:MOI.EqualTo}) = :zero
_moi_set_kind(::Type{<:MOI.Interval}) = :interval

function _moi_route_family(::Type{S}) where {S}
    kind = try
        _moi_set_kind(S)
    catch exception
        _recoverable(exception) || rethrow()
        return nothing
    end
    family = _route_family(kind)
    return family in (:free, :zero) ? nothing : family
end

@inline _moi_model_name(prefix::AbstractString, number::Integer) =
    Symbol(prefix, "_", number)

# ConstraintIndex values are only unique within one (function, set) type
# pair.  Native Model block names, however, share one namespace per block
# kind, so using `index.value` alone lets (for example) a ScalarAffineFunction
# and a VariableIndex constraint both claim `moi_scalar_constraint_1`.  Keep
# names deterministic while including the full source function/set type in
# the name.  The type spelling is sanitized only for readability; the source
# type parameters themselves (including the arithmetic type) remain encoded.
@inline function _moi_type_tag(::Type{T}) where {T}
    return replace(string(T), r"[^A-Za-z0-9_]+" => "_")
end

@inline function _moi_model_name(
    prefix::AbstractString,
    index::MOI.ConstraintIndex{F,S},
) where {F,S}
    return Symbol(
        prefix,
        "_",
        _moi_type_tag(F),
        "_",
        _moi_type_tag(S),
        "_",
        index.value,
    )
end

function _moi_scalar_expression(
    model::Model{T},
    function_value,
    entries::Dict{Int,VariableEntry{T}},
) where {T<:AbstractFloat}
    if function_value isa MOI.ScalarAffineFunction{T}
        expression = _constant_affine(model, function_value.constant)
        for term in function_value.terms
            entry = get(entries, term.variable.value, nothing)
            entry === nothing && throw(ArgumentError(
                "MOI scalar term references an unknown variable $(term.variable)",
            ))
            iszero(term.coefficient) && continue
            expression = expression + term.coefficient * entry
        end
        return expression
    elseif function_value isa MOI.VariableIndex
        entry = get(entries, function_value.value, nothing)
        entry === nothing && throw(ArgumentError(
            "MOI variable function references an unknown variable $function_value",
        ))
        return +entry
    end
    throw(MOIAdapterError(
        :unsupported_function,
        "MOI scalar function $(typeof(function_value)) is not supported by the Model bridge",
    ))
end

function _moi_vector_expressions(
    model::Model{T},
    function_value,
    entries::Dict{Int,VariableEntry{T}},
    dimension::Int,
) where {T<:AbstractFloat}
    expressions = Vector{ScalarAffine{T}}(undef, dimension)
    if function_value isa MOI.VectorAffineFunction{T}
        MOI.output_dimension(function_value) == dimension || throw(DimensionMismatch(
            "MOI vector function dimension $(MOI.output_dimension(function_value)) != $dimension",
        ))
        for output in 1:dimension
            expressions[output] = _constant_affine(model, function_value.constants[output])
        end
        for term in function_value.terms
            1 <= term.output_index <= dimension || throw(DimensionMismatch(
                "MOI vector term output $(term.output_index) is outside 1:$dimension",
            ))
            entry = get(entries, term.scalar_term.variable.value, nothing)
            entry === nothing && throw(ArgumentError(
                "MOI vector term references an unknown variable $(term.scalar_term.variable)",
            ))
            coefficient = term.scalar_term.coefficient
            iszero(coefficient) && continue
            expressions[term.output_index] = expressions[term.output_index] + coefficient * entry
        end
    elseif function_value isa MOI.VectorOfVariables
        length(function_value.variables) == dimension || throw(DimensionMismatch(
            "MOI VectorOfVariables length $(length(function_value.variables)) != $dimension",
        ))
        for (output, variable) in pairs(function_value.variables)
            entry = get(entries, variable.value, nothing)
            entry === nothing && throw(ArgumentError(
                "MOI vector variable references an unknown variable $variable",
            ))
            expressions[output] = +entry
        end
    else
        throw(MOIAdapterError(
            :unsupported_function,
            "MOI vector function $(typeof(function_value)) is not supported by the Model bridge",
        ))
    end
    return expressions
end

@inline function _moi_matrix_precision(::Type{BigFloat}, matrix)
    isempty(matrix) && return precision(BigFloat)
    return maximum(precision(value) for value in matrix)
end

@inline _moi_matrix_precision(::Type{T}, matrix) where {T<:AbstractFloat} =
    precision(T)

function _moi_psd_matrix_expressions(
    expressions::Vector{ScalarAffine{T}},
    side::Int,
    scaled::Bool;
    precision_bits::Int=precision(T),
) where {T<:AbstractFloat}
    coordinates = _triangle_coordinates(side)
    length(expressions) == length(coordinates) || throw(DimensionMismatch(
        "PSD expression length $(length(expressions)) != packed side $side",
    ))
    matrix = Matrix{ScalarAffine{T}}(undef, side, side)
    sqrt_two = _owned_sqrt_two(T, precision_bits)
    for output in eachindex(coordinates)
        row, column = coordinates[output]
        expression = if scaled && row != column
            inverse_sqrt_two = _owned_arithmetic_eval(
                T,
                () -> one(T) / sqrt_two;
                precision_bits=precision_bits,
            )
            inverse_sqrt_two * expressions[output]
        else
            expressions[output]
        end
        matrix[row, column] = expression
        matrix[column, row] = expression
    end
    return matrix
end

function _moi_psd_vector_matrix(
    values,
    side::Int,
    ::Type{T},
    scaled::Bool,
    ; precision_bits::Int=precision(T),
) where {T<:AbstractFloat}
    coordinates = _triangle_coordinates(side)
    length(values) == length(coordinates) || throw(DimensionMismatch(
        "PSD start length $(length(values)) != packed side $side",
    ))
    matrix = zeros(T, side, side)
    sqrt_two = _owned_sqrt_two(T, precision_bits)
    for output in eachindex(coordinates)
        row, column = coordinates[output]
        value = owned_arithmetic_copy(T, values[output]; precision_bits=precision_bits)
        if scaled && row != column
            value = _owned_arithmetic_eval(
                T,
                () -> value / sqrt_two;
                precision_bits=precision_bits,
            )
        end
        matrix[row, column] = value
        matrix[column, row] = value
    end
    return matrix
end

function _moi_psd_vector_from_matrix(
    matrix::AbstractMatrix{T},
    side::Int,
    scaled::Bool,
    ; precision_bits::Int=_moi_matrix_precision(T, matrix),
) where {T<:AbstractFloat}
    size(matrix) == (side, side) || throw(DimensionMismatch(
        "PSD result matrix size $(size(matrix)) != ($side, $side)",
    ))
    coordinates = _triangle_coordinates(side)
    sqrt_two = _owned_sqrt_two(T, precision_bits)
    values = Vector{T}(undef, length(coordinates))
    for output in eachindex(coordinates)
        row, column = coordinates[output]
        value = owned_arithmetic_copy(T, matrix[row, column]; precision_bits=precision_bits)
        if scaled && row != column
            value = _owned_arithmetic_eval(
                T,
                () -> value * sqrt_two;
                precision_bits=precision_bits,
            )
        end
        values[output] = value
    end
    return values
end

function _moi_validate_family_set(constraint_types)
    # Mixed symmetric-cone families (LP+SOC, SOC+PSD, LP+PSD) are a
    # first-class executable layout via the universal PSD lift; this
    # validation only guards the supported cone kinds, never rejects a
    # mixed family.
    for (_, set_type) in constraint_types
        family = _moi_route_family(set_type)
        # The direct public Model route supports these families, but this MOI
        # adapter does not claim them yet; reject them rather than letting
        # them reach a mixed lift.
        if family in (:exp_family, :power_family)
            throw(UnsupportedNativeConeRoute([family]))
        end
    end
    return nothing
end

function _moi_vector_variable_groups(
    model::Model{T},
    source,
    constraint_types,
    source_variables::Vector{MOI.VariableIndex},
    entries::Dict{Int,VariableEntry{T}},
) where {T<:AbstractFloat}
    groups = MOIVariableGroup{T}[]
    by_constraint = Dict{MOIModelConstraintKey,MOIVariableGroup{T}}()
    assigned = Dict{Int,MOI.ConstraintIndex}()
    known_variables = Set{Int}(variable.value for variable in source_variables)
    for (F, S) in constraint_types
        F === MOI.VectorOfVariables || continue
        family = _moi_route_family(S)
        kind = _moi_set_kind(S)
        family === nothing && kind !== :free && throw(MOI.UnsupportedConstraint{F,S}())
        for source_index in MOI.get(source, MOI.ListOfConstraintIndices{F,S}())
            function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
            function_value isa MOI.VectorOfVariables || throw(MOIAdapterError(
                :unsupported_function,
                "VectorOfVariables set $(S) has function $(typeof(function_value))",
            ))
            set = MOI.get(source, MOI.ConstraintSet(), source_index)
            variables = collect(function_value.variables)
            isempty(variables) && throw(ArgumentError("VectorOfVariables constraint cannot be empty"))
            for variable in variables
                variable.value in known_variables || throw(ArgumentError(
                    "VectorOfVariables constraint references unknown variable $variable",
                ))
                haskey(assigned, variable.value) && throw(ArgumentError(
                    "variable $variable appears in multiple VectorOfVariables product blocks",
                ))
            end

            scaled = kind === :psd_scaled
            scaled && throw(MOIAdapterError(
                :unsupported_set,
                "Scaled PositiveSemidefiniteConeTriangle product variables are not supported",
            ))
            block = if kind === :psd
                side = MOI.side_dimension(set)
                expected = psd_packed_length(side)
                length(variables) == expected || throw(DimensionMismatch(
                    "PSD product variable count $(length(variables)) != packed side $side",
                ))
                variable!(model, _moi_model_name("moi_psd", source_index),
                          side, side; domain=PSDCone())
            elseif kind === :nonnegative
                variable!(model, _moi_model_name("moi_nonnegative", source_index),
                          length(variables); domain=Nonnegative())
            elseif kind === :nonpositive
                variable!(model, _moi_model_name("moi_nonpositive", source_index),
                          length(variables); domain=Nonpositive())
            elseif kind === :zero
                variable!(model, _moi_model_name("moi_zero", source_index),
                          length(variables); domain=ZeroCone())
            elseif kind === :soc
                variable!(model, _moi_model_name("moi_soc", source_index),
                          length(variables); domain=LorentzCone())
            elseif kind === :rsoc
                variable!(model, _moi_model_name("moi_rsoc", source_index),
                          length(variables); domain=RotatedLorentzCone())
            # `:exp`/`:power` deliberately fall through to the
            # `UnsupportedConstraint` below.  The asymmetric cone solver is not
            # wired into this adapter, so a VectorOfVariables Exp/Power block
            # must never be constructed here: doing so would silently create an
            # unsolvable cone block instead of failing closed.
            elseif kind === :free
                variable!(model, _moi_model_name("moi_free_product", source_index),
                          length(variables); domain=Reals())
            else
                throw(MOI.UnsupportedConstraint{F,S}())
            end
            model_entries = if kind === :psd
                coordinates = _triangle_coordinates(MOI.side_dimension(set))
                VariableEntry{T}[
                    block[max(row, column), min(row, column)]
                    for (row, column) in coordinates
                ]
            else
                VariableEntry{T}[block[position] for position in eachindex(variables)]
            end
            group = MOIVariableGroup{T}(
                source_index,
                kind,
                variables,
                model_entries,
                block,
                false,
            )
            push!(groups, group)
            by_constraint[_moi_constraint_key(source_index)] = group
            for variable in variables
                assigned[variable.value] = source_index
            end
        end
    end

    # Add every source variable not covered by a product block as one free
    # Model block.  This keeps MOI variable identities stable even when source
    # variables are non-contiguous or appear in a sparse affine map.
    covered = Set{Int}(variable.value for group in groups for variable in group.source_variables)
    for variable in source_variables
        variable.value in covered && continue
        block = variable!(model, _moi_model_name("moi_free", variable.value), 1; domain=Reals())
        entries[variable.value] = block[1]
    end
    # Product groups were built before free blocks; fill their entry map now
    # so affine conversion can use the same source-variable dictionary.
    for group in groups
        for (variable, entry) in zip(group.source_variables, group.entries)
            entries[variable.value] = entry
        end
    end
    return groups, by_constraint
end

function _moi_register_info!(
    optimizer::Optimizer{T},
    source_index::MOI.ConstraintIndex{F,S},
    info::MOIModelConstraintInfo{T},
    counts::Dict{Tuple{DataType,DataType},Int},
    index_map,
) where {T<:AbstractFloat,F,S}
    destination = _new_constraint_index!(counts, F, S)
    index_map[source_index] = destination
    optimizer.model_constraint_records[_moi_constraint_key(destination)] = info
    return destination
end

function _moi_model_constraint_refs(block::ConstraintBlockRef)
    return constraint_refs(block)
end

function _moi_add_vector_constraint!(
    model::Model{T},
    optimizer::Optimizer{T},
    source,
    source_index::MOI.ConstraintIndex{F,S},
    entries::Dict{Int,VariableEntry{T}},
    counts,
    index_map,
) where {T<:AbstractFloat,F,S}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    kind = _moi_set_kind(S)
    # PSD triangle sets expose both a side dimension and a packed output
    # dimension.  Every other supported vector cone is represented directly
    # by its ordinary MOI dimension.
    dimension = kind === :psd || kind === :psd_scaled ?
                (MOI.side_dimension(set) * (MOI.side_dimension(set) + 1) ÷ 2) :
                MOI.dimension(set)
    expressions = _moi_vector_expressions(model, function_value, entries, dimension)
    if kind === :psd || kind === :psd_scaled
        side = MOI.side_dimension(set)
        scaled = kind === :psd_scaled
        matrix = _moi_psd_matrix_expressions(
            expressions,
            side,
            scaled;
            precision_bits=precision_bits(model),
        )
        block = constraint!(model, _moi_model_name("moi_psd_constraint", source_index),
                            matrix, PSDCone())
        info = MOIModelConstraintInfo{T}(
            :psd, kind, expressions, _moi_model_constraint_refs(block),
            ConstraintRef[], VariableEntry{T}[], nothing, scaled,
            _moi_identity_map(T),
        )
    elseif kind === :free
        info = MOIModelConstraintInfo{T}(
            :free, kind, expressions, ConstraintRef[],
            ConstraintRef[], VariableEntry{T}[], nothing, false,
            _moi_identity_map(T),
        )
    elseif kind === :nonnegative || kind === :nonpositive || kind === :zero ||
           kind === :soc || kind === :rsoc
        # `:exp`/`:power` are intentionally absent: the asymmetric solver is
        # not wired into this adapter, so such a VectorAffineFunction block
        # must fail closed (the `UnsupportedConstraint` branch below) rather
        # than construct an ExponentialCone block that no solver can solve.
        domain = kind === :nonnegative ? Nonnegative() :
                 kind === :nonpositive ? Nonpositive() :
                 kind === :zero ? ZeroCone() :
                 kind === :soc ? LorentzCone() : RotatedLorentzCone()
        block = constraint!(model, _moi_model_name("moi_vector_constraint", source_index),
                            expressions, domain)
        info = MOIModelConstraintInfo{T}(
            :vector, kind, expressions, _moi_model_constraint_refs(block),
            ConstraintRef[], VariableEntry{T}[], nothing, false,
            _moi_identity_map(T),
        )
    else
        throw(MOI.UnsupportedConstraint{F,S}())
    end
    return _moi_register_info!(optimizer, source_index, info, counts, index_map)
end

function _moi_add_scalar_constraint!(
    model::Model{T},
    optimizer::Optimizer{T},
    source,
    source_index::MOI.ConstraintIndex{F,S},
    entries::Dict{Int,VariableEntry{T}},
    counts,
    index_map,
) where {T<:AbstractFloat,F,S}
    set = MOI.get(source, MOI.ConstraintSet(), source_index)
    function_value = MOI.get(source, MOI.ConstraintFunction(), source_index)
    expression = _moi_scalar_expression(model, function_value, entries)
    kind = _moi_set_kind(S)
    if kind === :interval
        lower_block = constraint!(model, _moi_model_name("moi_interval_lower", source_index),
                                  expression - set.lower, Nonnegative())
        upper_block = constraint!(model, _moi_model_name("moi_interval_upper", source_index),
                                  expression - set.upper, Nonpositive())
        info = MOIModelConstraintInfo{T}(
            :interval, kind, ScalarAffine{T}[expression],
            _moi_model_constraint_refs(lower_block),
            _moi_model_constraint_refs(upper_block),
            VariableEntry{T}[], nothing, false,
            _moi_identity_map(T),
        )
    else
        domain = kind === :nonnegative ? Nonnegative() :
                 kind === :nonpositive ? Nonpositive() : ZeroCone()
        shifted = if set isa MOI.GreaterThan{T}
            expression - set.lower
        elseif set isa MOI.LessThan{T}
            expression - set.upper
        else
            expression - set.value
        end
        block = constraint!(model, _moi_model_name("moi_scalar_constraint", source_index),
                            shifted, domain)
        info = MOIModelConstraintInfo{T}(
            :scalar, kind, ScalarAffine{T}[expression],
            _moi_model_constraint_refs(block), ConstraintRef[],
            VariableEntry{T}[], nothing, false,
            _moi_identity_map(T),
        )
    end
    return _moi_register_info!(optimizer, source_index, info, counts, index_map)
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

function MOI.supports(
    optimizer::Optimizer,
    attribute::MOI.RawOptimizerAttribute,
)
    # bridge_plan is a read-only diagnostic attribute, not a settable option.
    attribute.name == "bridge_plan" && return false
    attribute.name == "engine" && return true
    attribute.name == "tolerance" && return true
    try
        _option_symbol(optimizer.options, attribute.name)
        return true
    catch error
        error isa MOI.UnsupportedAttribute && return false
        rethrow()
    end
end

# Starts are copied into the authoritative Model blocks below.  Declaring the
# attributes here is important for JuMP/CachingOptimizer: otherwise the cache
# refuses to retain values before `copy_to` can inspect them.
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart) = true
MOI.supports(::Optimizer, ::MOI.ConstraintDualStart) = true
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true
MOI.supports(::Optimizer, ::MOI.ConstraintDualStart, ::Type{<:MOI.ConstraintIndex}) = true

function MOI.get(optimizer::Optimizer, attribute::MOI.RawOptimizerAttribute)
    attribute.name == "bridge_plan" && return bridge_plan(optimizer)
    attribute.name == "engine" && return optimizer.engine
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

#=====================================================================#
# Model-backed MOI lifecycle, starts, and result accessors
#=====================================================================#

@inline function _moi_has_attribute(attributes, attribute::DataType)
    return any(value -> value isa attribute, attributes)
end

function _moi_apply_raw_optimizer_attributes!(optimizer::Optimizer, source)
    attributes = try
        MOI.get(source, MOI.ListOfOptimizerAttributesSet())
    catch error
        error isa MOI.UnsupportedAttribute && return nothing
        rethrow()
    end
    for attribute in attributes
        attribute isa MOI.RawOptimizerAttribute || continue
        # CachingOptimizer deliberately does not replay raw attributes when
        # it attaches an optimizer. Re-apply them at the concrete copy seam,
        # where this adapter can validate the lossless whitelist and reject
        # expert fields instead of silently dropping them.
        _set_raw_option!(optimizer, attribute.name, MOI.get(source, attribute))
    end
    return nothing
end

function _moi_attribute_list(source, attribute)
    try
        return MOI.get(source, attribute)
    catch error
        error isa MOI.UnsupportedAttribute && return Any[]
        rethrow()
    end
end

function _moi_record_start_error!(optimizer::Optimizer, reason::Symbol, message::String)
    optimizer.start_error === nothing && (optimizer.start_error = (reason, message))
    return nothing
end

function _moi_fetch_variable_start(source, variable, attributes)
    _moi_has_attribute(attributes, MOI.VariablePrimalStart) || return nothing
    return MOI.get(source, MOI.VariablePrimalStart(), variable)
end

function _moi_fetch_constraint_dual_start(source, index, attributes)
    _moi_has_attribute(attributes, MOI.ConstraintDualStart) || return nothing
    return MOI.get(source, MOI.ConstraintDualStart(), index)
end

function _moi_install_variable_starts!(
    optimizer::Optimizer{T},
    source,
    source_variables::Vector{MOI.VariableIndex},
    groups::Vector{MOIVariableGroup{T}},
    entries::Dict{Int,VariableEntry{T}},
    variable_attributes,
) where {T<:AbstractFloat}
    starts = Dict{Int,Union{Nothing,T}}()
    for variable in source_variables
        raw = _moi_fetch_variable_start(
            source,
            variable,
            variable_attributes,
        )
        starts[variable.value] = raw === nothing ? nothing : owned_arithmetic_copy(T, raw)
    end

    for group in groups
        values = Union{Nothing,T}[starts[variable.value] for variable in group.source_variables]
        supplied = any(value -> value !== nothing, values)
        supplied || continue
        if !all(value -> value !== nothing, values)
            _moi_record_start_error!(optimizer, :warm_start_incomplete,
                "MOI VariablePrimalStart for product block $(group.source_constraint) " *
                "must cover every coordinate")
            continue
        end
        block = group.block
        if group.set_kind === :psd
            side = size(block)[1]
            matrix = _moi_psd_vector_matrix(
                values,
                side,
                T,
                false;
                precision_bits=precision_bits(block.model),
            )
            set_start!(block, matrix)
        else
            set_start!(block, T[owned_arithmetic_copy(T, value) for value in values])
        end
    end

    covered = Set{Int}(variable.value for group in groups for variable in group.source_variables)
    for variable in source_variables
        variable.value in covered && continue
        value = starts[variable.value]
        value === nothing && continue
        entry = entries[variable.value]
        block = VariableBlockRef{T}(entry.model, entry.ref.block)
        set_start!(block, T[owned_arithmetic_copy(T, value)])
    end
    return nothing
end

function _moi_install_constraint_start!(
    optimizer::Optimizer{T},
    source,
    source_index,
    info::MOIModelConstraintInfo{T},
    constraint_attributes,
) where {T<:AbstractFloat}
    value = _moi_fetch_constraint_dual_start(source, source_index, constraint_attributes)
    value === nothing && return nothing
    values = value isa Number ? [value] : collect(value)
    optimizer.model_constraint_starts[_moi_constraint_key(source_index)] =
        T[owned_arithmetic_copy(T, item) for item in values]

    if info.kind === :variable
        block = info.variable_block::VariableBlockRef{T}
        if info.set_kind === :psd
            set_dual_slack_start!(
                block,
                _moi_psd_vector_matrix(
                    values,
                    size(block)[1],
                    T,
                    info.scaled;
                    precision_bits=precision_bits(block.model),
                ),
            )
        else
            set_dual_slack_start!(block, T[owned_arithmetic_copy(T, item) for item in values])
        end
        return nothing
    end

    if info.kind === :interval
        _moi_record_start_error!(optimizer, :warm_start_core_gap,
            "MOI ConstraintDualStart for interval constraint $source_index " *
            "cannot be represented by the two native affine rows")
        return nothing
    end

    isempty(info.refs) && throw(ArgumentError(
        "Model-backed MOI constraint metadata has no native reference",
    ))
    model = optimizer.model::Model{T}
    native = ConstraintBlockRef{T}(model, info.refs[1].block)
    if info.set_kind === :psd || info.set_kind === :psd_scaled
        side = size(native)[1]
        set_dual_start!(
            native,
            _moi_psd_vector_matrix(
                values,
                side,
                T,
                info.scaled;
                precision_bits=precision_bits(model),
            ),
        )
    else
        set_dual_start!(native, T[owned_arithmetic_copy(T, item) for item in values])
    end
    return nothing
end

@inline function _moi_option_symbol(value, on::Symbol, off::Symbol, default::Symbol=:auto)
    value === true && return on
    value === false && return off
    value isa Symbol && return value
    return default
end

@inline function _moi_model_has_starts(model::Model)
    any(record -> record.primal_start !== nothing ||
                  record.dual_slack_start !== nothing,
        model.variable_blocks) ||
    any(record -> record.dual_start !== nothing, model.constraint_blocks)
end

function _moi_settings(optimizer::Optimizer{T}) where {T<:AbstractFloat}
    options = optimizer.options
    precision_scope = T === BigFloat ? options.precision_bits : precision(T)
    tolerances = Tolerances{T}(
        owned_arithmetic_copy(T, options.ϵ_primal; precision_bits=precision_scope),
        owned_arithmetic_copy(T, options.ϵ_dual; precision_bits=precision_scope),
        owned_arithmetic_copy(T, options.ϵ_gap; precision_bits=precision_scope),
    )
    limits = Limits(
        iterations=options.iter_max,
        time=options.max_time,
        threads=options.threads,
    )
    return Settings{T}(
        tolerances,
        limits,
        optimizer.engine,
        _moi_option_symbol(options.scaling, :equilibrate, :none),
        options.formulation === :normal_equations ? :variable_space_schur :
            options.formulation === :augmented ? :dense_augmented_kkt : :auto,
        :bordered,
        options.linear_algebra_backend,
        _moi_model_has_starts(optimizer.model::Model{T}) ? :off :
            _moi_option_symbol(options.presolve, :on, :off),
        options.algorithm,
        _moi_option_symbol(options.sparse, :on, :off),
        options.equality_solver,
        options.working_precision_policy,
        # SolverOptions carries diagnostics as a Bool; the public Settings
        # surface wants the symbolic level (:summary retains plan/phase
        # payloads, :none drops them).
        options.diagnostics ? :summary : :none,
        options.verbosity,
        options.timing,
        options.certification,
        nothing,
    )
end

function _moi_require_executable_cone(model::Model, engine::Symbol=:auto)
    # The direct native HSD route has an analytic equality/all-free path and
    # must not inherit the legacy lowerer's dummy-cone restriction.
    engine === :native_hsd && return nothing
    # The pure LP lowerer intentionally does not synthesize a dummy
    # inequality for an all-free/equality-only model.  Detect that case at
    # the MOI seam so callers receive a typed adapter error before any public
    # solver/result object is allocated.  ZeroCone is an equality and Reals
    # is unconstrained; every other native domain contributes an executable
    # cone coordinate.
    has_cone = any(
        record -> !(record.domain isa Reals || record.domain isa ZeroCone),
        model.variable_blocks,
    ) || any(
        record -> !(record.domain isa Reals || record.domain isa ZeroCone),
        model.constraint_blocks,
    )
    has_cone || throw(MOIAdapterError(
        :no_dummy_cone,
        "MOI model has no executable nonfree cone; all-free/equality-only " *
        "LP models are unsupported without a dummy inequality cone",
    ))
    return nothing
end

function _moi_install_objective!(
    optimizer::Optimizer{T},
    model::Model{T},
    source,
    entries::Dict{Int,VariableEntry{T}},
) where {T<:AbstractFloat}
    sense = MOI.get(source, MOI.ObjectiveSense())
    optimizer.sense = sense
    optimizer.objective_constant = zero(T)
    sense === MOI.FEASIBILITY_SENSE && return nothing
    function_type = MOI.get(source, MOI.ObjectiveFunctionType())
    expression = if function_type === MOI.ScalarAffineFunction{T}
        _moi_scalar_expression(
            model,
            MOI.get(source, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}()),
            entries,
        )
    elseif function_type === MOI.VariableIndex
        _moi_scalar_expression(
            model,
            MOI.get(source, MOI.ObjectiveFunction{MOI.VariableIndex}()),
            entries,
        )
    else
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{function_type}()))
    end
    objective!(
        model,
        sense === MOI.MAX_SENSE ? Maximize() : Minimize(),
        expression,
    )
    optimizer.objective_constant = owned_arithmetic_copy(T, expression.constant)
    return nothing
end

function _moi_source_name(source)
    try
        return String(MOI.get(source, MOI.Name()))
    catch error
        error isa MOI.UnsupportedAttribute && return "MOI"
        rethrow()
    end
end

function _moi_constraint_metadata(
    optimizer::Optimizer{T},
    group::MOIVariableGroup{T},
) where {T<:AbstractFloat}
    expressions = ScalarAffine{T}[+entry for entry in group.entries]
    return MOIModelConstraintInfo{T}(
        :variable,
        group.set_kind,
        expressions,
        ConstraintRef[],
        ConstraintRef[],
        copy(group.entries),
        group.block,
        group.scaled,
        _moi_identity_map(T),
    )
end

function MOI.copy_to(optimizer::Optimizer{T}, source::MOI.ModelLike) where {T<:AbstractFloat}
    # BigFloat arithmetic is ambient-precision sensitive.  Keep the entire
    # source conversion (Model construction, affine arithmetic, PSD scaling,
    # and warm-start ownership) inside the optimizer's requested precision;
    # otherwise Julia can round coefficients before the typed Model owns them.
    if T === BigFloat && Base.precision(BigFloat) != optimizer.options.precision_bits
        return setprecision(BigFloat, optimizer.options.precision_bits) do
            MOI.copy_to(optimizer, source)
        end
    end
    _moi_apply_raw_optimizer_attributes!(optimizer, source)
    _check_copy_attributes(optimizer, source)

    # Route-family detection intentionally precedes Model allocation.  A
    # mixed LP/SOC/SDP source is a first-class executable layout (universal
    # PSD lift); only exponential/power families fail closed here before any
    # numerical buffer or native solver object can be created.
    constraint_types = MOI.get(source, MOI.ListOfConstraintTypesPresent())
    _moi_validate_family_set(constraint_types)

    MOI.empty!(optimizer)
    model = if T === BigFloat
        Model(BigFloat;
              precision_bits=optimizer.options.precision_bits,
              name=_moi_source_name(source))
    else
        Model(T; name=_moi_source_name(source))
    end
    source_variables = collect(MOI.get(source, MOI.ListOfVariableIndices()))
    entries = Dict{Int,VariableEntry{T}}()
    groups, groups_by_constraint = _moi_vector_variable_groups(
        model,
        source,
        constraint_types,
        source_variables,
        entries,
    )
    counts = Dict{Tuple{DataType,DataType},Int}()
    index_map = MOI.Utilities.IndexMap()
    for (position, variable) in enumerate(source_variables)
        index_map[variable] = MOI.VariableIndex(position)
    end

    # Register product-cone variable constraints first, preserving their
    # source function/set types.  Their duals are represented by Model
    # variable dual-slacks rather than affine row duals.
    for group in groups
        destination = _moi_register_info!(
            optimizer,
            group.source_constraint,
            _moi_constraint_metadata(optimizer, group),
            counts,
            index_map,
        )
        groups_by_constraint[_moi_constraint_key(group.source_constraint)] = group
        index_map[group.source_constraint] = destination
    end

    # All remaining supported constraints become one typed affine Model
    # block each.  No second route-specific canonicalizer is called here.
    for (F, S) in constraint_types
        F === MOI.VectorOfVariables && continue
        for source_index in MOI.get(source, MOI.ListOfConstraintIndices{F,S}())
            if F <: MOI.VectorAffineFunction
                _moi_add_vector_constraint!(
                    model,
                    optimizer,
                    source,
                    source_index,
                    entries,
                    counts,
                    index_map,
                )
            elseif F <: MOI.ScalarAffineFunction || F === MOI.VariableIndex
                _moi_add_scalar_constraint!(
                    model,
                    optimizer,
                    source,
                    source_index,
                    entries,
                    counts,
                    index_map,
                )
            else
                throw(MOI.UnsupportedConstraint{F,S}())
            end
        end
    end

    optimizer.model = model
    optimizer.model_variables = VariableEntry{T}[entries[variable.value] for variable in source_variables]
    optimizer.num_variables = length(source_variables)

    variable_attributes = _moi_attribute_list(source, MOI.ListOfVariableAttributesSet())
    _moi_install_variable_starts!(
        optimizer,
        source,
        source_variables,
        groups,
        entries,
        variable_attributes,
    )

    # Constraint attributes are fetched per source function/set type so that
    # a cache can retain starts independently for scalar, vector and product
    # constraints.
    for (F, S) in constraint_types
        attributes = _moi_attribute_list(source, MOI.ListOfConstraintAttributesSet{F,S}())
        for source_index in MOI.get(source, MOI.ListOfConstraintIndices{F,S}())
            destination = index_map[source_index]
            info = optimizer.model_constraint_records[_moi_constraint_key(destination)]
            _moi_install_constraint_start!(
                optimizer,
                source,
                source_index,
                info,
                attributes,
            )
        end
    end

    _moi_install_objective!(optimizer, model, source, entries)
    optimizer.public_result = nothing
    return index_map
end

@inline _moi_public_result(optimizer::Optimizer) =
    optimizer.public_result

@inline _moi_result_status_value(result::Result) = result.status

@inline function _moi_check_public_result(optimizer::Optimizer, attribute)
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    hasfield(typeof(attribute), :result_index) &&
        MOI.check_result_index_bounds(optimizer, attribute)
    return result
end

function MOI.optimize!(optimizer::Optimizer{T}) where {T<:AbstractFloat}
    model = optimizer.model
    model === nothing && error("no model has been copied to SDPX")
    if optimizer.start_error !== nothing
        reason, message = optimizer.start_error
        throw(MOIAdapterError(reason, message))
    end

    _moi_require_executable_cone(model, optimizer.engine)

    settings = _moi_settings(optimizer)
    outputs = Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
        history=false,
        trace=false,
    )
    optimizer.solve_time = @elapsed begin
        # This is the sole active MOI numerical path.  The public Model
        # optimizer consumes the explicit Settings engine exactly once;
        # native_hsd therefore takes its direct canonical route and cannot
        # reach a family lowerer/lift/legacy fallback.
        result = optimize!(model; settings=settings, outputs=outputs)
        optimizer.public_result = result
    end
    return nothing
end

function _moi_eval_affine(
    optimizer::Optimizer{T},
    result,
    expression::ScalarAffine{T},
) where {T<:AbstractFloat}
    model = optimizer.model::Model{T}
    expression.model == model_identity(model) || throw(ArgumentError(
        "MOI affine expression belongs to a different Model",
    ))
    acc = owned_arithmetic_copy(T, expression.constant; precision_bits=precision_bits(model))
    for (index, coefficient) in zip(expression.indices, expression.coefficients)
        1 <= index <= length(model.variables) || throw(BoundsError(model.variables, index))
        primal = value(result, model.variables[index])
        acc = _owned_arithmetic_eval(
            T,
            () -> acc + coefficient * primal;
            precision_bits=precision_bits(model),
        )
    end
    return acc
end

function _moi_model_constraint_block(
    optimizer::Optimizer{T},
    refs::Vector{ConstraintRef},
) where {T<:AbstractFloat}
    isempty(refs) && throw(ArgumentError("empty Model constraint reference list"))
    return ConstraintBlockRef{T}(optimizer.model::Model{T}, refs[1].block)
end

function _moi_variable_primal(
    optimizer::Optimizer{T},
    result,
    info::MOIModelConstraintInfo{T},
) where {T<:AbstractFloat}
    model = optimizer.model::Model{T}
    block = info.variable_block::VariableBlockRef{T}
    if info.set_kind === :psd
        matrix = value(result, block)
        return _moi_psd_vector_from_matrix(
            matrix,
            size(block)[1],
            info.scaled;
            precision_bits=precision_bits(model),
        )
    end
    return T[value(result, entry.ref) for entry in info.entries]
end

function _moi_variable_dual(
    optimizer::Optimizer{T},
    result,
    info::MOIModelConstraintInfo{T},
) where {T<:AbstractFloat}
    model = optimizer.model::Model{T}
    block = info.variable_block::VariableBlockRef{T}
    if info.set_kind === :psd
        matrix = dual_slack(result, block)
        return _moi_psd_vector_from_matrix(
            matrix,
            size(block)[1],
            info.scaled;
            precision_bits=precision_bits(model),
        )
    end
    return T[dual_slack(result, entry.ref) for entry in info.entries]
end

function MOI.get(
    optimizer::Optimizer,
    attribute::MOI.ResultCount,
)
    result = _moi_public_result(optimizer)
    result === nothing && return 0
    return 1
end

function MOI.get(optimizer::Optimizer, ::MOI.RawSolver)
    result = _moi_public_result(optimizer)
    return result
end

MOI.Utilities.map_indices(::Any, ::MOI.RawSolver, result::Result) = result

MOI.get(optimizer::Optimizer, ::MOI.BarrierIterations) = begin
    result = _moi_public_result(optimizer)
    result === nothing ? 0 : getproperty(result, :iterations)
end

MOI.get(optimizer::Optimizer, ::MOI.RawStatusString) = begin
    result = _moi_public_result(optimizer)
    result === nothing ? "optimize! not called" :
        try
            getproperty(termination(result), :message)
        catch exception
            _recoverable(exception) || rethrow()
            string(_moi_result_status_value(result))
        end
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    result = _moi_public_result(optimizer)
    result === nothing && return MOI.OPTIMIZE_NOT_CALLED
    if optimizer.engine === :native_hsd &&
       _moi_result_status_value(result) == InsufficientPrecision
        return MOI.NUMERICAL_ERROR
    end
    return _moi_termination_status(_moi_result_status_value(result))
end

function _moi_termination_status(status_value)
    status_value == Optimal && return MOI.OPTIMAL
    status_value == FeasibleCert && return MOI.OPTIMAL
    status_value == InfeasibleCert && return MOI.INFEASIBLE
    status_value == PrimalInfeasible && return MOI.INFEASIBLE
    status_value == DualInfeasible && return MOI.DUAL_INFEASIBLE
    status_value == IterLimit && return MOI.ITERATION_LIMIT
    status_value == TimeLimit && return MOI.TIME_LIMIT
    status_value == UserStopped && return MOI.INTERRUPTED
    status_value == Stalled && return MOI.SLOW_PROGRESS
    status_value == MaxRestartsExceeded && return MOI.SLOW_PROGRESS
    status_value == AlmostOptimal && return MOI.ALMOST_OPTIMAL
    status_value == InsufficientPrecision && return MOI.SLOW_PROGRESS
    status_value == NumericalFailure && return MOI.NUMERICAL_ERROR
    return MOI.NUMERICAL_ERROR
end

function MOI.get(optimizer::Optimizer, attribute::MOI.PrimalStatus)
    result = _moi_public_result(optimizer)
    result === nothing && return MOI.NO_SOLUTION
    1 <= attribute.result_index <= MOI.get(optimizer, MOI.ResultCount()) || return MOI.NO_SOLUTION
    return _moi_primal_status(_moi_result_status_value(result), attribute.result_index, optimizer)
end

function _moi_primal_status(status_value, index::Int, optimizer)
    status_value in (Optimal, FeasibleCert) && return MOI.FEASIBLE_POINT
    status_value == DualInfeasible && return MOI.INFEASIBILITY_CERTIFICATE
    status_value in (PrimalInfeasible, InfeasibleCert) && return MOI.NO_SOLUTION
    status_value == AlmostOptimal && return MOI.NEARLY_FEASIBLE_POINT
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(optimizer::Optimizer, attribute::MOI.DualStatus)
    result = _moi_public_result(optimizer)
    result === nothing && return MOI.NO_SOLUTION
    1 <= attribute.result_index <= MOI.get(optimizer, MOI.ResultCount()) || return MOI.NO_SOLUTION
    return _moi_dual_status(_moi_result_status_value(result), attribute.result_index, optimizer)
end

function _moi_dual_status(status_value, index::Int, optimizer)
    status_value == Optimal && return MOI.FEASIBLE_POINT
    status_value == PrimalInfeasible && return MOI.INFEASIBILITY_CERTIFICATE
    status_value in (DualInfeasible, InfeasibleCert) && return MOI.NO_SOLUTION
    status_value == AlmostOptimal && return MOI.NEARLY_FEASIBLE_POINT
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(
    optimizer::Optimizer{T},
    attribute::MOI.ObjectiveValue,
) where {T<:AbstractFloat}
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    _moi_check_public_result(optimizer, attribute)
    if _moi_result_status_value(result) == PrimalInfeasible
        model = optimizer.model::Model{T}
        return owned_arithmetic_copy(T, NaN; precision_bits=precision_bits(model))
    end
    return primal_objective(result)
end

function MOI.get(
    optimizer::Optimizer{T},
    attribute::MOI.DualObjectiveValue,
) where {T<:AbstractFloat}
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    _moi_check_public_result(optimizer, attribute)
    if _moi_result_status_value(result) == DualInfeasible
        model = optimizer.model::Model{T}
        return owned_arithmetic_copy(T, NaN; precision_bits=precision_bits(model))
    end
    return dual_objective(result)
end

function MOI.get(
    optimizer::Optimizer{T},
    attribute::MOI.ObjectiveBound,
) where {T<:AbstractFloat}
    # The best certified bound on the optimal objective value.
    #
    # At a certified optimum the HSD gap certificate ties the primal and
    # dual objectives, so the tightest known bound is the dual objective.
    # For a verified infeasible/unbounded certificate the standard MOI
    # convention applies: a primal-infeasible problem has no primal
    # solution and the bound is the signed infinity of the objective sense
    # (±Inf), while a dual-infeasible (primal unbounded) problem reports
    # the opposing signed infinity. Every other non-optimal status has no
    # certified primal point, so the best certified bound is the
    # corresponding signed infinity rather than a fabricated finite value.
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    _moi_check_public_result(optimizer, attribute)
    status = _moi_result_status_value(result)
    if status in (Optimal, FeasibleCert, AlmostOptimal) ||
       (status == InsufficientPrecision && optimizer.engine !== :native_hsd)
        return dual_objective(result)
    end
    model = optimizer.model::Model{T}
    bits = precision_bits(model)
    lower_bound = optimizer.sense == MOI.MIN_SENSE
    sign = if status in (PrimalInfeasible, InfeasibleCert)
        # no primal point exists: objective is +Inf for MIN, -Inf for MAX
        lower_bound ? 1 : -1
    else
        # dual-infeasible (primal unbounded) or no certificate: best known
        # finite bound is the trivial -Inf (MIN) / +Inf (MAX) ray
        lower_bound ? -1 : 1
    end
    return owned_arithmetic_copy(T, sign * Inf; precision_bits=bits)
end

function MOI.get(optimizer::Optimizer, attribute::MOI.RelativeGap)
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    _moi_check_public_result(optimizer, attribute)
    return getproperty(certificate(result), :relative_gap)
end

function MOI.get(
    optimizer::Optimizer,
    attribute::MOI.VariablePrimal,
    variable::MOI.VariableIndex,
)
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    _moi_check_public_result(optimizer, attribute)
    1 <= variable.value <= length(optimizer.model_variables) ||
        throw(BoundsError(optimizer.model_variables, variable.value))
    return value(result, optimizer.model_variables[variable.value].ref)
end

function MOI.get(
    optimizer::Optimizer,
    attribute::MOI.ConstraintPrimal,
    index::MOI.ConstraintIndex,
)
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    _moi_check_public_result(optimizer, attribute)
    info = get(optimizer.model_constraint_records, _moi_constraint_key(index), nothing)
    info === nothing && throw(MOI.InvalidIndex(index))
    info.kind === :variable && return _moi_variable_primal(optimizer, result, info)
    model = optimizer.model::Model
    T = eltype(model)
    values = T[_moi_eval_affine(optimizer, result, expression)
               for expression in info.expressions]
    info.kind === :psd && return values
    info.kind === :vector && return values
    info.kind === :free && return values
    info.kind === :interval && return values[1]
    return values[1]
end

function MOI.get(
    optimizer::Optimizer{T},
    attribute::MOI.ConstraintDual,
    index::MOI.ConstraintIndex,
) where {T<:AbstractFloat}
    result = _moi_public_result(optimizer)
    result === nothing && throw(MOI.GetAttributeNotAllowed(attribute))
    _moi_check_public_result(optimizer, attribute)
    info = get(optimizer.model_constraint_records, _moi_constraint_key(index), nothing)
    info === nothing && throw(MOI.InvalidIndex(index))
    model = optimizer.model::Model
    if info.kind === :variable
        return _moi_variable_dual(optimizer, result, info)
    elseif info.kind === :psd
        matrix = dual(result, _moi_model_constraint_block(optimizer, info.refs))
        return _moi_psd_vector_from_matrix(
            matrix,
            size(matrix, 1),
            info.scaled;
            precision_bits=precision_bits(model),
        )
    elseif info.kind === :interval
        lower_dual = dual(result, info.refs[1])
        upper_dual = dual(result, info.aux_refs[1])
        # The interval is bridged into (f - l) ∈ Nonnegative() and
        # (f - u) ∈ Nonpositive(); MOI's interval dual is their sum —
        # nonnegative when the lower bound is active, nonnegative-bound
        # sign flipped at the upper bound (upper_dual <= 0 there).
        return _owned_arithmetic_eval(
            T,
            () -> lower_dual + upper_dual;
            precision_bits=precision_bits(model),
        )
    elseif info.kind === :free
        return [
            owned_arithmetic_copy(T, 0; precision_bits=precision_bits(model))
            for _ in info.expressions
        ]
    elseif info.kind === :vector
        return [dual(result, ref) for ref in info.refs]
    end
    return dual(result, info.refs[1])
end

function _constraint_bridge_metadata(info::MOIModelConstraintInfo)
    return (kind=info.kind, representation=info.set_kind)
end

function bridge_plan(optimizer::Optimizer)
    if optimizer.model === nothing
        return (
            constraints=NamedTuple[],
            route=:none,
        )
    end
    model = optimizer.model
    families = Symbol[]
    for record in model.variable_blocks
        family = _route_family(record.domain)
        family === :free || family === :zero || (family in families || push!(families, family))
    end
    for record in model.constraint_blocks
        family = _route_family(record.domain)
        family === :free || family === :zero || (family in families || push!(families, family))
    end
    route = if isempty(families)
        :lp_family
    elseif length(families) == 1
        only(families)
    else
        # A heterogeneous symmetric-cone product (LP+SOC, SOC+PSD, LP+PSD).
        # Under the non-direct `:auto` engine this is served by the universal
        # PSD lift as a *fallback* route (plan §2.4/§5.11); the preferred
        # native mixed route is `engine=:native_hsd`.
        :mixed_family
    end
    return (
        constraints=[_constraint_bridge_metadata(info) for info in values(optimizer.model_constraint_records)],
        route=route,
    )
end

# Stable MOI optimizer/model attributes.  These deliberately report the
# copied Model state; no alternate problem/result object is consulted.
MOI.get(::Optimizer, ::MOI.SolverName) = "SDPX"
function MOI.get(::Optimizer, ::MOI.SolverVersion)
    version = Base.pkgversion(@__MODULE__)
    return version === nothing ? "unknown" : string(version)
end
MOI.get(optimizer::Optimizer, ::MOI.NumberOfVariables) = optimizer.num_variables
MOI.get(optimizer::Optimizer, ::MOI.ObjectiveSense) = optimizer.sense
function MOI.set(
    optimizer::Optimizer{T},
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
) where {T<:AbstractFloat}
    sense in (MOI.MIN_SENSE, MOI.MAX_SENSE, MOI.FEASIBILITY_SENSE) ||
        throw(ArgumentError("unsupported MOI objective sense $sense"))

    model = optimizer.model
    if model !== nothing
        if sense === MOI.FEASIBILITY_SENSE
            # MOI's feasibility sense removes the objective entirely.  The
            # copied Model has no feasibility marker, so represent that state
            # as an absent ObjectiveRecord; compilation then uses its typed
            # zero objective while the adapter reports FEASIBILITY_SENSE.
            model.objective = nothing
            optimizer.objective_constant = owned_arithmetic_copy(
                T,
                0;
                precision_bits=precision_bits(model),
            )
        elseif model.objective === nothing
            # A copied feasibility model has no ObjectiveRecord.  MOI's
            # default objective is the zero function, so installing that
            # record makes a subsequent MIN/MAX setter truthful and keeps the
            # public Model/optimizer senses aligned.
            objective!(
                model,
                sense === MOI.MAX_SENSE ? Maximize() : Minimize(),
                _constant_affine(model, zero(T)),
            )
        else
            objective = model.objective
            model.objective = ObjectiveRecord{T}(
                sense === MOI.MAX_SENSE ? Maximize() : Minimize(),
                objective.expression,
            )
        end
        # Any installed result corresponds to the old model objective.
        optimizer.public_result = nothing
        optimizer.solve_time = 0.0
    end
    optimizer.sense = sense
    return nothing
end
MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec) = optimizer.solve_time

function MOI.get(optimizer::Optimizer{T}, ::MOI.ObjectiveFunctionType) where {T}
    return MOI.ScalarAffineFunction{T}
end

function MOI.get(
    optimizer::Optimizer{T},
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
) where {T}
    model = optimizer.model
    bits = model === nothing ?
        (T === BigFloat ? optimizer.options.precision_bits : precision(T)) :
        precision_bits(model)
    owned_zero = owned_arithmetic_copy(T, 0; precision_bits=bits)
    model === nothing && return MOI.ScalarAffineFunction{T}(MOI.ScalarAffineTerm{T}[], owned_zero)
    objective = model.objective
    objective === nothing && return MOI.ScalarAffineFunction{T}(MOI.ScalarAffineTerm{T}[], owned_zero)
    expression = objective.expression
    terms = MOI.ScalarAffineTerm{T}[
        MOI.ScalarAffineTerm{T}(coefficient, MOI.VariableIndex(index))
        for (index, coefficient) in zip(expression.indices, expression.coefficients)
    ]
    return MOI.ScalarAffineFunction{T}(terms, expression.constant)
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.NumberOfConstraints{F,S},
) where {F,S}
    return count(
        key -> key[1] === MOI.ConstraintIndex{F,S},
        keys(optimizer.model_constraint_records),
    )
end

MOI.get(optimizer::Optimizer, ::MOI.ListOfVariableIndices) =
    MOI.VariableIndex[MOI.VariableIndex(index) for index in 1:optimizer.num_variables]

function MOI.get(optimizer::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    types = Tuple{DataType,DataType}[]
    for key in keys(optimizer.model_constraint_records)
        type = key[1]
        pair = (type.parameters[1], type.parameters[2])
        pair in types || push!(types, pair)
    end
    return types
end

function MOI.get(
    optimizer::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {F,S}
    indices = MOI.ConstraintIndex{F,S}[]
    for key in keys(optimizer.model_constraint_records)
        key[1] === MOI.ConstraintIndex{F,S} || continue
        push!(indices, MOI.ConstraintIndex{F,S}(key[2]))
    end
    sort!(indices, by=index -> index.value)
    return indices
end

MOI.get(optimizer::Optimizer, ::MOI.ListOfModelAttributesSet) = MOI.AbstractModelAttribute[]
MOI.get(optimizer::Optimizer, ::MOI.ListOfVariableAttributesSet) =
    MOI.AbstractVariableAttribute[MOI.VariablePrimalStart(), MOI.VariableName()]
function MOI.get(
    optimizer::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{F,S},
) where {F,S}
    found = any(
        key -> key[1] === MOI.ConstraintIndex{F,S} &&
               haskey(optimizer.model_constraint_starts, key),
        keys(optimizer.model_constraint_records),
    )
    return found ? MOI.AbstractConstraintAttribute[MOI.ConstraintDualStart()] :
        MOI.AbstractConstraintAttribute[]
end

for attribute in (
    :TerminationStatus,
    :PrimalStatus,
    :DualStatus,
    :ObjectiveValue,
    :DualObjectiveValue,
    :ObjectiveBound,
    :RelativeGap,
    :VariablePrimal,
    :ConstraintPrimal,
    :ConstraintDual,
)
    # Attribute-specific `get` methods above carry the actual behavior; this
    # loop only installs the required protocol declarations at runtime.
    @eval MOI.supports(::Optimizer, ::MOI.$attribute) = true
end
