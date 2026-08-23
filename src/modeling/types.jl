#=====================================================================#
#    v0.5 Model type and immutable arithmetic/precision metadata.
#
#    `Model{T}` is the user-facing mathematical model shell. It owns
#    one `ArithmeticSpec{T}` (the immutable arithmetic/precision
#    contract of every number that will enter the model) plus minimal
#    mutable state for the B1 builder: monotonically increasing
#    1-based id counters and the variable/constraint reference
#    registries. Every stored reference is a `VariableRef` /
#    `ConstraintRef` — concrete, typed vectors, never `Dict`/`Any`.
#
#    The model deliberately holds *no* constraint data, no objective,
#    no dualization state, and no pointer to any other runtime model.
#
#    Include order: after domains.jl and refs.jl; before ir/types.jl.
#=====================================================================#

const _MULTIFLOATS_UUID = Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5")
const _MULTIFLOATS_PKGID = Base.PkgId(_MULTIFLOATS_UUID, "MultiFloats")

"""
    _multifloats_loaded() -> Bool

Whether the `MultiFloats` package is currently loaded (by UUID). The
foundation never `import`s `MultiFloats`, so `Float64x2`/`Float64x4`
models are only constructible when the user has loaded the package —
this keeps the type system independent of the optional extension.
"""
function _multifloats_loaded()
    return get(Base.loaded_modules, _MULTIFLOATS_PKGID, nothing) !== nothing
end

"""
    _multifloat_type_available(::Type{T}) -> Bool

Whether `T` is a MultiFloat type available in the loaded `MultiFloats`
module. Uses the type's own name binding; never fabricates the type.
"""
function _multifloat_type_available(::Type{T}) where {T}
    module_ = get(Base.loaded_modules, _MULTIFLOATS_PKGID, nothing)
    module_ === nothing && return false
    return isdefined(module_, nameof(T))
end

"""
    SDPX.ArithmeticSpec{T<:AbstractFloat}

Immutable arithmetic/precision metadata owned by a `SDPX.Model`.

Fields
- `precision_bits::Int`: nominal precision of the arithmetic. `53` for
  `Float64` (IEEE binary64 mantissa bits), the requested
  `precision_bits` (any integer at least `2`, e.g. `128`, `256`, `412`,
  or `512`) for `BigFloat`, and an effective mantissa estimate
  `52 * (sizeof(T) ÷ 8)` for MultiFloat types
  (`104` for `Float64x2`, `208` for `Float64x4`).
- `supports_multifloat::Bool`: whether `T` is a MultiFloat type
  (requires the optional `MultiFloats` package to be loaded).
"""
struct ArithmeticSpec{T<:AbstractFloat}
    precision_bits::Int
    supports_multifloat::Bool
end

ArithmeticSpec(::Type{Float64}) = ArithmeticSpec{Float64}(53, false)

function ArithmeticSpec(::Type{BigFloat}; precision_bits::Int=256)
    precision_bits >= 2 ||
        throw(ArgumentError("BigFloat model precision_bits must be at least 2, got $precision_bits"))
    return ArithmeticSpec{BigFloat}(precision_bits, false)
end

function ArithmeticSpec(::Type{T}) where {T<:AbstractFloat}
    T === Float64 && return ArithmeticSpec(Float64)
    T === BigFloat &&
        throw(ArgumentError("BigFloat models require Model(BigFloat; precision_bits=...)"))
    if isbitstype(T) && sizeof(T) > sizeof(Float64)
        _multifloat_type_available(T) ||
            throw(ArgumentError("arithmetic type $T requires the MultiFloats package to be loaded"))
        nominal_bits = 52 * (sizeof(T) ÷ sizeof(Float64))
        return ArithmeticSpec{T}(nominal_bits, true)
    end
    throw(ArgumentError("unsupported arithmetic type $T; use Float64, BigFloat, Float64x2 or Float64x4"))
end

"""Internal, typed record for one native variable block."""
struct VariableBlockRecord{T<:AbstractFloat}
    name::Symbol
    domain::ProductConeDomain
    shape::Int
    offset::Int
    length::Int
    primal_start::Union{Nothing,Vector{T}}
    dual_slack_start::Union{Nothing,Vector{T}}
end

"""A scalar affine expression in the model's global packed variable order."""
struct ScalarAffine{T<:AbstractFloat}
    model::UInt64
    precision_bits::Int
    indices::Vector{Int}
    coefficients::Vector{T}
    constant::T
end

"""Internal, typed record for one affine-in-cone constraint block."""
struct AffineConstraintRecord{T<:AbstractFloat}
    name::Symbol
    domain::ProductConeDomain
    shape::Int
    expressions::Vector{ScalarAffine{T}}
    refs::Vector{ConstraintRef}
    dual_start::Union{Nothing,Vector{T}}
end

"""Internal, typed record for the single scalar affine objective."""
struct ObjectiveRecord{T<:AbstractFloat}
    sense::Union{Minimize,Maximize}
    expression::ScalarAffine{T}
end

"""
    SDPX.Model{T<:AbstractFloat}

User-facing mathematical model for the v0.5 frontend. `T` is the
arithmetic type of every number that enters the model; the immutable
`arithmetic::ArithmeticSpec{T}` records and validates the precision
contract.

Constructors
- `Model(Float64)` — IEEE binary64 arithmetic.
- `Model(BigFloat; precision_bits=256)` — arbitrary-precision arithmetic;
  any integer `precision_bits >= 2` (for example `128`, `256`, `412`, or `512`)
  is accepted.
- `Model(Float64x2)` / `Model(Float64x4)` — only when the optional
  `MultiFloats` package is loaded.

The mutable fields are the minimal builder state for B1: 1-based id
counters, concrete `Vector{VariableRef}` / `Vector{ConstraintRef}`
registries, and the typed variable-block records owned by the model
(`VariableBlockRecord{T}`; see modeling/model.jl). Constraint and
objective data, dualization state, and solver choices never live here.
The block registry uses concrete typed vectors and a `Dict{Symbol,Int}`
name→record map; no `Any`-typed state is stored.
"""
mutable struct Model{T<:AbstractFloat}
    arithmetic::ArithmeticSpec{T}
    identity::UInt64
    name::String
    next_variable_id::Int
    next_constraint_id::Int
    next_block_id::Int
    variables::Vector{VariableRef}
    constraints::Vector{ConstraintRef}
    variable_blocks::Vector{VariableBlockRecord{T}}
    block_names::Dict{Symbol,Int}
    constraint_blocks::Vector{AffineConstraintRecord{T}}
    constraint_names::Dict{Symbol,Int}
    objective::Union{Nothing,ObjectiveRecord{T}}
end

# A model identity must remain unique even after the garbage collector reuses
# an object address.  The process-local monotone counter is thread-safe and
# references still store only the resulting value, never a model pointer.
const _MODEL_ID_COUNTER = Base.Threads.Atomic{UInt64}(0)

function _next_model_identity()
    identity = Base.Threads.atomic_add!(_MODEL_ID_COUNTER, UInt64(1)) + UInt64(1)
    identity == 0 && throw(OverflowError("SDPX Model identity counter exhausted"))
    return identity
end

Model(::Type{Float64}; name::AbstractString="") =
    Model{Float64}(
        ArithmeticSpec(Float64),
        _next_model_identity(),
        String(name),
        1,
        1,
        1,
        VariableRef[],
        ConstraintRef[],
        VariableBlockRecord{Float64}[],
        Dict{Symbol,Int}(),
        AffineConstraintRecord{Float64}[],
        Dict{Symbol,Int}(),
        nothing,
    )

function Model(::Type{BigFloat}; precision_bits::Int=256, name::AbstractString="")
    spec = ArithmeticSpec(BigFloat; precision_bits=precision_bits)
    return Model{BigFloat}(
        spec,
        _next_model_identity(),
        String(name),
        1,
        1,
        1,
        VariableRef[],
        ConstraintRef[],
        VariableBlockRecord{BigFloat}[],
        Dict{Symbol,Int}(),
        AffineConstraintRecord{BigFloat}[],
        Dict{Symbol,Int}(),
        nothing,
    )
end

function Model(::Type{T}; name::AbstractString="") where {T<:AbstractFloat}
    return Model{T}(
        ArithmeticSpec(T),
        _next_model_identity(),
        String(name),
        1,
        1,
        1,
        VariableRef[],
        ConstraintRef[],
        VariableBlockRecord{T}[],
        Dict{Symbol,Int}(),
        AffineConstraintRecord{T}[],
        Dict{Symbol,Int}(),
        nothing,
    )
end

Base.eltype(::Type{Model{T}}) where {T} = T
Base.eltype(::Model{T}) where {T} = T

"""
    model_identity(model) -> UInt64

Opaque monotone identity of `model` used by every `VariableRef` /
`ConstraintRef` it owns. It is never a pointer to the model and is never
exposed to the solver layer as a handle.
"""
model_identity(model::Model) = model.identity

"""
    arithmetic(model) -> ArithmeticSpec{T}

Immutable arithmetic/precision metadata of `model`.
"""
arithmetic(model::Model) = model.arithmetic

"""
    precision_bits(model) -> Int

Nominal precision of `model`'s arithmetic (`53` for `Float64`, the
requested `precision_bits >= 2` for `BigFloat`, mantissa estimate for
MultiFloat types).
"""
precision_bits(model::Model) = model.arithmetic.precision_bits

"""
    num_variables(model), num_constraints(model)

Number of frontend variables / constraints registered so far. Both
counts are maintained by the B1 builder through the 1-based id
counters; the foundation only provides the accessors.
"""
num_variables(model::Model) = length(model.variables)
num_constraints(model::Model) = length(model.constraints)

function Base.show(io::IO, model::Model{T}) where {T}
    print(io, "Model{", T, "}(precision_bits=", model.arithmetic.precision_bits,
          ", name=", repr(model.name),
          ", variables=", num_variables(model), ", constraints=", num_constraints(model),
          ", blocks=", length(model.variable_blocks), ")")
end
