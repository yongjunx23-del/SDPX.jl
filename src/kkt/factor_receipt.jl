# Immutable ownership receipt for one actual numeric factor epoch.
#
# A receipt caches only factor-wide facts. RHS-dependent finite checks,
# factor-coordinate residuals, exact-operator residuals, and the authoritative
# five Newton equations are intentionally excluded and must run for every RHS.

mutable struct FactorReceipt{T<:AbstractFloat}
    matrix_epoch::Int
    factor_epoch::Int
    pattern_signature::UInt64
    route::Symbol
    provider::Symbol
    scalar_type::DataType
    precision_bits::Int
    regularization::T
    regularization_kind::Symbol
    factor_status::Symbol
    factor_backward_bound::T
    proof_valid::Bool
    # Explicit mutation tokens: every in-place rewrite of the operator or of
    # the numeric factor buffer bumps the corresponding generation *before*
    # any early exit, so a receipt can never survive a mutation that its
    # epoch counters cannot observe (e.g. an exact-border operator rewrite).
    operator_generation::Int
    factor_generation::Int
end

@inline function update_factor_receipt!(
    receipt::FactorReceipt{T},
    matrix_epoch::Int,
    factor_epoch::Int,
    pattern_signature::UInt64,
    route::Symbol,
    provider::Symbol,
    regularization::T,
    regularization_kind::Symbol,
    factor_status::Symbol,
    factor_backward_bound::T,
    proof_valid::Bool,
    operator_generation::Int=0,
    factor_generation::Int=0,
) where {T<:AbstractFloat}
    receipt.matrix_epoch = matrix_epoch
    receipt.factor_epoch = factor_epoch
    receipt.pattern_signature = pattern_signature
    receipt.route = route
    receipt.provider = provider
    receipt.scalar_type = T
    receipt.precision_bits = factor_receipt_precision(T)
    receipt.regularization = regularization
    receipt.regularization_kind = regularization_kind
    receipt.factor_status = factor_status
    receipt.factor_backward_bound = factor_backward_bound
    receipt.proof_valid = proof_valid
    receipt.operator_generation = operator_generation
    receipt.factor_generation = factor_generation
    return receipt
end

@inline function factor_receipt_precision(::Type{T}) where {T<:AbstractFloat}
    T === BigFloat && return precision(BigFloat)
    value = try
        precision(T)
    catch
        8 * sizeof(T)
    end
    return Int(value)
end

@inline function dense_factor_pattern_signature(
    rows::Integer, columns::Integer, route::Symbol,
)
    signature = UInt64(0xcbf29ce484222325)
    signature = (signature ⊻ UInt64(rows)) * UInt64(0x100000001b3)
    signature = (signature ⊻ UInt64(columns)) * UInt64(0x100000001b3)
    # Keep the signature reproducible without allocating `String(route)`.
    # Only the three dense factor routes call this helper; explicit tags avoid
    # depending on Julia's implementation-defined `hash(::Symbol)` seed.
    route_tag = route === :bordered ? UInt64(0x01) :
                route === :expanded ? UInt64(0x02) :
                route === :coupled ? UInt64(0x03) :
                throw(ArgumentError("unsupported dense factor route $(route)"))
    signature = (signature ⊻ route_tag) * UInt64(0x100000001b3)
    return signature
end

@inline function factor_receipt_owned(
    receipt::Union{Nothing,FactorReceipt{T}};
    matrix_epoch::Int,
    factor_epoch::Int,
    pattern_signature::UInt64,
    route::Symbol,
    provider::Symbol,
    regularization::T,
    factor_status::Symbol=:factored,
    require_proof::Bool=false,
    operator_generation::Int=0,
    factor_generation::Int=0,
) where {T<:AbstractFloat}
    receipt === nothing && return false
    receipt.matrix_epoch == matrix_epoch || return false
    receipt.factor_epoch == factor_epoch || return false
    receipt.pattern_signature == pattern_signature || return false
    receipt.route === route || return false
    receipt.provider === provider || return false
    receipt.scalar_type === T || return false
    receipt.precision_bits == factor_receipt_precision(T) || return false
    receipt.regularization == regularization || return false
    receipt.factor_status === factor_status || return false
    require_proof && !receipt.proof_valid && return false
    receipt.proof_valid && !isfinite(receipt.factor_backward_bound) && return false
    receipt.operator_generation == operator_generation || return false
    receipt.factor_generation == factor_generation || return false
    return true
end
