# Immutable ownership receipt for one actual numeric factor epoch.
#
# A receipt caches only factor-wide facts. RHS-dependent finite checks,
# factor-coordinate residuals, exact-operator residuals, and the authoritative
# five Newton equations are intentionally excluded and must run for every RHS.

struct FactorReceipt{T<:AbstractFloat}
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
    for byte in codeunits(String(route))
        signature = (signature ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
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
    return true
end
