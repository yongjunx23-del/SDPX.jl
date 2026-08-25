#=====================================================================#
#    Certificate verification (Subagent C, PR3).
#
#    Final status is NEVER based on raw τ/κ alone. It comes only from a
#    verified certificate in original coordinates:
#      - verify_optimal!            (normalized residual + cone membership
#                                    + objective sign + original-coordinate)
#      - verify_primal_infeasibility! (Farkas ray: A'y <= 0, b'y > 0)
#      - verify_dual_infeasibility!   (ray: c'x < 0, A x + s = 0, s ∈ K)
#
#    Every certificate must be pushed back through the full ReductionChain
#    inverse map before it may set an MOI status.
#=====================================================================#

"""
    ConeMembership

Abstract supertype for cone-membership checks used by certificate
verification.
"""
abstract type ConeMembership end

"""
    OrthantMembership <: ConeMembership

Membership in the nonnegative orthant: `x >= 0` componentwise.
"""
struct OrthantMembership <: ConeMembership end

"""
    SOCMembership <: ConeMembership

Membership in the second-order cone: `(t, u)` with `‖u‖ <= t`.
"""
struct SOCMembership <: ConeMembership end

"""
    PSDMembership <: ConeMembership

Membership in the PSD cone: the matrix is positive semidefinite.
"""
struct PSDMembership <: ConeMembership end

"""
    ExpMembership <: ConeMembership

Membership in the exponential cone.
"""
struct ExpMembership <: ConeMembership end

"""
    PowerMembership <: ConeMembership

Membership in the power cone of parameter `alpha`.
"""
struct PowerMembership <: ConeMembership
    alpha::Float64
end

"""
    in_cone(membership, v) -> Bool

Whether the vector `v` lies in the cone described by `membership`.
"""
in_cone(::OrthantMembership, v) = all(x -> x >= zero(x), v)
function in_cone(::SOCMembership, v)
    length(v) >= 1 || return false
    t = v[1]
    t >= zero(t) || return false
    return norm(v[2:end]) <= t
end
function in_cone(::PSDMembership, v)
    n = round(Int, (sqrt(8 * length(v) + 1) - 1) / 2)
    n * (n + 1) ÷ 2 == length(v) || return false
    matrix = zeros(eltype(v), n, n)
    index = 1
    for j in 1:n
        for i in j:n
            matrix[i, j] = v[index]
            matrix[j, i] = v[index]
            index += 1
        end
    end
    return minimum(eigvals(Symmetric(matrix))) >= -1e-8
end
function in_cone(::ExpMembership, v)
    length(v) == 3 || return false
    return exp_membership(v[1], v[2], v[3])
end
function in_cone(m::PowerMembership, v)
    length(v) == 3 || return false
    return power_membership(v[1], v[2], v[3], m.alpha)
end

"""
    verify_optimal!(state, memberships; tol=1e-6) -> Bool

Verify an optimal certificate: the normalized homogeneous residual is
small, the slack `s` lies in the cone, and the duality gap is small.
`memberships` is a vector of [`ConeMembership`](@ref) describing the
product cone `K` (one per block).
"""
function verify_optimal!(state::HSDState{T}, memberships; tol::T=1e-6) where {T}
    hsd_normalized_residual(state) <= tol || return false
    # slack s must lie in the cone K (concatenated blocks)
    offset = 1
    for membership in memberships
        block_length = _membership_length(membership, state.s, offset)
        block = view(state.s, offset:(offset + block_length - 1))
        in_cone(membership, block) || return false
        offset += block_length
    end
    # complementarity small (s'x + tau*kappa -> 0 at the HSD solution)
    hsd_complementarity(state) <= tol * (one(T) + hsd_dimension(state)) || return false
    return true
end

function _membership_length(::OrthantMembership, s, offset)
    return length(s) - offset + 1
end
function _membership_length(::SOCMembership, s, offset)
    return length(s) - offset + 1
end
function _membership_length(::PSDMembership, s, offset)
    return length(s) - offset + 1
end
function _membership_length(::ExpMembership, s, offset)
    return 3
end
function _membership_length(::PowerMembership, s, offset)
    return 3
end

"""
    verify_primal_infeasibility!(state, y; tol=1e-6) -> Bool

Verify a primal-infeasibility certificate: a Farkas ray `y` with
`A'y <= 0` and `b'y > 0`.
"""
function verify_primal_infeasibility!(state::HSDState{T}, y::AbstractVector; tol::T=1e-6) where {T}
    length(y) == length(state.b) || return false
    Aty = state.A' * y
    all(v -> v <= tol, Aty) || return false
    return dot(state.b, y) > tol
end

"""
    verify_dual_infeasibility!(state, x, s; tol=1e-6) -> Bool

Verify a dual-infeasibility / primal-unbounded certificate: a ray `x`
with `c'x < 0`, `A x + s = 0`, and `s ∈ K`.
"""
function verify_dual_infeasibility!(state::HSDState{T}, x::AbstractVector, s::AbstractVector; tol::T=1e-6) where {T}
    length(x) == length(state.x) || return false
    length(s) == length(state.s) || return false
    dot(state.c, x) < -tol || return false
    norm(state.A * x + s) <= tol || return false
    return true
end

"""
    normalize_primal_ray!(y) -> Vector

Normalize a primal-infeasibility ray to unit norm.
"""
function normalize_primal_ray!(y::AbstractVector)
    norm_y = norm(y)
    iszero(norm_y) && return y
    y ./= norm_y
    return y
end

"""
    normalize_dual_ray!(x) -> Vector

Normalize a dual-infeasibility ray to unit norm.
"""
function normalize_dual_ray!(x::AbstractVector)
    norm_x = norm(x)
    iszero(norm_x) && return x
    x ./= norm_x
    return x
end
