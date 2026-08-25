#=====================================================================#
#    Unified homogeneous self-dual (HSD) state (Subagent C, PR3).
#
#    Canonical form:
#        minimize    c'x
#        subject to  A*x + s = b
#                    s ∈ K
#
#    Homogeneous self-dual embedding (Nesterov–Todd):
#        A x + s − b τ = 0
#        −c'x − b's + κ = 0
#        (x, τ) ∈ K* × R_+
#        (s, κ) ∈ K × R_+
#
#    This module is solver-neutral: it defines the HSD state, the
#    residuals, and the certificate verification. It does not choose a
#    linear-algebra provider or a KKT factorization.
#=====================================================================#

"""
    HSDState{T}

The homogeneous self-dual state of a canonical conic program.

Fields
- `x::Vector{T}` — primal variables (free).
- `s::Vector{T}` — slack in the cone `K`.
- `tau::T` — homogeneous parameter.
- `kappa::T` — complementarity gap.
- `A::SparseMatrixCSC{T,Int}` — the equality map.
- `b::Vector{T}` — the right-hand side.
- `c::Vector{T}` — the objective.
"""
struct HSDState{T}
    x::Vector{T}
    s::Vector{T}
    tau::T
    kappa::T
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    c::Vector{T}
end

"""
    hsd_dimension(state) -> Int

The embedding dimension `n + 1` (variables plus `τ`).
"""
hsd_dimension(state::HSDState) = length(state.x) + 1

"""
    hsd_primal_residual(state) -> Vector

The primal homogeneous residual `A x + s − b τ`.
"""
function hsd_primal_residual(state::HSDState{T}) where {T}
    return state.A * state.x + state.s - state.b * state.tau
end

"""
    hsd_dual_residual(state) -> T

The dual homogeneous residual `−c'x − b's + κ`.
"""
function hsd_dual_residual(state::HSDState{T}) where {T}
    return -dot(state.c, state.x) - dot(state.b, state.s) + state.kappa
end

"""
    hsd_complementarity(state) -> T

The complementarity `(s'x + τκ) / (n+1)`.
"""
function hsd_complementarity(state::HSDState{T}) where {T}
    return (dot(state.s, state.x) + state.tau * state.kappa) / hsd_dimension(state)
end

"""
    hsd_normalized_residual(state) -> T

The normalized homogeneous residual: the max of the primal and dual
residuals scaled by the data norm. Used for certificate verification.
"""
function hsd_normalized_residual(state::HSDState{T}) where {T}
    rp = hsd_primal_residual(state)
    rd = hsd_dual_residual(state)
    data_norm = norm(state.A) + norm(state.b) + norm(state.c) + one(T)
    return max(norm(rp), abs(rd)) / data_norm
end

"""
    hsd_optimality_gap(state) -> T

The duality gap `c'x + b's` (should be ~0 at an optimal point).
"""
function hsd_optimality_gap(state::HSDState{T}) where {T}
    return dot(state.c, state.x) + dot(state.b, state.s)
end
