#=====================================================================#
#    Certificate verification in original coordinates (Subagent H).
#
#    Final status is NEVER based on raw τ/κ alone.  It comes only from a
#    verified certificate in original coordinates, pushed back through the
#    canonical program's reconstruction chain:
#
#      - verify_optimal!              normalized HSD residual + cone
#                                     membership (s ∈ K, y ∈ K*) +
#                                     complementarity small; recover
#                                     x/τ, s/τ, y/τ.
#      - verify_primal_infeasibility!  Farkas ray y: A'y ≈ 0, y ∈ K*,
#                                     b'y < 0, normalized by −b'y = 1.
#      - verify_dual_infeasibility!    ray x: −A x ∈ K, c'x < 0,
#                                     normalized by −c'x = 1.  Cone
#                                     membership of the ray IS checked
#                                     (the old draft skipped `s ∈ K`).
#
#    Cone membership is resolved per block through the canonical
#    `ConeProductLayout`; PSD tolerance derives from `T` and the requested
#    accuracy (no hardcoded 1e-8).
#=====================================================================#

# ---------------------------------------------------------------------------
# Tolerance from the element type + requested accuracy
# ---------------------------------------------------------------------------
"""
    default_certificate_tol(::Type{T}) -> T

A problem-free default certificate tolerance for the element type `T`.
"""
default_certificate_tol(::Type{Float64}) = 1e-6
default_certificate_tol(::Type{BigFloat}) = 1e-14
default_certificate_tol(::Type{T}) where {T} = 1e-8

# ---------------------------------------------------------------------------
# Small linear-algebra helpers (defined BEFORE the verifiers: Julia binds a
# function body eagerly at top level, so a helper referenced from a verify
# function must already exist when that function is defined).
# ---------------------------------------------------------------------------

# dst = A' v (m-length row space -> n-vector).
function _at_vec(A::SparseMatrixCSC{T}, v::AbstractVector) where {T}
    n = size(A, 2)
    out = zeros(T, n)
    _at_vec!(out, A, v)
    return out
end

function _at_vec!(out::AbstractVector{T}, A::SparseMatrixCSC{T}, v::AbstractVector) where {T}
    n = size(A, 2)
    vals = nonzeros(A)
    rows = rowvals(A)
    @inbounds for j in 1:n
        acc = zero(T)
        for idx in nzrange(A, j)
            i = rows[idx]
            acc += vals[idx] * v[i]
        end
        out[j] = acc
    end
    return out
end

# sr = −A v (m-vector)
function _at_negmul(A::SparseMatrixCSC{T}, v::AbstractVector) where {T}
    m = size(A, 1)
    out = zeros(T, m)
    _at_negmul!(out, A, v)
    return out
end

function _at_negmul!(out::AbstractVector{T}, A::SparseMatrixCSC{T}, v::AbstractVector) where {T}
    fill!(out, zero(T))
    n = size(A, 2)
    vals = nonzeros(A)
    rows = rowvals(A)
    @inbounds for j in 1:n
        val = v[j]
        iszero(val) && continue
        for idx in nzrange(A, j)
            i = rows[idx]
            out[i] -= vals[idx] * val
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Per-block cone membership resolved through the ConeProductLayout
# ---------------------------------------------------------------------------

@inline function _all_ge(v, tol)
    @inbounds for i in eachindex(v)
        v[i] < tol && return false
    end
    return true
end

# PSD membership of a packed-lower vector `v` (dim n ⇒ len n(n+1)/2).
function _packed_psd_membership(v, n::Int, tol::Real, ::Type{T}) where {T}
    len = div(n * (n + 1), 2)
    length(v) == len || return false
    n == 0 && return true
    if T === Float64
        M = Matrix{Float64}(undef, n, n)
        k = 1
        @inbounds for j in 1:n
            for i in j:n
                val = v[k]
                M[i, j] = val
                M[j, i] = val
                k += 1
            end
        end
        wmin = minimum(eigvals(Symmetric(M)))
        return wmin >= -tol
    else
        A = Matrix{T}(undef, n, n)
        V = Matrix{T}(undef, n, n)
        w = Vector{T}(undef, n)
        k = 1
        @inbounds for j in 1:n
            for i in j:n
                val = v[k]
                A[i, j] = val
                A[j, i] = val
                k += 1
            end
        end
        for j in 1:n, i in 1:n
            V[i, j] = i == j ? one(T) : zero(T)
        end
        try
            SymmetricCones._jacobi_eigen!(A, V, w; maxsweeps=100)
            return minimum(w) >= -tol
        catch e
            if e isa SymmetricCones._SymmetricEigenFailed
                return false
            end
            rethrow(e)
        end
    end
end

# Whether `v` (a slice view) lies in the block cone.  `dual` selects the dual
# cone `K*` (for the self-dual symmetric cones K* == K).  `tol` is the PSD
# slack / all-purpose numerical tolerance.
function _block_in_cone(block::ConeBlockDescriptor{T}, v, tol::T, dual::Bool) where {T}
    cone = block.cone
    if cone === :nonnegative
        return _all_ge(v, -tol)
    elseif cone === :soc
        t = v[1]
        t >= -tol || return false
        n = length(v)
        acc = zero(T)
        @inbounds for i in 2:n
            acc += v[i] * v[i]
        end
        return sqrt(acc) <= t + tol
    elseif cone === :psd
        return _packed_psd_membership(v, block.dimension, tol, T)
    elseif cone === :exp
        return dual ? exp_dual_membership(v[1], v[2], v[3]; tol=tol) : exp_membership(v[1], v[2], v[3])
    elseif cone === :power
        return dual ? power_dual_membership(v[1], v[2], v[3], block.parameter; tol=tol) : power_membership(v[1], v[2], v[3], block.parameter)
    elseif cone === :free
        return !dual # free dual is zero cone
    elseif cone === :zero
        return true  # zero dual is free cone
    end
    return false
end

"""
    in_canonical_cone(canonical, v; dual=false, tol=...) -> Bool

Check that the canonical slack vector `v` (length `m`) lies in the
product cone `K` (or its dual `K*` when `dual=true`), block by block
through the canonical `ConeProductLayout`.
"""
function in_canonical_cone(canonical::CanonicalConicProgram, v;
                           dual::Bool=false, tol::Real=default_certificate_tol(eltype(v)))
    length(v) == canonical_num_slack(canonical) ||
        throw(DimensionMismatch("v length != canonical slack m"))
    T = eltype(v)
    tol = convert(T, tol)
    for block in layout_blocks(canonical.cone_layout)
        off = block_offset(block); len = block_length(block)
        _block_in_cone(block, view(v, off:(off + len - 1)), tol, dual) || return false
    end
    return true
end

# ---------------------------------------------------------------------------
# Optimality certificate
# ---------------------------------------------------------------------------
"""
    verify_optimal!(canonical, state, x_orig, s_orig, y_orig; tol) -> Bool

Verify an optimality certificate in original coordinates: the normalized
HSD residual (`A x + s − b τ`, `A'y + c τ`, `−c'x − b'y + κ`) is small,
`s/τ ∈ K` and `y/τ ∈ K*` blockwise, and the complementarity `μ` is small.
On success, writes the original-coordinate recoveries `x_orig = x/τ`,
`s_orig = s/τ`, `y_orig = y/τ` (pushed through the reconstruction chain)
and returns `true`.
"""
function verify_optimal!(
    canonical::CanonicalConicProgram, state::HSDState,
    x_orig, s_orig, y_orig; tol=nothing,
)
    T = eltype(state.x)
    tol = tol === nothing ? default_certificate_tol(T) : T(tol)
    hsd_residual!(state)
    # τ must be clearly positive: a genuinely optimal recovery needs x/τ, s/τ,
    # y/τ to be finite and well-posed, which rules out the τ→0 (infeasibility)
    # faces where a coarse residual could pass spuriously.
    state.tau > tol || return false
    hsd_normalized_residual(state) <= tol || return false
    # cone membership: s/τ ∈ K, y/τ ∈ K* (using pre-allocated scratch buffers)
    inv_tau = inv(state.tau)
    @inbounds for k in 1:state.m
        state.st[k] = state.s[k] * inv_tau
        state.yt[k] = state.y[k] * inv_tau
    end
    @inbounds for j in 1:state.n
        state.xt[j] = state.x[j] * inv_tau
    end
    in_canonical_cone(canonical, state.st; dual=false, tol=tol) || return false
    in_canonical_cone(canonical, state.yt; dual=true, tol=tol) || return false
    # complementarity small (μ = (s'y + τκ)/(ν+1))
    state.mu <= tol * (one(T) + T(state.nu)) || return false
    # recover in original coordinates through the reconstruction chain
    primal_forward!(canonical, x_orig, s_orig, state.xt, state.st)
    dual_forward!(canonical, y_orig, state.yt)
    return true
end

# ---------------------------------------------------------------------------
# Primal-infeasibility (Farkas) certificate
# ---------------------------------------------------------------------------
"""
    verify_primal_infeasibility!(canonical, state, y_orig; tol) -> Bool

Verify a primal-infeasibility certificate: the canonical dual `y` is a
Farkas ray with `A'y ≈ 0`, `y ∈ K*`, and `b'y < 0`.  The ray is
normalized by `−b'y = 1` and re-verified, then pushed back to original
coordinates through the reconstruction chain (into `y_orig`).
"""
function verify_primal_infeasibility!(
    canonical::CanonicalConicProgram, state::HSDState, y_orig; tol=nothing,
)
    T = eltype(state.x)
    tol = tol === nothing ? default_certificate_tol(T) : T(tol)
    y = state.y
    by = dot(canonical_rhs(canonical), y)
    by < -tol || return false
    # A'y ≈ 0 (relative to the ray magnitude)
    _at_vec!(state.q, canonical_equality(canonical), y)
    res = _maxabs(state.q)
    (res / (one(T) + _maxabs(y))) <= tol || return false
    in_canonical_cone(canonical, y; dual=true, tol=tol) || return false
    # normalize by −b'y = 1 and re-verify
    scale = -one(T) / by
    @inbounds for k in 1:state.m
        state.yt[k] = scale * y[k]
    end
    dot(canonical_rhs(canonical), state.yt) ≈ -one(T) || return false
    _at_vec!(state.q, canonical_equality(canonical), state.yt)
    _maxabs(state.q) <= tol || return false
    in_canonical_cone(canonical, state.yt; dual=true, tol=tol) || return false
    # push the ray back into original coordinates
    certificate_backward!(canonical, y_orig, state.yt; ray_kind=:primal_infeasible)
    return true
end

# ---------------------------------------------------------------------------
# Dual-infeasibility / primal-unbounded certificate
# ---------------------------------------------------------------------------
"""
    verify_dual_infeasibility!(canonical, state, x_orig, s_orig; tol) -> Bool

Verify a dual-infeasibility / primal-unbounded certificate: the canonical
`x` is a ray with `−A x ∈ K` (the ray's own cone membership IS checked),
and `c'x < 0`.  The ray is normalized by `−c'x = 1` and re-verified, then
pushed back into original coordinates (into `x_orig` and the slack ray
`s_orig`).
"""
function verify_dual_infeasibility!(
    canonical::CanonicalConicProgram, state::HSDState, x_orig, s_orig; tol=nothing,
)
    T = eltype(state.x)
    tol = tol === nothing ? default_certificate_tol(T) : T(tol)
    x = state.x
    cx = dot(canonical_objective(canonical), x)
    cx < -tol || return false
    # the ray's cone member: s_r = −A x must lie in K
    _at_negmul!(state.e, canonical_equality(canonical), x)
    in_canonical_cone(canonical, state.e; dual=false, tol=tol) || return false
    # normalize by −c'x = 1 and re-verify (including the cone member again)
    scale = -one(T) / cx
    @inbounds for j in 1:state.n
        state.xt[j] = scale * x[j]
    end
    @inbounds for k in 1:state.m
        state.st[k] = scale * state.e[k]
    end
    dot(canonical_objective(canonical), state.xt) ≈ -one(T) || return false
    in_canonical_cone(canonical, state.st; dual=false, tol=tol) || return false
    # push the ray back into original coordinates
    certificate_backward!(canonical, x_orig, state.xt; ray_kind=:dual_infeasible)
    primal_forward!(canonical, x_orig, s_orig, state.xt, state.st)
    return true
end
