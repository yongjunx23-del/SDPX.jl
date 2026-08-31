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
default_certificate_tol(::Type{BigFloat}) = parse(BigFloat, "1e-14")
default_certificate_tol(::Type{T}) where {T} = one(T) / T(100_000_000)

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
# Centralized finite gate
# ---------------------------------------------------------------------------
# The finite gate must run before every tolerance comparison used for cone
# membership or certificate validity.  Under IEEE semantics `NaN < -tol` and
# `NaN > tol` are both `false`, so a non-finite coordinate would otherwise
# bypass every "reject when the tolerance is violated" branch (B1).
@inline function _all_finite(values)
    @inbounds for value in values
        isfinite(value) || return false
    end
    return true
end

@inline _valid_certificate_tolerance(tol) =
    isfinite(tol) && tol >= zero(tol)

# ---------------------------------------------------------------------------
# Per-block cone membership resolved through the ConeProductLayout
# ---------------------------------------------------------------------------

@inline function _all_ge(v, tol)
    _all_finite(v) || return false
    @inbounds for i in eachindex(v)
        v[i] < tol && return false
    end
    return true
end

# PSD membership of an HSD svec vector `v` (dim n => len n(n+1)/2).
# Off-diagonal coordinates must be mapped back by 1/sqrt(2) before the
# eigenvalue check; treating svec as raw packed coordinates changes the cone.
function _svec_psd_membership(
    v, map::PSDCoordinateMap{T}, tol::Real, ::Type{T},
) where {T}
    n = map.dimension
    len = map.length
    length(v) == len || return false
    _valid_certificate_tolerance(tol) || return false
    n == 0 && return true
    _all_finite(v) || return false
    if T === Float64
        M = Matrix{Float64}(undef, n, n)
        k = 1
        @inbounds for j in 1:n
            for i in j:n
                val = v[k] * map.primal_inverse[k]
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
                val = v[k] * map.primal_inverse[k]
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

function _svec_psd_membership(v, n::Int, tol::Real, ::Type{T}) where {T<:AbstractFloat}
    bits = _psd_default_precision_bits(T, v)
    return _svec_psd_membership(
        v, PSDCoordinateMap(T, n; precision_bits=bits), tol, T,
    )
end

# Whether `v` (a slice view) lies in the block cone.  `dual` selects the dual
# cone `K*` (for the self-dual symmetric cones K* == K).  `tol` is the PSD
# slack / all-purpose numerical tolerance.
function _block_in_cone(block::ConeBlockDescriptor{T}, v, tol::T, dual::Bool) where {T}
    # The free cone's primal and the zero cone's dual accept every finite
    # coordinate, so they must be gated explicitly: NaN/Inf are never valid
    # finite-dimensional cone certificate coordinates.
    _all_finite(v) || return false
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
        map = _canonical_psd_coordinate_map(
            block,
            T;
            precision_bits=_psd_default_precision_bits(T, v),
        )
        return _svec_psd_membership(v, map, tol, T)
    elseif cone === :exp
        return dual ? exp_dual_membership(v[1], v[2], v[3]; tol=tol) :
               exp_membership(v[1], v[2], v[3]; tol=tol)
    elseif cone === :power
        return dual ?
               power_dual_membership(v[1], v[2], v[3], block.parameter; tol=tol) :
               power_membership(v[1], v[2], v[3], block.parameter; tol=tol)
    elseif cone === :free
        # The primal free cone contains every finite coordinate, while its
        # dual is the zero cone.  A blanket `false` for the dual incorrectly
        # rejects the valid zero dual slack and makes certificates containing
        # free primal blocks impossible to verify.
        return !dual || _all_ge(v, -tol) && _all_ge(-v, -tol)
    elseif cone === :zero
        # ZeroCone itself contains only the origin; its dual is the full free
        # space.  Treating the primal as free would let an equality-violating
        # slack pass a canonical certificate.
        return dual || _all_ge(v, -tol) && _all_ge(-v, -tol)
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
    _valid_certificate_tolerance(tol) || return false
    _all_finite(v) || return false
    blocks = layout_blocks(canonical.cone_layout)
    nb = length(blocks)
    # Blockwise membership is a pure read predicate over disjoint slices, so
    # a multithreaded sweep is bit-identical to the serial loop (each block
    # evaluates its own membership independently; the conjunction is exact).
    # Large product cones with many small blocks gain the most; small ones
    # keep the zero-overhead serial loop.
    if nb >= 256 && Threads.nthreads() > 1
        failed = Threads.Atomic{Bool}(false)
        chunk = max(1, cld(nb, Threads.nthreads() * 4))
        Threads.@threads :static for start in 1:chunk:nb
            stop = min(start + chunk - 1, nb)
            failed[] && continue
            @inbounds for b in start:stop
                block = blocks[b]
                off = block_offset(block); len = block_length(block)
                if !_block_in_cone(
                    block, view(v, off:(off + len - 1)), tol, dual,
                )
                    failed[] = true
                    break
                end
            end
        end
        return !failed[]
    end
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
    _valid_certificate_tolerance(tol) || return false
    # Centralized finite gate: no tolerance comparison may run on non-finite
    # data.  NaN/Inf in any iterate coordinate or embedding scalar fails the
    # certificate closed before any residual or cone-membership check.
    _all_finite(state.x) && _all_finite(state.s) && _all_finite(state.y) &&
        isfinite(state.tau) && isfinite(state.kappa) && isfinite(state.mu) ||
        return false
    hsd_residual!(state)
    _all_finite(state.rP) && _all_finite(state.rD) && isfinite(state.rG) ||
        return false
    # τ must be clearly positive: a genuinely optimal recovery needs x/τ, s/τ,
    # y/τ to be finite and well-posed, which rules out the τ→0 (infeasibility)
    # faces where a coarse residual could pass spuriously.
    state.tau > tol || return false
    # HSD iterates are homogeneous: multiplying every iterate coordinate by
    # a positive scalar must not make a non-optimal point easier to certify.
    # Residuals therefore have to be measured after recovery (division by
    # tau), not in the arbitrarily scaled embedding coordinates.
    inv_tau = inv(state.tau)
    isfinite(inv_tau) || return false
    normalized_residual = hsd_normalized_residual(state) * inv_tau
    isfinite(normalized_residual) && normalized_residual <= tol || return false
    # cone membership: s/τ ∈ K, y/τ ∈ K* (using pre-allocated scratch buffers)
    @inbounds for k in 1:state.m
        state.st[k] = state.s[k] * inv_tau
        state.yt[k] = state.y[k] * inv_tau
    end
    @inbounds for j in 1:state.n
        state.xt[j] = state.x[j] * inv_tau
    end
    _all_finite(state.xt) && _all_finite(state.st) && _all_finite(state.yt) ||
        return false
    in_canonical_cone(canonical, state.st; dual=false, tol=tol) || return false
    in_canonical_cone(canonical, state.yt; dual=true, tol=tol) || return false
    # Check the recovered primal-dual gap explicitly.  The old absolute-mu
    # check admitted degenerate sequences with tau,kappa -> 0 but a finite
    # kappa/tau (and hence a finite recovered duality gap).
    primal_objective = dot(canonical_objective(canonical), state.xt)
    dual_pairing = dot(canonical_rhs(canonical), state.yt)
    isfinite(primal_objective) && isfinite(dual_pairing) || return false
    gap_scale = one(T) + abs(primal_objective) + abs(dual_pairing)
    isfinite(gap_scale) || return false
    gap_residual = abs(primal_objective + dual_pairing)
    gap_limit = tol * gap_scale
    isfinite(gap_residual) && isfinite(gap_limit) && gap_residual <= gap_limit ||
        return false
    # The affine residuals are data-normalized and can be tiny while their
    # pairing with a large primal/dual point still leaves a resolvable cone
    # complementarity.  Check the recovered pairing directly so an internal
    # HSD status can never outrun the full-canonical recovery authority.
    cone_complementarity = abs(dot(state.st, state.yt))
    isfinite(cone_complementarity) && cone_complementarity <= gap_limit ||
        return false
    kappa_recovered = state.kappa * inv_tau
    kappa_limit = tol * gap_scale
    isfinite(kappa_recovered) && isfinite(kappa_limit) &&
        abs(kappa_recovered) <= kappa_limit || return false
    # mu is quadratic under homogeneous rescaling.  Normalize by tau^2 so
    # this centrality gate is invariant under the same embedding symmetry.
    normalized_mu = state.mu * inv_tau * inv_tau
    mu_limit = tol * (one(T) + T(state.nu))
    isfinite(normalized_mu) && isfinite(mu_limit) && normalized_mu <= mu_limit ||
        return false
    # recover in original coordinates through the reconstruction chain
    primal_forward!(canonical, x_orig, s_orig, state.xt, state.st)
    dual_forward!(canonical, y_orig, state.yt)
    return _all_finite(x_orig) && _all_finite(s_orig) && _all_finite(y_orig)
end

@inline function _certificate_unit_normalization_ok(value::T, tol::T) where {T}
    isfinite(value) && _valid_certificate_tolerance(tol) || return false
    return abs(value + one(T)) <= tol * max(one(T), abs(value))
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
    _valid_certificate_tolerance(tol) || return false
    y = state.y
    # Centralized finite gate: a Farkas ray must be a finite vector and its
    # derived scalars must be finite before any tolerance comparison runs.
    _all_finite(y) || return false
    by = dot(canonical_rhs(canonical), y)
    isfinite(by) || return false
    by < -tol || return false
    # A'y ≈ 0 (relative to the ray magnitude)
    _at_vec!(state.q, canonical_equality(canonical), y)
    res = _maxabs(state.q)
    ray_scale = one(T) + _maxabs(y)
    isfinite(res) && isfinite(ray_scale) || return false
    relative_residual = res / ray_scale
    isfinite(relative_residual) && relative_residual <= tol || return false
    in_canonical_cone(canonical, y; dual=true, tol=tol) || return false
    # normalize by −b'y = 1 and re-verify
    scale = -one(T) / by
    isfinite(scale) || return false
    @inbounds for k in 1:state.m
        state.yt[k] = scale * y[k]
    end
    _all_finite(state.yt) || return false
    _certificate_unit_normalization_ok(
        dot(canonical_rhs(canonical), state.yt), tol,
    ) || return false
    _at_vec!(state.q, canonical_equality(canonical), state.yt)
    normalized_residual = _maxabs(state.q)
    isfinite(normalized_residual) && normalized_residual <= tol || return false
    in_canonical_cone(canonical, state.yt; dual=true, tol=tol) || return false
    # push the ray back into original coordinates
    certificate_backward!(canonical, y_orig, state.yt; ray_kind=:primal_infeasible)
    return _all_finite(y_orig)
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
    _valid_certificate_tolerance(tol) || return false
    x = state.x
    # Centralized finite gate: a recession ray must be a finite vector and its
    # derived scalars must be finite before any tolerance comparison runs.
    _all_finite(x) || return false
    cx = dot(canonical_objective(canonical), x)
    isfinite(cx) || return false
    cx < -tol || return false
    # the ray's cone member: s_r = −A x must lie in K
    _at_negmul!(state.e, canonical_equality(canonical), x)
    _all_finite(state.e) || return false
    in_canonical_cone(canonical, state.e; dual=false, tol=tol) || return false
    # normalize by −c'x = 1 and re-verify (including the cone member again)
    scale = -one(T) / cx
    isfinite(scale) || return false
    @inbounds for j in 1:state.n
        state.xt[j] = scale * x[j]
    end
    @inbounds for k in 1:state.m
        state.st[k] = scale * state.e[k]
    end
    _all_finite(state.xt) && _all_finite(state.st) || return false
    _certificate_unit_normalization_ok(
        dot(canonical_objective(canonical), state.xt), tol,
    ) || return false
    in_canonical_cone(canonical, state.st; dual=false, tol=tol) || return false
    # push the ray back into original coordinates
    certificate_backward!(canonical, x_orig, state.xt; ray_kind=:dual_infeasible)
    primal_forward!(canonical, x_orig, s_orig, state.xt, state.st)
    return _all_finite(x_orig) && _all_finite(s_orig)
end
function dual_objective(prob::SDPProblem{T}, y, Y) where {T}
    d = zero(T)
    for l in 1:prob.dims.L
        d += kdot(prob.C[l], Y[l])
    end
    prob.dims.n > 0 && (d += LinearAlgebra.dot(prob.b, y))
    return d
end
