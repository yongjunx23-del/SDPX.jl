# Common HSD runtime helpers shared by the Nonnegative-base state and the
# product-cone HSD state machine.  These kernels depend only on `HSDState`
# (the Nonnegative base of `ProductConeHSDState`) and the KKT route driver;
# they carry no cone-specific Newton algebra and no solver-family dispatch.

# Map a reduced row-space Newton direction into canonical/original
# coordinates.  Since `rank_basis` has orthonormal columns spanning
# `range(A')`, `V_r*dxr` is the unique minimum-norm representative of the
# reduced equality-map direction.  No original coordinate is selected or
# forced to zero.
@inline function _hsd_scatter_dx!(state::HSDState{T}) where {T}
    V = state.rank_basis
    @inbounds for i in 1:state.n
        acc = zero(T)
        for j in 1:state.nr
            acc += V[i, j] * state.dxr[j]
        end
        state.dx[i] = acc
    end
    return state.dx
end

@inline function _hsd_maxinf(v::AbstractVector{T}) where {T}
    a = zero(T)
    @inbounds for i in eachindex(v)
        b = v[i] < zero(T) ? -v[i] : v[i]
        b > a && (a = b)
    end
    return a
end

# Trial residual rPt = A xt + st − b τt, rDt = A' yt + c τt (for the guard).
@inline function _hsd_trial_residual!(state::HSDState{T}) where {T}
    A = state.A
    m = state.m; n = state.n
    fill!(state.rPt, zero(T))
    @inbounds for j in 1:n
        a = state.xt[j]
        iszero(a) && continue
        for ptr in nzrange(A, j)
            k = A.rowval[ptr]
            state.rPt[k] += A.nzval[ptr] * a
        end
    end
    @inbounds for k in 1:m
        state.rPt[k] += state.st[k] - state.b[k] * state.tau_t
    end
    fill!(state.rDt, zero(T))
    @inbounds for j in 1:n
        acc = zero(T)
        for ptr in nzrange(A, j)
            k = A.rowval[ptr]
            acc += A.nzval[ptr] * state.yt[k]
        end
        state.rDt[j] = acc + state.c[j] * state.tau_t
    end
    return nothing
end

@inline function _hsd_direction_finite(state::HSDState{T}) where {T}
    @inbounds for j in 1:state.n
        isfinite(state.dx[j]) || return false
    end
    @inbounds for k in 1:state.m
        (isfinite(state.dy[k]) && isfinite(state.ds[k])) || return false
    end
    return isfinite(state.dtau) && isfinite(state.dkappa)
end

@inline function _hsd_matrix_finite(H::Matrix{T}) where {T}
    @inbounds for v in H
        isfinite(v) || return false
    end
    return true
end

# Residual homotopy guard for a trial point: the admissible line-search step
# must keep every residual `rP, rD, rG` on the affine path `r(α) = (1-α)r(0)`
# up to an arithmetic tolerance derived from the element type.
@inline function _hsd_residual_homotopy_ok(
    state::HSDState{T}, alpha::T, p2::T, d2::T, gap2::T,
) where {T}
    weight = one(T) - alpha
    scale = max(
        one(T),
        _hsd_maxinf(state.rP),
        _hsd_maxinf(state.rD),
        abs(state.rG),
        p2,
        d2,
        abs(gap2),
    )
    tol = T(256) * sqrt(eps(T)) * scale
    isfinite(tol) || return false
    @inbounds for k in 1:state.m
        abs(state.rPt[k] - weight * state.rP[k]) <= tol || return false
    end
    @inbounds for j in 1:state.n
        abs(state.rDt[j] - weight * state.rD[j]) <= tol || return false
    end
    return abs(gap2 - weight * state.rG) <= tol
end

@inline function _hsd_update_record!(state::HSDState{T}) where {T}
    r = state.record
    r.p_res = _hsd_maxinf(state.rP)
    r.d_res = _hsd_maxinf(state.rD)
    r.mu = state.mu
    r.mu_aff = state.mu_aff
    r.complementarity = state.complementarity
    r.matrix_epoch = kkt_matrix_epoch(state.driver)
    r.factor_epoch = kkt_factor_epoch(state.driver)
    r.factorizations = kkt_factor_count(state.driver)
    return r
end
