#=====================================================================#
# Internal symmetric product-cone HSD solve loop.
#
# This file is deliberately not wired to the public/MOI route.  It drives
# `product_hsd_step!` and promotes a terminal status only after one of the
# original-coordinate certificate verifiers succeeds.  There is no legacy
# solve, PSD lift, projected-gradient ray search, or other fallback here.
#=====================================================================#

include("termination.jl")
include("recovery.jl")

@inline function _product_hsd_finite_ray_candidate(x::AbstractVector{T}) where {T}
    nonzero = false
    @inbounds for value in x
        isfinite(value) || return false
        nonzero |= !iszero(value)
    end
    return nonzero
end

"""
Run the authoritative original-coordinate dual-infeasibility verifier before
returning a terminal non-certificate status.  A collapsed τ/κ state and every
finite nonzero primal-ray candidate are both checked.  The requested failure,
time, or iteration status is preserved unless the unchanged verifier accepts
the ray; no HSD ratio or internal residual can promote DualInfeasible.
"""
function _product_hsd_termination_or_dual_ray!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    status::ProductHSDSolveStatus,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
) where {T}
    base = state.base
    candidate = _product_hsd_tau_collapsed(base, tol) ||
                _product_hsd_finite_ray_candidate(base.x)
    if candidate && verify_dual_infeasibility!(
        base.canonical, base, x_original, s_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDDualInfeasible,
            ProductHSDVerifiedTerminationRay, last_step, zero(T),
            x_original, s_original, y_original,
        )
    end
    return _product_hsd_make_result(
        state, status, reason, last_step, zero(T),
        x_original, s_original, y_original,
    )
end

@inline function _product_hsd_refinement_maxabs(values::AbstractVector{T}) where {T}
    magnitude_max = zero(T)
    @inbounds for value in values
        magnitude = abs(value)
        magnitude > magnitude_max && (magnitude_max = magnitude)
    end
    return magnitude_max
end

@inline function _product_hsd_refinement_scale(values::AbstractVector{T}) where {T}
    return max(one(T), _product_hsd_refinement_maxabs(values))
end

"""
Try one cold-path affine certificate refinement of an accepted HSD iterate.
The cone variables are held fixed while the primal variable is
corrected in `A*dx = -rP`; the dual is then corrected in the joint
stationarity/gap equations `[A'; b']*dy = [-rD; -gap]`. This removes the
homogeneous residual amplification which otherwise lets the data-normalized
HSD gate stop before the recovered original-coordinate gate is satisfied.

The refinement is fail closed: it is adopted only when the unchanged strict
optimal verifier accepts at the caller's requested tolerance. Exp dual blocks
also admit the exact `u=v` boundary representative when `u` is structurally
free in both affine equations; this removes a second-order cone residual
without altering stationarity or the gap. No cone-membership check or
tolerance is relaxed.
"""
function _product_hsd_owned_dense(
    A::SparseMatrixCSC{T,Int},
) where {T<:AbstractFloat}
    dense = alloc_zeros(T, size(A, 1), size(A, 2))
    @inbounds for column in axes(A, 2)
        for pointer in nzrange(A, column)
            _store_owned_scalar!(
                dense, CartesianIndex(A.rowval[pointer], column),
                A.nzval[pointer],
            )
        end
    end
    return dense
end

"""
    _product_hsd_terminal_primal_factor(A)

Factorize the primal least-squares operator exactly the way `\\` does for a
non-square dense operator, or return `nothing` when the cached path is not
provably identical to `\\`.

Only a dense non-square operator is cached: `A \\ rhs` is then exactly
`qr(A, ColumnNorm()) \\ rhs`, verified bit-identical. Square operators use LU.
Sparse operators are deliberately excluded: `qr(A)` without a tolerance routes
through SPQR's `_default_tol`, which reduces over `nonzeros(A)` and therefore
throws on an operator with no stored entries, whereas `A \\ rhs` does not take
that path. On the large dense operators this cache targets, `Ad` is a dense
`Matrix`, so the excluded case is not on the measured path.
"""
@inline function _product_hsd_terminal_primal_factor(A::Matrix{T}) where {T}
    size(A, 1) == size(A, 2) && return nothing
    return qr(A, ColumnNorm())
end

@inline _product_hsd_terminal_primal_factor(::AbstractMatrix) = nothing

"""
    _product_hsd_terminal_dual_operator(A, b)

Dense `(n+1) x m` operator `[A'; b']` used by the Float64 dual recovery. Loop
invariant, so it is materialized once per solve instead of once per attempt.
"""
function _product_hsd_terminal_dual_operator(
    A::AbstractMatrix{T}, b::AbstractVector{T},
) where {T}
    n, m = size(A, 2), size(A, 1)
    operator = alloc_zeros(T, n + 1, m)
    # The block is exactly `transpose(A)` with `b` appended as its last row:
    # a copy, never arithmetic. For a sparse `A` the elementwise form pays one
    # stored-entry search per *matrix* entry (`A[column, row]`), i.e. O(n * m)
    # searches, where iterating the stored entries is O(nnz) with direct dense
    # writes. Structurally zero entries would have been written as `zero(T)`
    # and `alloc_zeros` already left `zero(T)` there, so the result is
    # unchanged. On the measured C4/S256 operators this did NOT materially move
    # the recovery build, which is dominated by the dense dual QR, the cached
    # wide-QR reduction and the adapter self-check; it is kept because the
    # search count scales with n * m rather than nnz.
    if A isa SparseMatrixCSC
        # `operator[j, i] = A[i, j]`: the stored entry `A[i, j]` of column `j`
        # lands in row `j` and column `i`, exactly where the elementwise form
        # would have written it.
        @inbounds for j in 1:size(A, 2)
            for pointer in nzrange(A, j)
                _store_owned_scalar!(
                    operator,
                    CartesianIndex(j, rowvals(A)[pointer]),
                    nonzeros(A)[pointer],
                )
            end
        end
    else
        @inbounds for column in 1:m
            for row in 1:n
                _store_owned_scalar!(
                    operator, CartesianIndex(row, column), A[column, row],
                )
            end
        end
    end
    @inbounds for column in 1:m
        _store_owned_scalar!(
            operator, CartesianIndex(n + 1, column), b[column],
        )
    end
    return operator
end

@inline function _product_hsd_terminal_dual_factor(
    operator::Union{Nothing,Matrix{T}},
) where {T}
    operator === nothing && return nothing
    size(operator, 1) == size(operator, 2) && return nothing
    return qr(operator, ColumnNorm())
end

function _product_hsd_terminal_la_backend(::Type{T}) where {T<:AbstractFloat}
    T === Float64 && return nothing
    config = plan_la_backend(
        T; requested=:auto, route=:dense_cholesky,
        threads=max(Threads.nthreads(), 1), equality_solver=:qr,
    )
    return instantiate_la_backend(config, T, max(Threads.nthreads(), 1))
end

function _product_hsd_apply_primal_refinement!(
    x::Vector{T}, A::AbstractMatrix{T}, dense_A::AbstractMatrix{T},
    primal_residual::Vector{T}, backend,
) where {T<:AbstractFloat}
    if T === Float64
        x .+= dense_A \ (-primal_residual)
        return true
    end
    relative_tolerance = T(max(size(A)...)) * eps(T)
    factor = la_qr_factor!(
        backend, dense_A; pivoted=true,
        relative_tolerance=relative_tolerance,
    )
    factor === nothing && return false
    factor.rank == size(A, 2) || return false
    rhs = alloc_zeros(T, size(A, 2))
    @inbounds for column in axes(A, 2)
        value = zero(T)
        for pointer in nzrange(A, column)
            value -= A.nzval[pointer] * primal_residual[A.rowval[pointer]]
        end
        _store_owned_scalar!(rhs, column, value)
    end
    permuted = alloc_zeros(T, length(rhs))
    la_factor_solve!(factor, rhs, permuted)
    all(isfinite, rhs) || return false
    @inbounds for index in eachindex(x)
        _store_owned_scalar!(x, index, x[index] + rhs[index])
    end
    return all(isfinite, x)
end

function _product_hsd_apply_dual_refinement!(
    y::Vector{T}, A::AbstractMatrix{T}, dense_A::AbstractMatrix{T},
    b::Vector{T}, dual_residual::Vector{T}, gap::T, backend,
) where {T<:AbstractFloat}
    n = size(A, 2)
    m = size(A, 1)
    if T === Float64
        affine_dual = alloc_zeros(T, n + 1, m)
        @inbounds for column in 1:m
            for row in 1:n
                _store_owned_scalar!(
                    affine_dual, CartesianIndex(row, column), A[column, row],
                )
            end
            _store_owned_scalar!(
                affine_dual, CartesianIndex(n + 1, column), b[column],
            )
        end
        rhs = alloc_zeros(T, n + 1)
        @inbounds for row in 1:n
            _store_owned_scalar!(rhs, row, -dual_residual[row])
        end
        _store_owned_scalar!(rhs, n + 1, -gap)
        y .+= affine_dual \ rhs
        return true
    end

    # Minimum-norm correction for D*dy = rhs, D=[A'; b']:
    # dy = D' * (D*D')^-1 * rhs.  This is mathematically identical to the
    # previous wide least-squares solve, but factors only an (n+1)x(n+1)
    # target-arithmetic Gram matrix instead of generic QR on (n+1)xm.
    gram = alloc_zeros(T, n + 1, n + 1)
    top = @view gram[1:n, 1:n]
    la_syrk!(backend, top, dense_A, one(T), zero(T))
    atb = alloc_zeros(T, n)
    @inbounds for column in 1:n
        value = zero(T)
        for pointer in nzrange(A, column)
            row = A.rowval[pointer]
            value += A.nzval[pointer] * b[row]
        end
        _store_owned_scalar!(atb, column, value)
        _store_owned_scalar!(gram, CartesianIndex(n + 1, column), value)
        _store_owned_scalar!(gram, CartesianIndex(column, n + 1), value)
    end
    _store_owned_scalar!(
        gram, CartesianIndex(n + 1, n + 1), dot(b, b),
    )
    all(isfinite, gram) || return false
    factor = la_cholesky_factor!(backend, gram)
    factor === nothing && return false
    rhs = alloc_zeros(T, n + 1)
    @inbounds for row in 1:n
        rhs[row] = -dual_residual[row]
    end
    rhs[end] = -gap
    la_factor_solve!(factor, rhs)
    all(isfinite, rhs) || return false
    correction = alloc_zeros(T, m)
    @inbounds for row in 1:m
        _store_owned_scalar!(correction, row, b[row] * rhs[end])
    end
    @inbounds for column in 1:n
        coefficient = rhs[column]
        for pointer in nzrange(A, column)
            correction[A.rowval[pointer]] += A.nzval[pointer] * coefficient
        end
    end
    @inbounds for row in 1:m
        _store_owned_scalar!(y, row, y[row] + correction[row])
    end
    return all(isfinite, y)
end

function _product_hsd_refined_optimal_result!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
    terminal_alpha::T=zero(T),
) where {T}
    base = state.base
    recovered_floor = max(tol, sqrt(eps(T)))
    recovered_residual = _product_hsd_recovered_residual(state, recovered_floor)
    isfinite(recovered_residual) || return nothing
    recovered_residual <= sqrt(tol) || return nothing

    canonical = base.canonical
    # Float64 sparse least squares is handled by SPQR. Julia's sparse `\\`
    # does not support MultiFloat, so the cold terminal refinement makes an
    # explicit target-arithmetic dense copy rather than silently lowering to
    # Float64. The copy exists only for a terminal candidate; Newton epochs
    # remain sparse. Oversized candidates fail closed before allocation.
    A = base.workspace.Ad
    refinement_A = if A isa SparseMatrixCSC && T !== Float64
        scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
        required = saturating_bytes(scalar_bytes, size(A,1), size(A,2))
        required <= 512 * 1024^2 || return nothing
        _product_hsd_owned_dense(A)
    else
        A
    end
    backend = _product_hsd_terminal_la_backend(T)
    T !== Float64 && backend === nothing && return nothing
    # Loop-invariant terminal-recovery operators (see
    # `ProductHSDTerminalRecoveryCache`). `Ad`, `b` and the cone layout are
    # fixed for the solve, so the primal least-squares factorization and the
    # dense `(n+1) x m` dual operator `[A'; b']` (with its factorization) are
    # built once and reused by every later recovery attempt. The cached path is
    # bit-identical by construction: for a non-square operator `A \ rhs` is
    # exactly `qr(A[, ColumnNorm()]) \ rhs`. Float64 owns the cached path;
    # every other arithmetic keeps the existing provider-driven refinement
    # unchanged, and square operators keep the plain `\` dispatch.
    recovery = state.terminal_recovery
    primal_factor = nothing
    dual_operator = nothing
    dual_factor = nothing
    if T === Float64
        key = _product_hsd_terminal_recovery_key(T, A, canonical.b)
        if recovery.key != key
            recovery.primal = _product_hsd_terminal_primal_factor(refinement_A)
            recovery.dual_operator =
                _product_hsd_terminal_dual_operator(A, canonical.b)
            recovery.dual =
                _product_hsd_terminal_dual_factor(recovery.dual_operator)
            recovery.key = key
            recovery.builds += 1
            # Guarded wide pivoted-QR reduction, built only for Float64 and only
            # after a bitwise self-check against the ordinary solve on this
            # operator. Any refusal keeps the ordinary `F \ rhs` path.
            recovery.wide_qr = nothing
            recovery.wide_qr_active = false
            if T === Float64 && recovery.dual !== nothing
                reduction = _product_hsd_wide_qr_reduce(recovery.dual)
                if reduction === nothing
                    recovery.wide_qr_reason = :unsupported_or_version_gated
                else
                    operator = recovery.dual_operator
                    rows, columns = size(operator)
                    recovery.wide_qr_buffer = alloc_zeros(T, columns, 1)
                    if _product_hsd_wide_qr_selfcheck(
                        recovery.dual, reduction, rows, columns,
                    )
                        recovery.wide_qr = reduction
                        recovery.wide_qr_active = true
                        recovery.wide_qr_reason = :selfcheck_passed
                    else
                        recovery.wide_qr_reason = :selfcheck_mismatch
                    end
                end
            elseif T !== Float64
                recovery.wide_qr_reason = :unsupported_arithmetic
            else
                recovery.wide_qr_reason = :no_dual_factor
            end
        else
            recovery.reuses += 1
        end
        primal_factor = recovery.primal
        dual_operator = recovery.dual_operator
        dual_factor = recovery.dual
    end
    x = base.x ./ base.tau
    s = base.s ./ base.tau
    y = base.y ./ base.tau
    primal_residual = A * x + s - canonical.b

    try
        if base.n == 0
            _product_hsd_refinement_maxabs(primal_residual) <= tol || return nothing
        elseif primal_factor !== nothing
            x .+= primal_factor \ (-primal_residual)
        else
            _product_hsd_apply_primal_refinement!(
                x, A, refinement_A, primal_residual, backend,
            ) || return nothing
        end

        # Near the x=0 exposed face of K_exp, the exact boundary relation is
        # z=y. Scaling failure can occur a few ulps before that representative
        # is reached. Form it only as a cold certificate candidate, then
        # restore affine primal feasibility. The candidate is adopted below
        # only if every strict original-coordinate gate accepts it.
        exp_boundary_changed = false
        for block in canonical.cone_layout.blocks
            block.cone === :exp || continue
            u = block.offset
            v = u + 1
            w = u + 2
            scale = max(one(T), abs(s[u]), abs(s[v]), abs(s[w]))
            neighborhood = sqrt(tol) * scale
            if abs(s[u]) <= neighborhood &&
               abs(s[w] - s[v]) <= neighborhood && s[v] > zero(T)
                _store_owned_scalar!(s, u, zero(T))
                _store_owned_scalar!(s, w, s[v])
                exp_boundary_changed = true
            end
        end
        if exp_boundary_changed
            primal_residual = A * x + s - canonical.b
            if base.n == 0
                _product_hsd_refinement_maxabs(primal_residual) <= tol ||
                    return nothing
            else
                _product_hsd_apply_primal_refinement!(
                    x, A, _product_hsd_owned_dense(A), primal_residual, backend,
                ) || return nothing
            end
        end

        dual_residual = transpose(A) * y + canonical.c
        gap = dot(canonical.c, x) + dot(canonical.b, y)
        if dual_factor !== nothing
            # Same rhs and same operator the uncached helper builds; only the
            # factorization is reused instead of rebuilt per attempt.
            rhs = alloc_zeros(T, base.n + 1)
            @inbounds for row in 1:base.n
                _store_owned_scalar!(rhs, row, -dual_residual[row])
            end
            _store_owned_scalar!(rhs, base.n + 1, -gap)
            if recovery.wide_qr_active && recovery.wide_qr !== nothing
                buffer = recovery.wide_qr_buffer
                fill!(buffer, zero(T))
                @inbounds for row in 1:(base.n + 1)
                    buffer[row, 1] = rhs[row]
                end
                _product_hsd_wide_qr_solve!(buffer, dual_factor, recovery.wide_qr)
                @inbounds for row in 1:base.m
                    _store_owned_scalar!(y, row, y[row] + buffer[row, 1])
                end
            else
                recovery.wide_qr_fallbacks += 1
                y .+= dual_factor \ rhs
            end
        else
            dual_dense_A = T === Float64 ? refinement_A :
                           _product_hsd_owned_dense(A)
            _product_hsd_apply_dual_refinement!(
                y, A, dual_dense_A, canonical.b, dual_residual, gap, backend,
            ) || return nothing
        end

        # For K_exp^*, L_E(u,v,w)=(u-v,-u,w). When the u coordinate is
        # structurally absent from A' and b', replacing u by v preserves both
        # affine equations and selects the stable x=0 boundary representative.
        # The operator is only materialized when an Exp block actually exists:
        # on a pure SOC/linear model the loop below never reads it.
        if !in_canonical_cone(canonical, y; dual=true, tol=tol)
            exp_operator = dual_operator
            if exp_operator === nothing && !isempty(state.runtime.exp)
                exp_operator =
                    _product_hsd_terminal_dual_operator(A, canonical.b)
            end
            if exp_operator !== nothing
                for block in canonical.cone_layout.blocks
                    block.cone === :exp || continue
                    u = block.offset
                    structurally_free = true
                    @inbounds for row in axes(exp_operator, 1)
                        if !iszero(exp_operator[row, u])
                            structurally_free = false
                            break
                        end
                    end
                    structurally_free &&
                        _store_owned_scalar!(y, u, y[u + 1])
                end
            end
        end
    catch exception
        exception isa LinearAlgebra.SingularException ||
            exception isa LinearAlgebra.PosDefException ||
            exception isa LinearAlgebra.RankDeficientException || rethrow()
        return nothing
    end

    primal_residual = A * x + s - canonical.b
    dual_residual = transpose(A) * y + canonical.c
    primal_scale = max(
        _product_hsd_refinement_scale(x),
        _product_hsd_refinement_scale(s),
        _product_hsd_refinement_scale(canonical.b),
    )
    dual_scale = max(
        _product_hsd_refinement_scale(y),
        _product_hsd_refinement_scale(canonical.c),
    )
    _product_hsd_refinement_maxabs(primal_residual) <= tol * primal_scale ||
        return nothing
    _product_hsd_refinement_maxabs(dual_residual) <= tol * dual_scale ||
        return nothing
    abs(dot(canonical.c, x) + dot(canonical.b, y)) <=
        tol * (one(T) + abs(dot(canonical.c, x)) + abs(dot(canonical.b, y))) ||
        return nothing
    in_canonical_cone(canonical, s; dual=false, tol=tol) || return nothing
    in_canonical_cone(canonical, y; dual=true, tol=tol) || return nothing

    saved_x = copy_owned!(alloc_zeros(T, length(base.x)), base.x)
    saved_s = copy_owned!(alloc_zeros(T, length(base.s)), base.s)
    saved_y = copy_owned!(alloc_zeros(T, length(base.y)), base.y)
    saved_kappa = base.kappa
    @inbounds for index in eachindex(base.x)
        base.x[index] = base.tau * x[index]
    end
    @inbounds for index in eachindex(base.s)
        base.s[index] = base.tau * s[index]
    end
    @inbounds for index in eachindex(base.y)
        base.y[index] = base.tau * y[index]
    end
    # Once primal feasibility, stationarity, and the recovered gap have been
    # refined, the scalar HSD equation has the exact solution kappa=0. Keep a
    # small positive representative so the homogeneous point remains valid.
    base.kappa = min(base.kappa, base.tau * tol / T(8))
    # PR-01: x/s/y/kappa were overwritten above, so the cached residual is stale.
    _product_hsd_bump_point_epoch!(state)
    _product_hsd_residual!(state)
    if verify_optimal!(
        canonical, base, x_original, s_original, y_original; tol=tol,
    )
        return _product_hsd_make_result(
            state, ProductHSDOptimal, reason, last_step, terminal_alpha,
            x_original, s_original, y_original,
        )
    end

    copy_owned!(base.x, saved_x)
    copy_owned!(base.s, saved_s)
    copy_owned!(base.y, saved_y)
    base.kappa = saved_kappa
    _product_hsd_bump_point_epoch!(state)
    _product_hsd_residual!(state)
    return nothing
end

Base.@noinline function _product_hsd_candidate_result!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    reason::ProductHSDSolveReason,
    last_step::HSDStepCode,
    terminal_alpha::T=zero(T),
) where {T}
    # The unchanged accepted HSD point is always the first certificate
    # candidate, including Exp/Power products. Refinement is a recovery step,
    # never a prerequisite for invoking the authoritative verifier.
    direct = _product_hsd_verified_result(
        state, x_original, s_original, y_original, tol, reason, last_step,
        terminal_alpha; check_optimal=true,
    )
    direct === nothing || return direct
    return _product_hsd_refined_optimal_result!(
        state,x_original,s_original,y_original,tol,reason,last_step,
        terminal_alpha,
    )
end

@inline function _product_hsd_terminal_alpha(
    state::ProductConeHSDState{T}, tol::T,
) where {T}
    base = state.base
    # A failed nonsymmetric scaling update invalidates the runtime factor.
    # There is then no authoritative cone metric from which to compute a
    # terminal boundary step; fail closed instead of calling the throwing
    # max-step API on invalid state.
    state.runtime.valid || return T(NaN)
    ap = max_step_primal!(state.runtime, base.s, base.ds)
    ad = max_step_dual!(state.runtime, base.y, base.dy)
    (isfinite(ap) || ap == T(Inf)) || return T(NaN)
    (isfinite(ad) || ad == T(Inf)) || return T(NaN)
    (ap >= zero(T) && ad >= zero(T)) || return T(NaN)
    alpha = min(one(T), ap, ad)
    base.dtau < zero(T) && (alpha = min(alpha, -base.tau / base.dtau))
    base.dkappa < zero(T) &&
        (alpha = min(alpha, -base.kappa / base.dkappa))

    # A terminal certificate may be much closer to the boundary than a point
    # from which another NT factor must be computed.  It remains strictly
    # interior and uses one common alpha for every HSD coordinate.
    margin = max(sqrt(eps(T)), min(T(1) / T(1000), tol / T(10)))
    return (one(T) - margin) * alpha
end

@inline function _product_hsd_stage_terminal_trial!(
    state::ProductConeHSDState{T}, alpha::T,
) where {T}
    base = state.base
    @inbounds for j in 1:base.n
        _store_owned_scalar!(
            base.xt, j, base.x[j] + alpha * base.dx[j],
        )
    end
    @inbounds for k in 1:base.m
        _store_owned_scalar!(
            base.st, k, base.s[k] + alpha * base.ds[k],
        )
        _store_owned_scalar!(
            base.yt, k, base.y[k] + alpha * base.dy[k],
        )
    end
    base.tau_t = base.tau + alpha * base.dtau
    base.kappa_t = base.kappa + alpha * base.dkappa
    isfinite(base.tau_t) && isfinite(base.kappa_t) &&
        base.tau_t > zero(T) && base.kappa_t > zero(T) || return false
    product_strictly_interior(
        state.runtime, base.st, base.yt,
    ) || return false

    _product_hsd_trial_residual!(state)
    p2 = _hsd_maxinf(base.rPt)
    d2 = _hsd_maxinf(base.rDt)
    gap2 = dot(base.c, base.xt) + dot(base.b, base.yt) + base.kappa_t
    (isfinite(p2) && isfinite(d2) && isfinite(gap2)) || return false
    _hsd_residual_homotopy_ok(base, alpha, p2, d2, gap2) || return false
    scale = max(
        one(T), _hsd_maxinf(base.rP), _hsd_maxinf(base.rD), abs(base.rG),
    )
    guard = T(256) * sqrt(eps(T)) * scale
    return max(p2, d2, abs(gap2)) <= scale * T(1.0005) + guard
end

"""
Check the already-computed Newton direction after an ordinary line-search
breakdown.  This is not an alternative solver: it performs no factorization
and changes neither the direction nor the HSD equations.  A single global
strict-interior trial is promoted only if residual homotopy and an ordinary
original-coordinate certificate verifier both pass.
"""
@inline function _product_hsd_finish_terminal_restore!(state, result, restored::Bool)
    if !restored
        state.runtime.valid = false
        state.diagnostic = :post_result_state_restore_failed
    end
    return result
end

function _product_hsd_terminal_verified_result!(
    state::ProductConeHSDState{T},
    x_original::Vector{T},
    s_original::Vector{T},
    y_original::Vector{T},
    tol::T,
    last_step::HSDStepCode,
) where {T}
    base = state.base
    alpha = _product_hsd_terminal_alpha(state, tol)
    (isfinite(alpha) && alpha > zero(T)) || return nothing
    _product_hsd_stage_terminal_trial!(state, alpha) || return nothing

    saved_x = copy_owned!(alloc_zeros(T, length(base.x)), base.x)
    saved_s = copy_owned!(alloc_zeros(T, length(base.s)), base.s)
    saved_y = copy_owned!(alloc_zeros(T, length(base.y)), base.y)
    saved_tau = base.tau
    saved_kappa = base.kappa
    copy_owned!(base.x, base.xt)
    copy_owned!(base.s, base.st)
    copy_owned!(base.y, base.yt)
    base.tau = base.tau_t
    base.kappa = base.kappa_t
    # PR-01: terminal trial committed to base, so the cached residual is stale.
    _product_hsd_bump_point_epoch!(state)

    result = _product_hsd_candidate_result!(
        state, x_original, s_original, y_original, tol,
        ProductHSDVerifiedTerminalNewtonTrial, last_step, alpha,
    )

    # Keep the mutable state on its last accepted pair: unlike the terminal
    # certificate, that pair has an NT runtime which may safely be reused.
    copy_owned!(base.x, saved_x)
    copy_owned!(base.s, saved_s)
    copy_owned!(base.y, saved_y)
    base.tau = saved_tau
    base.kappa = saved_kappa
    # PR-01: restored to the saved pair; this is a different point than the
    # trial that was just certified, so the token advances and the explicit
    # residual below is required (not a duplicate).
    _product_hsd_bump_point_epoch!(state)
    _product_hsd_residual!(state)
    restored = if state.symmetric_core isa FixedTraceQ3CoreWorkspace
        _product_hsd_fixed_trace_hkm_neighborhood!(
            state, base.s, base.y, base.mu,
        )
    elseif try_update_scaling!(state.runtime, base.s, base.y, base.mu)
        true
    elseif isempty(state.runtime.exp) && isempty(state.runtime.power)
        try_update_scaling!(
            state.runtime, base.s, base.y, base.mu;
            allow_conditioned_soc=true,
        )
    else
        false
    end
    # `_product_hsd_make_result` has copied every verified coordinate. A
    # failure to restore this reusable mutable runtime cannot revoke that
    # immutable result; retain it and invalidate only the abandoned state.
    return _product_hsd_finish_terminal_restore!(state, result, restored)
end

"""
    product_hsd_solve!(state; max_iterations=300, max_time=Inf, tol=nothing)

Drive the internal native LP/SOC/PSD HSD step and return a typed cold-path
result.  Status promotion is certificate-only.  This routine never calls the
legacy orthant solve, a projected-gradient Farkas search, or a lifted cone
formulation, and it is intentionally not connected to a public solver route.
"""
function product_hsd_solve!(
    state::ProductConeHSDState{T};
    max_iterations::Integer=300,
    max_time::Real=Inf,
    tol::Union{Nothing,T}=nothing,
    initialization::Symbol=:auto,
    max_tau_collapse_recoveries::Integer=1,
) where {T}
    # @integration/core-cutover: this function's body WAS the HSD loop.  The
    # loop now lives in `src/solver/loop.jl` as `solver_run_session!`, which is
    # the ONE production loop; this wrapper keeps the existing callers, and the
    # black-box contract they rely on, working unchanged.
    #
    # `solver_run_session!` validates every argument this function validated
    # (identical messages), performs the same setup, and returns a
    # `SessionOutcome` whose `result` field is the `ProductHSDSolveResult` this
    # function has always returned.  `_session_outcome` documents itself as a
    # labelling step that never changes the carried result.
    #
    # `stagnation_limit` keeps its session default of 0, which reproduces the
    # production control flow.  Nothing here opts into a new behaviour or a new
    # default strategy.
    outcome = solver_run_session!(
        SessionState(state);
        max_iterations=max_iterations,
        max_time=max_time,
        tol=tol,
        initialization=initialization,
        max_tau_collapse_recoveries=max_tau_collapse_recoveries,
    )
    return outcome.result
end

"""Construct an internal product-HSD state and solve it."""
function product_hsd_solve(
    canonical::CanonicalConicProgram{T};
    kkt_route::Symbol=:bordered, kwargs...,
) where {T<:AbstractFloat}
    state = ProductConeHSDState(canonical; kkt_route=kkt_route)
    return product_hsd_solve!(state; kwargs...)
end
