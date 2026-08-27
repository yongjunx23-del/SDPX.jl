#=====================================================================#

@enum SymmetricBorderedReason::UInt8 begin
    SYMMETRIC_BORDERED_READY
    SYMMETRIC_BORDERED_ORDER_FAILED
    SYMMETRIC_BORDERED_EPOCH_MISMATCH
    SYMMETRIC_BORDERED_ASSEMBLY_NONFINITE
    SYMMETRIC_BORDERED_ZERO_ROW
    SYMMETRIC_BORDERED_SCALING_FAILED
    SYMMETRIC_BORDERED_TRANSFORM_FAILED
    SYMMETRIC_BORDERED_FACTOR_FAILED
    SYMMETRIC_BORDERED_FACTOR_CERT_FAILED
    SYMMETRIC_BORDERED_RHS_FAILED
    SYMMETRIC_BORDERED_SOLVE_FAILED
    SYMMETRIC_BORDERED_SOLVE_CERT_FAILED
    SYMMETRIC_BORDERED_ORIGINAL_CERT_FAILED
    SYMMETRIC_BORDERED_RECOVERY_FAILED
    SYMMETRIC_BORDERED_FIVE_EQUATION_FAILED
    SYMMETRIC_BORDERED_CONDITIONED_FINAL_CERTIFIED
end

const _SYMMETRIC_BORDERED_TRANSFORM_ORDER = UInt8(0x01)

"""Preallocated, pivoted original-border workspace for symmetric products."""
mutable struct SymmetricBorderedWorkspace{
    T,D<:HotRouteCache{T},
}
    nr::Int
    dimension::Int
    matrix::Matrix{T}
    factor_matrix::Matrix{T}
    rhs::Vector{T}
    factor_rhs::Vector{T}
    solution::Vector{T}
    certified_solution::Vector{T}
    previous_solution::Vector{T}
    correction_solution::Vector{T}
    residual::Vector{T}
    bound::Vector{T}
    certified_factor_bound::Vector{T}
    certified_physical_bound::Vector{T}
    row_scale::Vector{T}
    row_exponent::Vector{Int}
    permutation::Vector{Int}
    factor_error::Matrix{T}
    permuted_rhs::Vector{T}
    staged_y::Vector{T}
    forward_residual::Vector{T}
    backward_residual::Vector{T}
    identity_rhs::Vector{T}
    upper_work::Vector{T}
    lower_work::Vector{T}
    driver::D
    factor_certified::Bool
    original_solution_certified::Bool
    assembly_epoch::Int
    factor_epoch::Int
    transform_order::UInt8
    accumulated_candidate::Bool
    candidate_epoch::Int
    accumulations::Int
    solves::Int
    refinements::Int
    last_reason::SymmetricBorderedReason
end

function SymmetricBorderedWorkspace(::Type{T}, nr::Integer) where {T}
    reduced = Int(nr)
    reduced >= 0 || throw(ArgumentError("negative bordered reduced dimension"))
    dimension = reduced + 1
    route = LPLUCache{T}(dimension)
    driver = HotRouteCache(route; n=dimension)
    return SymmetricBorderedWorkspace{T,typeof(driver)}(
        reduced,
        dimension,
        zeros(T, dimension, dimension),
        zeros(T, dimension, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(Int, dimension),
        Vector{Int}(undef, dimension),
        zeros(T, dimension, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        zeros(T, dimension),
        driver,
        false,
        false,
        0,
        0,
        _SYMMETRIC_BORDERED_TRANSFORM_ORDER,
        false,
        0,
        0,
        0,
        0,
        SYMMETRIC_BORDERED_READY,
    )
end
# Native product-cone HSD core.
#
# This is a typed integration adapter around the already-audited `HSDState`.
# It deliberately reuses that state's canonical iterate, RRQR reduction,
# bordered solve, factor-cache driver, residual buffers, trial buffers and
# iteration record.  Only the cone-dependent shift, Schur assembly, recovery
# and boundary operations live here.  No public solve route or certificate
# status is selected in this phase.
#=====================================================================#

"""
    ProductConeHSDState{T,R,RT}

Typed native product-cone HSD execution state.  `base` owns the frozen HSD
embedding and reduced KKT storage; `runtime` owns pair-dependent scaling state
for every block. All vector, dense-three-block, and sparse-dyadic Schur scratch
is allocated at setup.

This adapter is the migration boundary for Phase 2: after the direction and
iteration gates are frozen, its runtime/scratch fields can be folded directly
into `HSDState` without changing the mathematical kernels below.
"""
mutable struct ProductConeHSDState{
    T,
    R<:AbstractFactorCache{T},
    RT<:ProductConeRuntime,
    NS,
    CW,
    SB,
}
    base::HSDState{T,R}
    runtime::RT
    h::Vector{T}
    g_input::Vector{T}
    g_output::Vector{T}
    gb::Vector{T}
    ds_hat::Vector{T}
    dy_hat::Vector{T}
    soc_g_error_bound::Vector{T}
    soc_roundtrip_bound::Vector{T}
    certified_soc_g_error_bound::Vector{T}
    certified_soc_roundtrip_bound::Vector{T}
    soc_bounds_certified::Bool
    ns_schur::NS
    ns_metrics::Array{T,3}
    ns_H::Matrix{T}
    ns_at_g_b::Vector{T}
    ns_bt_g_a::Vector{T}
    ns_at_g_rhs::Vector{T}
    ns_zero_rhs::Vector{T}
    coupled::CW
    symmetric_bordered::SB
end

function ProductConeHSDState(
    canonical::CanonicalConicProgram{T},
    driver::HotRouteCache{T,R},
) where {T<:AbstractFloat,R<:AbstractFactorCache{T}}
    base = HSDState(canonical, driver)
    return _product_cone_hsd_state(base)
end

function _product_cone_hsd_state(
    base::HSDState{T,R},
) where {T<:AbstractFloat,R<:AbstractFactorCache{T}}
    runtime = ProductConeRuntime(base.canonical.cone_layout, T)
    m = base.m
    block_count = length(runtime.exp) + length(runtime.power)
    offsets = Vector{Int}(undef, block_count)
    block_index = 0
    @inbounds for block in runtime.exp
        block_index += 1
        offsets[block_index] = block.offset
    end
    @inbounds for block in runtime.power
        block_index += 1
        offsets[block_index] = block.offset
    end
    ns_schur = NonsymmetricSchur3Workspace(base.Ar, offsets)
    coupled = NonsymmetricCoupledWorkspace(base.Ar, base.nr, offsets)
    # The symmetric product route intentionally owns a pivoted full-border
    # LPLU. `base.driver` remains part of the generic HSD storage contract but
    # is never factored by this route, even when a caller supplied it.
    symmetric_bordered = SymmetricBorderedWorkspace(T, base.nr)
    return ProductConeHSDState{
        T,R,typeof(runtime),typeof(ns_schur),typeof(coupled),
        typeof(symmetric_bordered),
    }(
        base,
        runtime,
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        false,
        ns_schur,
        zeros(T, 3, 3, block_count),
        zeros(T, base.nr, base.nr),
        zeros(T, base.nr),
        zeros(T, base.nr),
        zeros(T, base.nr),
        zeros(T, m),
        coupled,
        symmetric_bordered,
    )
end

function ProductConeHSDState(
    canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    reduction = _hsd_rowspace_reduction(canonical)
    cache = DenseSchurCholeskyCache{T}(reduction.rank)
    driver = HotRouteCache(cache; n=reduction.rank)
    base = _hsd_state_from_reduction(canonical, driver, reduction)
    return _product_cone_hsd_state(base)
end

@inline product_hsd_base(state::ProductConeHSDState) = state.base

"""Numeric factorizations performed by the active product-HSD route."""
@inline function product_hsd_factor_count(state::ProductConeHSDState)
    if state.coupled.nonsymmetric_dimension > 0
        return factor_epoch(state.coupled.cache)
    end
    return kkt_factor_count(state.symmetric_bordered.driver)
end

"""
    product_hsd_cold_start!(state)

Set `x=0`, `s=y=e`, `tau=kappa=1` using the actual product-cone identities.
The initial point is strictly interior but intentionally infeasible-start.
"""
function product_hsd_cold_start!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    fill!(base.x, zero(T))
    initialize_primal_dual!(state.runtime, base.s, base.y)
    base.tau = one(T)
    base.kappa = one(T)
    return state
end

@inline function _product_hsd_vector_finite(v)
    @inbounds for x in v
        isfinite(x) || return false
    end
    return true
end

@inline function _product_hsd_nonsymmetric_roundtrip_ok(target, block)
    T = eltype(target)
    theta = block.scaling.theta
    gamma = T(3) * eps(T) / (one(T) - T(3) * eps(T))
    forcing = T(128) * gamma
    isfinite(forcing) || return false

    @inbounds for i in 1:3
        qi = target[block.offset + i - 1]
        map_work = zero(T)
        for j in 1:3
            map_work += abs(theta[i, j]) * abs(block.input[j])
        end
        work = map_work + abs(qi)
        residual = block.output[i] - qi
        isfinite(work) && isfinite(residual) || return false
        if iszero(work)
            iszero(residual) || return false
        elseif abs(residual) > forcing * work
            return false
        end
    end
    return true
end

@inline function _product_hsd_soc_condition_budget(w, n::Int)
    T = eltype(w)
    w0 = T(w[1])
    tail2 = zero(T)
    @inbounds for i in 2:n
        wi = T(w[i])
        isfinite(wi) || return T(Inf)
        tail2 += wi * wi
    end
    radius = sqrt(tail2)
    lambda_plus = w0 + radius
    determinant = (w0 - radius) * lambda_plus
    lambda_minus = determinant / lambda_plus
    isfinite(lambda_plus) && isfinite(lambda_minus) &&
        lambda_plus > zero(T) && lambda_minus > zero(T) || return T(Inf)
    ratio = lambda_plus / lambda_minus
    kappa_theta = ratio * ratio
    gamma_argument = T(3n + 12) * eps(T)
    isfinite(gamma_argument) && gamma_argument < one(T) || return T(Inf)
    gamma = gamma_argument / (one(T) - gamma_argument)
    return T(64) * gamma * kappa_theta
end

@inline function _product_hsd_soc_q_coefficient(
    w::AbstractVector{T}, n::Int, i::Int, j::Int,
) where {T}
    w0 = w[1]
    ww = zero(T)
    @inbounds for k in 2:n
        ww += w[k] * w[k]
    end
    two = one(T) + one(T)
    if i == 1
        return j == 1 ? w0 * w0 + ww : two * w0 * w[j]
    elseif j == 1
        return two * w0 * w[i]
    end
    diagonal = i == j ? w0 * w0 - ww : zero(T)
    return diagonal + two * w[i] * w[j]
end

@inline function _product_hsd_soc_roundtrip_ok(
    target, block, g_error_bound, roundtrip_bound,
)
    T = eltype(target)
    n = block.dim
    w = block.state.w
    winv = block.state.winv
    z = block.input
    computed = block.output
    SymmetricCones._soc_q_condition_reliable(w, n) || return false

    # `z` was produced by Q_{w^{-1}} before this Q_w application.  Its
    # backward error is amplified by kappa(Q_w), so an output-only
    # sqrt(eps) forcing term is not a valid certificate near a curved cone
    # boundary.  Reuse the same finite, capped condition budget that admits
    # the frozen SOC scaling, and apply it to each row's actual map work.
    # Crossing the one-percent cap remains a hard fail-closed request for
    # more working precision.
    budget = _product_hsd_soc_condition_budget(w, n)
    isfinite(budget) && budget < one(T) / T(100) || return false
    gamma = _product_bordered_gamma(T, 12n + 24)
    isfinite(gamma) || return false
    offset = block.offset

    # Independently replay Q_{w^-1} as a dense polynomial in the stored NT
    # coordinates. Its componentwise actual-work bound is retained for the
    # later full dual/scalar equations; the observed solve residual is only a
    # gate and is never used to construct that bound.
    @inbounds for i in 1:n
        predicted = zero(T)
        work = zero(T)
        for j in 1:n
            coefficient = _product_hsd_soc_q_coefficient(winv, n, i, j)
            qi = target[offset + j - 1]
            term = coefficient * qi
            isfinite(coefficient) && isfinite(term) || return false
            predicted += term
            work += abs(term)
        end
        residual = z[i] - predicted
        allowance = gamma * work
        _product_bordered_zero_safe_close(residual, allowance) || return false
        g_error_bound[offset + i - 1] = allowance
    end

    # Certify Q_w*Q_{w^-1} by the same factor-reconstruction identity used by
    # the bordered LU: E = A*B-I, followed by E*q plus the independently
    # bounded B*q and A*z arithmetic. This is substantially tighter than the
    # a-priori kappa cap while retaining that cap as a hard reliability gate.
    @inbounds for i in 1:n
        theta_predicted = zero(T)
        theta_work = zero(T)
        propagated = zero(T)
        for k in 1:n
            aik = _product_hsd_soc_q_coefficient(w, n, i, k)
            term = aik * z[k]
            isfinite(aik) && isfinite(term) || return false
            theta_predicted += term
            theta_work += abs(term)
            propagated += abs(aik) * g_error_bound[offset + k - 1]
        end
        replay_residual = computed[i] - theta_predicted
        _product_bordered_zero_safe_close(
            replay_residual, gamma * theta_work,
        ) || return false

        reconstruction = zero(T)
        for j in 1:n
            product = zero(T)
            product_work = zero(T)
            for k in 1:n
                aik = _product_hsd_soc_q_coefficient(w, n, i, k)
                bkj = _product_hsd_soc_q_coefficient(winv, n, k, j)
                term = aik * bkj
                isfinite(term) || return false
                product += term
                product_work += abs(term)
            end
            identity = i == j ? one(T) : zero(T)
            factor_error = product - identity
            factor_bound = abs(factor_error) +
                           gamma * (product_work + abs(identity))
            qj = target[offset + j - 1]
            term = factor_bound * abs(qj)
            isfinite(factor_bound) && isfinite(term) || return false
            reconstruction += term
        end
        theta_rounding = gamma * theta_work
        subtotal = reconstruction + propagated + theta_rounding
        allowance = subtotal + gamma * subtotal
        residual = computed[i] - target[offset + i - 1]
        isfinite(reconstruction) && isfinite(propagated) &&
            isfinite(theta_rounding) && isfinite(allowance) || return false
        _product_bordered_zero_safe_close(residual, allowance) || return false
        roundtrip_bound[offset + i - 1] = allowance
    end
    return true
end

"""Map-aware backward gate for the frozen `Theta*(G*q)` round trip.

The residual is measured against the arithmetic work of each block map, not
against a cancellation-small `q`.  This is essential at cone-boundary
solutions, where a forward/output-relative test rejects a backward-stable
Lorentz or PSD map solely because its condition number is large.
"""
@inline function _product_hsd_roundtrip_backward_status(
    state::ProductConeHSDState{T},
) where {T}
    runtime = state.runtime
    target = state.g_input
    psd_budget_inconclusive = false
    state.soc_bounds_certified = false
    fill!(state.soc_g_error_bound, zero(T))
    fill!(state.soc_roundtrip_bound, zero(T))

    @inbounds for block in runtime.orthant
        gamma = T(5) * eps(T) / (one(T) - T(5) * eps(T))
        isfinite(gamma) || return false, false
        for i in 1:block.dim
            k = block.offset + i - 1
            ti = target[k]
            work = abs(block.state.theta[i] * block.input[i]) + abs(ti)
            allowance = T(64) * gamma * work
            isfinite(allowance) && abs(block.output[i] - ti) <= allowance ||
                return false, false
        end
    end

    @inbounds for block in runtime.soc
        for i in 1:block.dim
            block.direction[i] = target[block.offset + i - 1]
        end
        _product_hsd_soc_roundtrip_ok(
            target, block, state.soc_g_error_bound,
            state.soc_roundtrip_bound,
        ) || return false, false
    end

    @inbounds for block in runtime.psd
        n = block.dim
        for i in 1:block.len
            block.direction[i] = target[block.offset + i - 1]
        end
        pnorm = zero(T)
        pinvnorm = zero(T)
        for i in 1:n
            rowsum = zero(T)
            invrowsum = zero(T)
            for j in 1:n
                rowsum += abs(block.state.P[i, j])
                invrowsum += abs(block.state.Pinv[i, j])
            end
            pnorm = max(pnorm, rowsum)
            pinvnorm = max(pinvnorm, invrowsum)
        end
        qnorm = zero(T)
        residual = zero(T)
        for i in 1:block.len
            qnorm = max(qnorm, abs(block.direction[i]))
            residual = max(residual, abs(block.output[i] - block.direction[i]))
        end
        gamma = T(2n + 8) * eps(T) / (one(T) - T(2n + 8) * eps(T))
        condition_bound = pnorm * pinvnorm
        budget = T(64) * gamma * condition_bound * condition_bound
        isfinite(budget) && isfinite(qnorm) && isfinite(residual) ||
            return false, false
        if budget >= one(T) / T(100)
            # This cap is an a-priori reliability guard, not an observed
            # residual failure. Defer only this finite PSD case to the
            # componentwise original cone equation after `ds` is recovered.
            psd_budget_inconclusive = true
        elseif residual > budget * qnorm
            return false, false
        end
    end

    # A nonsymmetric block applies G by solving the accepted Theta system.
    # Validate that actual solve with the arithmetic work of Theta*dy; the
    # separately stored explicit inverse is diagnostic-only.
    @inbounds for block in runtime.exp
        _product_hsd_nonsymmetric_roundtrip_ok(target, block) ||
            return false, false
    end
    @inbounds for block in runtime.power
        _product_hsd_nonsymmetric_roundtrip_ok(target, block) ||
            return false, false
    end
    copyto!(state.certified_soc_g_error_bound, state.soc_g_error_bound)
    copyto!(
        state.certified_soc_roundtrip_bound,
        state.soc_roundtrip_bound,
    )
    state.soc_bounds_certified = true
    return !psd_budget_inconclusive, psd_budget_inconclusive
end

@inline function _product_hsd_roundtrip_backward_ok(
    state::ProductConeHSDState,
)
    certified, _ = _product_hsd_roundtrip_backward_status(state)
    return certified
end

"""Apply only the symmetric block metrics, leaving nonsymmetric rows zero.

The Exp/Power rows are assembled by the sparse three-row dyadic kernel below;
excluding them here prevents both a dense product-column pass and double
counting in mixed products.
"""
@inline function _product_hsd_apply_symmetric_G!(runtime, dst, src)
    fill!(dst, zero(eltype(dst)))
    @inbounds for block in runtime.orthant
        _runtime_copy_in!(block.input, src, block.offset, block.dim)
        SymmetricCones.g_apply!(
            block.cone, block.output, block.state, block.input,
        )
        _runtime_copy_out!(dst, block.offset, block.output, block.dim)
    end
    @inbounds for block in runtime.soc
        _runtime_copy_in!(block.input, src, block.offset, block.dim)
        SymmetricCones.g_apply!(
            block.cone, block.output, block.state, block.input,
        )
        _runtime_copy_out!(dst, block.offset, block.output, block.dim)
    end
    @inbounds for block in runtime.psd
        _runtime_copy_in!(block.input, src, block.offset, block.len)
        SymmetricCones.g_apply!(
            block.cone, block.output, block.state, block.input,
        )
        _runtime_copy_out!(dst, block.offset, block.output, block.len)
    end
    _product_hsd_vector_finite(dst) || throw(DomainError(
        dst, "symmetric G output contains non-finite data",
    ))
    return dst
end

@inline function _product_hsd_copy_nonsymmetric_thetas!(
    state::ProductConeHSDState,
)
    block_index = 0
    metrics = state.ns_metrics
    @inbounds for block in state.runtime.exp
        block_index += 1
        for j in 1:3, i in 1:3
            metrics[i, j, block_index] = block.scaling.theta[i, j]
        end
    end
    @inbounds for block in state.runtime.power
        block_index += 1
        for j in 1:3, i in 1:3
            metrics[i, j, block_index] = block.scaling.theta[i, j]
        end
    end
    return block_index
end

@inline function _product_hsd_add_nonsymmetric_schur!(
    state::ProductConeHSDState{T}, rhs,
) where {T}
    _product_hsd_copy_nonsymmetric_thetas!(state)
    result = try_assemble_nonsymmetric_schur3_theta!(
        state.ns_schur,
        state.ns_H,
        state.ns_at_g_b,
        state.ns_bt_g_a,
        state.ns_at_g_rhs,
        state.ns_metrics,
        state.base.b,
        rhs,
    )
    return result
end

"""
Assemble `H=Ar'GAr` and the shared homogeneous border without materialising
the global `m x m` operator `G`. Symmetric blocks retain the per-column block
map; every Exp/Power block is accumulated by the sparse three-row dyadic
assembler over the frozen reduced matrix.
"""
@inline function _product_hsd_form_schur_border!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    A = base.Ar
    H = base.H
    nr = base.nr
    fill!(H, zero(T))
    @inbounds for j in 1:nr
        fill!(state.g_input, zero(T))
        for ptr in nzrange(A, j)
            state.g_input[A.rowval[ptr]] = A.nzval[ptr]
        end
        _product_hsd_apply_symmetric_G!(
            state.runtime, state.g_output, state.g_input,
        )
        for i in 1:j
            acc = zero(T)
            for ptr in nzrange(A, i)
                acc += A.nzval[ptr] * state.g_output[A.rowval[ptr]]
            end
            H[i, j] = acc
            H[j, i] = acc
        end
    end

    copyto!(state.g_input, base.b)
    _product_hsd_apply_symmetric_G!(state.runtime, state.gb, state.g_input)
    bgb = zero(T)
    @inbounds for k in 1:base.m
        bgb += base.b[k] * state.gb[k]
    end

    has_nonsymmetric = !isempty(state.runtime.exp) ||
                       !isempty(state.runtime.power)
    if has_nonsymmetric
        fill!(state.ns_zero_rhs, zero(T))
        ns_result = _product_hsd_add_nonsymmetric_schur!(
            state, state.ns_zero_rhs,
        )
        ns_result.status === NS_SCHUR3_ASSEMBLED || return T(NaN)
        @inbounds for j in 1:nr, i in 1:nr
            H[i, j] += state.ns_H[i, j]
        end
        bgb += ns_result.b_g_b
    end

    @inbounds for j in 1:nr
        atgb = zero(T)
        for ptr in nzrange(A, j)
            atgb += A.nzval[ptr] * state.gb[A.rowval[ptr]]
        end
        has_nonsymmetric && (atgb += state.ns_at_g_b[j])
        cj = base.cr[j]
        base.qr[j] = cj - atgb
        base.rvec[j] = base.tau * (cj + atgb)
    end
    return base.kappa - base.tau * bgb
end

@inline function _product_bordered_gamma(
    ::Type{T}, operations::Int,
) where {T}
    scaled = T(operations) * eps(one(T))
    isfinite(scaled) && zero(T) <= scaled < one(T) || return T(Inf)
    return scaled / (one(T) - scaled)
end

@inline function _product_bordered_zero_safe_close(
    residual::T, allowance::T,
) where {T}
    isfinite(residual) && isfinite(allowance) && allowance >= zero(T) ||
        return false
    iszero(allowance) && return iszero(residual)
    return abs(residual) <= allowance
end

@inline function _product_bordered_transform_entry_ok(
    transformed::T, scale::T, original::T,
) where {T}
    isfinite(transformed) && isfinite(scale) && isfinite(original) &&
        scale > zero(T) || return false
    expected = scale * original
    isfinite(expected) || return false
    residual = transformed - expected
    work = abs(transformed) + abs(expected)
    gamma = _product_bordered_gamma(T, 2)
    _product_bordered_zero_safe_close(residual, gamma * work) || return false
    transformed == expected || return false
    if iszero(original)
        return iszero(transformed)
    end
    !iszero(transformed) || return false
    recovered = transformed / scale
    return isfinite(recovered) && recovered == original
end

@inline function _product_bordered_transform_matrix_ok(
    workspace::SymmetricBorderedWorkspace{T},
) where {T}
    workspace.transform_order === _SYMMETRIC_BORDERED_TRANSFORM_ORDER ||
        return false
    n = workspace.dimension
    @inbounds for i in 1:n
        power = workspace.row_exponent[i]
        scale = workspace.row_scale[i]
        expected_scale = try
            ldexp(one(T), power)
        catch
            return false
        end
        isfinite(expected_scale) && expected_scale > zero(T) &&
            scale == expected_scale || return false
        for j in 1:n
            _product_bordered_transform_entry_ok(
                workspace.factor_matrix[i, j], scale,
                workspace.matrix[i, j],
            ) || return false
        end
    end
    return true
end

@inline function _product_bordered_transform_rhs_ok(
    workspace::SymmetricBorderedWorkspace{T},
) where {T}
    n = workspace.dimension
    @inbounds for i in 1:n
        _product_bordered_transform_entry_ok(
            workspace.factor_rhs[i], workspace.row_scale[i],
            workspace.rhs[i],
        ) || return false
    end
    return true
end

@inline function _product_hsd_assemble_bordered!(
    state::ProductConeHSDState{T}, border_scalar::T,
) where {T}
    base = state.base
    workspace = state.symmetric_bordered
    n = workspace.dimension
    nr = workspace.nr
    workspace.factor_certified = false
    workspace.original_solution_certified = false
    workspace.last_reason = SYMMETRIC_BORDERED_READY
    workspace.assembly_epoch = 0
    workspace.factor_epoch = 0
    workspace.accumulated_candidate = false
    workspace.candidate_epoch = 0
    workspace.accumulations = 0
    workspace.solves = 0
    workspace.refinements = 0
    workspace.transform_order === _SYMMETRIC_BORDERED_TRANSFORM_ORDER || begin
        workspace.last_reason = SYMMETRIC_BORDERED_ORDER_FAILED
        return false
    end
    nr == base.nr && n == nr + 1 || begin
        workspace.last_reason = SYMMETRIC_BORDERED_ORDER_FAILED
        return false
    end

    @inbounds for j in 1:nr
        for i in 1:nr
            workspace.matrix[i, j] = base.H[i, j]
        end
        workspace.matrix[j, n] = base.qr[j]
        workspace.matrix[n, j] = base.rvec[j]
    end
    workspace.matrix[n, n] = border_scalar

    @inbounds for i in 1:n
        row_max = zero(T)
        for j in 1:n
            value = workspace.matrix[i, j]
            isfinite(value) || begin
                workspace.last_reason = SYMMETRIC_BORDERED_ASSEMBLY_NONFINITE
                return false
            end
            magnitude = abs(value)
            magnitude > row_max && (row_max = magnitude)
        end
        row_max > zero(T) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_ZERO_ROW
            return false
        end
        raw_exponent = try
            exponent(row_max)
        catch
            workspace.last_reason = SYMMETRIC_BORDERED_SCALING_FAILED
            return false
        end
        raw_exponent == typemin(Int) && begin
            workspace.last_reason = SYMMETRIC_BORDERED_SCALING_FAILED
            return false
        end
        power = -raw_exponent
        scale = try
            ldexp(one(T), power)
        catch
            workspace.last_reason = SYMMETRIC_BORDERED_SCALING_FAILED
            return false
        end
        isfinite(scale) && scale > zero(T) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_SCALING_FAILED
            return false
        end
        workspace.row_exponent[i] = power
        workspace.row_scale[i] = scale
        for j in 1:n
            transformed = scale * workspace.matrix[i, j]
            isfinite(transformed) || begin
                workspace.last_reason = SYMMETRIC_BORDERED_TRANSFORM_FAILED
                return false
            end
            workspace.factor_matrix[i, j] = transformed
        end
    end
    _product_bordered_transform_matrix_ok(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_TRANSFORM_FAILED
        return false
    end
    workspace.assembly_epoch = base.epoch
    return true
end

@inline function _product_bordered_permutation!(
    workspace::SymmetricBorderedWorkspace,
)
    n = workspace.dimension
    pivots = workspace.driver.route.ipiv
    @inbounds for i in 1:n
        workspace.permutation[i] = i
    end
    @inbounds for i in 1:n
        pivot = pivots[i]
        1 <= pivot <= n || return false
        if pivot != i
            workspace.permutation[i], workspace.permutation[pivot] =
                workspace.permutation[pivot], workspace.permutation[i]
        end
    end
    return true
end

@inline function _product_bordered_factor_certificate!(
    workspace::SymmetricBorderedWorkspace{T},
) where {T}
    n = workspace.dimension
    T(8n) * eps(one(T)) < one(T) || return false
    _product_bordered_transform_matrix_ok(workspace) || return false
    _product_bordered_permutation!(workspace) || return false
    F = workspace.driver.route.factors
    P = workspace.permutation
    E = workspace.factor_error
    gamma = _product_bordered_gamma(T, 4n)
    isfinite(gamma) || return false
    @inbounds for j in 1:n
        for i in 1:n
            product = zero(T)
            product_work = zero(T)
            for k in 1:min(i, j)
                lik = i == k ? one(T) : F[i, k]
                term = lik * F[k, j]
                product += term
                product_work += abs(term)
            end
            pm = workspace.factor_matrix[P[i], j]
            error = pm - product
            E[i, j] = error
            allowance = gamma * (abs(pm) + product_work)
            _product_bordered_zero_safe_close(error, allowance) || return false
        end
    end
    return true
end

@inline function _product_hsd_factor_bordered!(
    state::ProductConeHSDState,
)
    base = state.base
    workspace = state.symmetric_bordered
    workspace.factor_certified = false
    workspace.assembly_epoch == base.epoch || begin
        workspace.last_reason = SYMMETRIC_BORDERED_EPOCH_MISMATCH
        return false
    end
    _product_bordered_transform_matrix_ok(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_TRANSFORM_FAILED
        return false
    end
    factors_before = kkt_factor_count(workspace.driver)
    try
        kkt_epoch_factorize!(workspace.driver, workspace.factor_matrix)
    catch
        workspace.last_reason = SYMMETRIC_BORDERED_FACTOR_FAILED
        return false
    end
    kkt_factor_count(workspace.driver) == factors_before + 1 || begin
        workspace.last_reason = SYMMETRIC_BORDERED_FACTOR_FAILED
        return false
    end
    @inbounds for i in 1:workspace.dimension
        pivot = workspace.driver.route.factors[i, i]
        isfinite(pivot) && !iszero(pivot) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_FACTOR_FAILED
            return false
        end
    end
    if !_product_bordered_factor_certificate!(workspace)
        workspace.last_reason = SYMMETRIC_BORDERED_FACTOR_CERT_FAILED
        return false
    end
    workspace.factor_epoch = base.epoch
    workspace.factor_certified = true
    return true
end

@inline function _product_bordered_physical_snapshot_ok(
    workspace::SymmetricBorderedWorkspace{T},
) where {T}
    workspace.factor_certified &&
        workspace.original_solution_certified &&
        workspace.factor_epoch == workspace.assembly_epoch || return false
    _product_bordered_transform_matrix_ok(workspace) || return false
    _product_bordered_transform_rhs_ok(workspace) || return false
    _product_bordered_factor_certificate!(workspace) || return false
    n = workspace.dimension
    @inbounds for i in 1:n
        workspace.solution[i] == workspace.certified_solution[i] ||
            return false
        if workspace.accumulated_candidate
            workspace.candidate_epoch == workspace.factor_epoch ||
                return false
            workspace.certified_solution[i] ==
                workspace.previous_solution[i] +
                workspace.correction_solution[i] || return false
        end
        bound = workspace.bound[i]
        bound == workspace.certified_physical_bound[i] || return false
        isfinite(bound) && bound >= zero(T) || return false
    end
    return true
end

@inline function _product_hsd_has_nonsymmetric(
    state::ProductConeHSDState,
)
    return state.coupled.nonsymmetric_dimension > 0
end

"""Assemble the frozen original-Theta hybrid coupled matrix.

Rows are `[C_N; D; G; T]`, columns are `[dx_r; dy_N; dτ; dκ]`.  Only
symmetric blocks are eliminated; every Exp/Power dual direction remains in
the pivoted system.  Consequently the nonsymmetric operator appearing in the
matrix is the accepted `Theta` itself, never an explicit or factor-projected
inverse.
"""
@inline function _product_hsd_form_coupled_matrix!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    runtime = state.runtime
    workspace = state.coupled
    K = workspace.matrix
    nr = base.nr
    nsdim = workspace.nonsymmetric_dimension
    dual_row0 = nsdim
    dy_col0 = nr
    dtau_column = nr + nsdim + 1
    dkappa_column = dtau_column + 1
    gap_row = nsdim + nr + 1
    scalar_row = gap_row + 1
    fill!(K, zero(T))

    # Static A_N coefficients in C_N and D.
    @inbounds for j in 1:nr
        for pointer in nzrange(base.Ar, j)
            row = base.Ar.rowval[pointer]
            local_row = workspace.row_to_local[row]
            iszero(local_row) && continue
            value = base.Ar.nzval[pointer]
            K[local_row, j] -= value
            K[dual_row0 + j, dy_col0 + local_row] += value
        end
    end
    @inbounds for local_row in 1:nsdim
        row = workspace.offsets[div(local_row - 1, 3) + 1] +
              mod(local_row - 1, 3)
        K[local_row, dtau_column] = base.b[row]
        K[gap_row, dy_col0 + local_row] = -base.b[row]
    end

    # Accepted original nonsymmetric Theta blocks.
    @inbounds for block in runtime.exp
        local0 = workspace.row_to_local[block.offset] - 1
        for j in 1:3, i in 1:3
            K[local0 + i, dy_col0 + local0 + j] =
                block.scaling.theta[i, j]
        end
    end
    @inbounds for block in runtime.power
        local0 = workspace.row_to_local[block.offset] - 1
        for j in 1:3, i in 1:3
            K[local0 + i, dy_col0 + local0 + j] =
                block.scaling.theta[i, j]
        end
    end

    # d=A_S'G_S*b_S and beta=b_S'G_S*b_S.
    copyto!(state.g_input, base.b)
    _product_hsd_apply_symmetric_G!(runtime, state.gb, state.g_input)
    beta = zero(T)
    @inbounds for row in 1:base.m
        beta += base.b[row] * state.gb[row]
    end
    @inbounds for j in 1:nr
        dj = zero(T)
        for pointer in nzrange(base.Ar, j)
            row = base.Ar.rowval[pointer]
            dj += base.Ar.nzval[pointer] * state.gb[row]
        end
        state.ns_at_g_b[j] = dj
    end

    # H_S=A_S'G_S*A_S.  The symmetric-only map zeros all N rows, so the
    # sparse contractions below cannot double count an Exp/Power block.
    @inbounds for column in 1:nr
        fill!(state.g_input, zero(T))
        for pointer in nzrange(base.Ar, column)
            state.g_input[base.Ar.rowval[pointer]] +=
                base.Ar.nzval[pointer]
        end
        _product_hsd_apply_symmetric_G!(
            runtime, state.g_output, state.g_input,
        )
        for row_column in 1:nr
            value = zero(T)
            for pointer in nzrange(base.Ar, row_column)
                row = base.Ar.rowval[pointer]
                value += base.Ar.nzval[pointer] * state.g_output[row]
            end
            K[dual_row0 + row_column, column] = value
        end
    end

    @inbounds for j in 1:nr
        dj = state.ns_at_g_b[j]
        K[dual_row0 + j, dtau_column] = base.cr[j] - dj
        K[gap_row, j] = -(base.cr[j] + dj)
    end
    K[gap_row, dtau_column] = beta
    K[gap_row, dkappa_column] = one(T)
    K[scalar_row, dtau_column] = base.kappa
    K[scalar_row, dkappa_column] = base.tau

    _hsd_matrix_finite(K) || begin
        workspace.last_reason = COUPLED_ASSEMBLY_NONFINITE
        return false
    end
    # Freeze the accepted Theta factors before the global LU epoch.  The
    # source is the scaling workspace's certified lower Cholesky factor for
    # the current Theta; the coupled path keeps its own copy so line-search
    # checkpoints cannot mutate the matrix being factored.
    fill!(workspace.factor_coordinate_factor, zero(T))
    @inbounds for block in runtime.exp
        local0 = workspace.row_to_local[block.offset] - 1
        _product_coupled_copy_factor_block!(
            workspace, local0, block.scaling.factor, block.scaling.theta,
        ) || begin
            workspace.last_reason = COUPLED_TRANSFORM_FAILED
            return false
        end
    end
    @inbounds for block in runtime.power
        local0 = workspace.row_to_local[block.offset] - 1
        _product_coupled_copy_factor_block!(
            workspace, local0, block.scaling.factor, block.scaling.theta,
        ) || begin
            workspace.last_reason = COUPLED_TRANSFORM_FAILED
            return false
        end
    end
    _product_coupled_prepare_factor_coordinates!(workspace, base.epoch) ||
        return false
    workspace.last_reason = COUPLED_READY
    return true
end

"""Build one hybrid RHS for the already-scaled complementarity shift."""
@inline function _product_hsd_coupled_rhs!(
    state::ProductConeHSDState{T}, scalar_rhs::T,
) where {T}
    base = state.base
    workspace = state.coupled
    rhs = workspace.rhs
    nr = base.nr
    nsdim = workspace.nonsymmetric_dimension
    dual_row0 = nsdim
    gap_row = nsdim + nr + 1
    scalar_row = gap_row + 1

    @inbounds for row in 1:base.m
        state.g_input[row] = state.h[row] + base.rP[row]
    end
    _product_hsd_apply_symmetric_G!(
        state.runtime, state.g_output, state.g_input,
    )
    @inbounds for local_row in 1:nsdim
        row = workspace.offsets[div(local_row - 1, 3) + 1] +
              mod(local_row - 1, 3)
        rhs[local_row] = state.h[row] + base.rP[row]
    end
    @inbounds for j in 1:nr
        pj = zero(T)
        for pointer in nzrange(base.Ar, j)
            row = base.Ar.rowval[pointer]
            pj += base.Ar.nzval[pointer] * state.g_output[row]
        end
        state.ns_at_g_rhs[j] = pj
        rhs[dual_row0 + j] = -base.rDr[j] - pj
    end
    zeta = zero(T)
    @inbounds for row in 1:base.m
        zeta += base.b[row] * state.g_output[row]
    end
    rhs[gap_row] = -base.rG + zeta
    rhs[scalar_row] = scalar_rhs
    if !_product_hsd_vector_finite(rhs)
        workspace.last_reason = COUPLED_ASSEMBLY_NONFINITE
        return false
    end
    if workspace.transform_valid
        _product_coupled_transform_rhs!(workspace) || begin
            workspace.last_reason = COUPLED_TRANSFORM_FAILED
            return false
        end
    end
    return true
end

@inline function _product_hsd_coupled_recover!(
    state::ProductConeHSDState{T}, solution::Vector{T},
) where {T}
    base = state.base
    workspace = state.coupled
    nr = base.nr
    nsdim = workspace.nonsymmetric_dimension
    dtau_index = nr + nsdim + 1
    dkappa_index = dtau_index + 1
    _product_coupled_recover_physical!(workspace, solution) || begin
        workspace.last_reason = COUPLED_RECOVERY_FAILED
        return false
    end
    _product_coupled_original_solution_certificate!(
        workspace, workspace.physical_solution,
    ) || begin
        workspace.last_reason = COUPLED_RECOVERY_FAILED
        return false
    end
    physical = workspace.physical_solution
    @inbounds for j in 1:nr
        base.dxr[j] = physical[j]
    end
    base.dtau = physical[dtau_index]
    base.dkappa = physical[dkappa_index]
    _hsd_scatter_dx!(base)

    fill!(base.ax, zero(T))
    @inbounds for j in 1:base.n
        value = base.dx[j]
        iszero(value) && continue
        for pointer in nzrange(base.A, j)
            base.ax[base.A.rowval[pointer]] +=
                base.A.nzval[pointer] * value
        end
    end
    @inbounds for row in 1:base.m
        state.g_input[row] = base.ax[row] + state.h[row] + base.rP[row] -
                             base.b[row] * base.dtau
    end
    _product_hsd_apply_symmetric_G!(
        state.runtime, base.dy, state.g_input,
    )
    @inbounds for local_row in 1:nsdim
        row = workspace.offsets[div(local_row - 1, 3) + 1] +
              mod(local_row - 1, 3)
        base.dy[row] = physical[nr + local_row]
    end
    apply_Theta!(state.runtime, base.e, base.dy)
    roundtrip_certified, psd_budget_inconclusive =
        _product_hsd_roundtrip_backward_status(state)
    (roundtrip_certified || psd_budget_inconclusive) || return false
    @inbounds for row in 1:base.m
        base.ds[row] = -base.ax[row] - base.rP[row] +
                       base.b[row] * base.dtau
    end
    _hsd_direction_finite(base) || return false
    return roundtrip_certified ||
           _product_hsd_psd_cone_newton_residual_ok(state)
end

"""Build the reduced dual RHS for one already-scaled shift `h`."""
@inline function _product_hsd_rhs!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    @inbounds for k in 1:base.m
        state.g_input[k] = state.h[k] + base.rP[k]
    end
    _product_hsd_apply_symmetric_G!(
        state.runtime, state.g_output, state.g_input,
    )
    A = base.Ar
    @inbounds for j in 1:base.nr
        acc = zero(T)
        for ptr in nzrange(A, j)
            acc += A.nzval[ptr] * state.g_output[A.rowval[ptr]]
        end
        base.rhs[j] = -base.rDr[j] - acc
    end
    bsum = zero(T)
    @inbounds for k in 1:base.m
        bsum += base.b[k] * state.g_output[k]
    end
    if !isempty(state.runtime.exp) || !isempty(state.runtime.power)
        ns_result = _product_hsd_add_nonsymmetric_schur!(
            state, state.g_input,
        )
        ns_result.status === NS_SCHUR3_ASSEMBLED || return T(NaN)
        @inbounds for j in 1:base.nr
            base.rhs[j] -= state.ns_at_g_rhs[j]
        end
        bsum += ns_result.b_g_rhs
    end
    return bsum
end

"""
Recover the full Newton direction from `(dx,dtau)`.

Let `q=A*dx+h+rP-b*dtau`.  The eliminated equations give `dy=G*q` and the
algebraically equivalent primal recovery
`ds=h-q=-A*dx-rP+b*dtau`.  The latter avoids feeding the round-trip error of
`Theta*(G*q)` back into the primal Newton equation when a cone map is strongly
conditioned.  Every arithmetic type uses the stable formula only after a
condition-aware backward gate validates the composed `Theta*G` map.
"""
@inline function _product_hsd_recover!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    fill!(base.ax, zero(T))
    @inbounds for j in 1:base.n
        dxj = base.dx[j]
        iszero(dxj) && continue
        for ptr in nzrange(base.A, j)
            base.ax[base.A.rowval[ptr]] += base.A.nzval[ptr] * dxj
        end
    end
    @inbounds for k in 1:base.m
        state.g_input[k] = base.ax[k] + state.h[k] + base.rP[k] -
                           base.b[k] * base.dtau
    end
    apply_G!(state.runtime, base.dy, state.g_input)
    apply_Theta!(state.runtime, base.e, base.dy)

    roundtrip_certified, psd_budget_inconclusive =
        _product_hsd_roundtrip_backward_status(state)
    (roundtrip_certified || psd_budget_inconclusive) || return false
    @inbounds for k in 1:base.m
        base.ds[k] = -base.ax[k] - base.rP[k] + base.b[k] * base.dtau
    end
    cd = zero(T)
    bd = zero(T)
    @inbounds for j in 1:base.n
        cd += base.c[j] * base.dx[j]
    end
    @inbounds for k in 1:base.m
        bd += base.b[k] * base.dy[k]
    end
    base.dkappa = -base.rG + cd + bd
    isfinite(base.dkappa) || return false
    # A highly conditioned PSD scaling may make the conservative a-priori
    # round-trip condition cap inconclusive even when the actual map is
    # backward stable. Only that typed finite PSD outcome may continue, and
    # it must pass the independently recomputed componentwise PSD equation.
    # The caller still applies the authoritative full five-equation gate.
    return roundtrip_certified ||
           _product_hsd_psd_cone_newton_residual_ok(state)
end

@inline function _product_hsd_newton_close(
    residual::T, work::T,
) where {T}
    isfinite(residual) && isfinite(work) && work >= zero(T) || return false
    iszero(work) && return iszero(residual)
    return abs(residual) <= T(512) * sqrt(eps(T)) * work
end

@inline function _product_hsd_cone_newton_close(
    residual::T, work::T,
) where {T}
    isfinite(residual) && isfinite(work) && work >= zero(T) || return false
    iszero(work) && return iszero(residual)
    return abs(residual) <= T(512) * sqrt(eps(T)) * work
end

@inline function _product_hsd_conditioned_cone_newton_close(
    residual::T, work::T, budget::T,
) where {T}
    isfinite(residual) && isfinite(work) && work >= zero(T) &&
        isfinite(budget) && zero(T) <= budget < one(T) / T(100) ||
        return false
    iszero(work) && return iszero(residual)
    allowance = budget * work
    return isfinite(allowance) && abs(residual) <= allowance
end

@inline function _product_hsd_normalized_error(
    residual::T, work::T,
) where {T}
    isfinite(residual) && isfinite(work) && work >= zero(T) ||
        return T(Inf)
    iszero(work) && return iszero(residual) ? zero(T) : T(Inf)
    value = abs(residual) / work
    return isfinite(value) ? value : T(Inf)
end

"""Check `ds + Theta*dy = h` with the arithmetic work of `Theta*dy`.

The output coordinate `Theta*dy` may be cancellation-small near a curved
boundary.  It is therefore not a valid backward-error denominator.  Each
block below supplies either the exact absolute work of its map evaluation or
a conservative congruence bound; the zero-work case remains exact.
"""
@inline function _product_hsd_cone_newton_stats(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    runtime = state.runtime
    componentwise = true
    conditioned_family_failed = false
    group_residual = zero(T)
    group_work = zero(T)

    @inbounds for block in runtime.orthant
        for i in 1:block.dim
            k = block.offset + i - 1
            map_work = abs(block.state.theta[i]) * abs(base.dy[k])
            residual = base.ds[k] + base.e[k] - state.h[k]
            work = abs(base.ds[k]) + map_work + abs(state.h[k])
            componentwise &= _product_hsd_cone_newton_close(residual, work)
            group_residual = max(group_residual, abs(residual))
            group_work = max(group_work, work)
        end
    end

    @inbounds for block in runtime.soc
        n = block.dim
        w = block.state.w
        state.soc_bounds_certified ||
            return false, T(Inf), zero(T), true
        SymmetricCones._soc_q_condition_reliable(w, n) ||
            return false, T(Inf), zero(T), true
        condition_budget = _product_hsd_soc_condition_budget(w, n)
        isfinite(condition_budget) &&
            condition_budget < one(T) / T(100) ||
            return false, T(Inf), zero(T), true
        dot_work = zero(T)
        tail2 = zero(T)
        for i in 1:n
            zi = base.dy[block.offset + i - 1]
            dot_work += abs(w[i] * zi)
            i > 1 && (tail2 += w[i] * w[i])
        end
        radius = sqrt(tail2)
        determinant = (w[1] - radius) * (w[1] + radius)
        isfinite(dot_work) && isfinite(determinant) ||
            return false, T(Inf), zero(T), true
        two = one(T) + one(T)
        gamma = _product_bordered_gamma(T, 4n + 8)
        isfinite(gamma) || return false, T(Inf), zero(T), true
        for i in 1:n
            k = block.offset + i - 1
            map_work = two * abs(w[i]) * dot_work +
                       abs(determinant) * abs(base.dy[k])
            residual = base.ds[k] + base.e[k] - state.h[k]
            work = abs(base.ds[k]) + map_work + abs(state.h[k])
            map_bound = state.soc_roundtrip_bound[k]
            map_bound == state.certified_soc_roundtrip_bound[k] ||
                return false, T(Inf), zero(T), true
            recomputation = gamma * work
            allowance = map_bound + recomputation
            isfinite(map_bound) && map_bound >= zero(T) &&
                isfinite(recomputation) && isfinite(allowance) ||
                return false, T(Inf), zero(T), true
            local_ok = _product_bordered_zero_safe_close(
                residual, allowance,
            )
            componentwise &= local_ok
            conditioned_family_failed |= !local_ok
            group_residual = max(group_residual, abs(residual))
            group_work = max(group_work, work)
        end
    end

    @inbounds for block in runtime.psd
        n = block.dim
        pnorm = zero(T)
        for i in 1:n
            rowsum = zero(T)
            for j in 1:n
                rowsum += abs(block.state.P[i, j])
            end
            pnorm = max(pnorm, rowsum)
        end
        # The svec off diagonal represents two matrix entries of magnitude
        # `abs(x)/sqrt(2)`.  `matrix_work` therefore bounds the entrywise
        # absolute sum of the unpacked symmetric direction.
        matrix_work = zero(T)
        index = 1
        root2 = block.state.sqrt2
        for j in 1:n, i in j:n
            value = abs(base.dy[block.offset + index - 1])
            matrix_work += i == j ? value : root2 * value
            index += 1
        end
        entry_bound = pnorm * pnorm * matrix_work
        isfinite(entry_bound) ||
            return false, T(Inf), zero(T), conditioned_family_failed
        index = 1
        for j in 1:n, i in j:n
            k = block.offset + index - 1
            map_work = i == j ? entry_bound : root2 * entry_bound
            residual = base.ds[k] + base.e[k] - state.h[k]
            work = abs(base.ds[k]) + map_work + abs(state.h[k])
            componentwise &= _product_hsd_cone_newton_close(residual, work)
            group_residual = max(group_residual, abs(residual))
            group_work = max(group_work, work)
            index += 1
        end
    end

    @inbounds for block in runtime.exp
        theta = block.scaling.theta
        for i in 1:3
            map_work = zero(T)
            for j in 1:3
                map_work += abs(theta[i, j]) *
                            abs(base.dy[block.offset + j - 1])
            end
            k = block.offset + i - 1
            residual = base.ds[k] + base.e[k] - state.h[k]
            work = abs(base.ds[k]) + map_work + abs(state.h[k])
            componentwise &= _product_hsd_cone_newton_close(residual, work)
            group_residual = max(group_residual, abs(residual))
            group_work = max(group_work, work)
        end
    end
    @inbounds for block in runtime.power
        theta = block.scaling.theta
        for i in 1:3
            map_work = zero(T)
            for j in 1:3
                map_work += abs(theta[i, j]) *
                            abs(base.dy[block.offset + j - 1])
            end
            k = block.offset + i - 1
            residual = base.ds[k] + base.e[k] - state.h[k]
            work = abs(base.ds[k]) + map_work + abs(state.h[k])
            componentwise &= _product_hsd_cone_newton_close(residual, work)
            group_residual = max(group_residual, abs(residual))
            group_work = max(group_work, work)
        end
    end
    return componentwise, group_residual, group_work,
           conditioned_family_failed
end

"""Componentwise actual-work gate for a finite PSD-budget inconclusive map."""
@inline function _product_hsd_psd_cone_newton_residual_ok(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    saw_psd = false
    @inbounds for block in state.runtime.psd
        saw_psd = true
        n = block.dim
        pnorm = zero(T)
        for i in 1:n
            rowsum = zero(T)
            for j in 1:n
                rowsum += abs(block.state.P[i, j])
            end
            pnorm = max(pnorm, rowsum)
        end
        matrix_work = zero(T)
        index = 1
        root2 = block.state.sqrt2
        for j in 1:n, i in j:n
            value = abs(base.dy[block.offset + index - 1])
            matrix_work += i == j ? value : root2 * value
            index += 1
        end
        entry_bound = pnorm * pnorm * matrix_work
        isfinite(entry_bound) || return false
        index = 1
        for j in 1:n, i in j:n
            k = block.offset + index - 1
            map_work = i == j ? entry_bound : root2 * entry_bound
            residual = base.ds[k] + base.e[k] - state.h[k]
            work = abs(base.ds[k]) + map_work + abs(state.h[k])
            _product_hsd_cone_newton_close(residual, work) || return false
            index += 1
        end
    end
    return saw_psd
end

@inline function _product_hsd_cone_newton_residual_ok(
    state::ProductConeHSDState{T},
) where {T}
    componentwise, group_residual, group_work, conditioned_family_failed =
        _product_hsd_cone_newton_stats(state)
    conditioned_family_failed && return false
    return componentwise ||
           _product_hsd_cone_newton_close(group_residual, group_work)
end

@inline function _product_hsd_primal_newton_stats(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    componentwise = true
    group_residual = zero(T)
    rhs_norm = zero(T)
    direction_norm = abs(base.dtau)

    # `g_input` is dead after recovery's authoritative solve and is reused as
    # a preallocated row-sum scratch for ||[A I -b]||_infinity.
    @inbounds for k in 1:base.m
        state.g_input[k] = one(T) + abs(base.b[k])
        rhs_norm = max(rhs_norm, abs(base.rP[k]))
        direction_norm = max(
            direction_norm, abs(base.ds[k]),
        )
    end
    @inbounds for j in 1:base.n
        direction_norm = max(direction_norm, abs(base.dx[j]))
        for ptr in nzrange(base.A, j)
            state.g_input[base.A.rowval[ptr]] += abs(base.A.nzval[ptr])
        end
    end

    operator_norm = zero(T)
    @inbounds for k in 1:base.m
        bdt = base.b[k] * base.dtau
        residual = base.ax[k] + base.ds[k] - bdt + base.rP[k]
        local_work = abs(base.ax[k]) + abs(base.ds[k]) + abs(bdt) +
                     abs(base.rP[k])
        componentwise &= _product_hsd_newton_close(residual, local_work)
        group_residual = max(group_residual, abs(residual))
        operator_norm = max(operator_norm, state.g_input[k])
    end
    group_work = operator_norm * direction_norm + rhs_norm
    return componentwise, group_residual, group_work
end

@inline function _product_hsd_dual_newton_stats(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    componentwise = true
    group_residual = zero(T)
    rhs_norm = zero(T)
    direction_norm = abs(base.dtau)
    @inbounds for k in 1:base.m
        direction_norm = max(direction_norm, abs(base.dy[k]))
    end
    operator_norm = zero(T)
    @inbounds for j in 1:base.n
        cdt = base.c[j] * base.dtau
        residual = muladd(base.c[j], base.dtau, base.rD[j])
        local_work = abs(base.rD[j]) + abs(cdt)
        row_norm = abs(base.c[j])
        for ptr in nzrange(base.A, j)
            term = base.A.nzval[ptr] * base.dy[base.A.rowval[ptr]]
            residual = muladd(
                base.A.nzval[ptr], base.dy[base.A.rowval[ptr]], residual,
            )
            local_work += abs(term)
            row_norm += abs(base.A.nzval[ptr])
        end
        componentwise &= _product_hsd_newton_close(residual, local_work)
        group_residual = max(group_residual, abs(residual))
        rhs_norm = max(rhs_norm, abs(base.rD[j]))
        operator_norm = max(operator_norm, row_norm)
    end
    group_work = operator_norm * direction_norm + rhs_norm
    return componentwise, group_residual, group_work
end

@inline function _product_hsd_symmetric_dual_residual_ok(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    workspace = state.symmetric_bordered
    # This is the terminal full-equation authority, not the ordinary raw
    # solve gate.  A direct recomputation is admissible only after both of the
    # bounded correction solves for this factor epoch have been completed.
    # Keeping the guard here (rather than relying only on the caller) makes
    # accidental early use fail closed and prevents an observed raw residual
    # from being promoted to a certificate.
    workspace.refinements == 2 || return false
    _product_bordered_physical_snapshot_ok(workspace) || return false
    state.soc_bounds_certified || return false
    @inbounds for k in 1:base.m
        bound = state.soc_g_error_bound[k]
        bound == state.certified_soc_g_error_bound[k] || return false
        isfinite(bound) && bound >= zero(T) || return false
    end

    # The accepted reduced bordered residual lives in row-space coordinates.
    # Propagate its immutable physical bound through the actual RRQR basis.
    # Separately account for the condition-aware SOC G-map error and this
    # direct full-equation recomputation's own arithmetic work.
    @inbounds for block in state.runtime.soc
        SymmetricCones._soc_q_condition_reliable(
            block.state.w, block.dim,
        ) || return false
        budget = _product_hsd_soc_condition_budget(
            block.state.w, block.dim,
        )
        isfinite(budget) && budget < one(T) / T(100) || return false
    end

    @inbounds for i in 1:base.n
        residual = muladd(base.c[i], base.dtau, base.rD[i])
        local_work = abs(base.rD[i]) + abs(base.c[i] * base.dtau)
        conditioned = zero(T)
        operations = 2
        for pointer in nzrange(base.A, i)
            row = base.A.rowval[pointer]
            coefficient = base.A.nzval[pointer]
            term = coefficient * base.dy[row]
            residual = muladd(coefficient, base.dy[row], residual)
            local_work += abs(term)
            conditioned += abs(coefficient) *
                           state.soc_g_error_bound[row]
            operations += 2
        end
        isfinite(residual) && isfinite(local_work) || return false

        propagated = zero(T)
        for j in 1:base.nr
            term = abs(base.rank_basis[i, j]) *
                   workspace.certified_physical_bound[j]
            isfinite(term) || return false
            propagated += term
        end
        gamma = _product_bordered_gamma(T, operations + 2)
        recomputation = gamma * local_work
        allowance = propagated + conditioned + recomputation
        isfinite(propagated) && isfinite(conditioned) &&
            isfinite(recomputation) && isfinite(allowance) || return false
        _product_bordered_zero_safe_close(residual, allowance) ||
            return false
    end
    return true
end

@inline function _product_hsd_symmetric_scalar_residual_ok(
    state::ProductConeHSDState{T}, scalar_rhs::T,
) where {T}
    base = state.base
    workspace = state.symmetric_bordered
    # The scalar equation shares the same terminal authority as the direct
    # dual recomputation above: no early acceptance before the prescribed two
    # bounded corrections on this single numerical factor.
    workspace.refinements == 2 || return false
    _product_bordered_physical_snapshot_ok(workspace) || return false
    kdt = base.kappa * base.dtau
    tdk = base.tau * base.dkappa
    residual = muladd(
        base.kappa, base.dtau,
        muladd(base.tau, base.dkappa, -scalar_rhs),
    )
    work = abs(kdt) + abs(tdk) + abs(scalar_rhs)
    isfinite(residual) && isfinite(work) || return false

    soc_budget = zero(T)
    @inbounds for block in state.runtime.soc
        SymmetricCones._soc_q_condition_reliable(
            block.state.w, block.dim,
        ) || return false
        budget = _product_hsd_soc_condition_budget(
            block.state.w, block.dim,
        )
        isfinite(budget) && budget < one(T) / T(100) || return false
        soc_budget = max(soc_budget, budget)
    end
    metric_work = zero(T)
    @inbounds for k in 1:base.m
        term = base.b[k] * base.dy[k]
        isfinite(term) || return false
        metric_work += abs(term)
    end
    propagated = workspace.certified_physical_bound[end]
    gamma = _product_bordered_gamma(T, 8)
    recomputation = gamma * work
    conditioned = soc_budget * abs(base.tau) * metric_work
    allowance = propagated + conditioned + recomputation
    isfinite(propagated) && isfinite(conditioned) &&
        isfinite(recomputation) &&
        isfinite(allowance) || return false
    return _product_bordered_zero_safe_close(residual, allowance)
end

"""Fail-closed check of all five frozen full-Newton equation groups."""
@inline function _product_hsd_newton_residual_ok(
    state::ProductConeHSDState{T}, scalar_rhs::T,
    conditioned_authority::Bool=false,
) where {T}
    base = state.base

    # A*dx + ds - b*dτ = -rP.  Componentwise arithmetic-work backward
    # stability is preferred; the standard scale-free normwise alternative
    # handles a structurally near-null row without an absolute floor.
    primal_componentwise, primal_residual, primal_work =
        _product_hsd_primal_newton_stats(state)
    (primal_componentwise ||
     _product_hsd_newton_close(primal_residual, primal_work)) || return false

    # A'*dy + c*dτ = -rD.
    if _product_hsd_has_nonsymmetric(state) || !conditioned_authority
        dual_componentwise, dual_residual, dual_work =
            _product_hsd_dual_newton_stats(state)
        (dual_componentwise ||
         _product_hsd_newton_close(dual_residual, dual_work)) || return false
    else
        _product_hsd_symmetric_dual_residual_ok(state) || return false
    end

    # -c'*dx - b'*dy + dκ = -rG.
    gap_residual = base.rG + base.dkappa
    gap_work = abs(base.rG) + abs(base.dkappa)
    @inbounds for j in 1:base.n
        term = base.c[j] * base.dx[j]
        gap_residual = muladd(-base.c[j], base.dx[j], gap_residual)
        gap_work += abs(term)
    end
    @inbounds for k in 1:base.m
        term = base.b[k] * base.dy[k]
        gap_residual = muladd(-base.b[k], base.dy[k], gap_residual)
        gap_work += abs(term)
    end
    _product_hsd_newton_close(gap_residual, gap_work) || return false

    # ds + Theta*dy = h. Recovery has left Theta*dy in `base.e`.
    _product_hsd_cone_newton_residual_ok(state) || return false

    # κ*dτ + τ*dκ = scalar_rhs.
    kdt = base.kappa * base.dtau
    tdk = base.tau * base.dkappa
    scalar_residual = muladd(
        base.kappa, base.dtau,
        muladd(base.tau, base.dkappa, -scalar_rhs),
    )
    scalar_work = abs(kdt) + abs(tdk) + abs(scalar_rhs)
    if _product_hsd_has_nonsymmetric(state) || !conditioned_authority
        return _product_hsd_newton_close(scalar_residual, scalar_work)
    end
    return _product_hsd_symmetric_scalar_residual_ok(state, scalar_rhs)
end

@inline function _product_bordered_recompute_staged!(
    workspace::SymmetricBorderedWorkspace{T},
    solution::AbstractVector{T}, rhs::AbstractVector{T},
) where {T}
    n = workspace.dimension
    length(solution) == n && length(rhs) == n || return false
    F = workspace.driver.route.factors
    P = workspace.permutation
    py = workspace.permuted_rhs
    y = workspace.staged_y
    f = workspace.forward_residual
    u = workspace.backward_residual
    @inbounds for i in 1:n
        value = rhs[P[i]]
        isfinite(value) || return false
        py[i] = value
    end
    @inbounds for i in 1:n
        value = py[i]
        for j in 1:(i - 1)
            # LAPACK getrs uses fused triangular updates on the Float64 path;
            # replay the same rounding model so this staged certificate does
            # not mistake an FMA/non-FMA cancellation difference for solve
            # error. The generic LU kernel deliberately uses separate
            # multiply/subtract operations, so its replay must do the same.
            if T <: _LAPACK_LU
                value = muladd(-F[i, j], y[j], value)
            else
                value -= F[i, j] * y[j]
            end
        end
        isfinite(value) || return false
        y[i] = value
    end
    @inbounds for i in 1:n
        ly = y[i]
        for j in 1:(i - 1)
            if T <: _LAPACK_LU
                ly = muladd(F[i, j], y[j], ly)
            else
                ly += F[i, j] * y[j]
            end
        end
        f[i] = ly - py[i]
        isfinite(f[i]) || return false
        uz = zero(T)
        for j in i:n
            if T <: _LAPACK_LU
                uz = muladd(F[i, j], solution[j], uz)
            else
                uz += F[i, j] * solution[j]
            end
        end
        u[i] = uz - y[i]
        isfinite(u[i]) || return false
    end
    return true
end

@inline function _product_bordered_triangular_solution_ok!(
    workspace::SymmetricBorderedWorkspace{T},
    solution::AbstractVector{T},
    rhs::AbstractVector{T},
    operations::Int,
) where {T}
    _product_bordered_recompute_staged!(workspace, solution, rhs) ||
        return false
    n = workspace.dimension
    F = workspace.driver.route.factors
    py = workspace.permuted_rhs
    y = workspace.staged_y
    f = workspace.forward_residual
    u = workspace.backward_residual
    gamma = _product_bordered_gamma(T, operations)
    isfinite(gamma) || return false
    @inbounds for i in 1:n
        forward_work = abs(y[i]) + abs(py[i])
        for j in 1:(i - 1)
            forward_work += abs(F[i, j] * y[j])
        end
        _product_bordered_zero_safe_close(
            f[i], gamma * forward_work,
        ) || return false

        backward_work = abs(y[i])
        for j in i:n
            backward_work += abs(F[i, j] * solution[j])
        end
        _product_bordered_zero_safe_close(
            u[i], gamma * backward_work,
        ) || return false
    end
    return true
end

@inline function _product_bordered_staged_solve!(
    workspace::SymmetricBorderedWorkspace{T},
) where {T}
    n = workspace.dimension
    workspace.original_solution_certified = false
    workspace.factor_certified || return false
    workspace.factor_epoch == workspace.assembly_epoch || return false
    _product_bordered_transform_matrix_ok(workspace) || return false
    _product_bordered_transform_rhs_ok(workspace) || return false
    _product_bordered_factor_certificate!(workspace) || return false
    z = workspace.solution
    factors_before = kkt_factor_count(workspace.driver)
    try
        kkt_solve!(workspace.driver, z, workspace.factor_rhs)
    catch
        return false
    end
    kkt_factor_count(workspace.driver) == factors_before || return false
    @inbounds for value in z
        isfinite(value) || return false
    end
    _product_bordered_triangular_solution_ok!(
        workspace, z, workspace.factor_rhs, 8n,
    ) || return false
    copyto!(workspace.certified_solution, z)
    workspace.accumulated_candidate = false
    workspace.candidate_epoch = workspace.factor_epoch
    return true
end

@inline function _product_bordered_factor_solution_ok!(
    workspace::SymmetricBorderedWorkspace{T},
    solution::AbstractVector{T}=workspace.solution,
) where {T}
    n = workspace.dimension
    workspace.original_solution_certified = false
    workspace.factor_certified || return false
    workspace.factor_epoch == workspace.assembly_epoch || return false
    _product_bordered_transform_matrix_ok(workspace) || return false
    _product_bordered_transform_rhs_ok(workspace) || return false
    _product_bordered_factor_certificate!(workspace) || return false
    if workspace.accumulated_candidate
        workspace.candidate_epoch == workspace.factor_epoch || return false
        @inbounds for i in 1:n
            solution[i] == workspace.certified_solution[i] || return false
            workspace.certified_solution[i] ==
                workspace.previous_solution[i] +
                workspace.correction_solution[i] || return false
        end
        _product_bordered_recompute_staged!(
            workspace, solution, workspace.factor_rhs,
        ) || return false
    else
        # Every raw predictor/corrector/correction solve must retain its own
        # strict triangular actual-work certificate. Rechecking here also
        # catches any finite corruption after the cache solve returned.
        @inbounds for i in 1:n
            solution[i] == workspace.certified_solution[i] || return false
        end
        _product_bordered_triangular_solution_ok!(
            workspace, solution, workspace.factor_rhs, 8n,
        ) || return false
    end
    F = workspace.driver.route.factors
    E = workspace.factor_error
    P = workspace.permutation
    f = workspace.forward_residual
    u = workspace.backward_residual
    gamma = _product_bordered_gamma(T, 8n)
    isfinite(gamma) || return false

    @inbounds for i in 1:n
        upper = zero(T)
        for j in i:n
            upper += abs(F[i, j]) * abs(solution[j])
        end
        workspace.upper_work[i] = upper
    end
    @inbounds for i in 1:n
        lower = workspace.upper_work[i]
        for j in 1:(i - 1)
            lower += abs(F[i, j]) * workspace.upper_work[j]
        end
        workspace.lower_work[i] = lower
    end

    @inbounds for i in 1:n
        ez = zero(T)
        ez_work = zero(T)
        for j in 1:n
            term = E[i, j] * solution[j]
            ez += term
            ez_work += abs(term)
        end
        lu = f[i] + u[i]
        lu_work = abs(f[i]) + abs(u[i])
        for j in 1:(i - 1)
            term = F[i, j] * u[j]
            lu += term
            lu_work += abs(term)
        end
        identity = ez + lu
        identity_work = ez_work + lu_work
        isfinite(identity) && isfinite(identity_work) || return false
        workspace.identity_rhs[i] = identity
        original_row = P[i]
        acc = -workspace.factor_rhs[original_row]
        work = abs(workspace.factor_rhs[original_row])
        for j in 1:n
            term = workspace.factor_matrix[original_row, j] * solution[j]
            acc += term
            work += abs(term)
        end
        identity_error = acc - identity
        allowance = gamma * (
            work + identity_work + workspace.lower_work[i]
        )
        _product_bordered_zero_safe_close(identity_error, allowance) ||
            return false
        residual_allowance = identity_work +
            gamma * (work + workspace.lower_work[i])
        workspace.residual[original_row] = acc
        workspace.bound[original_row] = residual_allowance
        workspace.certified_factor_bound[original_row] = residual_allowance
        _product_bordered_zero_safe_close(acc, residual_allowance) ||
            return false
    end
    return true
end

@inline function _product_bordered_original_solution_ok!(
    workspace::SymmetricBorderedWorkspace{T},
    solution::AbstractVector{T}=workspace.solution,
) where {T}
    n = workspace.dimension
    workspace.original_solution_certified = false
    gamma = _product_bordered_gamma(T, 4n)
    isfinite(gamma) || return false
    @inbounds for i in 1:n
        solution[i] == workspace.certified_solution[i] || return false
        if workspace.accumulated_candidate
            workspace.certified_solution[i] ==
                workspace.previous_solution[i] +
                workspace.correction_solution[i] || return false
        end
    end
    @inbounds for i in 1:n
        balanced_factor_bound = workspace.bound[i]
        balanced_factor_bound == workspace.certified_factor_bound[i] ||
            return false
        isfinite(balanced_factor_bound) &&
            balanced_factor_bound >= zero(T) || return false
        residual = -workspace.rhs[i]
        work = abs(workspace.rhs[i])
        scaled_residual = -workspace.factor_rhs[i]
        for j in 1:n
            term = workspace.matrix[i, j] * solution[j]
            residual += term
            work += abs(term)
            scaled_residual += workspace.factor_matrix[i, j] * solution[j]
        end
        isfinite(residual) && isfinite(work) && isfinite(scaled_residual) ||
            return false
        recovered_residual = scaled_residual / workspace.row_scale[i]
        isfinite(recovered_residual) || return false
        if iszero(scaled_residual)
            iszero(recovered_residual) || return false
        else
            !iszero(recovered_residual) &&
                recovered_residual * workspace.row_scale[i] ==
                    scaled_residual || return false
        end
        reassociation = residual - recovered_residual
        reassociation_allowance = gamma * (
            work + abs(recovered_residual)
        )
        _product_bordered_zero_safe_close(
            reassociation, reassociation_allowance,
        ) || return false
        # Propagate the already-certified P*Mhat=L*U reconstruction, raw
        # triangular residuals, and factor error back through the exact
        # binary row scale, then add only this physical recomputation's
        # reassociation work. No output-relative or absolute floor appears.
        factor_allowance = balanced_factor_bound / workspace.row_scale[i]
        isfinite(factor_allowance) || return false
        if iszero(balanced_factor_bound)
            iszero(factor_allowance) || return false
        else
            !iszero(factor_allowance) &&
                factor_allowance * workspace.row_scale[i] ==
                    balanced_factor_bound || return false
        end
        allowance = factor_allowance + reassociation_allowance
        workspace.residual[i] = residual
        workspace.bound[i] = allowance
        workspace.certified_physical_bound[i] = allowance
        _product_bordered_zero_safe_close(residual, allowance) || return false
    end
    workspace.original_solution_certified = true
    return true
end

@inline function _product_hsd_prepare_bordered_rhs!(
    state::ProductConeHSDState{T}, scalar_rhs::T,
) where {T}
    base = state.base
    workspace = state.symmetric_bordered
    workspace.original_solution_certified = false
    workspace.factor_certified &&
        workspace.factor_epoch == base.epoch &&
        workspace.assembly_epoch == base.epoch || begin
            workspace.last_reason = SYMMETRIC_BORDERED_EPOCH_MISMATCH
            return false
        end
    _product_bordered_transform_matrix_ok(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_TRANSFORM_FAILED
        return false
    end
    bsum = _product_hsd_rhs!(state)
    rho = scalar_rhs + base.tau * base.rG - base.tau * bsum
    @inbounds for i in 1:base.nr
        value = base.rhs[i]
        isfinite(value) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_RHS_FAILED
            return false
        end
        workspace.rhs[i] = value
    end
    isfinite(rho) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_RHS_FAILED
        return false
    end
    workspace.rhs[end] = rho
    @inbounds for i in 1:workspace.dimension
        transformed = workspace.row_scale[i] * workspace.rhs[i]
        isfinite(transformed) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_RHS_FAILED
            return false
        end
        workspace.factor_rhs[i] = transformed
    end
    _product_bordered_transform_rhs_ok(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_RHS_FAILED
        return false
    end
    return true
end

@inline function _product_hsd_bordered_candidate_ok!(
    state::ProductConeHSDState{T}, scalar_rhs::T,
) where {T}
    base = state.base
    workspace = state.symmetric_bordered
    _product_hsd_prepare_bordered_rhs!(state, scalar_rhs) || return false
    @inbounds for i in 1:base.nr
        value = base.dxr[i]
        isfinite(value) || return false
        workspace.solution[i] = value
    end
    isfinite(base.dtau) || return false
    workspace.solution[end] = base.dtau
    _product_bordered_factor_solution_ok!(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_SOLVE_CERT_FAILED
        return false
    end
    _product_bordered_original_solution_ok!(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_ORIGINAL_CERT_FAILED
        return false
    end
    return true
end

@inline function _product_hsd_recover_dkappa!(base, scalar_rhs)
    # The scalar complementarity equation is well-conditioned because HSD
    # keeps tau strictly positive. Recover dkappa from it directly; computing
    # dkappa from the gap equation loses digits as the homogeneous border
    # becomes ill-conditioned near an SDP optimum. The independent gap
    # equation remains part of the strict five-equation direction certificate.
    base.dkappa = (scalar_rhs - base.kappa * base.dtau) / base.tau
    return isfinite(base.dkappa)
end

@inline function _product_hsd_solve_shift_raw!(
    state::ProductConeHSDState{T},
    scalar_rhs::T,
) where {T}
    base = state.base
    workspace = state.symmetric_bordered
    _product_hsd_prepare_bordered_rhs!(state, scalar_rhs) || return false
    _product_bordered_staged_solve!(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_SOLVE_FAILED
        return false
    end
    _product_bordered_factor_solution_ok!(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_SOLVE_CERT_FAILED
        return false
    end
    _product_bordered_original_solution_ok!(workspace) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_ORIGINAL_CERT_FAILED
        return false
    end
    @inbounds for i in 1:base.nr
        base.dxr[i] = workspace.solution[i]
    end
    base.dtau = workspace.solution[end]
    _hsd_scatter_dx!(base)
    _product_hsd_recover!(state) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_RECOVERY_FAILED
        return false
    end
    _product_hsd_recover_dkappa!(base, scalar_rhs) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_RECOVERY_FAILED
        return false
    end
    _hsd_direction_finite(base) || begin
        workspace.last_reason = SYMMETRIC_BORDERED_RECOVERY_FAILED
        return false
    end
    workspace.solves += 1
    return true
end

@inline function _product_hsd_max_normalized_newton_error(
    state::ProductConeHSDState{T}, scalar_rhs::T,
) where {T}
    base = state.base
    _, primal_residual, primal_work =
        _product_hsd_primal_newton_stats(state)
    _, dual_residual, dual_work = _product_hsd_dual_newton_stats(state)
    _, cone_residual, cone_work, _ =
        _product_hsd_cone_newton_stats(state)
    worst = max(
        _product_hsd_normalized_error(primal_residual, primal_work),
        _product_hsd_normalized_error(dual_residual, dual_work),
        _product_hsd_normalized_error(cone_residual, cone_work),
    )

    residual = base.rG + base.dkappa
    work = abs(base.rG) + abs(base.dkappa)
    @inbounds for j in 1:base.n
        term = base.c[j] * base.dx[j]
        residual = muladd(-base.c[j], base.dx[j], residual)
        work += abs(term)
    end
    @inbounds for k in 1:base.m
        term = base.b[k] * base.dy[k]
        residual = muladd(-base.b[k], base.dy[k], residual)
        work += abs(term)
    end
    worst = max(worst, _product_hsd_normalized_error(residual, work))
    residual = muladd(
        base.kappa, base.dtau,
        muladd(base.tau, base.dkappa, -scalar_rhs),
    )
    work = abs(base.kappa * base.dtau) + abs(base.tau * base.dkappa) +
           abs(scalar_rhs)
    return max(worst, _product_hsd_normalized_error(residual, work))
end

@inline function _product_hsd_restore_refinement_equations!(
    state::ProductConeHSDState{T}, original_rG::T,
) where {T}
    base = state.base
    copyto!(base.rP, base.rPt)
    copyto!(base.rD, base.rDt)
    copyto!(state.h, base.st)
    base.rG = original_rG
    @inbounds for j in 1:base.nr
        acc = zero(T)
        for i in 1:base.n
            acc += base.rank_basis[i, j] * base.rD[i]
        end
        base.rDr[j] = acc
    end
    return nothing
end

@inline function _product_hsd_rebuild_refined_direction!(
    state::ProductConeHSDState{T}, reduced_authoritative::Bool=false,
) where {T}
    base = state.base
    reduced_authoritative && _hsd_scatter_dx!(base)
    fill!(base.ax, zero(T))
    @inbounds for j in 1:base.n
        value = base.dx[j]
        iszero(value) && continue
        for ptr in nzrange(base.A, j)
            base.ax[base.A.rowval[ptr]] += base.A.nzval[ptr] * value
        end
    end
    if !reduced_authoritative
        @inbounds for j in 1:base.nr
            acc = zero(T)
            for i in 1:base.n
                acc += base.rank_basis[i, j] * base.dx[i]
            end
            base.dxr[j] = acc
        end
    end
    @inbounds for k in 1:base.m
        state.g_input[k] = base.ax[k] + state.h[k] + base.rP[k] -
                           base.b[k] * base.dtau
    end
    # Re-evaluate the frozen linear map on the accumulated correction RHS.
    # Summing two individually accurate `dy=G*q` vectors can lose every
    # significant digit when iterative refinement cancels them; applying G
    # once to the total q preserves the same Newton operator and gives the
    # subsequent round-trip/five-equation gates the actual refined direction.
    apply_G!(state.runtime, base.dy, state.g_input)
    apply_Theta!(state.runtime, base.e, base.dy)
    roundtrip_certified, psd_budget_inconclusive =
        _product_hsd_roundtrip_backward_status(state)
    (roundtrip_certified || psd_budget_inconclusive) || return false
    @inbounds for k in 1:base.m
        base.ds[k] = -base.ax[k] - base.rP[k] +
                     base.b[k] * base.dtau
    end
    cd = zero(T)
    bd = zero(T)
    @inbounds for j in 1:base.n
        cd += base.c[j] * base.dx[j]
    end
    @inbounds for k in 1:base.m
        bd += base.b[k] * base.dy[k]
    end
    base.dkappa = -base.rG + cd + bd
    _hsd_direction_finite(base) || return false
    return roundtrip_certified ||
           _product_hsd_psd_cone_newton_residual_ok(state)
end

@inline function _product_hsd_refine_shift!(
    state::ProductConeHSDState{T},
    scalar_rhs::T,
) where {T}
    base = state.base
    workspace = state.symmetric_bordered
    # Scope the bounded-correction counter to *this* solve.  One KKT epoch
    # runs two shifted solves (predictor and corrector) against the same
    # factor, so an epoch-cumulative counter reaches 4 whenever both need
    # refinement.  The terminal authorities below gate on `refinements == 2`
    # meaning "both bounded correction solves of the solve being certified
    # are done"; against a cumulative counter that gate silently stopped
    # admitting the conditioned authority from the second refined solve
    # onward, and a perfectly good direction was rejected as
    # SYMMETRIC_BORDERED_FIVE_EQUATION_FAILED.
    workspace.refinements = 0
    copyto!(base.rPt, base.rP)
    copyto!(base.rDt, base.rD)
    copyto!(base.st, state.h)
    original_rG = base.rG
    initial_error = _product_hsd_max_normalized_newton_error(
        state, scalar_rhs,
    )
    isfinite(initial_error) && initial_error > zero(T) || return false

    for refinement in 1:2
        copyto!(base.xt, base.dx)
        copyto!(base.yt, base.dy)
        previous_snapshot_ok = true
        @inbounds for i in 1:workspace.dimension
            if workspace.solution[i] != workspace.certified_solution[i]
                previous_snapshot_ok = false
                break
            end
        end
        previous_snapshot_ok || begin
            workspace.last_reason = SYMMETRIC_BORDERED_SOLVE_CERT_FAILED
            return false
        end
        copyto!(workspace.previous_solution, workspace.solution)
        total_dtau = base.dtau

        # Substitute the five current residuals into the same frozen Newton
        # operator.  The correction uses the already-factored global Schur;
        # no new KKT factorization is permitted.
        @inbounds for k in 1:base.m
            primal_residual = muladd(
                -base.b[k], base.dtau,
                base.ax[k] + base.ds[k] + base.rPt[k],
            )
            cone_residual = base.ds[k] + base.e[k] - base.st[k]
            base.rP[k] = primal_residual
            state.h[k] = -cone_residual
        end
        @inbounds for j in 1:base.n
            dual_residual = muladd(
                base.c[j], base.dtau, base.rDt[j],
            )
            for ptr in nzrange(base.A, j)
                dual_residual = muladd(
                    base.A.nzval[ptr],
                    base.dy[base.A.rowval[ptr]], dual_residual,
                )
            end
            base.rD[j] = dual_residual
        end
        @inbounds for j in 1:base.nr
            acc = zero(T)
            for i in 1:base.n
                acc = muladd(base.rank_basis[i, j], base.rD[i], acc)
            end
            base.rDr[j] = acc
        end
        gap_residual = original_rG + base.dkappa
        @inbounds for j in 1:base.n
            gap_residual = muladd(-base.c[j], base.dx[j], gap_residual)
        end
        @inbounds for k in 1:base.m
            gap_residual = muladd(-base.b[k], base.dy[k], gap_residual)
        end
        base.rG = gap_residual
        scalar_residual = muladd(
            base.kappa, base.dtau,
            muladd(base.tau, base.dkappa, -scalar_rhs),
        )

        correction_ok = _product_hsd_solve_shift_raw!(
            state, -scalar_residual,
        )
        if !correction_ok
            copyto!(base.dx, base.xt)
            copyto!(base.dy, base.yt)
            base.dtau = total_dtau
            _product_hsd_restore_refinement_equations!(state, original_rG)
            _product_hsd_rebuild_refined_direction!(state)
            return false
        end
        raw_snapshot_ok = true
        @inbounds for i in 1:workspace.dimension
            if workspace.solution[i] != workspace.certified_solution[i]
                raw_snapshot_ok = false
                break
            end
        end
        raw_snapshot_ok || begin
            copyto!(base.dx, base.xt)
            copyto!(base.dy, base.yt)
            base.dtau = total_dtau
            _product_hsd_restore_refinement_equations!(state, original_rG)
            _product_hsd_rebuild_refined_direction!(state)
            workspace.last_reason = SYMMETRIC_BORDERED_SOLVE_CERT_FAILED
            return false
        end
        copyto!(workspace.correction_solution, workspace.solution)
        accumulation_ok = true
        @inbounds for i in 1:workspace.dimension
            value = workspace.previous_solution[i] +
                    workspace.correction_solution[i]
            if !isfinite(value)
                accumulation_ok = false
                break
            end
            workspace.solution[i] = value
            workspace.certified_solution[i] = value
        end
        accumulation_ok || begin
            copyto!(base.dx, base.xt)
            copyto!(base.dy, base.yt)
            base.dtau = total_dtau
            _product_hsd_restore_refinement_equations!(state, original_rG)
            _product_hsd_rebuild_refined_direction!(state)
            workspace.last_reason = SYMMETRIC_BORDERED_SOLVE_FAILED
            return false
        end
        @inbounds for j in 1:base.nr
            base.dxr[j] = workspace.solution[j]
        end
        base.dtau = workspace.solution[end]
        workspace.accumulated_candidate = true
        workspace.candidate_epoch = workspace.factor_epoch
        workspace.accumulations += 1
        _product_hsd_restore_refinement_equations!(state, original_rG)
        _product_hsd_rebuild_refined_direction!(state, true) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_RECOVERY_FAILED
            return false
        end
        _product_hsd_recover_dkappa!(base, scalar_rhs) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_RECOVERY_FAILED
            return false
        end
        _hsd_direction_finite(base) || begin
            workspace.last_reason = SYMMETRIC_BORDERED_RECOVERY_FAILED
            return false
        end
        workspace.refinements += 1

        _product_hsd_bordered_candidate_ok!(state, scalar_rhs) || return false
        _product_hsd_newton_residual_ok(state, scalar_rhs) && return true
        if refinement == 2
            if _product_hsd_newton_residual_ok(
                state, scalar_rhs, true,
            )
                workspace.last_reason =
                    SYMMETRIC_BORDERED_CONDITIONED_FINAL_CERTIFIED
                return true
            end
        end
    end
    workspace.last_reason = SYMMETRIC_BORDERED_FIVE_EQUATION_FAILED
    return false
end

@inline function _product_hsd_solve_shift!(
    state::ProductConeHSDState{T},
    scalar_rhs::T,
) where {T}
    _product_hsd_solve_shift_raw!(
        state, scalar_rhs,
    ) || return false
    _product_hsd_newton_residual_ok(state, scalar_rhs) && return true
    return _product_hsd_refine_shift!(state, scalar_rhs)
end

@inline function _product_hsd_coupled_merit(
    state::ProductConeHSDState{T}, scalar_rhs::T, solve_merit::T,
) where {T}
    forcing = T(512) * sqrt(eps(one(T)))
    isfinite(forcing) && forcing > zero(T) || return T(Inf)
    five = _product_hsd_max_normalized_newton_error(state, scalar_rhs)
    isfinite(five) && isfinite(solve_merit) || return T(Inf)
    return max(five / forcing, solve_merit)
end

"""Solve one hybrid RHS and refine with the same pivoted factor at most twice."""
@inline function _product_hsd_coupled_solve_shift!(
    state::ProductConeHSDState{T}, scalar_rhs::T,
) where {T}
    workspace = state.coupled
    _product_hsd_coupled_rhs!(state, scalar_rhs) || return false
    solve_ok, solve_merit = _product_coupled_solve!(
        workspace, workspace.solution, workspace.rhs,
    )
    solve_ok || return false
    recovered = _product_hsd_coupled_recover!(
        state, workspace.solution,
    )
    if recovered && _product_hsd_newton_residual_ok(state, scalar_rhs)
        workspace.last_reason = COUPLED_READY
        return true
    end
    previous = recovered ?
        _product_hsd_coupled_merit(state, scalar_rhs, solve_merit) : T(Inf)

    @inbounds for _ in 1:2
        for i in 1:workspace.dimension
            workspace.correction_rhs[i] = -workspace.residual[i]
        end
        correction_ok, _ = _product_coupled_solve!(
            workspace, workspace.correction, workspace.correction_rhs,
        )
        correction_ok || return false
        for i in 1:workspace.dimension
            value = workspace.solution[i] + workspace.correction[i]
            isfinite(value) || begin
                workspace.last_reason = COUPLED_SOLVE_CERT_FAILED
                return false
            end
            workspace.solution[i] = value
        end
        workspace.refinements += 1
        total_ok, total_solve_merit =
            _product_coupled_solution_certificate!(
                workspace, workspace.solution, workspace.rhs,
            )
        if !total_ok
            workspace.last_reason = COUPLED_SOLVE_CERT_FAILED
            return false
        end
        recovered = _product_hsd_coupled_recover!(
            state, workspace.solution,
        )
        current = recovered ? _product_hsd_coupled_merit(
            state, scalar_rhs, total_solve_merit,
        ) : T(Inf)
        if recovered && _product_hsd_newton_residual_ok(state, scalar_rhs)
            current < previous || begin
                workspace.last_reason = COUPLED_REFINEMENT_STAGNATED
                return false
            end
            workspace.last_reason = COUPLED_READY
            return true
        end
        current < previous || begin
            workspace.last_reason = COUPLED_REFINEMENT_STAGNATED
            return false
        end
        previous = current
    end
    workspace.last_reason = COUPLED_FIVE_EQUATION_FAILED
    return false
end

@inline function _product_hsd_coupled_direction!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    _product_hsd_coupled_solve_shift!(state, predictor_scalar) || return false
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) || return false
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) || return false
    ratio = base.mu_aff / base.mu
    sigma = ratio * ratio * ratio
    sigma > one(T) && (sigma = one(T))
    sigma_mu = sigma * base.mu
    _product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    return _product_hsd_coupled_solve_shift!(state, corrector_scalar)
end

@inline function _product_hsd_boundary_alpha!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    ap = max_step_primal!(state.runtime, base.s, base.ds)
    ad = max_step_dual!(state.runtime, base.y, base.dy)
    (isfinite(ap) || ap == T(Inf)) || return T(NaN)
    (isfinite(ad) || ad == T(Inf)) || return T(NaN)
    (ap >= zero(T) && ad >= zero(T)) || return T(NaN)
    alpha = min(one(T), ap, ad)
    if base.dtau < zero(T)
        alpha = min(alpha, -base.tau / base.dtau)
    end
    if base.dkappa < zero(T)
        alpha = min(alpha, -base.kappa / base.dkappa)
    end
    return T(0.995) * alpha
end

@inline function _product_hsd_mu_aff!(
    state::ProductConeHSDState{T}, alpha::T,
) where {T}
    base = state.base
    acc = zero(T)
    @inbounds for k in 1:base.m
        sk = base.s[k] + alpha * base.ds[k]
        yk = base.y[k] + alpha * base.dy[k]
        acc += sk * yk
    end
    acc += (base.tau + alpha * base.dtau) *
           (base.kappa + alpha * base.dkappa)
    (isfinite(acc) && acc >= zero(T)) || return T(NaN)
    base.mu_aff = acc / T(base.nu + 1)
    return base.mu_aff
end

"""
Build the metric-consistent symmetric-cone corrector shift.

Canonical SOC coordinates use the ordinary Euclidean pairing, while the
Lorentz barrier `-log(t^2-‖u‖^2)` has degree two and
`-∇F(e) = 2e`.  Its central target is consequently `2σμe`; orthant and
PSD/svec blocks retain `σμe`.  This block weighting is what makes
`dot(s,y) = νμ` at a product-cone central point and is preserved by the
orthogonal RSOC-to-SOC canonical map.

All operands use state-owned product-runtime scratch.  In particular, this
does not materialise a product-cone matrix or allocate a block view.
"""
@inline function _product_hsd_corrector_shift!(
    state::ProductConeHSDState{T}, sigma_mu::T,
) where {T}
    runtime = state.runtime
    base = state.base

    # The generic runtime dispatches symmetric blocks through NT Jordan
    # algebra and Exp/Power blocks through the third-derivative higher-order
    # corrector. Keep the historical scratch-populating symmetric path for
    # the standalone scaled-frame oracle tests.
    if !isempty(runtime.exp) || !isempty(runtime.power)
        corrector_shift!(
            runtime, state.h, base.s, base.y,
            base.ds_a, base.dy_a, sigma_mu,
        )
        return state.h
    end

    apply_Rinv!(runtime, state.ds_hat, base.ds_a)
    apply_R!(runtime, state.dy_hat, base.dy_a)
    product_jordan!(runtime, state.h, state.ds_hat, state.dy_hat)

    # g_input = lambda, g_output = lambda∘lambda, gb = -∇F(e).
    apply_R!(runtime, state.g_input, base.y)
    product_jordan!(runtime, state.g_output, state.g_input, state.g_input)
    product_identity!(runtime, state.gb)
    @inbounds for block in runtime.soc
        state.gb[block.offset] += state.gb[block.offset]
    end
    @inbounds for k in 1:base.m
        state.g_input[k] = sigma_mu * state.gb[k] -
                           state.g_output[k] - state.h[k]
    end
    product_solve_Llambda!(runtime, state.g_output, state.g_input)
    apply_R!(runtime, state.h, state.g_output)
    return state.h
end

"""Predictor/corrector directions sharing one pivoted bordered factor."""
@inline function _product_hsd_direction!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    _product_hsd_solve_shift!(state, predictor_scalar) || return false
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) || return false
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) || return false
    ratio = base.mu_aff / base.mu
    sigma = ratio * ratio * ratio
    sigma > one(T) && (sigma = one(T))
    sigma_mu = sigma * base.mu

    _product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    return _product_hsd_solve_shift!(state, corrector_scalar)
end

@inline function _product_hsd_trial_scaling!(state::ProductConeHSDState)
    base = state.base
    return try_update_scaling!(state.runtime, base.st, base.yt, base.mu)
end

"""Single-alpha product-cone fraction-to-boundary line search."""
@inline function _product_hsd_line_search!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    alpha = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha) && alpha > zero(T)) || return false
    # Keep accepted iterates away from numerically unresolved PSD faces.  The
    # predictor still uses the aggressive 0.995 boundary estimate for the
    # frozen Mehrotra centering contract; only the accepted trial is damped.
    alpha *= T(0.9)
    p_norm = _hsd_maxinf(base.rP)
    d_norm = _hsd_maxinf(base.rD)
    g_norm = abs(base.rG)
    (isfinite(p_norm) && isfinite(d_norm) && isfinite(g_norm)) || return false
    backtracking = 0
    has_nonsymmetric = !isempty(state.runtime.exp) ||
                       !isempty(state.runtime.power)
    # PSD NT scaling can reject a strictly-interior trial until roundoff in a
    # near-boundary eigensystem is reduced below its backward-error gates (a
    # valid Lattice direction first becomes certifiable after 16 halvings),
    # so PSD- or nonsymmetric-containing products use the 64-trial budget.
    # Pure LP/SOC problems never needed more than the original 16 and the
    # wider budget over-damped their Mehrotra convergence (SOCP regression:
    # iteration_limit where 13 iterations previously converged).
    has_psd = !isempty(state.runtime.psd)
    max_backtracking = (has_psd || has_nonsymmetric) ? 64 : 16
    if has_nonsymmetric
        checkpoint_nonsymmetric_scaling!(state.runtime) || return false
    end
    accepted = false
    while !accepted
        @inbounds for j in 1:base.n
            base.xt[j] = base.x[j] + alpha * base.dx[j]
        end
        @inbounds for k in 1:base.m
            base.st[k] = base.s[k] + alpha * base.ds[k]
            base.yt[k] = base.y[k] + alpha * base.dy[k]
        end
        base.tau_t = base.tau + alpha * base.dtau
        base.kappa_t = base.kappa + alpha * base.dkappa
        ok = isfinite(base.tau_t) && isfinite(base.kappa_t) &&
             base.tau_t > zero(T) && base.kappa_t > zero(T) &&
             _product_hsd_trial_scaling!(state)
        if ok
            _hsd_trial_residual!(base)
            p2 = _hsd_maxinf(base.rPt)
            d2 = _hsd_maxinf(base.rDt)
            gap2 = -dot(base.c, base.xt) - dot(base.b, base.yt) + base.kappa_t
            scale = max(one(T), p_norm, d_norm, g_norm)
            tol = T(256) * sqrt(eps(T)) * scale
            accepted = isfinite(p2) && isfinite(d2) && isfinite(gap2) &&
                       _hsd_residual_homotopy_ok(base, alpha, p2, d2, gap2) &&
                       max(p2, d2, abs(gap2)) <= scale * T(1.0005) + tol
        end
        if !accepted
            # A failed nonsymmetric trial may leave the conjugate/scaling
            # workspace at a rejected point. Restore the last accepted pair
            # before reducing alpha so every trial is independent and a
            # validated accepted Fenchel shadow remains available as a warm
            # seed. This performs no factorization.
            if has_nonsymmetric
                restore_nonsymmetric_scaling_checkpoint!(state.runtime) ||
                    return false
            end
            alpha *= T(0.5)
            backtracking += 1
            backtracking >= max_backtracking && break
        end
    end
    if !accepted
        # Leave the pair runtime consistent with the unchanged base iterate.
        if has_nonsymmetric
            restore_nonsymmetric_scaling_checkpoint!(state.runtime)
        else
            try_update_scaling!(state.runtime, base.s, base.y, base.mu)
        end
        return false
    end
    @inbounds for j in 1:base.n
        base.x[j] = base.xt[j]
    end
    @inbounds for k in 1:base.m
        base.s[k] = base.st[k]
        base.y[k] = base.yt[k]
    end
    base.tau = base.tau_t
    base.kappa = base.kappa_t
    base.record.backtracking = backtracking
    base.record.primal_step = alpha
    base.record.dual_step = alpha
    base.record.step_size = alpha
    return true
end

"""
    product_hsd_step!(state) -> HSDStepCode

Execute one native LP/SOC/PSD/Exp/Power product-cone predictor/corrector epoch.
This is an explicit core API only: it does not assign solver status, recover a
public result, or fall back to a legacy/lifted route.
"""
function product_hsd_step!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    base.rank_ambiguous && return HSDStepDirectionFailed
    base.rank_incompatible && return HSDStepDirectionFailed
    hsd_residual!(base)
    if !isfinite(base.mu)
        return HSDStepDirectionFailed
    elseif base.mu <= zero(T)
        base.record.step_size = zero(T)
        return HSDStepAlreadyOptimal
    end
    scaling_ok = try_update_scaling!(state.runtime, base.s, base.y, base.mu)
    scaling_ok || return HSDStepDirectionFailed
    base.epoch += 1
    has_nonsymmetric = _product_hsd_has_nonsymmetric(state)
    direction_ok = if has_nonsymmetric
        assembled = try
            _product_hsd_form_coupled_matrix!(state)
        catch
            state.coupled.last_reason = COUPLED_ASSEMBLY_NONFINITE
            false
        end
        assembled ||
            return HSDStepDirectionFailed
        factor_ok = _product_coupled_factorize!(
            state.coupled, base.epoch,
        )
        if !factor_ok
            return state.coupled.last_reason === COUPLED_FACTOR_FAILED ?
                   HSDStepSingularKKT : HSDStepDirectionFailed
        end
        try
            _product_hsd_coupled_direction!(state)
        catch
            state.coupled.last_reason = COUPLED_FIVE_EQUATION_FAILED
            false
        end
    else
        border_scalar = try
            _product_hsd_form_schur_border!(state)
        catch
            return HSDStepDirectionFailed
        end
        (_hsd_matrix_finite(base.H) && isfinite(border_scalar) &&
         _product_hsd_vector_finite(base.qr) &&
         _product_hsd_vector_finite(base.rvec)) ||
            return HSDStepDirectionFailed
        assembled = try
            _product_hsd_assemble_bordered!(state, border_scalar)
        catch
            state.symmetric_bordered.last_reason =
                SYMMETRIC_BORDERED_ASSEMBLY_NONFINITE
            false
        end
        assembled || return HSDStepDirectionFailed
        factor_ok = _product_hsd_factor_bordered!(state)
        if !factor_ok
            return state.symmetric_bordered.last_reason ===
                   SYMMETRIC_BORDERED_FACTOR_FAILED ?
                   HSDStepSingularKKT : HSDStepDirectionFailed
        end
        try
            _product_hsd_direction!(state)
        catch
            state.symmetric_bordered.last_reason =
                SYMMETRIC_BORDERED_FIVE_EQUATION_FAILED
            false
        end
    end
    direction_ok || return HSDStepDirectionFailed
    accepted = try
        _product_hsd_line_search!(state)
    catch
        # Unexpected scaling-kernel failure remains fail-closed.  Restore the
        # runtime/base consistency when the original iterate is still valid.
        if !isempty(state.runtime.exp) || !isempty(state.runtime.power)
            restore_nonsymmetric_scaling_checkpoint!(state.runtime)
        else
            try_update_scaling!(state.runtime, base.s, base.y, base.mu)
        end
        false
    end
    accepted || return state.runtime.valid ?
                       HSDStepBreakdown : HSDStepDirectionFailed
    hsd_residual!(base)
    # The NT operators already correspond to the accepted `(s,y)` pair;
    # refresh the scalar metadata after recomputing the accepted-point gap.
    state.runtime.last_mu = base.mu
    _hsd_update_record!(base)
    base.record.iterations += 1
    return HSDStepOK
end
