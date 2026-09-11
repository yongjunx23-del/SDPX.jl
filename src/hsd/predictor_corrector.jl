# Predictor/corrector control for the product-cone HSD state machine.
# Extracted verbatim from product_cone_hsd.jl; frozen Newton equations live here.

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
    @inbounds for k in 1:base.m
        state.g_input[k] = sigma_mu * state.gb[k] -
                           state.g_output[k] - state.h[k]
    end
    product_solve_Llambda!(runtime, state.g_output, state.g_input)
    apply_R!(runtime, state.h, state.g_output)
    return state.h
end

@inline function _product_hsd_restore_affine_predictor!(
    state::ProductConeHSDState{T}, predictor_scalar::T,
) where {T}
    _runtime_ns_affine_fallback_reported(state.runtime) || return false
    base = state.base
    copy_owned!(base.dx, base.dx_a)
    copy_owned!(base.dy, base.dy_a)
    copy_owned!(base.ds, base.ds_a)
    base.dtau = base.dtau_a
    base.dkappa = base.dkappa_a
    fallback_result = state.runtime.last_nonsymmetric
    affine_shift!(state.runtime, state.h, base.s, base.y)
    _runtime_ns_restore_affine_fallback_report!(
        state.runtime, fallback_result,
    )
    core = state.symmetric_core
    if core isa FixedTraceQ3CoreWorkspace{T}
        _product_hsd_fixed_trace_hkm_linearization!(
            state, zero(T), false, false,
        ) || return false
    elseif core !== nothing
        cone = core.system.cone
        cone isa BlockProductConeLinearization{T} || return false
        copy_owned!(cone.corrector_rhs, state.h)
    else
        zero_owned!(base.ax)
        @inbounds for j in 1:base.n
            value = base.dx[j]
            iszero(value) && continue
            for pointer in nzrange(base.A, j)
                base.ax[base.A.rowval[pointer]] +=
                    base.A.nzval[pointer] * value
            end
        end
        apply_Theta!(state.runtime, base.e, base.dy)
    end
    restored = _product_hsd_newton_residual_ok(state, predictor_scalar)
    state.diagnostic = restored ?
        :corrector_fallback_to_affine_predictor :
        :corrector_affine_fallback_residual_failed
    return restored
end

@inline function _product_hsd_nonsymmetric_scaling(runtime, offset::Int)
    @inbounds for block in runtime.exp
        block.offset == offset && return block.scaling
    end
    @inbounds for block in runtime.power
        block.offset == offset && return block.scaling
    end
    return nothing
end

"""Predictor/corrector directions sharing one pivoted bordered factor."""
Base.@noinline function _product_hsd_direction!(
    state::ProductConeHSDState{T,R,RT,NS,CW,SB,SCW},
) where {T,R,RT,NS,CW,SB,SCW}
    base = state.base
    affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    _product_hsd_solve_shift!(state, predictor_scalar) || return false
    copy_owned!(base.dx_a, base.dx)
    copy_owned!(base.dy_a, base.dy)
    copy_owned!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) ||
        return (_sdpx_direction_failure_report(state, "boundary_alpha"); false)
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) ||
        return (_sdpx_direction_failure_report(state, "mu_aff"); false)
    ratio = base.mu_aff / base.mu
    sigma = _product_hsd_sigma(state, ratio)
    sigma_mu = sigma * base.mu
    # PR-07: record the actual per-step values (see HSDStepRecord).
    base.record.sigma_used = sigma
    base.record.alpha_aff = alpha_aff

    _product_hsd_corrector_shift!(state, sigma_mu)
    if _runtime_ns_affine_fallback_reported(state.runtime)
        return _product_hsd_restore_affine_predictor!(state, predictor_scalar)
    end
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    return _product_hsd_solve_shift!(state, corrector_scalar)
end

# =====================================================================
#    C7.2a: prepared symmetric-core production dispatch.
#
#    When `ProductConeHSDState` owns a prepared `SymmetricCoreWorkspace`,
#    the `:bordered` route executes the symmetric augmented core instead of
#    the legacy full-border LU.  This section owns no sign convention: every
#    Newton RHS is built from the frozen five-equation `residual_newton_rhs`
#    semantics and every direction is validated through the original
#    five-equation residual gate before line search/state update.
# =====================================================================

"""Fill the state-owned prepared core's block Theta operators and cone RHS.

Symmetric blocks materialize the accepted `Theta` action one local column at a
time through `apply_Theta!` (never a global m×m operator).  Exp/Power blocks
use the accepted fixed-size 3x3 `nonsymmetric_scaling_contribution3!` with
finite/symmetric/SPD acceptance.  The corrector stage may call this only to
rewrite the cone RHS (the operator values must be unchanged, verified by the
Theta signature guard in `_core_guard_ready!`).

Returns `true` on success; a non-converged or non-symmetric block fails closed.
"""
# Populate the state's fused residual scratch from the current core candidate.
# The state's `_product_hsd_newton_residual_ok` gate consumes `base.ax`
# (= A*dx), `base.e` (= Theta*dy) and `base.ds`; recompute them from the
# current `base.dx`/`base.dy`.  The core already enforces the frozen cone
# equation `ds + Theta*dy = h`, so `base.dy` stays the raw core dual
# direction and `base.e = Theta*dy` is formed from it directly (Clarabel
# semantics: the augmented solve is the authority for the dual direction).
# The `G(target)` recovery is a diagnostic scratch roundtrip only: it is
# written into `g_output`/`gb` and the per-block runtime scratch so the SOC/
# PSD/nonsymmetric roundtrip certificates can be (re)computed, but it never
# overwrites `base.dy`/`base.e`.  Returns the PSD-budget-inconclusive flag.
function _product_hsd_core_scatter!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    core = state.symmetric_core
    if core isa FixedTraceQ3CoreWorkspace{T}
        # The fixed-trace core already computed A*dx into `core.ax` for the
        # current ordinary candidate; the rescue path refreshes it explicitly
        # before entering this scatter.
        copy_owned!(base.ax, core.ax)
        @inbounds for row in 1:base.m
            _store_owned_scalar!(
                state.g_input, row,
                base.ax[row] + state.h[row] + base.rP[row] -
                base.b[row] * base.dtau,
            )
        end
        apply_cone_linearization!(base.e, core.system.cone, base.dy)
        return false
    end
    zero_owned!(base.ax)
    @inbounds for j in 1:base.n
        value = base.dx[j]
        iszero(value) && continue
        for pointer in nzrange(base.A, j)
            base.ax[base.A.rowval[pointer]] +=
                base.A.nzval[pointer] * value
        end
    end
    @inbounds for row in 1:base.m
        _store_owned_scalar!(
            state.g_input, row,
            base.ax[row] + state.h[row] + base.rP[row] -
            base.b[row] * base.dtau,
        )
    end
    # The cone complementarity term for the frozen five-equation gate must
    # use the exact operator carried by this NewtonSystem.  Fixed-trace HKM
    # intentionally differs from the runtime's generic SOC NT map.
    apply_Theta!(state.runtime, base.e, base.dy)
    # Diagnostic-only recovery: G(target) then Theta(G(target)) populate the
    # runtime block scratch and `gb`/`g_output` so the existing roundtrip
    # certificate machinery can be (re)evaluated.  The result is never copied
    # into `base.dy` or `base.e`; the raw core direction stays authoritative.
    apply_G!(state.runtime, state.g_output, state.g_input)
    apply_Theta!(state.runtime, state.gb, state.g_output)
    _, psd_inconclusive = _product_hsd_roundtrip_backward_status(state)
    return psd_inconclusive
end

# Rebuild the fixed-trace Ax cache only for the affine-predictor rescue.  The
# ordinary predictor/corrector scatter keeps the core-produced cache and avoids
# a second structured A scan.
@inline function _product_hsd_fixed_trace_rescue_scatter!(
    state::ProductConeHSDState{T},
) where {T}
    core = state.symmetric_core
    core isa FixedTraceQ3CoreWorkspace{T} || return false
    _fixed_trace_mul_A!(core.ax, core, state.base.dx)
    _product_hsd_core_scatter!(state)
    return true
end

# Fallback hook: the MultiFloat extension implements the 4-lane SIMD path.
function _hkm_vec4_linearization!(args...)
    return false
end

"""Prepare the exact HKM Q3 cone equation for one predictor/corrector RHS.

The complete map `M` satisfies `dy = r_HKM - M*ds`.  The frozen Newton
system therefore receives `Theta=M^-1` and `h=Theta*r_HKM`.  Predictor
refreshes `M/Theta`; corrector changes only `r_HKM/h`, preserving the one
factor and one homogeneous solve owned by the numeric epoch.
"""

function _product_hsd_fixed_trace_hkm_linearization!(
    state::ProductConeHSDState{T}, target::T,
    include_affine_product::Bool, refresh_metric::Bool,
) where {T}
    core = state.symmetric_core
    core isa FixedTraceQ3CoreWorkspace{T} || return false
    cone = core.system.cone
    cone isa BlockProductConeLinearization{T} || return false
    base = state.base
    plan = core.plan
    length(plan.soc_blocks) == length(plan.soc_operator_indices) || return false
    zero_owned!(state.h)
    zero_owned!(cone.corrector_rhs)
    if refresh_metric
        for operator in cone.operators
            zero_owned!(operator)
        end
    elseif core.linearization_epoch != base.epoch
        return false
    end

    # Optional 4-lane SIMD fast path (MultiFloat extension).  It mirrors the
    # scalar loop's fail-closed semantics exactly; returning false falls
    # through to the scalar path below.
    if _hkm_vec4_linearization!(
        state, target, include_affine_product, refresh_metric,
    )
        refresh_metric && (core.linearization_epoch = base.epoch)
        return true
    end

    blocks = plan.soc_blocks
    failed = Threads.Atomic{Bool}(false)
    run_block = function (block_index::Int)
        failed[] && return
        block = blocks[block_index]
        row0 = block.offset - 1
        rows = block.offset:(block.offset + 2)
        operator_index = plan.soc_operator_indices[block_index]
        cone.block_ranges[operator_index] == rows ||
            (failed[] = true; return)
        operator = cone.operators[operator_index]
        M = view(core.theta_inverse, :, :, block_index)
        primal = view(base.s, rows)
        dual = view(base.y, rows)
        if refresh_metric
            _soc_fixed_trace_hkm_full_metric!(M, primal, dual) ||
                (failed[] = true; return)
            _fixed_trace_spd3_inverse!(operator, M) ||
                (failed[] = true; return)
        end
        all(isfinite, operator) || (failed[] = true; return)
        affine_primal = view(base.ds_a, rows)
        affine_dual = view(base.dy_a, rows)
        r = view(core.hkm_rhs, :, block_index)
        _soc_fixed_trace_hkm_rhs!(
            r, primal, dual, affine_primal, affine_dual,
            target, include_affine_product,
        ) || (failed[] = true; return)
        for i in 1:3
            value = zero(T)
            for j in 1:3
                value += operator[i,j] * r[j]
            end
            isfinite(value) || (failed[] = true; return)
            state.h[row0 + i] = value
            cone.corrector_rhs[row0 + i] = _core_owned_value(value)
        end
        return
    end
    _q3_foreach(run_block, eachindex(blocks), core.worker_budget)
    failed[] && return false
    refresh_metric && (core.linearization_epoch = base.epoch)
    return true
end

function _product_hsd_symmetric_core_linearization!(
    state::ProductConeHSDState{T}, corrector_rhs::AbstractVector{T},
) where {T}
    core = state.symmetric_core
    core === nothing && return false
    cone = core.system.cone
    cone isa BlockProductConeLinearization{T} || return false
    m = state.base.m
    length(corrector_rhs) == m || return false
    basis = state.g_input
    image = state.g_output
    forcing = T(64) * eps(T)
    has_scalar_blocks=any(rows->length(rows)==1,cone.block_ranges)
    if has_scalar_blocks
        fill!(basis,one(T))
        apply_Theta!(state.runtime,image,basis)
    end
    for (index, rows) in enumerate(cone.block_ranges)
        operator = cone.operators[index]
        dimension=length(rows)
        dimension == size(operator, 1) == size(operator, 2) || return false
        if dimension==1
            row=first(rows)
            operator[1,1]=_core_owned_value(image[row])
            cone.corrector_rhs[row]=_core_owned_value(corrector_rhs[row])
            continue
        end
        scaling = _product_hsd_nonsymmetric_scaling(
            state.runtime, first(rows),
        )
        if scaling !== nothing
            dimension == 3 || return false
            reason = nonsymmetric_scaling_contribution3!(
                operator,
                view(cone.corrector_rhs, rows),
                scaling,
                view(corrector_rhs, rows),
            )
            reason === NS_SCALING_CONVERGED || return false
            continue
        end
        @inbounds for local_column in 1:dimension
            zero_distinct!(basis)
            _store_owned_scalar!(
                basis, rows[local_column], one(T),
            )
            apply_Theta!(state.runtime, image, basis)
            for local_row in 1:dimension
                operator[local_row, local_column] =
                    _core_owned_value(image[rows[local_row]])
            end
            cone.corrector_rhs[rows[local_column]] =
                _core_owned_value(corrector_rhs[rows[local_column]])
        end
    end
    # Certify each block operator is finite and self-adjoint to roundoff and
    # freeze the lower triangle as the single authority.
    for (index, rows) in enumerate(cone.block_ranges)
        operator = cone.operators[index]
        all(isfinite, operator) || return false
        @inbounds for local_column in 1:size(operator, 1)
            for local_row in (local_column + 1):size(operator, 1)
                lower = operator[local_row, local_column]
                upper = operator[local_column, local_row]
                work = abs(lower) + abs(upper)
                discrepancy = abs(lower - upper)
                if !(isfinite(work) && isfinite(discrepancy)) ||
                   (!iszero(work) && discrepancy > forcing * work) ||
                   (iszero(work) && !iszero(discrepancy))
                    return false
                end
                operator[local_column, local_row] =
                    _core_owned_value(lower)
            end
        end
    end
    return true
end

"""Build the semantic predictor/corrector NewtonSystem for the prepared core.

`cone_corrector_rhs` is the already-written cone corrector vector (predictor
`state.h` or the corrected shift).  Negated residuals are written into the
core-owned buffers; the frozen `residual_newton_rhs` signs are reproduced
exactly and no hidden sign convention exists.
"""
function _product_hsd_symmetric_core_system(
    state::ProductConeHSDState{T}, scalar_rhs::T,
) where {T}
    core = state.symmetric_core
    core === nothing && return nothing
    base = state.base
    all(isfinite, base.rP) && all(isfinite, base.rD) && isfinite(base.rG) &&
    all(isfinite, core.system.cone.corrector_rhs) && isfinite(scalar_rhs) ||
        throw(ArgumentError("HSD Newton RHS contains non-finite data"))
    @inbounds for index in 1:base.m
        _core_store_owned!(core.negated_primal, index, -base.rP[index])
    end
    @inbounds for index in 1:base.n
        _core_store_owned!(core.negated_dual, index, -base.rD[index])
    end
    rhs = HSDNewtonRHS(
        core.negated_primal, core.negated_dual, _core_owned_value(-base.rG),
        core.system.cone.corrector_rhs, _core_owned_value(scalar_rhs),
    )
    return NewtonSystem(
        base.A, base.b, base.c, core.system.cone,
        base.tau, base.kappa, rhs,
    )
end

"""Run one prepared-core predictor/corrector epoch.

Factor the core once, solve the homogeneous RHS once, then solve the
predictor and corrector variable RHS with the same factor.  The final
direction must pass the original five-equation residual gate; no legacy
bordered fallback is attempted when the core is present.
"""
function _product_hsd_symmetric_core_direction!(
    state::ProductConeHSDState{T,R,RT,NS,CW,SB,SCW},
) where {T,R,RT,NS,CW,SB,SCW}
    core = state.symmetric_core
    core === nothing && return false
    base = state.base
    fixed_trace = core isa FixedTraceQ3CoreWorkspace{T}
    timings = state.phase_timings
    refinement_iter0 = core.refinements

    # Predictor.
    predictor_scalar = -base.tau * base.kappa
    t0 = time_ns()
    predictor_linearized = if fixed_trace
        _product_hsd_fixed_trace_hkm_linearization!(
            state, zero(T), false, true,
        )
    else
        affine_shift!(state.runtime, state.h, base.s, base.y)
        _product_hsd_symmetric_core_linearization!(state, state.h)
    end
    predictor_linearized || begin
        state.diagnostic = fixed_trace ?
            :disjoint_fixed_head_q3_predictor_linearization_failed :
            :symmetric_core_predictor_linearization_failed
        return false
    end
    predictor_system = _product_hsd_symmetric_core_system(
        state, predictor_scalar,
    )
    predictor_system === nothing && return false
    timings.schur_assembly_seconds += Float64(time_ns() - t0) * 1.0e-9
    t0 = time_ns()
    factor_symmetric_core_epoch!(
        core, predictor_system, base.epoch,
    )
    timings.kkt_factorization_seconds +=
        Float64(time_ns() - t0) * 1.0e-9
    if core isa FixedTraceQ3CoreWorkspace
        # Exclusive Q3 sub-buckets (children of kkt_factorization_seconds),
        # accumulated in the workspace across epochs.  Assign cumulative
        # values. Worker metadata is an admitted maximum, not utilization.
        q3 = core.epoch_timing
        timings.q3_metric_seconds = q3.metric_seconds
        timings.q3_factor_seconds = q3.factor_seconds
        timings.q3_homogeneous_seconds = q3.homogeneous_seconds
        timings.q3_epochs = q3.epochs
        timings.q3_workers = max(timings.q3_workers, q3.workers)
        timings.q3_local_elimination_seconds = core.equality.local_elimination_seconds
        timings.q3_panel_transform_seconds = core.equality.panel_transform_seconds
        timings.q3_gram_seconds = core.equality.gram_seconds
    end
    t0 = time_ns()
    refinement_wall0 = timings.refinement_seconds
    _core_t0 = time_ns()
    predictor_candidate, predictor_residual, _ = fixed_trace ?
        _core_solve_raw!(core, predictor_system; compute_residual=false) :
        _core_solve_raw!(core, predictor_system)
    timings.core_solve_seconds += Float64(time_ns() - _core_t0) * 1.0e-9
    # Disjoint phase partition: the refine wall share inside this call was
    # accumulated directly into `refinement_seconds` by `_core_refine!`;
    # the solve bucket keeps the remainder of the call wall, extended
    # through direction materialization (copy/scatter/finite/residual
    # gates) so every wall fraction of the direction is attributed.
    copy_owned!(base.dx, predictor_candidate.dx)
    copy_owned!(base.dy, predictor_candidate.dy)
    copy_owned!(base.ds, predictor_candidate.ds)
    base.dtau = predictor_candidate.dtau
    base.dkappa = predictor_candidate.dkappa
    _scatter_t0 = time_ns()
    _product_hsd_core_scatter!(state)
    timings.core_scatter_seconds += Float64(time_ns() - _scatter_t0) * 1.0e-9
    _hsd_direction_finite(base) || begin
        state.diagnostic = fixed_trace ?
            :disjoint_fixed_head_q3_predictor_nonfinite :
            :symmetric_core_predictor_nonfinite
        return false
    end
    _gate_t0 = time_ns()
    _gate_ok = _product_hsd_newton_residual_ok(state, predictor_scalar)
    timings.core_gate_seconds += Float64(time_ns() - _gate_t0) * 1.0e-9
    if !_gate_ok
        state.diagnostic = fixed_trace ?
            :disjoint_fixed_head_q3_predictor_residual_failed :
            :symmetric_core_predictor_residual_failed
        return false
    end
    timings.predictor_linear_solve_seconds +=
        Float64(time_ns() - t0) * 1.0e-9 -
        (timings.refinement_seconds - refinement_wall0)
    copy_owned!(base.dx_a, base.dx)
    copy_owned!(base.dy_a, base.dy)
    copy_owned!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    # Affine-step / centering-parameter computation feeds the corrector RHS.
    t0 = time_ns()
    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) || begin
        state.diagnostic = :symmetric_core_affine_boundary_failed
        return false
    end
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) || begin
        state.diagnostic = :symmetric_core_affine_mu_failed
        return false
    end
    ratio = base.mu_aff / base.mu
    sigma = _product_hsd_sigma(state, ratio)
    sigma_mu = sigma * base.mu
    timings.corrector_rhs_seconds += Float64(time_ns() - t0) * 1.0e-9

    # Corrector: only the cone RHS and scalar shift change; the operator and
    # local/equality factor remain the predictor epoch's authority.
    t0 = time_ns()
    corrector_linearized = if fixed_trace
        _product_hsd_fixed_trace_hkm_linearization!(
            state, sigma_mu, true, false,
        )
    else
        _product_hsd_corrector_shift!(state, sigma_mu)
        _product_hsd_symmetric_core_linearization!(state, state.h)
    end
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    corrector_linearized || begin
        state.diagnostic = fixed_trace ?
            :disjoint_fixed_head_q3_corrector_linearization_failed :
            :symmetric_core_corrector_linearization_failed
        return false
    end
    if _runtime_ns_affine_fallback_reported(state.runtime)
        restored = _product_hsd_restore_affine_predictor!(
            state, predictor_scalar,
        )
        state.diagnostic = restored ?
            :corrector_fallback_to_affine_predictor :
            :corrector_affine_fallback_residual_failed
        return restored
    end
    corrector_system = _product_hsd_symmetric_core_system(
        state, corrector_scalar,
    )
    corrector_system === nothing && return false
    timings.corrector_rhs_seconds += Float64(time_ns() - t0) * 1.0e-9
    t0 = time_ns()
    refinement_wall0 = timings.refinement_seconds
    corrector_candidate, corrector_residual, _ = fixed_trace ?
        _core_solve_raw!(core, corrector_system; compute_residual=false) :
        _core_solve_raw!(core, corrector_system)
    timings.refinement_iterations =
        core.refinements - refinement_iter0
    copy_owned!(base.dx, corrector_candidate.dx)
    copy_owned!(base.dy, corrector_candidate.dy)
    copy_owned!(base.ds, corrector_candidate.ds)
    base.dtau = corrector_candidate.dtau
    base.dkappa = corrector_candidate.dkappa
    _product_hsd_core_scatter!(state)
    _hsd_direction_finite(base) || begin
        state.diagnostic = fixed_trace ?
            :disjoint_fixed_head_q3_corrector_nonfinite :
            :symmetric_core_corrector_nonfinite
        return false
    end
    if !_product_hsd_newton_residual_ok(state, corrector_scalar)
        copy_owned!(base.dx, base.dx_a)
        copy_owned!(base.dy, base.dy_a)
        copy_owned!(base.ds, base.ds_a)
        base.dtau = base.dtau_a
        base.dkappa = base.dkappa_a
        restored = if fixed_trace
            _product_hsd_fixed_trace_hkm_linearization!(
                state, zero(T), false, false,
            )
        else
            affine_shift!(state.runtime, state.h, base.s, base.y)
            cone = core.system.cone
            cone isa BlockProductConeLinearization{T} || false
            if cone isa BlockProductConeLinearization{T}
                copy_owned!(cone.corrector_rhs, state.h)
                true
            else
                false
            end
        end
        if restored
            # The fixed-trace core's Ax cache still belongs to the rejected
            # corrector until it is rebuilt from the restored predictor.
            if fixed_trace
                _product_hsd_fixed_trace_rescue_scatter!(state) || return false
            else
                _product_hsd_core_scatter!(state)
            end
            restored = _product_hsd_newton_residual_ok(state, predictor_scalar)
        end
        state.diagnostic = restored ?
            :corrector_fallback_to_affine_predictor :
            :corrector_affine_fallback_residual_failed
        return restored
    end
    timings.corrector_linear_solve_seconds +=
        Float64(time_ns() - t0) * 1.0e-9 -
        (timings.refinement_seconds - refinement_wall0)
    return true
end
