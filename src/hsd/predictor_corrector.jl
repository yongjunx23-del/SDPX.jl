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

@inline function _product_hsd_nonsymmetric_scaling(runtime, offset::Int)
    @inbounds for block in runtime.exp
        block.offset == offset && return block.scaling
    end
    @inbounds for block in runtime.power
        block.offset == offset && return block.scaling
    end
    return nothing
end

function _product_hsd_expanded_linearization(
    state::ProductConeHSDState{T}, corrector_rhs::AbstractVector{T},
) where {T}
    session = state.expanded
    m = state.base.m
    # Dense route: refill the block-diagonal product operator directly. Exp
    # and Power blocks use their accepted fixed-size 3x3 contribution; other
    # blocks retain the authoritative Theta action one local column at a time.
    operator = session.cone_operator
    fill!(operator, zero(T))
    copyto!(session.cone_corrector_rhs, corrector_rhs)
    basis = state.g_input
    image = state.g_output
    for rows in session.cone_block_ranges
        scaling = _product_hsd_nonsymmetric_scaling(
            state.runtime, first(rows),
        )
        if scaling !== nothing
            length(rows) == 3 || return nothing
            reason = nonsymmetric_scaling_contribution3!(
                view(operator, rows, rows),
                view(session.cone_corrector_rhs, rows),
                scaling,
                view(corrector_rhs, rows),
            )
            reason === NS_SCALING_CONVERGED || return nothing
            continue
        end
        @inbounds for column in rows
            fill!(basis, zero(T))
            basis[column] = one(T)
            apply_Theta!(state.runtime, image, basis)
            for row in rows
                operator[row, column] = image[row]
            end
        end
    end
    # Cone kernels promise a self-adjoint map. Independently materialized
    # columns can differ by their operation-order roundoff; certify that
    # skew part componentwise, then freeze the lower triangle as the single
    # self-adjoint authority. This is a setup backward-error check, not a
    # solver stopping tolerance.
    forcing = T(64) * eps(T)
    @inbounds for column in 1:m
        for row in (column + 1):m
            lower = operator[row, column]
            upper = operator[column, row]
            work = abs(lower) + abs(upper)
            discrepancy = abs(lower - upper)
            if !(isfinite(work) && isfinite(discrepancy)) ||
               (!iszero(work) && discrepancy > forcing * work) ||
               (iszero(work) && !iszero(discrepancy))
                return nothing
            end
            operator[column, row] = lower
        end
    end
    # Assemble into the session-owned product linearization. The same
    # self-adjointness/finite-data checks that the LocalConeLinearization and
    # assemble_cone_linearization seam ran are preserved verbatim below.
    issymmetric(operator) || throw(ArgumentError(
        "cone linearization must be self-adjoint",
    ))
    all(isfinite, operator) || throw(ArgumentError(
        "cone linearization contains non-finite data",
    ))
    all(isfinite, session.cone_corrector_rhs) || throw(ArgumentError(
        "cone corrector right-hand side contains non-finite data",
    ))
    return ProductConeLinearization{T}(
        operator, session.cone_corrector_rhs, session.cone_block_ranges,
    )
end

"""Materialize only independent dense cone blocks for sparse Schur assembly."""
function _product_hsd_sparse_linearization(
    state::ProductConeHSDState{T}, corrector_rhs::AbstractVector{T},
) where {T}
    m = state.base.m
    length(corrector_rhs) == m || throw(DimensionMismatch(
        "sparse cone corrector dimension mismatch",
    ))
    ranges = UnitRange{Int}[
        block.offset:(block.offset + block.length - 1)
        for block in state.base.canonical.cone_layout.blocks
    ]
    validate_product_cone_block_ranges(m, ranges)
    operators = Matrix{T}[]
    basis = state.g_input
    image = state.g_output
    forcing = T(64) * eps(T)
    copyto!(state.expanded.cone_corrector_rhs, corrector_rhs)
    for rows in ranges
        dimension = length(rows)
        operator = alloc_zeros(T, dimension, dimension)
        scaling = _product_hsd_nonsymmetric_scaling(
            state.runtime, first(rows),
        )
        if scaling !== nothing
            dimension == 3 || return nothing
            reason = nonsymmetric_scaling_contribution3!(
                operator,
                view(state.expanded.cone_corrector_rhs, rows),
                scaling,
                view(corrector_rhs, rows),
            )
            reason === NS_SCALING_CONVERGED || return nothing
        else
            @inbounds for local_column in 1:dimension
                fill!(basis, zero(T))
                basis[rows[local_column]] = one(T)
                apply_Theta!(state.runtime, image, basis)
                for local_row in 1:dimension
                    operator[local_row, local_column] = image[rows[local_row]]
                end
            end
        end
        @inbounds for column in 1:dimension
            for row in (column + 1):dimension
                lower = operator[row, column]
                upper = operator[column, row]
                work = abs(lower) + abs(upper)
                discrepancy = abs(lower - upper)
                if !(isfinite(work) && isfinite(discrepancy)) ||
                   (!iszero(work) && discrepancy > forcing * work) ||
                   (iszero(work) && !iszero(discrepancy))
                    return nothing
                end
                operator[column, row] = lower
            end
        end
        push!(operators, operator)
    end
    return BlockProductConeLinearization{T}(
        operators, state.expanded.cone_corrector_rhs, ranges,
    )
end

function _product_hsd_expanded_system(
    state::ProductConeHSDState{T}, cone, scalar_rhs::T,
) where {T}
    base = state.base
    session = state.expanded
    # `residual_newton_rhs` semantics are reproduced exactly (including its
    # fail-closed non-finite gate) but the negated residual vectors are
    # written into session-owned buffers instead of being allocated.
    all(isfinite, base.rP) && all(isfinite, base.rD) && isfinite(base.rG) &&
    all(isfinite, cone.corrector_rhs) && isfinite(scalar_rhs) ||
        throw(ArgumentError("HSD Newton RHS contains non-finite data"))
    @inbounds for index in 1:base.m
        session.negated_primal[index] = -base.rP[index]
    end
    @inbounds for index in 1:base.n
        session.negated_dual[index] = -base.rD[index]
    end
    rhs = HSDNewtonRHS(
        session.negated_primal, session.negated_dual, -base.rG,
        cone.corrector_rhs, scalar_rhs,
    )
    return NewtonSystem(
        base.A, base.b, base.c, cone, base.tau, base.kappa, rhs,
    )
end

function _product_hsd_expanded_solve_shift!(
    state::ProductConeHSDState{T}, cone, scalar_rhs::T;
    stage::Symbol=:predictor,
) where {T}
    session = state.expanded
    rhs = stage === :predictor ? session.predictor_rhs : session.corrector_rhs
    system = _product_hsd_expanded_system(state, cone, scalar_rhs)
    # The condensed RHS is constructed here, strictly after the previous
    # predictor solve has completed and been certified. Predictor and
    # corrector RHS are never built or solved simultaneously; each solve
    # reuses the single factorization (and its immutable receipt) below.
    expanded_rhs!(rhs, system)
    if stage === :predictor
        session.predictor_solve_count += 1
    else
        session.corrector_solve_count += 1
    end
    solution = session.solution
    # Predictor certification changes only the semantic status; the same
    # factor remains numerically valid for the corrector RHS, so no
    # re-factorization and no receipt rebuild happen between the solves.
    session.status = EXPANDED_KKT_FACTORED
    solve_expanded!(solution, session, rhs) || return false
    refined = refine_expanded!(solution, session, rhs; system=system)
    if !refined
        # A regularized factor can expose the adjacent homogeneous-border
        # inertia only during refinement. Retry the same RHS once with the
        # exact unregularized operator under the narrow diagnosed gate below.
        _product_hsd_factor_exact_expanded_border!(state, system) || return false
        solve_expanded!(solution, session, rhs) || return false
        refine_expanded!(solution, session, rhs; system=system) || return false
    end
    direction = recover_expanded_direction!(session, system, solution)
    base = state.base
    copyto!(base.dx, direction.dx)
    copyto!(base.dy, direction.dy)
    copyto!(base.ds, direction.ds)
    base.dtau = direction.dtau
    base.dkappa = direction.dkappa
    _hsd_direction_finite(base) || return false
    residual_workspace = product_hsd_residual_workspace(state)
    residual_workspace === nothing && return false
    evaluate_direction!(residual_workspace, system, direction)
    semantic_residual = session.newton_residual
    newton_residual!(
        semantic_residual, system, direction, residual_workspace,
    )
    scale = max(
        norm(rhs, Inf), norm(solution, Inf) *
        _expanded_operator_scale(session.unregularized), one(T),
    )
    return fused_max_newton_residual(residual_workspace) <=
           T(512) * eps(T) * scale
end

"""
Factor the expanded HSD condensation without mistaking the homogeneous border
for a cone-curvature inertia failure. The generic session expects `(n,m+1)`;
for a positive cone operator the unregularized symmetric companion can instead
have the finite adjacent inertia `(n+1,m)` solely because the tau border is not
a definite member of the y block. In that one diagnosed case, factor the exact
unregularized frozen-sign operator and retain the ordinary refinement plus
five-equation semantic checks as authority. Any zero pivot, other inertia, or
non-positive cone operator still fails closed.
"""
function _product_hsd_factor_exact_expanded_border!(
    state::ProductConeHSDState{T}, system::NewtonSystem{T},
) where {T}
    session = state.expanded
    observed = session.inertia_factor.inertia
    adjacent_border = observed == KKTInertia(session.n + 1, session.m, 0)
    diagnosed_border = adjacent_border || any(
        attempt -> attempt.reason === EXPANDED_ATTEMPT_WRONG_INERTIA &&
                   attempt.observed_inertia ==
                       KKTInertia(session.n + 1, session.m, 0),
        session.attempts,
    )
    diagnosed_border || return false
    try
        LinearAlgebra.cholesky(
            LinearAlgebra.Symmetric(system.cone.operator); check=true,
        )
    catch exception
        exception isa LinearAlgebra.PosDefException || rethrow()
        return false
    end

    # The exact-border retry rewrites the active operator in place while the
    # session may still own a receipt from the accepted regularized factor.
    # Bump the mutation token and revoke the receipt before the rewrite; the
    # rebuilt receipt below is the only ownership for the new factor.
    session.operator_generation += 1
    session.factor_receipt = nothing
    copy_owned!(session.regularized, session.unregularized)
    # The generic ladder's scale-relative pivot floor can reject an exact
    # homogeneous border whose small pivot is still resolvable componentwise.
    # Use the scalar arithmetic floor here; the subsequent unregularized
    # refinement and semantic Newton residual remain the acceptance gates.
    pivot_floor = T(32) * eps(T)
    _factor_expanded_exact!(session, pivot_floor) || return false
    session.regularization = zero(T)
    session.status = EXPANDED_KKT_FACTORED
    # The exact-border exception is still an ordinary numeric factor epoch.
    # Build only its immutable receipt; all RHS-dependent refinement and
    # five-equation acceptance checks remain unchanged below this seam.
    _build_expanded_factor_receipt!(session)
    state.diagnostic = :expanded_exact_border_inertia
    return true
end

function _product_hsd_factor_expanded!(
    state::ProductConeHSDState{T}, system::NewtonSystem{T},
) where {T}
    factor_expanded_kkt!(state.expanded, system) && return true
    state.expanded.status === EXPANDED_KKT_WRONG_INERTIA || return false
    return _product_hsd_factor_exact_expanded_border!(state, system)
end

"""Predictor/corrector directions sharing one expanded KKT factor."""
function _product_hsd_expanded_direction!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    cone = _product_hsd_expanded_linearization(state, state.h)
    cone === nothing && return false
    predictor_system = _product_hsd_expanded_system(
        state, cone, predictor_scalar,
    )
    _product_hsd_factor_expanded!(state, predictor_system) || return false
    _product_hsd_expanded_solve_shift!(
        state, cone, predictor_scalar,
    ) || return false
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
    sigma = _product_hsd_sigma(state, ratio)
    sigma_mu = sigma * base.mu
    _product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    # The corrector RHS is constructed only after the predictor direction has
    # been solved and certified: the corrected shift is copied into the
    # session-owned cone buffer and the predictor cone operator is shared.
    copyto!(state.expanded.cone_corrector_rhs, state.h)
    corrector_cone = ProductConeLinearization{T}(
        state.expanded.cone_operator, state.expanded.cone_corrector_rhs,
        state.expanded.cone_block_ranges,
    )
    return _product_hsd_expanded_solve_shift!(
        state, corrector_cone, corrector_scalar; stage=:corrector,
    )
end

function _product_hsd_sparse_solve_shift!(
    state::ProductConeHSDState{T}, cone, scalar_rhs::T;
    factor_operator::Bool,
) where {T}
    session = state.sparse_schur
    session === nothing && return false
    system = _product_hsd_expanded_system(state, cone, scalar_rhs)
    if factor_operator
        assemble_sparse_schur_operator!(session, system) || return false
        assemble_sparse_schur_rhs!(session, system) || return false
        factor_sparse_schur!(session) || return false
    else
        assemble_sparse_schur_rhs!(session, system) || return false
    end
    solution = session.solution_vector
    solve_sparse_schur!(session, solution) || return false
    direction = try
        recover_reduced_direction(system, solution, session)
    catch exception
        exception isa InterruptException && rethrow()
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_SOLVE_FAILED, :sparse_recovery_failed,
        )
    end
    base = state.base
    copyto!(base.dx, direction.dx)
    copyto!(base.dy, direction.dy)
    copyto!(base.ds, direction.ds)
    base.dtau = direction.dtau
    base.dkappa = direction.dkappa
    if !_hsd_direction_finite(base)
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_SOLVE_FAILED,
            :sparse_direction_nonfinite,
        )
    end
    residual_workspace = product_hsd_residual_workspace(state)
    residual_workspace === nothing && return _invalidate_sparse_schur_factor!(
        session, SPARSE_SCHUR_SOLVE_FAILED,
        :sparse_residual_workspace_unavailable,
    )
    evaluate_direction!(residual_workspace, system, direction)
    scale = max(
        norm(session.rhs, Inf),
        _schur_operator_scale(session.schur) * norm(solution, Inf), one(T),
    )
    if fused_max_newton_residual(residual_workspace) >
       T(512) * eps(T) * scale
        return _invalidate_sparse_schur_factor!(
            session, SPARSE_SCHUR_REFINEMENT_STAGNATED,
            :sparse_semantic_residual_failed,
        )
    end
    session.status = factor_operator ? SPARSE_SCHUR_FACTORED :
                     SPARSE_SCHUR_UNREGULARIZED_CERTIFIED
    session.last_reason = :none
    return true
end

"""Predictor/corrector directions sharing one sparse reduced-Schur factor."""
function _product_hsd_sparse_direction!(state::ProductConeHSDState{T}) where {T}
    state.sparse_schur === nothing && return false
    base = state.base
    affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    cone = _product_hsd_sparse_linearization(state, state.h)
    cone === nothing && return _invalidate_sparse_schur_factor!(
        state.sparse_schur, SPARSE_SCHUR_FACTOR_FAILED,
        :sparse_cone_linearization_failed,
    )
    _product_hsd_sparse_solve_shift!(
        state, cone, predictor_scalar; factor_operator=true,
    ) || return false
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) ||
        return _invalidate_sparse_schur_factor!(
            state.sparse_schur, SPARSE_SCHUR_REFINEMENT_STAGNATED,
            :sparse_affine_boundary_failed,
        )
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) ||
        return _invalidate_sparse_schur_factor!(
            state.sparse_schur, SPARSE_SCHUR_REFINEMENT_STAGNATED,
            :sparse_affine_mu_failed,
        )
    ratio = base.mu_aff / base.mu
    sigma = _product_hsd_sigma(state, ratio)
    sigma_mu = sigma * base.mu
    _product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    # The corrected shift is copied into the session-owned cone buffer; the
    # sparse block operators are shared with the predictor cone and the same
    # reduced-Schur factor is reused for the corrector solve.
    copyto!(state.expanded.cone_corrector_rhs, state.h)
    corrector_cone = BlockProductConeLinearization{T}(
        cone.operators, state.expanded.cone_corrector_rhs, cone.block_ranges,
    )
    return _product_hsd_sparse_solve_shift!(
        state, corrector_cone, corrector_scalar; factor_operator=false,
    )
end

"""Predictor/corrector directions sharing one pivoted bordered factor."""
Base.@noinline function _product_hsd_direction!(
    state::ProductConeHSDState{T,R,RT,NS,CW,SB,EW,SW,SCW},
) where {T,R,RT,NS,CW,SB,EW,SW,SCW}
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
    (isfinite(alpha_aff) && alpha_aff > zero(T)) || return false
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) || return false
    ratio = base.mu_aff / base.mu
    sigma = _product_hsd_sigma(state, ratio)
    sigma_mu = sigma * base.mu

    _product_hsd_corrector_shift!(state, sigma_mu)
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
        # The fixed-trace core already computed A*dx into `core.ax`;
        # reuse it instead of a second full sparse scan.
        copyto!(base.ax, core.ax)
        @inbounds for row in 1:base.m
            state.g_input[row] = base.ax[row] + state.h[row] +
                                 base.rP[row] - base.b[row] * base.dtau
        end
        apply_cone_linearization!(base.e, core.system.cone, base.dy)
        return false
    end
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
        state.g_input[row] = base.ax[row] + state.h[row] +
                             base.rP[row] - base.b[row] * base.dtau
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
    fill!(state.h, zero(T))
    fill!(cone.corrector_rhs, zero(T))
    if refresh_metric
        for operator in cone.operators
            fill!(operator, zero(T))
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
    next_block = Threads.Atomic{Int}(1)
    failed = Threads.Atomic{Bool}(false)
    run_block = function (block_index::Int)
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
    if Threads.nthreads() <= 1 || length(blocks) < 256
        for block_index in eachindex(blocks)
            run_block(block_index)
            failed[] && return false
        end
    else
        @sync for _ in 1:Threads.nthreads()
            Threads.@spawn begin
                while !failed[]
                    block_index = Threads.atomic_add!(next_block, 1)
                    block_index > length(blocks) && break
                    run_block(block_index)
                end
            end
        end
        failed[] && return false
    end
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
            fill!(basis, zero(T))
            basis[rows[local_column]] = one(T)
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
    state::ProductConeHSDState{T,R,RT,NS,CW,SB,EW,SW,SCW},
) where {T,R,RT,NS,CW,SB,EW,SW,SCW}
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
    t0 = time_ns()
    refinement_wall0 = timings.refinement_seconds
    predictor_candidate, predictor_residual, _ = fixed_trace ?
        _core_solve_raw!(core, predictor_system; compute_residual=false) :
        _core_solve_raw!(core, predictor_system)
    # Disjoint phase partition: the refine wall share inside this call was
    # accumulated directly into `refinement_seconds` by `_core_refine!`;
    # the solve bucket keeps the remainder of the call wall, extended
    # through direction materialization (copy/scatter/finite/residual
    # gates) so every wall fraction of the direction is attributed.
    copyto!(base.dx, predictor_candidate.dx)
    copyto!(base.dy, predictor_candidate.dy)
    copyto!(base.ds, predictor_candidate.ds)
    base.dtau = predictor_candidate.dtau
    base.dkappa = predictor_candidate.dkappa
    _product_hsd_core_scatter!(state)
    _hsd_direction_finite(base) || begin
        state.diagnostic = fixed_trace ?
            :disjoint_fixed_head_q3_predictor_nonfinite :
            :symmetric_core_predictor_nonfinite
        return false
    end
    if !_product_hsd_newton_residual_ok(state, predictor_scalar)
        state.diagnostic = fixed_trace ?
            :disjoint_fixed_head_q3_predictor_residual_failed :
            :symmetric_core_predictor_residual_failed
        return false
    end
    timings.predictor_linear_solve_seconds +=
        Float64(time_ns() - t0) * 1.0e-9 -
        (timings.refinement_seconds - refinement_wall0)
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
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
    copyto!(base.dx, corrector_candidate.dx)
    copyto!(base.dy, corrector_candidate.dy)
    copyto!(base.ds, corrector_candidate.ds)
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
        copyto!(base.dx, base.dx_a)
        copyto!(base.dy, base.dy_a)
        copyto!(base.ds, base.ds_a)
        base.dtau = base.dtau_a
        base.dkappa = base.dkappa_a
        _product_hsd_core_scatter!(state)
        state.diagnostic = :corrector_fallback_to_predictor
        return true
    end
    timings.corrector_linear_solve_seconds +=
        Float64(time_ns() - t0) * 1.0e-9 -
        (timings.refinement_seconds - refinement_wall0)
    return true
end
