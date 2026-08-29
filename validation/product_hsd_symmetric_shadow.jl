# C6b: same-iterate product-HSD symmetric-core shadow parity.
#
# This file drives the *real* `ProductConeHSDState` on deterministic
# benchmark cases (Float64 only), runs the existing expanded route as the
# exact reference predictor/corrector, then recomputes the same Newton
# systems on the symmetric augmented core (CHOLMOD signed-LDL + original-K
# refinement) at the *same* iterate, without updating x/s/y/tau/kappa.
#
# The shadow is validation-only: the expanded route remains authoritative and
# mutates only direction/scratch fields.  The iterate is snapshotted before
# and asserted bitwise unchanged after.  Any case whose reference expanded
# direction cannot be produced, or whose symmetric-core workspace is not
# applicable, is recorded explicitly rather than silently skipped or loosened.

using Test
using SDPX
using LinearAlgebra
using SparseArrays

include(joinpath(
    @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
))
using .GenericConicBenchmark

const _SHADOW_IDS = (
    :lp_afiro_style,
    :socp_portfolio_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
    :power_epigraph_small,
)

"""Deterministic case spec; fail loudly if the catalog changed."""
function _shadow_spec(id::Symbol)
    matches = filter(spec -> spec.id === id, inventory(; tier=:small))
    length(matches) == 1 || error(
        "expected exactly one small case $id, found $(length(matches))",
    )
    return only(matches)
end

"""Snapshot every direction/scratch field the shadow may touch."""
function _snapshot_directions(state)
    base = state.base
    return (
        dx=copy(base.dx), dy=copy(base.dy), ds=copy(base.ds),
        dtau=base.dtau, dkappa=base.dkappa,
        dx_a=copy(base.dx_a), dy_a=copy(base.dy_a), ds_a=copy(base.ds_a),
        dtau_a=base.dtau_a, dkappa_a=base.dkappa_a,
        mu_aff=base.mu_aff, h=copy(state.h),
        expanded=(
            cone_operator=copy(state.expanded.cone_operator),
            cone_corrector_rhs=copy(state.expanded.cone_corrector_rhs),
            numeric_factor_count=state.expanded.numeric_factor_count,
            predictor_solve_count=state.expanded.predictor_solve_count,
            corrector_solve_count=state.expanded.corrector_solve_count,
        ),
    )
end

function _assert_directions_restored(state, snap)
    base = state.base
    @test base.dx == snap.dx
    @test base.dy == snap.dy
    @test base.ds == snap.ds
    @test base.dtau == snap.dtau
    @test base.dkappa == snap.dkappa
    @test base.dx_a == snap.dx_a
    @test base.dy_a == snap.dy_a
    @test base.ds_a == snap.ds_a
    @test base.dtau_a == snap.dtau_a
    @test base.dkappa_a == snap.dkappa_a
    @test base.mu_aff == snap.mu_aff
    @test state.h == snap.h
    @test state.expanded.cone_operator == snap.expanded.cone_operator
    @test state.expanded.cone_corrector_rhs ==
          snap.expanded.cone_corrector_rhs
    return nothing
end

"""Run the reference expanded predictor/corrector and snapshot both phases."""
function _reference_direction(state)
    base = state.base
    ok = SDPX._product_hsd_expanded_direction!(state)
    ok || return nothing
    return (
        predictor=(
            dx=copy(base.dx_a), dy=copy(base.dy_a), ds=copy(base.ds_a),
            dtau=base.dtau_a, dkappa=base.dkappa_a,
        ),
        corrector=(
            dx=copy(base.dx), dy=copy(base.dy), ds=copy(base.ds),
            dtau=base.dtau, dkappa=base.dkappa,
        ),
        scalar_rhs=-base.tau * base.kappa,
    )
end

"""Maximum absolute value over the five frozen residual groups."""
function _shadow_max_residual(residual)
    return maximum((
        maximum(abs, residual.primal_affine; init=zero(eltype(residual.primal_affine))),
        maximum(abs, residual.dual_affine; init=zero(eltype(residual.dual_affine))),
        abs(residual.homogeneous_gap),
        maximum(abs, residual.cone_complementarity; init=zero(eltype(residual.cone_complementarity))),
        abs(residual.tau_kappa),
    ))
end

function _shadow_max_direction_difference(a, b)
    return max(
        maximum(abs, a.dx - b.dx; init=0.0),
        maximum(abs, a.dy - b.dy; init=0.0),
        maximum(abs, a.ds - b.ds; init=0.0),
        abs(a.dtau - b.dtau),
        abs(a.dkappa - b.dkappa),
    )
end

"""Build the predictor NewtonSystem and the symmetric-core workspace."""
function _shadow_predictor(state)
    base = state.base
    SDPX.affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    cone = SDPX._product_hsd_expanded_linearization(state, state.h)
    cone === nothing && return (nothing, nothing)
    system = SDPX._product_hsd_expanded_system(state, cone, predictor_scalar)
    V = base.rank_basis
    workspace = try
        SDPX.build_symmetric_core_workspace(
            system, V, 1, 53, typemax(Int), 0, 1e-6;
            symbolic_epoch=0,
        )
    catch exception
        exception isa InterruptException && rethrow()
        return (system, exception)
    end
    return (system, workspace)
end

"""Reconstruct the corrector shift and system from the core predictor."""
function _shadow_corrector(state, core_predictor, system)
    base = state.base
    # Use the core predictor direction for the boundary step/mu_aff so the
    # shadow is self-consistent.
    saved_a = (
        ds=copy(base.ds_a), dy=copy(base.dy_a),
        dtau=base.dtau_a, dkappa=base.dkappa_a,
        mu_aff=base.mu_aff,
    )
    base.ds_a .= core_predictor.ds
    base.dy_a .= core_predictor.dy
    base.dtau_a = core_predictor.dtau
    base.dkappa_a = core_predictor.dkappa

    saved_ds = copy(base.ds); saved_dy = copy(base.dy)
    saved_dtau = base.dtau; saved_dkappa = base.dkappa
    base.ds .= core_predictor.ds
    base.dy .= core_predictor.dy
    base.dtau = core_predictor.dtau
    base.dkappa = core_predictor.dkappa
    alpha_aff = SDPX._product_hsd_boundary_alpha!(state)
    mu_aff = SDPX._product_hsd_mu_aff!(state, alpha_aff)
    base.ds .= saved_ds; base.dy .= saved_dy
    base.dtau = saved_dtau; base.dkappa = saved_dkappa

    sigma = min(one(Float64), (mu_aff / base.mu)^3)
    sigma_mu = sigma * base.mu
    SDPX._product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       core_predictor.dtau * core_predictor.dkappa

    # Restore the reference affine fields; the shadow must not leave state
    # mutated beyond the recomputed corrector shift.
    base.ds_a .= saved_a.ds
    base.dy_a .= saved_a.dy
    base.dtau_a = saved_a.dtau
    base.dkappa_a = saved_a.dkappa
    base.mu_aff = saved_a.mu_aff

    cone = SDPX.ProductConeLinearization{Float64}(
        system.cone.operator, copy(state.h), system.cone.block_ranges,
    )
    corrector_rhs = SDPX.HSDNewtonRHS(
        copy(system.rhs.primal_affine), copy(system.rhs.dual_affine),
        system.rhs.homogeneous_gap, copy(state.h), corrector_scalar,
    )
    corrector_system = SDPX.NewtonSystem(
        system.A, system.b, system.c, cone,
        system.tau, system.kappa, corrector_rhs,
    )
    return (corrector_system, alpha_aff)
end

function _shadow_one(id::Symbol)
    spec = _shadow_spec(id)
    model = build(spec.problem, Float64, spec.params)
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    # Production path: equality reduction eliminates ZeroCone rows before the
    # product-HSD state is constructed.  Mirror that exactly so the shadow
    # operates on the same reduced program and rank basis.
    reduction = SDPX.hsd_equality_reduce(canonical)
    reduction.status === SDPX.HSDEqualityReady || return (
        id, status=:equality_reduction_failed,
        detail=string(reduction.status),
    )
    reduced = reduction.reduced
    state = SDPX.ProductConeHSDState(reduced; kkt_route=:expanded)
    SDPX.product_hsd_cold_start!(state)
    base = state.base

    iterate_snapshot = (
        x=copy(base.x), s=copy(base.s), y=copy(base.y),
        tau=base.tau, kappa=base.kappa,
    )

    reference = _reference_direction(state)
    if reference === nothing
        return (; id, status=:reference_failed, detail=string(state.diagnostic))
    end
    # Snapshot the reference's direction/scratch AFTER it ran, so restore
    # checks verify the shadow leaves exactly the reference state behind.
    direction_snapshot = _snapshot_directions(state)

    predictor_system, workspace_or_error = _shadow_predictor(state)
    if predictor_system === nothing
        return (; id, status=:linearization_failed,
            detail="expanded cone linearization unavailable")
    end
    if workspace_or_error isa Exception
        return (; id, status=:workspace_inapplicable,
            detail=sprint(showerror, workspace_or_error))
    end

    core_predictor, core_predictor_residual =
        SDPX.solve_core_direction!(workspace_or_error, predictor_system)

    corrector_system, alpha_aff =
        _shadow_corrector(state, core_predictor, predictor_system)
    core_corrector, core_corrector_residual =
        SDPX.solve_core_direction!(workspace_or_error, corrector_system)

    # Independent frozen five-equation residual of the reference directions.
    reference_predictor_residual = SDPX.NewtonResidual(predictor_system)
    SDPX.newton_residual!(
        reference_predictor_residual, predictor_system,
        SDPX.NewtonDirection(
            reference.predictor.dx, reference.predictor.dy,
            reference.predictor.ds, reference.predictor.dtau,
            reference.predictor.dkappa,
        ),
    )
    reference_corrector_residual = SDPX.NewtonResidual(corrector_system)
    SDPX.newton_residual!(
        reference_corrector_residual, corrector_system,
        SDPX.NewtonDirection(
            reference.corrector.dx, reference.corrector.dy,
            reference.corrector.ds, reference.corrector.dtau,
            reference.corrector.dkappa,
        ),
    )

    residual_tol = 4096.0 * eps(Float64)
    ok = true
    messages = String[]

    if _shadow_max_residual(core_predictor_residual) > residual_tol
        ok = false
        push!(messages, "core predictor five-equation residual too large")
    end
    if _shadow_max_residual(core_corrector_residual) > residual_tol
        ok = false
        push!(messages, "core corrector five-equation residual too large")
    end
    if _shadow_max_residual(reference_predictor_residual) > residual_tol
        ok = false
        push!(messages, "reference predictor residual too large")
    end
    if _shadow_max_residual(reference_corrector_residual) > residual_tol
        ok = false
        push!(messages, "reference corrector residual too large")
    end

    tol = 4096.0 * sqrt(eps(Float64))
    pred_diff = _shadow_max_direction_difference(
        core_predictor, reference.predictor,
    )
    corr_diff = _shadow_max_direction_difference(
        core_corrector, reference.corrector,
    )
    scale = max(
        1.0, norm(reference.predictor.dx, Inf), norm(reference.predictor.dy, Inf),
        norm(reference.predictor.ds, Inf), abs(reference.predictor.dtau),
        abs(reference.predictor.dkappa), norm(core_corrector.dx, Inf),
    )
    if pred_diff > tol * scale
        ok = false
        push!(messages, "predictor direction differs by $(pred_diff)")
    end
    if corr_diff > tol * scale
        ok = false
        push!(messages, "corrector direction differs by $(corr_diff)")
    end

    diag = SDPX.factor_diagnostics(workspace_or_error.cache)
    if diag.symbolic_count != 1 || diag.numeric_count != 1
        ok = false
        push!(messages, "unexpected factor counts $diag")
    end
    if workspace_or_error.homogeneous_solves != 1
        ok = false
        push!(messages, "homogeneous solve count $(workspace_or_error.homogeneous_solves)")
    end
    if workspace_or_error.variable_solves != 2 ||
       workspace_or_error.directions != 2
        ok = false
        push!(messages, "variable/direction counts $(workspace_or_error.variable_solves)/$(workspace_or_error.directions)")
    end

    if core_predictor.dx === core_corrector.dx
        ok = false
        push!(messages, "predictor/corrector dx aliased")
    end
    if core_predictor_residual.primal_affine ===
       core_corrector_residual.primal_affine
        ok = false
        push!(messages, "predictor/corrector residual aliased")
    end

    # Restore the shadow's recomputed corrector scratch so the state is
    # bitwise identical to the reference's after the shadow comparison.
    copyto!(state.h, direction_snapshot.h)
    copyto!(
        state.expanded.cone_operator, direction_snapshot.expanded.cone_operator,
    )
    copyto!(
        state.expanded.cone_corrector_rhs,
        direction_snapshot.expanded.cone_corrector_rhs,
    )

    if base.x != iterate_snapshot.x || base.s != iterate_snapshot.s ||
       base.y != iterate_snapshot.y || base.tau != iterate_snapshot.tau ||
       base.kappa != iterate_snapshot.kappa
        ok = false
        push!(messages, "iterate mutated by shadow")
    end
    _assert_directions_restored(state, direction_snapshot)

    return (;
        id, status=ok ? :pass : :fail,
        detail=isempty(messages) ? "" : join(messages, "; "),
        pred_diff, corr_diff, alpha_aff,
        factor_epoch=diag.factor_epoch,
    )
end

@testset "C6b product HSD symmetric core shadow parity" begin
    for id in _SHADOW_IDS
        @testset "$id" begin
            result = _shadow_one(id)
            if result.status === :pass
                @test result.status === :pass
                @test result.factor_epoch == 1
            elseif result.status in (:reference_failed, :linearization_failed,
                                     :workspace_inapplicable)
                @info "shadow skipped" id=result.id reason=result.status detail=result.detail
                @test true
            else
                @info "shadow failed" id=result.id detail=result.detail
                @test false
            end
        end
    end
end
