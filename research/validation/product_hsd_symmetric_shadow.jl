# C6b: same-iterate product-HSD symmetric-core shadow parity.
#
# This file drives the *real* `ProductConeHSDState` on deterministic
# benchmark cases (Float64 only) and replays the exact production expanded
# predictor/corrector sequence step by step.  The reference stage returns the
# exact predictor and corrector `NewtonSystem`s together with the exact
# reference boundary/sigma/scalar facts.  The symmetric augmented core then
# solves exactly those two systems at the same iterate, without updating
# x/s/y/tau/kappa and without rebuilding a different corrector system.
#
# The shadow is validation-only: the expanded route remains authoritative.
# Every catalog case is mandatory: any equality/reference/linearization/
# workspace failure fails the test with a diagnostic; nothing is skipped.
# After the shadow the reference state (iterate, direction/scratch arrays,
# runtime mutable state, expanded cone buffers/counters) must be exactly
# restored.

using Test
using SDPX
using LinearAlgebra
using SparseArrays
using Serialization

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


"""Return an ownership-independent copy of a semantic Newton system.

`_product_hsd_expanded_system` reuses `state.expanded` session buffers for the
negated residuals and the cone operator/RHS.  The corrector stage of the
reference rewrites those buffers, which would otherwise corrupt the predictor
system consumed by the shadow.  This copy freezes every cone/RHS vector used
by `solve_core_direction!` so the shadow and the reference residual use
independent, unchanged data.
"""
function _own_system(system::SDPX.NewtonSystem{T}) where {T}
    cone = SDPX.ProductConeLinearization{T}(
        copy(system.cone.operator), copy(system.cone.corrector_rhs),
        UnitRange{Int}[rows for rows in system.cone.block_ranges],
    )
    rhs = SDPX.HSDNewtonRHS(
        copy(system.rhs.primal_affine), copy(system.rhs.dual_affine),
        system.rhs.homogeneous_gap, copy(system.rhs.cone_corrector),
        system.rhs.tau_kappa,
    )
    return SDPX.NewtonSystem(
        system.A, system.b, system.c, cone,
        system.tau, system.kappa, rhs,
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

function _serialize_value(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end

"""Maximum absolute difference over the five direction components."""
function _shadow_max_direction_difference(a, b)
    return max(
        maximum(abs, a.dx - b.dx; init=0.0),
        maximum(abs, a.dy - b.dy; init=0.0),
        maximum(abs, a.ds - b.ds; init=0.0),
        abs(a.dtau - b.dtau),
        abs(a.dkappa - b.dkappa),
    )
end

"""
    _reference_direction(state)

Replay the exact production expanded predictor/corrector sequence one step at
a time and return the two exact `NewtonSystem`s plus every reference scalar
fact:

    (predictor_system, corrector_system, facts)

`facts` carries `predictor`/`corrector` `NewtonDirection` snapshots,
`alpha_aff`, `mu_aff`, `sigma`, `sigma_mu`, `predictor_scalar`,
`corrector_scalar`.  This must stay in lock-step with
`_product_hsd_expanded_direction!`; the shadow consumes the returned systems
unchanged.
"""
function _reference_direction(state)
    base = state.base
    SDPX.affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    cone = SDPX._product_hsd_expanded_linearization(state, state.h)
    cone === nothing && return nothing
    predictor_system = SDPX._product_hsd_expanded_system(
        state, cone, predictor_scalar,
    )
    SDPX._product_hsd_factor_expanded!(state, predictor_system) || return nothing
    SDPX._product_hsd_expanded_solve_shift!(
        state, cone, predictor_scalar; stage=:predictor,
    ) || return nothing
    predictor = SDPX.NewtonDirection(
        copy(base.dx), copy(base.dy), copy(base.ds),
        base.dtau, base.dkappa,
    )
    # Freeze the predictor system before the corrector stage mutates the
    # shared session buffers.
    predictor_system = _own_system(predictor_system)
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = SDPX._product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(Float64)) || return nothing
    mu_aff = SDPX._product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(Float64)) || return nothing
    ratio = base.mu_aff / base.mu
    sigma = min(one(Float64), ratio * ratio * ratio)
    sigma_mu = sigma * base.mu
    SDPX._product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    copyto!(state.expanded.cone_corrector_rhs, state.h)
    corrector_cone = SDPX.ProductConeLinearization{Float64}(
        state.expanded.cone_operator, state.expanded.cone_corrector_rhs,
        state.expanded.cone_block_ranges,
    )
    corrector_system = SDPX._product_hsd_expanded_system(
        state, corrector_cone, corrector_scalar,
    )
    SDPX._product_hsd_expanded_solve_shift!(
        state, corrector_cone, corrector_scalar; stage=:corrector,
    ) || return nothing
    corrector_system = _own_system(corrector_system)
    corrector = SDPX.NewtonDirection(
        copy(base.dx), copy(base.dy), copy(base.ds),
        base.dtau, base.dkappa,
    )
    facts = (
        predictor=predictor,
        corrector=corrector,
        alpha_aff=alpha_aff,
        mu_aff=mu_aff,
        sigma=sigma,
        sigma_mu=sigma_mu,
        predictor_scalar=predictor_scalar,
        corrector_scalar=corrector_scalar,
    )
    return (predictor_system, corrector_system, facts)
end

"""Collect every reference state field the shadow must leave untouched."""
function _reference_state_snapshot(state)
    base = state.base
    return (
        iterate=(x=copy(base.x), s=copy(base.s), y=copy(base.y),
                 tau=base.tau, kappa=base.kappa),
        directions=(dx=copy(base.dx), dy=copy(base.dy), ds=copy(base.ds),
                    dtau=base.dtau, dkappa=base.dkappa,
                    dx_a=copy(base.dx_a), dy_a=copy(base.dy_a),
                    ds_a=copy(base.ds_a), dtau_a=base.dtau_a,
                    dkappa_a=base.dkappa_a, mu_aff=base.mu_aff),
        h=copy(state.h),
        scratch=(g_input=copy(state.g_input), g_output=copy(state.g_output),
                 gb=copy(state.gb), ds_hat=copy(state.ds_hat),
                 dy_hat=copy(state.dy_hat),
                 soc_g=copy(state.soc_g_error_bound),
                 soc_rt=copy(state.soc_roundtrip_bound),
                 cert_g=copy(state.certified_soc_g_error_bound),
                 cert_rt=copy(state.certified_soc_roundtrip_bound),
                 soc_bounds=state.soc_bounds_certified,
                 ns_metrics=copy(state.ns_metrics),
                 ns_H=copy(state.ns_H), ns_atgb=copy(state.ns_at_g_b),
                 ns_btga=copy(state.ns_bt_g_a),
                 ns_atgr=copy(state.ns_at_g_rhs),
                 ns_zero=copy(state.ns_zero_rhs)),
        expanded=(cone_operator=copy(state.expanded.cone_operator),
                  cone_rhs=copy(state.expanded.cone_corrector_rhs),
                  numeric=state.expanded.numeric_factor_count,
                  predictor_solves=state.expanded.predictor_solve_count,
                  corrector_solves=state.expanded.corrector_solve_count),
        runtime_ref=state.runtime,
        runtime_snapshot=deepcopy(state.runtime),
    )
end

function _assert_reference_state_unchanged(state, snap)
    base = state.base
    @test base.x == snap.iterate.x
    @test base.s == snap.iterate.s
    @test base.y == snap.iterate.y
    @test base.tau == snap.iterate.tau
    @test base.kappa == snap.iterate.kappa
    @test base.dx == snap.directions.dx
    @test base.dy == snap.directions.dy
    @test base.ds == snap.directions.ds
    @test base.dtau == snap.directions.dtau
    @test base.dkappa == snap.directions.dkappa
    @test base.dx_a == snap.directions.dx_a
    @test base.dy_a == snap.directions.dy_a
    @test base.ds_a == snap.directions.ds_a
    @test base.dtau_a == snap.directions.dtau_a
    @test base.dkappa_a == snap.directions.dkappa_a
    @test base.mu_aff == snap.directions.mu_aff
    @test state.h == snap.h
    @test state.g_input == snap.scratch.g_input
    @test state.g_output == snap.scratch.g_output
    @test state.gb == snap.scratch.gb
    @test state.ds_hat == snap.scratch.ds_hat
    @test state.dy_hat == snap.scratch.dy_hat
    @test state.soc_g_error_bound == snap.scratch.soc_g
    @test state.soc_roundtrip_bound == snap.scratch.soc_rt
    @test state.certified_soc_g_error_bound == snap.scratch.cert_g
    @test state.certified_soc_roundtrip_bound == snap.scratch.cert_rt
    @test state.soc_bounds_certified == snap.scratch.soc_bounds
    @test state.ns_metrics == snap.scratch.ns_metrics
    @test state.ns_H == snap.scratch.ns_H
    @test state.ns_at_g_b == snap.scratch.ns_atgb
    @test state.ns_bt_g_a == snap.scratch.ns_btga
    @test state.ns_at_g_rhs == snap.scratch.ns_atgr
    @test state.ns_zero_rhs == snap.scratch.ns_zero
    @test state.expanded.cone_operator == snap.expanded.cone_operator
    @test state.expanded.cone_corrector_rhs == snap.expanded.cone_rhs
    @test state.expanded.numeric_factor_count == snap.expanded.numeric
    @test state.expanded.predictor_solve_count == snap.expanded.predictor_solves
    @test state.expanded.corrector_solve_count == snap.expanded.corrector_solves
    @test state.runtime === snap.runtime_ref
    # Runtime value equality: the restored object must serialize identically
    # to the pre-shadow deep snapshot (ownership-safe value comparison).
    @test _serialize_value(state.runtime) == _serialize_value(snap.runtime_snapshot)
    return nothing
end

"""Direction scale used by the parity bound."""
function _shadow_scale(facts, core_predictor)
    return max(
        1.0, norm(facts.predictor.dx, Inf), norm(facts.predictor.dy, Inf),
        norm(facts.predictor.ds, Inf), abs(facts.predictor.dtau),
        abs(facts.predictor.dkappa),
        norm(facts.corrector.dx, Inf), norm(facts.corrector.dy, Inf),
        norm(facts.corrector.ds, Inf), abs(facts.corrector.dtau),
        abs(facts.corrector.dkappa),
        norm(core_predictor.dx, Inf), norm(core_predictor.dy, Inf),
        norm(core_predictor.ds, Inf), abs(core_predictor.dtau),
        abs(core_predictor.dkappa),
    )
end

function _shadow_one(id::Symbol)
    spec = _shadow_spec(id)
    model = build(spec.problem, Float64, spec.params)
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    reduction = SDPX.hsd_equality_reduce(canonical)
    reduction.status === SDPX.HSDEqualityReady || return (
        id=id, status=:equality_reduction_failed,
        detail=string(reduction.status),
    )
    state = SDPX.ProductConeHSDState(reduction.reduced; kkt_route=:expanded)
    SDPX.product_hsd_cold_start!(state)

    reference = _reference_direction(state)
    reference === nothing && return (
        id=id, status=:reference_failed,
        detail="expanded predictor/corrector reference unavailable",
    )
    (predictor_system, corrector_system, facts) = reference
    snap = _reference_state_snapshot(state)

    # Shadow: solve the exact reference systems on the symmetric core.  The
    # core boundary/centering facts are computed on a *copied* runtime so the
    # reference runtime object is never touched and needs no field restore.
    V = state.base.rank_basis
    workspace = SDPX.build_symmetric_core_workspace(
        predictor_system, V, 1, 53, typemax(Int), 0, 1e-6;
        symbolic_epoch=0,
    )

    core_predictor, core_predictor_residual =
        SDPX.solve_core_direction!(workspace, predictor_system)

    # Core predictor boundary/centering facts, staged into a runtime copy.
    shadow_runtime = deepcopy(snap.runtime_ref)
    state.runtime = shadow_runtime
    base = state.base
    saved = (
        ds=copy(base.ds), dy=copy(base.dy), dtau=base.dtau, dkappa=base.dkappa,
        mu_aff=base.mu_aff,
    )
    base.ds .= core_predictor.ds
    base.dy .= core_predictor.dy
    base.dtau = core_predictor.dtau
    base.dkappa = core_predictor.dkappa
    alpha_core = SDPX._product_hsd_boundary_alpha!(state)
    (isfinite(alpha_core) && alpha_core > zero(Float64)) ||
        return (id=id, status=:shadow_boundary_failed,
                detail="core predictor boundary alpha unavailable")
    mu_aff_core = SDPX._product_hsd_mu_aff!(state, alpha_core)
    base.ds .= saved.ds
    base.dy .= saved.dy
    base.dtau = saved.dtau
    base.dkappa = saved.dkappa
    base.mu_aff = saved.mu_aff
    # Restore the original runtime object; the copy is discarded.
    state.runtime = snap.runtime_ref

    core_corrector, core_corrector_residual =
        SDPX.solve_core_direction!(workspace, corrector_system)

    # Exact reference five-equation residuals on the exact reference systems.
    reference_predictor_residual = SDPX.NewtonResidual(predictor_system)
    SDPX.newton_residual!(
        reference_predictor_residual, predictor_system, facts.predictor,
    )
    reference_corrector_residual = SDPX.NewtonResidual(corrector_system)
    SDPX.newton_residual!(
        reference_corrector_residual, corrector_system, facts.corrector,
    )

    ok = true
    messages = String[]
    residual_tol = 4096.0 * eps(Float64)
    for (name, value) in (
        ("core predictor", _shadow_max_residual(core_predictor_residual)),
        ("core corrector", _shadow_max_residual(core_corrector_residual)),
        ("reference predictor", _shadow_max_residual(reference_predictor_residual)),
        ("reference corrector", _shadow_max_residual(reference_corrector_residual)),
    )
        if value > residual_tol
            ok = false
            push!(messages, "$name five-equation residual $value > $residual_tol")
        end
    end

    scale = _shadow_scale(facts, core_predictor)
    parity_tol = 1e-9 * scale
    pred_diff = _shadow_max_direction_difference(core_predictor, facts.predictor)
    corr_diff = _shadow_max_direction_difference(core_corrector, facts.corrector)
    if pred_diff > parity_tol
        ok = false
        push!(messages, "predictor direction differs by $pred_diff > $parity_tol")
    end
    if corr_diff > parity_tol
        ok = false
        push!(messages, "corrector direction differs by $corr_diff > $parity_tol")
    end

    # Scalar/fact parity: the core predictor must reproduce the reference
    # boundary and centering facts within the same tight bound.
    fact_scale = max(
        1.0, abs(facts.alpha_aff), abs(facts.mu_aff), abs(facts.sigma),
        abs(facts.sigma_mu), abs(facts.predictor_scalar),
        abs(facts.corrector_scalar), abs(mu_aff_core),
    )
    fact_tol = 1e-9 * fact_scale
    if abs(alpha_core - facts.alpha_aff) > fact_tol
        ok = false
        push!(messages, "alpha differs $alpha_core vs $(facts.alpha_aff)")
    end
    if abs(mu_aff_core - facts.mu_aff) > fact_tol
        ok = false
        push!(messages, "mu_aff differs $mu_aff_core vs $(facts.mu_aff)")
    end
    if abs(facts.sigma - min(one(Float64), (mu_aff_core / base.mu)^3)) > fact_tol
        ok = false
        push!(messages, "sigma inconsistent with core mu_aff")
    end

    diag = SDPX.factor_diagnostics(workspace.cache)
    if diag.symbolic_count != 1 || diag.numeric_count != 1
        ok = false
        push!(messages, "unexpected factor counts $diag")
    end
    if workspace.homogeneous_solves != 1
        ok = false
        push!(messages, "homogeneous solve count $(workspace.homogeneous_solves)")
    end
    if workspace.variable_solves != 2 || workspace.directions != 2
        ok = false
        push!(messages, "variable/direction counts $(workspace.variable_solves)/$(workspace.directions)")
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

    # Every reference field must be bitwise/value identical after the shadow.
    _assert_reference_state_unchanged(state, snap)

    return (
        id=id, status=ok ? :pass : :fail,
        detail=isempty(messages) ? "" : join(messages, "; "),
        pred_diff, corr_diff, alpha_aff=facts.alpha_aff, alpha_core,
        mu_aff_ref=facts.mu_aff, mu_aff_core,
        sigma=facts.sigma, sigma_mu=facts.sigma_mu,
        factor_epoch=diag.factor_epoch,
    )
end

@testset "C6b product HSD symmetric core shadow parity" begin
    for id in _SHADOW_IDS
        @testset "$id" begin
            result = _shadow_one(id)
            @info "C6b shadow evidence" result
            @test result.status === :pass
            @test result.factor_epoch == 1
        end
    end
end
