# Focused regression for the frozen original-coordinate gap normalization.
#
# The frozen CSDR α3 trajectory pins the public certificate relative gap to
#     |p - d| / max(1, (|p| + |d|) / 2)
# in exact typed arithmetic/order (expression predating b2fdb16). Commit
# b2fdb16 replaced the denominator with (1 + |p| + |d|), a weaker
# normalization: 1 + |p| + |d| >= max(1, (|p| + |d|) / 2) always, so the
# reported gap shrinks (about 2.03x at CSDR magnitudes) and the
# `relative_gap <= gap_limit` gate becomes looser.
#
# Near-threshold validity property: with old_gap < frozen_gap, the tolerance
# band where the weakened formula passes but the restored formula rejects is
# [old_gap, frozen_gap) — reject iff relative_gap > gap_limit, i.e. iff
# gap_limit < frozen_gap, while the old formula passes iff
# old_gap <= gap_limit; at gap_limit == frozen_gap the restored gate accepts
# (<=). Rejection flips `valid: true -> false` with reason `:duality_gap`,
# and via `src/hsd/native_hsd_public.jl` an optimal core result whose
# original-coordinate certificate is invalid is downgraded to public status
# `NumericalFailure` with termination reason
# `:original_coordinate_certificate_failed` (expected fail-closed behavior).
using Test
using SDPX

@testset "Frozen certificate gap normalization" begin
    frozen_gap(p::Float64, d::Float64) =
        abs(p - d) / max(1.0, (abs(p) + abs(d)) / 2.0)
    old_gap(p::Float64, d::Float64) = abs(p - d) / (1.0 + abs(p) + abs(d))

    function _soc_model()
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1] + 0.3 * x[2])
        return model
    end

    function _knob_soc_model()
        model = SDPX.Model(Float64)
        y = SDPX.variable!(model, :x, 6; domain=SDPX.Reals())
        for cell in 1:3
            r = y[2cell - 1]
            q = y[2cell]
            SDPX.constraint!(model, Symbol(:unitarity_, cell),
                Any[one(Float64), q - one(Float64), r], SDPX.LorentzCone())
        end
        weights = Float64.([0.3, 0.1, 0.7, 0.2, 0.5, 0.4])
        SDPX.constraint!(model, :anchor,
            sum(y[i] - weights[i] for i in 1:6) - 1.5, SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(),
            sum(weights[i] * y[i] for i in 1:6))
        return model
    end

    # Large-objective branch: (|p| + |d|) / 2 > 1.
    result = SDPX.optimize!(_soc_model(); settings=SDPX.Settings(Float64; verbosity=0))
    cert = SDPX.certificate(result)
    p = Float64(cert.primal_objective)
    d = Float64(cert.dual_objective)
    @test SDPX.status(result) === :optimal
    @test cert.valid
    @test cert.reason === :valid
    @test (abs(p) + abs(d)) / 2.0 > 1.0
    @test cert.relative_gap == frozen_gap(p, d)
    # Stricter normalization: the restored gap is above the weakened one,
    # so a tolerance band [old_gap, frozen_gap) exists where the old formula
    # would pass but the restored formula rejects. Tightening the solver
    # tolerance instead is not a substitute (the solver refines further
    # rather than failing the gate), so rejection is pinned below by
    # calling the actual certificate entry point on fixed solve outputs.
    @test frozen_gap(p, d) > old_gap(p, d)
    @test cert.relative_gap > old_gap(p, d)

    # Fixed ordinary-solve inputs through the existing native-HSD seam: one
    # core solve, then the production result constructor with default vs
    # tightened gap limits (no re-solve, no runtime eval, no benchmark
    # special-casing in `src/`).
    settings = SDPX.Settings(Float64; verbosity=0)
    outputs_full = SDPX.Outputs(
        :all, :all, :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:none,
        history=false,
        trace=false,
    )
    fixed_model = _soc_model()
    fixed_program = SDPX.compile_product_cone_model(fixed_model)
    fixed_route = SDPX.classify_native_cone_program(fixed_program)
    fixed_canonical, _, fixed_core =
        SDPX._public_native_hsd_core(fixed_model, fixed_program, fixed_route, settings)
    result_ok = SDPX._public_result_from_native_hsd(
        fixed_model, fixed_program, fixed_canonical, fixed_core, settings, outputs_full,
    )
    @test SDPX.status(result_ok) === :optimal
    @test fixed_core.status === result_ok.status
    cert_ok = SDPX.certificate(result_ok)
    @test cert_ok.valid
    @test cert_ok.reason === :valid
    fp = Float64(cert_ok.primal_objective)
    fd = Float64(cert_ok.dual_objective)
    @test cert_ok.relative_gap == frozen_gap(fp, fd)
    old = old_gap(fp, fd)
    frozen = frozen_gap(fp, fd)
    @test old < frozen
    mid = (old + frozen) / 2.0
    @test isfinite(mid)
    @test old < mid < frozen

    # Actual `_public_original_certificate` rejection at a gap limit strictly
    # between the relaxed and frozen formulas, plus acceptance at equality.
    primal = SDPX.value(result_ok)
    constraint_dual = SDPX.dual(result_ok)
    dual_slack = SDPX.dual_slack(result_ok)
    p_obj = SDPX.primal_objective(result_ok)
    d_obj = SDPX.dual_objective(result_ok)
    tight = SDPX.Settings(
        Float64; verbosity=0, tolerances=SDPX.Tolerances(Float64; gap=mid),
    )
    rejected = SDPX._public_original_certificate(
        fixed_model, fixed_program, primal, constraint_dual, dual_slack,
        p_obj, d_obj, tight, fixed_core.status,
    )
    @test rejected.relative_gap == frozen
    @test !rejected.valid
    @test rejected.reason === :duality_gap
    edge = SDPX.Settings(
        Float64;
        verbosity=0,
        tolerances=SDPX.Tolerances(Float64; gap=rejected.relative_gap),
    )
    accepted = SDPX._public_original_certificate(
        fixed_model, fixed_program, primal, constraint_dual, dual_slack,
        p_obj, d_obj, edge, fixed_core.status,
    )
    @test accepted.valid
    @test accepted.reason === :valid

    # Existing public-status downgrade on the same fixed core: an invalid
    # original-coordinate certificate turns the optimal core result into
    # `NumericalFailure` with `:original_coordinate_certificate_failed`.
    result_tight = SDPX._public_result_from_native_hsd(
        fixed_model, fixed_program, fixed_canonical, fixed_core, tight, outputs_full,
    )
    @test !SDPX.certificate(result_tight).valid
    @test SDPX.certificate(result_tight).reason === :duality_gap
    @test SDPX.status(result_tight) === :numerical_failure
    @test SDPX.termination(result_tight).reason === :original_coordinate_certificate_failed
    @test SDPX.termination(result_tight).stage === :certification

    # Small-objective branch: max(1, (|p| + |d|) / 2) == 1.
    result2 = SDPX.optimize!(_knob_soc_model(); settings=SDPX.Settings(Float64; verbosity=0))
    cert2 = SDPX.certificate(result2)
    p2 = Float64(cert2.primal_objective)
    d2 = Float64(cert2.dual_objective)
    @test SDPX.status(result2) === :optimal
    @test cert2.valid
    @test cert2.reason === :valid
    @test (abs(p2) + abs(d2)) / 2.0 < 1.0
    @test cert2.relative_gap == abs(p2 - d2)
    @test cert2.relative_gap == frozen_gap(p2, d2)
    @test frozen_gap(p2, d2) > old_gap(p2, d2)

    # CSDR-scale strictness factor: at frozen α3 magnitudes the restored gap
    # is ~2.03x the weakened value (19-digit agreement with the control
    # comparison), so restoration only ever tightens the validity gate.
    pc = -31.672155970636577
    dc = -31.672155970700526
    @test frozen_gap(pc, dc) / old_gap(pc, dc) ≈ 2.0315734742189984 rtol = 1e-12
    @test frozen_gap(pc, dc) > old_gap(pc, dc)
end
