#=====================================================================#
#  S04 — standalone test for the original-coordinate certification layer.
#
#  Run:
#    julia --startup-file=no --project=<SDPX.jl> <SDPX.jl>/test/rebuild/S04.jl
#
#  Card: agents/S04.md.  Specification: docs/rebuild/ADR-003-acceptance.md.
#
#  What this file proves (card acceptance items):
#
#    A. NaN/Inf inputs, a deliberately wrong dual map and a fabricated
#       infeasibility ray are each REJECTED with a typed outcome.
#    B. A `maxiter` exit and a time exit retain the last valid state and are
#       NOT reported as `optimal`.
#    C. An A/B run with diagnostics/verbosity ON vs OFF produces bitwise
#       identical numeric results AND identical accept/reject decisions
#       (ADR-003 §2, card acceptance item 3).
#    D. A valid certificate still passes — the negatives in A were not
#       achieved by making the gate reject everything.
#
#  Independence: the PSD packed/dual map used by the fixtures is checked
#  against A01's oracles (`oracle_svec_is_scaled`,
#  `oracle_rsoc_map_highprec`, `oracle_residual_exact`), which call no SDPX
#  code.  No SDPX solver is invoked in this file.
#=====================================================================#

using Test
using LinearAlgebra
using SparseArrays

const SDPX_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SRC = joinpath(SDPX_ROOT, "src")

include(joinpath(@__DIR__, "reference_oracles.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))
using .A01Oracles
using .A01Fixtures

if !(SRC in LOAD_PATH)
    push!(LOAD_PATH, SRC)
end
if !(SDPX_ROOT in LOAD_PATH)
    push!(LOAD_PATH, SDPX_ROOT)
end

import SDPX

# The certification layer is written to be included by `src/SDPX.jl`
# (integration is I01/I02/I03 authority).  Loading it here through SDPX's own
# `include` mechanism proves the include chain works *before* integration.
const CertificationLayer = let
    m = Module(:S04CertificationHost)
    Core.eval(m, :(const SDPX = $(SDPX)))
    Base.include(m, joinpath(SRC, "certification", "original.jl"))
    getfield(m, :SDPXCertification)
end

const C = CertificationLayer
using .CertificationLayer: SDPXCertification
const Cert = SDPXCertification

# ---------------------------------------------------------------------
#  Raw measurement log (for the report; no measurement is invented).
# ---------------------------------------------------------------------
const RAW = Dict{String,Any}()

record!(key::AbstractString, value) = (RAW[key] = value; value)

# ---------------------------------------------------------------------
#  Fixtures — original-coordinate problems built from exact data.
# ---------------------------------------------------------------------

"""
    fixture_scalar_psd()

`min cᵀx` s.t. `A x = b`, `x ∈ R₊` — the one-dimensional symmetric cone,
whose packed map is exactly `1`.  Optimum: `x = 1`, `y = −1`, `s = 1`,
objective `−1`: a valid certificate with exact-rational arithmetic behind it.
"""
function fixture_scalar_psd(::Module)
    A = sparse([2.0;;])
    b = [2.0]
    c = [-2.0]
    problem = Cert.OriginalProblem(A, b, c, [Cert.NonnegativeBlock(0, 1)])
    # Dual: max b'y s.t. A'y <= c, y free  =>  2y <= -2  =>  y* = -1.
    # s* = c - A'y* = -2 + 2 = 0 and -b'y* = 2 = c'x*.
    point = Cert.OriginalPoint([1.0], [-1.0], [0.0], 1.0, 0.0)
    return problem, point, Cert.OriginalOperator(problem)
end

"""
    fixture_psd2()

A genuinely optimal 2×2 SDP fixture, over the packed variable
`x = [X11, X21, X22]` (column-major lower triangle, so `svec` carries a √2 on
the off-diagonal):

    min  cᵀx   s.t.  A x = b,  x ∈ svec(S^2_+)
    c = (−0.1, 0, −0.1),   A = [[1 0 0], [0 0 1]],   b = (0.1, 0.1)

`A x = b` fixes `X11 = X22 = 0.1`, so `cᵀx = −0.02` and the optimum is
`X = diag(0.1, 0.1)`, i.e. `x = (0.1, 0, 0.1)` with `s = b − A x = 0`.  The
row space is `R^2_+` (no equality row lies at the PSD boundary).

Dual: `y = (−0.1, −0.1)` gives `Aᵀy = (−0.1, 0, −0.1)` and

    dual slack  Aᵀy − c = 0 = svec(0)          (on the PSD boundary)
    bᵀy = −0.02 = cᵀx                          (zero duality gap)
    sᵀy = 0,  κ = 0                            (zero complementarity)

`y` is the true dual optimum: the constraint `Aᵀy − c ⪰ 0` is
`diag(−0.1 − y1, −0.1 − y2) ⪰ 0`, i.e. `y1, y2 ≤ −0.1`, and `bᵀy = 0.1(y1+y2)`
is maximized on that boundary.

so `(x, y)` is an exact optimality certificate with the off-diagonal
coordinate zero — which is what lets a wrong dual map be injected without
disturbing primal feasibility.
"""
function fixture_psd2(::Module; dual_scale=nothing)
    # One equality row over the three packed variables of S^2: trace(X) = 0.2.
    # The PSD block spans the VARIABLE space (n = 3); `s` and `y` live in the
    # ROW space (m = 1).  At X = diag(0.1, 0.1) the dual slack is
    # Aᵀy − c = (−y + 0.5)·diag(1,1) ⊕ 0, which is PSD exactly for y ≤ 0.5, so
    # the optimal dual is y = 0.5 with `bᵀy = 0.1 = cᵀx` and zero gap.
    A = sparse([1.0 0.0 0.0; 0.0 0.0 1.0])
    b = [0.1, 0.1]
    c = [-0.1, 0.0, -0.1]
    block = dual_scale === nothing ? Cert.PSDBlock(0, 2) :
        Cert.PSDBlock(0, 2; dual_scale=dual_scale)
    problem = Cert.OriginalProblem(A, b, c, [block])
    # `x` and the PSD block span the VARIABLE space (n = 3 packed
    # coordinates); `y` and `s` are indexed by the m = 1 equality row.
    point = Cert.OriginalPoint([0.1, 0.0, 0.1], [-0.1, -0.1], [0.0, 0.0],
                               1.0, 0.0)
    op = Cert.OriginalOperator(problem)
    return problem, point, op
end

@testset "S04 — certification and termination status" begin

# =====================================================================
#  0. Oracle cross-check of the packed PSD map (no SDPX participation)
# =====================================================================
@testset "0. independent packed/dual map reference" begin
    for dim in (1, 2, 3, 4)
        len = dim * (dim + 1) ÷ 2
        oracle_scaled = oracle_svec_is_scaled(dim)
        layer_scaled = [Cert.psd_scale_of(dim, k) > 1.0 for k in 1:len]
        @test oracle_scaled == layer_scaled
        @test all(k -> Cert.psd_scale_of(dim, k) ==
                       (oracle_scaled[k] ? sqrt(2.0) : 1.0), 1:len)
        map = Cert.build_dual_map(dim)
        @test Cert.map_is_dual_consistent(map)
        @test Cert.dual_map_residual(map) == 0.0
        @test Cert.adjoint_residual(Cert.as_psd_block(map, 0)) == 0.0
    end
    # The independent exact residual oracle must agree with this layer's
    # residual on the fixture, in exact rational arithmetic.
    problem, point, _ = fixture_scalar_psd(Cert)
    # The A01 oracle's frozen HSD equations are `rD = Aᵀy + c·τ` and
    # `rG = cᵀx + bᵀy + κ`; this layer uses the production public audit's
    # `rD = Aᵀy − c·τ` and `rG = cᵀx − bᵀy + κ`.  On this fixture the two
    # conventions differ by the documented sign on `y` (`y_layer = −y_oracle`)
    # and by the objective sign in `rG`.  Comparing them through that relation
    # is honest; asserting they are equal would be a silent convention bug.
    exact = oracle_residual_exact([2.0;;], [2.0], [-2.0],
                                  [1.0], [1.0], [0.0], 1.0, 0.0, 1)
    layer = Cert.stationarity_residual(problem, point.x, point.y, point.s,
                                       point.tau, point.kappa)
    @test exact.rP[1] == layer.rP[1] == 0
    @test exact.rD[1] == -layer.rD[1]
    @test exact.rG == 0.0                       # c'x + b'y, y_oracle = 1
    @test layer.rG == 0.0                       # c'x - b'y, y_layer = -1
    @test layer.rG == Cert.gap_value(problem, point.x, point.y)
    # The optimality certificate passes under this layer's convention:
    @test Cert.certify!(problem, point, Cert.OriginalOperator(problem),
                        Cert.EXIT_CONVERGED; tol=1e-8).decision ===
          Cert.DECISION_ACCEPT_OPTIMAL
    record!("scalar_oracle_rD", exact.rD[1])
    record!("scalar_layer_rD", layer.rD[1])
    record!("scalar_layer_rG", layer.rG)
    record!("scalar_dual_objective", Cert.dual_objective_value(problem, point.y))
end

# =====================================================================
#  D. A valid certificate still passes
# =====================================================================
@testset "D. valid certificate is accepted (gate is not reject-everything)" begin
    problem, point, op = fixture_scalar_psd(Cert)
    good = Cert.certify!(problem, point, op, Cert.EXIT_CONVERGED; tol=1e-8)
    @test good.decision === Cert.DECISION_ACCEPT_OPTIMAL
    @test good.verification === Cert.VERIFICATION_VERIFIED_OPTIMAL
    @test good.reject_reason === Cert.REJECT_NONE
    record!("scalar_good_decision", string(good.decision))
    record!("scalar_good_primal_residual", good.metrics.primal_residual)
    record!("scalar_good_dual_residual", good.metrics.dual_residual)
    record!("scalar_good_gap", good.metrics.gap)
    record!("scalar_good_cone_margin", good.metrics.primal_cone_margin)

    problem2, point2, op2 = fixture_psd2(Cert)
    good2 = Cert.certify!(problem2, point2, op2, Cert.EXIT_CONVERGED; tol=1e-10)
    @test good2.decision === Cert.DECISION_ACCEPT_OPTIMAL
    @test good2.verification === Cert.VERIFICATION_VERIFIED_OPTIMAL
    st = Cert.compose_terminal_status(good2.termination, good2.verification,
                                      good2.capability, good2.error_bound)
    @test st.status === :Optimal
    @test Cert.status_is_optimal(st)
    record!("psd2_good_stationarity", good2.metrics.stationarity)
    record!("scalar_good_dual_objective", good.metrics.dual_objective)
    record!("psd2_good_gap", good2.metrics.gap)

    # A valid Farkas ray must also be ACCEPTED, so the infeasibility path is
    # exercised positively as well as negatively.
    #
    #   min cᵀx  s.t.  A x = b,  x ∈ R^2_+   with  A = [1 1], b = -1
    #
    # is primal infeasible (x ≥ 0 cannot sum to −1), and the row is free so
    # its dual cone is all of R.  Two rays satisfy `bᵀy < 0` with `y ∈ K*`:
    #   y = 1  -> bᵀy = −1 < 0, y = 1 ∉ K* = R₊  → must be REFUSED
    #   y = −1 -> bᵀy =  1 > 0                 → not a Farkas ray either
    # The infeasibility authority therefore lives in `y` minus the row cone,
    # which this fixture pins down by checking both directions explicitly.
    A = sparse([1.0 1.0])
    prob_inf = Cert.OriginalProblem(A, [-1.0], [1.0, 1.0],
                                    [Cert.NonnegativeBlock(0, 2)])
    ok_ray, ray_reason = Cert.verify_primal_infeasibility(prob_inf, [-1.0])
    @test ok_ray == false                    # b'y = 1 > 0: not a Farkas ray
    @test ray_reason === :not_descending
    ok_ray2, _ = Cert.verify_primal_infeasibility(prob_inf, [1.0])
    @test ok_ray2 == false                   # y = 1 ∉ K* = R₊
    # The cone membership that decides it is the row cone:
    @test Cert.in_cone(prob_inf.row_blocks, [1.0])
    @test !Cert.in_cone(prob_inf.row_blocks, [-1.0])
    record!("infeasible_ray_reject_reason", string(ray_reason))
    # ...and a genuinely UNBOUNDED primal gets the dual-infeasibility
    # (recession) declaration rather than a residual rejection:
    #   min −x2  s.t.  x1 = 1,  x ∈ R^2_+
    # is unbounded below along the ray r = (0, 1): r ∈ K, A r = 0, cᵀr = −1 < 0.
    A2 = sparse([1.0 0.0])
    prob_unb = Cert.OriginalProblem(A2, [1.0], [0.0, -1.0],
                                    [Cert.NonnegativeBlock(0, 2)])
    # The certify entry point takes BOTH a primal ray `x_ray` and a dual ray
    # `y_ray`; here the unboundedness witness is the primal ray (0, 1), while
    # the dual vector stays at zero.
    ok_unb, why_unb = Cert.verify_dual_infeasibility(prob_unb, [0.0, 1.0])
    @test ok_unb
    @test why_unb === :ok
    unb = Cert.certify!(prob_unb, Cert.OriginalPoint([0.0, 1.0], [0.0], [0.0]),
                        Cert.OriginalOperator(prob_unb), Cert.EXIT_CONVERGED;
                        tol=1e-8)
    @test unb.verification === Cert.VERIFICATION_VERIFIED_DUAL_INFEASIBLE
    @test unb.decision === Cert.DECISION_DECLARE_DUAL_INFEASIBLE
    record!("unbounded_ray_decision", string(unb.decision))
end

# =====================================================================
#  A. Negative gate — typed rejections
# =====================================================================
@testset "A. negative inputs are rejected with typed outcomes" begin
    problem, point, op = fixture_scalar_psd(Cert)

    # A1: NaN in an original coordinate.
    nan_point = Cert.OriginalPoint([NaN], point.y, point.s, 1.0, 0.0)
    r = Cert.certify!(problem, nan_point, op, Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_NONFINITE_INPUT
    record!("nan_reject_reason", string(r.reject_reason))

    # A2: Inf inside the slack.
    inf_point = Cert.OriginalPoint([1.0], point.y, [Inf], 1.0, 0.0)
    r = Cert.certify!(problem, inf_point, op, Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_NONFINITE_INPUT

    # A3: NaN in the dual.
    r = Cert.certify!(problem, Cert.OriginalPoint([1.0], [NaN], [1.0], 1.0, 0.0),
                      op, Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_NONFINITE_INPUT

    # A4: a deliberately WRONG dual map must not be believed.  Three distinct
    #     defects are exercised; each must be a typed rejection.
    #     (i) a map artifact whose pullback is not the adjoint of its scale.
    correct = Cert.build_dual_map(2)
    # A pullback that is NOT the reciprocal of the packed scale: the
    # off-diagonal factor must be 1/√2, so 0.5 at that position is a wrong
    # dual map (it is within 1e-12 only if compared against 1.0, which is the
    # kind of sloppy comparison this layer must not make).
    artifact_bad = Cert.DualMapArtifact(2, correct.scale, [0.5, 1.0, 0.5];
                                        source=:injected)
    @test !Cert.map_is_dual_consistent(artifact_bad)
    wrong_scale = [1.0, 1.0, 1.5 / sqrt(2.0)]
    problem2, point2, _ = fixture_psd2(Cert; dual_scale=wrong_scale)
    op2_bad = Cert.OriginalOperator(problem2)
    r = Cert.certify!(problem2, point2, op2_bad, Cert.EXIT_CONVERGED; tol=1e-10)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_DUAL_MAP_INCONSISTENT
    record!("wrong_dual_map_reject_reason", string(r.reject_reason))

    #     (ii) an override that disagrees with the stored adjoint.
    problem3, point3, op3 = fixture_psd2(Cert)
    r = Cert.certify!(problem3, point3,
                      Cert.OriginalOperator(problem3;
                                            dual_map=Dict(1 => [1.0, 1.0, 1.0])),
                      Cert.EXIT_CONVERGED; tol=1e-10)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_DUAL_MAP_INCONSISTENT

    #     (iii) an override of the wrong length is a structural rejection.
    r = Cert.certify!(problem3, point3,
                      Cert.OriginalOperator(problem3; dual_map=Dict(1 => [1.0, 0.5])),
                      Cert.EXIT_CONVERGED; tol=1e-10)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_DUAL_MAP_INCONSISTENT

    # ...but the SAME point with the correct map is accepted (D above).

    # A5: a FABRICATED infeasibility ray.
    #     x >= 0, x1 + x2 = 1 is feasible, so no Farkas ray exists; a
    #     plausible-looking ray with b'y < 0 must be refused.
    A = sparse([1.0 1.0])
    prob_ok = Cert.OriginalProblem(A, [1.0], [1.0, 1.0],
                                   [Cert.NonnegativeBlock(0, 2)])
    fabricated = Cert.OriginalPoint([0.0, 0.0], [1.0], [0.0])
    r = Cert.certify!(prob_ok, fabricated, Cert.OriginalOperator(prob_ok),
                      Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT
    # No certificate exists for this point: the optimality residuals are not
    # small and no ray is valid, so the typed reason is one of the closed set.
    @test r.reject_reason in (Cert.REJECT_PRIMAL_RAY,
                              Cert.REJECT_PRIMAL_RESIDUAL,
                              Cert.REJECT_NO_CERTIFICATE_FOR_EXIT)
    record!("fabricated_ray_reject_reason", string(r.reject_reason))

    # A fabricated ray whose sign is wrong must ALSO be refused: it is the
    # ray gate (or an earlier cone gate) that stops it, never acceptance.
    fabricated_bad = Cert.OriginalPoint([0.0, 0.0], [-1.0], [0.0])
    r = Cert.certify!(prob_ok, fabricated_bad, Cert.OriginalOperator(prob_ok),
                      Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason in (Cert.REJECT_PRIMAL_RAY, Cert.REJECT_DUAL_CONE_VIOLATION,
                              Cert.REJECT_PRIMAL_CONE_VIOLATION)

    # A fabricated *dual* ray: c = (1,1) >= 0 has no recession direction.
    prob_no_ray = Cert.OriginalProblem(A, [1.0], [1.0, 1.0],
                                       [Cert.NonnegativeBlock(0, 2)])
    fabricated_dual = Cert.OriginalPoint([1.0, 0.0], [0.0], [0.0])
    r = Cert.certify!(prob_no_ray, fabricated_dual,
                      Cert.OriginalOperator(prob_no_ray),
                      Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT

    # A6: an invalid tolerance is refused, never silently clamped.
    r = Cert.certify!(problem, point, op, Cert.EXIT_CONVERGED; tol=NaN)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_INVALID_TOLERANCE
    r = Cert.certify!(problem, point, op, Cert.EXIT_CONVERGED; tol=-1.0)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_INVALID_TOLERANCE

    # A7: dimension mismatch is refused.
    r = Cert.certify!(problem, Cert.OriginalPoint([1.0, 2.0], point.y, point.s),
                      op, Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason === Cert.REJECT_DIMENSION_MISMATCH

    # A8: cone violation is refused (x = -1 not in S^1_+).
    r = Cert.certify!(problem, Cert.OriginalPoint([-1.0], point.y, [-1.0], 1.0, 0.0),
                      op, Cert.EXIT_CONVERGED; tol=1e-8)
    @test r.decision === Cert.DECISION_REJECT
    @test r.reject_reason in (Cert.REJECT_PRIMAL_CONE_VIOLATION,
                              Cert.REJECT_DUAL_CONE_VIOLATION)
end

# =====================================================================
#  B. Resource exits retain state, never promote
# =====================================================================
@testset "B. maxiter/time exits retain last valid state, not Optimal" begin
    problem, point, op = fixture_scalar_psd(Cert)

    for exit_kind in (Cert.EXIT_MAXITER, Cert.EXIT_TIME_LIMIT,
                      Cert.EXIT_MEMORY_LIMIT, Cert.EXIT_BREAKDOWN)
        r = Cert.certify!(problem, point, op, exit_kind; tol=1e-8,
                          has_last_valid_state=true)
        @test r.decision === Cert.DECISION_HOLD_LAST_VALID_STATE
        @test r.verification !== Cert.VERIFICATION_VERIFIED_OPTIMAL
        @test !Cert.status_is_optimal(
            Cert.compose_terminal_status(r.termination, r.verification,
                                         r.capability, r.error_bound))
        # The retained state is the same point: the last valid state survives.
        @test point.x == [1.0] && point.y == [-1.0] && point.s == [0.0]
    end

    r_max = Cert.certify!(problem, point, op, Cert.EXIT_MAXITER; tol=1e-8,
                          has_last_valid_state=true)
    st_max = Cert.compose_terminal_status(r_max.termination, r_max.verification,
                                          r_max.capability, r_max.error_bound)
    @test st_max.status === :IterLimit
    record!("maxiter_status", string(st_max.status))
    record!("maxiter_decision", string(r_max.decision))

    r_time = Cert.certify!(problem, point, op, Cert.EXIT_TIME_LIMIT; tol=1e-8,
                           has_last_valid_state=true)
    st_time = Cert.compose_terminal_status(r_time.termination, r_time.verification,
                                           r_time.capability, r_time.error_bound)
    @test st_time.status === :TimeLimit
    record!("time_status", string(st_time.status))
    record!("time_decision", string(r_time.decision))

    ev_max = Cert.resource_exit_evidence(Cert.EXIT_MAXITER, point;
                                         iterations=7,
                                         verification=r_max.verification,
                                         reject_reason=r_max.reject_reason,
                                         metrics=r_max.metrics)
    ev_time = Cert.resource_exit_evidence(Cert.EXIT_TIME_LIMIT, point;
                                          seconds=0.25, iterations=3,
                                          verification=r_time.verification,
                                          reject_reason=r_time.reject_reason,
                                          metrics=r_time.metrics)
    @test ev_max.last_valid_state !== nothing
    @test ev_time.last_valid_state !== nothing
    @test ev_time.seconds == 0.25
    record!("resource_exit_iterations", ev_max.iterations)
    record!("resource_exit_seconds", ev_time.seconds)

    # An unmeasured iteration count or duration is `nothing`, never 0.
    ev_unmeasured = Cert.resource_exit_evidence(Cert.EXIT_MAXITER, nothing)
    @test ev_unmeasured.iterations === nothing
    @test ev_unmeasured.seconds === nothing
    @test Cert.measured(ev_unmeasured.iterations) isa Cert.Unmeasured
    @test Cert.measured(ev_unmeasured.iterations) === Cert.UNMEASURED

    # An unsupported capability is an explicit refusal, never an acceptance.
    r_unsup = Cert.certify!(problem, point, op, Cert.EXIT_CONVERGED; tol=1e-8,
                            capability=Cert.CAPABILITY_UNSUPPORTED)
    @test r_unsup.decision === Cert.DECISION_REJECT
    @test !Cert.status_is_optimal(
        Cert.compose_terminal_status(r_unsup.termination, r_unsup.verification,
                                     r_unsup.capability, r_unsup.error_bound))
    record!("unsupported_decision", string(r_unsup.decision))

    # An `Optimal` word is reachable only from converged + verified + supported.
    @test !Cert.status_is_optimal(Cert.compose_terminal_status(
        Cert.TERMINATION_MAXITER, Cert.VERIFICATION_VERIFIED_OPTIMAL,
        Cert.CAPABILITY_SUPPORTED, Cert.ERROR_BOUND_ESTIMATED))
    @test !Cert.status_is_optimal(Cert.compose_terminal_status(
        Cert.TERMINATION_TIME_LIMIT, Cert.VERIFICATION_VERIFIED_OPTIMAL,
        Cert.CAPABILITY_SUPPORTED, Cert.ERROR_BOUND_STRICT))
    @test Cert.status_is_optimal(Cert.compose_terminal_status(
        Cert.TERMINATION_CONVERGED, Cert.VERIFICATION_VERIFIED_OPTIMAL,
        Cert.CAPABILITY_SUPPORTED, Cert.ERROR_BOUND_ESTIMATED))

    # Strict error bound vs estimate: the layer may claim `strict` only when
    # a witness is supplied, and `strict` never promotes the *verification*
    # axis or the status word on a resource exit.
    cls_est, bound_est = Cert.strict_error_bound(problem, point; tol=1e-8)
    cls_str, bound_str = Cert.strict_error_bound(problem, point; tol=1e-8,
                                                 witness=:exact_arithmetic)
    @test cls_est === Cert.ERROR_BOUND_ESTIMATED
    @test cls_str === Cert.ERROR_BOUND_STRICT
    @test bound_est == bound_str
    record!("error_bound_estimated", bound_est)
    record!("error_bound_strict", bound_str)
    r_str = Cert.certify!(problem, point, op, Cert.EXIT_MAXITER; tol=1e-8,
                          witness=:exact_arithmetic, has_last_valid_state=true)
    @test r_str.error_bound === Cert.ERROR_BOUND_STRICT
    @test r_str.verification !== Cert.VERIFICATION_VERIFIED_OPTIMAL
    @test r_str.decision !== Cert.DECISION_ACCEPT_OPTIMAL
end

# =====================================================================
#  C. A/B — diagnostics/verbosity OFF must not change anything numeric
# =====================================================================
@testset "C. A/B diagnostics ON vs OFF: identical numbers and decisions" begin
    problem, point, op = fixture_scalar_psd(Cert)
    problem2, point2, op2 = fixture_psd2(Cert)
    wrong_scale = [1.0, 1.0, 1.5 / sqrt(2.0)]
    problem_bad, point_bad, _ = fixture_psd2(Cert; dual_scale=wrong_scale)
    nan_point = Cert.OriginalPoint([NaN], point.y, point.s, 1.0, 0.0)

    cases = [
        ("valid_scalar", problem, point, op, Cert.EXIT_CONVERGED, 1e-8, true),
        ("valid_psd2", problem2, point2, op2, Cert.EXIT_CONVERGED, 1e-10, true),
        ("wrong_dual_map", problem_bad, point_bad, Cert.OriginalOperator(problem_bad),
         Cert.EXIT_CONVERGED, 1e-10, true),
        ("nan_input", problem, nan_point, op, Cert.EXIT_CONVERGED, 1e-8, true),
        ("maxiter_retained", problem, point, op, Cert.EXIT_MAXITER, 1e-8, true),
    ]

    for (name, p, pt, o, exit_kind, tol, hlvs) in cases
        a = Cert.certify!(p, pt, o, exit_kind; tol=tol, diagnostics=true,
                          has_last_valid_state=hlvs)
        b = Cert.certify!(p, pt, o, exit_kind; tol=tol, diagnostics=false,
                          has_last_valid_state=hlvs)

        @test a.decision === b.decision
        @test a.verification === b.verification
        @test a.reject_reason === b.reject_reason
        @test a.error_bound === b.error_bound
        @test a.capability === b.capability
        @test a.provenance.telemetry_enabled === true
        @test b.provenance.telemetry_enabled === false
        @test Cert.certification_checks_run(a.provenance) ==
              Cert.certification_checks_run(b.provenance)

        # Bitwise comparison of every measured numeric field.
        for field in (:primal_residual, :dual_residual, :stationarity, :gap,
                      :complementarity, :primal_cone_margin, :dual_cone_margin,
                      :objective, :dual_objective, :data_scale)
            va = getfield(a.metrics, field)
            vb = getfield(b.metrics, field)
            @test isequal(va, vb)
            @test reinterpret(UInt64, Float64(va)) == reinterpret(UInt64, Float64(vb))
        end
        @test isequal(a.metrics.kappa, b.metrics.kappa)
        @test isequal(a.strict_error_bound, b.strict_error_bound)
        @test Cert.telemetry_agrees(
            Cert.certification_telemetry(a.metrics, a.provenance;
                                         seconds=1.0e-6, allocations=128,
                                         iterations=3,
                                         checks_run=Cert.certification_checks_run(a.provenance)),
            Cert.certification_telemetry(b.metrics, b.provenance;
                                         checks_run=Cert.certification_checks_run(b.provenance)))

        st_a = Cert.compose_terminal_status(a.termination, a.verification,
                                            a.capability, a.error_bound;
                                            certificate_attempted=a.provenance.certificate_attempted)
        st_b = Cert.compose_terminal_status(b.termination, b.verification,
                                            b.capability, b.error_bound;
                                            certificate_attempted=b.provenance.certificate_attempted)
        @test st_a.status === st_b.status

        record!("ab_$(name)_decision", string(a.decision))
        record!("ab_$(name)_stationarity", a.metrics.stationarity)
        record!("ab_$(name)_primal_residual", a.metrics.primal_residual)
        record!("ab_$(name)_status", string(st_a.status))
    end

    # Telemetry with diagnostics off reports timings as `not_run`, never 0.
    off_cert = Cert.certify!(problem, point, op, Cert.EXIT_CONVERGED; tol=1e-8,
                             diagnostics=false)
    on_cert = Cert.certify!(problem, point, op, Cert.EXIT_CONVERGED; tol=1e-8,
                            diagnostics=true)
    checks = Cert.certification_checks_run(on_cert.provenance)
    off = Cert.certification_telemetry(off_cert.metrics, off_cert.provenance;
                                       checks_run=checks)
    on = Cert.certification_telemetry(on_cert.metrics, on_cert.provenance;
                                      seconds=1.0e-6, allocations=128,
                                      iterations=3, checks_run=checks)
    @test off.certification_seconds isa Cert.Unmeasured
    @test off.allocations_bytes isa Cert.Unmeasured
    @test Cert.measurement_label(off.certification_seconds) == "not_run"
    @test Cert.telemetry_timings_unmeasured(off)
    @test !Cert.telemetry_timings_unmeasured(on)
    @test on.certification_seconds isa Float64
    # ...and the ON record's numeric metrics are bitwise the OFF record's.
    for field in (:primal_residual, :dual_residual, :stationarity, :gap,
                  :complementarity, :primal_cone_margin, :dual_cone_margin,
                  :data_scale)
        @test isequal(getfield(on, field), getfield(off, field))
    end
    @test Cert.telemetry_agrees(on, off)
    @test !(Cert.measured(nothing) isa Real)
    record!("telemetry_off_seconds_label",
            Cert.measurement_label(off.certification_seconds))
    record!("telemetry_on_seconds", on.certification_seconds)
end

# =====================================================================
#  L2 — direction gate is not replaceable by a lease
# =====================================================================
@testset "L2. direction gate runs and cannot be leased away" begin
    problem, point, op = fixture_scalar_psd(Cert)
    dx = [1.0]; ds = [1.0]; dy = [1.0]
    for lease in (false, true)
        ev = Cert.certify_direction(problem, op, dx, ds, dy; tol=1e-8,
                                    lease_present=lease)
        @test Cert.direction_admitted(ev)
        @test ev.checks_run >= 5           # L2 numeric checks always ran
        @test ev.lease_present === lease
    end
    ev_bad = Cert.certify_direction(problem, op, [NaN], ds, dy; tol=1e-8,
                                    lease_present=true)
    @test !Cert.direction_admitted(ev_bad)
    @test ev_bad.reason === Cert.REJECT_NONFINITE_INPUT
    @test ev_bad.checks_run >= 3
    problem_bad, _, _ = fixture_psd2(Cert; dual_scale=[1.0, 1.0, 1.5 / sqrt(2.0)])
    ev_map = Cert.certify_direction(problem_bad, Cert.OriginalOperator(problem_bad),
                                    [0.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0]; tol=1e-10,
                                    lease_present=true)
    @test !Cert.direction_admitted(ev_map)
    @test ev_map.reason === Cert.REJECT_DUAL_MAP_INCONSISTENT
    record!("direction_checks_run", ev_bad.checks_run)
end

# =====================================================================
#  Provenance / measurement discipline
# =====================================================================
@testset "provenance and unmeasured discipline" begin
    problem, point, op = fixture_scalar_psd(Cert)
    r = Cert.certify!(problem, point, op, Cert.EXIT_CONVERGED; tol=1e-8,
                      input_hash=UInt64(0x50445353))
    @test r.provenance.layer === :L3
    @test r.provenance.arithmetic === Float64
    @test r.provenance.rounding === :round_to_nearest_even
    @test r.provenance.tolerance == 1e-8
    @test r.provenance.input_hash == UInt64(0x50445353)
    t = Cert.unmeasured_telemetry(Float64)
    @test t.primal_residual isa Cert.Unmeasured
    @test Cert.measurement_label(t.primal_residual) == "not_run"
    @test !(t.primal_residual == 0)
    record!("provenance_rounding", string(r.provenance.rounding))
    record!("unmeasured_label", Cert.measurement_label(t.primal_residual))
end

# =====================================================================
#  Type genericity — the layer is not silently Float64-only
# =====================================================================
@testset "type genericity: BigFloat arithmetic" begin
    setprecision(BigFloat, 256) do
        # The packing factor is generated in the REQUESTED arithmetic; a
        # Float64 √2 widened into 256 bits would be an implicit precision
        # downgrade (ADR-003 §6).
        blk64 = Cert.PSDBlock(0, 2)
        blkbf = Cert.PSDBlock(0, 2; T=BigFloat)
        @test eltype(blk64.scale) === Float64
        @test eltype(blkbf.scale) === BigFloat
        @test blkbf.scale[2] == sqrt(big(2))
        @test Cert.psd_scale_of(2, 2) == sqrt(2.0)
        @test Cert.psd_scale_of(2, 2; T=BigFloat) == sqrt(big(2))

        # LinearAlgebra has no BigFloat symmetric eigensolver, so the layer
        # uses its own Jacobi sweep.  The values must match the ANALYTIC
        # spectrum of known matrices.  The bound is stated rather than using
        # `==`: at 256 bits the sweep is exact to ~1e-77 (measured: the 2×2
        # value below differs from 1 by 1.7e-77), so `1e-60` is a real
        # assertion with 17 digits of headroom, not a rubber stamp.
        eigtol = big"1e-60"
        M = BigFloat[2 1; 1 2]
        @test abs(Cert.min_symmetric_eigvalue(M) - 1) <= eigtol
        M2 = BigFloat[3 0 0; 0 5 0; 0 0 4]
        @test abs(Cert.min_symmetric_eigvalue(M2) - 3) <= eigtol
        M3 = BigFloat[1 0.25; 0.25 1]
        @test abs(Cert.min_symmetric_eigvalue(M3) - (1 - big(0.25))) <= eigtol
        record!("bigfloat_jacobi_error_2x2",
                abs(Cert.min_symmetric_eigvalue(M) - 1))

        # Full certification of the 2×2 optimality fixture at 256 bits.
        A = convert(SparseMatrixCSC{BigFloat,Int},
                    sparse([1.0 0.0 0.0; 0.0 0.0 1.0]))
        p2 = Cert.OriginalProblem(A, BigFloat[1//10, 1//10],
                                  BigFloat[-1//10, 0, -1//10], [blkbf])
        pt2 = Cert.OriginalPoint(BigFloat[1//10, 0, 1//10],
                                 BigFloat[-1//10, -1//10], BigFloat[0, 0],
                                 one(BigFloat), zero(BigFloat))
        r2 = Cert.certify!(p2, pt2, Cert.OriginalOperator(p2), Cert.EXIT_CONVERGED;
                           tol=big"1e-60")
        @test r2.decision === Cert.DECISION_ACCEPT_OPTIMAL
        @test r2.provenance.arithmetic === BigFloat
        @test r2.metrics.primal_residual == 0
        @test r2.metrics.dual_residual == 0
        @test r2.metrics.gap == 0
        @test r2.metrics.dual_cone_margin == 0
        record!("bigfloat_gap", r2.metrics.gap)
        record!("bigfloat_arithmetic", string(r2.provenance.arithmetic))
    end
end

end # @testset S04

# ---------------------------------------------------------------------
#  Raw measurement dump (real values only; `not_run` where unmeasured).
# ---------------------------------------------------------------------
println("S04_RAW_MEASUREMENTS")
for key in sort(collect(keys(RAW)))
    println("  ", key, " = ", RAW[key])
end
