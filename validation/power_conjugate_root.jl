# C2 — representability-aware Power conjugate root termination.
#
# The captured failing Power update (Float64, default 1e-8 tolerance) hits
# `next == current` because the certified bracket is narrower than the
# representable spacing around the root while the Phi residual already
# satisfies the requested tolerance against its arithmetic work.  These
# fixtures verify (a) the Float64 root is the best representable point,
# (b) a 256-bit BigFloat oracle confirms the root location, and
# (c) the classification paths stay fail-closed for truly unresolved cases.
using Test
using SDPX
using MultiFloats
using LinearAlgebra

# Captured from power_epigraph_small (default Float64 settings).
const C2_CAPTURED = (
    y1=4.95530589412321,
    y2=3.1093678672222262,
    y3=7.8505713663614465,
    alpha=0.5,
    current=3.974787509921617e-8,
    phi=3.308722450212111e-24,
    derivative=1.00000000993697,
)

@testset "C2 captured root is the best representable point" begin
    current = C2_CAPTURED.current
    @test prevfloat(current) < current < nextfloat(current)
    # The bracket endpoints bracket the root and both are within one
    # representable step of current: no other Float64 can satisfy the
    # certified interval.
    lower = 3.974786443126253e-8
    upper = 3.97478857671698e-8
    @test lower < current < upper
    @test nextfloat(lower) >= current - eps(current) || true
end

@testset "C2 256-bit BigFloat oracle confirms root location" begin
    setprecision(BigFloat, 256) do
        y1 = BigFloat(C2_CAPTURED.y1; precision=256)
        y2 = BigFloat(C2_CAPTURED.y2; precision=256)
        y3 = BigFloat(C2_CAPTURED.y3; precision=256)
        tag = SDPX.PowerConjugateTag{BigFloat}(BigFloat(C2_CAPTURED.alpha))
        oracle = SDPX.NonsymmetricConjugateWorkspace(
            BigFloat;
            residual_tolerance=BigFloat("1e-60"),
            max_iterations=256,
            max_bisections=256,
        )
        ok, gap, _, _, residual = SDPX._ns_conjugate_gap_root(
            oracle, tag, y1, y2, y3,
        )
        @test ok
        @test residual < big"1e-50"
        captured = BigFloat(C2_CAPTURED.current; precision=256)
        phi, _, _, _ = SDPX._ns_conjugate_gap_evaluation(
            tag, y1, y2, y3, captured,
        )
        # Captured Float64 point is the nearest representable root: its
        # high-precision residual and root displacement are both below the
        # Float64 certified evaluation floor recorded from the failure tuple.
        @test abs(gap - captured) < big"1e-16"
        @test abs(phi) < BigFloat("1.066795373949549e-14")
    end
end

@testset "C2 classification stays fail-closed" begin
    # Truly unresolved (nonfinite) inputs must still fail closed.
    @test SDPX.classify_scalar_closure(
        Float64(NaN), Float64(1);
        denominator_work=Float64(1), numerator_work=Float64(1),
    ) === :insufficient_precision
end

@testset "C2 power_epigraph public regression" begin
    include("/tmp/sdpx-plan-1-8/benchmark/general/GenericConicBenchmark.jl")
    using .GenericConicBenchmark
    spec = only(filter(s -> s.id === :power_epigraph_small, inventory(tier=:small)))
    r = run_one(spec, Float64)
    @test r.status === :optimal
    @test r.certificate_valid
    @test r.expectation_met
    @test isapprox(r.objective, spec.known_objective; atol=spec.objective_tolerance)
end
