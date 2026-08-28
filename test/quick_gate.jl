# Fast per-commit regression gate (target: < 60 s end to end).
#
# Covers one representative native-HSD solve per currently-green family
# (LP / SOCP / PSD) plus the certificate-promotion invariant, reusing the
# deterministic general benchmark fixtures. Full suites remain authoritative for
# merges; this gate gives structural/performance refactors honest signal
# in about a minute. No tolerance is loosened: shipped default Tolerances
# and certificate gates only.
#
# Known-failing families deliberately NOT asserted in the fast gate (owned by
# the kernel-restructure wave; add them back as they turn green):
#   * tiny degenerate SOC/PSD instances (rank-1 boundary optima)
#   * mixed Reals+PSD coefficient-matching models in native HSD
#   * affine orthant slack rows with positive rhs, including Nonpositive rows
#
# The last gap has a precise minimal repro: free x, x1+x2=1, and
# x1<=1/x2<=1 as Nonpositive rows. The equality and both caps imply
# 0<=x1,x2<=1, so this is a bounded LP with optimum 2.0 (an earlier audit
# incorrectly called it unbounded). The bordered route still breaks down;
# the opt-in expanded route now returns the certified optimum. An exact
# slack-variable formulation (u>=0, x+u=1) also solves, but is intentionally
# not hidden as a fallback.
using SDPX
using Test
using SHA

function _quick_gate_provenance()
    root = normpath(joinpath(@__DIR__, ".."))
    source_sha = readchomp(`git -C $root rev-parse HEAD`)
    source_state = success(`git -C $root diff --quiet HEAD --`) ? "clean" : "dirty"
    project = Base.active_project()
    manifest = joinpath(dirname(project), "Manifest.toml")
    manifest_sha = isfile(manifest) ? bytes2hex(sha256(read(manifest))) : "missing"
    report = join((
        "source_sha=$source_sha",
        "source_state=$source_state",
        "active_project=$project",
        "test_manifest_sha256=$manifest_sha",
    ), '\n') * '\n'
    artifact = get(
        ENV, "SDPX_QUICK_GATE_ARTIFACT",
        joinpath(tempdir(), "sdpx-quick-gate-provenance.txt"),
    )
    write(artifact, report)
    println("SDPX quick-gate provenance:\n", report, "artifact=", artifact)
    return artifact
end

const _QUICK_GATE_PROVENANCE = _quick_gate_provenance()

if !isdefined(Main, :GenericConicBenchmark)
    include(joinpath(
        @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
    ))
end
const _QUICK_GENERAL = Main.GenericConicBenchmark

function _quick_general_model(id::Symbol)
    spec = only(filter(
        candidate -> candidate.id === id,
        _QUICK_GENERAL.inventory(; tier=:small),
    ))
    return _QUICK_GENERAL.build(spec.problem, Float64, spec.params)
end

function _native_solve(id::Symbol)
    return SDPX.optimize!(
        _quick_general_model(id);
        settings=SDPX.Settings{Float64}(engine=:native_hsd),
    )
end

@testset "quick gate" begin
    # --- LP: free variables + nonnegativity + equality (native HSD) ---
    r = _native_solve(:lp_afiro_style)
    @test SDPX.status(r) === :optimal
    @test r.certificate.valid
    @test isapprox(SDPX.primal_objective(r), 9.0; atol=1e-6)

    # --- SOCP: second-order blocks (native HSD) ---
    r = _native_solve(:socp_portfolio_small)
    @test SDPX.status(r) === :optimal
    @test r.certificate.valid
    @test isapprox(SDPX.primal_objective(r), 1.0; atol=1e-5)

    # --- PSD: semidefinite block (native HSD) ---
    r = _native_solve(:sdp_maxcut_k4)
    @test SDPX.status(r) === :optimal
    @test r.certificate.valid
    @test isapprox(SDPX.primal_objective(r), 4.0; atol=1e-5)
end

@testset "known native gaps (documented)" begin
    if get(ENV, "SDPX_RUN_KNOWN_GAPS", "0") == "1"
        # Bounded capped LP: x2=1-x1 together with x2<=1 implies x1>=0,
        # and symmetrically x2>=0. The optimum is therefore 2 at (0,1).
        capped = SDPX.Model(Float64)
        x = SDPX.variable!(capped, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(capped, :eq, x[1] + x[2] - 1.0, SDPX.ZeroCone())
        SDPX.constraint!(capped, :upper1, x[1] - 1.0, SDPX.Nonpositive())
        SDPX.constraint!(capped, :upper2, x[2] - 1.0, SDPX.Nonpositive())
        SDPX.objective!(capped, SDPX.Maximize(), x[1] + 2 * x[2])
        result = SDPX.optimize!(capped;
            settings=SDPX.Settings{Float64}(engine=:native_hsd))
        @test_broken SDPX.status(result) === :optimal &&
            result.certificate.valid &&
            isapprox(SDPX.primal_objective(result), 2.0; atol=1e-6)

        # Bounded mixed-family repro: lower Nonnegative and upper Nonpositive
        # affine rows, with the upper bounds redundant but semantically valid.
        bounded = SDPX.Model(Float64)
        y = SDPX.variable!(bounded, :y, 2; domain=SDPX.Reals())
        SDPX.constraint!(bounded, :sum, y[1] + y[2] - 1.0, SDPX.ZeroCone())
        SDPX.constraint!(bounded, :lower1, y[1], SDPX.Nonnegative())
        SDPX.constraint!(bounded, :lower2, y[2], SDPX.Nonnegative())
        SDPX.constraint!(bounded, :upper1, y[1] - 1.0, SDPX.Nonpositive())
        SDPX.constraint!(bounded, :upper2, y[2] - 1.0, SDPX.Nonpositive())
        SDPX.objective!(bounded, SDPX.Maximize(), y[1] + 2 * y[2])
        result = SDPX.optimize!(bounded;
            settings=SDPX.Settings{Float64}(engine=:native_hsd))
        bounded_ok = SDPX.status(result) === :optimal &&
            result.certificate.valid &&
            isapprox(SDPX.primal_objective(result), 2.0; atol=1e-6)
        @test_broken bounded_ok
    end
end
