# Driver: run the experimental half-Power step context on the exact canonical
# Power problem (min sum t, (t,1,a) in POW3^{0.5}).  Records every accepted
# step, the useful-progress floor crossing, and the committed pair lineage.
using Test, TOML, LinearAlgebra, SparseArrays, SHA, SDPX
include(joinpath(@__DIR__, "..", "factor_preserving_affine.jl"))
include(joinpath(@__DIR__, "..", "factor_affine_reference.jl"))
include(joinpath(@__DIR__, "..", "native_factor_affine_certificate.jl"))
include(joinpath(@__DIR__, "..", "half_power_native_corrector.jl"))
include(joinpath(@__DIR__, "..", "factor_combined_epoch.jl"))
include(joinpath(@__DIR__, "..", "native_half_pair.jl"))
include(joinpath(@__DIR__, "experimental_power_step.jl"))
const EPS = ExperimentalPowerStep
const NP = NativeHalfPair
const FA = FactorPreservingAffine
const NC = NativeFactorAffineCertificate
const FAR = FactorAffineReference
const Q = Rational{BigInt}

floatword(s) = reinterpret(Float64, parse(UInt64, s; base = 16))
row = TOML.parsefile(joinpath(@__DIR__, "..", "fixtures", "factor_affine_trial_17.toml"))
a = [floatword(row["b_bits"][k]) for k in (6, 9, 12)]
A = sparse([1, 4, 2, 7, 3, 10], [1, 1, 2, 2, 3, 3], fill(-1.0, 6), 12, 3)
b = zeros(12)
for (i, v) in enumerate(a)
    b[3i + 2] = 1.0   # second component of power block i (rows 5,8,11)
    b[3i + 3] = v     # third component of power block i (rows 6,9,12)
end
c = ones(3)
problem = (A, b, c)
layout = NP.Layout(3, (0.5, 0.5, 0.5))
exact_obj = sum(Q(v)^2 for v in a)

function sprint_res(prefix, res)
    if res.ok
        println(prefix, " ok alpha=", res.alpha, " merit=", res.merit,
            " bt=", res.backtracking)
    else
        println(prefix, " FAILED stage=", res.stage,
            hasproperty(res, :alpha) ? " alpha=" * string(res.alpha) : "",
            hasproperty(res, :backtracking) ? " bt=" * string(res.backtracking) : "")
    end
end

target = 1e-8
function terminal_audit(ctx; target = 1e-8)
    e = EPS.build_epoch(ctx.pair, problem, ctx.x, ctx.tau, ctx.kappa)
    rP, rD, rG = EPS.current_residuals(e, ctx.x, ctx.pair.s, ctx.pair.y, ctx.tau, ctx.kappa)
    m = max(EPS.maxinf(rP), EPS.maxinf(rD), abs(rG))
    sN = ctx.pair.s ./ ctx.tau; yN = ctx.pair.y ./ ctx.tau; xN = ctx.x ./ ctx.tau
    obj = dot(c, xN)
    sNy = dot(ctx.pair.s, ctx.pair.y) / (ctx.tau^2)
    scale = max(1.0, maximum(abs, sN), maximum(abs, yN), maximum(abs, xN), abs(obj))
    tol = 1e-7 * scale
    # original-coordinate feasibility and cone membership within the band
    membership = true
    for i in 1:3
        membership &= (sN[i] >= -tol && yN[i] >= -tol)
    end
    for blk in 0:2
        rows = 3 + 3*blk + 1 : 3 + 3*blk + 3
        x, y, z = sN[rows]; u, v, w = yN[rows]
        membership &= (x >= -tol && y >= -tol && x*y - z*z >= -tol)
        membership &= (u >= -tol && v >= -tol && 4.0*u*v - w*w >= -tol)
    end
    primal_feas = EPS.maxinf(ctx.rP) / max(1.0, ctx.tau)
    dual_feas = EPS.maxinf(ctx.rD) / max(1.0, ctx.tau)
    obj_gap = abs(obj + dot(b, yN))      # c'x* + b'y* = -kappa/tau ~ 0
    (; m, membership, primal_feas, dual_feas, obj, sNy, obj_gap,
        obj_err = abs(obj - Float64(exact_obj)))
end

@testset "experimental half-Power step context" begin
    start = EPS.cold_start(problem, layout)
    @test start.ok
    ctx = start.ctx
    @test ctx.pair isa NP.PairReceipt && EPS.NP.verify(ctx.pair)
    reached_floor = false
    terminal_ok = false
    for iter in 1:80
        res = EPS.step!(ctx)
        if !res.ok
            sprint_res("ITER $(iter):", res)
            break
        end
        reached_floor |= res.alpha >= EPS.PROG_FLOOR
        if res.merit <= target
            aud = terminal_audit(ctx)
            if aud.membership && aud.sNy <= 1e-6 && aud.obj_err <= 1e-6 &&
               aud.primal_feas <= 1e-7 && aud.dual_feas <= 1e-7
                terminal_ok = true
                println("TERMINATED iter=", iter, " merit=", res.merit, " obj_err=", aud.obj_err)
                break
            end
        end
        println("ITER ", iter, ": accepted alpha=", res.alpha,
            " (floor ", EPS.PROG_FLOOR, ") merit=", res.merit,
            " sigma_mu=", ctx.history[end].sigma_mu,
            " mu=", ctx.pair.mu, " tau=", ctx.tau, " kappa=", ctx.kappa)
        # verify committed pair + next-epoch five equations at the committed state
        e = EPS.build_epoch(ctx.pair, problem, ctx.x, ctx.tau, ctx.kappa)
        aff = FA.solve(e, FA.affine_rhs(e))
        @test NC.certify(e, aff).status === :certified
        @test all(v -> v <= Q(FactorPreservingAffine.PHYSICAL_FORCING), FAR.physical(e, aff).errors)
        @test EPS.NP.verify(ctx.pair)
        @test ctx.owner.anchor === ctx.pair
        if ctx.iterations >= 2
            # committed trial is the new anchor; a stale-token rebuild must refuse
            bad = EPS.NP.build(copy(ctx.pair.s), copy(ctx.pair.y), ctx.pair.mu,
                ctx.layout; policy = NP.POLICY, settings = ctx.settings,
                owner = NP.Owner(), warm = ctx.owner.tokens)
            @test bad isa NP.PairRefusal
        end
    end
    @test reached_floor
    @test terminal_ok
    @test ctx.iterations >= 1
    terminal = terminal_audit(ctx)
    println("TERMINAL merit=", terminal.m, " membership=", terminal.membership,
        " pr=", terminal.primal_feas, " dr=", terminal.dual_feas,
        " sNy=", terminal.sNy, " obj=", terminal.obj, " obj_err=", terminal.obj_err)
    @test terminal.membership
    @test terminal.sNy <= 1e-6
    @test terminal.primal_feas <= 1e-7 && terminal.dual_feas <= 1e-7
    @test terminal.obj_err <= 1e-4
    println("ACCEPTED_STEPS ", length(ctx.history))
    for h in ctx.history
        println("  alpha=", h.alpha, " merit=", h.merit, " sigma_mu=", h.sigma_mu)
    end
end