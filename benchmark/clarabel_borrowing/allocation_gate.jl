# PR-06: a small, stable allocation gate that runs in the DEFAULT environment.
#
# The plan asks to "恢复一个小而稳定的 allocation gate，不将全部大型性能测试塞回
# 普通 E2E" -- restore a small stable allocation gate rather than pushing the
# whole performance suite back into the ordinary E2E run.
#
# The existing `benchmark/general/performance/hsd_allocation.jl` cannot serve
# that role: it imports MultiFloats, MultiFloatLinearAlgebra and
# BigFloatLinearAlgebra, so it does not load in the default project at all and
# therefore protects nothing in ordinary CI.
#
# This gate is Float64-only and self-contained, so it runs anywhere. It asserts
# two ceilings that were MEASURED, not chosen:
#
#   * per-step allocation in the Newton loop;
#   * allocation of a complete warm public solve.
#
# Both are ceilings with headroom, not equality assertions, because a gate that
# fails on a one-off allocator difference teaches people to ignore it. The
# measured values are printed on every run so drift is visible before the
# ceiling is hit.
#
#   julia --startup-file=no --project=. benchmark/clarabel_borrowing/allocation_gate.jl
#
# Exits non-zero if a ceiling is exceeded.

using SDPX
using Printf

"""Measured 2026-09-11 at HEAD 4e5938f: 54.4 bytes/step, 272 bytes over 5 steps.

The residual is NOT zero and is not claimed to be. It is identical before and
after the PR-01/PR-07 changes (verified by measuring a stashed build), so it is
pre-existing. The plan's instruction is to *locate* per-step allocation and say
whether it is solver scratch, logging/boxing, or a provider `factor \\ rhs`
return value -- not to assume the wrapper's zero-allocation claim transfers to
the bottom layer. Locating it further is open work; this gate pins it so it
cannot grow unnoticed.
"""
const PER_STEP_CEILING_BYTES = 256

"""Measured 2026-09-11 at HEAD 4e5938f: 326160 bytes for a complete warm solve.

This is a whole-solve figure: it includes the public wrapper, certificate
assembly, result construction and the recovered-point copies, none of which the
per-step loop pays. It is therefore a regression ceiling for the public path,
not a claim about the inner loop.
"""
const WARM_SOLVE_CEILING_BYTES = 1_500_000

function _soc_model()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

"""Warmed per-step allocation of the product-HSD Newton loop."""
function _per_step_bytes(steps::Int=5)
    model = _soc_model()
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    state = SDPX.ProductConeHSDState(canonical)
    SDPX.product_hsd_cold_start!(state)
    SDPX._product_hsd_residual!(state)
    # Warm the loop past compilation before measuring, otherwise the first-call
    # compilation dominates and the number is meaningless.
    for _ in 1:4
        SDPX.product_hsd_step!(state)
    end
    measured = @allocated begin
        for _ in 1:steps
            SDPX.product_hsd_step!(state)
        end
    end
    return measured, div(measured, steps)
end

"""Warmed allocation of a complete public solve."""
function _warm_solve_bytes()
    settings = SDPX.Settings(Float64; verbosity=0,
        limits=SDPX.Limits(iterations=200, time=60.0, threads=1))
    # One untimed warm solve so only the steady-state path is measured.
    SDPX.optimize!(_soc_model(); settings)
    return @allocated SDPX.optimize!(_soc_model(); settings)
end

function main()
    failures = String[]

    total, per_step = _per_step_bytes()
    @printf("per-step allocation      : %6d bytes/step  (ceiling %d)\n",
        per_step, PER_STEP_CEILING_BYTES)
    per_step <= PER_STEP_CEILING_BYTES ||
        push!(failures, "per-step $per_step > $PER_STEP_CEILING_BYTES")

    warm = _warm_solve_bytes()
    @printf("complete warm solve      : %6d bytes     (ceiling %d)\n",
        warm, WARM_SOLVE_CEILING_BYTES)
    warm <= WARM_SOLVE_CEILING_BYTES ||
        push!(failures, "warm solve $warm > $WARM_SOLVE_CEILING_BYTES")

    # The per-step residual must stay bounded relative to the step count, not
    # merely below the ceiling: a leak that grows with steps would pass a
    # per-step average check on a short run.
    _, per_step_long = _per_step_bytes(20)
    @printf("per-step (20-step run)   : %6d bytes/step\n", per_step_long)
    per_step_long <= PER_STEP_CEILING_BYTES ||
        push!(failures, "per-step over 20 steps $per_step_long > $PER_STEP_CEILING_BYTES")

    if isempty(failures)
        @printf("ALLOCATION GATE OK\n")
        return 0
    end
    for failure in failures
        @printf("ALLOCATION GATE FAILED: %s\n", failure)
    end
    return 1
end

exit(main())
