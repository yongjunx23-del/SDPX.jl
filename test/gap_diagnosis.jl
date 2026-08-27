# WAVE-D gap diagnosis harness — per-iteration trace of the two remaining Phase 0 gaps.
#
# Env-gated (like SDPX_RUN_KNOWN_GAPS): runs only when SDPX_RUN_GAP_DIAGNOSIS=1.
# Drives the internal ProductConeHSDState manually so we can observe tau/kappa/mu
# and residuals per iteration for :bordered and :expanded routes.
#
# FINDING (see docs/design/GAP_DIAGNOSIS_EXPANDED.md): both remaining gaps converge
# in primal/dual residuals (rP/rD/rG -> ~1e-14) but collapse tau -> denormal while
# kappa stays O(1), i.e. the HSD iterate reaches the tau=0 infeasibility face instead
# of the optimal face. The working bounded-capped probe keeps tau=2.0 and kappa->0.
using SDPX, Test, LinearAlgebra

const _ENV_GATE = "SDPX_RUN_GAP_DIAGNOSIS"

function _p0_mixed_free_psd()
    model = SDPX.Model(Float64)
    t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())
    M = SDPX.variable!(model, :M, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :link, M[1, 1] - t[1], SDPX.ZeroCone())
    SDPX.constraint!(
        model, :upper_psd,
        [1.0 - M[1, 1] -M[1, 2]; -M[1, 2] 1.0 - M[2, 2]],
        SDPX.PSDCone(),
    )
    SDPX.objective!(model, SDPX.Maximize(), t[1])
    return model
end

function _p0_bounded_nonpositive()
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :sum, y[1] + y[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :lower_one, y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :lower_two, y[2], SDPX.Nonnegative())
    SDPX.constraint!(model, :upper_one, y[1] - 1.0, SDPX.Nonpositive())
    SDPX.constraint!(model, :upper_two, y[2] - 1.0, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), y[1] + 2.0 * y[2])
    return model
end

function _p0_bounded_capped()  # WORKING contrast case
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :sum, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :upper_one, x[1] - 1.0, SDPX.Nonpositive())
    SDPX.constraint!(model, :upper_two, x[2] - 1.0, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + 2.0 * x[2])
    return model
end

struct _Traj
    name::String
    route::Symbol
    tau_end::Float64
    kappa_end::Float64
    mu_end::Float64
    rp_end::Float64
    rd_end::Float64
    tau_collapsed::Bool
end

function _trace(name, model; maxiter=40)
    out = _Traj[]
    prog = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(prog)
    red = SDPX.hsd_equality_reduce(canon)
    redc = red.reduced
    redc === nothing && return out
    for route in (:bordered, :expanded)
        st = SDPX.ProductConeHSDState(redc; kkt_route=route)
        SDPX.kkt_derived_start!(st)
        b = st.base
        for _ in 1:maxiter
            SDPX.product_hsd_step!(st)
        end
        push!(out, _Traj(
            name, route, b.tau, b.kappa, b.mu,
            maximum(abs, b.rP), maximum(abs, b.rD),
            b.tau < 1e-12,
        ))
    end
    return out
end

@testset "WAVE-D gap diagnosis" begin
    if get(ENV, _ENV_GATE, "0") == "1"
        for (name, build) in [
            ("mixed_free_psd", _p0_mixed_free_psd),
            ("bounded_nonpositive", _p0_bounded_nonpositive),
            ("bounded_capped_contrast", _p0_bounded_capped),
        ]
            for t in _trace(name, build())
                println("DIAG ", t.name, " route=", t.route,
                        " tau=", t.tau_end, " kappa=", t.kappa_end,
                        " mu=", t.mu_end, " rP=", t.rp_end, " rD=", t.rd_end,
                        " tau_collapsed=", t.tau_collapsed)
            end
        end
        # Structural assertions: broken probes collapse tau; contrast keeps it.
        broken = vcat(
            _trace("mixed_free_psd", _p0_mixed_free_psd()),
            _trace("bounded_nonpositive", _p0_bounded_nonpositive()),
        )
        @test all(t.tau_collapsed for t in broken)
        contrast = _trace("bounded_capped_contrast", _p0_bounded_capped())
        @test all(!t.tau_collapsed for t in contrast)
    end
end