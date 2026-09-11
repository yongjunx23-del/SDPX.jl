# Exact-duplicate equality reduction repro (diagnosis only; no solver changes).
# Run from this worktree with the temporary test environment used by CI:
# julia --gcthreads=1 --startup-file=no --project=. -e \
#   'using Pkg; Pkg.activate(; temp=true); Pkg.develop(; path=pwd()); include("benchmark/robustness/repro_duplicate_row.jl")'
using SDPX

function duplicate_lp(::Type{T}=Float64; perturb=zero(T)) where {T<:AbstractFloat}
    model = SDPX.Model(T; name="exact_duplicate_row_repro")
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    b = one(T) + T(perturb)
    SDPX.constraint!(model, :eq1, x[1] + x[2] - one(T), SDPX.ZeroCone())
    SDPX.constraint!(model, :eq2, x[1] + b * x[2] - b, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + x[2])
    return model
end

function inspect(perturb)
    model = duplicate_lp(Float64; perturb=perturb)
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    reduction = SDPX.hsd_equality_reduce(canonical)
    println("perturb=", perturb,
        " canonical=(n=", SDPX.canonical_num_variables(canonical),
        ",m=", SDPX.canonical_num_slack(canonical), ")",
        " reduction=(status=", reduction.status,
        ",rank=", reduction.rank,
        ",independent=", reduction.independent,
        ",dependent=", reduction.dependent,
        ",rank_tol=", reduction.rank_tolerance, ")")
    println("  x_particular=", reduction.x_particular,
        " null_size=", size(reduction.null_basis),
        " reduced=(n=", SDPX.canonical_num_variables(reduction.reduced),
        ",m=", SDPX.canonical_num_slack(reduction.reduced),
        ",A=", Matrix(reduction.reduced.A),
        ",b=", reduction.reduced.b,
        ",c=", reduction.reduced.c, ")")
    for route in (:bordered, :expanded)
        settings = SDPX.Settings{Float64}(kkt_route=route, verbosity=0,
            limits=SDPX.Limits(iterations=100, time=15.0))
        _, route_reduction, core = SDPX._public_native_hsd_core(
            model, program, SDPX.NativeConeRoute(route), settings)
        println("  route=", route,
            " reduction_rank=", route_reduction.rank,
            " status=", core.status,
            " reason=", core.reason,
            " iterations=", core.iterations,
            " factorizations=", core.factorizations,
            " product_status=", core.product_status)
    end
end

inspect(0.0)
inspect(1.0e-8)
