# PR-04B: start-point policy comparison with the accounting the plan requires.
#
# Plan Section 5, PR-04B is explicit that the candidates are *different startup
# equations* and must be compared as separate policies, and its acceptance
# criterion is explicit too: "统计 T_start + T_iteration + T_final_cert，同时
# 比较迭代数与总时间", with "只有总成本/稳健性有代表性改善后才更新 auto".
#
# ## Why the comparison goes through the public API
#
# The two starts are reached through the public `kkt_route`, because
# `product_cone_solve.jl` binds `initialization === :auto` to the route:
#
#     bordered, sparse_augmented -> :identity
#     expanded, sparse_schur     -> :kkt
#
# Driving `product_hsd_solve!` directly with `initialization=:kkt` was tried
# first and REJECTED: it bypasses equilibration and formulation selection, and
# on these cases BOTH policies then fail (LP: `line_search_breakdown` at 194
# iterations for identity and `direction_breakdown` at 16 for kkt, objective
# -214 instead of -21). Reporting "kkt is 12x faster" from that would have been
# a comparison of two failures. The public routes are the ones that actually
# solve, so they are the ones compared.
#
# ## Confound, stated rather than hidden
#
# Route and start are not independently selectable on the public surface today,
# so a route change also changes the executor (bordered sparse core vs expanded
# dense). This receipt therefore measures the *route-and-start pair*, which is
# exactly the decision `auto` makes. Isolating the start alone would require a
# new public knob, and PR-04B must not add public surface while comparing.
#
#   julia --startup-file=no --project=. benchmark/clarabel_borrowing/start_point_comparison.jl
#
# Writes `benchmark/clarabel_borrowing/start_point_comparison.toml`. Changes no
# default.

using SDPX
using Dates
using TOML
using Printf
using Statistics: median
using LinearAlgebra: dot

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", ".."))

function _head_sha()
    try
        return strip(read(`git -C $REPO rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

struct StartCase
    id::Symbol
    family::Symbol
    build::Function
    expected_objective::Float64
end

function _lp_afiro_style()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :slack, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :capacity_1, x[1] + x[2] + s[1] - 4.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :capacity_2, 2.0 * x[1] + x[2] + s[2] - 5.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), 3.0 * x[1] + 2.0 * x[2])
    return model
end

function _soc_disk()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _soc_k64()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 63; domain=SDPX.Reals())
    SDPX.constraint!(model, :cone, Any[1.0; collect(x)], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _soc_many_small()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 6; domain=SDPX.Reals())
    for block in 1:3
        SDPX.constraint!(
            model, Symbol(:cone, block),
            Any[1.0, x[2 * block - 1], x[2 * block]], SDPX.LorentzCone(),
        )
    end
    SDPX.objective!(model, SDPX.Minimize(),
        -(1.0 / 3.0) * (x[1] + x[3] + x[5]))
    return model
end

function _psd_2x2()
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :trace, X[1, 1] + X[2, 2] - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    return model
end

function _psd_blockdiag()
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 4, 4; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :trace,
        X[1, 1] + X[2, 2] + X[3, 3] + X[4, 4] - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1] + X[3, 3])
    return model
end

const CASES = (
    StartCase(:lp_afiro_style, :lp, _lp_afiro_style, -21.0),
    StartCase(:soc_disk, :soc, _soc_disk, -1.0),
    StartCase(:soc_k64, :soc, _soc_k64, -1.0),
    StartCase(:soc_many_small, :soc, _soc_many_small, -1.0),
    StartCase(:psd_2x2, :psd, _psd_2x2, 0.0),
    StartCase(:psd_blockdiag, :psd, _psd_blockdiag, 0.0),
)

"""Route/start pairs compared. The start each route selects is recorded."""
const POLICIES = (
    (name=:identity, route=:bordered),
    (name=:kkt, route=:expanded),
)

function _run_policy(case::StartCase, route::Symbol, repeats::Int)
    rows = NamedTuple[]
    # Warmup pass, not timed. Every case here runs in well under a second, so
    # the first call is dominated by compilation and would otherwise appear as
    # a large spurious win for whichever policy happens to run first. The
    # receipt records that warmup happened; it is not part of the medians.
    try
        SDPX.optimize!(case.build(); settings=SDPX.Settings(Float64;
            kkt_route=route, verbosity=0,
            limits=SDPX.Limits(iterations=400, time=180.0, threads=1)))
    catch
        # A warmup failure is handled by the timed loop, which records it.
    end
    for _ in 1:repeats
        started = time_ns()
        local result
        try
            result = SDPX.optimize!(case.build(); settings=SDPX.Settings(Float64;
                kkt_route=route, verbosity=0,
                limits=SDPX.Limits(iterations=400, time=180.0, threads=1)))
        catch exception
            push!(rows, (
                seconds=Float64(time_ns() - started) * 1.0e-9,
                status="threw", reason=sprint(showerror, exception),
                iterations=-1, factorizations=-1, backtracking=-1,
                certificate_valid=false, objective_error=NaN,
                start_seconds=NaN, iteration_seconds=NaN, cert_seconds=NaN,
                executed_route=:not_executed,
            ))
            continue
        end
        elapsed = Float64(time_ns() - started) * 1.0e-9
        diagnostics = SDPX.diagnostics(result)
        timings = diagnostics.timings
        termination = diagnostics.termination
        certificate = SDPX.certificate(result)

        # The plan's T_start + T_iteration + T_final_cert split, from the
        # engine's own phase counters.
        cert_seconds = Float64(get(timings, :certification_seconds, NaN))
        iteration_seconds = Float64(get(timings, :direction_seconds, 0.0)) +
                            Float64(get(timings, :line_search_seconds, 0.0)) +
                            Float64(get(timings, :residual_seconds, 0.0)) +
                            Float64(get(timings, :scaling_seconds, 0.0)) +
                            Float64(get(timings, :accepted_update_seconds, 0.0))
        start_seconds = Float64(get(timings, :schur_assembly_seconds, 0.0))

        push!(rows, (
            seconds=elapsed,
            status=String(SDPX.status(result)),
            reason=String(termination.reason),
            iterations=Int(termination.iterations),
            factorizations=Int(termination.factorizations),
            backtracking=Int(termination.backtracking),
            certificate_valid=Bool(certificate.valid),
            objective_error=abs(Float64(certificate.primal_objective) -
                                case.expected_objective),
            start_seconds=start_seconds,
            iteration_seconds=iteration_seconds,
            cert_seconds=cert_seconds,
            executed_route=Symbol(diagnostics.selected_algorithms.executed_kkt_route),
        ))
    end
    return rows
end

_SUCCESS = ("optimal",)

function _summarize(rows)
    solved = filter(row -> row.status in _SUCCESS && row.certificate_valid, rows)
    summary = (
        runs=length(rows),
        solved=length(solved),
        failures=length(rows) - length(solved),
        statuses=join(sort(unique(row.status for row in rows)), ","),
        reasons=join(sort(unique(row.reason for row in rows)), ","),
    )
    # Timing columns are taken over the SUCCESSFUL repeats only, and the receipt
    # records how many there were, so a fast failure can never masquerade as a
    # speed win. Cases where both policies fail get no ratio at all.
    isempty(solved) && return merge(summary, (
        seconds=NaN, iterations=-1, factorizations=-1, backtracking=-1,
        start_seconds=NaN, iteration_seconds=NaN, cert_seconds=NaN,
        objective_error=NaN, executed_routes="",
    ))
    return merge(summary, (
        seconds=median(row.seconds for row in solved),
        iterations=median(row.iterations for row in solved),
        factorizations=median(row.factorizations for row in solved),
        backtracking=median(row.backtracking for row in solved),
        start_seconds=median(row.start_seconds for row in solved),
        iteration_seconds=median(row.iteration_seconds for row in solved),
        cert_seconds=median(row.cert_seconds for row in solved),
        objective_error=maximum(row.objective_error for row in solved),
        executed_routes=join(sort(unique(String(row.executed_route) for row in solved)), ","),
    ))
end

function main()
    repeats = something(tryparse(Int, get(ENV, "SDPX_START_COMPARISON_REPEATS", "5")), 5)
    repeats >= 1 || (repeats = 1)
    @printf("PR-04B start-point comparison — HEAD %s, %d repeats\n",
        _head_sha(), repeats)
    @printf("  identity start = kkt_route :bordered | kkt start = kkt_route :expanded\n")

    entries = Dict{String,Any}[]
    both_solved = 0
    for case in CASES
        entry = Dict{String,Any}(
            "id" => String(case.id), "family" => String(case.family),
        )
        for policy in POLICIES
            summary = _summarize(_run_policy(case, policy.route, repeats))
            for (key, value) in pairs(summary)
                entry["$(policy.name)_$(key)"] = value
            end
            entry["$(policy.name)_route"] = String(policy.route)
        end
        identity_ok = get(entry, "identity_solved", 0) > 0
        kkt_ok = get(entry, "kkt_solved", 0) > 0
        if identity_ok && kkt_ok
            both_solved += 1
            entry["comparable"] = true
            entry["kkt_seconds_ratio"] = entry["kkt_seconds"] / entry["identity_seconds"]
            entry["kkt_iterations_delta"] = entry["kkt_iterations"] - entry["identity_iterations"]
            entry["kkt_start_seconds_delta"] = entry["kkt_start_seconds"] - entry["identity_start_seconds"]
            entry["kkt_cert_seconds_delta"] = entry["kkt_cert_seconds"] - entry["identity_cert_seconds"]
        else
            # No ratio for a case one side failed: that is the plan's explicit
            # rule, not an omission.
            entry["comparable"] = false
        end
        push!(entries, entry)
        @printf("  %-16s id: %-9s %6.3fs %4d it | kkt: %-9s %6.3fs %4d it | ratio %s\n",
            entry["id"],
            entry["identity_statuses"], get(entry, "identity_seconds", NaN),
            entry["identity_iterations"],
            entry["kkt_statuses"], get(entry, "kkt_seconds", NaN),
            entry["kkt_iterations"],
            get(entry, "comparable", false) ?
                @sprintf("%.3f", entry["kkt_seconds_ratio"]) : "n/a (unequal success)")
    end

    payload = Dict{String,Any}(
        "schema" => "sdpx-start-point-comparison/1",
        "generated" => string(now()),
        "sdpx_head" => _head_sha(),
        "julia_version" => string(VERSION),
        "repeats" => repeats,
        "warmup" => "one untimed warmup solve per (case, policy) before timing",
        "identity_policy" => "kkt_route=:bordered selects initialization=:identity",
        "kkt_policy" => "kkt_route=:expanded selects initialization=:kkt",
        "confound" => "route and start are not independently selectable on the " *
                      "public surface, so this measures the route-and-start pair " *
                      "that `auto` actually chooses",
        "note" => "Timing medians are over SUCCESSFUL repeats only; `solved` and " *
                  "`failures` are reported per policy and no ratio is emitted " *
                  "unless both policies solved. No default is changed here.",
        "cases" => entries,
        "comparable_cases" => both_solved,
        "total_cases" => length(CASES),
    )
    target = joinpath(HERE, "start_point_comparison.toml")
    open(target, "w") do io
        TOML.print(io, payload; sorted=true)
    end
    @printf("wrote %s (%d/%d cases comparable)\n",
        target, both_solved, length(CASES))
    return 0
end

exit(main())
