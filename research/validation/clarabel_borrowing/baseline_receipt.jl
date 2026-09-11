# PR-00 baseline receipt: a small, deterministic, runnable cost record.
#
# The plan requires that a baseline be captured *before* any optimization, with
# the exact SHAs, provider versions and hardware recorded, and with per-phase
# accounting rather than a single wall-clock number (Section 7.4). This script
# produces that record for a set of small analytic cases so it is cheap to rerun
# and comparable across commits.
#
# It deliberately records failures rather than filtering them: a row that fails
# here must stay in the table, because "the new version deleted the failing
# row" is exactly the outcome the plan forbids.
#
#   julia --startup-file=no --project=. validation/clarabel_borrowing/baseline_receipt.jl
#
# Writes `validation/clarabel_borrowing/baseline_receipt.json`.

using SDPX
using Dates
using TOML
using Printf
using LinearAlgebra: BLAS

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", ".."))

"""`git rev-parse HEAD`, or `"unknown"` outside a checkout."""
function _head_sha()
    try
        return strip(read(`git -C $REPO rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

"""Whether the tracked worktree has uncommitted changes."""
function _worktree_dirty()
    try
        return !isempty(strip(read(`git -C $REPO status --porcelain`, String)))
    catch
        return true
    end
end

"""One analytic case with a known optimum and an original-coordinate certificate."""
struct BaselineCase
    id::Symbol
    family::Symbol
    build::Function
    expected_objective::Float64
end

function _soc_unit_disk()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _lp_simple()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :sum, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + 2 * x[2])
    return model
end

function _soc_scaled(k::Int)
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, k - 1; domain=SDPX.Reals())
    SDPX.constraint!(
        model, :cone, Any[1.0; collect(x)], SDPX.LorentzCone(),
    )
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _psd_2x2()
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :trace, X[1, 1] + X[2, 2] - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    return model
end

const CASES = (
    BaselineCase(:lp_simple, :lp, _lp_simple, 1.0),
    BaselineCase(:soc_unit_disk, :soc, _soc_unit_disk, -1.0),
    BaselineCase(:soc_k32, :soc, () -> _soc_scaled(32), -1.0),
    BaselineCase(:soc_k128, :soc, () -> _soc_scaled(128), -1.0),
    BaselineCase(:psd_2x2, :psd, _psd_2x2, 0.0),
)

"""Run one case and project the receipt fields the plan asks for."""
function _receipt(case::BaselineCase)
    row = Dict{String,Any}(
        "id" => String(case.id),
        "family" => String(case.family),
    )
    started = time_ns()
    local result
    try
        model = case.build()
        settings = SDPX.Settings(Float64;
            verbosity=0,
            limits=SDPX.Limits(iterations=500, time=120.0, threads=1),
        )
        # The native HSD engine publishes neither iteration history nor a
        # performance trace, so request only what it does publish. Asking for
        # `history`/`trace` is a hard error, not a silent downgrade.
        outputs = SDPX.Outputs(
            :all, :all, :all; objectives=true, certificate=:full,
            diagnostics=:full, history=false, trace=false,
        )
        result = SDPX.optimize!(model; settings, outputs)
    catch exception
        row["status"] = "threw"
        row["exception"] = sprint(showerror, exception)
        row["seconds"] = Float64(time_ns() - started) * 1.0e-9
        return row
    end
    row["seconds"] = Float64(time_ns() - started) * 1.0e-9
    row["status"] = String(SDPX.status(result))
    certificate = SDPX.certificate(result)
    row["certificate_valid"] = Bool(certificate.valid)
    row["certificate_method"] = String(certificate.method)
    row["primal_objective"] = Float64(certificate.primal_objective)
    row["dual_objective"] = Float64(certificate.dual_objective)
    row["objective_error"] =
        abs(Float64(certificate.primal_objective) - case.expected_objective)
    # The native engine does not publish per-iteration history; record that
    # honestly with an explicit sentinel rather than substituting a different
    # quantity or omitting the field.
    row["iterations"] = "not_published_by_engine"

    # Effective route facts: the plan wants the *executed* route, not the
    # requested one, because those differ under fallback.
    selected = SDPX.diagnostics(result).selected_algorithms
    for field in (
        :requested_kkt_route, :planned_kkt_route, :executed_kkt_route,
        :executed_kkt_storage, :planned_factorization_kernel,
        :executed_factorization_kernel, :la_executed_provider,
        :equilibration, :fallback_reason,
    )
        value = hasproperty(selected, field) ? getproperty(selected, field) : nothing
        row[String(field)] = value === nothing ? "absent" : String(value)
    end

    # Phase timings: the public trace is not published by this engine, so they
    # are not available here. Recorded as absent rather than omitted silently.
    row["phase_timings_available"] = false
    return row
end

function main()
    @printf("SDPX baseline receipt — HEAD %s (dirty=%s)\n",
        _head_sha(), _worktree_dirty())
    rows = Dict{String,Any}[]
    failures = String[]
    for case in CASES
        row = _receipt(case)
        push!(rows, row)
        ok = get(row, "certificate_valid", false) === true
        ok || push!(failures, String(case.id))
        @printf("  %-16s %-10s cert=%-5s obj_err=%-10.3g iters=%-5s %.3fs\n",
            row["id"], row["status"],
            string(get(row, "certificate_valid", "n/a")),
            get(row, "objective_error", NaN),
            string(get(row, "iterations", "-")),
            row["seconds"])
    end
    payload = Dict{String,Any}(
        "schema" => "sdpx-baseline-receipt/1",
        "generated" => string(now()),
        "sdpx_head" => _head_sha(),
        "worktree_dirty" => _worktree_dirty(),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "precision_bits" => precision(Float64),
        "cases" => rows,
        "failures" => failures,
        "case_count" => length(rows),
        "failure_count" => length(failures),
    )
    target = joinpath(HERE, "baseline_receipt.toml")
    open(target, "w") do io
        TOML.print(io, payload; sorted=true)
    end
    @printf("wrote %s (%d cases, %d failures)\n",
        target, length(rows), length(failures))
    return isempty(failures) ? 0 : 1
end

exit(main())
