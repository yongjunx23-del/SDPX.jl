# Q01 — case manifest and result schema.
#
# Packet requirement: "建立真实输入shape trace与完整结果schema；允许统计失败，
# 不删除慢/失败案例."
#
# ## The rule this file exists to enforce
#
# ADR-003 §6 and the Q01 card both forbid reading a benchmark NAME to decide a
# numeric route. The manifest therefore describes only *inputs*. It has no field
# for a route, a formulation, a provider or a strategy, and `case_settings`
# constructs `Settings` from tolerances and limits alone — never from the case
# id. A regression test in `test/rebuild/dependency_rules.jl` asserts that this
# file contains no route-selection vocabulary, so the rule is checked rather
# than promised.
#
# ## Failures are data
#
# A case that fails is recorded with its outcome. Nothing here filters, skips or
# deletes on failure — `summarize` counts failures and keeps every row.

module RebuildManifest

using SDPX

export RebuildCase, rebuild_cases, result_row, failure_row, summarize,
       cases_fingerprint, case_settings, RESULT_SCHEMA, shape_trace

"""
    RebuildCase

An input descriptor. Deliberately contains no strategy field.

- `id`: stable identifier, used for reporting only, never for dispatch.
- `family`: `:lp`/`:soc`/`:psd`/`:exp`/`:power`/`:mixed` — describes the cone,
  not the algorithm.
- `n`, `m`: regression and cone dimensions, recorded as a **shape trace** so a
  manifest row can be compared across revisions without re-solving.
- `cones`: the ordered cone block signature, e.g. `[(:soc, 3), (:nonneg, 4)]`.
- `build`: constructs the model. Pure input construction.
- `known_objective`: an independently known optimum, or `nothing`. `nothing`
  means "no oracle" and is recorded as such — it is never 0.
"""
struct RebuildCase
    id::Symbol
    family::Symbol
    n::Int
    m::Int
    cones::Vector{Tuple{Symbol,Int}}
    build::Function
    known_objective::Union{Nothing,Float64}
end

shape_trace(case::RebuildCase) = (
    id=case.id, family=case.family, n=case.n, m=case.m,
    cone_count=length(case.cones),
    cones=case.cones,
)

# ---------------------------------------------------------------------------
# Case builders. Every one is a pure input constructor.
# ---------------------------------------------------------------------------

function _lp_afiro_style()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :slack, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :capacity_1, x[1] + x[2] + s[1] - 4.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :capacity_2, 2.0 * x[1] + x[2] + s[2] - 5.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), 3.0 * x[1] + 2.0 * x[2])
    return model
end

function _lp_degenerate()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :slack, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :dup_1, x[1] + x[2] + s[1] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :dup_2, x[1] + x[2] + s[2] - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + x[2])
    return model
end

function _soc_disk()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _soc_large(k::Int)
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, k - 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :cone, Any[1.0; collect(x)], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _soc_many_small()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 6; domain=SDPX.Reals())
    for block in 1:3
        SDPX.constraint!(model, Symbol(:cone, block),
            Any[1.0, x[2 * block - 1], x[2 * block]], SDPX.LorentzCone())
    end
    SDPX.objective!(model, SDPX.Minimize(), -(1.0 / 3.0) * (x[1] + x[3] + x[5]))
    return model
end

function _psd_2x2()
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :trace, X[1, 1] + X[2, 2] - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    return model
end

function _mixed_cones()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 3; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
    SDPX.constraint!(model, :orth, Any[1.0 + x[3]], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), -x[1] - 0.5 * x[2])
    return model
end

"""
    rebuild_cases() -> Vector{RebuildCase}

The deterministic case set. Ordered by `id` so a manifest is reproducible.

Slow and degenerate cases are INCLUDED, per the packet rule that a manifest may
count failures but may not delete them.
"""
function rebuild_cases()
    cases = RebuildCase[
        RebuildCase(:lp_afiro_style, :lp, 4, 2,
            [(:nonneg, 2), (:nonneg, 2), (:zero, 2)], _lp_afiro_style, -21.0),
        RebuildCase(:lp_degenerate, :lp, 4, 2,
            [(:nonneg, 2), (:nonneg, 2), (:zero, 2)], _lp_degenerate, 1.0),
        RebuildCase(:soc_disk, :soc, 2, 3,
            [(:zero, 2), (:soc, 3)], _soc_disk, -1.0),
        RebuildCase(:soc_k32, :soc, 31, 32,
            [(:zero, 31), (:soc, 32)], () -> _soc_large(32), -1.0),
        RebuildCase(:soc_k128, :soc, 127, 128,
            [(:zero, 127), (:soc, 128)], () -> _soc_large(128), -1.0),
        RebuildCase(:soc_many_small, :soc, 6, 9,
            [(:zero, 6), (:soc, 3), (:soc, 3), (:soc, 3)], _soc_many_small, -1.0),
        RebuildCase(:psd_2x2, :psd, 3, 1,
            [(:zero, 3), (:psd, 2)], _psd_2x2, 0.0),
        RebuildCase(:mixed_soc_nonneg, :mixed, 3, 4,
            [(:zero, 3), (:soc, 3), (:nonneg, 1)], _mixed_cones, nothing),
    ]
    sort!(cases; by=case -> case.id)
    return cases
end

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

"""
    RESULT_SCHEMA

The complete set of fields one case run produces. Every field is always present;
a field that could not be measured is `nothing`, never `0` (ADR-003 §3).
"""
const RESULT_SCHEMA = (
    # identity
    :id, :family, :family_cone_count,
    # outcome — recorded whatever it is, including failure
    :status, :certificate_valid, :certificate_method,
    :objective, :objective_error, :iterations, :factorizations, :backtracking,
    :termination_reason,
    # executed route facts, read from the receipt rather than assumed
    :requested_kkt_route, :executed_kkt_route, :executed_kkt_storage,
    :executed_factorization_kernel, :fallback_reason,
    # diagnostics
    :sigma_used, :alpha_aff, :alpha_combined, :correction_norm, :retry_reason,
    # cost
    :seconds, :allocated_bytes, :rss_delta_bytes,
    # failure bookkeeping
    :threw, :exception,
)

"""
    result_row(case, result; seconds, allocated_bytes, rss_delta_bytes) -> NamedTuple

Project a solve result into `RESULT_SCHEMA`. A field the engine does not publish
is `nothing` with the reason recorded in `unavailable`, never a fabricated zero.
"""
function result_row(case::RebuildCase, result;
    seconds::Float64, allocated_bytes::Int, rss_delta_bytes::Int,
)
    unavailable = Symbol[]
    diagnostics = try
        SDPX.diagnostics(result)
    catch
        nothing
    end
    termination = diagnostics === nothing ? nothing : diagnostics.termination
    selected = diagnostics === nothing ? nothing : diagnostics.selected_algorithms

    get_or_nothing(source, field) = begin
        if source === nothing || !hasproperty(source, field)
            push!(unavailable, field)
            return nothing
        end
        value = getproperty(source, field)
        value isa Symbol ? String(value) : value
    end

    certificate = SDPX.certificate(result)
    objective = Float64(certificate.primal_objective)
    return (
        id=case.id,
        family=case.family,
        family_cone_count=length(case.cones),
        status=String(SDPX.status(result)),
        certificate_valid=Bool(certificate.valid),
        certificate_method=String(certificate.method),
        objective=objective,
        # `nothing` when there is no independent oracle, never 0.
        objective_error=case.known_objective === nothing ? nothing :
                        abs(objective - case.known_objective),
        iterations=get_or_nothing(termination, :iterations),
        factorizations=get_or_nothing(termination, :factorizations),
        backtracking=get_or_nothing(termination, :backtracking),
        termination_reason=get_or_nothing(termination, :reason),
        requested_kkt_route=get_or_nothing(selected, :requested_kkt_route),
        executed_kkt_route=get_or_nothing(selected, :executed_kkt_route),
        executed_kkt_storage=get_or_nothing(selected, :executed_kkt_storage),
        executed_factorization_kernel=get_or_nothing(selected, :executed_factorization_kernel),
        fallback_reason=get_or_nothing(selected, :fallback_reason),
        sigma_used=get_or_nothing(termination, :sigma_used),
        alpha_aff=get_or_nothing(termination, :alpha_aff),
        alpha_combined=get_or_nothing(termination, :alpha_combined),
        correction_norm=get_or_nothing(termination, :correction_norm),
        retry_reason=get_or_nothing(termination, :retry_reason),
        seconds=seconds,
        allocated_bytes=allocated_bytes,
        rss_delta_bytes=rss_delta_bytes,
        threw=false,
        exception=nothing,
    ), unavailable
end

"""A row for a case whose construction or solve threw. Kept, not dropped."""
function failure_row(case::RebuildCase, exception; phase::Symbol, seconds::Float64)
    return (
        id=case.id, family=case.family,
        family_cone_count=length(case.cones),
        status="threw", certificate_valid=false, certificate_method="none",
        objective=nothing, objective_error=nothing,
        iterations=nothing, factorizations=nothing, backtracking=nothing,
        termination_reason=nothing,
        requested_kkt_route=nothing, executed_kkt_route=nothing,
        executed_kkt_storage=nothing, executed_factorization_kernel=nothing,
        fallback_reason=nothing,
        sigma_used=nothing, alpha_aff=nothing, alpha_combined=nothing,
        correction_norm=nothing, retry_reason=nothing,
        seconds=seconds, allocated_bytes=nothing, rss_delta_bytes=nothing,
        threw=true, exception="$(phase): $(sprint(showerror, exception))",
    )
end

"""
    case_settings(case; kwargs...) -> Settings

Build `Settings` from tolerances and limits ONLY.

This function must never branch on `case.id`, `case.family` or any name. That is
the packet's no-name-dispatch rule, and `test/rebuild/dependency_rules.jl`
asserts it structurally.
"""
function case_settings(case::RebuildCase;
    primal::Float64=1e-8, dual::Float64=1e-8, gap::Float64=1e-8,
    iterations::Int=400, time::Float64=180.0, threads::Int=1,
)
    return SDPX.Settings(Float64;
        verbosity=0,
        tolerances=SDPX.Tolerances(Float64; primal=primal, dual=dual, gap=gap),
        limits=SDPX.Limits(iterations=iterations, time=time, threads=threads),
    )
end

"""
    summarize(rows) -> NamedTuple

Count successes and failures. **Nothing is filtered out**: `rows` is returned in
full alongside the counts, so a caller cannot accidentally report only the
successes.
"""
function summarize(rows)
    total = length(rows)
    solved = count(row -> !row.threw && row.status == "optimal" &&
                          row.certificate_valid === true, rows)
    return (
        total=total,
        solved=solved,
        failed=total - solved,
        threw=count(row -> row.threw, rows),
        without_certificate=count(row -> !row.threw &&
            row.certificate_valid !== true, rows),
        rows=collect(rows),
    )
end

"""
    cases_fingerprint(cases) -> UInt64

Stable hash over the input shape trace only. Two manifests with the same
fingerprint describe the same inputs, so a rerun is comparable (Q01 acceptance
item 1). It deliberately does not hash `build`, which is a closure.
"""
function cases_fingerprint(cases)
    h = UInt64(0xcbf29ce484222325)
    for case in cases
        for value in (UInt64(case.n), UInt64(case.m),
                      UInt64(hash(case.id)), UInt64(hash(case.family)))
            h = (h ⊻ value) * UInt64(0x100000001b3)
        end
        for (name, size) in case.cones
            h = (h ⊻ UInt64(hash(name))) * UInt64(0x100000001b3)
            h = (h ⊻ UInt64(size)) * UInt64(0x100000001b3)
        end
    end
    return h
end

end # module
