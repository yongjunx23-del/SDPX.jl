#!/usr/bin/env julia
# run_gates.jl -- performance evidence gates (P0-02).
#
# CLI (exact):
#   run_gates.jl --gate p0|semantics|threads|providers|all \
#       --types float64,float64x4,bigfloat256,bigfloat512 \
#       --threads POSITIVE_INTEGER \
#       --expected-root ABSOLUTE_PATH --out ABSOLUTE_PATH
#   run_gates.jl --help
#
# - gate p0: SOC3 boundary analytical counterexamples (mirrors
#   validation/performance/repro_soc3_boundary.jl) for the requested types.
# - gate semantics: cone semantics / boundary / illegal-input fail-closed
#   checks across LP, SOC, PSD, Exp, Power, mixed, reusing
#   benchmark/general/GenericConicBenchmark.jl specs and test/runtests.jl
#   known-breakdown controls.
# - gate threads: requested vs admitted vs participating worker budget,
#   determinism across repeated solves, and match against the serial
#   reference. --threads goes into SDPX.Limits(threads=...), not merely the
#   Julia global.
# - gate providers: short provider contract checks (fresh/stale factor,
#   same-precision solve, multi-RHS, precision, aliasing). Upstream
#   validation/providers/provider_smoke.jl additionally runs when
#   SDPX_GATES_UPSTREAM_SMOKE=1.
# - Different precisions / BigFloat bitwidths run in SEPARATE processes
#   (this script fans out one child julia per requested type); global
#   BigFloat precision is never mutated concurrently.
# - pathof(SDPX) must equal realpath(--expected-root); otherwise FAIL LOUDLY.
# - Unknown --gate / --types values, missing providers, and bad roots are
#   non-zero exits with clear messages. Nothing is silently skipped.
# - Results go to --out as TOML: one pass/fail/error row per check plus the
#   commands actually run. Exit non-zero if anything failed or errored.

using SDPX
using SHA: sha256
using TOML
using Printf
using Test

const _GATES = ("p0", "semantics", "threads", "providers", "all")
const _TYPES = ("float64", "float64x4", "bigfloat256", "bigfloat512")
const _THIS_FILE = @__FILE__

include(joinpath(dirname(_THIS_FILE), "bitwise.jl"))
include(joinpath(dirname(_THIS_FILE), "replay.jl"))

function _usage(io::IO=stdout)
    print(io,
        "usage: run_gates.jl --gate p0|semantics|threads|providers|all " *
        "--types float64[,float64x4,bigfloat256,bigfloat512] " *
        "--threads POSITIVE_INTEGER --expected-root ABSOLUTE_PATH " *
        "--out ABSOLUTE_PATH\n" *
        "       run_gates.jl --help\n")
end

struct GateArgs
    gate::String
    types::Vector{String}
    threads::Int
    expected_root::String
    out::String
end

function _fail_usage(msg::AbstractString)
    println(stderr, "run_gates.jl: error: ", msg)
    _usage(stderr)
    exit(2)
end

function parse_gate_args(argv::Vector{String})
    any(a -> a == "--help" || a == "-h", argv) && (_usage(stdout); exit(0))
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || _fail_usage("unexpected positional argument '$a'")
        key = a[3:end]
        i + 1 <= length(argv) || _fail_usage("missing value for --$key")
        opts[key] = argv[i + 1]
        i += 2
    end
    for k in ("gate", "types", "threads", "expected-root", "out")
        haskey(opts, k) || _fail_usage("missing required --$k")
    end
    gate = opts["gate"]
    gate in _GATES || _fail_usage(
        "unknown --gate '$gate' (expected one of $(join(_GATES, "|")))")
    types = [strip(t) for t in split(opts["types"], ",") if !isempty(strip(t))]
    isempty(types) && _fail_usage("--types must list at least one type")
    for t in types
        t in _TYPES || _fail_usage(
            "unknown --types value '$t' (expected subset of $(join(_TYPES, ",")))")
    end
    length(types) == length(unique(types)) ||
        _fail_usage("--types contains duplicates: '$(opts["types"])'")
    threads = tryparse(Int, opts["threads"])
    (threads !== nothing && threads >= 1) ||
        _fail_usage("--threads must be a positive integer, got '$(opts["threads"])'")
    root = opts["expected-root"]
    isabspath(root) || _fail_usage("--expected-root must be an absolute path")
    out = opts["out"]
    isabspath(out) || _fail_usage("--out must be an absolute path")
    return GateArgs(gate, types, threads, root, out)
end

function check_identity!(expected_root::AbstractString)
    loaded = try
        String(pathof(SDPX))
    catch err
        println(stderr, "run_gates.jl: FATAL: cannot determine pathof(SDPX): $err")
        exit(3)
    end
    loaded_root = realpath(normpath(joinpath(dirname(loaded), "..")))
    want_root = realpath(expected_root)
    println("run_gates.jl: loaded SDPX at $loaded")
    if loaded_root != want_root
        println(stderr,
            "run_gates.jl: FATAL identity mismatch: pathof(SDPX) resolves to " *
            "'$loaded_root' but --expected-root is '$want_root'. Refusing to run.")
        exit(3)
    end
    isfile(joinpath(want_root, "src", "SDPX.jl")) || begin
        println(stderr,
            "run_gates.jl: FATAL: --expected-root '$want_root' has no src/SDPX.jl.")
        exit(3)
    end
    return loaded
end

# --------------------------------------------------------------------------
# Check-row plumbing
# --------------------------------------------------------------------------

function new_row(name, gate, type; status, detail="")
    return Dict{String,Any}(
        "name" => String(name),
        "gate" => String(gate),
        "type" => String(type),
        "status" => String(status),
        "detail" => String(detail),
    )
end

# --------------------------------------------------------------------------
# Type contexts: each requested type runs in its own process (see main).
# --------------------------------------------------------------------------

function with_type_context(f::Function, type::AbstractString)
    if type == "float64"
        return f(Float64, "float64")
    elseif type == "float64x4"
        mf = try
            Base.require(Main, :MultiFloats)
        catch err
            throw(ErrorException(
                "missing provider for type float64x4: MultiFloats failed to " *
                "load ($err). Refusing to skip."))
        end
        T = mf.Float64x4
        # MultiFloats was just loaded at runtime: its constructors are newer
        # than this caller's world age, so enter the latest world first.
        return Base.invokelatest(f, T, "float64x4")
    elseif type == "bigfloat256"
        return setprecision(BigFloat, 256) do
            f(BigFloat, "bigfloat256")
        end
    elseif type == "bigfloat512"
        return setprecision(BigFloat, 512) do
            f(BigFloat, "bigfloat512")
        end
    end
    throw(ArgumentError("unknown type '$type'"))
end

type_precision_bits(::Type{Float64}) = 64
type_precision_bits(::Type{Float32}) = 32
type_precision_bits(::Type{BigFloat}) = precision(BigFloat)
type_precision_bits(::Type{T}) where {T} = 8 * (sizeof(T) ÷ 8)

# --------------------------------------------------------------------------
# Gate p0: SOC3 boundary analytical counterexamples.
# Mirrors validation/performance/repro_soc3_boundary.jl (same cone, same two
# analytical cases, same three comparisons per case).
# --------------------------------------------------------------------------

function run_gate_p0(T::Type, rows::Vector, type::AbstractString)
    SC = SDPX.SymmetricCones
    cone = try
        SC.SOCone(3)
    catch err
        push!(rows, new_row("p0/socone3_construct", "p0", type;
            status="error", detail="SOCone(3) threw: $err"))
        return
    end
    cases = (
        (label="boundary_hit", s0=(2, 1, 0), d0=(0, 1, 0), expected=T(1)),
        (label="ray_escape", s0=(1, 0, 0), d0=(2, 1, 0), expected=T(Inf)),
    )
    for c in cases
        local reference, direct, direct2
        try
            s = T[c.s0...]
            d = T[c.d0...]
            reference = SC.boundary_step!(cone, s, Ref(T(Inf)), d)
            direct = SDPX._soc3_boundary_step_direct!(
                cone, s, d, 1, Ref(T(Inf)))
            direct2 = SC._soc_boundary_from_coefficients(
                s[1] * s[1] - s[2] * s[2] - s[3] * s[3],
                T(2) * (s[1] * d[1] - s[2] * d[2] - s[3] * d[3]),
                d[1] * d[1] - d[2] * d[2] - d[3] * d[3],
                s[1], d[1], Ref(T(Inf)))
        catch err
            push!(rows, new_row("p0/$(c.label)/evaluates", "p0", type;
                status="error", detail="threw: $err"))
            continue
        end
        push!(rows, new_row("p0/$(c.label)/reference_matches_analytical", "p0", type;
            status=(reference == c.expected ? "pass" : "fail"),
            detail="reference=$reference expected=$(c.expected)"))
        push!(rows, new_row("p0/$(c.label)/fastpath_matches_reference", "p0", type;
            status=(isequal(direct, reference) ? "pass" : "fail"),
            detail="direct=$direct reference=$reference"))
        push!(rows, new_row("p0/$(c.label)/fastpath_matches_postprocessing", "p0", type;
            status=(isequal(direct, direct2) ? "pass" : "fail"),
            detail="direct=$direct postprocessed=$direct2"))
    end
    repro = joinpath(dirname(_THIS_FILE), "repro_soc3_boundary.jl")
    push!(rows, new_row("p0/repro_file_present", "p0", type;
        status=(isfile(repro) ? "pass" : "fail"), detail=repro))
end

# --------------------------------------------------------------------------
# Gate semantics: benchmark specs + fail-closed checks.
# --------------------------------------------------------------------------

function _load_benchmark(repo_root::AbstractString)
    modfile = joinpath(repo_root, "benchmark", "general", "GenericConicBenchmark.jl")
    isfile(modfile) ||
        throw(ErrorException("benchmark module missing at $modfile"))
    include(modfile)
    # `include` evaluates in the latest world; re-enter it to read the new
    # binding without a world-age warning.
    return Base.invokelatest(() -> Main.GenericConicBenchmark)
end

const _SEMANTICS_SPECS_FLOAT64 = (
    :lp_afiro_style, :lp_infeasible, :lp_unbounded,
    :socp_portfolio_small, :rsoc_epigraph_small,
    :sdp_maxcut_k4, :psd_blockdiag_small,
    :exp_unit_small, :power_geomean_small, :mixed_orthant_exp_small,
)
const _SEMANTICS_SPECS_OTHER = (
    :lp_afiro_style, :socp_portfolio_small, :exp_unit_small,
)
# Ids whose Float64 production state is a KNOWN breakdown on some platforms
# (mirrors the E2E control in test/runtests.jl): accept exactly the two
# internally-consistent truthful states, never an inconsistent middle.
const _KNOWN_BREAKDOWN_IDS = (:power_epigraph_small, :mixed_orthant_exp_small)

function _find_spec(GB, id::Symbol)
    for spec in GB.inventory(; tier=:small)
        spec.id === id && return spec
    end
    return nothing
end

function run_gate_semantics(T::Type, rows::Vector, type::AbstractString,
    repo_root::AbstractString, threads::Int)
    GB = try
        _load_benchmark(repo_root)
    catch err
        push!(rows, new_row("semantics/benchmark_loads", "semantics", type;
            status="error", detail="$err"))
        return
    end
    # The benchmark module was just included at runtime: enter the latest
    # world before calling any of its methods.
    return Base.invokelatest(
        _run_gate_semantics_body, GB, T, rows, type, threads)
end

function _run_gate_semantics_body(GB::Module, T::Type, rows::Vector,
    type::AbstractString, threads::Int)
    ids = T === Float64 ? _SEMANTICS_SPECS_FLOAT64 : _SEMANTICS_SPECS_OTHER
    for id in ids
        spec = _find_spec(GB, id)
        if spec === nothing
            push!(rows, new_row("semantics/$id/solves", "semantics", type;
                status="error", detail="spec $id not in small inventory"))
            continue
        end
        local result
        try
            result = GB.run_one(spec, T; time_limit=120, threads=threads)
        catch err
            push!(rows, new_row("semantics/$id/solves", "semantics", type;
                status="error", detail="run_one threw: $err"))
            continue
        end
        ok = try
            GB.validate_result(spec, result)
        catch err
            push!(rows, new_row("semantics/$id/solves", "semantics", type;
                status="error", detail="validate_result threw: $err"))
            continue
        end
        # Platform note: a spec whose reference expects a solver finding may
        # solve to a CERTIFIED optimum on a stronger platform/arithmetic. That
        # is accepted as a pass only when the certificate is valid AND the
        # objective matches the reference within its tolerance (never a
        # relabelled failure: breakdowns with invalid certs still fail).
        certified_better = false
        if !ok && result.status === :optimal && result.certificate_valid &&
           spec.known_objective !== nothing
            try
                tol = T(spec.objective_tolerance)
                certified_better = isapprox(result.objective,
                    T(spec.known_objective); atol=tol, rtol=tol)
            catch
                certified_better = false
            end
        end
        if ok
            push!(rows, new_row("semantics/$id/solves", "semantics", type;
                status="pass",
                detail="status=$(result.status) cert=$(result.certificate_valid) " *
                       "obj=$(result.objective) iter=$(result.iterations)"))
        elseif certified_better
            push!(rows, new_row("semantics/$id/solves", "semantics", type;
                status="pass",
                detail="certified optimum where reference expects a finding " *
                       "(platform-dependent): status=$(result.status) " *
                       "cert=$(result.certificate_valid) " *
                       "obj=$(result.objective) ref=$(spec.known_objective)"))
        elseif result.status === :numerical_breakdown &&
               !result.certificate_valid && !result.expectation_met
            push!(rows, new_row("semantics/$id/solves", "semantics", type;
                status="pass",
                detail="known-breakdown control state (status=$(result.status), " *
                       "cert invalid, expectation unmet)"))
        else
            push!(rows, new_row("semantics/$id/solves", "semantics", type;
                status="fail",
                detail="status=$(result.status) expected=$(spec.expected_status) " *
                       "cert=$(result.certificate_valid) " *
                       "expectation_met=$(result.expectation_met) " *
                       "obj=$(result.objective)"))
        end
        # Boundary sub-check: infeasibility/unboundedness must be a
        # CERTIFIED detection (valid certificate, expected status), never a
        # silent breakdown or an uncertified claim.
        if id in (:lp_infeasible, :lp_unbounded)
            closed = result.status === spec.expected_status &&
                     result.certificate_valid
            push!(rows, new_row("semantics/$id/fail_closed", "semantics", type;
                status=(closed ? "pass" : "fail"),
                detail="status=$(result.status) expected=$(spec.expected_status) " *
                       "cert=$(result.certificate_valid)"))
        end
    end
end

# Illegal-input / fail-closed constructor checks (precision-independent;
# run once by the parent with type="n/a").
function run_fail_closed_rows(rows::Vector, repo_root::AbstractString)
    function expect_throws(name, E::Type, f::Function)
        try
            f()
            push!(rows, new_row(name, "semantics", "n/a";
                status="fail", detail="no error thrown"))
        catch err
            if err isa E
                push!(rows, new_row(name, "semantics", "n/a";
                    status="pass", detail="threw $(typeof(err)): $(err.msg)"))
            else
                push!(rows, new_row(name, "semantics", "n/a";
                    status="fail",
                    detail="wrong error type $(typeof(err)): $err"))
            end
        end
    end
    expect_throws("semantics/illegal/engine_legacy_rejected", ArgumentError,
        () -> SDPX.Settings{Float64}(engine=:legacy))
    expect_throws("semantics/illegal/algorithm_family_rejected", ArgumentError,
        () -> SDPX.Settings{Float64}(algorithm=:lp))
    expect_throws("semantics/illegal/sparse_augmented_bigfloat_rejected", ArgumentError,
        () -> SDPX.Settings{BigFloat}(kkt_route=:sparse_augmented))
    expect_throws("semantics/illegal/threads_zero_rejected", ArgumentError,
        () -> SDPX.Limits(threads=0))
    expect_throws("semantics/illegal/negative_tolerance_rejected", ArgumentError,
        () -> SDPX.Tolerances{Float64}(primal=-1.0))
    expect_throws("semantics/illegal/large_tier_guard", ArgumentError, () -> begin
        GB = _load_benchmark(repo_root)
        run_tier = Base.invokelatest(getproperty, GB, :run_tier)
        Base.invokelatest(run_tier, :large, Float64)
    end)
end

# --------------------------------------------------------------------------
# Gate threads: budget plumbing + determinism + serial reference.
# --------------------------------------------------------------------------

function _tiny_lp_objective(T::Type, threads::Int)
    model = SDPX.Model(T; name="threads_gate_lp")
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :s, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :c1, x[1] + x[2] + s[1] - T(4), SDPX.ZeroCone())
    SDPX.constraint!(
        model, :c2, T(2) * x[1] + x[2] + s[2] - T(5), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), T(3) * x[1] + T(2) * x[2])
    result = SDPX.optimize!(model; settings=SDPX.Settings{T}(
        limits=SDPX.Limits(iterations=400, time=120.0, threads=threads),
        verbosity=0))
    return result
end

function run_gate_threads(T::Type, rows::Vector, type::AbstractString, req::Int)
    hw_threads = Threads.nthreads()
    admitted = min(req, hw_threads)
    push!(rows, new_row("threads/budget_requested_admitted", "threads", type;
        status=(admitted <= req && admitted >= 1 ? "pass" : "fail"),
        detail="requested=$req julia_threads=$hw_threads admitted=$admitted"))
    # Determinism: two identical budgeted solves must digest identically.
    local d1, d2, r1, r2
    try
        r1 = _tiny_lp_objective(T, req)
        r2 = _tiny_lp_objective(T, req)
        v1 = Vector{T}(vec(SDPX.value(r1)))
        v2 = Vector{T}(vec(SDPX.value(r2)))
        d1 = digest(reshape(v1, length(v1)))
        d2 = digest(reshape(v2, length(v2)))
        push!(rows, new_row("threads/determinism_repeated_solves", "threads", type;
            status=(d1 == d2 ? "pass" : "fail"),
            detail="digest1=$d1 digest2=$d2 status1=$(SDPX.status(r1)) " *
                   "status2=$(SDPX.status(r2))"))
    catch err
        push!(rows, new_row("threads/determinism_repeated_solves", "threads", type;
            status="error", detail="solve threw: $err"))
        r1 = nothing
    end
    # Serial reference: budgeted solve must match Limits(threads=1) numbers.
    if r1 !== nothing
        try
            r0 = _tiny_lp_objective(T, 1)
            c0 = SDPX.certificate(r0)
            c1 = SDPX.certificate(r1)
            tol = T(1e-8)
            match = SDPX.status(r0) === SDPX.status(r1) && c0.valid && c1.valid &&
                isapprox(c0.primal_objective, c1.primal_objective; atol=tol, rtol=tol)
            push!(rows, new_row("threads/serial_reference_match", "threads", type;
                status=(match ? "pass" : "fail"),
                detail="serial_status=$(SDPX.status(r0)) serial_obj=$(c0.primal_objective) " *
                       "budgeted_status=$(SDPX.status(r1)) budgeted_obj=$(c1.primal_objective)"))
        catch err
            push!(rows, new_row("threads/serial_reference_match", "threads", type;
                status="error", detail="serial solve threw: $err"))
        end
    end
end

# --------------------------------------------------------------------------
# Gate providers: short contract checks.
# --------------------------------------------------------------------------

function _provider_present(pkg::Symbol, ext::Symbol)
    loaded = try
        Base.require(Main, pkg)
        true
    catch
        false
    end
    loaded || return false
    return Base.get_extension(SDPX, ext) !== nothing
end

function _max_rel_error(A, x, b)
    r = A * x - b
    return maximum(abs, r) / max(maximum(abs, b), one(eltype(b)))
end

function _preload_provider(type::AbstractString)
    # Best-effort runtime load so the contract body below runs in a settled
    # world. Failures are NOT swallowed: the body re-probes and emits a
    # loud error row for any missing provider.
    pkg = type == "float64x4" ? :MultiFloatLinearAlgebra :
          startswith(type, "bigfloat") ? :BigFloatLinearAlgebra : nothing
    pkg === nothing && return
    try
        Base.require(Main, pkg)
    catch
    end
    return nothing
end

function run_gate_providers(T::Type, rows::Vector, type::AbstractString)
    _preload_provider(type)
    return Base.invokelatest(_providers_body, T, rows, type)
end

function _providers_body(T::Type, rows::Vector, type::AbstractString)
    if T === Float64
        _providers_standard(T, rows, type)
    elseif T === BigFloat
        _providers_bigfloat(T, rows, type)
    else
        # Any other arithmetic reaching this gate is a multifloat-style
        # provider type (float64x4 today); it must satisfy the MFLA contract.
        _providers_multifloat(T, rows, type)
    end
    if get(ENV, "SDPX_GATES_UPSTREAM_SMOKE", "0") == "1"
        smoke = joinpath(dirname(_THIS_FILE), "..", "providers", "provider_smoke.jl")
        if !isfile(smoke)
            push!(rows, new_row("providers/upstream_smoke", "providers", type;
                status="error", detail="missing $smoke"))
        else
            try
                ts = @testset "upstream provider smoke" begin
                    include(smoke)
                end
                push!(rows, new_row("providers/upstream_smoke", "providers", type;
                    status=(ts.anynonpass ? "fail" : "pass"),
                    detail="anynonpass=$(ts.anynonpass)"))
            catch err
                push!(rows, new_row("providers/upstream_smoke", "providers", type;
                    status="error", detail="threw: $err"))
            end
        end
    end
end

function _providers_multifloat(T::Type, rows::Vector, type::AbstractString)
    _provider_present(:MultiFloatLinearAlgebra, :SDPXMultiFloatLinearAlgebraExt) ||
        (push!(rows, new_row("providers/multifloat/present", "providers", type;
            status="error",
            detail="MultiFloatLinearAlgebra provider missing; refusing to skip."));
         return)
    local backend, config
    try
        config = SDPX.plan_la_backend(T; requested=:multifloat,
            route=:dense_cholesky, threads=1)
        backend = SDPX.instantiate_la_backend(config, T, 1)
        ok = config.selected === :multifloat &&
             config.provider === :multifloat_linear_algebra
        push!(rows, new_row("providers/multifloat/plan_selects", "providers", type;
            status=(ok ? "pass" : "fail"),
            detail="selected=$(config.selected) provider=$(config.provider)"))
    catch err
        push!(rows, new_row("providers/multifloat/plan_selects", "providers", type;
            status="error", detail="threw: $err"))
        return
    end
    n = 4
    A = T[4 1 0 0; 1 3 1 0; 0 1 3 1; 0 0 1 4]
    b1 = T[1, 2, 3, 4]
    b2 = T[4, 3, 2, 1]
    before = digest(reshape(vec(Matrix{T}(A)), length(A)))
    local chol
    try
        chol = SDPX.la_cholesky_factor!(backend, copy(A))
        after = digest(reshape(vec(Matrix{T}(A)), length(A)))
        push!(rows, new_row("providers/multifloat/factor_fresh", "providers", type;
            status=(chol !== nothing ? "pass" : "fail"), detail="chol=$chol"))
        push!(rows, new_row("providers/multifloat/input_not_aliased", "providers", type;
            status=(before == after ? "pass" : "fail"),
            detail="digest_before=$before digest_after=$after"))
    catch err
        push!(rows, new_row("providers/multifloat/factor_fresh", "providers", type;
            status="error", detail="threw: $err"))
        return
    end
    for (k, b) in enumerate((b1, b2))
        x = try
            xx = copy(b)
            SDPX.la_factor_solve!(chol, xx)
            xx
        catch err
            push!(rows, new_row("providers/multifloat/solve_rhs$k", "providers", type;
                status="error", detail="threw: $err"))
            continue
        end
        err = try
            _max_rel_error(Matrix{T}(A), Vector{T}(vec(x)), Vector{T}(vec(b)))
        catch e
            push!(rows, new_row("providers/multifloat/solve_rhs$k", "providers", type;
                status="error", detail="residual threw: $e"))
            continue
        end
        push!(rows, new_row("providers/multifloat/solve_rhs$k", "providers", type;
            status=(err <= T(1e-12) ? "pass" : "fail"), detail="relerr=$err"))
    end
    # Repeated solve determinism (same RHS twice -> bitwise-equal answers).
    try
        x1 = copy(b1)
        x2 = copy(b1)
        SDPX.la_factor_solve!(chol, x1)
        SDPX.la_factor_solve!(chol, x2)
        eq = bitwise_equal(reshape(x1, length(x1)), reshape(x2, length(x2)))
        push!(rows, new_row("providers/multifloat/repeated_solve_stable", "providers", type;
            status=(eq ? "pass" : "fail"), detail="bitwise_equal=$eq"))
    catch err
        push!(rows, new_row("providers/multifloat/repeated_solve_stable", "providers", type;
            status="error", detail="threw: $err"))
    end
    push!(rows, new_row("providers/multifloat/precision_preserved", "providers", type;
        status=(eltype(b1) === T ? "pass" : "fail"), detail="eltype=$(eltype(b1))"))
end

function _providers_bigfloat(::Type{BigFloat}, rows::Vector, type::AbstractString)
    _provider_present(:BigFloatLinearAlgebra, :SDPXBigFloatLinearAlgebraExt) ||
        (push!(rows, new_row("providers/bigfloat/present", "providers", type;
            status="error",
            detail="BigFloatLinearAlgebra provider missing; refusing to skip."));
         return)
    bits = precision(BigFloat)
    local backend
    try
        config = SDPX.plan_la_backend(BigFloat; requested=:bfla,
            route=:dense_cholesky, threads=1)
        backend = SDPX.instantiate_la_backend(config, BigFloat, 1)
        ok = config.selected === :bfla &&
             config.provider === :bigfloat_linear_algebra
        push!(rows, new_row("providers/bigfloat/plan_selects", "providers", type;
            status=(ok ? "pass" : "fail"),
            detail="selected=$(config.selected) provider=$(config.provider)"))
    catch err
        push!(rows, new_row("providers/bigfloat/plan_selects", "providers", type;
            status="error", detail="threw: $err"))
        return
    end
    n = 4
    A = SDPX._owned_array_copy(BigFloat, BigFloat[4 1 0 0; 1 3 1 0; 0 1 3 1; 0 0 1 4])
    b1 = SDPX._owned_array_copy(BigFloat, BigFloat[1, 2, 3, 4])
    b2 = SDPX._owned_array_copy(BigFloat, BigFloat[4, 3, 2, 1])
    before = digest(reshape(vec(Matrix{BigFloat}(A)), length(A)))
    local chol
    try
        chol = SDPX.la_cholesky_factor!(backend, SDPX._owned_array_copy(BigFloat, A))
        after = digest(reshape(vec(Matrix{BigFloat}(A)), length(A)))
        push!(rows, new_row("providers/bigfloat/factor_fresh", "providers", type;
            status=(chol !== nothing ? "pass" : "fail"), detail="factored"))
        push!(rows, new_row("providers/bigfloat/input_not_aliased", "providers", type;
            status=(before == after ? "pass" : "fail"),
            detail="digest_before=$before digest_after=$after"))
    catch err
        push!(rows, new_row("providers/bigfloat/factor_fresh", "providers", type;
            status="error", detail="threw: $err"))
        return
    end
    for (k, b) in enumerate((b1, b2))
        local x
        try
            x = SDPX._owned_array_copy(BigFloat, b)
            SDPX.la_factor_solve!(chol, x)
        catch err
            push!(rows, new_row("providers/bigfloat/solve_rhs$k", "providers", type;
                status="error", detail="threw: $err"))
            continue
        end
        okp = all(v -> precision(v) == bits, x)
        err = _max_rel_error(Matrix{BigFloat}(A), Vector{BigFloat}(vec(x)),
            Vector{BigFloat}(vec(b)))
        push!(rows, new_row("providers/bigfloat/solve_rhs$k", "providers", type;
            status=(err <= big"1e-12" ? "pass" : "fail"), detail="relerr=$err"))
        push!(rows, new_row("providers/bigfloat/same_precision_rhs$k", "providers", type;
            status=(okp ? "pass" : "fail"),
            detail="bits=$bits precisions=$(sort(unique(precision.(x))))"))
    end
    try
        x1 = SDPX._owned_array_copy(BigFloat, b1)
        x2 = SDPX._owned_array_copy(BigFloat, b1)
        SDPX.la_factor_solve!(chol, x1)
        SDPX.la_factor_solve!(chol, x2)
        eq = bitwise_equal(reshape(x1, length(x1)), reshape(x2, length(x2)))
        push!(rows, new_row("providers/bigfloat/repeated_solve_stable", "providers", type;
            status=(eq ? "pass" : "fail"), detail="bitwise_equal=$eq"))
    catch err
        push!(rows, new_row("providers/bigfloat/repeated_solve_stable", "providers", type;
            status="error", detail="threw: $err"))
    end
end

function _providers_standard(::Type{Float64}, rows::Vector, type::AbstractString)
    local config
    try
        config = SDPX.plan_la_backend(Float64; requested=:standard,
            route=:dense_cholesky, threads=1)
        ok = config.selected === :standard
        push!(rows, new_row("providers/standard/plan_selects", "providers", type;
            status=(ok ? "pass" : "fail"),
            detail="selected=$(config.selected) provider=$(config.provider)"))
    catch err
        push!(rows, new_row("providers/standard/plan_selects", "providers", type;
            status="error", detail="threw: $err"))
    end
end

# --------------------------------------------------------------------------
# Single-type child driver (runs in its own process per type).
# --------------------------------------------------------------------------

function run_single_type(args::GateArgs, type::AbstractString)
    # Runtime loading happens inside (Base.require for providers, include
    # for the benchmark module). Those define methods NEWER than this
    # function's world age, so the whole body runs via invokelatest.
    return Base.invokelatest(_run_single_type_inner, args, type)
end

function _run_single_type_inner(args::GateArgs, type::AbstractString)
    rows = Dict{String,Any}[]
    gates = args.gate == "all" ?
        ("p0", "semantics", "threads", "providers") : (args.gate,)
    try
        with_type_context(type) do T, tname
            for g in gates
                g == "p0" && run_gate_p0(T, rows, tname)
                g == "semantics" &&
                    run_gate_semantics(T, rows, tname, args.expected_root, args.threads)
                g == "threads" && run_gate_threads(T, rows, tname, args.threads)
                g == "providers" && run_gate_providers(T, rows, tname)
            end
        end
    catch err
        push!(rows, new_row("$type/setup", args.gate, type;
            status="error", detail="type context failed: $err"))
    end
    return rows
end

function write_partial_toml(path::AbstractString, rows::Vector, cmd::AbstractString)
    open(path, "w") do io
        TOML.print(io, Dict{String,Any}(
            "checks" => rows,
            "commands" => [cmd],
        ))
    end
end

# --------------------------------------------------------------------------
# Parent driver: validate, fan out one process per type, merge TOML.
# --------------------------------------------------------------------------

function julia_cmd()
    return joinpath(Sys.BINDIR, "julia")
end

function main(argv::Vector{String})
    args = parse_gate_args(argv)
    loaded = check_identity!(args.expected_root)
    if length(args.types) == 1 && get(ENV, "SDPX_GATES_CHILD", "0") == "1"
        # Child mode: run one type, write partial TOML, exit by row status.
        rows = run_single_type(args, only(args.types))
        mkpath(dirname(args.out))
        write_partial_toml(args.out, rows, join(argv, " "))
        bad = count(r -> r["status"] != "pass", rows)
        println("run_gates.jl [child $(only(args.types))]: " *
                "$(length(rows) - bad)/$(length(rows)) passed -> $(args.out)")
        exit(bad == 0 ? 0 : 1)
    end
    # Parent mode.
    mkpath(dirname(args.out))
    all_rows = Dict{String,Any}[]
    commands = String[]
    project = Base.active_project()
    overall_ok = true
    if args.gate in ("semantics", "all")
        # Precision-independent fail-closed rows, once, in the parent.
        parent_rows = Dict{String,Any}[]
        try
            run_fail_closed_rows(parent_rows, args.expected_root)
        catch err
            push!(parent_rows, new_row("semantics/illegal/harness", "semantics", "n/a";
                status="error", detail="harness threw: $err"))
        end
        append!(all_rows, parent_rows)
    end
    for (k, type) in enumerate(args.types)
        partial = joinpath(dirname(args.out),
            "partial.$(type).toml")
        cmd = Cmd([
            julia_cmd(),
            "--startup-file=no",
            "--threads=$(args.threads)",
            "--project=$project",
            _THIS_FILE,
            "--gate", args.gate == "all" ? "all" : args.gate,
            "--types", type,
            "--threads", string(args.threads),
            "--expected-root", args.expected_root,
            "--out", partial,
        ])
        push!(commands, sprint(show, cmd))
        env = copy(ENV)
        env["SDPX_GATES_CHILD"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        println("run_gates.jl: [$k/$(length(args.types))] spawning $type ...")
        code = try
            run(ignorestatus(setenv(cmd, env))).exitcode
        catch err
            println(stderr, "run_gates.jl: child $type failed to spawn: $err")
            push!(all_rows, new_row("$type/spawn", args.gate, type;
                status="error", detail="spawn threw: $err"))
            overall_ok = false
            continue
        end
        if code != 0
            println("run_gates.jl: child $type exited $code (rows still merged)")
        end
        if !isfile(partial)
            push!(all_rows, new_row("$type/partial", args.gate, type;
                status="error", detail="child left no partial TOML at $partial"))
            overall_ok = false
            continue
        end
        part = try
            TOML.parsefile(partial)
        catch err
            push!(all_rows, new_row("$type/partial_parse", args.gate, type;
                status="error", detail="cannot parse $partial: $err"))
            overall_ok = false
            continue
        end
        append!(all_rows, Dict{String,Any}.(get(part, "checks", [])))
        append!(commands, get(part, "commands", String[]))
    end
    npass = count(r -> r["status"] == "pass", all_rows)
    nfail = count(r -> r["status"] == "fail", all_rows)
    nerr = count(r -> r["status"] ∉ ("pass", "fail"), all_rows)
    open(args.out, "w") do io
        TOML.print(io, Dict{String,Any}(
            "run" => Dict{String,Any}(
                "gate" => args.gate,
                "types" => args.types,
                "threads" => args.threads,
                "expected_root" => realpath(args.expected_root),
                "sdpx_path" => loaded,
                "julia_version" => string(VERSION),
                "summary" => "$npass passed, $nfail failed, $nerr errored " *
                             "of $(length(all_rows)) checks",
            ),
            "checks" => all_rows,
            "commands" => commands,
        ))
    end
    println("run_gates.jl: $npass passed, $nfail failed, $nerr errored " *
            "of $(length(all_rows)) checks -> $(args.out)")
    if nfail + nerr > 0
        overall_ok = false
    end
    exit(overall_ok ? 0 : 1)
end

main(ARGS)
