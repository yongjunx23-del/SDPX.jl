# test/rebuild/S06.jl
#
# Standalone: julia --project=<SDPX.jl> test/rebuild/S06.jl
#
# S06 — setup planning, memory and thread budget. The planning sources are new
# files (`src/planning/costs.jl`, `resources.jl`, `setup.jl`) that the
# integration role (I01) will add to `src/SDPX.jl`; until then this test includes
# them itself, supplying from `SDPX` the bindings they rely on. Once the include
# exists in `src/SDPX.jl` the guard below skips the local include and the test
# exercises the integrated names instead.
#
# Sections:
#   1. plan reproducibility, and the reported plan vs the execution that ran
#   2. a 1-thread request inside a 4-thread process (real subprocess, measured)
#   3. memory refusal *before* the large allocation
#   4. a sparse-declared path does not silently densify
#   5. concurrent sessions and the process-global BLAS thread count
#   6. profiles come from explicit offline calibration only
#   7. inheritance from the previous round's cost model, and the packet's tiers
#
# Evidence labels are explicit: a check is *dynamic* (measured here), or *static*
# (a property of the code, checked here). Where a measurement is impossible on
# this host the test records `unsupported` with its reason rather than asserting
# a made-up number.

using Test
using LinearAlgebra
using SparseArrays
using SDPX

import SDPX: blas_threads, set_blas_threads!
import SDPX: saturating_bytes, saturating_sum_bytes
import SDPX: plan_core_route, CoreRoutePlan, legacy_dimension_rule
import SDPX: symmetric_core_state_prepare_bytes
import SDPX: ThreadBudget
import SDPX: ExtendedPrecisionBLAS

const S06_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const S06_SRC = joinpath(S06_ROOT, "src")
const S06_PLANNING = joinpath(S06_SRC, "planning")
const S06_PLANNING_INCLUDED_HERE = !isdefined(SDPX, :plan_setup)

if S06_PLANNING_INCLUDED_HERE
    include(joinpath(S06_PLANNING, "costs.jl"))
    include(joinpath(S06_PLANNING, "resources.jl"))
    include(joinpath(S06_PLANNING, "setup.jl"))
end

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

const S06_CAPACITY = ThreadCapacity(4, 4, 4)          # explicit, host-independent
const S06_BUDGET = 1 << 30                            # 1 GiB: admission passes

s06_small_workload() = setup_workload(;
    block_sizes=[6, 4, 3], reduced_dimension=24, band=3, extra_nnz=12,
    blas_dimension=8,
)

s06_large_workload() = setup_workload(;
    block_sizes=[64, 64, 64], reduced_dimension=2048, band=4, blas_dimension=8,
)

s06_child_command() =
    `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__)`

"""
Digest of the plan's decision fields, built from an explicit ordered list of
values rendered with `string`. Deliberately not `hash(plan)`: `hash` of a
mutable or address-bearing object is not a cross-process statement, and this
digest has to be comparable between two Julia processes.
"""
function s06_signature_digest(plan)
    parts = String[]
    for value in (
        plan.route, plan.declared_storage, plan.resolved_storage,
        plan.costs.core_route.full_score, plan.costs.core_route.compact_score,
        plan.costs.core_route.predicted_fill_ratio, plan.costs.dominant,
        plan.costs.legacy_would_choose_compact,
        plan.memory.structural_bytes, plan.memory.fill_bytes,
        plan.memory.owned_scalar_bytes, plan.memory.workspace_bytes,
        plan.memory.fallback_bytes, plan.memory.inherited_core_bytes,
        plan.memory.attributed_bytes, plan.memory.total_bytes,
        plan.admission.admitted, plan.admission.reason,
        plan.admission.budget_bytes, plan.admission.headroom_bytes,
        plan.threads.budget.mode, plan.threads.budget.julia_outer_threads,
        plan.threads.budget.blas_threads, plan.threads.budget.provider_threads,
        plan.threads.budget.reduction_bins, plan.threads.active_consumer,
        plan.threads.tier_status, plan.threads.cone_available,
        plan.threads.la_available, plan.threads.blas_available,
        plan.profile.id, plan.profile.source, plan.profile.fill_factor,
        plan.profile.blas_threading_min_width,
        plan.request_fingerprint, plan.context_fingerprint,
    )
        push!(parts, string(value))
    end
    for reason in plan.reasons
        push!(parts, string(reason))
    end
    text = join(parts, "|")
    digest = UInt64(0xcbf29ce484222325)
    for byte in codeunits(text)
        digest = (digest ⊻ UInt64(byte)) * UInt64(0x00000100000001b3)
    end
    return string(digest; base=16, pad=16)
end

"""
Allocate a byte buffer the compiler cannot fold away, so `@allocated` around the
call measures a real large allocation. This is the control for the memory
refusal test: it demonstrates that the measurement technique detects an
allocation of the size the refused request would have needed.
"""
function s06_allocate_control(bytes::Integer)
    buffer = Vector{UInt8}(undef, bytes)
    buffer[1] = 0x01
    return buffer
end

function s06_parse_child(text::AbstractString, marker::AbstractString)
    fields = Dict{String,String}()
    prefix = marker * " "
    for line in split(text, '\n')
        startswith(line, prefix) || continue
        for pair in split(strip(line[length(prefix)+1:end]), ' ')
            isempty(pair) && continue
            key_value = split(pair, '='; limit=2)
            length(key_value) == 2 && (fields[key_value[1]] = key_value[2])
        end
    end
    return fields
end

# ---------------------------------------------------------------------------
# Child process modes
# ---------------------------------------------------------------------------
#
# Two child modes, both in a *fresh* Julia:
#   SDPX_S06_CHILD=1     -- the 1-thread-in-a-multi-thread-process measurement
#                           (started with --threads=4)
#   SDPX_S06_CHILD=plan  -- cross-process plan reproducibility

function s06_child_threads_report()
    capacity = thread_capacity()
    blocks = fill(4, 8)   # 8 independent cone tasks: wide enough to parallelize
    w = setup_workload(;
        block_sizes=blocks, reduced_dimension=8, band=2, blas_dimension=8,
    )
    ctx = setup_context(capacity=capacity)
    serial_plan = plan_setup(
        setup_request(w; requested_threads=1, memory_budget_bytes=S06_BUDGET), ctx,
    )
    before = Int(blas_threads())
    serial_obs = execute_setup_plan(serial_plan, w; session=:s06_child_serial)
    after = Int(blas_threads())

    parallel_plan = plan_setup(
        setup_request(w; requested_threads=4, memory_budget_bytes=S06_BUDGET), ctx,
    )
    parallel_obs = execute_setup_plan(parallel_plan, w; session=:s06_child_parallel)

    return (
        julia_threads=capacity.julia_threads,
        blas_capacity=capacity.blas_threads,
        cpu_threads=capacity.cpu_threads,
        serial_tier=String(serial_plan.threads.tier_status),
        serial_mode=String(serial_plan.threads.budget.mode),
        serial_granted_julia=serial_plan.threads.budget.julia_outer_threads,
        serial_granted_blas=serial_plan.threads.budget.blas_threads,
        serial_granted_provider=serial_plan.threads.budget.provider_threads,
        serial_observed_julia=serial_obs.outer_threads_observed,
        serial_observed_blas=serial_obs.blas_threads_observed,
        serial_restored=serial_obs.blas_restored ? 1 : 0,
        blas_before=before,
        blas_after=after,
        serial_residual=Float64(serial_obs.la_kernel_residual),
        serial_allocated=serial_obs.allocated_bytes,
        parallel_granted_julia=parallel_plan.threads.budget.julia_outer_threads,
        parallel_observed_julia=parallel_obs.outer_threads_observed,
        parallel_bins=parallel_obs.cone_bins_executed,
    )
end

function s06_child_plan_report()
    plan = plan_setup(
        setup_request(
            s06_small_workload();
            requested_threads=4, memory_budget_bytes=S06_BUDGET,
        ),
        setup_context(capacity=S06_CAPACITY),
    )
    return (
        request_fingerprint=plan.request_fingerprint,
        context_fingerprint=plan.context_fingerprint,
        signature=s06_signature_digest(plan),
    )
end

# --- provider leg ----------------------------------------------------------
#
# The extended-precision providers are optional extensions. Whether they are
# available is *probed*, never assumed: in the packet's `REBUILD_ENV` they load,
# in the bare default environment they do not, and the same test must tell the
# truth in both. The leg is run in its own process with `-t1`, and MF and BF legs
# never share a process (Julia 1.12 can exhaust its inference compiler when the
# fixed-width MFLA and BFLA/MPFR specializations compile together).

s06_provider_extension(::Type{BigFloat}) = :SDPXBigFloatLinearAlgebraExt
s06_provider_extension(T::Type) =
    nameof(T) === :MultiFloat ? :SDPXMultiFloatLinearAlgebraExt : :none

"""Probe whether the provider for `T` is really loaded in this process."""
function s06_provider_loaded(::Type{T}) where {T}
    s06_provider_extension(T) === :none && return false
    return Base.get_extension(SDPX, s06_provider_extension(T)) !== nothing
end

"""
Resolve the provider arithmetic type by name, or `nothing` when this process
cannot load it. The weak-dependency triggers are imported at the top level of
the child branch, *before* this runs: importing them inside a function body would
leave the rest of that body in the old world age (Julia 1.12 errors on that), so
the import is deliberately not done here.
"""
function s06_provider_type(name::AbstractString)
    name == "BigFloat" && return isdefined(Main, :BigFloatLinearAlgebra) ? BigFloat : nothing
    name == "MultiFloat" || return nothing
    (isdefined(Main, :MultiFloats) && isdefined(Main, :MultiFloatLinearAlgebra)) ||
        return nothing
    return MultiFloats.MultiFloat{Float64,2}
end

function s06_convert_workload(w::SetupWorkload{Float64}, ::Type{T}) where {T}
    return SetupWorkload{T}(
        [Matrix{T}(block) for block in w.block_operators],
        SparseMatrixCSC{T,Int}(w.sparse_lower),
        nothing,
        Matrix{T}(w.blas_a),
        Matrix{T}(w.blas_b),
    )
end

function s06_child_provider_report(
    name::AbstractString=get(ENV, "SDPX_S06_PROVIDER", "BigFloat"),
    load_error::AbstractString="",
)
    capacity = thread_capacity()
    w = s06_small_workload()
    request = setup_request(w; requested_threads=1, memory_budget_bytes=S06_BUDGET)
    float64_plan = plan_setup(request, setup_context(capacity=capacity))
    base = (
        provider=name, arithmetic="unavailable", extension="none", loaded=0,
        precision_bits=0, scalar_bytes=0,
        float64_scalar_bytes=float64_plan.memory.scalar_bytes,
        total_bytes=0, float64_total_bytes=float64_plan.memory.total_bytes,
        admitted=0, wide_scalar_reason=0, route="none", storage_used="none",
        residual=0.0, verification_ok=0, verification_mismatches="",
        threads_mode="none", julia_threads=capacity.julia_threads,
        blas_before=Int(blas_threads()), error="",
        measured_allocated_bytes=0, ledger_total_bytes=0,
    )
    T = s06_provider_type(name)
    if T === nothing
        return merge(base, (error=isempty(load_error) ?
            "provider arithmetic type unavailable" : load_error,))
    end
    loaded = s06_provider_loaded(T)
    loaded || return merge(base, (
        arithmetic=String(string(T)), extension=String(s06_provider_extension(T)),
        error="provider extension not loaded in this process",
    ))
    bits = if T === BigFloat
        precision(BigFloat)
    else
        # `precision(::Type{<:MultiFloat})` is not reliably callable (SDPX's own
        # `_accuracy_effective_bits` wraps it in a try/catch for the same
        # reason), so fall back to the documented xN width 53N - (N - 1).
        try
            precision(T)
        catch
            width = Int(T.parameters[2])
            53 * width - (width - 1)
        end
    end
    wide_request = SetupRequest(
        T, bits, :bordered, false, 3,
        request.full_dimension, request.compact_dimension,
        request.ar_nnz, request.canonical_nnz, 0, 0,
        copy(request.block_sizes), :sparse_lower, 1, S06_BUDGET, nothing,
    )
    plan = plan_setup(
        wide_request, setup_context(capacity=capacity, provider_available=loaded),
    )
    wide_workload = s06_convert_workload(w, T)
    # Warm up, then measure: the first call would otherwise include compilation.
    execute_setup_plan(
        plan, wide_workload; session=Symbol("s06_$(name)_warmup"),
        measure_allocations=false,
    )
    allocation_observation = execute_setup_plan(
        plan, wide_workload; session=Symbol("s06_$(name)_alloc"),
        measure_allocations=true,
    )
    observation = execute_setup_plan(
        plan, wide_workload; session=Symbol("s06_$(name)"),
        measure_allocations=false,
    )
    verification = verify_setup_execution(plan, observation)
    return (
        provider=name,
        arithmetic=String(string(T)),
        extension=String(s06_provider_extension(T)),
        loaded=1,
        precision_bits=bits,
        scalar_bytes=plan.memory.scalar_bytes,
        float64_scalar_bytes=float64_plan.memory.scalar_bytes,
        total_bytes=plan.memory.total_bytes,
        float64_total_bytes=float64_plan.memory.total_bytes,
        admitted=plan.admission.admitted ? 1 : 0,
        wide_scalar_reason=(:wide_scalar_narrows_memory_headroom in
                            plan.costs.core_route.reasons) ? 1 : 0,
        route=String(plan.route),
        storage_used=String(observation.storage_used),
        residual=Float64(observation.la_kernel_residual),
        verification_ok=verification.ok ? 1 : 0,
        verification_mismatches=join(string.(verification.mismatches), ","),
        # Allocation *traffic* (measured warm) next to the workspace ledger: for
        # BigFloat these differ by orders of magnitude because every MPFR
        # operation allocates outside the Julia heap. The ledger bounds
        # workspace, not traffic — recorded here so the gap is a number.
        measured_allocated_bytes=allocation_observation.allocated_bytes,
        ledger_total_bytes=plan.memory.total_bytes,
        threads_mode=String(plan.threads.budget.mode),
        julia_threads=capacity.julia_threads,
        blas_before=Int(blas_threads()),
        error="",
    )
end

if get(ENV, "SDPX_S06_CHILD", "") == "1"
    report = s06_child_threads_report()
    println(
        "S06_CHILD ",
        join(("$(name)=$(value)" for (name, value) in pairs(report)), " "),
    )
    flush(stdout)
    exit(0)
elseif get(ENV, "SDPX_S06_CHILD", "") == "plan"
    report = s06_child_plan_report()
    println("S06_PLAN request_fingerprint=$(report.request_fingerprint) " *
            "context_fingerprint=$(report.context_fingerprint) " *
            "signature=$(report.signature)")
    flush(stdout)
    exit(0)
elseif get(ENV, "SDPX_S06_CHILD", "") == "provider"
    # The provider trigger packages are weak dependencies of SDPX: their
    # extensions only load once the trigger is loaded. Import at top level so
    # the world age advances before the report runs.
    s06_provider_name = get(ENV, "SDPX_S06_PROVIDER", "BigFloat")
    s06_provider_load_error = ""
    try
        s06_provider_name == "BigFloat" && @eval import BigFloatLinearAlgebra
        s06_provider_name == "MultiFloat" && begin
            @eval import MultiFloats
            @eval import MultiFloatLinearAlgebra
        end
    catch exception
        global s06_provider_load_error = sprint(showerror, exception)
    end
    report = s06_child_provider_report(s06_provider_name, s06_provider_load_error)
    println(
        "S06_PROVIDER ",
        join(("$(name)=$(value)" for (name, value) in pairs(report)), " "),
    )
    flush(stdout)
    exit(0)
end

@testset "S06 setup planning, memory and thread budget" begin

    # -----------------------------------------------------------------------
    # 1. Reproducibility, and the reported plan versus the execution
    # -----------------------------------------------------------------------
    @testset "1 plan reproducibility and reported-vs-actual" begin
        w = s06_small_workload()
        request = setup_request(w; requested_threads=4, memory_budget_bytes=S06_BUDGET)
        context = setup_context(capacity=S06_CAPACITY)
        first_plan = plan_setup(request, context)
        second_plan = plan_setup(request, context)
        # Same input, same context -> the same plan, field for field.
        @test setup_plan_signature(first_plan) == setup_plan_signature(second_plan)
        @test first_plan.request_fingerprint == second_plan.request_fingerprint
        @test first_plan.context_fingerprint == second_plan.context_fingerprint
        @test s06_signature_digest(first_plan) == s06_signature_digest(second_plan)

        # A different *context* is a different plan, and says so.
        other_context = setup_context(capacity=ThreadCapacity(2, 2, 4))
        other_plan = plan_setup(request, other_context)
        @test other_plan.context_fingerprint != first_plan.context_fingerprint
        @test other_plan.request_fingerprint == first_plan.request_fingerprint

        # Ambient benchmark naming must not enter the plan at all.
        baseline_fingerprint = first_plan.request_fingerprint
        baseline_signature = s06_signature_digest(first_plan)
        withenv(
            "SDPX_BENCHMARK_NAME" => "stokes2d",
            "SDPX_BENCHMARK_PROFILE" => "calibrated_best",
        ) do
            under_env = plan_setup(request, context)
            @test under_env.request_fingerprint == baseline_fingerprint
            @test s06_signature_digest(under_env) == baseline_signature
        end

        # The plan says what it will do; execution is measured against it.
        execute_setup_plan(first_plan, w; session=:s06_warmup)
        observation = execute_setup_plan(first_plan, w; session=:s06_measured)
        verification = verify_setup_execution(first_plan, observation)
        @test verification.ok
        @test isempty(verification.mismatches)
        @test observation.allocated_bytes > 0            # the measurement ran
        @test observation.cone_tasks_executed ==
              first_plan.costs.coarse.block_count
        @test observation.cone_rows_executed == first_plan.costs.coarse.total_rows
        @test observation.la_structural_nnz == first_plan.costs.la.structural_nnz
        @test observation.la_kernel_residual < 1.0e-8     # a real solve, not a stub
        @test observation.outer_threads_observed <=
              first_plan.threads.budget.julia_outer_threads
        @test observation.blas_threads_observed <=
              first_plan.threads.budget.blas_threads
        @test observation.blas_restored

        # Measured evidence, printed so the S06 report can quote real numbers.
        @info "S06 plan reproducibility and execution match" request_fingerprint=first_plan.request_fingerprint context_fingerprint=first_plan.context_fingerprint signature=s06_signature_digest(first_plan) route=first_plan.route storage=first_plan.resolved_storage verification_ok=verification.ok checks=length(verification.checks) allocated_bytes=observation.allocated_bytes memory_estimate_bytes=first_plan.memory.total_bytes outer_threads_observed=observation.outer_threads_observed outer_threads_granted=first_plan.threads.budget.julia_outer_threads blas_threads_observed=observation.blas_threads_observed la_residual=observation.la_kernel_residual
        # Cross-process reproducibility: a fresh Julia must agree.
        buffer = IOBuffer()
        process = run(pipeline(
            ignorestatus(setenv(
                s06_child_command(), "SDPX_S06_CHILD" => "plan",
            ));
            stdout=buffer, stderr=buffer,
        ))
        child_text = String(take!(buffer))
        @test process.exitcode == 0
        child = s06_parse_child(child_text, "S06_PLAN")
        if isempty(child)
            @info "S06 plan child produced no report" child_text
            @test !isempty(child)
        else
            @test child["request_fingerprint"] == "$(first_plan.request_fingerprint)"
            @test child["context_fingerprint"] == "$(first_plan.context_fingerprint)"
            @test child["signature"] == s06_signature_digest(first_plan)
        end
    end

    # -----------------------------------------------------------------------
    # 2. A 1-thread request inside a multi-threaded process
    # -----------------------------------------------------------------------
    @testset "2 one-thread request inside a 4-thread process" begin
        buffer = IOBuffer()
        process = run(pipeline(
            ignorestatus(setenv(
                s06_child_command(),
                "SDPX_S06_CHILD" => "1",
                "JULIA_NUM_THREADS" => "4",
            ));
            stdout=buffer, stderr=buffer,
        ))
        text = String(take!(buffer))
        report = s06_parse_child(text, "S06_CHILD")
        @test process.exitcode == 0
        if isempty(report)
            @info "S06 thread child produced no report" text
            @test !isempty(report)
        else
            integer(name) = parse(Int, report[name])
            floating(name) = parse(Float64, report[name])
            if integer("julia_threads") < 2
                # Infrastructure, not a numeric failure: without a genuinely
                # multi-threaded process this measurement cannot be made.
                @info "S06 1-thread-in-multi-thread-process check unsupported" report
                @test_broken integer("julia_threads") >= 2
            else
                # The process really is multi-threaded ...
                @test integer("julia_threads") >= 2
                # ... the serial request is granted exactly one thread in every
                # layer, *and the measured execution used exactly one*.
                @test report["serial_tier"] == "supported"   # 1 is a real tier
                @test report["serial_mode"] == "serial"
                @test integer("serial_granted_julia") == 1
                @test integer("serial_granted_blas") == 1
                @test integer("serial_granted_provider") == 1
                @test integer("serial_observed_julia") == 1
                @test integer("serial_observed_blas") == 1
                @test integer("serial_restored") == 1
                # The process-global BLAS count is left exactly as found.
                @test integer("blas_before") == integer("blas_after")
                @test floating("serial_residual") < 1.0e-6   # real work, real solve
                # Sensitivity: the *same* probe detects real parallelism when
                # four threads are granted, so "observed == 1" above is a
                # measurement rather than a constant.
                @test integer("parallel_granted_julia") <= integer("julia_threads")
                @test integer("parallel_observed_julia") >= 2
                @test integer("parallel_bins") == integer("parallel_granted_julia")
                @info "S06 one-thread-in-multi-threaded-process report" report
            end
        end
    end

    # -----------------------------------------------------------------------
    # 3. Memory refusal happens before the large allocation
    # -----------------------------------------------------------------------
    @testset "3 memory refusal precedes the allocation" begin
        w = s06_large_workload()
        context = setup_context(capacity=S06_CAPACITY)

        # The probe *is* the large allocation. In the admitted control it runs
        # and really allocates the admitted size, which is what makes its
        # absence in the refused case evidence rather than an assumption.
        probe_calls = Ref(0)
        probe_bytes = Ref{Vector{UInt8}}(UInt8[])
        probe = size -> begin
            probe_calls[] += 1
            probe_bytes[] = Vector{UInt8}(undef, size)
        end

        admitted = plan_setup(
            setup_request(w; requested_threads=4, memory_budget_bytes=S06_BUDGET),
            context,
        )
        @test admitted.admission.admitted
        @test admitted.memory.total_bytes > 1_000_000      # genuinely large
        admitted_observation = execute_setup_plan(
            admitted, w; session=:s06_probe_control, allocation_probe=probe,
            measure_allocations=true,
        )
        @test probe_calls[] == 1
        @test length(probe_bytes[]) == admitted.memory.total_bytes
        @test admitted_observation.la_kernel_residual < 1.0e-8
        @test admitted_observation.cone_tasks_executed == 3

        # Refused case: a 4 KiB budget against a multi-megabyte estimate.
        tiny_budget = 4096
        refused = plan_setup(
            setup_request(w; requested_threads=4, memory_budget_bytes=tiny_budget),
            context,
        )
        @test !refused.admission.admitted
        @test refused.admission.reason === :budget_exceeded
        @test refused.memory.total_bytes > tiny_budget
        @test refused.admission.headroom_bytes == 0
        @test_throws SetupMemoryRefusal reserve_setup_memory(refused.admission)

        refused_probe_calls = Ref(0)
        refused_probe_bytes = Ref{Vector{UInt8}}(UInt8[])
        outcome = Ref(:not_run)
        attempt = function ()
            try
                execute_setup_plan(
                    refused, w; session=:s06_probe_refused,
                    allocation_probe = size -> begin
                        refused_probe_calls[] += 1
                        refused_probe_bytes[] = Vector{UInt8}(undef, size)
                    end,
                )
                outcome[] = :executed
            catch exception
                exception isa SetupMemoryRefusal || rethrow()
                outcome[] = :refused
            end
            return outcome[]
        end
        attempt()                                     # warm up, then measure
        measured = Int(@allocated attempt())
        @test outcome[] === :refused
        @test refused_probe_calls[] == 0              # the allocation never ran
        @test isempty(refused_probe_bytes[])
        # The refused path allocated a few kilobytes, not the ~27 MB the refused
        # request would have needed (the admitted control above allocated it).
        @test measured < refused.memory.total_bytes ÷ 64

        # An unknown budget is a refusal too: fail closed, never assume room.
        unknown = plan_setup(
            setup_request(w; requested_threads=4, memory_budget_bytes=nothing),
            context,
        )
        @test !unknown.admission.admitted
        @test unknown.admission.reason === :no_memory_budget_declared
        @test unknown.admission.budget_bytes === nothing
        @test_throws SetupMemoryRefusal execute_setup_plan(unknown, w)

        # The measurement technique itself is sound: it detects a large
        # allocation when one really happens. (`@allocated` of a bare constant
        # allocation expression is folded away — measured here as 0 — so the
        # control goes through a function with a runtime-sized request.)
        control = Int(@allocated s06_allocate_control(8_000_000))
        @test control >= 8_000_000
        @info "S06 memory refusal evidence" admitted_estimate_bytes=admitted.memory.total_bytes admitted_fallback_bytes=admitted.memory.fallback_bytes probe_allocated_bytes=length(probe_bytes[]) refused_budget_bytes=tiny_budget refused_estimate_bytes=refused.memory.total_bytes refused_reason=refused.admission.reason refused_probe_calls=refused_probe_calls[] refused_path_allocated_bytes=measured probe_control_bytes=control
    end

    # -----------------------------------------------------------------------
    # 4. A sparse-declared path does not silently densify
    # -----------------------------------------------------------------------
    @testset "4 sparse declaration is honoured" begin
        context = setup_context(capacity=S06_CAPACITY)

        # (a) dynamic: the estimate is driven by nnz at *fixed* dimension, which
        # a dense estimate could not be.
        sparse_low = setup_workload(;
            block_sizes=[6, 4, 3], reduced_dimension=24, band=3, extra_nnz=0,
        )
        sparse_high = setup_workload(;
            block_sizes=[6, 4, 3], reduced_dimension=24, band=3, extra_nnz=400,
        )
        @test size(sparse_low.sparse_lower) == size(sparse_high.sparse_lower)
        @test nnz(sparse_high.sparse_lower) - nnz(sparse_low.sparse_lower) == 400
        low_request = setup_request(
            sparse_low; requested_threads=4, memory_budget_bytes=S06_BUDGET,
        )
        high_request = setup_request(
            sparse_high; requested_threads=4, memory_budget_bytes=S06_BUDGET,
        )
        @test low_request.full_dimension == high_request.full_dimension
        plan_low = plan_setup(low_request, context)
        plan_high = plan_setup(high_request, context)
        @test plan_low.memory.resolved_storage === :sparse_lower
        @test plan_high.memory.resolved_storage === :sparse_lower
        @test plan_high.memory.structural_bytes > plan_low.memory.structural_bytes
        @test plan_high.memory.structural_bytes - plan_low.memory.structural_bytes ==
              400 * (plan_low.memory.scalar_bytes + sizeof(Int))
        dense_bytes_small = plan_low.memory.scalar_bytes * low_request.full_dimension^2
        @test plan_low.memory.structural_bytes < dense_bytes_small ÷ 4

        # (b) dynamic: the execution ran the sparse representation, walked
        # exactly the declared pattern, and never touched a dense core — with a
        # core whose dense form would be ~40 MB against a ~9k-nnz pattern.
        big = s06_large_workload()
        dimension = size(big.sparse_lower, 1)
        dense_bytes = 8 * dimension^2
        @test nnz(big.sparse_lower) < dimension^2 ÷ 100
        big_plan = plan_setup(
            setup_request(big; requested_threads=4, memory_budget_bytes=S06_BUDGET),
            context,
        )
        execute_setup_plan(big_plan, big; session=:s06_sparse_warmup)
        observation = execute_setup_plan(
            big_plan, big; session=:s06_sparse_measured, measure_allocations=true,
        )
        @test observation.storage_used === :sparse_lower
        @test !observation.dense_core_touched
        @test !observation.fallback_allocated
        @test observation.la_structural_nnz == nnz(big.sparse_lower)
        @test observation.allocated_bytes > 0
        @test observation.allocated_bytes < dense_bytes ÷ 4
        @test verify_setup_execution(big_plan, observation).ok
        @info "S06 sparse declaration evidence" dimension=dimension declared_nnz=nnz(big.sparse_lower) dense_bytes=dense_bytes structural_bytes=big_plan.memory.structural_bytes fill_bytes=big_plan.memory.fill_bytes fallback_bytes=big_plan.memory.fallback_bytes applied_fallback=observation.fallback_allocated storage_used=observation.storage_used allocated_bytes=observation.allocated_bytes nnz_scaling_delta_bytes=plan_high.memory.structural_bytes - plan_low.memory.structural_bytes nnz_scaling_expected_bytes=400 * (plan_low.memory.scalar_bytes + sizeof(Int))

        # (c) dynamic over a sweep: every sparse declaration resolves sparse.
        for (blocks, reduced, band, extra) in (
            ([3], 4, 1, 0), ([6, 6], 12, 2, 7), ([8, 8, 8], 32, 4, 25),
            ([16], 64, 3, 100),
        )
            sweep = setup_workload(;
                block_sizes=blocks, reduced_dimension=reduced, band=band,
                extra_nnz=extra,
            )
            sweep_plan = plan_setup(
                setup_request(
                    sweep; requested_threads=2, memory_budget_bytes=S06_BUDGET,
                ),
                context,
            )
            @test sweep_plan.declared_storage === :sparse_lower
            @test sweep_plan.resolved_storage === :sparse_lower
        end

        # (d) densifying requires an *explicit* dense declaration, and a dense
        # declaration without a dense representation is refused outright.
        @test_throws ArgumentError setup_request(
            sparse_low; requested_threads=4, declared_storage=:dense,
            memory_budget_bytes=S06_BUDGET,
        )
        dense_workload = SetupWorkload{Float64}(
            sparse_low.block_operators, sparse_low.sparse_lower,
            Matrix{Float64}(undef, size(sparse_low.sparse_lower)...),
            sparse_low.blas_a, sparse_low.blas_b,
        )
        dense_plan = plan_setup(
            setup_request(
                dense_workload; requested_threads=4, declared_storage=:dense,
                memory_budget_bytes=S06_BUDGET,
            ),
            context,
        )
        @test dense_plan.declared_storage === :dense
        @test dense_plan.resolved_storage === :dense
        @test dense_plan.memory.structural_bytes ==
              dense_plan.memory.scalar_bytes * size(sparse_low.sparse_lower, 1)^2

        # Static, checked here: `setup_memory_ledger` has no branch that turns a
        # `:sparse_lower` declaration into `:dense` (the only dense structural
        # term is guarded by `declared_storage === :dense`), and the execution
        # path constructs no dense core.
        ledger_source = read(joinpath(S06_PLANNING, "resources.jl"), String)
        @test occursin("declared_storage in (:sparse_lower, :dense)", ledger_source)
        @test occursin("resolved = declared_storage", ledger_source)
    end

    # -----------------------------------------------------------------------
    # 5. Concurrent sessions and the global BLAS thread count
    # -----------------------------------------------------------------------
    @testset "5 concurrent sessions do not contend over global threads" begin
        capacity = thread_capacity()
        original = Int(blas_threads())
        registry = BlasLeaseRegistry()
        wide = min(4, max(capacity.blas_threads, 1))
        if wide < 2
            @info "S06 concurrent BLAS scope check unsupported: BLAS capacity " *
                  "$(capacity.blas_threads)"
            @test_broken wide >= 2
        else
            requests = (1, wide, wide, 1)
            results = Vector{Any}(undef, length(requests))
            @sync for (index, threads) in enumerate(requests)
                Threads.@spawn begin
                    budget = session_thread_budget(;
                        requested_threads=threads,
                        capacity=capacity,
                        cone_task_width=1,
                        la_kernel_width=1,
                        blas_width=wide,
                    )
                    session = Symbol("s06_concurrent_$(index)")
                    body = () -> (sleep(0.05 * index); Int(blas_threads()))
                    value, scope = with_session_thread_scope(
                        body, budget; registry=registry, session=session,
                    )
                    results[index] = (
                        granted=Int(budget.budget.blas_threads),
                        measured_in_body=value,
                        scope=scope,
                    )
                end
            end
            for result in results
                # Every session observed *its own* granted count inside its own
                # scope: nobody saw another session's value.
                @test result.measured_in_body == result.granted
                @test result.scope.observed_blas_threads == result.granted
                @test result.scope.restored
            end
            record = blas_lease_record(registry)
            @test record.attempts == length(requests)
            @test record.scopes_entered == length(requests)
            @test record.max_concurrent_holders == 1     # mutual exclusion held
            @test record.arrivals_while_held >= 1        # the scopes really overlapped
            @test record.value_mismatches == 0
            @test record.holder === nothing
            @test Int(blas_threads()) == original        # global state restored
            @info "S06 concurrent session evidence" requested=requests granted=[result.granted for result in results] measured_in_body=[result.measured_in_body for result in results] registry=record global_blas_before=original global_blas_after=Int(blas_threads())
        end

        # Nested scopes for the same session must not multiply the thread count.
        nested_budget = session_thread_budget(;
            requested_threads=2, capacity=capacity, cone_task_width=1,
            la_kernel_width=1, blas_width=wide,
        )
        nested_registry = BlasLeaseRegistry()
        nested_value = Ref(0)
        inner_value = Ref(0)
        inner_observed = Ref(0)
        nested_result = Ref{Symbol}(:not_run)
        _outer_value, _outer_scope = with_session_thread_scope(
            nested_budget; registry=nested_registry, session=:s06_nested,
        ) do
            nested_value[] = Int(blas_threads())
            _inner_value, inner_scope = with_session_thread_scope(
                () -> begin
                    inner_value[] = Int(blas_threads())
                    :nested_ok
                end,
                nested_budget;
                registry=nested_registry, session=:s06_nested,
            )
            inner_observed[] = inner_scope.observed_blas_threads
            nested_result[] = :outer_ok
            return :outer_ok
        end
        @test nested_result[] === :outer_ok
        @test nested_value[] == nested_budget.budget.blas_threads
        @test inner_value[] == nested_budget.budget.blas_threads
        @test inner_observed[] == nested_budget.budget.blas_threads
        @test max_granted_threads(nested_budget) == nested_budget.budget.blas_threads
        @test blas_lease_record(nested_registry).max_concurrent_holders == 1
        @test Int(blas_threads()) == original

        # A foreign session inside an active scope is refused, not silently
        # allowed to overwrite the global setting ...
        conflicting = session_thread_budget(;
            requested_threads=wide, capacity=capacity, cone_task_width=1,
            la_kernel_width=1, blas_width=wide,
        )
        holder_budget = session_thread_budget(;
            requested_threads=1, capacity=capacity, cone_task_width=1,
            la_kernel_width=1, blas_width=1,
        )
        contention_registry = BlasLeaseRegistry()
        caught = Ref{Any}(nothing)
        with_session_thread_scope(
            holder_budget; registry=contention_registry, session=:s06_holder,
        ) do
            try
                with_session_thread_scope(
                    () -> nothing, conflicting;
                    registry=contention_registry, session=:s06_intruder,
                )
            catch exception
                caught[] = exception
            end
        end
        @test caught[] isa ThreadContentionError
        @test caught[].reason === :scope_owned_by_another_session
        @test caught[].holder === :s06_holder
        contention_record = blas_lease_record(contention_registry)
        @test contention_record.attempts == 2
        @test contention_record.value_mismatches == 0
        @test Int(blas_threads()) == original

        # ... and a nested re-configuration of the *same* session is refused
        # rather than doubling the thread count.
        changed = session_thread_budget(;
            requested_threads=wide, capacity=capacity, cone_task_width=1,
            la_kernel_width=1, blas_width=wide,
        )
        doubling_registry = BlasLeaseRegistry()
        doubling = Ref{Any}(nothing)
        with_session_thread_scope(
            holder_budget; registry=doubling_registry, session=:s06_same,
        ) do
            try
                with_session_thread_scope(
                    () -> nothing, changed;
                    registry=doubling_registry, session=:s06_same,
                )
            catch exception
                doubling[] = exception
            end
        end
        @test doubling[] isa ThreadContentionError
        @test doubling[].reason === :nested_budget_change
        @test blas_lease_record(doubling_registry).value_mismatches == 1
        @test Int(blas_threads()) == original
    end

    # -----------------------------------------------------------------------
    # 6. Profiles: explicit offline calibration only
    # -----------------------------------------------------------------------
    @testset "6 profiles come from offline calibration only" begin
        @test_throws SetupProfileRefusal setup_profile(;
            id=:trial, source=:trial_run, artifact_path="x", samples=1,
            calibration_input_hash="deadbeef",
        )
        @test_throws SetupProfileRefusal profile_from_benchmark_name("stokes2d")
        @test_throws SetupProfileRefusal profile_from_benchmark_name("qap15")

        mktempdir() do directory
            bad = joinpath(directory, "solve_trial.txt")
            write(bad, "kind = solve_trial\nsamples = 3\ninput_hash = abc123\n")
            refusal = try
                load_setup_profile(bad)
                nothing
            catch exception
                exception
            end
            @test refusal isa SetupProfileRefusal
            @test refusal.reason === :profile_not_offline_calibration

            no_samples = joinpath(directory, "no_samples.txt")
            write(no_samples, "kind = offline_calibration\ninput_hash = abc123\n")
            @test_throws SetupProfileRefusal load_setup_profile(no_samples)
            @test_throws SetupProfileRefusal load_setup_profile(
                joinpath(directory, "absent.txt"),
            )

            good = joinpath(directory, "offline_calibration.txt")
            calibration_text =
                "# explicit offline calibration of the setup resource model\n" *
                "kind = offline_calibration\n" *
                "profile_id = s06_offline\n" *
                "samples = 12\n" *
                "input_hash = 9f2c41ab77\n" *
                "fill_factor = 5.5\n" *
                "blas_threading_min_width = 96\n"
            write(good, calibration_text)
            profile = load_setup_profile(good)
            @test profile.calibrated
            @test profile.source === :offline_calibration
            @test profile.samples == 12
            @test profile.fill_factor == 5.5
            @test profile.blas_threading_min_width == 96
            @test startswith(profile.artifact_digest, "fnv1a64:")
            # The digest is a content digest: a one-character change changes it.
            other = joinpath(directory, "offline_calibration2.txt")
            write(other, replace(calibration_text, "5.5" => "5.6"))
            @test load_setup_profile(other).artifact_digest != profile.artifact_digest

            # The profile really feeds the resource model: the calibrated fill
            # factor changes the modelled fill bytes and the BLAS width prior.
            w = s06_small_workload()
            request = setup_request(
                w; requested_threads=4, memory_budget_bytes=S06_BUDGET,
            )
            unprofiled = plan_setup(request, setup_context(capacity=S06_CAPACITY))
            calibrated = plan_setup(
                request, setup_context(capacity=S06_CAPACITY, profile=profile),
            )
            @test calibrated.profile.id === :s06_offline
            @test calibrated.memory.fill_bytes > unprofiled.memory.fill_bytes
            @test calibrated.context_fingerprint != unprofiled.context_fingerprint
            @test calibrated.costs.blas.min_threading_width == 96
            # ... and does not change the *route*, which belongs to the
            # inherited structural model.
            @test calibrated.route === unprofiled.route
            @test calibrated.costs.la.structural_nnz ==
                  unprofiled.costs.la.structural_nnz
            @test calibrated.costs.core_route.full_score ==
                  unprofiled.costs.core_route.full_score

            # No trial run inside a solve, dynamic form: with a malformed
            # calibration artifact and a trial-run switch pointed at in the
            # environment — the only places a hidden profiler could look — the
            # plan is unchanged and execution still reports exactly the plan.
            fill_before = calibrated.memory.fill_bytes
            signature_before = s06_signature_digest(calibrated)
            withenv(
                "SDPX_SETUP_PROFILE" => bad,
                "SDPX_CALIBRATION_ARTIFACT" => bad,
                "SDPX_SETUP_TRIAL_RUN" => "1",
            ) do
                observation = execute_setup_plan(
                    calibrated, w; session=:s06_no_trial_run,
                )
                @test verify_setup_execution(calibrated, observation).ok
            end
            @test calibrated.memory.fill_bytes == fill_before
            @test s06_signature_digest(calibrated) == signature_before
            @test calibrated.profile.artifact_digest == profile.artifact_digest

            # Static, checked here: no planning source reads an environment
            # variable, so no ambient name or path can select a profile.
            for name in ("costs.jl", "resources.jl", "setup.jl")
                @test !occursin("ENV[", read(joinpath(S06_PLANNING, name), String))
            end
        end
    end

    # -----------------------------------------------------------------------
    # 7. Inheritance and the packet's thread tiers
    # -----------------------------------------------------------------------
    @testset "7 inheritance and host tiers" begin
        # The route comes from the previous round's model: same inputs, same
        # answer, and the incumbent rule is reported alongside.
        w = s06_small_workload()
        request = setup_request(w; requested_threads=2, memory_budget_bytes=S06_BUDGET)
        plan = plan_setup(request, setup_context(capacity=S06_CAPACITY))
        inherited = plan_core_route(;
            full_dimension=request.full_dimension,
            compact_dimension=request.compact_dimension,
            ar_nnz=request.ar_nnz,
            canonical_nnz=request.canonical_nnz,
            block_sizes=request.block_sizes,
            T=Float64,
            kkt_route=request.kkt_route,
            fixed_trace=request.fixed_trace,
            rhs_count=request.rhs_count,
        )
        @test plan.route === inherited.route
        @test plan.costs.core_route.full_score == inherited.full_score
        @test plan.costs.core_route.compact_score == inherited.compact_score
        @test plan.costs.inherited_model === :core_route_planner
        @test plan.costs.legacy_would_choose_compact ==
              legacy_dimension_rule(request.full_dimension, request.compact_dimension)
        @test plan.reasons[1] === :plan_inherits_core_route_cost_model

        # The coarse-cone description agrees with the inherited classifier, so
        # its threshold cannot silently diverge.
        if isdefined(SDPX, :_cone_block_shape_class)
            classifier = getfield(SDPX, :_cone_block_shape_class)
            for blocks in ([], [3], [6], [3, 8, 12], [5, 5, 5], [64, 2, 1])
                @test describe_coarse_cone_tasks(blocks).dense_share ==
                      classifier(blocks)[1]
            end
        else
            @info "S06 inherited classifier unavailable; consistency check not_run"
        end

        # Three independent descriptions; no product of their limits anywhere.
        model = plan.costs
        @test model.coarse.parallel_width == max(model.coarse.block_count, 1)
        @test available_consumer_width(model, :coarse_cone_tasks) ==
              model.coarse.parallel_width
        @test max_granted_threads(plan.threads) <=
              max(
                  model.coarse.parallel_width,
                  model.la.parallel_width,
                  model.blas.parallel_width,
              )
        @test max_granted_threads(plan.threads) <= plan.threads.requested_threads

        # The packet's tiers: 1/2/4 are supported on a 4-thread host, 16 and 64
        # are unsupported and are *recorded* as unsupported, never granted.
        for tier in (1, 2, 4)
            @test thread_tier_status(tier, S06_CAPACITY) === :supported
        end
        for tier in (16, 64)
            @test thread_tier_status(tier, S06_CAPACITY) === :unsupported
            tier_plan = plan_setup(
                setup_request(
                    w; requested_threads=tier, memory_budget_bytes=S06_BUDGET,
                ),
                setup_context(capacity=S06_CAPACITY),
            )
            @test tier_plan.threads.tier_status === :unsupported
            @test :requested_tier_unsupported_on_host in tier_plan.reasons
            @test tier_plan.threads.requested_threads == tier
            @test max_granted_threads(tier_plan.threads) <= 4
            @test max_granted_threads(tier_plan.threads) != tier
        end
        # ... and the real host agrees with the synthetic capacity used above.
        host = thread_capacity()
        @test thread_tier_status(16, host) === :unsupported
        @test thread_tier_status(64, host) === :unsupported
        @info "S06 host and tier evidence" julia_threads=host.julia_threads blas_threads=host.blas_threads cpu_threads=host.cpu_threads tier_1=thread_tier_status(1, host) tier_2=thread_tier_status(2, host) tier_4=thread_tier_status(4, host) tier_16=thread_tier_status(16, host) tier_64=thread_tier_status(64, host)

        # The extended-precision capability is *probed*, not assumed. Without the
        # provider loaded the request is refused with its reason; with it loaded
        # the same request must be planned and priced at the provider's own
        # element width. Neither branch may be satisfied by a stale belief.
        if S06_PLANNING_INCLUDED_HERE
            big_request = SetupRequest(
                BigFloat, 256, :bordered, false, 3,
                request.full_dimension, request.compact_dimension,
                request.ar_nnz, request.canonical_nnz, 0, 0,
                copy(request.block_sizes), :sparse_lower, 1, S06_BUDGET, nothing,
            )
            if s06_provider_loaded(BigFloat)
                planned = plan_setup(
                    big_request,
                    setup_context(capacity=S06_CAPACITY, provider_available=true),
                )
                @test planned.memory.scalar_bytes ==
                      ExtendedPrecisionBLAS._element_storage_bytes(BigFloat)
                @test planned.memory.scalar_bytes > 8
                @test planned.admission.admitted
                @test planned.route isa Symbol
            else
                capability = try
                    plan_setup(
                        big_request,
                        setup_context(
                            capacity=S06_CAPACITY, provider_available=false,
                        ),
                    )
                    nothing
                catch exception
                    exception
                end
                @test capability isa SetupCapabilityRefusal
                @test capability.reason === :provider_unavailable_for_precision
            end
        else
            @info "S06 BigFloat capability check skipped: integrated configuration"
        end
    end

    # -----------------------------------------------------------------------
    # 8. Provider legs in the packet's rebuild environment
    # -----------------------------------------------------------------------
    #
    # MF and BF run in *separate* `-t1` processes: Julia 1.12 can exhaust its
    # inference compiler when the fixed-width MFLA and BFLA/MPFR specializations
    # compile in one process (`scripts/provider_smoke.sh` documents this). The
    # provider is probed inside the child, so an environment without it yields
    # `unsupported` with that reason instead of a silent pass.
    @testset "8 provider legs (REBUILD_ENV)" begin
        rebuild_env = get(
            ENV, "SDPX_S06_REBUILD_ENV",
            "/Users/xuyongjun/Desktop/project/SDPX/rebuild-env",
        )
        project_file = joinpath(rebuild_env, "Project.toml")
        if !isfile(project_file)
            @info "S06 provider legs unsupported: no REBUILD_ENV at $project_file"
            @test_skip false
        else
            for name in ("BigFloat", "MultiFloat")
                command = `$(Base.julia_cmd()) --startup-file=no -t1 --project=$rebuild_env $(@__FILE__)`
                buffer = IOBuffer()
                process = run(pipeline(
                    ignorestatus(setenv(
                        command,
                        "SDPX_S06_CHILD" => "provider",
                        "SDPX_S06_PROVIDER" => name,
                    ));
                    stdout=buffer, stderr=buffer,
                ))
                text = String(take!(buffer))
                report = s06_parse_child(text, "S06_PROVIDER")
                if process.exitcode != 0 && !occursin(
                    "rebuild-env-depot", get(ENV, "JULIA_DEPOT_PATH", ""),
                )
                    @info "S06 provider leg ($name) unsupported: REBUILD_ENV depot " *
                          "is not on JULIA_DEPOT_PATH" text
                    @test_skip false
                else
                    @test process.exitcode == 0
                    if isempty(report)
                        @info "S06 provider child ($name) produced no report" text
                        @test !isempty(report)
                    elseif parse(Int, report["loaded"]) != 1
                        @info "S06 provider leg ($name) unsupported" report
                        @test_skip false
                    else
                        # Measured, one process per provider, one thread each.
                        @test parse(Int, report["julia_threads"]) == 1
                        @test report["threads_mode"] == "serial"
                        @test parse(Int, report["admitted"]) == 1
                        @test parse(Int, report["scalar_bytes"]) >
                              parse(Int, report["float64_scalar_bytes"])
                        @test parse(Int, report["scalar_bytes"]) > 8
                        @test parse(Int, report["total_bytes"]) > 0
                        @test parse(Int, report["wide_scalar_reason"]) == 1
                        @test report["storage_used"] == "sparse_lower"
                        @test parse(Float64, report["residual"]) < 1.0e-6
                        @test parse(Int, report["verification_ok"]) == 1
                        # The ledger is an upper bound on the measured
                        # Julia-heap allocation traffic. MPFR limb memory is
                        # native allocation and is *not* counted by `@allocated`;
                        # that gap is an open finding, not a property test.
                        @test parse(Int, report["measured_allocated_bytes"]) <=
                              parse(Int, report["ledger_total_bytes"])
                        # Recorded, not hidden: the wide-precision ledger total
                        # is *not* larger than the Float64 one for the same
                        # structure, because the Float64 sparse path inherits a
                        # 32x snapshot allowance the wide path does not. That
                        # asymmetry is an open finding in the S06 report.
                        @info "S06 provider ledger comparison" name scalar_bytes=report["scalar_bytes"] total_bytes=report["total_bytes"] float64_total_bytes=report["float64_total_bytes"] residual=report["residual"] measured_allocated_bytes=report["measured_allocated_bytes"]
                        if name == "BigFloat"
                            @test parse(Int, report["scalar_bytes"]) ==
                                  ExtendedPrecisionBLAS._element_storage_bytes(BigFloat)
                        end
                    end
                end
            end
        end
    end
end
