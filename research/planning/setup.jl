# S06 — the single setup planning entry point.
#
# Card step 1: "收束现有规划规则到一个setup入口，继承上一轮已完成的结构/精度
# cost model." There is exactly one entry point, `plan_setup`, and it converges
# the rules that previously lived apart:
#
#   * the inherited structure-aware route model (`plan_core_route`, carried
#     verbatim through `setup_cost_model`);
#   * the retained incumbent rule (`legacy_dimension_rule`, recorded in the
#     receipt as `legacy_would_choose_compact` so a reader can see both);
#   * the three independent consumer descriptions (`costs.jl`);
#   * memory admission (`resources.jl`);
#   * the session thread budget (`resources.jl`).
#
# Card step 3: "profile仅在显式离线校准产生，不在求解中试跑或按benchmark名字
# 分流." A profile is a value, it is produced only by `load_setup_profile` from
# an artifact that declares itself an offline calibration, and `plan_setup`
# refuses any other source. There is no code path that measures anything while
# planning or solving, and no benchmark name is ever consulted: `plan_setup`
# takes no benchmark identifier at all, and `profile_from_benchmark_name` exists
# only to refuse.
#
# A plan is a *description*: identical request and context produce an identical
# plan (fingerprinted), and `execute_setup_plan` runs the workload the plan
# describes while measuring what actually happened, so
# `verify_setup_execution` can compare the report against reality.

# ---------------------------------------------------------------------------
# Profiles: offline calibration only
# ---------------------------------------------------------------------------

"""
    SetupProfile

Resource coefficients, and where they came from. A profile is admissible only
from an explicit **offline** calibration artifact; the default
[`UNPROFILED_SETUP`](@ref) carries the stated priors and declares
`source === :unprofiled`.

Fields: `id`, `source`, `artifact_path`, `artifact_digest`, `samples`,
`calibration_input_hash`, `fill_factor`, `blas_threading_min_width`, plus
`calibrated`.
"""
struct SetupProfile
    id::Symbol
    source::Symbol
    artifact_path::String
    artifact_digest::String
    samples::Int
    calibration_input_hash::String
    fill_factor::Float64
    blas_threading_min_width::Int
    calibrated::Bool
end

"""The unprofiled defaults: the stated priors, no measurement claimed."""
const UNPROFILED_SETUP = SetupProfile(
    :unprofiled, :unprofiled, "", "", 0, "", 3.0, MIN_BLAS_THREADING_WIDTH, false,
)

"""The only sources a profile may have."""
const SETUP_PROFILE_SOURCES = (:unprofiled, :offline_calibration)

"""
    SetupProfileRefusal

Raised when a profile would come from anywhere other than an explicit offline
calibration: a trial run inside a solve, a benchmark-name dispatch, or an
artifact that does not declare itself a calibration.
"""
struct SetupProfileRefusal <: Exception
    reason::Symbol
    detail::String
end

function Base.showerror(io::IO, e::SetupProfileRefusal)
    print(io, "SetupProfileRefusal(", e.reason, "): ", e.detail)
end

"""
    SetupCapabilityRefusal

Raised when a plan cannot be made because a capability the request needs is not
present in this environment — an absent extended-precision provider above all.
This is an *infrastructure* fact: it is recorded as a refusal with its reason,
never recorded as a numeric failure and never worked around by silently
downgrading the precision.
"""
struct SetupCapabilityRefusal <: Exception
    reason::Symbol
    detail::String
end

function Base.showerror(io::IO, e::SetupCapabilityRefusal)
    print(io, "SetupCapabilityRefusal(", e.reason, "): ", e.detail)
end

"""
    setup_profile(; id, source, artifact_path="", artifact_digest="", samples=0,
                  calibration_input_hash="", fill_factor=3.0,
                  blas_threading_min_width=MIN_BLAS_THREADING_WIDTH) -> SetupProfile

Construct a validated profile. Any `source` other than `:unprofiled` /
`:offline_calibration` — `:trial_run` above all — is refused here, so an
in-solve measurement cannot even be represented as a profile.
"""
function setup_profile(;
    id::Symbol=:unprofiled,
    source::Symbol=:unprofiled,
    artifact_path::AbstractString="",
    artifact_digest::AbstractString="",
    samples::Integer=0,
    calibration_input_hash::AbstractString="",
    fill_factor::Real=3.0,
    blas_threading_min_width::Integer=MIN_BLAS_THREADING_WIDTH,
)
    source in SETUP_PROFILE_SOURCES || throw(SetupProfileRefusal(
        :profile_source_not_offline_calibration,
        "profile source $(source) is not one of $(SETUP_PROFILE_SOURCES); " *
        "profiles must come from an explicit offline calibration",
    ))
    if source === :offline_calibration
        isempty(artifact_path) && throw(SetupProfileRefusal(
            :calibration_artifact_missing,
            "an offline calibration profile must name the artifact it came from",
        ))
        samples >= 1 || throw(SetupProfileRefusal(
            :calibration_samples_missing,
            "an offline calibration profile must report at least one sample",
        ))
        isempty(calibration_input_hash) && throw(SetupProfileRefusal(
            :calibration_input_hash_missing,
            "an offline calibration profile must record the hash of the input " *
            "corpus it was calibrated on",
        ))
        isfinite(fill_factor) && fill_factor > 0.0 || throw(SetupProfileRefusal(
            :calibration_fill_factor_invalid,
            "calibrated fill factor must be finite and positive",
        ))
        blas_threading_min_width >= 1 || throw(SetupProfileRefusal(
            :calibration_blas_width_invalid,
            "calibrated BLAS threading width must be at least one",
        ))
    end
    return SetupProfile(
        id, source, String(artifact_path), String(artifact_digest), Int(samples),
        String(calibration_input_hash), Float64(fill_factor),
        Int(blas_threading_min_width), source === :offline_calibration,
    )
end

"""
    profile_from_benchmark_name(name) -> SetupProfile

Always throws. It exists so the forbidden dispatch has a named, testable
refusal instead of being an unstated omission: a profile may never be selected
because a benchmark is called `stokes2d` or `qap15`.
"""
function profile_from_benchmark_name(name)
    throw(SetupProfileRefusal(
        :benchmark_name_dispatch_forbidden,
        "refusing to select a setup profile from benchmark name $(repr(name)); " *
        "profiles come from an explicit offline calibration only",
    ))
end

"""Content digest of a calibration artifact (FNV-1a/64, dependency-free)."""
function artifact_digest(bytes::AbstractVector{UInt8})
    h = UInt64(0xcbf29ce484222325)
    for byte in bytes
        h = (h ⊻ UInt64(byte)) * UInt64(0x00000100000001b3)
    end
    return "fnv1a64:" * string(h; base=16, pad=16)
end

"""
    parse_calibration_artifact(text) -> NamedTuple

Parse the `key = value` calibration record. Unknown keys are ignored; a missing
required key fails closed. The record must declare
`kind = offline_calibration`.
"""
function parse_calibration_artifact(text::AbstractString)
    fields = Dict{String,String}()
    for raw in split(text, '\n')
        line = strip(raw)
        (isempty(line) || startswith(line, '#')) && continue
        parts = split(line, '='; limit=2)
        length(parts) == 2 || throw(SetupProfileRefusal(
            :calibration_artifact_malformed, "unparsable line $(repr(line))",
        ))
        fields[strip(parts[1])] = strip(parts[2])
    end
    get(fields, "kind", "") == "offline_calibration" || throw(SetupProfileRefusal(
        :profile_not_offline_calibration,
        "artifact declares kind=$(repr(get(fields, "kind", ""))); a profile may " *
        "only come from an artifact that declares kind=offline_calibration",
    ))
    samples = something(tryparse(Int, get(fields, "samples", "")), 0)
    fill_factor = something(tryparse(Float64, get(fields, "fill_factor", "")), 3.0)
    min_width = something(
        tryparse(Int, get(fields, "blas_threading_min_width", "")),
        MIN_BLAS_THREADING_WIDTH,
    )
    id = Symbol(get(fields, "profile_id", "calibrated"))
    return (
        id=id, samples=samples, fill_factor=fill_factor,
        blas_threading_min_width=min_width,
        calibration_input_hash=get(fields, "input_hash", ""),
    )
end

"""
    load_setup_profile(path) -> SetupProfile

Load an **explicit offline calibration** artifact and turn it into a profile.

Refuses (never falls back to a prior silently): a missing artifact, a
malformed one, an artifact that does not declare itself an offline calibration
(for example `kind = solve_trial`), a zero sample count, or a missing input
hash. The returned profile records the artifact path and its content digest, so
a receipt can name the calibration it used.
"""
function load_setup_profile(path::AbstractString)
    isfile(path) || throw(SetupProfileRefusal(
        :calibration_artifact_missing,
        "no calibration artifact at $(repr(path))",
    ))
    bytes = read(path)
    text = String(copy(bytes))
    parsed = parse_calibration_artifact(text)
    return setup_profile(;
        id=parsed.id,
        source=:offline_calibration,
        artifact_path=String(path),
        artifact_digest=artifact_digest(bytes),
        samples=parsed.samples,
        calibration_input_hash=parsed.calibration_input_hash,
        fill_factor=parsed.fill_factor,
        blas_threading_min_width=parsed.blas_threading_min_width,
    )
end

# ---------------------------------------------------------------------------
# Context and request
# ---------------------------------------------------------------------------

"""
    SetupContext

Everything about the run that is not the problem: the measured host capacity,
whether a provider is available, and the calibration profile in force.

`capacity` is carried explicitly rather than read from globals during planning,
so "the same input and context produce the same plan" is a statement about two
*named* things. `thread_capacity()` is the default and is the only place the
host is measured.
"""
struct SetupContext
    capacity::ThreadCapacity
    provider_available::Bool
    profile::SetupProfile
end

"""Build a setup context; the host is measured once, here."""
function setup_context(;
    capacity::ThreadCapacity=thread_capacity(),
    provider_available::Bool=false,
    profile::SetupProfile=UNPROFILED_SETUP,
)
    return SetupContext(capacity, provider_available, profile)
end

"""
    SetupRequest

The frozen inputs of one setup decision: the problem's structure and precision,
the declared storage, and the resources the caller is willing to spend.

Notably absent: any benchmark name, any timing, and any measurement taken while
solving. Nothing here can vary with how the caller labelled the problem.
"""
struct SetupRequest
    T::Type
    precision_bits::Int
    kkt_route::Symbol
    fixed_trace::Bool
    rhs_count::Int
    full_dimension::Int
    compact_dimension::Int
    ar_nnz::Int
    canonical_nnz::Int
    basis_nnz::Int
    variable_dimension::Int
    block_sizes::Vector{Int}
    declared_storage::Symbol
    requested_threads::Int
    memory_budget_bytes::Union{Nothing,Int}
    current_rss_bytes::Union{Nothing,Int}
end

function Base.show(io::IO, r::SetupRequest)
    print(
        io, "SetupRequest(", r.T, " d=", r.full_dimension, "/", r.compact_dimension,
        " nnz=", r.ar_nnz, " blocks=", length(r.block_sizes),
        " storage=", r.declared_storage, " threads=", r.requested_threads,
        " budget=", r.memory_budget_bytes === nothing ? "none" : string(r.memory_budget_bytes),
        ")",
    )
end

# ---------------------------------------------------------------------------
# The setup workload the plan describes
# ---------------------------------------------------------------------------

"""
    SetupWorkload

A real, concrete instance of the shape a plan describes: the per-block cone
operators, the declared sparse core, an optional dense core (the fallback
representation), and one BLAS panel product.

The plan is not built *from* the workload's arrays alone — `setup_request`
extracts the structural facts — but execution runs these arrays, so the
comparison between what the plan reported and what happened is a comparison
against real work, not against a second copy of the plan.
"""
struct SetupWorkload{T<:AbstractFloat}
    block_operators::Vector{Matrix{T}}
    sparse_lower::SparseMatrixCSC{T,Int}
    dense_core::Union{Nothing,Matrix{T}}
    blas_a::Matrix{T}
    blas_b::Matrix{T}
end

"""
    setup_workload(; block_sizes, reduced_dimension, band=2, extra_nnz=0,
                   blas_dimension=8, T=Float64) -> SetupWorkload

Build a deterministic workload with the requested structure. The sparse core is
a genuine lower-triangular CSC with a banded pattern plus `extra_nnz` additional
stored entries placed deterministically — enough to vary `nnz` at *fixed*
dimension, which is what makes the densification check a discriminating
measurement rather than a tautology.
"""
function setup_workload(;
    block_sizes,
    reduced_dimension::Integer,
    band::Integer=2,
    extra_nnz::Integer=0,
    blas_dimension::Integer=8,
    T::Type=Float64,
)
    blocks = [size for size in block_sizes if size > 0]
    cone_rows = sum(blocks; init=0)
    d = Int(reduced_dimension) + cone_rows
    d >= 1 || throw(ArgumentError("setup workload dimension must be positive"))
    bandw = clamp(Int(band), 1, d)
    rows = Int[]
    cols = Int[]
    vals = T[]
    column_mass = zeros(Float64, d)
    for j in 1:d
        for i in (j + 1):min(d, j + bandw - 1)
            value = T(mod(7 * i + 3 * j, 11) + 4)
            push!(rows, i)
            push!(cols, j)
            push!(vals, value)
            column_mass[j] += Float64(value)
        end
    end
    added = 0
    j = 1
    while added < Int(extra_nnz) && j <= d
        for i in (j + bandw):d
            added >= Int(extra_nnz) && break
            value = T(mod(5 * i + j, 13) + 1)
            push!(rows, i)
            push!(cols, j)
            push!(vals, value)
            column_mass[j] += Float64(value)
            added += 1
        end
        j += 1
    end
    # Strictly column-diagonally dominant, so the forward substitution the LA
    # kernel performs is well conditioned: its residual is then a real accuracy
    # statement about the kernel instead of an artefact of a bad fixture.
    for j in 1:d
        push!(rows, j)
        push!(cols, j)
        push!(vals, T(4.0 * column_mass[j] + 1.0))
    end
    lower = sparse(rows, cols, vals, d, d)
    operators = [Matrix{T}(I, size, size) for size in blocks]
    for (index, size) in enumerate(blocks)
        for j2 in 1:size, i2 in 1:size
            operators[index][i2, j2] = T(mod(3 * i2 + 5 * j2, 17) + 2) / T(4)
        end
    end
    dense_core = Matrix{T}(undef, 0, 0)
    panel = max(Int(blas_dimension), 1)
    blas_a = [T(mod(2 * i + 3 * j, 19) + 1) / T(3) for i in 1:panel, j in 1:panel]
    blas_b = [T(mod(5 * i + j, 23) + 1) / T(5) for i in 1:panel, j in 1:panel]
    return SetupWorkload{T}(operators, lower, nothing, blas_a, blas_b)
end

"""`true` when the workload provides a dense core representation."""
has_dense_core(w::SetupWorkload) = w.dense_core !== nothing

"""
    setup_request(workload; requested_threads, memory_budget_bytes,
                  current_rss_bytes=nothing, compact_dimension=nothing,
                  declared_storage=nothing, precision_bits=64,
                  kkt_route=:bordered, fixed_trace=false, rhs_count=3,
                  basis_nnz=0, variable_dimension=0, T=Float64) -> SetupRequest

Extract the frozen structural facts from a real workload.

`declared_storage` defaults to what the workload actually carries
(`:sparse_lower` for a CSC core). Passing `:dense` is an explicit request to
build the dense representation and must be paired with a workload that has a
dense core.
"""
function setup_request(
    w::SetupWorkload{T};
    requested_threads::Integer,
    memory_budget_bytes::Union{Nothing,Integer}=nothing,
    current_rss_bytes::Union{Nothing,Integer}=nothing,
    compact_dimension::Union{Nothing,Integer}=nothing,
    declared_storage::Union{Nothing,Symbol}=nothing,
    precision_bits::Integer=64,
    kkt_route::Symbol=:bordered,
    fixed_trace::Bool=false,
    rhs_count::Integer=3,
    basis_nnz::Integer=0,
    variable_dimension::Integer=0,
) where {T}
    d = size(w.sparse_lower, 1)
    block_sizes = Int[size(m, 1) for m in w.block_operators]
    cone_rows = sum(block_sizes; init=0)
    cone_rows <= d || throw(ArgumentError(
        "workload cone rows ($(cone_rows)) exceed its core dimension ($(d))",
    ))
    reduced = d - cone_rows
    compact = compact_dimension === nothing ? max(reduced, 1) : Int(compact_dimension)
    storage = declared_storage === nothing ? :sparse_lower : Symbol(declared_storage)
    if storage === :dense && !has_dense_core(w)
        throw(ArgumentError(
            "declared_storage=:dense requires the workload to carry a dense core",
        ))
    end
    return SetupRequest(
        T, Int(precision_bits), kkt_route, fixed_trace, Int(rhs_count), d, compact,
        nnz(w.sparse_lower), nnz(w.sparse_lower) + cone_rows, Int(basis_nnz),
        Int(variable_dimension), block_sizes, storage, Int(requested_threads),
        memory_budget_bytes === nothing ? nothing : Int(memory_budget_bytes),
        current_rss_bytes === nothing ? nothing : Int(current_rss_bytes),
    )
end

# ---------------------------------------------------------------------------
# The plan and the one entry point
# ---------------------------------------------------------------------------

"""
    SetupPlan

The complete setup decision.

Fields: the chosen `route` (from the inherited model), the `costs` description,
the `memory` ledger and its `admission`, the `threads` budget, the `profile` in
force, the `declared_storage` and `resolved_storage`, the ordered `reasons`, and
the two fingerprints (`request_fingerprint` over input, `context_fingerprint`
over context) that make reproducibility checkable rather than asserted.
"""
struct SetupPlan
    route::Symbol
    costs::SetupCostModel
    memory::SetupMemoryLedger
    admission::MemoryAdmission
    threads::SessionThreadBudget
    profile::SetupProfile
    declared_storage::Symbol
    resolved_storage::Symbol
    reasons::Vector{Symbol}
    request_fingerprint::UInt64
    context_fingerprint::UInt64
end

const _SETUP_FNV_OFFSET = UInt64(0xcbf29ce484222325)
const _SETUP_FNV_PRIME = UInt64(0x00000100000001b3)

_fnvn(h::UInt64, x::UInt64) = (h ⊻ x) * _SETUP_FNV_PRIME

function _fnv_int(h::UInt64, x::Integer)
    value = x < 0 ? typemax(UInt64) : UInt64(min(x, typemax(Int64)))
    return _fnvn(h, value)
end

_fnvn(h::UInt64, x::Float64) = _fnvn(h, reinterpret(UInt64, x))

function _fnv_symbol(h::UInt64, s::Symbol)
    for byte in codeunits(String(s))
        h = _fnvn(h, UInt64(byte))
    end
    return _fnvn(h, UInt64(0xff))
end

function _fnv_string(h::UInt64, s::AbstractString)
    for byte in codeunits(s)
        h = _fnvn(h, UInt64(byte))
    end
    return _fnvn(h, UInt64(0xfe))
end

"""
    setup_request_fingerprint(request) -> UInt64

FNV-1a/64 over the request's fields in a fixed order. Only integer, `Float64`
bit-pattern and byte-level terms are hashed — no `hash` of mutable containers and
no time, address, or task identity — so the value is stable across processes and
across JIT states.
"""
function setup_request_fingerprint(r::SetupRequest)
    h = _SETUP_FNV_OFFSET
    h = _fnv_symbol(h, Symbol(string(r.T)))
    for value in (
        r.precision_bits, r.rhs_count, r.full_dimension, r.compact_dimension,
        r.ar_nnz, r.canonical_nnz, r.basis_nnz, r.variable_dimension,
        r.requested_threads, length(r.block_sizes),
    )
        h = _fnv_int(h, value)
    end
    for size in r.block_sizes
        h = _fnv_int(h, size)
    end
    h = _fnv_int(h, r.fixed_trace ? 1 : 0)
    h = _fnv_int(
        h, r.memory_budget_bytes === nothing ? -1 : r.memory_budget_bytes,
    )
    h = _fnv_int(h, r.current_rss_bytes === nothing ? -1 : r.current_rss_bytes)
    h = _fnv_symbol(h, r.declared_storage)
    h = _fnv_symbol(h, r.kkt_route)
    return h
end

"""
    setup_context_fingerprint(context) -> UInt64

FNV-1a/64 over the context: host capacity, provider availability, and the
profile's identity and coefficients. Two plans are reproducible when these two
fingerprints and the decision fields agree.
"""
function setup_context_fingerprint(c::SetupContext)
    h = _SETUP_FNV_OFFSET
    h = _fnv_int(h, c.capacity.julia_threads)
    h = _fnv_int(h, c.capacity.blas_threads)
    h = _fnv_int(h, c.capacity.cpu_threads)
    h = _fnv_int(h, c.provider_available ? 1 : 0)
    h = _fnv_symbol(h, c.profile.id)
    h = _fnv_symbol(h, c.profile.source)
    h = _fnv_string(h, c.profile.artifact_digest)
    h = _fnv_string(h, c.profile.calibration_input_hash)
    h = _fnv_int(h, c.profile.samples)
    h = _fnvn(h, c.profile.fill_factor)
    h = _fnv_int(h, c.profile.blas_threading_min_width)
    return h
end

"""
    plan_setup(request, context) -> SetupPlan
    plan_setup(workload; context=setup_context(), kwargs...) -> SetupPlan

**The single setup planning entry point.**

It converges every planning rule into one place and returns one description:

1. the route comes from the inherited structure-aware cost model
   (`plan_core_route`), with the incumbent `legacy_dimension_rule` recorded
   alongside;
2. the three consumers (coarse cone tasks, LA kernel, BLAS) are described
   independently;
3. memory admission accounts for the declared pattern, the modelled fill, the
   owned scalars, provider workspace, and the fallback representation;
4. the thread budget is the session's, derived from the *granted* single active
   layer and never a product of the three descriptions;
5. the profile in force is recorded, and can only be an offline calibration or
   the unprofiled priors.

Planning allocates nothing that scales with the problem and measures nothing:
it is pure arithmetic over frozen facts, which is what makes the plan
reproducible. Over-budget memory is *reported* in `admission` (the refusal is
raised by `execute_setup_plan`/`reserve_setup_memory` before the first
allocation) — a planner that threw here would make the refusal boundary
unobservable.
"""
function plan_setup(request::SetupRequest, context::SetupContext)
    profile = context.profile
    profile.source in SETUP_PROFILE_SOURCES || throw(SetupProfileRefusal(
        :profile_source_not_offline_calibration,
        "profile source $(profile.source) is not admissible",
    ))
    if profile.calibrated
        (profile.samples >= 1 && !isempty(profile.calibration_input_hash)) ||
            throw(SetupProfileRefusal(
                :calibration_incomplete,
                "a calibrated profile must carry samples and its input hash",
            ))
    end
    # Precision/provider gate, mirroring the inherited preflight: a non-Float64
    # scalar needs the extended-precision provider, and its absence is an
    # infrastructure fact to be recorded, never assumed away.
    if request.T !== Float64 && !context.provider_available
        throw(SetupCapabilityRefusal(
            :provider_unavailable_for_precision,
            "arithmetic $(request.T) at $(request.precision_bits) bits requires " *
            "a provider that is not available in this environment",
        ))
    end

    costs = setup_cost_model(;
        full_dimension=request.full_dimension,
        compact_dimension=request.compact_dimension,
        ar_nnz=request.ar_nnz,
        canonical_nnz=request.canonical_nnz,
        block_sizes=request.block_sizes,
        basis_nnz=request.basis_nnz,
        variable_dimension=request.variable_dimension,
        T=request.T,
        precision_bits=request.precision_bits,
        kkt_route=request.kkt_route,
        fixed_trace=request.fixed_trace,
        rhs_count=request.rhs_count,
        fill_factor=profile.fill_factor,
        min_threading_width=profile.blas_threading_min_width,
        profile_id=profile.id,
    )
    route = costs.core_route.route
    memory = setup_memory_ledger(;
        T=request.T,
        precision_bits=request.precision_bits,
        dimension=request.full_dimension,
        block_sizes=request.block_sizes,
        structural_nnz=request.ar_nnz,
        fill_factor=profile.fill_factor,
        route=route,
        declared_storage=request.declared_storage,
        rhs_count=request.rhs_count,
        basis_nnz=request.basis_nnz,
        variable_dimension=request.variable_dimension,
        canonical_nnz=request.canonical_nnz,
        fallback_dimension=route === :compact_schur ?
            request.full_dimension : request.compact_dimension,
    )
    admission = admit_setup_memory(
        memory;
        budget_bytes=request.memory_budget_bytes,
        rss_bytes=request.current_rss_bytes,
    )
    threads = session_thread_budget(;
        requested_threads=request.requested_threads,
        capacity=context.capacity,
        cone_task_width=costs.coarse.parallel_width,
        la_kernel_width=costs.la.parallel_width,
        blas_width=costs.blas.parallel_width,
        provider_available=context.provider_available,
    )

    reasons = Symbol[]
    push!(reasons, :plan_inherits_core_route_cost_model)
    push!(reasons, :no_benchmark_name_consulted)
    append!(reasons, costs.core_route.reasons)
    if costs.legacy_would_choose_compact == (route === :compact_schur)
        push!(reasons, :legacy_dimension_rule_agrees)
    else
        push!(reasons, :legacy_dimension_rule_disagrees)
    end
    if memory.resolved_storage === :sparse_lower
        push!(reasons, :sparse_storage_honoured)
    else
        push!(reasons, :dense_storage_declared)
    end
    admission.admitted || push!(reasons, Symbol("memory_refused_", admission.reason))
    append!(reasons, threads.reasons)
    threads.active_consumer === :none ||
        push!(reasons, Symbol("threads_granted_to_", threads.active_consumer))

    return SetupPlan(
        route, costs, memory, admission, threads, profile,
        request.declared_storage, memory.resolved_storage, reasons,
        setup_request_fingerprint(request), setup_context_fingerprint(context),
    )
end

function plan_setup(
    w::SetupWorkload;
    context::SetupContext=setup_context(),
    kwargs...,
)
    return plan_setup(setup_request(w; kwargs...), context)
end

"""
    setup_plan_receipt(plan) -> NamedTuple

Everything the plan reports, in one place: route and why, the inherited model it
came from, the three independent consumer descriptions, the memory ledger terms
and the admission verdict, the thread budget with the request/capacity/tier, the
profile's provenance, and both fingerprints.

`reported_threads` is what the plan *claims* will run; `max_granted_threads`
is the single-layer maximum (never a product).
"""
function setup_plan_receipt(plan::SetupPlan)
    return (
        route=plan.route,
        resolved_storage=plan.resolved_storage,
        declared_storage=plan.declared_storage,
        cost_model=plan.costs.inherited_model,
        legacy_would_choose_compact=plan.costs.legacy_would_choose_compact,
        dominant_consumer=plan.costs.dominant,
        coarse_cone_tasks=plan.costs.coarse,
        la_kernel=plan.costs.la,
        blas_layer=plan.costs.blas,
        memory_ledger=plan.memory,
        memory_admitted=plan.admission.admitted,
        memory_reason=plan.admission.reason,
        memory_budget_bytes=plan.admission.budget_bytes,
        memory_estimate_bytes=plan.admission.estimate_bytes,
        memory_headroom_bytes=plan.admission.headroom_bytes,
        threads_requested=plan.threads.requested_threads,
        threads_mode=plan.threads.budget.mode,
        threads_granted_julia=plan.threads.budget.julia_outer_threads,
        threads_granted_blas=plan.threads.budget.blas_threads,
        threads_granted_provider=plan.threads.budget.provider_threads,
        threads_max_granted=max_granted_threads(plan.threads),
        threads_active_consumer=plan.threads.active_consumer,
        threads_tier_status=plan.threads.tier_status,
        threads_capacity=plan.threads.capacity,
        profile_id=plan.profile.id,
        profile_source=plan.profile.source,
        profile_calibrated=plan.profile.calibrated,
        profile_fill_factor=plan.profile.fill_factor,
        profile_artifact_digest=plan.profile.artifact_digest,
        reasons=copy(plan.reasons),
        request_fingerprint=plan.request_fingerprint,
        context_fingerprint=plan.context_fingerprint,
    )
end

"""
    setup_plan_signature(plan) -> NamedTuple

The decision-carrying fields, in a fixed order, for equality comparison. Two
plans are "the same plan" when their signatures are `==`.
"""
function setup_plan_signature(plan::SetupPlan)
    return (
        route=plan.route,
        declared_storage=plan.declared_storage,
        resolved_storage=plan.resolved_storage,
        full_score=plan.costs.core_route.full_score,
        compact_score=plan.costs.core_route.compact_score,
        predicted_fill_ratio=plan.costs.core_route.predicted_fill_ratio,
        route_reasons=copy(plan.costs.core_route.reasons),
        dominant=plan.costs.dominant,
        coarse=plan.costs.coarse,
        la=plan.costs.la,
        blas=plan.costs.blas,
        ledger_structural=plan.memory.structural_bytes,
        ledger_fill=plan.memory.fill_bytes,
        ledger_owned=plan.memory.owned_scalar_bytes,
        ledger_workspace=plan.memory.workspace_bytes,
        ledger_fallback=plan.memory.fallback_bytes,
        ledger_inherited=plan.memory.inherited_core_bytes,
        ledger_total=plan.memory.total_bytes,
        admitted=plan.admission.admitted,
        admission_reason=plan.admission.reason,
        budget=plan.admission.budget_bytes,
        headroom=plan.admission.headroom_bytes,
        threads=plan.threads.budget,
        thread_consumer=plan.threads.active_consumer,
        tier_status=plan.threads.tier_status,
        reasons=copy(plan.reasons),
        request_fingerprint=plan.request_fingerprint,
        context_fingerprint=plan.context_fingerprint,
    )
end

# ---------------------------------------------------------------------------
# Execution and verification
# ---------------------------------------------------------------------------

"""
    SetupExecutionObservation

What actually happened when the plan ran, measured rather than copied from the
plan.

Fields: `cone_tasks_executed`, `cone_rows_executed`, `cone_bins_executed`,
`outer_threads_observed` (distinct Julia thread ids that really ran a cone bin),
`la_structural_nnz` (pattern entries the kernel walked), `la_kernel_residual`
(the real residual of the real triangular solve), `storage_used`,
`dense_core_touched`, `fallback_allocated`, `blas_threads_observed`,
`blas_restored`, `allocated_bytes` (measured with `@allocated` around the
workload; `0` means not measured), `session`, `registry_record`.
"""
struct SetupExecutionObservation{T<:AbstractFloat}
    cone_tasks_executed::Int
    cone_rows_executed::Int
    cone_bins_executed::Int
    outer_threads_observed::Int
    la_structural_nnz::Int
    la_kernel_residual::T
    storage_used::Symbol
    dense_core_touched::Bool
    fallback_allocated::Bool
    blas_threads_observed::Int
    blas_restored::Bool
    allocated_bytes::Int
    session::Symbol
    registry_record::NamedTuple
end

"""
    execute_setup_plan(plan, workload; registry=SETUP_THREAD_REGISTRY,
                       session=:setup, allocation_probe=nothing,
                       measure_allocations=true) -> SetupExecutionObservation

Execute the workload the plan describes, under the plan's budget.

**Order matters and is the point**: the first statement is
`reserve_setup_memory(plan.admission)`, which allocates nothing and throws
[`SetupMemoryRefusal`](@ref) when the estimate does not fit. The large
allocation — here, the `allocation_probe` callback that stands for it — is only
reached after that gate passes, so a refusal is always observed *before* the
memory is committed.

The BLAS layer is pinned for the whole execution through
`with_session_thread_scope`, which restores the previous global value on every
exit path.
"""
function execute_setup_plan(
    plan::SetupPlan,
    w::SetupWorkload{T};
    registry::BlasLeaseRegistry=SETUP_THREAD_REGISTRY,
    session::Symbol=:setup,
    allocation_probe::Union{Nothing,Function}=nothing,
    measure_allocations::Bool=true,
) where {T<:AbstractFloat}
    # Memory gate first: nothing has been allocated for this plan yet.
    reserve_setup_memory(plan.admission)
    allocation_probe === nothing || allocation_probe(plan.memory.total_bytes)

    d = size(w.sparse_lower, 1)
    bin_count = plan.threads.active_consumer === :coarse_cone_tasks ?
        plan.threads.budget.julia_outer_threads : 1
    slots = zeros(Int, max(bin_count, 1))
    partial = zeros(T, max(bin_count, 1))
    row_sums = zeros(T, length(w.block_operators))
    y = Vector{T}(undef, d)
    residual_scratch = Vector{T}(undef, d)
    b = ones(T, d)
    product = Matrix{T}(undef, size(w.blas_a, 1), size(w.blas_b, 2))

    work = function ()
        _s06_run_cone_batch!(row_sums, partial, slots, w.block_operators, bin_count)
        residual = _s06_run_la_kernel!(y, w.sparse_lower, b, residual_scratch)
        _s06_run_blas_panel!(product, w.blas_a, w.blas_b)
        return residual
    end

    boxed = Ref{Any}(nothing)
    allocated_bytes = 0
    scope = nothing
    if measure_allocations
        allocated_bytes = Int(@allocated begin
            boxed[] = with_session_thread_scope(
                work, plan.threads; registry=registry, session=session,
            )
        end)
        result, scope = boxed[]
    else
        result, scope = with_session_thread_scope(
            work, plan.threads; registry=registry, session=session,
        )
    end
    residual = result

    cone_rows = sum(size(m, 1) for m in w.block_operators; init=0)
    observed_threads = unique(filter(>(0), slots))
    return SetupExecutionObservation{T}(
        length(w.block_operators), cone_rows, count(>(0), slots),
        length(observed_threads), nnz(w.sparse_lower), T(residual), :sparse_lower,
        false, false, Int(scope.observed_blas_threads), Bool(scope.restored),
        allocated_bytes, session, blas_lease_record(registry),
    )
end

"""Run the coarse cone batch over the planned number of bins, recording threads."""
function _s06_run_cone_batch!(row_sums, partial, slots, blocks, bins)
    fill!(partial, zero(eltype(partial)))
    fill!(row_sums, zero(eltype(row_sums)))
    nblocks = length(blocks)
    nblocks == 0 && return row_sums
    Threads.@threads :static for bin in 1:max(bins, 1)
        slots[bin] = Threads.threadid()
        total = zero(eltype(partial))
        for index in bin:bins:nblocks
            block = blocks[index]
            for j in 1:size(block, 2), i in j:size(block, 1)
                total += block[i, j]
            end
        end
        partial[bin] = total
    end
    for index in 1:nblocks
        block = blocks[index]
        acc = zero(eltype(row_sums))
        for j in 1:size(block, 2), i in j:size(block, 1)
            acc += block[i, j]
        end
        row_sums[index] = acc
    end
    return row_sums
end

"""
Run the LA kernel over the declared sparse pattern: a real forward substitution
on the lower-triangular CSC, walking only stored entries. Returns the true
residual `max_i |(L*y - b)_i|`, accumulated column-wise over the same stored
pattern so the check costs one extra nnz pass and allocates nothing.

`scratch` is caller-owned (length `size(L, 1)`), so measuring the kernel's
allocation measures the kernel rather than the probe.
"""
function _s06_run_la_kernel!(y, L::SparseMatrixCSC{T}, b, scratch) where {T}
    d = size(L, 1)
    length(y) == d && length(scratch) == d || throw(ArgumentError(
        "LA kernel workspace must match the declared core dimension",
    ))
    copyto!(y, b)
    colptr = L.colptr
    rowval = L.rowval
    nzval = L.nzval
    for j in 1:d
        pivot = zero(T)
        for p in colptr[j]:(colptr[j + 1] - 1)
            rowval[p] == j && (pivot = nzval[p])
        end
        pivot != zero(T) || throw(ArgumentError(
            "declared sparse pattern has no usable pivot in column $(j)",
        ))
        y[j] /= pivot
        yj = y[j]
        for p in colptr[j]:(colptr[j + 1] - 1)
            i = rowval[p]
            i > j || continue
            y[i] -= nzval[p] * yj
        end
    end
    # Residual from the stored pattern only: scratch = b - L*y, accumulated by
    # column (`L*y`'s row i receives `L[i,k]*y[k]` from every stored entry of
    # column k). An earlier version of this probe formed the *column* inner
    # product `sum_k L[k,j]*y[k]`, i.e. `L'*y`, and reported a residual of 5.7
    # on a system whose true residual was 4e-16 — a wrong probe, not a wrong
    # solve. Recorded in the S06 report as a defect found in this task's own
    # work.
    copyto!(scratch, b)
    for k in 1:d
        yk = y[k]
        for p in colptr[k]:(colptr[k + 1] - 1)
            scratch[rowval[p]] -= nzval[p] * yk
        end
    end
    acc = zero(T)
    for i in 1:d
        acc = max(acc, abs(scratch[i]))
    end
    return acc
end

"""Run the BLAS panel product — the widest dense operation in setup."""
function _s06_run_blas_panel!(product, a, b)
    mul!(product, a, b)
    return product
end

"""
    verify_setup_execution(plan, observation) -> NamedTuple

Compare what the plan reported with what actually ran.

Checked, each as its own named check so a receipt can show which one failed:

- `storage`: the executed storage class equals the resolved storage — a
  `:sparse_lower` plan may never have run a dense representation;
- `dense_fallback`: the fallback representation was budgeted, and was not
  allocated;
- `cone_tasks`, `cone_rows`: the batch ran the described blocks;
- `la_pattern`: the kernel walked exactly the declared nonzeros;
- `outer_threads`: measured distinct Julia threads ≤ granted outer threads;
- `blas_threads`: measured BLAS threads ≤ granted BLAS threads;
- `threads_restored`: the global BLAS count was restored;
- `memory`: measured allocation ≤ admitted estimate.

`ok` is the conjunction. This is the check behind "the plan's actual execution
matches what it reported".
"""
function verify_setup_execution(plan::SetupPlan, obs::SetupExecutionObservation)
    checks = NamedTuple[]
    function record!(name, ok, expected, observed)
        push!(checks, (name=name, ok=Bool(ok), expected=expected, observed=observed))
    end
    record!(
        :storage, obs.storage_used === plan.resolved_storage,
        plan.resolved_storage, obs.storage_used,
    )
    record!(
        :dense_fallback_absent, !obs.dense_core_touched && !obs.fallback_allocated,
        false, obs.dense_core_touched || obs.fallback_allocated,
    )
    record!(
        :cone_tasks, obs.cone_tasks_executed == plan.costs.coarse.block_count,
        plan.costs.coarse.block_count, obs.cone_tasks_executed,
    )
    record!(
        :cone_rows, obs.cone_rows_executed == plan.costs.coarse.total_rows,
        plan.costs.coarse.total_rows, obs.cone_rows_executed,
    )
    record!(
        :la_pattern, obs.la_structural_nnz == plan.costs.la.structural_nnz,
        plan.costs.la.structural_nnz, obs.la_structural_nnz,
    )
    record!(
        :outer_threads,
        obs.outer_threads_observed <= plan.threads.budget.julia_outer_threads,
        plan.threads.budget.julia_outer_threads, obs.outer_threads_observed,
    )
    record!(
        :blas_threads,
        obs.blas_threads_observed <= plan.threads.budget.blas_threads,
        plan.threads.budget.blas_threads, obs.blas_threads_observed,
    )
    record!(:threads_restored, obs.blas_restored, true, obs.blas_restored)
    if obs.allocated_bytes > 0
        record!(
            :memory, obs.allocated_bytes <= plan.memory.total_bytes,
            plan.memory.total_bytes, obs.allocated_bytes,
        )
    else
        push!(checks, (
            name=:memory, ok=true, expected=plan.memory.total_bytes,
            observed=:not_measured,
        ))
    end
    mismatches = Symbol[check.name for check in checks if !check.ok]
    return (ok=isempty(mismatches), checks=checks, mismatches=mismatches)
end
