# Q01 — case manifest, arithmetic axis, environment identity and result schema.
#
# Packet requirement: "建立真实输入shape trace与完整结果schema；允许统计失败，
# 不删除慢/失败案例."
#
# ## The rule this file exists to enforce
#
# ADR-003 §6 and the Q01 card both forbid reading a benchmark NAME to decide a
# numeric route. The manifest therefore describes only *inputs*. It has no field
# for a route, a formulation or a provider, and `case_settings` constructs
# `Settings` from an arithmetic type, tolerances and limits alone — never from
# the case id or family. `test/rebuild/dependency_rules.jl` asserts that
# structurally.
#
# ## Three identity axes, kept separate
#
# A number is only comparable to another number taken on the same
#   (1) manifest      — which inputs, hashed into `inputs_fingerprint`;
#   (2) arithmetic    — which `T`, precision and provider revision;
#   (3) environment   — which Julia project resolved those providers;
#   (4) threads       — requested vs actually executed.
#
# `environment_facts()` records (3) with the active `Project.toml`/`Manifest.toml`
# paths, their SHA-256s, and the resolved provider versions *and git revisions*.
# A figure measured under `$REBUILD_ENV` (capability-enabled: MFLA, BFLA, QDLDL)
# is NOT comparable to one measured in the default project (provider-free), so
# every payload carries the environment it came from.
#
# ## Failures are data
#
# A case that fails is recorded with its outcome. Nothing here filters, skips or
# deletes on failure — `summarize` counts failures and keeps every row.
#
# ## What this revision changed (2026-09-11, Q01 re-run for the provider env)
#
# The declared shape trace of the eight cases below is retained **verbatim** from
# the first Q01 revision, and `observed_shape_trace` now derives the real trace
# from the compiled canonical program. They do not agree for every case; the
# disagreement is returned as data (`declared_vs_observed`) instead of being
# silently rewritten, because "the hand-written table was right" is a claim that
# has to be checkable.

module RebuildManifest

using SDPX
using SHA
using TOML
using Serialization
using LinearAlgebra: BLAS
using SparseArrays: nnz

export RebuildCase, rebuild_cases, shape_trace, observed_shape_trace,
       declared_vs_observed, case_settings, result_row, failure_row, summarize,
       cases_fingerprint, inputs_fingerprint, RESULT_SCHEMA, ARITHMETIC_ARMS,
       arithmetic_arm, resolve_arithmetic, environment_facts, provider_facts,
       PROVIDER_PACKAGES, load_provider!, provider_extension_active,
       bigfloat_cell_facts, cell_identity_snapshot, null_reason

# ---------------------------------------------------------------------------
# Arithmetic axis
# ---------------------------------------------------------------------------
#
# Arithmetic is an ENVIRONMENT axis, not a property of a case name: the same
# eight inputs are measured once per arithmetic, and nothing selects a route.
# An arm whose provider packages are not loadable is reported `not_run` with the
# resolution fact as its reason — never estimated, never zero.

"""
    ArithmeticArm

One measurement arm. `T` is resolved at run time (`resolve_arithmetic`) because
a fixed-width MultiFloat type does not exist in an environment that has not
loaded `MultiFloats`; naming it in source would make this file unloadable in the
default project.
"""
struct ArithmeticArm
    id::Symbol
    precision_bits::Int
    provider::Symbol
    provider_packages::Vector{String}
    rounding_mode::String
    note::String
end

const ARITHMETIC_ARMS = (
    ArithmeticArm(:float64, 53, :stdlib, String[],
        "ieee754_round_nearest_ties_even",
        "default project or REBUILD_ENV; stdlib LAPACK/CHOLMOD path"),
    ArithmeticArm(:multifloat_x2, 106, :mfla, ["MultiFloats", "MultiFloatLinearAlgebra"],
        "round_nearest_ties_even_float64_lanes",
        "requires the MFLA provider extension SDPXMultiFloatLinearAlgebraExt"),
    ArithmeticArm(:bigfloat_256, 256, :bfla, ["BigFloatLinearAlgebra"],
        "mpfr_round_nearest",
        "requires the BFLA provider extension SDPXBigFloatLinearAlgebraExt"),
)

function arithmetic_arm(id::Symbol)
    for arm in ARITHMETIC_ARMS
        arm.id === id && return arm
    end
    throw(ArgumentError("unknown arithmetic arm $(repr(id)); known: " *
        join((String(a.id) for a in ARITHMETIC_ARMS), ", ")))
end

"""
    resolve_arithmetic(id) -> Type or nothing

The Julia type for an arm, or `nothing` when the packages it names cannot be
loaded here. `nothing` is the honest answer in a provider-free project and is
reported as `not_run` with the resolution fact, never as a zero.
"""
function resolve_arithmetic(id::Symbol)
    id === :float64 && return Float64
    id === :bigfloat_256 && return BigFloat
    if id === :multifloat_x2
        mf = load_provider!("MultiFloats")
        mf === nothing && return nothing
        return getproperty(mf, :Float64x2)
    end
    throw(ArgumentError("unknown arithmetic arm $(repr(id))"))
end

# ---------------------------------------------------------------------------
# Provider and environment identity
# ---------------------------------------------------------------------------

const PROVIDER_PACKAGES = (
    (name="MultiFloats", uuid="bdf0d083-296b-4888-a5b6-7498122e68a5",
     role="fixed-width arithmetic types (Float64x2/3/4)"),
    (name="MultiFloatLinearAlgebra", uuid="642d9d30-8e28-45ca-9d81-256429ea358f",
     role="MFLA factorizations; provider revision matters for every MF claim"),
    (name="BigFloatLinearAlgebra", uuid="44d352a4-380e-4c6a-9c2a-31e5bfe329aa",
     role="BFLA factorizations and MPFR context"),
    (name="QDLDL", uuid="bfc457fd-c171-5ab7-bd9e-d5dbfc242d63",
     role="sparse LDL provider; changes which sparse routes can execute"),
    (name="GenericLinearAlgebra", uuid="14197337-ba66-59df-a3e3-ca00e7dcff7a",
     role="generic-precision dense fallback kernels"),
)

"""
    load_provider!(name) -> module or nothing

Resolve and load a provider package by name **without** naming its types in
source. Returns `nothing` when the active environment cannot resolve it, which
is an infrastructure fact about the environment, not a numeric failure
(ADR-003 §3).
"""
function load_provider!(name::AbstractString)
    entry = nothing
    for candidate in PROVIDER_PACKAGES
        candidate.name == name && (entry = candidate)
    end
    entry === nothing &&
        throw(ArgumentError("$name is not in the tracked provider list"))
    try
        return Base.require(Base.PkgId(Base.UUID(entry.uuid), name))
    catch
        return nothing
    end
end

"""Is SDPX's provider extension for `name` active in this process?"""
function provider_extension_active(name::AbstractString)
    extension = name == "MultiFloatLinearAlgebra" ? :SDPXMultiFloatLinearAlgebraExt :
                name == "BigFloatLinearAlgebra" ? :SDPXBigFloatLinearAlgebraExt :
                name == "GenericLinearAlgebra" ? :SDPXGenericLinearAlgebraExt :
                name == "MultiFloats" ? :SDPXMultiFloatsExt :
                nothing
    extension === nothing && return nothing
    return Base.get_extension(SDPX, extension) !== nothing
end

function _sha256_file(path)
    (path === nothing || !isfile(path)) && return nothing
    return bytes2hex(SHA.sha256(read(path)))
end

function _git_revision(path)
    (path === nothing || !isdir(path)) && return nothing
    try
        return strip(read(`git -C $path rev-parse HEAD`, String))
    catch
        return nothing
    end
end

"""Is a provider checkout dirty? A result from a dirty provider is marked as such."""
function _git_dirty(path)
    (path === nothing || !isdir(path)) && return nothing
    try
        return !isempty(strip(read(`git -C $path status --porcelain`, String)))
    catch
        return nothing
    end
end

"""
    provider_facts(manifest_path) -> Vector{Dict}

What the **active environment** resolves, read from its `Manifest.toml`:
version, dev path, and the git revision of that path. Reading the manifest does
not load the packages, so a Float64 arm can report the provider configuration
that a *different* arm would get without compiling it — which matters because
Julia 1.12 must not compile the MFLA and BFLA specialisations in one process.
"""
function provider_facts(manifest_path)
    rows = Dict{String,Any}[]
    deps = Dict{String,Any}()
    if manifest_path !== nothing && isfile(manifest_path)
        try
            parsed = TOML.parsefile(manifest_path)
            deps = get(parsed, "deps", Dict{String,Any}())
        catch
            deps = Dict{String,Any}()
        end
    end
    # Julia's TOML parser represents the `[[deps.X]]` array-of-tables syntax as a
    # one-element vector, NOT as the dict Python's `tomllib` produces. Reading
    # `get(record, "version")` on the vector silently yields `nothing`, which
    # would erase exactly the provider version and revision a measurement has to
    # name. Unwrap it, and return `nothing` rather than a plausible-looking
    # default when the shape is unexpected.
    unwrap(record) = record isa AbstractVector ?
        (isempty(record) ? nothing : first(record)) : record
    for entry in PROVIDER_PACKAGES
        record = unwrap(get(deps, entry.name, nothing))
        path = record isa AbstractDict ? get(record, "path", nothing) : nothing
        # Two different facts, both recorded: a package can be resolvable from
        # the whole load path (`@v#.#` is on it) while being absent from the
        # active project's Manifest. Only the second one is what the active
        # environment "resolves", and only the first one decides whether
        # `Base.require` succeeds.
        resolvable = try
            Base.identify_package(entry.name) !== nothing
        catch
            nothing
        end
        push!(rows, Dict{String,Any}(
            "name" => entry.name,
            "role" => entry.role,
            "resolved" => record !== nothing,
            "resolvable_in_load_path" => resolvable,
            "version" => record isa AbstractDict ? get(record, "version", nothing) : nothing,
            "path" => path,
            "revision" => _git_revision(path),
            "worktree_dirty" => _git_dirty(path),
            "loaded_in_process" => false, # filled by the caller after the fact
            "extension_active" => provider_extension_active(entry.name),
        ))
    end
    return rows
end

"""
    environment_facts() -> NamedTuple

The environment half of a measurement's identity. `label` is what a reader
compares against: a timing taken under a capability-enabled environment must
never be attributed to the provider-free default project.
"""
function environment_facts()
    project = Base.active_project()
    env_dir = project === nothing ? nothing : dirname(project)
    manifest_path = env_dir === nothing ? nothing : joinpath(env_dir, "Manifest.toml")
    label = get(ENV, "SDPX_MEASURE_ENV_LABEL", "")
    if isempty(label)
        # Path comparison via filesystem identity, not string equality:
        # `normpath` keeps a trailing separator when the last component is "..".
        repo_root = normpath(joinpath(@__DIR__, "..", ".."))
        label = if env_dir === nothing
            "unknown"
        elseif occursin("rebuild-env", env_dir)
            # Capability-enabled: SDPX provider extensions active, QDLDL present.
            "rebuild_env"
        elseif try Base.samefile(normpath(env_dir), repo_root) catch; false end
            # Provider-free: Pkg.test() exercises Float64 only.
            "sdpx_default_project"
        else
            "other:" * basename(env_dir)
        end
    end
    return (
        label=label,
        project_path=project,
        environment_dir=env_dir,
        manifest_path=manifest_path,
        project_sha256=_sha256_file(project),
        manifest_sha256=_sha256_file(manifest_path),
        depot_path=get(ENV, "JULIA_DEPOT_PATH", nothing),
        julia_version=string(VERSION),
        julia_exe=joinpath(Sys.BINDIR, Base.julia_exename()),
        kernel=string(Sys.KERNEL),
        machine=string(Sys.MACHINE),
        cpu_threads=Sys.CPU_THREADS,
        total_memory_bytes=Int(Sys.total_memory()),
        providers=provider_facts(manifest_path),
    )
end

# ---------------------------------------------------------------------------
# Case builders. Every one is a pure input constructor, parameterised by `T`.
# ---------------------------------------------------------------------------

"""
    RebuildCase

An input descriptor. Deliberately contains no strategy field.

- `id`: stable identifier, used for reporting only, never for dispatch.
- `family`: `:lp`/`:soc`/`:psd`/`:exp`/`:power`/`:mixed` — describes the cone,
  not the algorithm.
- `n`, `m`, `cones`: the **declared** shape trace, retained verbatim from the
  first Q01 revision. `observed_shape_trace` compiles the model and reports what
  the canonical program actually contains; the two are returned side by side.
- `build`: constructs the model in a requested arithmetic `T`. Pure input
  construction.
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

function _lp_afiro_style(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :slack, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :capacity_1, x[1] + x[2] + s[1] - T(4), SDPX.ZeroCone())
    SDPX.constraint!(model, :capacity_2, T(2) * x[1] + x[2] + s[2] - T(5), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), T(3) * x[1] + T(2) * x[2])
    return model
end

function _lp_degenerate(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :slack, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :dup_1, x[1] + x[2] + s[1] - one(T), SDPX.ZeroCone())
    SDPX.constraint!(model, :dup_2, x[1] + x[2] + s[2] - one(T), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + x[2])
    return model
end

function _soc_disk(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[one(T), x[1], x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _soc_large(::Type{T}, k::Int) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, k - 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :cone, Any[one(T); collect(x)], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1])
    return model
end

function _soc_many_small(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 6; domain=SDPX.Reals())
    for block in 1:3
        SDPX.constraint!(model, Symbol(:cone, block),
            Any[one(T), x[2 * block - 1], x[2 * block]], SDPX.LorentzCone())
    end
    SDPX.objective!(model, SDPX.Minimize(),
        -(one(T) / T(3)) * (x[1] + x[3] + x[5]))
    return model
end

function _psd_2x2(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :trace, X[1, 1] + X[2, 2] - one(T), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    return model
end

function _mixed_cones(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 3; domain=SDPX.Reals())
    SDPX.constraint!(model, :disk, Any[one(T), x[1], x[2]], SDPX.LorentzCone())
    SDPX.constraint!(model, :orth, Any[one(T) + x[3]], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), -x[1] - T(0.5) * x[2])
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
            [(:zero, 31), (:soc, 32)], (T) -> _soc_large(T, 32), -1.0),
        RebuildCase(:soc_k128, :soc, 127, 128,
            [(:zero, 127), (:soc, 128)], (T) -> _soc_large(T, 128), -1.0),
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
# Real shape trace and input fingerprint
# ---------------------------------------------------------------------------

"""
    observed_shape_trace(case, T) -> NamedTuple or nothing

Compile the case in arithmetic `T` and report what the **canonical program**
actually contains: variable count, canonical slack dimension, barrier degree,
constraint-matrix shape and nnz, and the canonical block list.

This is the trace the first Q01 revision only *declared*. When the two disagree
the disagreement is reported, never silently reconciled.
"""
function observed_shape_trace(case::RebuildCase, ::Type{T}) where {T<:AbstractFloat}
    model = case.build(T)
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    blocks = [(String(block.cone), Int(block.dimension), Int(block.length))
              for block in canonical.cone_layout.blocks]
    return (
        num_variables=length(canonical.c),
        canonical_slack_dimension=Int(canonical.cone_layout.dimension),
        barrier_degree=Int(canonical.cone_layout.barrier_degree),
        equality_rows=size(canonical.A, 1),
        nnz_constraint_matrix=Int(nnz(canonical.A)),
        precision_bits=Int(canonical.precision_bits),
        cone_blocks=blocks,
    )
end

"""Field-level differences between the declared and the observed trace."""
function declared_vs_observed(case::RebuildCase, observed)
    observed === nothing && return ["observed_trace: not measured"]
    mismatches = String[]
    observed.num_variables == case.n ||
        push!(mismatches, "num_variables declared=$(case.n) observed=$(observed.num_variables)")
    observed.equality_rows == case.m ||
        push!(mismatches, "equality_rows declared=$(case.m) observed=$(observed.equality_rows)")
    declared_blocks = [(String(kind), size) for (kind, size) in case.cones]
    observed_blocks = [(kind, dimension) for (kind, dimension, _) in observed.cone_blocks]
    declared_blocks == observed_blocks ||
        push!(mismatches, "cone_blocks declared=$(declared_blocks) observed=$(observed_blocks)")
    return mismatches
end

"""
    inputs_fingerprint(case, T, precision_bits; tolerances, limits) -> String

SHA-256 over the *solver's actual numeric input* — the canonical `A`, `b`, `c`,
the canonical block layout — plus the arithmetic, precision, tolerances and
limits. Two runs with the same fingerprint describe the same problem at the same
target, so a re-run is comparable (Q01 acceptance item 1).

`Serialization` is used as the byte encoding; it is deterministic within a Julia
version, and the Julia version is recorded beside the hash. Route, provider and
formulation are deliberately NOT hashed: they are outcomes, not inputs.
"""
function inputs_fingerprint(case::RebuildCase, ::Type{T}, precision_bits::Int;
    tolerances::NamedTuple, limits::NamedTuple,
) where {T<:AbstractFloat}
    model = case.build(T)
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    blocks = [(String(block.cone), Int(block.dimension), Int(block.length))
              for block in canonical.cone_layout.blocks]
    buffer = IOBuffer()
    Serialization.serialize(buffer, (
        String(case.id), string(T), precision_bits,
        canonical.c, canonical.b,
        canonical.A.colptr, canonical.A.rowval, canonical.A.nzval,
        blocks,
        (tol=string(tolerances.primal), dual=string(tolerances.dual),
         gap=string(tolerances.gap)),
        (iterations=limits.iterations, time=string(limits.time),
         threads=limits.threads),
    ))
    return bytes2hex(SHA.sha256(take!(buffer)))
end

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

"""
    RESULT_SCHEMA

The complete set of fields one case run produces. Every field is always present;
a field that could not be measured is `nothing`, never `0` (ADR-003 §3), and its
reason is carried in the row's `unavailable` list.
"""
const RESULT_SCHEMA = (
    # identity
    :id, :family, :family_cone_count, :arithmetic, :precision_bits,
    # outcome — recorded whatever it is, including failure
    :status, :certificate_valid, :certificate_method,
    :objective, :objective_text, :objective_error, :objective_error_text,
    :iterations, :factorizations, :backtracking, :termination_reason,
    # executed route facts, read from the receipt rather than assumed
    :requested_kkt_route, :executed_kkt_route, :executed_kkt_storage,
    :executed_provider, :executed_factorization_kernel, :fallback_reason,
    # diagnostics
    :sigma_used, :alpha_aff, :alpha_combined, :correction_norm, :retry_reason,
    # cost
    :seconds, :allocated_bytes, :rss_delta_bytes,
    # resident workspace capacity (packet §5.6: workspace 常驻容量)
    :workspace_bytes, :estimated_workspace_bytes,
    # BigFloat cell identity (ADR-003 §7). `nothing` for Float64 arithmetic.
    :bigfloat_value_cells, :bigfloat_distinct_cells,
    # failure bookkeeping
    :threw, :exception,
)

"""Format one `nothing` reason as `"field — reason"` for the `unavailable` list."""
null_reason(field::Symbol, reason::AbstractString) = "$(field) — $(reason)"

"""
    result_row(case, result; arithmetic, seconds, allocated_bytes, rss_delta_bytes)

Project a solve result into `RESULT_SCHEMA`. A field the engine does not publish
is `nothing` with the reason recorded in `unavailable`, never a fabricated zero.
"""
function result_row(case::RebuildCase, result;
    arithmetic::Symbol, seconds::Float64, allocated_bytes::Int,
    rss_delta_bytes,
)
    unavailable = String[]
    diagnostics = try
        SDPX.diagnostics(result)
    catch
        nothing
    end
    termination = diagnostics === nothing ? nothing : diagnostics.termination
    selected = diagnostics === nothing ? nothing : diagnostics.selected_algorithms
    memory = (diagnostics === nothing || !hasproperty(diagnostics, :memory)) ?
        nothing : diagnostics.memory

    function get_or_nothing(source, field, reason::AbstractString)
        if source === nothing || !hasproperty(source, field)
            push!(unavailable, null_reason(field, reason))
            return nothing
        end
        value = getproperty(source, field)
        value isa Symbol ? String(value) : value
    end
    no_diagnostics = "engine diagnostics did not publish this field"

    certificate = SDPX.certificate(result)
    # Objective values stay in the requested arithmetic; the Float64 projection
    # is for reporting only and is labelled as such in the receipt.
    objective_full = certificate.primal_objective
    objective = Float64(objective_full)
    error_full = case.known_objective === nothing ? nothing :
                 abs(objective_full - typeof(objective_full)(case.known_objective))
    cell_facts = bigfloat_cell_facts(result)

    return (
        id=case.id,
        family=case.family,
        family_cone_count=length(case.cones),
        arithmetic=String(arithmetic),
        precision_bits=get_or_nothing(selected, :requested_precision_bits, no_diagnostics),
        status=String(SDPX.status(result)),
        certificate_valid=Bool(certificate.valid),
        certificate_method=String(certificate.method),
        objective=objective,
        objective_text=string(objective_full),
        # `nothing` when there is no independent oracle, never 0.
        objective_error=error_full === nothing ? nothing : Float64(error_full),
        objective_error_text=error_full === nothing ? nothing : string(error_full),
        iterations=get_or_nothing(termination, :iterations, no_diagnostics),
        factorizations=get_or_nothing(termination, :factorizations, no_diagnostics),
        backtracking=get_or_nothing(termination, :backtracking, no_diagnostics),
        termination_reason=get_or_nothing(termination, :reason, no_diagnostics),
        requested_kkt_route=get_or_nothing(selected, :requested_kkt_route, no_diagnostics),
        executed_kkt_route=get_or_nothing(selected, :executed_kkt_route, no_diagnostics),
        executed_kkt_storage=get_or_nothing(selected, :executed_kkt_storage, no_diagnostics),
        executed_provider=get_or_nothing(selected, :la_executed_provider, no_diagnostics),
        executed_factorization_kernel=get_or_nothing(selected, :executed_factorization_kernel, no_diagnostics),
        fallback_reason=get_or_nothing(selected, :fallback_reason, no_diagnostics),
        sigma_used=get_or_nothing(termination, :sigma_used, no_diagnostics),
        alpha_aff=get_or_nothing(termination, :alpha_aff, no_diagnostics),
        alpha_combined=get_or_nothing(termination, :alpha_combined, no_diagnostics),
        correction_norm=get_or_nothing(termination, :correction_norm, no_diagnostics),
        retry_reason=get_or_nothing(termination, :retry_reason, no_diagnostics),
        seconds=seconds,
        allocated_bytes=allocated_bytes,
        rss_delta_bytes=rss_delta_bytes,
        workspace_bytes=get_or_nothing(memory, :workspace_bytes, no_diagnostics),
        estimated_workspace_bytes=get_or_nothing(memory, :estimated_workspace_bytes, no_diagnostics),
        bigfloat_value_cells=cell_facts === nothing ? nothing : cell_facts.elements,
        bigfloat_distinct_cells=cell_facts === nothing ? nothing : cell_facts.distinct_cells,
        threw=false,
        exception="none",
    ), unavailable
end

"""A row for a case whose construction or solve threw. Kept, not dropped."""
function failure_row(case::RebuildCase, exception; arithmetic::Symbol,
    phase::Symbol, seconds::Float64, precision_bits,
)
    return (
        id=case.id, family=case.family,
        family_cone_count=length(case.cones),
        arithmetic=String(arithmetic), precision_bits=precision_bits,
        status="threw", certificate_valid=false, certificate_method="none",
        objective=nothing, objective_text=nothing,
        objective_error=nothing, objective_error_text=nothing,
        iterations=nothing, factorizations=nothing, backtracking=nothing,
        termination_reason=nothing,
        requested_kkt_route=nothing, executed_kkt_route=nothing,
        executed_kkt_storage=nothing, executed_provider=nothing,
        executed_factorization_kernel=nothing, fallback_reason=nothing,
        sigma_used=nothing, alpha_aff=nothing, alpha_combined=nothing,
        correction_norm=nothing, retry_reason=nothing,
        seconds=seconds, allocated_bytes=nothing, rss_delta_bytes=nothing,
        workspace_bytes=nothing, estimated_workspace_bytes=nothing,
        bigfloat_value_cells=nothing, bigfloat_distinct_cells=nothing,
        threw=true, exception="$(phase): $(sprint(showerror, exception))",
    )
end

"""
    case_settings(case, T; kwargs...) -> Settings{T}

Build `Settings` from an arithmetic type, tolerances and limits ONLY.

This function must never branch on `case.id`, `case.family` or any name, and it
must not name a route. That is the packet's no-name-dispatch rule, and
`test/rebuild/dependency_rules.jl` asserts it structurally.
"""
function case_settings(case::RebuildCase, ::Type{T};
    primal::Float64=1e-8, dual::Float64=1e-8, gap::Float64=1e-8,
    iterations::Int=400, time::Float64=180.0, threads::Int=1,
) where {T<:AbstractFloat}
    return SDPX.Settings(T;
        verbosity=0,
        tolerances=SDPX.Tolerances(T; primal=T(primal), dual=T(dual), gap=T(gap)),
        limits=SDPX.Limits(iterations=iterations, time=time, threads=threads),
    )
end

# ---------------------------------------------------------------------------
# BigFloat allocation disaggregation (ADR-003 §7, packet §5.6)
# ---------------------------------------------------------------------------

"""
    cell_identity_snapshot(result) -> Set{UInt} or nothing

`objectid` of every published primal and dual value. BigFloat is a mutable
struct, so two distinct cells holding equal values are distinguishable — that is
the "cell identity" axis the packet requires to be reported separately from
Julia heap bytes. Returns `nothing` for non-BigFloat results (not applicable,
not zero).
"""
function cell_identity_snapshot(result)
    arrays = try
        (SDPX.value(result), SDPX.dual(result))
    catch
        # Published values not retained: the axis is not reported, never zeroed.
        return nothing
    end
    eltype(arrays[1]) === BigFloat || return nothing
    ids = Set{UInt}()
    for array in arrays
        for element in array
            push!(ids, objectid(element))
        end
    end
    return ids
end

"""
    bigfloat_cell_facts(result) -> NamedTuple or nothing

Element / distinct-cell counts for a BigFloat result. `elements` minus
`distinct_cells` is the number of aliased pairs; a positive count would mean two
published entries share one mutable cell, which the packet forbids. `nothing`
for other arithmetics.
"""
function bigfloat_cell_facts(result)
    arrays = try
        (SDPX.value(result), SDPX.dual(result))
    catch
        # Published values not retained: the axis is not reported, never zeroed.
        return nothing
    end
    eltype(arrays[1]) === BigFloat || return nothing
    ids = UInt[]
    elements = 0
    for array in arrays
        for element in array
            push!(ids, objectid(element))
            elements += 1
        end
    end
    return (elements=elements, distinct_cells=length(unique(ids)))
end

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

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

Stable hash over the **declared** input shape trace only: dimensions, cone
signature, id, family. It deliberately does not hash `build`, which is a
closure. Unchanged from the first Q01 revision so the manifest identity remains
comparable across the two revisions; the per-arm content hash is
`inputs_fingerprint`.
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
