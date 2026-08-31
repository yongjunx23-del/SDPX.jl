#=====================================================================#
#    Typed public solve settings (B5).
#
#    The layer contains *policy types only*.  Every keyword is validated
#    on construction, every field has a concrete type (no `Any`), and the
#    typed policy is lowered to the existing all-auto frontend
#    `SolveOptions` through a single deterministic one-to-one mapping.
#    This file owns exactly the settings half of B5:
#      - `Tolerances{T}` / `Limits` / `Settings{T}`
#      - defaults and constructor validation
#      - `SolveOptions(settings)` / `resolve_solve_options`
#
#    It does not own and will not touch:
#      - `SolverOptions` / `ResolvedSolveOptions` / execution planning
#      - `SDPResult` / result retention consumption
#      - BLAS mutation (`blas_threads` is stored as request metadata only)
#      - orientation / dual-model compat, conic dual models, providers
#
#    The parent contract explicitly rejects `:dual` and orientation
#    keywords *by absence*: the strict constructors below do not accept
#    keyword `...`, so `Settings(; orientation=:dual)` and
#    `SolveOptions(settings)` are MethodErrors.
#=====================================================================#

"""
    Tolerances{T<:AbstractFloat}

Typed public stopping tolerances in arithmetic `T`.  Each field is
optional (`Tolerances(; primal=..., dual=..., gap=...)`); omitted fields
are `nothing` and mean: use SDPX's deterministic automatic target.

Validation on construction:
- explicit tolerances are finite and strictly positive in `T`;
- `nothing` is the only way to request automatic selection.

The default constructor is `Tolerances{T}()` — every tolerance automatic.
"""
struct Tolerances{T<:AbstractFloat}
    primal::Union{Nothing,T}
    dual::Union{Nothing,T}
    gap::Union{Nothing,T}

    function Tolerances{T}(
        primal::Union{Nothing,T},
        dual::Union{Nothing,T},
        gap::Union{Nothing,T},
    ) where {T<:AbstractFloat}
        _validate_tolerance(primal, "primal")
        _validate_tolerance(dual, "dual")
        _validate_tolerance(gap, "gap")
        return new{T}(primal, dual, gap)
    end
end

_validate_tolerance(::Nothing, ::AbstractString) = nothing

function _validate_tolerance(value::T, label::AbstractString) where {T<:AbstractFloat}
    isfinite(value) || throw(ArgumentError("$label tolerance must be finite, got $value"))
    value > zero(T) || throw(ArgumentError("$label tolerance must be strictly positive, got $value"))
    return nothing
end

function Tolerances(
    ::Type{T};
    primal::Union{Nothing,T}=nothing,
    dual::Union{Nothing,T}=nothing,
    gap::Union{Nothing,T}=nothing,
) where {T<:AbstractFloat}
    return Tolerances{T}(primal, dual, gap)
end

Tolerances{T}(; primal=nothing, dual=nothing, gap=nothing) where {T<:AbstractFloat} =
    Tolerances{T}(primal, dual, gap)

"""
    Limits

Typed public iteration and time limits.

- `iterations`: nonnegative `Int`; `0` (or `nothing`) requests the
  automatic default, which is currently the existing `SolveOptions`
  default of 200.
- `time`: finite and nonnegative `Float64`, or `Inf` for no limit.
- `threads`: strictly positive `Int`; `Limits(; threads=nothing)` stores
  the process auto count (`Base.Threads.nthreads()`), and the existing
  per-solve scheduling limit is inherited from that stored value.

`threads` and maximum iteration/time are scheduling/weight limits
interpreted by the existing option lowerer; no Julia thread pool or BLAS
thread count is mutated anywhere in this file.
"""
struct Limits
    iterations::Int
    time::Float64
    threads::Int

    function Limits(
        iterations::Int,
        time::Float64,
        threads::Int,
    )
        iterations >= 0 ||
            throw(ArgumentError("iterations must be a nonnegative Int (0 or nothing means automatic), got $iterations"))
        (isinf(time) || isfinite(time)) && time >= 0.0 ||
            throw(ArgumentError("time must be finite and nonnegative, or Inf, got $time"))
        threads >= 1 ||
            throw(ArgumentError("threads must be a positive Int (`nothing` requests automatic resolution in `Limits(; ...)`), got $threads"))
        return new(iterations, time, threads)
    end
end

function Limits(;
    iterations::Union{Nothing,Integer}=nothing,
    time::Union{Nothing,Real}=nothing,
    threads::Union{Nothing,Integer}=nothing,
)
    resolved_iterations = iterations === nothing ? 0 : Int(iterations)
    resolved_time = time === nothing ? Inf : Float64(time)
    resolved_threads = threads === nothing ? Base.Threads.nthreads() : Int(threads)
    return Limits(resolved_iterations, resolved_time, resolved_threads)
end

"""
    Settings{T<:AbstractFloat}

Typed public solve settings in arithmetic `T`.  Every field is concrete;
there is intentionally no `Any` field and no catch-all keyword sink, so
orientation / `:dual` and every other unknown keyword fail loudly by
absence.

Fields
- `tolerances::Tolerances{T}` — primal/dual/gap stopping targets.
- `limits::Limits` — iterations, wall time, per-solve threads.
- `engine::Symbol` — solver engine selector: `:auto` (the default) or
  `:native_hsd`.  Native product HSD is the only public engine; the
  historical `:legacy` selector is deprecated and rejected at construction
  with a migration error (there is no hidden legacy fallback).  An explicit
  `:native_hsd` request is fail-closed and is never retried through a
  family lowerer, PSD lift, or legacy solver.
- `scaling::Symbol` — compatibility storage: `:auto`, `:none`, or `:equilibrate`.
- `equilibration` — public policy view/keyword: `:off` (default) or `:ruiz`;
  `:ruiz` is stored as `scaling=:equilibrate` without a duplicate field.
- `formulation::Symbol` — numerical formulation selector.  Public names
  are `:auto`, `:variable_space_schur` (variable-space Schur complement;
  lowers to the existing `:normal_equations` policy) and
  `:dense_augmented_kkt` (dense augmented KKT; lowers to the existing
  `:augmented` policy).  The historical `:primal` and any dual-model
  `:dual` are deliberately not part of the public surface and are
  rejected.  `:sparse_normal_equations` is also not exposed: the existing
  policy space has no exact sparse-formulation name (sparse storage is
  selected separately by the `sparse` field).  An explicit
  `formulation=:psd_lift` is accepted for mixed symmetric products to
  force the universal PSD-lift route (the plan's explicit, non-default
  fallback; the default `:auto` keeps the family lowerer's mixed behavior).
- `provider::Symbol` — `:auto` / `:standard` / `:bfla` / `:multifloat` /
  `:legacy` dense linear-algebra provider policy (documented metadata
  carried to `SolveOptions.linear_algebra_backend`; this file never
  instantiates or probes a provider).
- `presolve::Symbol` — structural presolve: `:auto`, `:on`, or `:off`.
- `algorithm::Symbol` — read-only diagnostic label; `:auto` is the only
  accepted value.  Algorithm-family selection no longer exists because every
  public solve executes the native product-HSD engine; the historical family
  selectors `:lp`, `:socp`, and `:sdp` are deprecated and rejected at
  construction with a migration error.
- `sparse::Symbol` — sparse storage preference `:auto`, `:on`, or `:off`.
- `equality_solver::Symbol` — `:auto` / `:normal_equations` / `:qr`.
- `working_precision_policy::Symbol` — `:auto` / `:fixed`.
- `diagnostics::Symbol` — retention level `:none`, `:summary`, or
  `:full` for planning/diagnostic payloads.
- `verbosity::Int` — nonnegative output level.
- `timing::Bool` — record frontend/solver phase timings.
- `certification::Bool` — run post-solve certification.
- `blas_threads::Union{Nothing,Int}` — *requested* BLAS/LAPACK thread
  metadata only.  This layer never calls `set_blas_threads!`; an
  integrator may apply the value through the existing
  `SDPX.set_blas_threads!` seam without this file mutating global state.

Construct with `Settings{Float64}()` / `Settings{Float64}(...)`, or
`Settings(model; ...)` where the model is any object whose element type
selects the arithmetic (`SDPProblem`, `ConicProblem`, `Model`).
"""
struct Settings{T<:AbstractFloat}
    tolerances::Tolerances{T}
    limits::Limits
    engine::Symbol
    scaling::Symbol
    formulation::Symbol
    kkt_route::Symbol
    provider::Symbol
    presolve::Symbol
    algorithm::Symbol
    sparse::Symbol
    equality_solver::Symbol
    working_precision_policy::Symbol
    diagnostics::Symbol
    verbosity::Int
    timing::Bool
    certification::Bool
    blas_threads::Union{Nothing,Int}
    # Additive native product-HSD iteration controls.  `nothing` delegates
    # exactly to the historical numeric path; predictor=:classic is the
    # historical predictor policy.
    iteration_knobs::NamedTuple

    function Settings{T}(
        tolerances::Tolerances{T},
        limits::Limits,
        engine::Symbol,
        scaling::Symbol,
        formulation::Symbol,
        kkt_route::Symbol,
        provider::Symbol,
        presolve::Symbol,
        algorithm::Symbol,
        sparse::Symbol,
        equality_solver::Symbol,
        working_precision_policy::Symbol,
        diagnostics::Symbol,
        verbosity::Int,
        timing::Bool,
        certification::Bool,
        blas_threads::Union{Nothing,Int},
        iteration_knobs::NamedTuple,
    ) where {T<:AbstractFloat}
        _validate_engine(engine)
        _validate_symbol(scaling, (:auto, :none, :equilibrate), "scaling")
        _validate_symbol(
            formulation,
            (:auto, :variable_space_schur, :dense_augmented_kkt, :psd_lift),
            "formulation",
        )
        _validate_symbol(kkt_route, (:bordered, :expanded, :sparse_schur), "kkt_route")
        _validate_symbol(provider, (:auto, :standard, :bfla, :multifloat, :legacy), "provider")
        _validate_algorithm(algorithm)
        _validate_symbol(presolve, (:auto, :on, :off), "presolve")
        _validate_symbol(sparse, (:auto, :on, :off), "sparse")
        _validate_symbol(equality_solver, (:auto, :normal_equations, :qr), "equality_solver")
        _validate_symbol(working_precision_policy, (:auto, :fixed), "working_precision_policy")
        _validate_symbol(diagnostics, (:none, :summary, :full), "diagnostics")
        iteration_knobs isa NamedTuple || throw(ArgumentError(
            "iteration_knobs must be a NamedTuple",
        ))
        verbosity >= 0 ||
            throw(ArgumentError("verbosity must be nonnegative, got $verbosity"))
        blas_threads === nothing || blas_threads >= 1 ||
            throw(ArgumentError("blas_threads must be nothing or at least 1, got $blas_threads"))
        return new{T}(
            tolerances,
            limits,
            engine,
            scaling,
            formulation,
            kkt_route,
            provider,
            presolve,
            algorithm,
            sparse,
            equality_solver,
            working_precision_policy,
            diagnostics,
            verbosity,
            timing,
            certification,
            blas_threads,
            iteration_knobs,
        )
    end
end

function _validate_symbol(value::Symbol, allowed::Tuple, label::AbstractString)
    value in allowed ||
        throw(ArgumentError("$label must be one of $allowed, got $(repr(value))"))
    return nothing
end

# ---------------------------------------------------------------------------
# Phase-9 public API simplification: engine and algorithm selectors.
#
# Native product HSD is the only public engine.  `engine=:legacy` and the
# algorithm-family selectors (`algorithm=:lp/:socp/:sdp`) are removed from
# the user-facing surface: they are rejected with explicit migration errors
# instead of being silently remapped, so no hidden fallback can be reached
# through a deprecated spelling.  The `algorithm` field is retained as a
# read-only diagnostic label (always `:auto`); `kkt_route` and `provider`
# remain expert policy fields and are unchanged.
# ---------------------------------------------------------------------------

const _ENGINE_VALUES = (:auto, :native_hsd)
const _ALGORITHM_VALUES = (:auto,)
const _DEPRECATED_ENGINE_VALUES = (:legacy,)
const _DEPRECATED_ALGORITHM_VALUES = (:lp, :socp, :sdp)

function _validate_engine(value::Symbol)
    value in _DEPRECATED_ENGINE_VALUES && throw(ArgumentError(
        "engine=$(repr(value)) is deprecated and no longer selectable: " *
        "native product HSD is the only public engine.  Use " *
        "engine=:auto or engine=:native_hsd.  There is no hidden legacy " *
        "fallback; legacy execution engines are not reachable through " *
        "the public Settings surface.",
    ))
    _validate_symbol(value, _ENGINE_VALUES, "engine")
    return nothing
end

function _validate_algorithm(value::Symbol)
    value in _DEPRECATED_ALGORITHM_VALUES && throw(ArgumentError(
        "algorithm=$(repr(value)) is deprecated: algorithm-family selection " *
        "no longer exists because every public solve executes the native " *
        "product-HSD engine.  Use algorithm=:auto (the only accepted " *
        "value); the field is retained as a read-only diagnostic label.",
    ))
    _validate_symbol(value, _ALGORITHM_VALUES, "algorithm")
    return nothing
end

const _ITERATION_KNOB_NAMES = (:sigma, :beta, :gamma, :predictor)
const _DEFAULT_ITERATION_KNOBS = (
    sigma=nothing, beta=nothing, gamma=nothing, predictor=:classic,
)

function _normalize_iteration_knobs(::Type{T}, knobs::NamedTuple) where {T<:AbstractFloat}
    unknown = setdiff(propertynames(knobs), _ITERATION_KNOB_NAMES)
    isempty(unknown) || throw(ArgumentError(
        "iteration_knobs has unsupported fields $(Tuple(unknown)); " *
        "supported fields are $(_ITERATION_KNOB_NAMES)",
    ))
    sigma = get(knobs, :sigma, nothing)
    beta = get(knobs, :beta, nothing)
    gamma = get(knobs, :gamma, nothing)
    predictor = get(knobs, :predictor, :classic)
    for (name, value) in ((:sigma, sigma), (:beta, beta), (:gamma, gamma))
        value === nothing && continue
        value isa Real || throw(ArgumentError(
            "iteration_knobs.$name must be nothing or a real number",
        ))
        valueT = try T(value) catch; throw(ArgumentError(
            "iteration_knobs.$name is not representable in $(T)")) end
        isfinite(valueT) || throw(ArgumentError(
            "iteration_knobs.$name must be finite"))
        lower_ok = name === :sigma ? valueT >= zero(T) : valueT > zero(T)
        upper_ok = name === :sigma ? valueT <= one(T) : valueT < one(T)
        lower_ok && upper_ok || throw(ArgumentError(
            "iteration_knobs.$name must lie in the valid open/closed unit interval"))
        if name === :sigma; sigma = valueT
        elseif name === :beta; beta = valueT
        else; gamma = valueT end
    end
    predictor in (:classic, :sdpb) || throw(ArgumentError(
        "iteration_knobs.predictor must be :classic or :sdpb",
    ))
    return (sigma=sigma, beta=beta, gamma=gamma, predictor=predictor)
end

function Settings(
    ::Type{T};
    tolerances::Tolerances{T}=Tolerances{T}(),
    limits::Limits=Limits(),
    engine::Symbol=:auto,
    scaling::Symbol=:auto,
    equilibration::Symbol=:off,
    formulation::Symbol=:auto,
    kkt_route::Symbol=:bordered,
    provider::Symbol=:auto,
    presolve::Symbol=:auto,
    algorithm::Symbol=:auto,
    sparse::Symbol=:auto,
    equality_solver::Symbol=:auto,
    working_precision_policy::Symbol=:auto,
    diagnostics::Symbol=:summary,
    verbosity::Int=1,
    timing::Bool=true,
    certification::Bool=true,
    blas_threads::Union{Nothing,Int}=nothing,
    iteration_knobs::NamedTuple=(;
        sigma=nothing, beta=nothing, gamma=nothing, predictor=:classic,
    ),
) where {T<:AbstractFloat}
    _validate_symbol(equilibration, (:off, :ruiz), "equilibration")
    equilibration === :ruiz && !(scaling in (:auto, :equilibrate)) &&
        throw(ArgumentError("equilibration=:ruiz conflicts with scaling=$scaling"))
    effective_scaling = equilibration === :ruiz ? :equilibrate : scaling
    iteration_knobs = _normalize_iteration_knobs(T, iteration_knobs)
    return Settings{T}(
        tolerances,
        limits,
        engine,
        effective_scaling,
        formulation,
        kkt_route,
        provider,
        presolve,
        algorithm,
        sparse,
        equality_solver,
        working_precision_policy,
        diagnostics,
        verbosity,
        timing,
        certification,
        blas_threads,
        iteration_knobs,
    )
end

Settings{T}(; kwargs...) where {T<:AbstractFloat} = Settings(T; kwargs...)

# `scaling` predates the prepared Phase-4 map and remains ABI-compatible.
# The narrow `equilibration` view gives the native route one unambiguous public
# spelling without adding a duplicate stored policy field.
@inline function Base.getproperty(settings::Settings, name::Symbol)
    if name === :equilibration
        return getfield(settings, :scaling) === :equilibrate ? :ruiz : :off
    end
    return getfield(settings, name)
end
@inline function Base.propertynames(settings::Settings, private::Bool=false)
    return (fieldnames(typeof(settings))..., :equilibration)
end

"""
    Settings(model; kwargs...)

Construct typed settings from any model whose `eltype` is
`AbstractFloat` (`SDPProblem`, `ConicProblem`, or the v0.5 `Model`).
"""
function Settings(model; kwargs...)
    T = eltype(model)
    T <: AbstractFloat ||
        throw(ArgumentError("Settings(model) requires a model with AbstractFloat eltype, got $T"))
    return Settings(T; kwargs...)
end

"""
    SolveOptions(settings::Settings{T}) where {T}

Exact deterministic one-to-one lowering of the typed public policy to the
existing all-auto frontend options.  No policy is invented and no
existing policy name is reused with a different meaning; the mapping is
documented per field and is the only place the typed layer talks to
`SolveOptions`.

Mapping (typed -> existing):
- `settings.tolerances.primal`   -> `primal_error_threshold`
- `settings.tolerances.dual`     -> `dual_error_threshold`
- `settings.tolerances.gap`      -> `duality_gap_threshold`
- `settings.limits.iterations`   -> `maximum_iterations`
- `settings.limits.time`         -> `max_runtime`
- `settings.limits.threads`      -> `threads`
- `settings.scaling`             -> `scaling`
- `settings.formulation`         -> `formulation` (mapped:
  `:variable_space_schur` -> `:normal_equations`,
  `:dense_augmented_kkt` -> `:augmented`, `:auto` -> `:auto`)
- `settings.provider`            -> `linear_algebra_backend`
- `settings.presolve`            -> `presolve`
- `settings.algorithm`           -> `algorithm`
- `settings.sparse`              -> `sparse`
- `settings.equality_solver`     -> `equality_solver`
- `settings.working_precision_policy` -> `working_precision_policy`
- `settings.diagnostics`         -> `diagnostics`
- `settings.verbosity`           -> `verbosity`
- `settings.timing`              -> `timing`
- `settings.certification`       -> `certification`

`settings.engine` is consumed by the public execution router and is
intentionally not lowered into the legacy `SolveOptions` policy object.
`settings.blas_threads` is request metadata and is intentionally NOT part
of `SolveOptions`; the resolver surfaces it only in the summary NamedTuple
so integration code can apply it through the existing BLAS seam.
"""
function Base.convert(::Type{SolveOptions}, settings::Settings{T}) where {T<:AbstractFloat}
    return SolveOptions(
        precision=:auto,
        duality_gap_threshold=settings.tolerances.gap === nothing ? :auto : settings.tolerances.gap,
        primal_error_threshold=settings.tolerances.primal === nothing ? :auto : settings.tolerances.primal,
        dual_error_threshold=settings.tolerances.dual === nothing ? :auto : settings.tolerances.dual,
        maximum_iterations=settings.limits.iterations == 0 ? :auto : settings.limits.iterations,
        max_runtime=settings.limits.time,
        threads=settings.limits.threads == 0 ? :auto : settings.limits.threads,
        verbosity=settings.verbosity,
        presolve=settings.presolve,
        scaling=settings.scaling,
        algorithm=settings.algorithm,
        sparse=settings.sparse,
        formulation=_map_formulation(settings.formulation),
        equality_solver=settings.equality_solver,
        linear_algebra_backend=settings.provider,
        working_precision_policy=settings.working_precision_policy,
        diagnostics=_map_diagnostics(settings.diagnostics),
        timing=settings.timing,
        certification=settings.certification,
    )
end

SolveOptions(settings::Settings{T}) where {T<:AbstractFloat} = convert(SolveOptions, settings)

function _map_formulation(value::Symbol)
    value === :auto && return :auto
    value === :variable_space_schur && return :normal_equations
    value === :dense_augmented_kkt && return :augmented
    # `:psd_lift` is a public *routing* request (universal PSD lift for mixed
    # symmetric), not a numerical KKT formulation; it selects the numerical
    # formulation automatically.
    value === :psd_lift && return :auto
    throw(ArgumentError("unknown formulation $(repr(value))"))
end

function _map_diagnostics(value::Symbol)
    value === :none && return false
    value === :summary && return true
    value === :full && return true
    throw(ArgumentError("unknown diagnostics level $(repr(value))"))
end

"""
    resolve_solve_options(::Type{T}, settings::Settings{T}) -> ResolvedSolveOptions{T}

Workflow-preserving overload: typed settings lower through the SAME
`SolveOptions` mapping so the current deterministic resolver remains the
single authoritative lowerer.  This overload is re-exported in the
existing `public_api.jl` style (`SDPX.Settings`/`SDPX.Outputs` may be
imported qualified) and carries `settings.blas_threads` as additional
summary metadata without adding a field to `SolverOptions`.
"""
function resolve_solve_options(::Type{T}, settings::Settings{T}) where {T<:AbstractFloat}
    resolved = resolve_solve_options(T, SolveOptions(settings))
    summary = merge(
        resolved.summary,
        (blas_threads=settings.blas_threads, kkt_route=settings.kkt_route,),
    )
    return ResolvedSolveOptions{T}(resolved.core, resolved.certification, summary)
end

function Base.show(io::IO, settings::Settings{T}) where {T}
    print(
        io,
        "Settings{", T, "}(",
        "engine=", settings.engine,
        ", ",
        "formulation=", settings.formulation,
        ", kkt_route=", settings.kkt_route,
        ", provider=", settings.provider,
        ", blas_threads=", settings.blas_threads,
        ")",
    )
end
