# ---------------------------------------------------------------------------
# KKT storage is an execution dimension independent of formulation and LA
# provider.  The small descriptors below are intentionally immutable and carry
# only structural facts; numeric factors live in the sparse execution layer.
# ---------------------------------------------------------------------------

abstract type AbstractKKTStorage end

"""Dense KKT storage marker used by the planner and diagnostics."""
struct DenseKKTStorage <: AbstractKKTStorage end

"""Frozen CSC storage marker used by sparse KKT execution."""
struct SparseCSCStorage <: AbstractKKTStorage end

"""
    KKTStoragePlan

Structural storage choice, deliberately separate from `FormulationPlan` and
`LABackendConfiguration`.  `dimension`/`input_nnz` are optional estimates used
by diagnostics and the simple `:auto` policy; no numeric factorization is run
while constructing the plan.
"""
struct KKTStoragePlan
    storage::Symbol
    dimension::Int
    input_nnz::Int
    density::Float64
    reason::Symbol
    provenance::Symbol
    requested::Symbol
end

function KKTStoragePlan(
    storage::Symbol;
    dimension::Integer=0,
    input_nnz::Integer=0,
    density::Real=0.0,
    reason::Symbol=:explicit,
    provenance::Symbol=:storage_planner,
    requested::Symbol=storage,
)
    storage in (:dense, :sparse) || throw(ArgumentError(
        "KKT storage must be :dense or :sparse",
    ))
    dim = Int(dimension)
    nnz_value = Int(input_nnz)
    dim >= 0 || throw(ArgumentError("KKT storage dimension must be nonnegative"))
    nnz_value >= 0 || throw(ArgumentError("KKT storage nnz must be nonnegative"))
    return KKTStoragePlan(
        storage,
        dim,
        nnz_value,
        Float64(density),
        reason,
        provenance,
        requested,
    )
end

KKTStoragePlan(storage::Symbol, dimension::Integer, input_nnz::Integer) =
    KKTStoragePlan(storage; dimension=dimension, input_nnz=input_nnz)
KKTStoragePlan(::DenseKKTStorage; kwargs...) = KKTStoragePlan(:dense; kwargs...)
KKTStoragePlan(::SparseCSCStorage; kwargs...) = KKTStoragePlan(:sparse; kwargs...)
storage_symbol(::DenseKKTStorage) = :dense
storage_symbol(::SparseCSCStorage) = :sparse
storage_symbol(plan::KKTStoragePlan) = plan.storage

@inline function _normalize_kkt_storage_request(value)
    value isa Bool && return value ? :sparse : :dense
    value === :on && return :sparse
    value === :off && return :dense
    value in (:auto, :dense, :sparse) || throw(ArgumentError(
        "sparse/storage policy must be :auto, :dense, :sparse, :on, :off, or Bool",
    ))
    return value
end

"""Typed mathematical KKT formulation, independent of its implementation."""
abstract type AbstractKKTFormulation end

struct DenseNormalEquations <: AbstractKKTFormulation end
"""Explicit dense symmetric-indefinite Newton system, factored by pivoted LDLT."""
struct DenseAugmentedKKT <: AbstractKKTFormulation end
struct SparseNormalEquations <: AbstractKKTFormulation end
struct BlockArrowElimination <: AbstractKKTFormulation end
struct NoKKTFormulation <: AbstractKKTFormulation end

"""Compatibility marker retained only so malformed old plans fail in setup."""
struct UnsupportedKKTFormulation <: AbstractKKTFormulation
    name::Symbol
end

"""
    FormulationPlan

Planner-owned mathematical formulation. `reason` explains the structural
choice and `provenance` names the planner layer that made it. LA provider and
factorization capabilities are deliberately absent.
"""
struct FormulationPlan{F<:AbstractKKTFormulation}
    formulation::F
    reason::Symbol
    provenance::Symbol
end

formulation_symbol(::DenseNormalEquations) = :dense_normal_equations
formulation_symbol(::DenseAugmentedKKT) = :dense_augmented_kkt
formulation_symbol(::SparseNormalEquations) = :sparse_normal_equations
formulation_symbol(::BlockArrowElimination) = :block_arrow
formulation_symbol(::NoKKTFormulation) = :not_applicable
formulation_symbol(formulation::UnsupportedKKTFormulation) = formulation.name
formulation_symbol(plan::FormulationPlan) =
    formulation_symbol(plan.formulation)

function FormulationPlan(
    formulation::Symbol;
    reason::Symbol=:compatibility,
    provenance::Symbol=:compatibility,
)
    typed = formulation === :dense_normal_equations ?
            DenseNormalEquations() :
            formulation === :dense_augmented_kkt ?
            DenseAugmentedKKT() :
            formulation === :sparse_normal_equations ?
            SparseNormalEquations() :
            formulation === :block_arrow ?
            BlockArrowElimination() :
            formulation === :not_applicable ?
            NoKKTFormulation() : UnsupportedKKTFormulation(formulation)
    return FormulationPlan(typed, reason, provenance)
end

"""Map a typed formulation to its current implementation after planning."""
function kkt_backend_from_formulation(
    plan::FormulationPlan,
    algorithm::Symbol,
    equalities::Integer,
)
    formulation = plan.formulation
    sdp_algorithms = (:sdp_primal_dual,)
    if formulation isa Union{
        DenseNormalEquations,
        DenseAugmentedKKT,
        SparseNormalEquations,
        BlockArrowElimination,
    }
        algorithm in sdp_algorithms || throw(ArgumentError(
            "SDP KKT formulation $(formulation_symbol(plan)) is incompatible " *
            "with algorithm $(algorithm)",
        ))
    end
    formulation isa DenseNormalEquations && return :dense_cholesky
    formulation isa DenseAugmentedKKT && return :dense_augmented_ldlt
    formulation isa SparseNormalEquations && return :sparse_schur_cholesky
    formulation isa BlockArrowElimination && return :block_arrow
    if formulation isa NoKKTFormulation
        algorithm === :lp_primal_dual &&
            return equalities == 0 ?
                   :positive_definite_cholesky : :dense_lu
    end
    throw(ArgumentError(
        "KKT formulation $(formulation_symbol(plan)) does not implement " *
        "algorithm $(algorithm)",
    ))
end

"""Compatibility equality for backend aliases implementing one formulation."""
function kkt_backend_matches_formulation(
    backend::Symbol,
    plan::FormulationPlan,
    algorithm::Symbol,
    equalities::Integer,
)
    planned = kkt_backend_from_formulation(plan, algorithm, equalities)
    backend === planned && return true
    return planned === :dense_cholesky &&
           backend === :dense_cholesky_fallback
end

"""
    kkt_formulation_from_backend(kkt_backend) -> Symbol

Stable route mapping used by compatibility positional `ExecutionPlan`
constructors and by Workspace validation.  Dense Cholesky and its historical
fallback label execute the same dense normal-equation route, so both map to
`:dense_normal_equations`.  Deferred LP and native Q3 plans have no SDP KKT
formulation and map to `:not_applicable`.
"""
function kkt_formulation_from_backend(kkt_backend::Symbol)
    kkt_backend === :block_arrow && return :block_arrow
    kkt_backend === :sparse_schur_cholesky && return :sparse_normal_equations
    kkt_backend in (:dense_cholesky, :dense_cholesky_fallback) &&
        return :dense_normal_equations
    kkt_backend === :dense_augmented_ldlt && return :dense_augmented_kkt
    return :not_applicable
end

"""Compatibility-only inverse mapping for historical positional plans."""
formulation_plan_from_backend(kkt_backend::Symbol) = FormulationPlan(
    kkt_formulation_from_backend(kkt_backend);
    reason=:backend_compatibility,
    provenance=:compatibility_constructor,
)

"""
    AbstractExecutionPlanPayload

Optional solver-family payload carried by one top-level `ExecutionPlan`.
Subtypes (for example `NativeSOCPlan`) freeze family-specific planning inside
the single authoritative execution-plan object.  The abstract type lives in
`types.jl` so `ExecutionPlan` can hold the field without any include-order
dependency on the family modules that define concrete payload types.
"""
abstract type AbstractExecutionPlanPayload end

"""
    LPRoutePlan

Finalized route payload for the dedicated LP path, carried by
`ExecutionPlan.payload` on the single post-presolve LP execution plan.

`route` is one of `:diagonal_reduced_cholesky`, `:positive_definite_cholesky`,
`:dense_lu`, or `:sparse_normal`. The dense and diagonal routes always report
`:dense` storage with their dense provider. `storage` is `:dense` or
`:sparse`; `provider` is the executed provider. `sparse_probe_count` counts
the measured-pattern sparse probes performed while finalizing this route
(exactly zero for routes decided structurally, and exactly one for an
auto-sparse candidate that was probed and rejected or accepted).

The payload is finalized exactly once, after LP row presolve and `_scale_lp!`
have settled `G`/`B`, and is frozen for the whole solve: workspace and backend
construction assert parity with it instead of re-planning.
"""
struct LPRoutePlan <: AbstractExecutionPlanPayload
    route::Symbol
    storage::Symbol
    provider::Symbol
    sparse_probe_count::Int
    variables::Int
    equalities::Int
    inequalities::Int
end

"""
    ExecutionPlan

Algorithms selected before a solve. This is deliberately descriptive: it is
returned to callers in diagnostics so automatic decisions are inspectable and
reproducible.
"""
struct ExecutionPlan{F<:AbstractKKTFormulation}
    classification::ProblemClassification
    algorithm::Symbol
    scaling::Symbol
    kkt_backend::Symbol
    backend_config::BackendConfiguration
    formulation_plan::FormulationPlan{F}
    la_config::LABackendConfiguration
    gram_kernel::Symbol
    schedule::Symbol
    threads::Int
    parameter_profile::Symbol
    memory_budget_bytes::Int
    parameters::NamedTuple
    # Optional solver-family payload. `nothing` for plans that describe a
    # generic SDP/LP route; solver families that need a typed specialization
    # (NativeSOC) carry their exact plan here so the ExecutionPlan remains the
    # sole top-level planning authority.
    payload::Union{Nothing,AbstractExecutionPlanPayload}

    function ExecutionPlan(
        classification::ProblemClassification,
        algorithm::Symbol,
        scaling::Symbol,
        kkt_backend::Symbol,
        backend_config::BackendConfiguration,
        formulation_plan::FormulationPlan{F},
        la_config::LABackendConfiguration,
        gram_kernel::Symbol,
        schedule::Symbol,
        threads::Int,
        parameter_profile::Symbol,
        memory_budget_bytes::Int,
        parameters::NamedTuple,
        payload::Union{Nothing,AbstractExecutionPlanPayload},
    ) where {F<:AbstractKKTFormulation}
        return new{F}(
            classification,
            algorithm,
            scaling,
            kkt_backend,
            backend_config,
            formulation_plan,
            la_config,
            gram_kernel,
            schedule,
            threads,
            parameter_profile,
            memory_budget_bytes,
            parameters,
            payload,
        )
    end
end


function Base.getproperty(plan::ExecutionPlan, name::Symbol)
    name === :kkt_formulation &&
        return formulation_symbol(getfield(plan, :formulation_plan))
    if name === :storage_plan
        classification = getfield(plan, :classification)
        parameters = getfield(plan, :parameters)
        requested = hasproperty(parameters, :storage_policy) ?
                    parameters.storage_policy : :auto
        requested = _normalize_kkt_storage_request(requested)
        selected = hasproperty(parameters, :storage_selected) ?
                   parameters.storage_selected : classification.storage
        selected in (:dense, :sparse) || (selected = classification.storage)
        dimension = hasproperty(parameters, :storage_dimension) ?
                    parameters.storage_dimension : classification.variables
        input_nnz = hasproperty(parameters, :storage_input_nnz) ?
                    parameters.storage_input_nnz : 0
        density = hasproperty(parameters, :storage_density) ?
                  parameters.storage_density : classification.expected_schur_density
        reason = hasproperty(parameters, :storage_reason) ?
                 parameters.storage_reason : :route_storage_policy
        return KKTStoragePlan(
            selected;
            dimension=dimension,
            input_nnz=input_nnz,
            density=density,
            reason=reason,
            provenance=:execution_plan,
            requested=requested,
        )
    end
    return getfield(plan, name)
end

function Base.propertynames(plan::ExecutionPlan, private::Bool=false)
    names = fieldnames(typeof(plan))
    return (names..., :kkt_formulation, :storage_plan)
end

"""Immutable copy of `plan` carrying the given solver-family payload.
The mathematical formulation, backend, and parameter fields are copied
verbatim; only the payload slot is replaced."""
function ExecutionPlan(
    plan::ExecutionPlan{F},
    payload::Union{Nothing,AbstractExecutionPlanPayload},
) where {F<:AbstractKKTFormulation}
    return ExecutionPlan(
        plan.classification,
        plan.algorithm,
        plan.scaling,
        plan.kkt_backend,
        plan.backend_config,
        plan.formulation_plan,
        plan.la_config,
        plan.gram_kernel,
        plan.schedule,
        plan.threads,
        plan.parameter_profile,
        plan.memory_budget_bytes,
        plan.parameters,
        payload,
    )
end

# Source compatibility for the modern typed 13-field constructor: planners
# that do not need a solver-family payload keep working with `payload=nothing`.
function ExecutionPlan(
    classification::ProblemClassification,
    algorithm::Symbol,
    scaling::Symbol,
    kkt_backend::Symbol,
    backend_config::BackendConfiguration,
    formulation_plan::FormulationPlan{F},
    la_config::LABackendConfiguration,
    gram_kernel::Symbol,
    schedule::Symbol,
    threads::Int,
    parameter_profile::Symbol,
    memory_budget_bytes::Int,
    parameters::NamedTuple,
) where {F<:AbstractKKTFormulation}
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
        backend_config,
        formulation_plan,
        la_config,
        gram_kernel,
        schedule,
        threads,
        parameter_profile,
        memory_budget_bytes,
        parameters,
        nothing,
    )
end

# Source compatibility for the former 13-field constructor. Modern planner
# code passes a typed FormulationPlan directly; historical callers may still
# pass the formulation Symbol in the same position.
function ExecutionPlan(
    classification::ProblemClassification,
    algorithm::Symbol,
    scaling::Symbol,
    kkt_backend::Symbol,
    backend_config::BackendConfiguration,
    kkt_formulation::Symbol,
    la_config::LABackendConfiguration,
    gram_kernel::Symbol,
    schedule::Symbol,
    threads::Int,
    parameter_profile::Symbol,
    memory_budget_bytes::Int,
    parameters::NamedTuple,
)
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
        backend_config,
        FormulationPlan(
            kkt_formulation;
            reason=:explicit_descriptor,
            provenance=:compatibility_constructor,
        ),
        la_config,
        gram_kernel,
        schedule,
        threads,
        parameter_profile,
        memory_budget_bytes,
        parameters,
    )
end

# Compatibility constructor for the v0.4.1-dev 11-field plan.  New planner
# code supplies `la_config` explicitly; old callers retain the legacy path.
function ExecutionPlan(
    classification::ProblemClassification,
    algorithm::Symbol,
    scaling::Symbol,
    kkt_backend::Symbol,
    backend_config::BackendConfiguration,
    gram_kernel::Symbol,
    schedule::Symbol,
    threads::Int,
    parameter_profile::Symbol,
    memory_budget_bytes::Int,
    parameters::NamedTuple,
)
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
        backend_config,
        formulation_plan_from_backend(kkt_backend),
        _compat_la_backend_configuration(
            classification.arithmetic,
            backend_config.equality_solver,
        ),
        gram_kernel,
        schedule,
        threads,
        parameter_profile,
        memory_budget_bytes,
        parameters,
    )
end

# Compatibility constructor for the v0.4.1-dev plan carrying `la_config` but
# no formulation field.  New planner code supplies the formulation explicitly;
# old callers retain the backend-derived route mapping.
function ExecutionPlan(
    classification::ProblemClassification,
    algorithm::Symbol,
    scaling::Symbol,
    kkt_backend::Symbol,
    backend_config::BackendConfiguration,
    la_config::LABackendConfiguration,
    gram_kernel::Symbol,
    schedule::Symbol,
    threads::Int,
    parameter_profile::Symbol,
    memory_budget_bytes::Int,
    parameters::NamedTuple,
)
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
        backend_config,
        formulation_plan_from_backend(kkt_backend),
        la_config,
        gram_kernel,
        schedule,
        threads,
        parameter_profile,
        memory_budget_bytes,
        parameters,
    )
end

# Source compatibility for the pre-`backend_config` positional form.  The
# compatibility route is intentionally conservative: callers constructing an
# `ExecutionPlan` directly get the same backend named by `kkt_backend`, while
# plans built by `build_execution_plan` carry the complete configuration.
function ExecutionPlan(
    classification::ProblemClassification,
    algorithm::Symbol,
    scaling::Symbol,
    kkt_backend::Symbol,
    gram_kernel::Symbol,
    schedule::Symbol,
    threads::Int,
    parameter_profile::Symbol,
    memory_budget_bytes::Int,
    parameters::NamedTuple,
)
    equality_solver = get(parameters, :equality_solver, :auto)
    deferred = algorithm === :lp_primal_dual
    config = BackendConfiguration(
        kkt_backend,
        equality_solver,
        false,
        false,
        :off,
        (),
        deferred,
    )
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
        config,
        formulation_plan_from_backend(kkt_backend),
        _compat_la_backend_configuration(
            classification.arithmetic,
            equality_solver,
        ),
        gram_kernel,
        schedule,
        threads,
        parameter_profile,
        memory_budget_bytes,
        parameters,
    )
end

