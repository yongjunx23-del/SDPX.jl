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

"""
    AbstractNativeHSDFormulation

Family-specific formulation descriptors for the native homogeneous
self-dual route.  These are deliberately separate from the mature SDP KKT
formulations above: a native HSD system contains the homogeneous border and
its factorization/recovery contract, so reporting it as a normal-equation or
Cholesky route is misleading.
"""
abstract type AbstractNativeHSDFormulation <: AbstractKKTFormulation end

"""
    DenseHomogeneousBordered

Descriptor for the symmetric native-HSD route.  `matrix_dimension` is the
actual full bordered matrix dimension (the reduced variable rank plus the
homogeneous scalar border), or zero when equality reduction is unavailable or
the reduced problem has no product-cone rows.  `row_scaling`, `factorization`,
`pivoting`, and `factorization_reuse` are execution facts, not requests.
"""
struct DenseHomogeneousBordered <: AbstractNativeHSDFormulation
    dimension::Int
    reduced_rank::Int
    active_rows::Int
    layout::Symbol
    row_scaling::Symbol
    border_structure::Symbol
    factorization::Symbol
    pivoting::Symbol
    factor_reuse::Symbol
    gram_or_metric::Symbol
    backend::Symbol
    route::Symbol
    reason::Symbol
    available::Bool
end

"""Typed descriptor for `K = [0 Ar'; Ar -Theta]`."""
struct SymmetricAugmentedHSD <: AbstractNativeHSDFormulation
    dimension::Int
    reduced_rank::Int
    active_rows::Int
    layout::Symbol
    row_scaling::Symbol
    border_structure::Symbol
    factorization::Symbol
    pivoting::Symbol
    factor_reuse::Symbol
    gram_or_metric::Symbol
    backend::Symbol
    route::Symbol
    reason::Symbol
    available::Bool
end

function SymmetricAugmentedHSD(
    reduced_rank::Integer, active_rows::Integer, dimension::Integer;
    storage::Symbol=:sparse,
    reason::Symbol=:ready,
    available::Bool=true,
)
    nr = _native_hsd_nonnegative_dimension(reduced_rank, :reduced_rank)
    rows = _native_hsd_nonnegative_dimension(active_rows, :active_rows)
    dim = _native_hsd_nonnegative_dimension(dimension, :matrix_dimension)
    return SymmetricAugmentedHSD(
        available ? dim : 0,
        available ? nr : 0,
        rows,
        available ? :rank_reduced_product_rows : :not_applicable,
        :none,
        :none,
        available ? :symmetric_ldl : :not_applicable,
        available ? :symmetric_pivoting : :not_applicable,
        available ? :factor_once_homogeneous_predictor_corrector : :not_applicable,
        available ? :native_product_theta : :not_applicable,
        available ? :symmetric_augmented_core : :not_applicable,
        :symmetric_augmented_hsd_core,
        reason,
        available,
    )
end

"""
    DenseHybridCoupled

Descriptor for the future mixed symmetric/nonsymmetric native-HSD route.  The
descriptor records the factor-coordinate contract used by Exp/Power blocks,
while keeping the public policy fail-closed until that route is enabled.
`matrix_dimension` is the full coupled matrix dimension, including the
nonsymmetric rows and the two homogeneous scalar columns/rows.
"""
struct DenseHybridCoupled <: AbstractNativeHSDFormulation
    dimension::Int
    reduced_rank::Int
    active_rows::Int
    layout::Symbol
    symmetric_dimension::Int
    nonsymmetric_dimension::Int
    nonsymmetric_blocks::Int
    row_scaling::Symbol
    border_structure::Symbol
    factorization::Symbol
    pivoting::Symbol
    coordinate_system::Symbol
    factor_reuse::Symbol
    gram_or_metric::Symbol
    backend::Symbol
    route::Symbol
    reason::Symbol
    available::Bool
end

@inline function _native_hsd_nonnegative_dimension(value::Integer, label::Symbol)
    value >= 0 || throw(ArgumentError("native HSD $label must be nonnegative"))
    return Int(value)
end

function DenseHomogeneousBordered(
    reduced_variables::Integer,
    active_rows::Integer;
    dimension::Union{Nothing,Integer}=nothing,
    matrix_dimension::Union{Nothing,Integer}=nothing,
    layout::Symbol=:equality_reduced,
    row_scaling::Symbol=:exact_binary_row_scaling,
    border_structure::Symbol=:full_homogeneous_border,
    factorization::Symbol=:lu,
    pivoting::Symbol=:partial,
    factor_reuse::Symbol=:factor_once_predictor_corrector_refinement,
    gram_or_metric::Symbol=:native_product_metric,
    backend::Symbol=:native_hsd_binary_row_scaled_border,
    route::Symbol=:dense_homogeneous_bordered,
    reason::Symbol=:ready,
    available::Bool=true,
)
    nr = _native_hsd_nonnegative_dimension(reduced_variables, :reduced_variables)
    rows = _native_hsd_nonnegative_dimension(active_rows, :active_rows)
    matrix_dim = dimension === nothing ?
                 (matrix_dimension === nothing ?
                  (active_rows > 0 && available ? reduced_variables + 1 : 0) :
                  matrix_dimension) :
                 dimension
    dimension !== nothing && matrix_dimension !== nothing &&
        Int(dimension) == Int(matrix_dimension) ||
        (dimension === nothing || matrix_dimension === nothing) ||
        throw(ArgumentError("native HSD dimension and matrix_dimension disagree"))
    matrix_dim = _native_hsd_nonnegative_dimension(matrix_dim, :matrix_dimension)
    if !available
        matrix_dim = 0
        row_scaling = :none
        border_structure = :none
        factorization = :not_applicable
        pivoting = :not_applicable
        factor_reuse = :not_applicable
        gram_or_metric = :not_applicable
        backend = :not_applicable
    end
    return DenseHomogeneousBordered(
        matrix_dim, nr, rows, layout, row_scaling, border_structure,
        factorization, pivoting, factor_reuse, gram_or_metric, backend, route,
        reason, available,
    )
end

function DenseHomogeneousBordered()
    return DenseHomogeneousBordered(
        0,
        0;
        matrix_dimension=0,
        layout=:not_applicable,
        row_scaling=:none,
        border_structure=:none,
        factorization=:not_applicable,
        pivoting=:not_applicable,
        factor_reuse=:not_applicable,
        gram_or_metric=:not_applicable,
        backend=:not_applicable,
        route=:dense_homogeneous_bordered,
        reason=:not_applicable,
        available=false,
    )
end

function DenseHybridCoupled(
    reduced_variables::Integer,
    active_rows::Integer;
    dimension::Union{Nothing,Integer}=nothing,
    matrix_dimension::Union{Nothing,Integer}=nothing,
    layout::Symbol=:equality_reduced,
    symmetric_dimension::Integer=0,
    nonsymmetric_dimension::Integer=0,
    nonsymmetric_blocks::Integer=0,
    row_scaling::Symbol=:nonsymmetric_factor_coordinates,
    border_structure::Symbol=:full_homogeneous_border,
    factorization::Symbol=:lu,
    pivoting::Symbol=:partial,
    coordinate_system::Symbol=:factor_coordinate,
    factor_reuse::Symbol=:factor_once_predictor_corrector_refinement,
    gram_or_metric::Symbol=:hybrid_factor_coordinate_metric,
    backend::Symbol=:native_hsd_factor_coordinate_coupled,
    route::Symbol=:dense_hybrid_coupled,
    reason::Symbol=:ready,
    available::Bool=true,
)
    nr = _native_hsd_nonnegative_dimension(reduced_variables, :reduced_variables)
    rows = _native_hsd_nonnegative_dimension(active_rows, :active_rows)
    matrix_dim = dimension === nothing ?
                 (matrix_dimension === nothing ?
                  (active_rows > 0 && nonsymmetric_dimension > 0 && available ?
                   reduced_variables + nonsymmetric_dimension + 2 : 0) :
                  matrix_dimension) :
                 dimension
    dimension !== nothing && matrix_dimension !== nothing &&
        Int(dimension) == Int(matrix_dimension) ||
        (dimension === nothing || matrix_dimension === nothing) ||
        throw(ArgumentError("native HSD dimension and matrix_dimension disagree"))
    matrix_dim = _native_hsd_nonnegative_dimension(matrix_dim, :matrix_dimension)
    symmetric = _native_hsd_nonnegative_dimension(symmetric_dimension, :symmetric_dimension)
    nonsymmetric = _native_hsd_nonnegative_dimension(
        nonsymmetric_dimension, :nonsymmetric_dimension,
    )
    blocks = _native_hsd_nonnegative_dimension(nonsymmetric_blocks, :nonsymmetric_blocks)
    if !available
        matrix_dim = 0
        row_scaling = :none
        border_structure = :none
        factorization = :not_applicable
        pivoting = :not_applicable
        coordinate_system = :not_applicable
        factor_reuse = :not_applicable
        gram_or_metric = :not_applicable
        backend = :not_applicable
    end
    return DenseHybridCoupled(
        matrix_dim, nr, rows, layout, symmetric, nonsymmetric, blocks,
        row_scaling, border_structure, factorization, pivoting,
        coordinate_system, factor_reuse, gram_or_metric, backend, route,
        reason, available,
    )
end

function DenseHybridCoupled()
    return DenseHybridCoupled(
        0,
        0;
        matrix_dimension=0,
        layout=:not_applicable,
        symmetric_dimension=0,
        nonsymmetric_dimension=0,
        nonsymmetric_blocks=0,
        row_scaling=:none,
        border_structure=:none,
        factorization=:not_applicable,
        pivoting=:not_applicable,
        coordinate_system=:not_applicable,
        factor_reuse=:not_applicable,
        gram_or_metric=:not_applicable,
        backend=:not_applicable,
        route=:dense_hybrid_coupled,
        reason=:not_applicable,
        available=false,
    )
end

function Base.getproperty(
    descriptor::Union{DenseHomogeneousBordered,SymmetricAugmentedHSD},
    name::Symbol,
)
    name === :formulation && return getfield(descriptor, :route)
    name === :matrix_dimension && return getfield(descriptor, :dimension)
    name === :reduced_variables && return getfield(descriptor, :reduced_rank)
    name === :reduced_layout && return getfield(descriptor, :layout)
    name === :border && return getfield(descriptor, :border_structure)
    name === :factorization_reuse && return getfield(descriptor, :factor_reuse)
    name === :transform && return getfield(descriptor, :row_scaling)
    name === :metric && return getfield(descriptor, :gram_or_metric)
    name === :pivoting_strategy && return getfield(descriptor, :pivoting)
    name === :reuse && return getfield(descriptor, :factor_reuse)
    return getfield(descriptor, name)
end

function Base.getproperty(
    descriptor::DenseHybridCoupled,
    name::Symbol,
)
    name === :formulation && return getfield(descriptor, :route)
    name === :matrix_dimension && return getfield(descriptor, :dimension)
    name === :reduced_variables && return getfield(descriptor, :reduced_rank)
    name === :reduced_layout && return getfield(descriptor, :layout)
    name === :border && return getfield(descriptor, :border_structure)
    name === :factorization_reuse && return getfield(descriptor, :factor_reuse)
    name === :transform && return getfield(descriptor, :row_scaling)
    name === :metric && return getfield(descriptor, :gram_or_metric)
    name === :pivoting_strategy && return getfield(descriptor, :pivoting)
    name === :reuse && return getfield(descriptor, :factor_reuse)
    return getfield(descriptor, name)
end

function Base.propertynames(
    descriptor::Union{DenseHomogeneousBordered,DenseHybridCoupled,SymmetricAugmentedHSD},
    private::Bool=false,
)
    return (fieldnames(typeof(descriptor))..., :matrix_dimension,
            :formulation, :reduced_variables, :reduced_layout, :border,
            :factorization_reuse, :transform, :metric, :pivoting_strategy, :reuse)
end

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
formulation_symbol(::DenseHomogeneousBordered) = :dense_homogeneous_bordered
formulation_symbol(::SymmetricAugmentedHSD) = :symmetric_augmented_hsd_core
formulation_symbol(::DenseHybridCoupled) = :dense_hybrid_coupled
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
    formulation isa DenseHomogeneousBordered &&
        return formulation.backend
    formulation isa SymmetricAugmentedHSD &&
        return formulation.backend
    formulation isa DenseHybridCoupled &&
        return formulation.backend
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

Optional typed payload carried by one top-level `ExecutionPlan`. Concrete
payloads freeze route-specific implementation facts inside the single
execution-plan object without changing the product-HSD equations.
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

"""Build a typed storage descriptor for historical plan constructors.

Modern planners pass a `KKTStoragePlan` directly.  This adapter is retained at
the construction boundary only so qualified callers using the former
`parameters.storage_*` convention receive the same plan without making those
loose parameters a second runtime authority.
"""
function _compat_storage_plan(
    classification::ProblemClassification,
    parameters::NamedTuple,
)
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
        dimension,
        input_nnz,
        density,
        reason,
        provenance=:compatibility_constructor,
        requested,
    )
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
    backend_config::BackendConfiguration
    formulation_plan::FormulationPlan{F}
    la_config::LABackendConfiguration
    storage_plan::KKTStoragePlan
    gram_kernel::Symbol
    schedule::Symbol
    threads::Int
    parameter_profile::Symbol
    memory_budget_bytes::Int
    parameters::NamedTuple
    # Optional solver-family payload. `nothing` for plans that describe a
    # generic SDP/LP route; solver families that need a typed specialization
    # Specialized compatibility adapters carry their exact plan here so the ExecutionPlan remains the
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
        storage_plan::KKTStoragePlan,
        gram_kernel::Symbol,
        schedule::Symbol,
        threads::Int,
        parameter_profile::Symbol,
        memory_budget_bytes::Int,
        parameters::NamedTuple,
        payload::Union{Nothing,AbstractExecutionPlanPayload},
    ) where {F<:AbstractKKTFormulation}
        backend_config.route === kkt_backend || throw(ArgumentError(
            "execution plan backend configuration $(backend_config.route) " *
            "does not match compatibility kkt_backend $(kkt_backend)",
        ))
        if payload isa LPRoutePlan
            algorithm === :lp_primal_dual || throw(ArgumentError(
                "LP route payload requires algorithm=:lp_primal_dual",
            ))
            backend_config.deferred && throw(ArgumentError(
                "finalized LP route payload cannot carry a deferred backend",
            ))
            payload.route in (
                :diagonal_reduced_cholesky,
                :positive_definite_cholesky,
                :dense_lu,
                :sparse_normal,
            ) || throw(ArgumentError(
                "unknown finalized LP route $(payload.route)",
            ))
            expected_storage = payload.route === :sparse_normal ?
                               :sparse : :dense
            payload.storage === expected_storage || throw(ArgumentError(
                "LP route $(payload.route) requires $(expected_storage) " *
                "storage, got $(payload.storage)",
            ))
            backend_config.route === payload.route || throw(ArgumentError(
                "execution-plan route $(backend_config.route) does not " *
                "match LP payload route $(payload.route)",
            ))
            storage_plan.storage === payload.storage || throw(ArgumentError(
                "execution-plan storage $(storage_plan.storage) does not " *
                "match LP payload storage $(payload.storage)",
            ))
            if payload.storage === :sparse
                payload.provider in (:cholmod, :generic) || throw(ArgumentError(
                    "unknown sparse LP provider $(payload.provider)",
                ))
                la_config.provider === payload.provider || throw(ArgumentError(
                    "execution-plan LA provider $(la_config.provider) does " *
                    "not match sparse LP provider $(payload.provider)",
                ))
            end
        end
        return new{F}(
            classification,
            algorithm,
            scaling,
            backend_config,
            formulation_plan,
            la_config,
            storage_plan,
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


"""Canonical `ExecutionPlan` constructor with typed route authorities."""
function ExecutionPlan(
    classification::ProblemClassification,
    algorithm::Symbol,
    scaling::Symbol,
    backend_config::BackendConfiguration,
    formulation_plan::FormulationPlan{F},
    la_config::LABackendConfiguration,
    storage_plan::KKTStoragePlan,
    gram_kernel::Symbol,
    schedule::Symbol,
    threads::Int,
    parameter_profile::Symbol,
    memory_budget_bytes::Int,
    parameters::NamedTuple,
    payload::Union{Nothing,AbstractExecutionPlanPayload}=nothing,
) where {F<:AbstractKKTFormulation}
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        backend_config.route,
        backend_config,
        formulation_plan,
        la_config,
        storage_plan,
        gram_kernel,
        schedule,
        threads,
        parameter_profile,
        memory_budget_bytes,
        parameters,
        payload,
    )
end

# Compatibility for the former typed constructor, where storage facts lived
# inside the loose `parameters` NamedTuple.
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
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
        backend_config,
        formulation_plan,
        la_config,
        _compat_storage_plan(classification, parameters),
        gram_kernel,
        schedule,
        threads,
        parameter_profile,
        memory_budget_bytes,
        parameters,
        payload,
    )
end


function Base.getproperty(plan::ExecutionPlan, name::Symbol)
    name === :kkt_backend &&
        return getfield(getfield(plan, :backend_config), :route)
    if name === :kkt_formulation
        payload = getfield(plan, :payload)
        return payload isa LPRoutePlan ?
               payload.route :
               formulation_symbol(getfield(plan, :formulation_plan))
    end
    return getfield(plan, name)
end

function Base.propertynames(plan::ExecutionPlan, private::Bool=false)
    names = fieldnames(typeof(plan))
    return (names..., :kkt_backend, :kkt_formulation)
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
        plan.backend_config,
        plan.formulation_plan,
        plan.la_config,
        plan.storage_plan,
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
