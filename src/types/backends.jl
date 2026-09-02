# ---------------------------------------------------------------------------
# Linear-algebra backend description
#
# The configuration is deliberately non-parametric and immutable.  It is part
# of an ExecutionPlan (and is therefore serialized in diagnostics/checkpoints)
# rather than inferred again by a Workspace.  `provider` is an optional
# symbols supplied by an arithmetic extension; concrete setup payloads live
# only in the instantiated MultiFloatLABackend, never in the plan.
# ---------------------------------------------------------------------------

abstract type AbstractLABackend end
abstract type KKTBackend end

"""
    LAProviderCapabilities

Pure, immutable description of the operations and numerical contracts exposed
by one linear-algebra provider.  These are facts, not preferences: capability
lookup never benchmarks the machine and never installs a runtime fallback.

The semantic fields (`cholesky`, `lu`, `qr`, `factor_solve`, ... ) are the
planner boundary.  The final six fields describe the dense primitives already
used by SDPX's KKT implementation.  A provider may therefore be useful for a
Cholesky route without being advertised as a robust symmetric-indefinite
solver.
"""
Base.@kwdef struct LAProviderCapabilities
    cholesky::Bool = false
    lu::Bool = false
    qr::Bool = false
    rank_revealing_qr::Bool = false
    pivoted_symmetric_ldlt::Bool = false
    ldlt_inertia::Bool = false
    factor_solve::Bool = false
    multi_rhs::Bool = false
    iterative_refinement::Bool = false
    higher_precision_residual::Bool = false
    refinement_correction::Bool = false
    mixed_precision_residual::Bool = false
    sparse_factorization::Bool = false
    threading::Bool = false
    dot::Bool = false
    norminf::Bool = false
    mul::Bool = false
    mul_owned::Bool = false
    syrk::Bool = false
    triangular_solve::Bool = false
    axpby::Bool = false
end

const _LA_CAPABILITY_NAMES = fieldnames(LAProviderCapabilities)

@inline function _canonical_la_capability(capability::Symbol)
    capability in (:chol, :cholesky_factor!) && return :cholesky
    capability in (:solve, :cholsolve_owned) && return :factor_solve
    capability in (:trsm, :trsv_lower, :trsv_transpose) &&
        return :triangular_solve
    capability in (:axpby_owned,) && return :axpby
    return capability
end

"""Return whether `capabilities` satisfies one semantic capability name."""
function la_provider_supports(
    capabilities::LAProviderCapabilities,
    capability::Symbol,
)
    name = _canonical_la_capability(capability)
    name in _LA_CAPABILITY_NAMES || throw(ArgumentError(
        "unknown linear-algebra capability $(capability)",
    ))
    return getfield(capabilities, name)
end

Base.in(capability::Symbol, capabilities::LAProviderCapabilities) =
    la_provider_supports(capabilities, capability)

"""Stable tuple projection used by diagnostics and compatibility adapters."""
la_capability_symbols(capabilities::LAProviderCapabilities) = Tuple(
    name for name in _LA_CAPABILITY_NAMES if getfield(capabilities, name)
)

"""Conservatively translate a historical operation tuple into semantic facts."""
function la_capabilities_from_symbols(symbols::Tuple)
    enabled = Set{Symbol}()
    for symbol in symbols
        symbol isa Symbol || throw(ArgumentError(
            "linear-algebra capability names must be Symbols",
        ))
        push!(enabled, _canonical_la_capability(symbol))
    end
    return LAProviderCapabilities(;
        (name => (name in enabled) for name in _LA_CAPABILITY_NAMES)...,
    )
end

struct StandardLABackend <: AbstractLABackend
    arithmetic::Symbol
    provider::Symbol
    mode::Symbol
end

StandardLABackend(arithmetic::Symbol) = begin
    provider = arithmetic in (:float32, :float64) ? :blas_lapack :
               :generic_linear_algebra
    ownership = arithmetic in (:float32, :float64) ? :immutable_scalars :
                 :owned_mutable_scalars
    StandardLABackend(arithmetic, provider, ownership)
end

struct LegacyLABackend{P} <: AbstractLABackend
    arithmetic::Symbol
    reason::Symbol
    provider::P

    function LegacyLABackend{P}(
        arithmetic::Symbol,
        reason::Symbol,
        provider::P,
    ) where {P}
        return new{P}(arithmetic, reason, provider)
    end
end

struct MultiFloatLABackend{P} <: AbstractLABackend
    arithmetic::Symbol
    provider::P
end

"""Optional native BigFloat provider instantiated by the BFLA extension."""
struct BFLALABackend{P} <: AbstractLABackend
    arithmetic::Symbol
    provider::P
end

struct LABackendConfiguration
    arithmetic::Symbol
    requested::Symbol
    selected::Symbol
    provider::Symbol
    capabilities::Tuple{Vararg{Symbol}}
    capability_model::LAProviderCapabilities
    required_capabilities::Tuple{Vararg{Symbol}}
    provider_implementation::Symbol
    fallback_chain::Tuple{Vararg{Symbol}}
    fallback_reason::Symbol
    ownership::Symbol
end

# Source compatibility for the v0.4.1 eight-field descriptor.  Modern planner
# code supplies an explicit semantic model and requirement set; historical
# callers remain conservative and do not gain new route authority.
function LABackendConfiguration(
    arithmetic::Symbol,
    requested::Symbol,
    selected::Symbol,
    provider::Symbol,
    capabilities::Tuple{Vararg{Symbol}},
    fallback_chain::Tuple{Vararg{Symbol}},
    fallback_reason::Symbol,
    ownership::Symbol,
)
    return LABackendConfiguration(
        arithmetic,
        requested,
        selected,
        provider,
        capabilities,
        la_capabilities_from_symbols(capabilities),
        (),
        provider,
        fallback_chain,
        fallback_reason,
        ownership,
    )
end

"""Conservative configuration used by historical positional plan builders."""
function _compat_la_backend_configuration(
    arithmetic::Symbol,
    equality_solver::Symbol=:auto,
)
    return LABackendConfiguration(
        arithmetic, :legacy, :legacy, :sdpx_legacy_la,
        SDPX_LEGACY_LA_CAPABILITIES,
        SDPX_LEGACY_LA_CAPABILITY_MODEL,
        (),
        :bundled_sdpx_legacy,
        equality_solver === :auto ? (:rank_revealing_qr,) : (),
        :compatibility,
        _legacy_la_symbol_ownership(arithmetic),
    )
end

"""
    fine_grained_block_bins(T, requested, reduced_arrow_panel, block_count)

Return the number of LPT bins used by short per-block solver phases. The
default preserves the requested width. Arithmetic extensions may apply a
measured, structure-aware cap without limiting the coarser Schur/SYRK kernels,
which have a different parallel crossover.
"""
fine_grained_block_bins(
    ::Type,
    requested::Int,
    reduced_arrow_panel::Bool,
    block_count::Int,
) = max(requested, 1)

"""
    fine_grained_block_partition(
        T, reduced_arrow_panel, block_dimensions, bin_count,
    )

Choose the static partition used by short block-local solver phases. The
default LPT partition preserves load balance for heterogeneous PSD blocks.
Arithmetic extensions may select contiguous ownership for measured uniform
block-arrow geometries where cache and NUMA locality dominate imbalance.
"""
fine_grained_block_partition(
    ::Type,
    reduced_arrow_panel::Bool,
    block_dimensions,
    bin_count::Int,
) = :lpt

"""
    reduced_arrow_worker_count(T, requested, block_count, columns)

Return the effective worker width for reduced-arrow panel packing and SYRK.
The default preserves the requested width; fixed-width arithmetic extensions
may cap demonstrably oversubscribed geometries without changing the number of
Julia threads available to other solver phases.
"""
reduced_arrow_worker_count(
    ::Type,
    requested::Int,
    block_count::Int,
    columns::Int,
) = max(requested, 1)

"""
    reduced_arrow_factor_worker_count(T, requested, dimension)

Return the effective worker width for a reduced-arrow Schur factorization.
The generic factor is serial. Arithmetic extensions may opt into a measured,
bounded panel-factor team for dimensions where coarse trailing updates exceed
task-launch overhead.
"""
reduced_arrow_factor_worker_count(
    ::Type,
    requested::Int,
    dimension::Int,
) = 1

"""
    reduced_arrow_solver_worker_count(
        T, requested, block_count, shared_columns,
    )

Return the effective whole-solver worker width for a reduced-arrow problem.
The default respects the request. Arithmetic extensions may cap a measured
oversubscribed geometry so every solver phase and workspace uses the same
transparent limit rather than leaving synchronization-heavy regions wider
than Schur assembly.
"""
reduced_arrow_solver_worker_count(
    ::Type,
    requested::Int,
    block_count::Int,
    shared_columns::Int,
) = max(requested, 1)

"""
    EqualityQRFactor{T}

Rank-revealing Householder QR factor used by the guarded equality fallback.
The packed provider-produced matrix, column permutation, and numerical-rank
diagnostics are retained. `coefficients` may be empty for providers whose
public contract exposes packed `R` but not reflector coefficients: SDPX's
Newton solve uses only the leading triangular `R` block to solve the semantic
`R'R` system and never treats this as a generic QR least-squares handle.
"""
abstract type AbstractLAFactorization{T} end
abstract type AbstractLACholeskyFactor{T} <: AbstractLAFactorization{T} end

"""Abstract marker for provider-neutral QR factor handles."""
abstract type AbstractLAQRFactor{T} <: AbstractLAFactorization{T} end

struct EqualityQRFactor{T,P} <: AbstractLAQRFactor{T}
    provider::P
    factors::Matrix{T}
    coefficients::Vector{T}
    permutation::Vector{Int}
    rank::Int
    quality::T
end

EqualityQRFactor{T}(
    factors::Matrix{T},
    coefficients::Vector{T},
    permutation::Vector{Int},
    rank::Int,
    quality::T,
) where {T} = EqualityQRFactor{T,Symbol}(
    :compatibility,
    factors,
    coefficients,
    permutation,
    rank,
    quality,
)

"""Standard generic factor handle wrapping Julia's Cholesky object."""
struct StandardLACholeskyFactor{T,F<:LinearAlgebra.Cholesky{T}} <:
       AbstractLACholeskyFactor{T}
    factor::F
    factors::Matrix{T}
end

"""
Provider factor payload supplied by an optional arithmetic extension.

The external factor may borrow the SDPX-owned workspace buffer that was passed
to its in-place factorization. The wrapper keeps the factor payload and storage
alive together; `provider_owned` describes execution authority, not a promise
that the provider allocated a second copy of the matrix.
"""
struct ProviderLACholeskyFactor{T,P,M<:AbstractMatrix{T}} <: AbstractLACholeskyFactor{T}
    provider::P
    factors::M
end

"""Standard LU handle exposed through the provider-neutral factor API."""
struct StandardLALUFactor{T,F<:LinearAlgebra.Factorization{T}} <:
       AbstractLAFactorization{T}
    factor::F
end

"""Provider-owned dense LU factor handle (e.g. MultiFloatLinearAlgebra)."""
struct ProviderLALUFactor{T,P,M<:AbstractMatrix{T}} <:
       AbstractLAFactorization{T}
    provider::P
    factors::M
end

"""Bundled-provider LU handle used by explicit Legacy dense LP plans."""
struct LegacyLALUFactor{T,P,F<:LinearAlgebra.Factorization{T}} <:
       AbstractLAFactorization{T}
    provider::P
    factor::F
end

"""
Successful-factor status for the provider-neutral handles.

Only handles that can wrap an unsuccessful factorization delegate to the
underlying `LinearAlgebra` object; provider-owned and borrowed-legacy handles
are constructed exclusively after their factor kernels have reported success,
so they are success by construction.
"""
LinearAlgebra.issuccess(factor::StandardLACholeskyFactor) =
    LinearAlgebra.issuccess(factor.factor)
LinearAlgebra.issuccess(::ProviderLACholeskyFactor) = true
LinearAlgebra.issuccess(factor::StandardLALUFactor) =
    LinearAlgebra.issuccess(factor.factor)
LinearAlgebra.issuccess(::ProviderLALUFactor) = true
LinearAlgebra.issuccess(factor::LegacyLALUFactor) =
    LinearAlgebra.issuccess(factor.factor)

"""Standard generic QR handle; `pivoted` records rank-revealing selection."""
struct StandardLAQRFactor{T,P,F<:LinearAlgebra.Factorization{T}} <:
       AbstractLAQRFactor{T}
    provider::P
    factor::F
    pivoted::Bool
end

"""Provider-owned symmetric-indefinite LDLT factor handle."""
struct ProviderLALDLTFactor{T,P,M<:AbstractMatrix{T}} <:
       AbstractLAFactorization{T}
    provider::P
    factors::M
end

LinearAlgebra.issuccess(::ProviderLALDLTFactor) = true

"""
    DenseAugmentedKKTWorkspace{T}

Owned storage for the explicit symmetric-indefinite Newton system.  The
unknown ordering is `[dx; dy]`; only the lower triangle of `matrix` is
authoritative.  The provider may borrow `factor_buffer` for the lifetime of
`factor`, while SDPX retains the unfactored matrix for residual evaluation.
"""
