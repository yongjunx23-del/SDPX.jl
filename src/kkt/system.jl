# Semantic homogeneous self-dual Newton system.
#
# This file is the only authority for signs in the five Newton equations.
# Numerical routes may change coordinates or eliminate variables, but they
# must obtain their right-hand sides from `HSDNewtonRHS` and validate the
# recovered direction with `newton_residual!`.

"""A self-adjoint local cone linearization and its corrector right-hand side."""
abstract type AbstractConeLinearization{T<:AbstractFloat} end

"""
    LocalConeLinearization(rows, operator, corrector_rhs)

One cone kernel contribution. `operator` is the self-adjoint map `H` in

    ds + H*dy = corrector_rhs.

`rows` names the canonical cone coordinates owned by the contribution. A
product assembly rejects gaps, overlaps, non-finite data, and non-symmetric
operators rather than guessing a layout.
"""
struct LocalConeLinearization{
    T<:AbstractFloat,M<:AbstractMatrix{T},V<:AbstractVector{T},
} <: AbstractConeLinearization{T}
    rows::UnitRange{Int}
    operator::M
    corrector_rhs::V

    function LocalConeLinearization(
        rows::UnitRange{Int}, operator::M, corrector_rhs::V,
    ) where {T<:AbstractFloat,M<:AbstractMatrix{T},V<:AbstractVector{T}}
        dimension = length(rows)
        first(rows) >= 1 || throw(ArgumentError(
            "cone contribution rows must be positive canonical indices",
        ))
        size(operator) == (dimension, dimension) || throw(DimensionMismatch(
            "cone operator size $(size(operator)) does not match row range $rows",
        ))
        length(corrector_rhs) == dimension || throw(DimensionMismatch(
            "cone corrector length $(length(corrector_rhs)) does not match row range $rows",
        ))
        issymmetric(operator) || throw(ArgumentError(
            "cone linearization must be self-adjoint",
        ))
        all(isfinite, operator) || throw(ArgumentError(
            "cone linearization contains non-finite data",
        ))
        all(isfinite, corrector_rhs) || throw(ArgumentError(
            "cone corrector right-hand side contains non-finite data",
        ))
        return new{T,M,V}(rows, operator, corrector_rhs)
    end
end

"""Setup-owned block-diagonal product-cone linearization."""
struct ProductConeLinearization{T<:AbstractFloat} <:
       AbstractConeLinearization{T}
    operator::Matrix{T}
    corrector_rhs::Vector{T}
    block_ranges::Vector{UnitRange{Int}}
end

cone_dimension(linearization::LocalConeLinearization) =
    length(linearization.rows)
cone_dimension(linearization::ProductConeLinearization) =
    length(linearization.corrector_rhs)

"""
    assemble_cone_linearization(T, dimension, contributions)

Assemble local cone kernels in canonical row order. Every row must be covered
exactly once. This is the setup invariant that prevents a ZeroCone, free row,
or untransformed cone from silently reaching the runtime KKT map.
"""
function assemble_cone_linearization(
    ::Type{T}, dimension::Int,
    contributions::AbstractVector{<:LocalConeLinearization{T}},
) where {T<:AbstractFloat}
    dimension >= 0 || throw(ArgumentError("cone dimension must be nonnegative"))
    operator = zeros(T, dimension, dimension)
    corrector_rhs = zeros(T, dimension)
    ranges = UnitRange{Int}[]
    expected = 1
    for contribution in contributions
        rows = contribution.rows
        first(rows) == expected || throw(ArgumentError(
            "cone row coverage gap or overlap: expected row $expected, got $rows",
        ))
        last(rows) <= dimension || throw(DimensionMismatch(
            "cone contribution $rows exceeds product dimension $dimension",
        ))
        operator[rows, rows] .= contribution.operator
        corrector_rhs[rows] .= contribution.corrector_rhs
        push!(ranges, rows)
        expected = last(rows) + 1
    end
    expected == dimension + 1 || throw(ArgumentError(
        "cone row coverage ends at $(expected - 1), expected $dimension",
    ))
    return ProductConeLinearization{T}(operator, corrector_rhs, ranges)
end

function apply_cone_linearization!(
    destination::AbstractVector{T},
    linearization::AbstractConeLinearization{T},
    source::AbstractVector{T},
) where {T<:AbstractFloat}
    dimension = cone_dimension(linearization)
    length(destination) == dimension || throw(DimensionMismatch(
        "cone destination length does not match linearization",
    ))
    length(source) == dimension || throw(DimensionMismatch(
        "cone source length does not match linearization",
    ))
    mul!(destination, linearization.operator, source)
    return destination
end

"""
    HSDNewtonRHS

The right-hand sides of the five authoritative HSD Newton equations:

    A*dx + ds - b*dτ       = primal_affine
    A'*dy + c*dτ           = dual_affine
   -c'*dx - b'*dy + dκ     = homogeneous_gap
    ds + H*dy              = cone_corrector
    κ*dτ + τ*dκ            = tau_kappa
"""
struct HSDNewtonRHS{
    T<:AbstractFloat,VP<:AbstractVector{T},VD<:AbstractVector{T},
    VC<:AbstractVector{T},
}
    primal_affine::VP
    dual_affine::VD
    homogeneous_gap::T
    cone_corrector::VC
    tau_kappa::T
end

"""
    residual_newton_rhs(rP, rD, rG, cone_corrector, scalar_rhs)

Construct the semantic RHS from the current HSD residuals. This conversion is
kept beside the equation definition so factorization routes never rederive the
residual signs.
"""
function residual_newton_rhs(
    rP::AbstractVector{T}, rD::AbstractVector{T}, rG::T,
    cone_corrector::AbstractVector{T}, scalar_rhs::T,
) where {T<:AbstractFloat}
    all(isfinite, rP) && all(isfinite, rD) && isfinite(rG) &&
    all(isfinite, cone_corrector) && isfinite(scalar_rhs) ||
        throw(ArgumentError("HSD Newton RHS contains non-finite data"))
    return HSDNewtonRHS(-rP, -rD, -rG, cone_corrector, scalar_rhs)
end

"""Typed semantic Newton system; numerical routes own no independent signs."""
struct NewtonSystem{
    T<:AbstractFloat,MA<:AbstractMatrix{T},VB<:AbstractVector{T},
    VC<:AbstractVector{T},L<:AbstractConeLinearization{T},
    R<:HSDNewtonRHS{T},
}
    A::MA
    b::VB
    c::VC
    cone::L
    tau::T
    kappa::T
    rhs::R

    function NewtonSystem(
        A::MA, b::VB, c::VC, cone::L, tau::T, kappa::T, rhs::R,
    ) where {
        T<:AbstractFloat,MA<:AbstractMatrix{T},VB<:AbstractVector{T},
        VC<:AbstractVector{T},L<:AbstractConeLinearization{T},
        R<:HSDNewtonRHS{T},
    }
        m, n = size(A)
        length(b) == m || throw(DimensionMismatch("A/b dimensions disagree"))
        length(c) == n || throw(DimensionMismatch("A/c dimensions disagree"))
        cone_dimension(cone) == m || throw(DimensionMismatch(
            "cone linearization dimension does not match rows of A",
        ))
        length(rhs.primal_affine) == m || throw(DimensionMismatch(
            "primal Newton RHS dimension does not match rows of A",
        ))
        length(rhs.dual_affine) == n || throw(DimensionMismatch(
            "dual Newton RHS dimension does not match columns of A",
        ))
        length(rhs.cone_corrector) == m || throw(DimensionMismatch(
            "cone Newton RHS dimension does not match rows of A",
        ))
        all(isfinite, A) && all(isfinite, b) && all(isfinite, c) &&
        isfinite(tau) && isfinite(kappa) || throw(ArgumentError(
            "Newton system contains non-finite data",
        ))
        tau > zero(T) && kappa > zero(T) || throw(ArgumentError(
            "HSD scalar iterate must be strictly interior",
        ))
        return new{T,MA,VB,VC,L,R}(A, b, c, cone, tau, kappa, rhs)
    end
end

"""Recovered direction in the original semantic Newton coordinates."""
struct NewtonDirection{
    T<:AbstractFloat,VX<:AbstractVector{T},VY<:AbstractVector{T},
    VS<:AbstractVector{T},
}
    dx::VX
    dy::VY
    ds::VS
    dtau::T
    dkappa::T
end

"""Preallocated residuals for all five semantic equation groups."""
mutable struct NewtonResidual{T<:AbstractFloat}
    primal_affine::Vector{T}
    dual_affine::Vector{T}
    homogeneous_gap::T
    cone_complementarity::Vector{T}
    tau_kappa::T
    cone_work::Vector{T}
end

function NewtonResidual(system::NewtonSystem{T}) where {T}
    m, n = size(system.A)
    return NewtonResidual{T}(
        zeros(T, m), zeros(T, n), zero(T), zeros(T, m), zero(T),
        zeros(T, m),
    )
end

"""
    newton_residual!(residual, system, direction)

Evaluate the unregularized five-equation residual. This check is authoritative
for every reduced, expanded, regularized, or mixed-precision route.
"""
function newton_residual!(
    residual::NewtonResidual{T}, system::NewtonSystem{T},
    direction::NewtonDirection{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(direction.dx) == n || throw(DimensionMismatch("dx dimension mismatch"))
    length(direction.dy) == m || throw(DimensionMismatch("dy dimension mismatch"))
    length(direction.ds) == m || throw(DimensionMismatch("ds dimension mismatch"))
    length(residual.primal_affine) == m || throw(DimensionMismatch(
        "primal residual workspace dimension mismatch",
    ))
    length(residual.dual_affine) == n || throw(DimensionMismatch(
        "dual residual workspace dimension mismatch",
    ))
    length(residual.cone_complementarity) == m || throw(DimensionMismatch(
        "cone residual workspace dimension mismatch",
    ))

    mul!(residual.primal_affine, system.A, direction.dx)
    @inbounds for i in 1:m
        residual.primal_affine[i] += direction.ds[i] -
                                     system.b[i] * direction.dtau -
                                     system.rhs.primal_affine[i]
    end

    mul!(residual.dual_affine, transpose(system.A), direction.dy)
    @inbounds for j in 1:n
        residual.dual_affine[j] += system.c[j] * direction.dtau -
                                   system.rhs.dual_affine[j]
    end

    gap = direction.dkappa - system.rhs.homogeneous_gap
    @inbounds for j in 1:n
        gap -= system.c[j] * direction.dx[j]
    end
    @inbounds for i in 1:m
        gap -= system.b[i] * direction.dy[i]
    end
    residual.homogeneous_gap = gap

    apply_cone_linearization!(
        residual.cone_work, system.cone, direction.dy,
    )
    @inbounds for i in 1:m
        residual.cone_complementarity[i] = direction.ds[i] +
            residual.cone_work[i] - system.rhs.cone_corrector[i]
    end

    residual.tau_kappa = system.kappa * direction.dtau +
                         system.tau * direction.dkappa -
                         system.rhs.tau_kappa
    return residual
end

"""Infinity norm over all five unregularized Newton equation groups."""
function max_newton_residual(residual::NewtonResidual{T}) where {T}
    maximum_residual = max(
        abs(residual.homogeneous_gap), abs(residual.tau_kappa),
    )
    @inbounds for value in residual.primal_affine
        maximum_residual = max(maximum_residual, abs(value))
    end
    @inbounds for value in residual.dual_affine
        maximum_residual = max(maximum_residual, abs(value))
    end
    @inbounds for value in residual.cone_complementarity
        maximum_residual = max(maximum_residual, abs(value))
    end
    return maximum_residual
end
