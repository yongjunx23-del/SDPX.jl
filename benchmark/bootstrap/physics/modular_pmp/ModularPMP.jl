module ModularPMP

using SHA
using LinearAlgebra
import SDPX

const _HELLERMAN_PATH = joinpath(@__DIR__, "..", "modular_lp", "HellermanModularLP.jl")
if !isdefined(@__MODULE__, :HellermanModularLP)
    Base.include(@__MODULE__, _HELLERMAN_PATH)
end
using .HellermanModularLP

export ModularPMPSpec, ModularPMPArtifact, modular_pmp_specs
export build_modular_pmp, build_modular_pmp_sdp
export halfline_gram_dimensions, reconstruct_even_coefficients
export reconstruct_odd_coefficients, chi_positive_on_halfline
export validate_artifact, canonical_text, stable_fingerprint

const ARTIFACT_SCHEMA_VERSION = 2
const PRIMARY_ARXIV = "0902.2790"
const PRIMARY_VERSION = "0902.2790v2"
const SOURCE_STATUS = :build_only
const ETA_PRODUCT_TERMS = HellermanModularLP.ETA_PRODUCT_TERMS

Base.@kwdef struct ModularPMPSpec{T<:AbstractFloat}
    id::String
    degree::Int
    gap::T
    basis::Matrix{T}                 # coefficient rows, low power first
    normalization::Vector{T}          # b_p(Ehat0), actual vacuum data
    derivative_orders::Vector{Int}
    left_central_charge::T = T(2)
    right_central_charge::T = T(2)
    eta_product_terms::Int = ETA_PRODUCT_TERMS
    source::Symbol = :primary_paper
    source_version::String = PRIMARY_VERSION
    reference_status::Symbol = SOURCE_STATUS
    paper_equivalent::Bool = false
end

struct ModularPMPArtifact{T<:AbstractFloat}
    schema_version::Int
    spec::ModularPMPSpec{T}
    gram_dimensions::NamedTuple
    witness_alpha::Vector{T}
    witness_polynomial::Vector{T}
    witness_Q::Matrix{T}
    witness_R::Matrix{T}
    witness_strict::Bool
    provenance::NamedTuple
    fingerprint::String
end

function halfline_gram_dimensions(degree::Int, matrix_dimension::Int=1)
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    matrix_dimension >= 1 || throw(ArgumentError("matrix dimension must be positive"))
    if iseven(degree)
        d = degree ÷ 2
        return (parity=:even, q=matrix_dimension * (d + 1), r=matrix_dimension * d,
            q_blocks=d + 1, r_blocks=d)
    end
    d = (degree - 1) ÷ 2
    return (parity=:odd, q=matrix_dimension * (d + 1), r=matrix_dimension * (d + 1),
        q_blocks=d + 1, r_blocks=d + 1)
end

function reconstruct_even_coefficients(Q::AbstractMatrix, R::AbstractMatrix)
    qdim = size(Q, 1)
    size(Q, 2) == qdim || throw(DimensionMismatch("Q must be square"))
    d = qdim - 1
    size(R) == (d, d) || throw(DimensionMismatch("R has wrong parity dimension"))
    out = zeros(promote_type(eltype(Q), eltype(R)), 2d + 1)
    for i in 0:d, j in 0:d
        out[i + j + 1] += Q[i + 1, j + 1]
    end
    for i in 0:(d - 1), j in 0:(d - 1)
        out[i + j + 2] += R[i + 1, j + 1]
    end
    return out
end

function reconstruct_odd_coefficients(Q::AbstractMatrix, R::AbstractMatrix)
    qdim = size(Q, 1)
    size(Q, 2) == qdim || throw(DimensionMismatch("Q must be square"))
    size(R) == size(Q) || throw(DimensionMismatch("Q and R dimensions differ"))
    d = qdim - 1
    out = zeros(promote_type(eltype(Q), eltype(R)), 2d + 2)
    for i in 0:d, j in 0:d
        out[i + j + 2] += Q[i + 1, j + 1]
        out[i + j + 1] += R[i + 1, j + 1]
    end
    return out
end

chi_positive_on_halfline(gap::Real) = isfinite(gap) && gap > zero(gap)

function _canonical_spec_components(::Type{T}, gap, degree, orders, cL, cR) where {T<:AbstractFloat}
    degree == 15 || throw(ArgumentError("the pinned Hellerman jet is audited through degree 15"))
    orders == collect(1:2:15) || throw(ArgumentError("the pinned Hellerman jet uses odd orders 1:2:15"))
    cL == T(2) && cR == T(2) || throw(ArgumentError("the pinned fixture uses c_L=c_R=2"))
    gap == T(1) / T(4) || throw(ArgumentError("the pinned fixture uses gap=1/4"))
    ehat0 = (T(2) - cL - cR) / T(24)
    shift = gap + ehat0
    basis = zeros(T, degree + 1, length(orders))
    for (column, order) in pairs(orders)
        coefficients = _poly_coefficients(order, shift)
        basis[1:length(coefficients), column] = coefficients
    end
    normalization = T[HellermanModularLP.vacuum_polynomial(
        order, ehat0; eta_product_terms=ETA_PRODUCT_TERMS,
    ) for order in orders]
    return basis, normalization
end

function _validate_spec(spec::ModularPMPSpec{T}) where {T<:AbstractFloat}
    spec.source === :primary_paper || throw(ArgumentError("source must be :primary_paper"))
    spec.source_version == PRIMARY_VERSION || throw(ArgumentError("source version is not pinned"))
    spec.reference_status === SOURCE_STATUS || throw(ArgumentError("reference status must be :build_only"))
    !spec.paper_equivalent || throw(ArgumentError("finite eta/truncation artifact is not paper-equivalent"))
    spec.left_central_charge > one(spec.left_central_charge) || throw(ArgumentError("c_L must exceed one"))
    spec.right_central_charge > one(spec.right_central_charge) || throw(ArgumentError("c_R must exceed one"))
    chi_positive_on_halfline(spec.gap) || throw(ArgumentError("gap must be positive"))
    spec.eta_product_terms == ETA_PRODUCT_TERMS || throw(ArgumentError("eta product term count is frozen"))
    !isempty(spec.derivative_orders) || throw(ArgumentError("derivative orders cannot be empty"))
    all(order -> order >= 1 && isodd(order) && order <= 15, spec.derivative_orders) ||
        throw(ArgumentError("orders must be odd and within the audited Hellerman jet"))
    issorted(spec.derivative_orders) && length(unique(spec.derivative_orders)) == length(spec.derivative_orders) ||
        throw(ArgumentError("derivative orders must be sorted and unique"))
    size(spec.basis) == (spec.degree + 1, length(spec.derivative_orders)) ||
        throw(DimensionMismatch("basis dimensions do not match degree/orders"))
    length(spec.normalization) == length(spec.derivative_orders) ||
        throw(DimensionMismatch("vacuum normalization dimension mismatch"))
    all(isfinite, spec.basis) && all(isfinite, spec.normalization) ||
        throw(ArgumentError("basis and normalization must be finite"))
    try
        rank(spec.basis) == length(spec.derivative_orders) ||
            throw(ArgumentError("functional basis is rank deficient"))
    catch error
        error isa ArgumentError && rethrow()
        throw(ArgumentError("functional basis rank is unavailable for $(T)"))
    end
    canonical_basis, canonical_normalization = setprecision(256) do
        _canonical_spec_components(
            T, spec.gap, spec.degree, spec.derivative_orders,
            spec.left_central_charge, spec.right_central_charge,
        )
    end
    spec.basis == canonical_basis || throw(ArgumentError(
        "basis does not equal the pinned Hellerman derivative-polynomial rebuild",
    ))
    spec.normalization == canonical_normalization || throw(ArgumentError(
        "normalization does not equal the pinned Hellerman vacuum rebuild",
    ))
    return nothing
end

function _poly_coefficients(order::Int, shift::T) where {T<:AbstractFloat}
    d = order
    V = T[x^k for x in 0:d, k in 0:d]
    values = T[HellermanModularLP.derivative_polynomial(order, T(x) + shift) for x in 0:d]
    return V \ values
end

function modular_pmp_specs(::Type{T}=BigFloat) where {T<:AbstractFloat}
    build = function ()
        setprecision(256) do
            orders = collect(1:2:15)
            degree = maximum(orders)
            gap = T(1) / T(4)
            basis, normalization = _canonical_spec_components(
                T, gap, degree, orders, T(2), T(2),
            )
            return (fixed_gap=ModularPMPSpec{T}(
                id="hellerman09/fixed_gap_degree15_odd1_15",
                degree=degree, gap=gap, basis=basis,
                normalization=normalization, derivative_orders=orders,
                left_central_charge=T(2), right_central_charge=T(2),
            ),)
        end
    end
    T === BigFloat || throw(ArgumentError(
        "actual Hellerman PMP coefficients require BigFloat construction at 256 bits",
    ))
    return build()
end

function _provenance(spec::ModularPMPSpec{T}) where {T<:AbstractFloat}
    return (
        title="A Universal Inequality for CFT and Quantum Gravity",
        author="Simeon Hellerman", arxiv=PRIMARY_ARXIV, source_version=PRIMARY_VERSION,
        reference_status=SOURCE_STATUS, paper_equivalent=false,
        central_charge_assumptions=(c_L_gt_one=true, c_R_gt_one=true,
            c_L=spec.left_central_charge, c_R=spec.right_central_charge,
            Ehat0=(spec.right_central_charge + spec.left_central_charge - T(2)) / T(-24)),
        implemented_equations=(
            "Sec. 2.2: tau=i and beta=2pi fixed-point odd derivatives",
            "Eqs. 3.18-3.21: Virasoro character decomposition",
            "Eqs. 3.22-3.24: derivative polynomials f_p from the eta q-product",
            "Eq. 3.28: vacuum polynomial b_p",
            "Eq. 3.29: checked p=1,3 spectral rows with exp(-2*pi*Delta) factor",
            "odd orders 5..15 are a literal derivative-polynomial extension, not an Eq. 3.29 quotation",
        ),
        eta_product_terms=ETA_PRODUCT_TERMS,
        derivative_orders=Tuple(spec.derivative_orders),
        positivity=(domain="Delta >= gap", factored_character="exp(-2*pi*Delta) * p(Delta-gap)",
            exp_factor_strictly_positive=true),
        lifting=(parity=iseven(spec.degree) ? :markov_lukacs_even : :markov_lukacs_odd,
            gram_coefficient_order=:low_power_first),
        claims=(status=:build_only, continuum_bound=false, published_objective=false,
            paper_equivalent=false),
    )
end

function _strict_witness(spec::ModularPMPSpec{T}) where {T}
    # The actual Hellerman derivative basis is retained. Construction fails
    # closed unless an independently certified strict polynomial exists.
    throw(ArgumentError("no certified strict witness exists for the default actual Hellerman basis at gap=$(spec.gap); refusing synthetic witness"))
end

build_modular_pmp(spec::ModularPMPSpec) = (_validate_spec(spec); _strict_witness(spec))
build_modular_pmp(scale::Symbol, ::Type{T}=BigFloat) where {T<:AbstractFloat} =
    build_modular_pmp(getproperty(modular_pmp_specs(T), scale))

function canonical_text(artifact::ModularPMPArtifact)
    io = IOBuffer(); println(io, "modular-pmp-schema=", artifact.schema_version)
    println(io, "artifact.gram_dimensions=", repr(artifact.gram_dimensions))
    for name in fieldnames(typeof(artifact.spec)); println(io, "spec.", name, '=', repr(getfield(artifact.spec, name))); end
    for (name, value) in pairs(artifact.provenance); println(io, "provenance.", name, '=', repr(value)); end
    println(io, "witness.strict=", artifact.witness_strict)
    for (name, values) in (("alpha",artifact.witness_alpha),("p",artifact.witness_polynomial))
        for (i,v) in pairs(values); println(io, "witness.",name,"[",i,"]=",repr(v)); end
    end
    for (name, matrix) in (("basis",artifact.spec.basis),("Q",artifact.witness_Q),("R",artifact.witness_R))
        for i in axes(matrix,1), j in axes(matrix,2); println(io, name,"[",i,",",j,"]=",repr(matrix[i,j])); end
    end
    for (i,v) in pairs(artifact.spec.normalization); println(io,"normalization[",i,"]=",repr(v)); end
    return String(take!(io))
end
stable_fingerprint(artifact::ModularPMPArtifact) = bytes2hex(SHA.sha256(codeunits(canonical_text(artifact))))

function _validate_gram(X::AbstractMatrix{T}, strict::Bool) where {T}
    T <: AbstractFloat || return false
    size(X, 1) == size(X, 2) || return false
    all(isfinite, X) || return false
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        X[i, j] == X[j, i] || return false
    end
    n = size(X, 1)
    n == 0 && return !strict
    # Symmetric Schur-complement PSD test. This is deliberately elementary:
    # it preserves T for Float64 and BigFloat and does not rely on the
    # unavailable BigFloat eigvals implementation in Julia 1.12. For a PSD
    # matrix, a zero diagonal pivot must have an identically zero row/column;
    # otherwise the matrix is indefinite. Positive pivots recurse on the
    # T-native Schur complement. Strict mode requires every pivot to be
    # strictly positive (a positive-definite certificate).
    work = Matrix{T}(X)
    @inbounds for k in 1:n
        pivot = work[k, k]
        isfinite(pivot) || return false
        if pivot < zero(T)
            return false
        elseif pivot == zero(T)
            strict && return false
            for i in (k + 1):n
                work[i, k] == zero(T) || return false
            end
            continue
        end
        for i in (k + 1):n
            mik = work[i, k]
            for j in i:n
                work[j, i] -= mik * work[j, k] / pivot
            end
        end
    end
    return true
end

function _all_finite(value)
    try
        return all(isfinite, value)
    catch
        return false
    end
end

function validate_artifact(artifact::ModularPMPArtifact)
    failures = String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures, "schema_version")
    try
        _validate_spec(artifact.spec)
    catch
        push!(failures, "spec")
    end
    dims = try
        halfline_gram_dimensions(artifact.spec.degree)
    catch
        push!(failures, "gram_dimensions")
        nothing
    end
    if dims !== nothing
        artifact.gram_dimensions == dims || push!(failures, "gram_dimensions")
        size(artifact.witness_Q) == (dims.q_blocks, dims.q_blocks) || push!(failures, "Q_dimensions")
        size(artifact.witness_R) == (dims.r_blocks, dims.r_blocks) || push!(failures, "R_dimensions")
    end
    _all_finite(artifact.witness_alpha) || push!(failures, "alpha_finite")
    _all_finite(artifact.witness_polynomial) || push!(failures, "polynomial_finite")
    _all_finite(artifact.witness_Q) || push!(failures, "Q_finite")
    _all_finite(artifact.witness_R) || push!(failures, "R_finite")
    if dims !== nothing && size(artifact.witness_Q) == (dims.q_blocks, dims.q_blocks) &&
       size(artifact.witness_R) == (dims.r_blocks, dims.r_blocks)
        expected = try
            iseven(artifact.spec.degree) ?
                reconstruct_even_coefficients(artifact.witness_Q, artifact.witness_R) :
                reconstruct_odd_coefficients(artifact.witness_Q, artifact.witness_R)
        catch
            nothing
        end
        expected === nothing && push!(failures, "gram_reconstruction")
        expected !== nothing && expected != artifact.witness_polynomial &&
            push!(failures, "gram_reconstruction")
    else
        push!(failures, "gram_reconstruction")
    end
    if size(artifact.spec.basis, 2) == length(artifact.witness_alpha)
        try
            artifact.spec.basis * artifact.witness_alpha == artifact.witness_polynomial ||
                push!(failures, "basis_alpha_reconstruction")
        catch
            push!(failures, "basis_alpha_reconstruction")
        end
    else
        push!(failures, "basis_alpha_dimensions")
    end
    try
        dot(artifact.spec.normalization, artifact.witness_alpha) == one(eltype(artifact.witness_alpha)) ||
            push!(failures, "vacuum_normalization")
    catch
        push!(failures, "vacuum_normalization")
    end
    _validate_gram(artifact.witness_Q, artifact.witness_strict) || push!(failures, "Q_psd")
    _validate_gram(artifact.witness_R, artifact.witness_strict) || push!(failures, "R_psd")
    try
        artifact.provenance == _provenance(artifact.spec) || push!(failures, "provenance")
    catch
        push!(failures, "provenance")
    end
    artifact.fingerprint == stable_fingerprint(artifact) || push!(failures, "fingerprint")
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

function build_modular_pmp_sdp(spec::ModularPMPSpec{R}, ::Type{T}=Float64) where {R<:AbstractFloat,T<:AbstractFloat}
    _validate_spec(spec); throw(ArgumentError("no certified strict witness; modular PMP remains build-only and fail-closed until a valid functional is supplied"))
end

end
