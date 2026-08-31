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

function _validate_spec(spec::ModularPMPSpec)
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
    rank(Matrix{Float64}(spec.basis)) == length(spec.derivative_orders) ||
        throw(ArgumentError("functional basis is rank deficient"))
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
            ehat0 = (T(2) - T(2) - T(2)) / T(24)
            shift = gap + ehat0
            basis = zeros(T, degree + 1, length(orders))
            for (column, order) in pairs(orders)
                coefficients = _poly_coefficients(order, shift)
                basis[1:length(coefficients), column] = coefficients
            end
            normalization = T[HellermanModularLP.vacuum_polynomial(
                order, ehat0; eta_product_terms=ETA_PRODUCT_TERMS,
            ) for order in orders]
            return (fixed_gap=ModularPMPSpec{T}(
                id="hellerman09/fixed_gap_pmp_d4",
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

function _provenance(spec::ModularPMPSpec)
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
            "Eq. 3.29: spectral equations with exp(-2*pi*Delta) factor",
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

function _validate_gram(X, strict)
    size(X,1)==size(X,2) || return false
    issymmetric(X) || return false
    # LinearAlgebra's generic BigFloat eigensolver is not available on all
    # supported Julia builds; PSD is a structural validator here, so inspect
    # the exact stored matrix through a Float64 shadow without changing any
    # solver arithmetic or certificate tolerance.
    λ = eigvals(Symmetric(Matrix{Float64}(X)))
    all(isfinite, λ) && (strict ? minimum(λ) > 0 : minimum(λ) >= 0)
end

function validate_artifact(artifact::ModularPMPArtifact)
    failures=String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures,"schema_version")
    valid_spec=true; try _validate_spec(artifact.spec) catch; valid_spec=false; push!(failures,"spec") end
    dims=halfline_gram_dimensions(artifact.spec.degree)
    artifact.gram_dimensions == dims || push!(failures,"gram_dimensions")
    length(artifact.witness_alpha)==length(artifact.spec.derivative_orders) || push!(failures,"alpha_dimensions")
    size(artifact.witness_Q)==(dims.q_blocks,dims.q_blocks) || push!(failures,"Q_dimensions")
    size(artifact.witness_R)==(dims.r_blocks,dims.r_blocks) || push!(failures,"R_dimensions")
    expected = iseven(artifact.spec.degree) ? reconstruct_even_coefficients(artifact.witness_Q,artifact.witness_R) : reconstruct_odd_coefficients(artifact.witness_Q,artifact.witness_R)
    expected == artifact.witness_polynomial || push!(failures,"gram_reconstruction")
    size(artifact.spec.basis,2)==length(artifact.witness_alpha) || push!(failures,"basis_alpha_dimensions")
    (artifact.spec.basis * artifact.witness_alpha == artifact.witness_polynomial) || push!(failures,"basis_alpha_reconstruction")
    dot(artifact.spec.normalization,artifact.witness_alpha) == one(eltype(artifact.witness_alpha)) || push!(failures,"vacuum_normalization")
    _validate_gram(artifact.witness_Q,artifact.witness_strict) || push!(failures,"Q_psd")
    _validate_gram(artifact.witness_R,artifact.witness_strict) || push!(failures,"R_psd")
    artifact.fingerprint == stable_fingerprint(artifact) || push!(failures,"fingerprint")
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

function build_modular_pmp_sdp(spec::ModularPMPSpec{R}, ::Type{T}=Float64) where {R<:AbstractFloat,T<:AbstractFloat}
    _validate_spec(spec); throw(ArgumentError("no certified strict witness; modular PMP remains build-only and fail-closed until a valid functional is supplied"))
end

end
