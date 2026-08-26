module HellermanModularLP

using SHA
using SparseArrays
import SDPX

export ModularLPSpec, ModularLPArtifact
export modular_lp_specs, build_modular_lp, build_lp_problem, validate_artifact
export derivative_polynomial, vacuum_polynomial
export canonical_text, stable_fingerprint

const ARTIFACT_SCHEMA_VERSION = 1
const PRIMARY_ARXIV = "0902.2790"
const PRIMARY_VERSION = "0902.2790v2"
const SOURCE_STATUS = :build_only
const FORMULA_ORACLE_STATUS = :verified
const ETA_PRODUCT_TERMS = 64

Base.@kwdef struct ModularLPSpec{T}
    id::String
    scale::Symbol
    left_central_charge::T = T(2)
    right_central_charge::T = T(2)
    maximum_derivative_order::Int
    dimension_points::Int
    dimension_minimum::T
    dimension_maximum::T
    eta_product_terms::Int = ETA_PRODUCT_TERMS
    source::Symbol = :primary_paper
    source_version::String = PRIMARY_VERSION
    reference_status::Symbol = SOURCE_STATUS
    paper_equivalent::Bool = false
end

struct ModularLPArtifact{T}
    schema_version::Int
    spec::ModularLPSpec{T}
    derivative_orders::Vector{Int}
    dimensions::Vector{T}
    coefficients::Matrix{T}
    rhs::Vector{T}
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

function _validate_spec(spec::ModularLPSpec)
    spec.source === :primary_paper || throw(ArgumentError(
        "source must be :primary_paper",
    ))
    spec.source_version == PRIMARY_VERSION || throw(ArgumentError(
        "source_version must be $PRIMARY_VERSION",
    ))
    spec.reference_status === SOURCE_STATUS || throw(ArgumentError(
        "reference_status must be :build_only",
    ))
    spec.paper_equivalent && throw(ArgumentError(
        "finite dimension-grid artifacts are not paper-equivalent continuum bounds",
    ))
    spec.left_central_charge > one(spec.left_central_charge) || throw(ArgumentError(
        "Hellerman's character decomposition assumes c_L > 1",
    ))
    spec.right_central_charge > one(spec.right_central_charge) || throw(ArgumentError(
        "Hellerman's character decomposition assumes c_R > 1",
    ))
    spec.maximum_derivative_order >= 1 && isodd(spec.maximum_derivative_order) ||
        throw(ArgumentError("maximum_derivative_order must be positive and odd"))
    spec.maximum_derivative_order <= 15 || throw(ArgumentError(
        "maximum_derivative_order exceeds the audited Taylor-jet range",
    ))
    spec.dimension_points >= 2 || throw(ArgumentError(
        "dimension_points must be at least two",
    ))
    spec.dimension_minimum > zero(spec.dimension_minimum) || throw(ArgumentError(
        "the non-vacuum primary grid must start at positive dimension",
    ))
    spec.dimension_maximum > spec.dimension_minimum || throw(ArgumentError(
        "dimension_maximum must exceed dimension_minimum",
    ))
    spec.eta_product_terms == ETA_PRODUCT_TERMS || throw(ArgumentError(
        "eta_product_terms is frozen at $ETA_PRODUCT_TERMS for reproducibility",
    ))
    return nothing
end

function modular_lp_specs(::Type{T}=Float64) where {T}
    common = (
        left_central_charge=T(2),
        right_central_charge=T(2),
        dimension_minimum=T(1) / T(4),
    )
    return (
        tiny=ModularLPSpec{T}(;
            id="hellerman09/c4_grid16_p1",
            scale=:tiny,
            maximum_derivative_order=1,
            dimension_points=16,
            dimension_maximum=T(4),
            common...,
        ),
        small=ModularLPSpec{T}(;
            id="hellerman09/c4_grid32_p3",
            scale=:small,
            maximum_derivative_order=3,
            dimension_points=32,
            dimension_maximum=T(6),
            common...,
        ),
        medium=ModularLPSpec{T}(;
            id="hellerman09/c4_grid64_p5",
            scale=:medium,
            maximum_derivative_order=5,
            dimension_points=64,
            dimension_maximum=T(8),
            common...,
        ),
        stress=ModularLPSpec{T}(;
            id="hellerman09/c4_grid128_p7",
            scale=:stress,
            maximum_derivative_order=7,
            dimension_points=128,
            dimension_maximum=T(10),
            common...,
        ),
    )
end

function _series_exp(input::Vector{T}) where {T}
    degree = length(input) - 1
    output = zeros(T, degree + 1)
    output[1] = exp(input[1])
    for n in 1:degree
        value = zero(T)
        for k in 1:n
            value += T(k) * input[k + 1] * output[n - k + 1]
        end
        output[n + 1] = value / T(n)
    end
    return output
end

function _series_inverse(input::Vector{T}) where {T}
    iszero(input[1]) && throw(ArgumentError("formal series is not invertible"))
    degree = length(input) - 1
    output = zeros(T, degree + 1)
    output[1] = inv(input[1])
    for n in 1:degree
        value = zero(T)
        for k in 1:n
            value += input[k + 1] * output[n - k + 1]
        end
        output[n + 1] = -value / input[1]
    end
    return output
end

function _series_log(input::Vector{T}) where {T}
    input[1] > zero(T) || throw(ArgumentError(
        "real formal logarithm requires a positive constant term",
    ))
    degree = length(input) - 1
    inverse = _series_inverse(input)
    output = zeros(T, degree + 1)
    output[1] = log(input[1])
    for n in 1:degree
        value = zero(T)
        for k in 1:n
            value += T(k) * input[k + 1] * inverse[n - k + 1]
        end
        output[n + 1] = value / T(n)
    end
    return output
end

"""Evaluate Hellerman's `f_p(z)` defined by Eq. (3.22).

The derivative `β∂β` is represented exactly as `∂t` at
`β = 2π exp(t)`. The Dedekind eta factor uses a frozen 64-term q-product;
this numerical truncation is recorded in every artifact and is not presented
as an exact symbolic reconstruction.
"""
function derivative_polynomial(
    order::Int,
    z::T;
    eta_product_terms::Int=ETA_PRODUCT_TERMS,
) where {T}
    order >= 0 || throw(ArgumentError("derivative order must be nonnegative"))
    order <= 15 || throw(ArgumentError("derivative order exceeds audited range"))
    eta_product_terms == ETA_PRODUCT_TERMS || throw(ArgumentError(
        "eta_product_terms is frozen at $ETA_PRODUCT_TERMS",
    ))

    pi_t = T(pi)
    beta0 = T(2) * pi_t
    exp_t = T[inv(T(factorial(k))) for k in 0:order]
    beta = beta0 .* exp_t
    delta_beta = copy(beta)
    delta_beta[1] -= beta0

    log_normalized = -(z - inv(T(12))) .* delta_beta
    for n in 1:eta_product_terms
        exponent = -T(n) .* beta
        qn = _series_exp(exponent)
        one_minus_qn = -qn
        one_minus_qn[1] += one(T)
        log_factor = _series_log(one_minus_qn)
        # The eta ratio is normalized to one at t=0.
        log_factor[1] = zero(T)
        log_normalized .-= T(2) .* log_factor
    end
    normalized = _series_exp(log_normalized)
    sign = isodd(order) ? -one(T) : one(T)
    return sign * T(factorial(order)) * normalized[order + 1]
end

"""Vacuum polynomial `b_p(z)` from Hellerman Eq. (3.28)."""
function vacuum_polynomial(
    order::Int,
    z::T;
    eta_product_terms::Int=ETA_PRODUCT_TERMS,
) where {T}
    q = exp(-T(2) * T(pi))
    return derivative_polynomial(order, z; eta_product_terms) -
           T(2) * q * derivative_polynomial(order, z + one(T); eta_product_terms) +
           q * q * derivative_polynomial(order, z + T(2); eta_product_terms)
end

function _dimension_grid(spec::ModularLPSpec{T}) where {T}
    denominator = T(spec.dimension_points - 1)
    width = spec.dimension_maximum - spec.dimension_minimum
    return T[
        spec.dimension_minimum + width * T(index - 1) / denominator
        for index in 1:spec.dimension_points
    ]
end

function _provenance()
    return (
        title="A Universal Inequality for CFT and Quantum Gravity",
        author="Simeon Hellerman",
        arxiv=PRIMARY_ARXIV,
        source_version=PRIMARY_VERSION,
        reference_status=SOURCE_STATUS,
        formula_oracle_status=FORMULA_ORACLE_STATUS,
        paper_equivalent=false,
        implemented_equations=(
            "Sec. 2.2: tau=i, beta=2pi modular fixed-point odd derivatives",
            "3.18-3.21: Virasoro-character partition-function decomposition",
            "3.22: derivative polynomials f_p",
            "3.28: vacuum polynomials b_p",
            "3.29: p=1,3 spectral equations; extended to higher odd p by the preceding fixed-point identity",
        ),
        discretization=(
            "non-vacuum primary dimensions restricted to a finite positive grid",
            "nonnegative variables represent grid spectral weights",
            "Dedekind eta q-product truncated to 64 terms",
        ),
        excluded_claims=(
            "not a rigorous continuum functional bound",
            "not the analytic Hellerman Delta_1 inequality",
            "not a reconstruction of an actual integer-degeneracy CFT spectrum",
        ),
    )
end

function build_modular_lp(spec::ModularLPSpec{T}) where {T}
    _validate_spec(spec)
    orders = collect(1:2:spec.maximum_derivative_order)
    dimensions = _dimension_grid(spec)
    equations = length(orders)
    variables = length(dimensions)
    coefficients = Matrix{T}(undef, equations, variables)
    rhs = Vector{T}(undef, equations)
    ehat0 = (T(2) - spec.left_central_charge - spec.right_central_charge) / T(24)
    for (row, order) in pairs(orders)
        rhs[row] = -vacuum_polynomial(
            order,
            ehat0;
            eta_product_terms=spec.eta_product_terms,
        )
        for (column, dimension) in pairs(dimensions)
            coefficients[row, column] = derivative_polynomial(
                order,
                dimension + ehat0;
                eta_product_terms=spec.eta_product_terms,
            ) * exp(-T(2) * T(pi) * dimension)
        end
    end
    counts = (
        variables=variables,
        nonnegative_rows=variables,
        equality_rows=equations,
        derivative_orders=Tuple(orders),
        dimension_points=variables,
        eta_product_terms=spec.eta_product_terms,
        discretization=:finite_dimension_grid_not_continuum,
        paper_equivalent=false,
    )
    provisional = ModularLPArtifact{T}(
        ARTIFACT_SCHEMA_VERSION,
        spec,
        orders,
        dimensions,
        coefficients,
        rhs,
        _provenance(),
        counts,
        "",
    )
    return ModularLPArtifact{T}(
        provisional.schema_version,
        provisional.spec,
        provisional.derivative_orders,
        provisional.dimensions,
        provisional.coefficients,
        provisional.rhs,
        provisional.provenance,
        provisional.counts,
        stable_fingerprint(provisional),
    )
end

build_modular_lp(scale::Symbol, ::Type{T}=Float64) where {T} =
    build_modular_lp(getproperty(modular_lp_specs(T), scale))

_number_token(value) = string(value)

function canonical_text(artifact::ModularLPArtifact)
    io = IOBuffer()
    spec = artifact.spec
    println(io, "hellerman-modular-lp-schema=", artifact.schema_version)
    for name in fieldnames(typeof(spec))
        println(io, "spec.", name, '=', repr(getfield(spec, name)))
    end
    for (name, value) in pairs(artifact.provenance)
        println(io, "provenance.", name, '=', repr(value))
    end
    for (name, value) in pairs(artifact.counts)
        println(io, "count.", name, '=', repr(value))
    end
    for (index, order) in pairs(artifact.derivative_orders)
        println(io, "derivative[", index, "]=", order)
    end
    for (index, dimension) in pairs(artifact.dimensions)
        println(io, "delta[", index, "]=", _number_token(dimension))
    end
    for row in axes(artifact.coefficients, 1), column in axes(artifact.coefficients, 2)
        println(io, "Aeq[", row, ',', column, "]=",
            _number_token(artifact.coefficients[row, column]))
    end
    for (row, value) in pairs(artifact.rhs)
        println(io, "beq[", row, "]=", _number_token(value))
    end
    return String(take!(io))
end

stable_fingerprint(artifact::ModularLPArtifact) =
    bytes2hex(SHA.sha256(codeunits(canonical_text(artifact))))

function validate_artifact(artifact::ModularLPArtifact)
    failures = String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures, "schema_version")
    valid_spec = true
    try
        _validate_spec(artifact.spec)
    catch
        valid_spec = false
        push!(failures, "spec")
    end
    all(isfinite, artifact.dimensions) || push!(failures, "dimension_nonfinite")
    all(isfinite, artifact.coefficients) || push!(failures, "coefficient_nonfinite")
    all(isfinite, artifact.rhs) || push!(failures, "rhs_nonfinite")
    size(artifact.coefficients) ==
        (length(artifact.derivative_orders), length(artifact.dimensions)) ||
        push!(failures, "coefficient_dimensions")
    if valid_spec
        try
            expected = build_modular_lp(artifact.spec)
            artifact.derivative_orders == expected.derivative_orders ||
                push!(failures, "derivative_semantics")
            artifact.dimensions == expected.dimensions ||
                push!(failures, "grid_semantics")
            artifact.coefficients == expected.coefficients ||
                push!(failures, "coefficient_semantics")
            artifact.rhs == expected.rhs || push!(failures, "rhs_semantics")
            artifact.provenance == expected.provenance ||
                push!(failures, "provenance_semantics")
            artifact.counts == expected.counts || push!(failures, "counts_semantics")
        catch
            push!(failures, "semantic_rebuild")
        end
    end
    stable_fingerprint(artifact) == artifact.fingerprint || push!(failures, "fingerprint")
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

"""Lower to `min 0` with `G=I`, `h=0`, hence nonnegative grid weights."""
function build_lp_problem(artifact::ModularLPArtifact{T}) where {T}
    verdict = validate_artifact(artifact)
    verdict.valid || throw(ArgumentError(
        "invalid modular LP artifact: $(join(verdict.failures, ", "))",
    ))
    variables = length(artifact.dimensions)
    return SDPX.linear_program(
        zeros(T, variables),
        spdiagm(0 => ones(T, variables)),
        zeros(T, variables);
        Aeq=artifact.coefficients,
        beq=artifact.rhs,
        T,
        sparse=true,
        validate=true,
        verbosity=0,
    )
end

end # module
