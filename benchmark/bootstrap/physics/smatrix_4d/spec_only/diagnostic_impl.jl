module SMatrix4DSpecDiagnostic

using SHA
using SparseArrays
import SDPX

# SPEC-ONLY diagnostic implementation. It is not included by any catalog.

const ARTIFACT_SCHEMA_VERSION = 1
const PRIMARY_ARXIV = "2605.06613"
const PRIMARY_VERSION = "2605.06613v1"
const SOURCE_STATUS = :sampled_build_only
const PREDECESSOR_REFERENCES = (
    "2210.01502v2", "1708.06765v1", "2106.10257",
)
const S_STAR_FACTOR = 4 / 3
const WITNESS_MARGIN_FLOOR_FACTOR = 100

# Immutable Appendix-A parameter provenance.  These values are not silently
# substituted into the implemented single-anchor diagnostic basis.
const PAPER_DEFAULTS = (
    basis_centers=(20 / 3, 10, 20, 30, 40, 50, 60, 86),
    nmax_values=(10, 12, 14, 16, 18, 20),
    lmax_values=(16, 18),
    s_max=300,
    alpha_th=:zero,
    a7_t_grid=(0.27, 1.0, 2.0, 3.0, 3.73, 3.99994, 3.99996, 3.99998, 3.99999, 4.0),
    dual_mu2=12,
    dual_grid_count=200,
    dual_spin_max=32,
)

Base.@kwdef struct SMatrix4DSpec{T}
    id::String
    scale::Symbol
    external_mass::T = one(T)
    ansatz_degree::Int
    energy_samples::Int
    spin_max::Int
    quadrature_order::Int
    # The implemented diagnostic is a single-anchor symmetric triple-rho
    # orbit basis.  The 2605 multi-wavelet basis is metadata only.
    basis_kind::Symbol = :single_anchor_triple_rho
    basis_centers::Tuple = PAPER_DEFAULTS.basis_centers
    nmax_values::Tuple{Vararg{Int}} = PAPER_DEFAULTS.nmax_values
    lmax_values::Tuple{Vararg{Int}} = PAPER_DEFAULTS.lmax_values
    s_max::T = T(PAPER_DEFAULTS.s_max)
    alpha_th::Symbol = PAPER_DEFAULTS.alpha_th
    a7_t_grid::Tuple = PAPER_DEFAULTS.a7_t_grid
    dual_mu2::T = T(PAPER_DEFAULTS.dual_mu2)
    dual_grid_count::Int = PAPER_DEFAULTS.dual_grid_count
    dual_spin_max::Int = PAPER_DEFAULTS.dual_spin_max
    s_grid_min::Union{Nothing,T} = nothing
    s_grid_max::Union{Nothing,T} = nothing
    s_grid_count::Union{Nothing,Int} = nothing
    spin_set::Tuple{Vararg{Int}} = ()
    beta_min::T = T(1) / T(10)
    beta_max::T = T(9) / T(10)
    source::Symbol = :primary_paper
    source_version::String = PRIMARY_VERSION
    formulation::Symbol = :primal_full_unitarity
    witness_mode::Symbol = :none
    scaling::Symbol = :tiny
    rho_map_s0::Union{Nothing,T} = nothing
    objective::Symbol = :none
    objective_min::Union{Nothing,T} = nothing
    objective_max::Union{Nothing,T} = nothing
    reference_status::Symbol = SOURCE_STATUS
    paper_equivalent::Bool = false
    # This flag is derived/checked against the immutable paper-default table;
    # callers cannot claim that custom diagnostic tuples match the paper.
    matches_paper_defaults::Bool = true
end

struct SMatrix4DArtifact{T}
    schema_version::Int
    spec::SMatrix4DSpec{T}
    betas::Vector{T}
    # In this normalization beta is the two-body phase-space factor rho(s).
    phase_space::Vector{T}
    energies_squared::Vector{T}
    quadrature_nodes::Vector{T}
    quadrature_weights::Vector{T}
    basis_indices::Vector{NTuple{3,Int}}
    projection_real::Matrix{T}
    projection_imag::Matrix{T}
    strict_witness::Vector{T}
    zero_witness::Vector{T}
    strict_margins::Vector{T}
    witness_certified::Bool
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

function _matches_paper_defaults(spec::SMatrix4DSpec)
    return spec.basis_centers == PAPER_DEFAULTS.basis_centers &&
           spec.nmax_values == PAPER_DEFAULTS.nmax_values &&
           spec.lmax_values == PAPER_DEFAULTS.lmax_values &&
           spec.s_max == typeof(spec.s_max)(PAPER_DEFAULTS.s_max) &&
           spec.alpha_th == PAPER_DEFAULTS.alpha_th &&
           spec.a7_t_grid == PAPER_DEFAULTS.a7_t_grid &&
           spec.dual_mu2 == typeof(spec.dual_mu2)(PAPER_DEFAULTS.dual_mu2) &&
           spec.dual_grid_count == PAPER_DEFAULTS.dual_grid_count &&
           spec.dual_spin_max == PAPER_DEFAULTS.dual_spin_max &&
           spec.beta_min == typeof(spec.beta_min)(1 / 10) &&
           spec.beta_max == typeof(spec.beta_max)(9 / 10) &&
           spec.rho_map_s0 === nothing
end

function _validate_spec(spec::SMatrix4DSpec)
    spec.source === :primary_paper || throw(ArgumentError("source must be :primary_paper"))
    spec.source_version == PRIMARY_VERSION || throw(ArgumentError(
        "source_version must be $PRIMARY_VERSION",
    ))
    spec.reference_status === SOURCE_STATUS || throw(ArgumentError(
        "reference_status must be :sampled_build_only",
    ))
    spec.paper_equivalent && throw(ArgumentError(
        "finite sampling is not a continuous-domain proof",
    ))
    spec.external_mass > zero(spec.external_mass) || throw(ArgumentError(
        "external_mass must be positive",
    ))
    0 <= spec.ansatz_degree <= 64 || throw(ArgumentError("ansatz_degree out of range"))
    spec.energy_samples >= 1 || throw(ArgumentError("energy_samples must be positive"))
    spec.basis_kind === :single_anchor_triple_rho || throw(ArgumentError(
        "only basis_kind=:single_anchor_triple_rho is implemented",
    ))
    spec.matches_paper_defaults == _matches_paper_defaults(spec) || throw(ArgumentError(
        "matches_paper_defaults must agree with the immutable paper-default table",
    ))
    !isempty(spec.basis_centers) && all(isfinite, spec.basis_centers) &&
        all(>(4), spec.basis_centers) || throw(ArgumentError(
            "basis_centers must be nonempty finite centers above threshold in m^2 units",
        ))
    all(>(0), spec.nmax_values) && issorted(spec.nmax_values) &&
        length(unique(spec.nmax_values)) == length(spec.nmax_values) ||
        throw(ArgumentError("nmax_values must be sorted and unique positive values"))
    all(>(0), spec.lmax_values) && all(iseven, spec.lmax_values) &&
        issorted(spec.lmax_values) && length(unique(spec.lmax_values)) == length(spec.lmax_values) ||
        throw(ArgumentError("lmax_values must be sorted and unique positive even values"))
    spec.ansatz_degree <= maximum(spec.nmax_values) || throw(ArgumentError(
        "ansatz_degree exceeds the declared nmax_values range",
    ))
    spec.spin_max <= maximum(spec.lmax_values) || throw(ArgumentError(
        "spin_max exceeds the declared lmax_values range",
    ))
    spec.s_max > 4 * spec.external_mass^2 || throw(ArgumentError(
        "s_max must be above the two-particle threshold",
    ))
    spec.alpha_th in (:zero, :max) || throw(ArgumentError(
        "alpha_th must be one of the discrete paper branches :zero or :max",
    ))
    !isempty(spec.a7_t_grid) && all(isfinite, spec.a7_t_grid) ||
        throw(ArgumentError("A7 t-grid must be finite and nonempty"))
    4 * spec.external_mass^2 < spec.dual_mu2 < spec.s_max ||
        throw(ArgumentError("dual_mu2 must lie above threshold and below s_max"))
    spec.dual_grid_count >= 1 || throw(ArgumentError("dual_grid_count must be positive"))
    spec.dual_spin_max >= 0 && iseven(spec.dual_spin_max) ||
        throw(ArgumentError("dual_spin_max must be nonnegative and even"))
    spec.objective === :none || throw(ArgumentError(
        "objectives are not implemented in this spec-only builder",
    ))
    spec.objective_min === nothing && spec.objective_max === nothing ||
        throw(ArgumentError("objective bounds require a real implemented objective"))
    spec.formulation in (:primal_full_unitarity, :dual_linearized, :finite_conic_dual) ||
        throw(ArgumentError("unsupported formulation"))
    spec.witness_mode in (:none, :free_boundary, :max_margin) || throw(ArgumentError("unsupported witness_mode"))
    spec.scaling in (:tiny, :small, :medium, :stress) || throw(ArgumentError("unsupported scaling tier"))
    (isempty(spec.spin_set) || (all(iseven, spec.spin_set) && all(>=(0), spec.spin_set) &&
      issorted(spec.spin_set) && length(unique(spec.spin_set)) == length(spec.spin_set) &&
      maximum(spec.spin_set) <= spec.spin_max)) ||
        throw(ArgumentError(
            "spin_set must be sorted, unique, nonnegative even, and <= spin_max",
        ))
    (spec.s_grid_min === nothing) == (spec.s_grid_max === nothing) ||
        throw(ArgumentError("s_grid_min and s_grid_max must be supplied together"))
    if spec.s_grid_min !== nothing
        spec.s_grid_min > 4 * spec.external_mass^2 || throw(ArgumentError("s_grid_min must be above threshold"))
        spec.s_grid_max >= spec.s_grid_min || throw(ArgumentError("invalid s grid"))
        spec.s_grid_count === nothing || spec.s_grid_count == spec.energy_samples ||
            throw(ArgumentError("s_grid_count must equal energy_samples"))
    end
    iseven(spec.spin_max) && spec.spin_max >= 0 ||
        throw(ArgumentError("spin_max must be a nonnegative even integer"))
    effective_spin_max = isempty(spec.spin_set) ? spec.spin_max : maximum(spec.spin_set)
    spec.quadrature_order >= max(4, spec.ansatz_degree + effective_spin_max + 2) ||
        throw(ArgumentError("quadrature_order is too small for the declared truncation"))
    zero(spec.beta_min) < spec.beta_min < spec.beta_max < one(spec.beta_max) ||
        throw(ArgumentError("beta interval must lie strictly inside (0,1)"))
    sstar = spec.rho_map_s0 === nothing ?
        S_STAR_FACTOR * spec.external_mass^2 : spec.rho_map_s0
    zero(sstar) < sstar < 4 * spec.external_mass^2 || throw(ArgumentError(
        "rho_map_s0 must satisfy 0 < s_star < 4m^2",
    ))
    return nothing
end

function smatrix_4d_specs(::Type{T}=Float64) where {T}
    return (
        tiny=SMatrix4DSpec{T}(id="paulos19/4d_tiny", scale=:tiny,
            ansatz_degree=4, energy_samples=8, spin_max=4, quadrature_order=32),
        small=SMatrix4DSpec{T}(id="paulos19/4d_small", scale=:small,
            ansatz_degree=8, energy_samples=24, spin_max=8, quadrature_order=64),
        medium=SMatrix4DSpec{T}(id="paulos19/4d_medium", scale=:medium,
            ansatz_degree=12, energy_samples=64, spin_max=16, quadrature_order=128),
        stress=SMatrix4DSpec{T}(id="paulos19/4d_stress", scale=:stress,
            ansatz_degree=16, energy_samples=192, spin_max=32, quadrature_order=256),
    )
end

function _basis_indices(degree::Int)
    # One coefficient per canonical S_3 orbit; repeated exponents therefore
    # never duplicate a monomial in the crossing-symmetric ansatz.
    result = NTuple{3,Int}[]
    for a in degree:-1:0
        for b in min(a, degree - a):-1:0
            for c in min(b, degree - a - b):-1:0
                push!(result, (a, b, c))
            end
        end
    end
    return result
end


# Deterministic Newton construction. Nodes/weights are generated at fixed
# BigFloat precision, then converted to T; no platform LAPACK eigensolver is
# involved in the frozen artifact.
function _gauss_legendre(n::Int)
    setprecision(BigFloat, 256) do
        nodes = zeros(BigFloat, n)
        weights = zeros(BigFloat, n)
        half = (n + 1) ÷ 2
        for i in 1:half
            x = cos(BigFloat(pi) * (BigFloat(i) - BigFloat("0.25")) /
                    (BigFloat(n) + BigFloat("0.5")))
            for _ in 1:64
                p0 = one(BigFloat)
                p1 = x
                for k in 2:n
                    p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                    p0, p1 = p1, p2
                end
                dp = n * (x * p1 - p0) / (x * x - 1)
                delta = p1 / dp
                x -= delta
                abs(delta) < eps(BigFloat) && break
            end
            p0 = one(BigFloat)
            p1 = x
            for k in 2:n
                p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                p0, p1 = p1, p2
            end
            dp = n * (x * p1 - p0) / (x * x - 1)
            w = 2 / ((1 - x * x) * dp * dp)
            nodes[i] = -x
            nodes[n + 1 - i] = x
            weights[i] = w
            weights[n + 1 - i] = w
        end
        return nodes, weights
    end
end

@inline function _legendre(l::Int, z::T) where {T}
    l == 0 && return one(T)
    l == 1 && return z
    p0, p1 = one(T), z
    for k in 2:l
        p0, p1 = p1, ((T(2k - 1) * z * p1) - T(k - 1) * p0) / T(k)
    end
    return p1
end

@inline function rho_coordinate(
    x::T, mass2::T, s_star::T; rim::Symbol=:upper,
) where {T<:Real}
    threshold = T(4) * mass2
    zero(s_star) < s_star < threshold || throw(ArgumentError(
        "s_star must satisfy 0 < s_star < 4m^2",
    ))
    rim in (:upper, :lower) || throw(ArgumentError(
        "rim must be :upper or :lower",
    ))
    root = if x > threshold
        sign = rim === :upper ? -one(T) : one(T)
        complex(zero(T), sign * sqrt(x - threshold))
    else
        sqrt(complex(threshold - x, zero(T)))
    end
    anchor = sqrt(complex(threshold - s_star, zero(T)))
    return (anchor - root) / (anchor + root)
end

@inline _rho(x::T, mass2::T, s_star::T) where {T<:Real} =
    rho_coordinate(x, mass2, s_star; rim=:upper)

function _orbit_permutations(index::NTuple{3,Int})
    a, b, c = index
    a == b == c && return ((a, b, c),)
    a == b && return ((a, b, c), (a, c, b), (c, a, b))
    b == c && return ((a, b, c), (b, a, c), (b, c, a))
    a == c && return ((a, b, c), (b, a, c), (a, c, b))
    return ((a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a))
end

function _orbit_value(rhos, index::NTuple{3,Int})
    total = zero(eltype(rhos))
    for (i, j, k) in _orbit_permutations(index)
        total += rhos[1]^i * rhos[2]^j * rhos[3]^k
    end
    return total
end

@inline function alpha_threshold_value(spec::SMatrix4DSpec{T}) where {T}
    spec.alpha_th === :zero && return zero(T)
    return T(64) * T(pi) / sqrt(T(2) / T(3))
end

function _provenance(spec::SMatrix4DSpec)
    return (
        title="The Phases of the Scalar S-Matrix Island",
        authors=("Joan Elias Miro", "Andrea Guerrieri", "Mehmet Asim Gumus"),
        arxiv=PRIMARY_ARXIV,
        source_version=PRIMARY_VERSION,
        reference_status=SOURCE_STATUS,
        predecessor_references=PREDECESSOR_REFERENCES,
        formula_oracle_status=:verified_conventions_only,
        formulation_source=:miro_guerrieri_gumus_2605_06613v1,
        equations=(ansatz="single-anchor triple-rho diagnostic; 2605 A2-A4 multi-wavelet is provenance-only",
            partial_wave="A1", unitarity="A5-A6", primal="A1 plus sampled A5-A6",
            fixed_t_dual="A8-A15_placeholder"),
        unitarity_level=:sampled_partial_wave,
        duality_relation=:finite_conic_dual_distinct_from_paper_linearized_dual,
        basis_implementation=:single_anchor_triple_rho_diagnostic,
        paper_defaults=PAPER_DEFAULTS,
        matches_paper_defaults=_matches_paper_defaults(spec),
        primal_analyticity_assumption=:maximal_landau,
        dual_analyticity_assumption=:axiomatic_fixed_t,
        paper_equivalent=false,
        dimension=4,
        implemented_equations=(
            "s+t+u=4m^2 and COM t,u kinematics",
            "rho_s,t,u uniformizer with explicit s_star anchor (A3)",
            "single-anchor fully symmetric triple-rho orbit basis (implemented)",
            "M=16pi sum_l (2l+1) f_l P_l",
            "f_l=(32pi)^-1 integral P_l M (A1)",
            "S_l=1+i*sqrt((s-4m^2)/s)*f_l and A5-A6 disk",
            "2605 A2-A4 multi-wavelet centers/truncations and per-center N_sigma rule are provenance-only",
            "A7 subtracted-positivity t-grid is provenance-only and not imposed",
            "A8-A15 fixed-t dual parameters are recorded but not lowered here",
            "primal objective/maximization is not implemented; objective=:none only",
            "sampled |S_l|<=1 for even l",
        ),
        excluded_claims=(
            "finite energy/spin/quadrature sampling is not continuous unitarity",
            "no paper numerical bound, pole spectrum, or asymptotic theorem is claimed",
            "no 4D coupled-channel resonance result is claimed",
            "arXiv:2605.06613 is an S-matrix reference only and is unrelated to CSDR",
        ),
    )
end

function build_smatrix_4d(spec::SMatrix4DSpec{T}) where {T}
    _validate_spec(spec)
    spec.formulation === :primal_full_unitarity || throw(ArgumentError(
        "$(spec.formulation) is parameterized but not implemented in SPEC-ONLY builder; " *
        "do not conflate paper Section 4 linearized dual with finite conic dual",
    ))
    mass2 = spec.external_mass * spec.external_mass
    s_star = spec.rho_map_s0 === nothing ? T(4) * mass2 / T(3) : spec.rho_map_s0
    betas = collect(range(spec.beta_min, spec.beta_max; length=spec.energy_samples))
    energies = if spec.s_grid_min === nothing
        T[4 * mass2 / (one(T) - beta^2) for beta in betas]
    else
        collect(range(spec.s_grid_min, spec.s_grid_max; length=spec.energy_samples))
    end
    betas = T[sqrt(one(T) - 4 * mass2 / energy) for energy in energies]
    nodes_big, weights_big = _gauss_legendre(spec.quadrature_order)
    nodes = T.(nodes_big)
    weights = T.(weights_big)
    basis = _basis_indices(spec.ansatz_degree)
    spins = isempty(spec.spin_set) ? collect(0:2:spec.spin_max) : collect(spec.spin_set)
    effective_spin_max = isempty(spins) ? 0 : maximum(spins)
    rows = spec.energy_samples * length(spins)
    pr = zeros(T, rows, length(basis))
    pim = zeros(T, rows, length(basis))
    row = 0
    for (energy, beta) in zip(energies, betas)
        for spin in spins
            row += 1
            for q in eachindex(nodes)
                z = nodes[q]
                t = -(energy - 4 * mass2) * (one(T) - z) / 2
                u = -(energy - 4 * mass2) * (one(T) + z) / 2
                rhos = (_rho(energy, mass2, s_star), _rho(t, mass2, s_star), _rho(u, mass2, s_star))
                leg = _legendre(spin, z)
                for col in eachindex(basis)
                    value = _orbit_value(rhos, basis[col])
                    factor = weights[q] * leg / (T(32) * T(pi))
                    pr[row, col] += factor * real(value)
                    pim[row, col] += factor * imag(value)
                end
            end
        end
    end
    # Diagnostic mode records failed strict-witness searches instead of
    # constructing a certified artifact. M=0 is a boundary witness only.
    candidate = pim \ ones(T, rows)
    im_candidate = pim * candidate
    re_candidate = pr * candidate
    strict = zeros(T, length(basis))
    if all(>(zero(T)), im_candidate)
        scale = one(T)
        for i in eachindex(im_candidate)
            beta = betas[1 + (i - 1) ÷ length(spins)]
            denominator = beta * (re_candidate[i]^2 + im_candidate[i]^2)
            denominator > zero(T) && (scale = min(scale, T(2) * im_candidate[i] / denominator))
        end
        scale > zero(T) && (strict .= (scale / T(2)) .* candidate)
    end
    zero_coefficients = zeros(T, length(basis))
    margins = cone_margins_from_projection(betas, pr, pim, strict)
    expected_width = Dict(4 => 11, 8 => 41, 12 => 102, 16 => 204)
    if haskey(expected_width, spec.ansatz_degree)
        length(basis) == expected_width[spec.ansatz_degree] ||
            throw(ArgumentError("canonical basis width mismatch"))
    end
    counts = (variables=length(basis), basis_width=length(basis), ansatz_degree=spec.ansatz_degree,
        energy_samples=length(betas), even_spins=length(spins), spin_max=effective_spin_max,
        spin_set=Tuple(spins), quadrature_order=spec.quadrature_order,
        basis_centers=spec.basis_centers, nmax_values=spec.nmax_values,
        lmax_values=spec.lmax_values, s_max=spec.s_max,
        alpha_th=spec.alpha_th, alpha_th_value=alpha_threshold_value(spec),
        a7_t_grid=spec.a7_t_grid, dual_mu2=spec.dual_mu2,
        dual_grid_count=spec.dual_grid_count, dual_spin_max=spec.dual_spin_max,
        lorentz_cones=rows, cone_dimension=3,
        dimension=4, crossing=:structural_symmetric_orbit,
        projection=:frozen_gauss_legendre, unitarity=:sampled_partial_wave_disk,
        paper_equivalent=false)
    provisional = SMatrix4DArtifact(ARTIFACT_SCHEMA_VERSION, spec, betas, copy(betas), energies,
        nodes, weights, basis, pr, pim, strict, zero_coefficients, margins,
        false, _provenance(spec), counts, "")
    return SMatrix4DArtifact(provisional.schema_version, provisional.spec,
        provisional.betas, provisional.phase_space, provisional.energies_squared,
        provisional.quadrature_nodes,
        provisional.quadrature_weights, provisional.basis_indices,
        provisional.projection_real, provisional.projection_imag,
        provisional.strict_witness, provisional.zero_witness, provisional.strict_margins,
        provisional.witness_certified, provisional.provenance, provisional.counts,
        stable_fingerprint(provisional))
end

build_smatrix_4d(scale::Symbol, ::Type{T}=Float64) where {T} =
    build_smatrix_4d(getproperty(smatrix_4d_specs(T), scale))

function evaluate_amplitude(artifact::SMatrix4DArtifact{T}, coefficients::AbstractVector,
                           s::T, t::T, u::T) where {T}
    length(coefficients) == length(artifact.basis_indices) || throw(DimensionMismatch())
    s_star = artifact.spec.rho_map_s0 === nothing ?
        T(4) * artifact.spec.external_mass^2 / T(3) : artifact.spec.rho_map_s0
    rhos = (_rho(s, artifact.spec.external_mass^2, s_star),
        _rho(t, artifact.spec.external_mass^2, s_star),
        _rho(u, artifact.spec.external_mass^2, s_star))
    value = zero(Complex{T})
    for (index, coefficient) in zip(artifact.basis_indices, coefficients)
        value += T(coefficient) * _orbit_value(rhos, index)
    end
    return value
end

evaluate_amplitude(artifact::SMatrix4DArtifact{T}, coefficients::AbstractVector,
                   s::T, z::T) where {T} = begin
    mass2 = artifact.spec.external_mass^2
    t = -(s - 4 * mass2) * (one(T) - z) / 2
    u = -(s - 4 * mass2) * (one(T) + z) / 2
    evaluate_amplitude(artifact, coefficients, s, t, u)
end

function partial_wave_matrices(artifact::SMatrix4DArtifact)
    return (real=artifact.projection_real, imag=artifact.projection_imag)
end

function partial_wave_values(artifact::SMatrix4DArtifact{T}, coefficients::AbstractVector) where {T}
    length(coefficients) == size(artifact.projection_real, 2) || throw(DimensionMismatch())
    return (real=artifact.projection_real * T.(coefficients),
        imag=artifact.projection_imag * T.(coefficients))
end

function cone_margins_from_projection(betas, pr, pi, coefficients)
    values_r = pr * coefficients
    values_i = pi * coefficients
    result = similar(values_r)
    for i in eachindex(result)
        # S=1+i beta f, so Re(S)=1-beta*Im(f), Im(S)=beta*Re(f).
        beta = betas[1 + (i - 1) ÷ (size(pr, 1) ÷ length(betas))]
        x = one(eltype(result)) - beta * values_i[i]
        y = beta * values_r[i]
        result[i] = one(eltype(result)) - x*x - y*y
    end
    return result
end

function cone_margins(artifact::SMatrix4DArtifact{T}, coefficients::AbstractVector) where {T}
    return cone_margins_from_projection(artifact.betas, artifact.projection_real,
        artifact.projection_imag, T.(coefficients))
end

function build_soc_problem(artifact::SMatrix4DArtifact{T}) where {T}
    verdict = validate_artifact(artifact)
    verdict.valid || throw(ArgumentError("invalid 4D artifact: $(join(verdict.failures, ", "))"))
    variables = length(artifact.basis_indices)
    cones = SDPX.SOCConstraint{T}[]
    for row in 1:size(artifact.projection_real, 1)
        matrix = zeros(T, 3, variables)
        # [1, Re S, Im S] in Q3, with S=1+i beta f.
        beta = artifact.betas[1 + (row - 1) ÷ (size(artifact.projection_real, 1) ÷ length(artifact.betas))]
        matrix[2, :] .= -beta .* artifact.projection_imag[row, :]
        matrix[3, :] .= beta .* artifact.projection_real[row, :]
        # [1, 1-beta*Im(f), beta*Re(f)] in Q3.
        push!(cones, SDPX.SOCConstraint(matrix, T[one(T), one(T), zero(T)]; T))
    end
    return SDPX.second_order_program(zeros(T, variables), cones; T)
end

"""Return the exact real 2-by-2 PSD affine block for every sampled disk.
This is a representation only; this SPEC-ONLY package does not solve it."""
function build_sdp_blocks(artifact::SMatrix4DArtifact{T}) where {T}
    blocks = Array{T,3}[]
    constants = Matrix{T}[]
    rows = size(artifact.projection_real, 1)
    spins_per_energy = rows ÷ length(artifact.betas)
    for row in 1:rows
        beta = artifact.betas[1 + (row - 1) ÷ spins_per_energy]
        x = -beta .* vec(artifact.projection_imag[row, :])
        y = beta .* vec(artifact.projection_real[row, :])
        # [1+x,y; y,1-x] >= 0, represented as affine constant plus rows.
        # With x_delta=-beta*Im(f), Re(S)=1+x_delta.  Thus the
        # realification [[1+Re(S), Im(S)],[Im(S),1-Re(S)]] has constant
        # [[2,0],[0,0]] at the free boundary S=1.
        push!(constants, T[2 0; 0 0])
        block = zeros(T, 2, 2, length(artifact.basis_indices))
        for j in axes(block, 3)
            block[1, 1, j] = x[j]
            block[1, 2, j] = y[j]
            block[2, 1, j] = y[j]
            block[2, 2, j] = -x[j]
        end
        push!(blocks, block)
    end
    return (constants=constants, coefficients=blocks)
end

"""Build the sampled max-margin SOCP without inventing independent S variables.
The extra variable is delta and every disk uses the same affine ansatz rows."""
function require_strict_witness(artifact::SMatrix4DArtifact{T}) where {T}
    artifact.witness_certified || throw(ArgumentError(
        "strict witness is not independently certified; status is a blocker",
    ))
    floor = T(WITNESS_MARGIN_FLOOR_FACTOR) * sqrt(eps(T))
    minimum(artifact.strict_margins) > floor || throw(ArgumentError(
        "strict witness margin is not robustly above machine precision",
    ))
    return artifact.strict_witness
end

function build_max_margin_problem(artifact::SMatrix4DArtifact{T}) where {T}
    variables = length(artifact.basis_indices)
    cones = SDPX.SOCConstraint{T}[]
    rows = size(artifact.projection_real, 1)
    spins_per_energy = rows ÷ length(artifact.betas)
    for row in 1:rows
        beta = artifact.betas[1 + (row - 1) ÷ spins_per_energy]
        matrix = zeros(T, 3, variables + 1)
        matrix[1, end] = -one(T)
        matrix[2, 1:variables] .= -beta .* artifact.projection_imag[row, :]
        matrix[3, 1:variables] .= beta .* artifact.projection_real[row, :]
        push!(cones, SDPX.SOCConstraint(matrix, T[one(T), one(T), zero(T)]; T))
    end
    lower = zeros(T, 1, variables + 1)
    lower[1, end] = one(T)
    upper = zeros(T, 1, variables + 1)
    upper[1, end] = -one(T)
    push!(cones, SDPX.SOCConstraint(lower, T[zero(T)]; T))
    push!(cones, SDPX.SOCConstraint(upper, T[one(T)]; T))
    objective = zeros(T, variables + 1)
    objective[end] = -one(T)
    return SDPX.second_order_program(objective, cones; T)
end

function canonical_text(artifact::SMatrix4DArtifact)
    # Every persisted field participates in the identity.  In particular this
    # includes derived grids, phase space, witnesses, provenance and counts;
    # changing any field cannot silently preserve an artifact fingerprint.
    spec = artifact.spec
    payload = (
        schema_version=artifact.schema_version,
        spec=(id=spec.id, scale=spec.scale, external_mass=spec.external_mass,
            ansatz_degree=spec.ansatz_degree, energy_samples=spec.energy_samples,
            spin_max=spec.spin_max, quadrature_order=spec.quadrature_order,
            basis_kind=spec.basis_kind, basis_centers=spec.basis_centers,
            nmax_values=spec.nmax_values, lmax_values=spec.lmax_values,
            s_max=spec.s_max, alpha_th=spec.alpha_th, a7_t_grid=spec.a7_t_grid,
            dual_mu2=spec.dual_mu2, dual_grid_count=spec.dual_grid_count,
            dual_spin_max=spec.dual_spin_max, s_grid_min=spec.s_grid_min,
            s_grid_max=spec.s_grid_max, s_grid_count=spec.s_grid_count,
            spin_set=spec.spin_set, beta_min=spec.beta_min, beta_max=spec.beta_max,
            source=spec.source, source_version=spec.source_version,
            formulation=spec.formulation, witness_mode=spec.witness_mode,
            scaling=spec.scaling, rho_map_s0=spec.rho_map_s0,
            objective=spec.objective, objective_min=spec.objective_min,
            objective_max=spec.objective_max, reference_status=spec.reference_status,
            paper_equivalent=spec.paper_equivalent,
            matches_paper_defaults=spec.matches_paper_defaults),
        betas=artifact.betas, phase_space=artifact.phase_space,
        energies_squared=artifact.energies_squared,
        quadrature_nodes=artifact.quadrature_nodes,
        quadrature_weights=artifact.quadrature_weights,
        basis_indices=artifact.basis_indices,
        projection_real=artifact.projection_real,
        projection_imag=artifact.projection_imag,
        strict_witness=artifact.strict_witness,
        zero_witness=artifact.zero_witness,
        strict_margins=artifact.strict_margins,
        witness_certified=artifact.witness_certified,
        provenance=artifact.provenance, counts=artifact.counts,
    )
    return repr(payload)
end

stable_fingerprint(artifact::SMatrix4DArtifact) =
    bytes2hex(sha256(codeunits(canonical_text(artifact))))

function validate_artifact(artifact::SMatrix4DArtifact{T}) where {T}
    failures = String[]
    try
        _validate_spec(artifact.spec)
        spec = artifact.spec
        artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures, "schema_version")
        expected_basis = _basis_indices(spec.ansatz_degree)
        artifact.basis_indices == expected_basis || push!(failures, "basis_rebuild")
        expected_spins = isempty(spec.spin_set) ? collect(0:2:spec.spin_max) : collect(spec.spin_set)
        expected_rows = spec.energy_samples * length(expected_spins)
        artifact.counts.dimension == 4 || push!(failures, "dimension")
        artifact.counts.basis_width == length(expected_basis) || push!(failures, "basis_width")
        artifact.counts.variables == length(expected_basis) || push!(failures, "variables")
        artifact.counts.lorentz_cones == expected_rows || push!(failures, "cone_count")
        size(artifact.projection_real) == (expected_rows, length(expected_basis)) ||
            push!(failures, "projection_shape")
        size(artifact.projection_imag) == size(artifact.projection_real) ||
            push!(failures, "projection_imag_shape")
        artifact.counts.spin_set == Tuple(expected_spins) || push!(failures, "spin_set_count")
        (length(artifact.betas), length(artifact.phase_space), length(artifact.energies_squared)) ==
            (spec.energy_samples, spec.energy_samples, spec.energy_samples) ||
            push!(failures, "grid_lengths")
        artifact.betas == artifact.phase_space || push!(failures, "phase_space")
        all(isfinite, artifact.projection_real) && all(isfinite, artifact.projection_imag) ||
            push!(failures, "finite_projection")
        all(isfinite, artifact.betas) && all(beta -> zero(T) < beta < one(T), artifact.betas) ||
            push!(failures, "beta_domain")
        mass2 = spec.external_mass^2
        expected_energies = T[4 * mass2 / (one(T) - beta^2) for beta in artifact.betas]
        all(isapprox.(artifact.energies_squared, expected_energies;
            atol=100eps(T), rtol=100eps(T))) || push!(failures, "energy_beta")
        for energy in artifact.energies_squared
            energy > 4 * mass2 || push!(failures, "threshold")
        end
        for (energy, z) in Iterators.product(artifact.energies_squared, artifact.quadrature_nodes)
            t = -(energy - 4 * mass2) * (one(T) - z) / 2
            u = -(energy - 4 * mass2) * (one(T) + z) / 2
            isapprox(energy + t + u, 4 * mass2; atol=100eps(T), rtol=100eps(T)) ||
                push!(failures, "mandelstam_sum")
        end
        # Compare every derived array against a deterministic rebuild, including
        # projection matrices and the diagnostic witness/margin vectors.
        rebuilt = build_smatrix_4d(spec)
        artifact.provenance == rebuilt.provenance || push!(failures, "provenance_rebuild")
        artifact.counts == rebuilt.counts || push!(failures, "counts_rebuild")
        for (name, actual, expected) in (("betas", artifact.betas, rebuilt.betas),
            ("phase_space", artifact.phase_space, rebuilt.phase_space),
            ("energies_squared", artifact.energies_squared, rebuilt.energies_squared),
            ("quadrature_nodes", artifact.quadrature_nodes, rebuilt.quadrature_nodes),
            ("quadrature_weights", artifact.quadrature_weights, rebuilt.quadrature_weights),
            ("basis_indices", artifact.basis_indices, rebuilt.basis_indices),
            ("projection_real", artifact.projection_real, rebuilt.projection_real),
            ("projection_imag", artifact.projection_imag, rebuilt.projection_imag),
            ("strict_witness", artifact.strict_witness, rebuilt.strict_witness),
            ("zero_witness", artifact.zero_witness, rebuilt.zero_witness),
            ("strict_margins", artifact.strict_margins, rebuilt.strict_margins))
            actual == expected || push!(failures, name * "_rebuild")
        end
        all(iszero, artifact.zero_witness) || push!(failures, "zero_witness")
        all(isapprox.(cone_margins(artifact, artifact.zero_witness), zero(T);
            atol=100eps(T), rtol=100eps(T))) || push!(failures, "zero_boundary")
        artifact.witness_certified == false || push!(failures, "uncertified_witness_flag")
        stable_fingerprint(artifact) == artifact.fingerprint || push!(failures, "fingerprint")
    catch err
        push!(failures, "semantic_rebuild:" * string(typeof(err)))
    end
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

end # module SMatrix4DSpecDiagnostic
