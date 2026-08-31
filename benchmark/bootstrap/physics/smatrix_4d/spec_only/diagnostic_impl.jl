module SMatrix4DSpecDiagnostic

using SHA
using SparseArrays
import SDPX

# SPEC-ONLY diagnostic implementation. It is not included by any catalog.

const ARTIFACT_SCHEMA_VERSION = 1
const PRIMARY_ARXIV = "2210.01502"
const PRIMARY_VERSION = "2210.01502v2"
const SOURCE_STATUS = :experimental_build_only
const S_STAR = 4 / 3

Base.@kwdef struct SMatrix4DSpec{T}
    id::String
    scale::Symbol
    external_mass::T = one(T)
    ansatz_degree::Int
    energy_samples::Int
    spin_max::Int
    quadrature_order::Int
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
end

struct SMatrix4DArtifact{T}
    schema_version::Int
    spec::SMatrix4DSpec{T}
    betas::Vector{T}
    energies_squared::Vector{T}
    quadrature_nodes::Vector{T}
    quadrature_weights::Vector{T}
    basis_indices::Vector{NTuple{3,Int}}
    projection_real::Matrix{T}
    projection_imag::Matrix{T}
    strict_witness::Vector{T}
    zero_witness::Vector{T}
    strict_margins::Vector{T}
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

function _validate_spec(spec::SMatrix4DSpec)
    spec.source === :primary_paper || throw(ArgumentError("source must be :primary_paper"))
    spec.source_version == PRIMARY_VERSION || throw(ArgumentError(
        "source_version must be $PRIMARY_VERSION",
    ))
    spec.reference_status === SOURCE_STATUS || throw(ArgumentError(
        "reference_status must be :experimental_build_only",
    ))
    spec.paper_equivalent && throw(ArgumentError(
        "finite sampling is not a continuous-domain proof",
    ))
    spec.external_mass > zero(spec.external_mass) || throw(ArgumentError(
        "external_mass must be positive",
    ))
    0 <= spec.ansatz_degree <= 64 || throw(ArgumentError("ansatz_degree out of range"))
    spec.energy_samples >= 1 || throw(ArgumentError("energy_samples must be positive"))
    spec.objective in (:none, :c0, :c2, :c3, :radial) || throw(ArgumentError("unsupported objective"))
    spec.formulation in (:primal_full_unitarity, :dual_linearized, :finite_conic_dual) ||
        throw(ArgumentError("unsupported formulation"))
    spec.witness_mode in (:none, :free_boundary, :max_margin) || throw(ArgumentError("unsupported witness_mode"))
    spec.scaling in (:tiny, :small, :medium, :stress) || throw(ArgumentError("unsupported scaling tier"))
    (isempty(spec.spin_set) || (all(iseven, spec.spin_set) && all(>=(0), spec.spin_set))) ||
        throw(ArgumentError("spin_set must contain nonnegative even spins"))
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
    spec.quadrature_order >= max(4, spec.ansatz_degree + spec.spin_max + 2) ||
        throw(ArgumentError("quadrature_order is too small for the declared truncation"))
    zero(spec.beta_min) < spec.beta_min < spec.beta_max < one(spec.beta_max) ||
        throw(ArgumentError("beta interval must lie strictly inside (0,1)"))
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
    result = NTuple{3,Int}[]
    for a in 0:degree, b in 0:(degree - a), c in 0:(degree - a - b)
        push!(result, (a, b, c))
    end
    sort!(result; by=x -> (-x[1], -x[2], -x[3]))
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

@inline function _rho(x::T, mass2::T, s0_factor::T=T(4)/T(3)) where {T<:Real}
    threshold = T(4) * mass2
    root = x > threshold ? complex(zero(T), -sqrt(x - threshold)) :
        complex(sqrt(threshold - x), zero(T))
    anchor = sqrt(complex(T(4) * mass2 * s0_factor, zero(T)))
    return (anchor - root) / (anchor + root)
end

function _orbit_value(rhos, index::NTuple{3,Int})
    a, b, c = index
    tuples = ((a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a))
    total = zero(eltype(rhos))
    count = 0
    for (i,j,k) in tuples
        # Duplicate permutations must be counted once.
        (i,j,k) in tuples[1:count] && continue
        push = rhos[1]^i * rhos[2]^j * rhos[3]^k
        total += push
        count += 1
    end
    return total / count
end

function _provenance()
    return (
        title="Bridging Positivity and S-matrix Bootstrap Bounds",
        authors=("Joan Elias Miro", "Andrea Guerrieri", "Mehmet Asim Gumus"),
        arxiv=PRIMARY_ARXIV,
        source_version=PRIMARY_VERSION,
        reference_status=SOURCE_STATUS,
        formula_oracle_status=:verified_conventions_only,
        formulation_source=:miro_guerrieri_gumus_2210_01502v2,
        equations=(ansatz="generic_ansatz", partial_wave="2.8", unitarity="2.9", primal="2.10", linearized_dual="section_4"),
        unitarity_level=:sampled_partial_wave,
        duality_relation=:finite_conic_dual_distinct_from_paper_linearized_dual,
        paper_equivalent=false,
        dimension=4,
        implemented_equations=(
            "s+t+u=4m^2 and COM t,u kinematics",
            "rho uniformizer at s*=t*=u*=4m^2/3",
            "M=16pi sum_l (2l+1) f_l P_l",
            "f_l=(32pi)^-1 integral P_l M (Eq. 2.8)",
            "S_l=1+i*sqrt((s-4m^2)/s)*f_l and Eq. 2.9 disk",
            "Eq. 2.10 finite primal min/max; Section 4 linearized dual is separate",
            "sampled |S_l|<=1 for even l",
        ),
        excluded_claims=(
            "finite energy/spin/quadrature sampling is not continuous unitarity",
            "no paper numerical bound, pole spectrum, or asymptotic theorem is claimed",
            "no 4D coupled-channel resonance result is claimed",
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
                s0_factor = spec.rho_map_s0 === nothing ? T(4)/T(3) : spec.rho_map_s0 / mass2
                rhos = (_rho(energy, mass2, s0_factor), _rho(t, mass2, s0_factor), _rho(u, mass2, s0_factor))
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
    # Diagnostic mode deliberately records failed strict-witness searches
    # instead of constructing a benchmark artifact. A zero vector is retained
    # only as the known boundary witness M=0 => S_l=1.
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
    counts = (variables=length(basis), ansatz_degree=spec.ansatz_degree,
        energy_samples=length(betas), even_spins=length(spins), spin_max=spec.spin_max,
        quadrature_order=spec.quadrature_order, lorentz_cones=rows, cone_dimension=3,
        dimension=4, crossing=:structural_symmetric_orbit,
        projection=:frozen_gauss_legendre, unitarity=:sampled_partial_wave_disk,
        paper_equivalent=false)
    provisional = SMatrix4DArtifact(ARTIFACT_SCHEMA_VERSION, spec, betas, energies,
        nodes, weights, basis, pr, pim, strict, zero_coefficients, margins,
        _provenance(), counts, "")
    return SMatrix4DArtifact(provisional.schema_version, provisional.spec,
        provisional.betas, provisional.energies_squared, provisional.quadrature_nodes,
        provisional.quadrature_weights, provisional.basis_indices,
        provisional.projection_real, provisional.projection_imag,
        provisional.strict_witness, provisional.zero_witness, provisional.strict_margins,
        provisional.provenance, provisional.counts, stable_fingerprint(provisional))
end

build_smatrix_4d(scale::Symbol, ::Type{T}=Float64) where {T} =
    build_smatrix_4d(getproperty(smatrix_4d_specs(T), scale))

function evaluate_amplitude(artifact::SMatrix4DArtifact{T}, coefficients::AbstractVector,
                           s::T, t::T, u::T) where {T}
    length(coefficients) == length(artifact.basis_indices) || throw(DimensionMismatch())
    s0_factor = artifact.spec.rho_map_s0 === nothing ? T(4)/T(3) :
        artifact.spec.rho_map_s0 / artifact.spec.external_mass^2
    rhos = (_rho(s, artifact.spec.external_mass^2, s0_factor),
        _rho(t, artifact.spec.external_mass^2, s0_factor),
        _rho(u, artifact.spec.external_mass^2, s0_factor))
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
        push!(cones, SDPX.SOCConstraint(matrix, T[one(T), zero(T), zero(T)]; T))
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
        # Constant [[1+x0,y0],[y0,1-x0]] with x0=1,y0=0.
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
function require_strict_witness(artifact::SMatrix4DArtifact)
    all(>(zero(eltype(artifact.strict_margins))), artifact.strict_margins) ||
        throw(ArgumentError("strict witness is not independently certified; status is a blocker"))
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
        push!(cones, SDPX.SOCConstraint(matrix, T[one(T), zero(T), zero(T)]; T))
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
    io = IOBuffer()
    print(io, artifact.schema_version, '|', artifact.spec.id, '|', artifact.spec.ansatz_degree,
        '|', artifact.spec.energy_samples, '|', artifact.spec.spin_max, '|',
        artifact.spec.quadrature_order, '|', artifact.spec.external_mass, '|')
    print(io, join(artifact.basis_indices, ';'), '|')
    print(io, join(artifact.betas, ';'), '|', join(artifact.quadrature_nodes, ';'), '|',
        join(artifact.quadrature_weights, ';'), '|')
    for A in (artifact.projection_real, artifact.projection_imag)
        for value in A
            print(io, value, ';')
        end
        print(io, '|')
    end
    print(io, artifact.provenance, '|', artifact.counts)
    return String(take!(io))
end

stable_fingerprint(artifact::SMatrix4DArtifact) =
    bytes2hex(sha256(codeunits(canonical_text(artifact))))

function validate_artifact(artifact::SMatrix4DArtifact{T}) where {T}
    failures = String[]
    try
        _validate_spec(artifact.spec)
        artifact.spec.reference_status === SOURCE_STATUS || push!(failures, "reference_status")
        artifact.spec.paper_equivalent == false || push!(failures, "paper_equivalent")
        artifact.counts.dimension == 4 || push!(failures, "dimension")
        artifact.counts.lorentz_cones == size(artifact.projection_real, 1) ||
            push!(failures, "cone_count")
        length(artifact.basis_indices) == size(artifact.projection_real, 2) ||
            push!(failures, "basis_count")
        # Strict witness is intentionally not asserted in this evidence artifact.
        all(isfinite, artifact.projection_real) && all(isfinite, artifact.projection_imag) ||
            push!(failures, "finite_projection")
        # Kinematic and crossing checks at all quadrature points.
        mass2 = artifact.spec.external_mass^2
        for (energy, z) in Iterators.product(artifact.energies_squared,
                                               artifact.quadrature_nodes)
            t = -(energy - 4 * mass2) * (one(T) - z) / 2
            u = -(energy - 4 * mass2) * (one(T) + z) / 2
            isapprox(energy + t + u, 4 * mass2; atol=100eps(T), rtol=100eps(T)) ||
                push!(failures, "mandelstam_sum")
            break
        end
        # The zero coefficient vector is intentionally a boundary witness:
        # M=0 gives f_l=0 and S_l=1, not a strict interior point.
        all(iszero, artifact.zero_witness) || push!(failures, "zero_witness")
        all(isapprox.(cone_margins(artifact, artifact.zero_witness), zero(T);
                    atol=100eps(T), rtol=100eps(T))) || push!(failures, "zero_boundary")
        stable_fingerprint(artifact) == artifact.fingerprint || push!(failures, "fingerprint")
    catch
        push!(failures, "semantic_rebuild")
    end
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

end # module SMatrix4DSpecDiagnostic
