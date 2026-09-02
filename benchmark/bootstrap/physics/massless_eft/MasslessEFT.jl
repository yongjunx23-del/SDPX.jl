module MasslessEFT

using LinearAlgebra
using SHA
using SparseArrays
import SDPX

export MasslessEFTSpec, MasslessEFTArtifact
export massless_eft_specs, build_massless_eft, generate_rows, pair_basis
export build_model, build_soc_problem, objective_map, evaluate_amplitude
export audit_enforced, audit_heldout, audit_unitarity
export validate_artifact, canonical_text, stable_fingerprint, manifest_sha256

const ARTIFACT_SCHEMA_VERSION = 1
const SOURCE_GENERATOR_SHA256 = "72e0e5f16c1b0f5f3e671a9f0599d06425977d672437a265fbab1b5f07827fcd"
const SOURCE_AUDITOR_SHA256 = "407c4ff154c69989e24a28ec764bcb17dbf7c71d17c3f35858588c318231fbba"
const SOURCE_RESULT_SHA256 = "03399fed09bcf535fce61084536ff8c18d9aa260243cd657e89f76f7b69525ba"
const REPOSITORY_HEAD = "1541ab4fa666eedebbdc16706b668040c963f04d"
const PHI_STAR_TEXT = "1.211181126328405459400596274944243656708155553391151020347001165400401467736505165976615558893255859960761265007439555544360454547892731605960723621992436454230335579603439355355902902167744019602228266127625962008995580574403143672466108440121909510412577047350841810190489360423107563025994226830489924990538e-05"
# The canonical string above intentionally comes from the reviewed source.
const MANIFEST_PATH = joinpath(@__DIR__, "source_oracle_manifest.toml")
manifest_sha256() = open(MANIFEST_PATH, "r") do io
    bytes2hex(SHA.sha256(io))
end

Base.@kwdef struct MasslessEFTSpec{T<:AbstractFloat}
    id::String
    scale::Symbol
    maxN::Int
    lmax::Int
    ngrid::Int
    quadrature_order::Int
    heldout_ngrid::Int = 599
    heldout_lmax::Int = 100
    phi_star::T
    precision_bits::Int = 256
    source::Symbol = :external_pole_ansatz
    source_version::String = "external-snapshot-1541ab4"
    reference_status::Symbol = :sampled_build_only
    paper_equivalent::Bool = false
    normalization::Symbol = :tau_equals_4_T_physical
end

struct MasslessEFTArtifact{T<:AbstractFloat}
    schema_version::Int
    spec::MasslessEFTSpec{T}
    phis::Vector{T}
    heldout_phis::Vector{T}
    spins::Vector{Int}
    pairs::Vector{Tuple{Int,Int}}
    real_rows::Matrix{T}
    imag_rows::Matrix{T}
    cone_rhs::Vector{T}
    g0_map::Vector{T}
    g2_map::Vector{T}
    objective_maps::NamedTuple
    strict_witness::Vector{T}
    witness_certified::Bool
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

# Compatibility aliases make the row contract unambiguous without duplicating arrays.
function Base.getproperty(a::MasslessEFTArtifact, name::Symbol)
    name === :Rre && return getfield(a, :real_rows)
    name === :Rim && return getfield(a, :imag_rows)
    name === :rhs && return getfield(a, :cone_rhs)
    name === :heldout_grid && return getfield(a, :heldout_phis)
    return getfield(a, name)
end

pair_basis(maxN::Int) = Tuple{Int,Int}[(a, t - a) for t in 0:maxN for a in t:-1:0 if a >= t - a]

function _validate_spec(spec::MasslessEFTSpec)
    spec.maxN >= 0 || throw(ArgumentError("maxN must be nonnegative"))
    spec.lmax >= 0 && iseven(spec.lmax) || throw(ArgumentError("lmax must be a nonnegative even integer"))
    spec.ngrid >= 2 || throw(ArgumentError("ngrid must be at least two"))
    spec.heldout_ngrid >= 3 && isodd(spec.heldout_ngrid) || throw(ArgumentError("heldout_ngrid must be odd and at least three"))
    spec.quadrature_order >= 2 || throw(ArgumentError("quadrature_order must be at least two"))
    spec.heldout_lmax >= spec.lmax && iseven(spec.heldout_lmax) || throw(ArgumentError("heldout_lmax must be even and cover lmax"))
    FT = typeof(spec.phi_star)
    zero(spec.phi_star) < spec.phi_star < FT(pi) / 2 || throw(ArgumentError("phi_star must lie in (0,pi/2)"))
    spec.source === :external_pole_ansatz || throw(ArgumentError("unsupported source"))
    spec.normalization === :tau_equals_4_T_physical || throw(ArgumentError("normalization is mandatory"))
    spec.reference_status in (:build_only, :sampled_build_only) || throw(ArgumentError("massless EFT has no solve-eligible reference"))
    spec.paper_equivalent && throw(ArgumentError("finite sampled rows are not paper-equivalent"))
    return nothing
end

function _spec(::Type{T}, scale, id, maxN, lmax, ngrid, Q) where {T<:AbstractFloat}
    MasslessEFTSpec{T}(id=id, scale=scale, maxN=maxN, lmax=lmax,
        ngrid=ngrid, quadrature_order=Q, phi_star=parse(T, PHI_STAR_TEXT),
        precision_bits=(T === BigFloat ? precision(BigFloat) : 53))
end

function massless_eft_specs(::Type{T}=Float64) where {T<:AbstractFloat}
    return (
        smoke=_spec(T, :smoke, "massless_eft/smoke_N2_L4_grid9", 2, 4, 9, 32),
        train=_spec(T, :train, "massless_eft/train_N6_L12_grid33", 6, 12, 33, 64),
        production=_spec(T, :production, "massless_eft/production_N14_L60_grid300", 14, 60, 300, 2048),
    )
end

function _grid(::Type{T}, n::Int, phi_star::T) where {T<:AbstractFloat}
    lo, hi = phi_star, T(pi) - phi_star
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    T[mid - half * cos(T(k - 1) * T(pi) / T(n - 1)) for k in 1:n]
end

function _gauss_legendre(::Type{T}, n::Int) where {T<:AbstractFloat}
    iseven(n) || throw(ArgumentError("quadrature order must be even"))
    x, w = zeros(T, n), zeros(T, n)
    half = (n + 1) >> 1
    for i in 1:half
        z = cos(T(pi) * (T(i) - T(0.25)) / (T(n) + T(0.5)))
        for _ in 1:60
            p0, p1 = one(T), z
            for k in 1:n-1
                p0, p1 = p1, ((T(2k + 1) * z * p1) - T(k) * p0) / T(k + 1)
            end
            dp = T(n) * (z * p1 - p0) / (z*z - one(T))
            dz = p1 / dp
            z -= dz
            abs(dz) <= eps(T) * T(8) && break
        end
        p0, p1 = one(T), z
        for k in 1:n-1
            p0, p1 = p1, ((T(2k + 1) * z * p1) - T(k) * p0) / T(k + 1)
        end
        dp = T(n) * (z * p1 - p0) / (z*z - one(T))
        wi = T(2) / ((one(T) - z*z) * dp*dp)
        x[i], w[i] = -z, wi
        x[n + 1 - i], w[n + 1 - i] = z, wi
    end
    return x, w
end

function _legendre_table(x::Vector{T}, lmax::Int) where {T}
    p = Matrix{T}(undef, length(x), lmax + 1)
    p[:, 1] .= one(T)
    lmax >= 1 && (p[:, 2] .= x)
    for k in 1:lmax-1
        @inbounds for q in axes(p, 1)
            p[q, k + 2] = (T(2k + 1) * x[q] * p[q, k + 1] - T(k) * p[q, k]) / T(k + 1)
        end
    end
    p
end

_rbelow(z::T) where {T<:AbstractFloat} = (one(T) - sqrt(one(T) - z)) / (one(T) + sqrt(one(T) - z))

"""Generate the reviewed pole-augmented `(Re tau, Im tau)` row matrices.

Rows are spin-major, then phase-major. Column one is `alpha_pole`; remaining
columns follow the exact symmetric pair basis in `pair_basis`.
"""
function generate_rows(spec::MasslessEFTSpec{T}, phis::AbstractVector{T};
                       lmax::Int=spec.lmax, quadrature_order::Int=spec.quadrature_order) where {T<:AbstractFloat}
    _validate_spec(spec)
    all(isfinite, phis) || throw(ArgumentError("phase grid contains nonfinite values"))
    all(zero(T) < phi < T(pi) for phi in phis) || throw(ArgumentError("phase grid must avoid endpoints"))
    pair_list = pair_basis(spec.maxN)
    x, weights = _gauss_legendre(T, quadrature_order)
    spins = collect(0:2:lmax)
    leg = _legendre_table(x, lmax)
    ns, nv = length(phis), 1 + length(pair_list)
    rre, rim = Matrix{T}(undef, ns * length(spins), nv), Matrix{T}(undef, ns * length(spins), nv)
    inv32pi = one(T) / (T(32) * T(pi))
    weighted = leg .* reshape(weights, :, 1)
    for (phase_index, phi) in Base.pairs(phis)
        rs = complex(cos(phi), sin(phi))
        s = one(T) / cos(phi / T(2))^2
        rt = Matrix{T}(undef, quadrature_order, spec.maxN + 1)
        ru = Matrix{T}(undef, quadrature_order, spec.maxN + 1)
        rt[:, 1] .= one(T); ru[:, 1] .= one(T)
        pole = Vector{T}(undef, quadrature_order)
        for q in 1:quadrature_order
            t = -s * (one(T) - x[q]) / T(2)
            u = -s * (one(T) + x[q]) / T(2)
            r_t, r_u = _rbelow(t), _rbelow(u)
            pole[q] = -T(0.5) + one(T) / (r_t - one(T)) + one(T) / (r_u - one(T))
            spec.maxN >= 1 && (rt[q, 2] = r_t; ru[q, 2] = r_u)
        end
        for b in 2:spec.maxN
            @inbounds for q in 1:quadrature_order
                rt[q, b + 1] = rt[q, b] * rt[q, 2]
                ru[q, b + 1] = ru[q, b] * ru[q, 2]
            end
        end
        # Moments for all spins are one BLAS contraction.  The prototype
        # formed a matrix product separately for every spin; flattening
        # the (a,b) products makes this production-sized build bounded.
        # (The rows themselves remain in the reviewed spin-major order.)
        K = Matrix{T}(undef, (spec.maxN + 1)^2, quadrature_order)
        @inbounds for q in 1:quadrature_order, a in 0:spec.maxN, b in 0:spec.maxN
            K[a * (spec.maxN + 1) + b + 1, q] = rt[q, a + 1] * ru[q, b + 1]
        end
        all_moments = K * weighted[:, 1:length(spins)]
        for (spin_index, ell) in enumerate(spins)
            row = (spin_index - 1) * ns + phase_index
            wp = @view weighted[:, spin_index]
            rim[row, 1] = ell == 0 ? -cos(phi / 2) / sin(phi / 2) * inv32pi : zero(T)
            rre[row, 1] = dot(wp, pole) * inv32pi
            J = reshape(@view(all_moments[:, spin_index]), spec.maxN + 1, spec.maxN + 1)
            for (j, (a, b)) in enumerate(pair_list)
                value = if a == b
                    rs^a * (J[a + 1, 1] + J[1, a + 1]) + J[a + 1, a + 1]
                else
                    rs^a * (J[b + 1, 1] + J[1, b + 1]) + rs^b * (J[a + 1, 1] + J[1, a + 1]) + J[a + 1, b + 1] + J[b + 1, a + 1]
                end * inv32pi
                rre[row, j + 1], rim[row, j + 1] = real(value), imag(value)
            end
        end
    end
    return (real=rre, imag=rim, pairs=pair_list, spins=spins)
end

generate_rows(spec::MasslessEFTSpec{T}; heldout=false) where {T<:AbstractFloat} =
    generate_rows(spec, _grid(T, heldout ? spec.heldout_ngrid : spec.ngrid, spec.phi_star);
                  lmax=heldout ? spec.heldout_lmax : spec.lmax)

function _objective_maps(::Type{T}, pairs) where {T}
    n = 1 + length(pairs)
    g0, g2 = zeros(T, n), zeros(T, n)
    g0[1] = -T(3)
    g0[findfirst(==((0, 0)), pairs) + 1] = T(3)
    for (pair, coefficient) in (((1, 0), T(1)/2), ((2, 0), T(1)/4), ((1, 1), -T(1)/32))
        index = findfirst(==(pair), pairs)
        index === nothing || (g2[index + 1] = coefficient)
    end
    g2[1] = -T(3)/T(8)
    return (g0=g0, g2=g2, min_g0=g0, max_g0=-g0)
end

function build_massless_eft(spec::MasslessEFTSpec{T}) where {T<:AbstractFloat}
    _validate_spec(spec)
    phis = _grid(T, spec.ngrid, spec.phi_star)
    heldout = _grid(T, spec.heldout_ngrid, spec.phi_star)
    generated = generate_rows(spec, phis)
    maps = _objective_maps(T, generated.pairs)
    provisional = MasslessEFTArtifact{T}(
        ARTIFACT_SCHEMA_VERSION, spec, phis, heldout, generated.spins,
        generated.pairs, generated.real, generated.imag, T[one(T), zero(T), one(T)],
        maps.g0, maps.g2, maps, zeros(T, length(maps.g0)), false,
        (source_identifier="pole_ansatz_scan", source_generator_sha256=SOURCE_GENERATOR_SHA256,
         source_auditor_sha256=SOURCE_AUDITOR_SHA256, source_result_json_sha256=SOURCE_RESULT_SHA256,
         repository_head=REPOSITORY_HEAD, manifest_sha256=manifest_sha256(),
         normalization="tau=4*T_physical; implemented disk abs(1+i*tau)<=1",
         claim_boundary="finite sampled grid only; held-out diagnostic; no continuum certificate"),
        (variables=length(maps.g0), original_phases=length(phis), heldout_phases=length(heldout),
         spins=length(generated.spins), heldout_spins=length(0:2:spec.heldout_lmax),
         original_cones=size(generated.real, 1), cone_dimension=3, pair_count=length(generated.pairs),
         original_grid=:chebyshev_lobatto, heldout_grid=:chebyshev_lobatto,
         quadrature_order=spec.quadrature_order, witness=:non_strict_uncertified), "")
    return MasslessEFTArtifact(provisional.schema_version, provisional.spec, provisional.phis,
        provisional.heldout_phis, provisional.spins, provisional.pairs, provisional.real_rows,
        provisional.imag_rows, provisional.cone_rhs, provisional.g0_map, provisional.g2_map,
        provisional.objective_maps, provisional.strict_witness, provisional.witness_certified,
        provisional.provenance, provisional.counts, stable_fingerprint(provisional))
end
build_massless_eft(scale::Symbol, ::Type{T}=Float64) where {T<:AbstractFloat} =
    build_massless_eft(getproperty(massless_eft_specs(T), scale))

function evaluate_amplitude(artifact::MasslessEFTArtifact{T}, coefficients::AbstractVector; heldout=false,
                            lmax=artifact.spec.lmax) where {T}
    length(coefficients) == length(artifact.g0_map) || throw(DimensionMismatch("coefficient vector has wrong length"))
    rows = heldout ? generate_rows(artifact.spec, artifact.heldout_phis; lmax=lmax).real : artifact.real_rows
    imrows = heldout ? generate_rows(artifact.spec, artifact.heldout_phis; lmax=lmax).imag : artifact.imag_rows
    # Return complex values while preserving the declared arithmetic type.
    return complex.(rows * T.(coefficients), imrows * T.(coefficients))
end

function _audit_rows(real_rows, imag_rows, coefficients::AbstractVector{T}, spins, phis) where {T}
    y = T.(coefficients)
    re, im = real_rows * y, imag_rows * y
    excess = T[sqrt((one(T) - im[k])^2 + re[k]^2) - one(T) for k in eachindex(re)]
    index = isempty(excess) ? 0 : argmax(excess)
    return (status=:diagnostic, max_positive_excess=max(zero(T), maximum(excess)),
            worst_excess=excess[index], worst_spin=isempty(excess) ? -1 : spins[(index - 1) ÷ length(phis) + 1],
            worst_phase_index=isempty(excess) ? 0 : (index - 1) % length(phis) + 1,
            endpoint_distances=(first(phis), T(pi)-last(phis)),
            phases=length(phis), spins=length(spins), radius_excess=excess,
            normalization=:tau_equals_4_T_physical)
end

function audit_enforced(artifact::MasslessEFTArtifact{T}, coefficients::AbstractVector) where {T}
    length(coefficients) == length(artifact.g0_map) || throw(DimensionMismatch("coefficient vector has wrong length"))
    _audit_rows(artifact.real_rows, artifact.imag_rows, coefficients, artifact.spins, artifact.phis)
end

function audit_heldout(artifact::MasslessEFTArtifact{T}, coefficients::AbstractVector;
                       quadrature_order::Int=artifact.spec.quadrature_order) where {T}
    length(coefficients) == length(artifact.g0_map) || throw(DimensionMismatch("coefficient vector has wrong length"))
    generated = generate_rows(artifact.spec, artifact.heldout_phis;
                              lmax=artifact.spec.heldout_lmax, quadrature_order=quadrature_order)
    all(isfinite, generated.real) && all(isfinite, generated.imag) || throw(ArgumentError("held-out rows are nonfinite"))
    heldout_indices = collect(2:2:length(artifact.heldout_phis))
    phase_count = length(artifact.heldout_phis)
    selected_rows = [((spin - 1) * phase_count + phase) for spin in 1:length(generated.spins) for phase in heldout_indices]
    selected_phis = artifact.heldout_phis[heldout_indices]
    audit = _audit_rows(generated.real[selected_rows, :], generated.imag[selected_rows, :],
                        coefficients, generated.spins, selected_phis)
    union_audit = _audit_rows(generated.real, generated.imag, coefficients,
                              generated.spins, artifact.heldout_phis)
    # Every other point of the 599 grid is the original 300-node grid's phase
    # interval; the even Julia indices are independently regenerated holdouts.
    return merge(audit, (heldout_indices=heldout_indices,
                         union=union_audit, union_phases=length(artifact.heldout_phis),
                         endpoint_limits=(phi_zero=:not_represented, phi_pi=:not_represented),
                         policy=:diagnostic_only_no_declared_threshold))
end

audit_unitarity(artifact::MasslessEFTArtifact, coefficients::AbstractVector) =
    (enforced=audit_enforced(artifact, coefficients), heldout=audit_heldout(artifact, coefficients))

function build_model(artifact::MasslessEFTArtifact{T}) where {T<:AbstractFloat}
    validate_artifact(artifact).valid || throw(ArgumentError("invalid massless EFT artifact"))
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits=artifact.spec.precision_bits, name=artifact.spec.id) : SDPX.Model(T; name=artifact.spec.id)
    variables = SDPX.variable!(model, :coefficients, length(artifact.g0_map); domain=SDPX.Reals())
    for row in axes(artifact.real_rows, 1)
        A = zeros(T, 3, length(artifact.g0_map))
        A[2, :] .= artifact.real_rows[row, :]
        A[3, :] .= -artifact.imag_rows[row, :]
        SDPX.constraint!(model, Symbol(:unitarity_, row), A * variables .+ artifact.cone_rhs, SDPX.LorentzCone())
    end
    SDPX.objective!(model, SDPX.Minimize(), LinearAlgebra.dot(zeros(T, length(artifact.g0_map)), variables))
    return model
end

function build_soc_problem(artifact::MasslessEFTArtifact{T}) where {T<:AbstractFloat}
    validate_artifact(artifact).valid || throw(ArgumentError("invalid massless EFT artifact"))
    n = length(artifact.g0_map)
    cones = SDPX.SOCConstraint{T}[]
    sizehint!(cones, size(artifact.real_rows, 1))
    for row in axes(artifact.real_rows, 1)
        entries = Int[]; rows = Int[]; vals = T[]
        for j in 1:n
            artifact.real_rows[row, j] == zero(T) || (push!(rows, 2); push!(entries, j); push!(vals, artifact.real_rows[row, j]))
            artifact.imag_rows[row, j] == zero(T) || (push!(rows, 3); push!(entries, j); push!(vals, -artifact.imag_rows[row, j]))
        end
        A = sparse(rows, entries, vals, 3, n)
        push!(cones, SDPX.SOCConstraint(A, artifact.cone_rhs; T))
    end
    return SDPX.second_order_program(zeros(T, n), cones; T)
end

objective_map(artifact::MasslessEFTArtifact, which::Symbol=:g0) =
    which in (:g0, :g2, :min_g0, :max_g0) ? getproperty(artifact.objective_maps, which) : throw(ArgumentError("objective must be :g0, :g2, :min_g0, or :max_g0"))

function canonical_text(artifact::MasslessEFTArtifact)
    io = IOBuffer(); println(io, "massless-eft-schema=", artifact.schema_version)
    for name in fieldnames(typeof(artifact.spec)); println(io, "spec.", name, "=", repr(getfield(artifact.spec, name))); end
    println(io, "pairs=", repr(artifact.pairs)); println(io, "phis=", join(string.(artifact.phis), ",")); println(io, "heldout=", join(string.(artifact.heldout_phis), ","))
    println(io, "spins=", repr(artifact.spins)); println(io, "rhs=", join(string.(artifact.cone_rhs), ","))
    for name in (:g0_map, :g2_map, :strict_witness); println(io, name, "=", join(string.(getfield(artifact, name)), ",")); end
    for row in axes(artifact.real_rows, 1), col in axes(artifact.real_rows, 2)
        println(io, "row[", row, ",", col, "]=", artifact.real_rows[row,col], ";", artifact.imag_rows[row,col])
    end
    println(io, "witness_certified=", artifact.witness_certified); println(io, "provenance=", repr(artifact.provenance)); println(io, "counts=", repr(artifact.counts))
    String(take!(io))
end
stable_fingerprint(artifact::MasslessEFTArtifact) = bytes2hex(SHA.sha256(codeunits(canonical_text(artifact))))

function validate_artifact(artifact::MasslessEFTArtifact; rebuild::Bool=true)
    failures = String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures, "schema_version")
    try _validate_spec(artifact.spec) catch; push!(failures, "spec") end
    all(isfinite, artifact.phis) && all(isfinite, artifact.heldout_phis) || push!(failures, "grid_nonfinite")
    all(isfinite, artifact.real_rows) && all(isfinite, artifact.imag_rows) || push!(failures, "rows_nonfinite")
    size(artifact.real_rows) == size(artifact.imag_rows) || push!(failures, "row_dimensions")
    FT = eltype(artifact.phis)
    length(artifact.cone_rhs) == 3 || push!(failures, "cone_rhs")
    artifact.cone_rhs == FT[one(FT), zero(FT), one(FT)] || push!(failures, "cone_normalization")
    length(artifact.g0_map) == length(artifact.g2_map) || push!(failures, "objective_dimensions")
    artifact.witness_certified == false || push!(failures, "uncertified_witness_claim")
    artifact.provenance.manifest_sha256 == manifest_sha256() || push!(failures, "manifest_digest")
    if rebuild && isempty(failures)
        expected = build_massless_eft(artifact.spec)
        artifact.phis == expected.phis || push!(failures, "phase_semantics")
        artifact.heldout_phis == expected.heldout_phis || push!(failures, "heldout_semantics")
        artifact.pairs == expected.pairs || push!(failures, "pair_ordering")
        artifact.real_rows == expected.real_rows || push!(failures, "real_row_semantics")
        artifact.imag_rows == expected.imag_rows || push!(failures, "imag_row_semantics")
        artifact.g0_map == expected.g0_map || push!(failures, "g0_semantics")
        artifact.g2_map == expected.g2_map || push!(failures, "g2_semantics")
    end
    stable_fingerprint(artifact) == artifact.fingerprint || push!(failures, "fingerprint")
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

end # module
