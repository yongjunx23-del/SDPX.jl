module MasslessEFT

using LinearAlgebra
using SHA
using SparseArrays
import SDPX

export MasslessEFTSpec, MasslessEFTArtifact
export massless_eft_specs, build_massless_eft, generate_rows, pair_basis
export build_model, build_sdp_model, build_psd_model, build_soc_problem, objective_map, evaluate_amplitude
export audit_enforced, audit_heldout, audit_unitarity
export validate_artifact, canonical_text, stable_fingerprint, manifest_sha256
export spec_fingerprint, reference_generate_rows, generator_parity_gate, build_production_massless_eft
export prove_representation_parity

const ARTIFACT_SCHEMA_VERSION = 1
const SOURCE_GUARD_PRECISION_BITS = 1024
const SOURCE_GENERATOR_SHA256 = "72e0e5f16c1b0f5f3e671a9f0599d06425977d672437a265fbab1b5f07827fcd"
const SOURCE_AUDITOR_SHA256 = "407c4ff154c69989e24a28ec764bcb17dbf7c71d17c3f35858588c318231fbba"
const SOURCE_RESULT_SHA256 = "03399fed09bcf535fce61084536ff8c18d9aa260243cd657e89f76f7b69525ba"
const SDPX_IMPORT_BASE = "1541ab4fa666eedebbdc16706b668040c963f04d"
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
    heldout_ngrid::Int = 2 * ngrid - 1
    heldout_lmax::Int = 100
    phi_star::T
    precision_bits::Int = SOURCE_GUARD_PRECISION_BITS
    source::Symbol = :external_pole_ansatz
    source_version::String = "external-generator-snapshot"
    reference_status::Symbol = :sampled_build_only
    paper_equivalent::Bool = false
    normalization::Symbol = :physical_factor_four_unresolved
end

"""Cheap catalog identity: spec metadata and manifest only, never generated rows."""
function spec_fingerprint(spec::MasslessEFTSpec, manifest_digest::AbstractString=manifest_sha256())
    payload = join((spec.id, spec.scale, spec.maxN, spec.lmax, spec.ngrid,
        spec.heldout_ngrid, spec.quadrature_order, spec.heldout_lmax,
        repr(spec.phi_star), spec.precision_bits, spec.source, spec.source_version,
        spec.reference_status, spec.paper_equivalent, spec.normalization,
        manifest_digest), "|")
    return bytes2hex(SHA.sha256(codeunits(payload)))
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
    witness_candidate::Vector{T}
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
    spec.normalization === :physical_factor_four_unresolved || throw(ArgumentError("physical factor-four normalization is unresolved"))
    spec.reference_status in (:build_only, :sampled_build_only) || throw(ArgumentError("massless EFT has no solve-eligible reference"))
    spec.paper_equivalent && throw(ArgumentError("finite sampled rows are not paper-equivalent"))
    return nothing
end

function _spec(::Type{T}, scale, id, maxN, lmax, ngrid, Q) where {T<:AbstractFloat}
    MasslessEFTSpec{T}(id=id, scale=scale, maxN=maxN, lmax=lmax,
        ngrid=ngrid, heldout_ngrid=2 * ngrid - 1, quadrature_order=Q,
        phi_star=parse(T, PHI_STAR_TEXT), precision_bits=SOURCE_GUARD_PRECISION_BITS)
end

function massless_eft_specs(::Type{T}=Float64) where {T<:AbstractFloat}
    return (
        smoke=_spec(T, :smoke, "massless_eft/smoke_N2_L4_grid9", 2, 4, 9, 32),
        train=_spec(T, :train, "massless_eft/train_N6_L12_grid33", 6, 12, 33, 64),
        production=_spec(T, :production, "massless_eft/production_N14_L60_grid300", 14, 60, 300, 2048),
    )
end

function _source_grid(n::Int, phi_star)
    setprecision(BigFloat, SOURCE_GUARD_PRECISION_BITS) do
        source_phi = if phi_star isa Float64 && phi_star == parse(Float64, PHI_STAR_TEXT)
            parse(BigFloat, PHI_STAR_TEXT)
        else
            BigFloat(phi_star)
        end
        lo, hi = source_phi, BigFloat(pi) - source_phi
        mid, half = (hi + lo) / 2, (hi - lo) / 2
        [mid - half * cos(BigFloat(k - 1) * BigFloat(pi) / BigFloat(n - 1)) for k in 1:n]
    end
end

function _grid(::Type{T}, n::Int, phi_star::T) where {T<:AbstractFloat}
    # Convert each guarded source coordinate once to the target arithmetic.
    T[value for value in _source_grid(n, phi_star)]
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
function _legacy_generate_rows(spec::MasslessEFTSpec{T}, phis::AbstractVector{T};
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

# The optimized candidate is gated against the literal implementation before
# it can be used by any caller.  Until the candidate's order-preserving parity
# is demonstrated, this safe wrapper is the reference conversion.
function _generate_rows_optimized(spec::MasslessEFTSpec{T}, phis::AbstractVector{T};
                                  lmax::Int=spec.lmax, quadrature_order::Int=spec.quadrature_order) where {T<:AbstractFloat}
    _convert_rows(T, reference_generate_rows(spec, phis; lmax, quadrature_order))
end

# This is deliberately a literal, scalar-order reference implementation of
# the external BigFloat generator.  It is the source of truth for persisted
# rows; optimized implementations may only be used after the parity gate.
function _reference_gauss_legendre(n::Int)
    x, w = Vector{BigFloat}(undef, n), Vector{BigFloat}(undef, n)
    half = (n + 1) >> 1
    for i in 1:half
        z = BigFloat(cos(pi * (i - 0.25) / (n + 0.5)))
        dp = zero(BigFloat)
        for _ in 1:50
            p0, p1 = one(BigFloat), z
            for k in 1:(n - 1)
                p0, p1 = p1, ((2k + 1) * z * p1 - k * p0) / (k + 1)
            end
            dp = n * (z * p1 - p0) / (z * z - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < eps(BigFloat) * 8 && break
        end
        w_i = 2 / ((1 - z * z) * dp * dp)
        x[i], w[i] = -z, w_i
        x[n + 1 - i], w[n + 1 - i] = z, w_i
    end
    return x, w
end

function _reference_generate_rows_bigfloat(spec::MasslessEFTSpec, phis, lmax::Int, quadrature_order::Int)
    T = BigFloat
    pair_list = pair_basis(spec.maxN)
    x, weights = _reference_gauss_legendre(quadrature_order)
    spins = collect(0:2:lmax)
    leg = _legendre_table(x, lmax)
    ns, nv = length(phis), 1 + length(pair_list)
    rre, rim = Matrix{T}(undef, ns * length(spins), nv), Matrix{T}(undef, ns * length(spins), nv)
    inv32pi = one(T) / (T(32) * T(pi))
    wp_table = weights .* leg[:, [ell + 1 for ell in spins]]
    for (phase_index, phi) in Base.pairs(phis)
        rs = Complex{T}(cos(phi), sin(phi))
        s = one(T) / cos(phi / 2)^2
        rt = Matrix{T}(undef, quadrature_order, spec.maxN + 1)
        ru = Matrix{T}(undef, quadrature_order, spec.maxN + 1)
        rt[:, 1] .= one(T); ru[:, 1] .= one(T)
        pole = Vector{T}(undef, quadrature_order)
        for q in 1:quadrature_order
            t = -s * (one(T) - x[q]) / 2
            u = -s * (one(T) + x[q]) / 2
            r_t = _rbelow(t)
            r_u = _rbelow(u)
            pole[q] = -T(0.5) + one(T) / (r_t - one(T)) + one(T) / (r_u - one(T))
            if spec.maxN >= 1
                rt[q, 2] = r_t; ru[q, 2] = r_u
            end
        end
        for b in 2:spec.maxN
            @inbounds for q in 1:quadrature_order
                rt[q, b + 1] = rt[q, b] * rt[q, 2]
                ru[q, b + 1] = ru[q, b] * ru[q, 2]
            end
        end
        for (spin_index, ell) in enumerate(spins)
            wp = @view wp_table[:, spin_index]
            row = phase_index + (spin_index - 1) * ns
            rre[row, 1] = dot(wp, pole) * inv32pi
            rim[row, 1] = ell == 0 ? (-cot(phi / 2) * inv32pi) : zero(T)
            J = (rt .* wp)' * ru
            for (j, (a, b)) in enumerate(pair_list)
                value = if a == b
                    rs^a * (J[a + 1, 1] + J[1, a + 1]) + J[a + 1, a + 1]
                else
                    rs^a * (J[b + 1, 1] + J[1, b + 1]) +
                    rs^b * (J[a + 1, 1] + J[1, a + 1]) +
                    J[a + 1, b + 1] + J[b + 1, a + 1]
                end
                value *= inv32pi
                rre[row, j + 1] = real(value)
                rim[row, j + 1] = imag(value)
            end
        end
    end
    return (real=rre, imag=rim, pairs=pair_list, spins=spins)
end

function reference_generate_rows(spec::MasslessEFTSpec{T}, phis::AbstractVector{T};
                                 lmax::Int=spec.lmax, quadrature_order::Int=spec.quadrature_order) where {T<:AbstractFloat}
    _validate_spec(spec)
    all(isfinite, phis) || throw(ArgumentError("phase grid contains nonfinite values"))
    setprecision(BigFloat, SOURCE_GUARD_PRECISION_BITS) do
        source_phis = BigFloat[BigFloat(phi) for phi in phis]
        _reference_generate_rows_bigfloat(spec, source_phis, lmax, quadrature_order)
    end
end

function _convert_rows(::Type{T}, generated) where {T<:AbstractFloat}
    real_rows = Matrix{T}(undef, size(generated.real))
    imag_rows = Matrix{T}(undef, size(generated.imag))
    for index in eachindex(real_rows)
        real_rows[index] = T(generated.real[index])
        imag_rows[index] = T(generated.imag[index])
    end
    return (real=real_rows, imag=imag_rows, pairs=generated.pairs, spins=generated.spins)
end

function generator_parity_gate(spec::MasslessEFTSpec{T}, phis::AbstractVector{T};
                               lmax::Int=spec.lmax, quadrature_order::Int=spec.quadrature_order) where {T<:AbstractFloat}
    reference = _convert_rows(T, reference_generate_rows(spec, phis; lmax, quadrature_order))
    optimized = _generate_rows_optimized(spec, phis; lmax, quadrature_order)
    same_shape = size(reference.real) == size(optimized.real) && reference.pairs == optimized.pairs
    scale = max(one(T), maximum(abs, reference.real), maximum(abs, reference.imag))
    error_bound = T(128) * eps(T) * scale
    passed = same_shape && maximum(abs, reference.real - optimized.real) <= error_bound &&
             maximum(abs, reference.imag - optimized.imag) <= error_bound
    return (passed=passed, reference=reference, optimized=optimized, error_bound=error_bound)
end

# Source rows always come from the guarded reference and are converted once to
# the requested arithmetic.  The optimized implementation is never silently
# substituted for source data.
function generate_rows(spec::MasslessEFTSpec{T}, phis::AbstractVector{T};
                       lmax::Int=spec.lmax, quadrature_order::Int=spec.quadrature_order) where {T<:AbstractFloat}
    _validate_spec(spec)
    all(zero(T) < phi < T(pi) for phi in phis) || throw(ArgumentError("phase grid must avoid endpoints"))
    _convert_rows(T, reference_generate_rows(spec, phis; lmax, quadrature_order))
end

function _generate_rows_on_source_grid(spec::MasslessEFTSpec{T}, n::Int, lmax::Int,
                                         quadrature_order::Int) where {T<:AbstractFloat}
    source_phis = _source_grid(n, spec.phi_star)
    generated = _reference_generate_rows_bigfloat(spec, source_phis, lmax, quadrature_order)
    return _convert_rows(T, generated)
end

function generate_rows(spec::MasslessEFTSpec{T}; heldout=false) where {T<:AbstractFloat}
    n = heldout ? spec.heldout_ngrid : spec.ngrid
    lmax = heldout ? spec.heldout_lmax : spec.lmax
    _generate_rows_on_source_grid(spec, n, lmax, spec.quadrature_order)
end

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
    source_phis = _source_grid(spec.ngrid, spec.phi_star)
    source_heldout = _source_grid(spec.heldout_ngrid, spec.phi_star)
    phis = T[value for value in source_phis]
    heldout = T[value for value in source_heldout]
    generated = _convert_rows(T, setprecision(BigFloat, SOURCE_GUARD_PRECISION_BITS) do
        _reference_generate_rows_bigfloat(spec, source_phis, spec.lmax, spec.quadrature_order)
    end)
    maps = _objective_maps(T, generated.pairs)
    provisional = MasslessEFTArtifact{T}(
        ARTIFACT_SCHEMA_VERSION, spec, phis, heldout, generated.spins,
        generated.pairs, generated.real, generated.imag, T[one(T), zero(T), one(T)],
        maps.g0, maps.g2, maps, zeros(T, length(maps.g0)), false,
        (source_identifier="pole_ansatz_scan", source_generator_sha256=SOURCE_GENERATOR_SHA256,
         source_auditor_sha256=SOURCE_AUDITOR_SHA256, external_receipt_json_sha256=SOURCE_RESULT_SHA256,
         sdpx_import_base=SDPX_IMPORT_BASE, source_guard_precision_bits=SOURCE_GUARD_PRECISION_BITS,
         manifest_sha256=manifest_sha256(),
         normalization="implemented source variable tau; physical factor-four normalization unresolved",
         claim_boundary="finite sampled grid only; held-out diagnostic; no continuum certificate"),
        (variables=length(maps.g0), original_phases=length(phis), heldout_phases=length(heldout),
         spins=length(generated.spins), heldout_spins=length(0:2:spec.heldout_lmax),
         original_cones=size(generated.real, 1), cone_dimension=3, pair_count=length(generated.pairs),
         original_grid=:chebyshev_lobatto, heldout_grid=:chebyshev_lobatto,
         quadrature_order=spec.quadrature_order, witness=:candidate_uncertified), "")
    return MasslessEFTArtifact(provisional.schema_version, provisional.spec, provisional.phis,
        provisional.heldout_phis, provisional.spins, provisional.pairs, provisional.real_rows,
        provisional.imag_rows, provisional.cone_rhs, provisional.g0_map, provisional.g2_map,
        provisional.objective_maps, provisional.witness_candidate, provisional.witness_certified,
        provisional.provenance, provisional.counts, stable_fingerprint(provisional))
end
build_massless_eft(scale::Symbol, ::Type{T}=Float64) where {T<:AbstractFloat} =
    build_massless_eft(getproperty(massless_eft_specs(T), scale))

"""Explicit opt-in production build; ordinary catalog suites never call this."""
function build_production_massless_eft(::Type{T}=BigFloat; confirm::Bool=false) where {T<:AbstractFloat}
    confirm || throw(ArgumentError("production build requires confirm=true"))
    build_massless_eft(:production, T)
end

function evaluate_amplitude(artifact::MasslessEFTArtifact{T}, coefficients::AbstractVector; heldout=false,
                            lmax=artifact.spec.lmax) where {T}
    length(coefficients) == length(artifact.g0_map) || throw(DimensionMismatch("coefficient vector has wrong length"))
    heldout_rows = heldout ? _generate_rows_on_source_grid(
        artifact.spec, artifact.spec.heldout_ngrid, lmax, artifact.spec.quadrature_order) : nothing
    rows = heldout ? heldout_rows.real : artifact.real_rows
    imrows = heldout ? heldout_rows.imag : artifact.imag_rows
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
            normalization=:physical_factor_four_unresolved)
end

function audit_enforced(artifact::MasslessEFTArtifact{T}, coefficients::AbstractVector) where {T}
    length(coefficients) == length(artifact.g0_map) || throw(DimensionMismatch("coefficient vector has wrong length"))
    _audit_rows(artifact.real_rows, artifact.imag_rows, coefficients, artifact.spins, artifact.phis)
end

function audit_heldout(artifact::MasslessEFTArtifact{T}, coefficients::AbstractVector;
                       quadrature_order::Int=artifact.spec.quadrature_order) where {T}
    length(coefficients) == length(artifact.g0_map) || throw(DimensionMismatch("coefficient vector has wrong length"))
    generated = _generate_rows_on_source_grid(
        artifact.spec, artifact.spec.heldout_ngrid, artifact.spec.heldout_lmax, quadrature_order)
    all(isfinite, generated.real) && all(isfinite, generated.imag) || throw(ArgumentError("held-out rows are nonfinite"))
    heldout_indices = collect(2:2:length(artifact.heldout_phis))
    phase_count = length(artifact.heldout_phis)
    selected_rows = [((spin - 1) * phase_count + phase) for spin in 1:length(generated.spins) for phase in heldout_indices]
    selected_phis = artifact.heldout_phis[heldout_indices]
    audit = _audit_rows(generated.real[selected_rows, :], generated.imag[selected_rows, :],
                        coefficients, generated.spins, selected_phis)
    union_audit = _audit_rows(generated.real, generated.imag, coefficients,
                              generated.spins, artifact.heldout_phis)
    # Even Julia indices are independently regenerated holdouts; no endpoint,
    # continuum, or extrapolation claim is attached to this diagnostic.
    return merge(audit, (heldout_indices=heldout_indices,
                         union=union_audit, union_phases=length(artifact.heldout_phis),
                         endpoint_limits=(phi_zero=:not_represented, phi_pi=:not_represented),
                         policy=:diagnostic_only_no_declared_threshold))
end

audit_unitarity(artifact::MasslessEFTArtifact, coefficients::AbstractVector) =
    (enforced=audit_enforced(artifact, coefficients), heldout=audit_heldout(artifact, coefficients))

function _objective_vector(artifact::MasslessEFTArtifact{T}, objective::Symbol) where {T}
    objective === :none && return zeros(T, length(artifact.g0_map))
    objective === :min_g0 && return artifact.objective_maps.g0
    objective === :max_g0 && return artifact.objective_maps.max_g0
    throw(ArgumentError("objective must be :none, :min_g0, or :max_g0"))
end

function build_model(artifact::MasslessEFTArtifact{T}; objective::Symbol=:none) where {T<:AbstractFloat}
    validate_artifact(artifact).valid || throw(ArgumentError("invalid massless EFT artifact"))
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits=artifact.spec.precision_bits, name=artifact.spec.id) : SDPX.Model(T; name=artifact.spec.id)
    variables = SDPX.variable!(model, :coefficients, length(artifact.g0_map); domain=SDPX.Reals())
    for row in axes(artifact.real_rows, 1)
        A = zeros(T, 3, length(artifact.g0_map))
        A[2, :] .= artifact.real_rows[row, :]
        A[3, :] .= -artifact.imag_rows[row, :]
        SDPX.constraint!(model, Symbol(:unitarity_, row), A * variables .+ artifact.cone_rhs, SDPX.LorentzCone())
    end
    SDPX.objective!(model, SDPX.Minimize(), LinearAlgebra.dot(_objective_vector(artifact, objective), variables))
    return model
end
build_model(artifact::MasslessEFTArtifact, objective::Symbol) = build_model(artifact; objective)

function build_sdp_model(artifact::MasslessEFTArtifact{T}; objective::Symbol=:none) where {T<:AbstractFloat}
    validate_artifact(artifact).valid || throw(ArgumentError("invalid sampled massless EFT artifact"))
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits=artifact.spec.precision_bits, name=artifact.spec.id * ".sdp") : SDPX.Model(T; name=artifact.spec.id * ".sdp")
    variables = SDPX.variable!(model, :coefficients, length(artifact.g0_map); domain=SDPX.Reals())
    for row in axes(artifact.real_rows, 1)
        re = sum((artifact.real_rows[row, j] * variables[j] for j in axes(artifact.real_rows, 2)); init=zero(T))
        im = sum((artifact.imag_rows[row, j] * variables[j] for j in axes(artifact.imag_rows, 2)); init=zero(T))
        matrix = Any[T(2) - im re; re im]
        SDPX.constraint!(model, Symbol(:unitarity_psd_, row), matrix, SDPX.PSDCone())
    end
    SDPX.objective!(model, SDPX.Minimize(), LinearAlgebra.dot(_objective_vector(artifact, objective), variables))
    return model
end
build_sdp_model(artifact::MasslessEFTArtifact, objective::Symbol) = build_sdp_model(artifact; objective)
const build_psd_model = build_sdp_model

function build_soc_problem(artifact::MasslessEFTArtifact{T}; objective::Symbol=:none) where {T<:AbstractFloat}
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
    return SDPX.second_order_program(_objective_vector(artifact, objective), cones; T)
end
build_soc_problem(artifact::MasslessEFTArtifact, objective::Symbol) = build_soc_problem(artifact; objective)

objective_map(artifact::MasslessEFTArtifact{T}, which::Symbol=:g0) where {T} =
    which === :none ? zeros(T, length(artifact.g0_map)) :
    which in (:g0, :g2, :min_g0, :max_g0) ? getproperty(artifact.objective_maps, which) :
    throw(ArgumentError("objective must be :none, :min_g0, or :max_g0; :g2 is diagnostic only"))

function prove_representation_parity(artifact::MasslessEFTArtifact{T}) where {T}
    n = length(artifact.g0_map)
    affine = size(artifact.real_rows) == size(artifact.imag_rows) && size(artifact.real_rows, 2) == n
    rhs = artifact.cone_rhs == T[one(T), zero(T), one(T)]
    determinant_identity = "det([2-Im(tau) Re(tau; Re(tau) Im(tau)]) = 1 - (Re(tau)^2 + (1-Im(tau))^2)"
    return (valid=affine && rhs, affine=affine, determinant_parity=determinant_identity,
            source_psd_block="[2-Im(tau) Re(tau); Re(tau) Im(tau)]",
            source_q3_vector="[1, Re(tau), 1-Im(tau)]", source_disk="abs(1+i*tau)<=1")
end

function canonical_text(artifact::MasslessEFTArtifact)
    io = IOBuffer(); println(io, "massless-eft-schema=", artifact.schema_version)
    for name in fieldnames(typeof(artifact.spec)); println(io, "spec.", name, "=", repr(getfield(artifact.spec, name))); end
    println(io, "pairs=", repr(artifact.pairs)); println(io, "phis=", join(string.(artifact.phis), ",")); println(io, "heldout=", join(string.(artifact.heldout_phis), ","))
    println(io, "spins=", repr(artifact.spins)); println(io, "rhs=", join(string.(artifact.cone_rhs), ","))
    for name in (:g0_map, :g2_map, :witness_candidate); println(io, name, "=", join(string.(getfield(artifact, name)), ",")); end
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
