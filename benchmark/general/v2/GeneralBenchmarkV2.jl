module GeneralBenchmarkV2

using SHA
import SDPX

export V2_SCHEMA_VERSION, V2Axis, V2Tier, V2Precision, V2Reference,
    AbstractV2SourceArtifact, AbstractV2SmallArtifact, LPArtifact,
    SOCPArtifact, RSOCArtifact, SDPArtifact, IllConditionedArtifact, MixedArtifact,
    V2LPOracle, V2RSOCOracle, V2SDPOracle, V2MixedOracle, V2ConicArtifact, native_v2_catalog,
    V2Transform, V2Family, V2Instance, V2Catalog, V2Built, V2Validation,
    V2RunResult, expand, validate_catalog, catalog_fingerprint,
    lp_tranche_catalog, ill_conditioned_tranche_catalog, socp_tranche_catalog,
    V2SOCOracle,

    lp_tranche_catalog, ill_conditioned_tranche_catalog,
    rsoc_tranche_catalog, sdp_tranche_catalog, mixed_tranche_catalog,
    input_fingerprint, mathematical_fingerprint, execution_fingerprint, adapt_generic_specs,
    build_instance, run_instance, reference_interval, resource_tiers,
    precision_matrix, reviewed_precision_specs, certificate_gate,
    validate_manifest, manifest_fingerprint, compatibility_tier,
    training_instances, holdout_instances, sentinel_instances,
    classify_disposition

const V2_SCHEMA_VERSION = 2

"""A finite deterministic parameter axis used to expand benchmark families."""
struct V2Axis{F}
    name::Symbol
    values::Vector{Any}
    canonicalize::F
    function V2Axis(name::Symbol, values, canonicalize=identity)
        isempty(values) && throw(ArgumentError("axis $name must not be empty"))
        vals = Any[canonicalize(v) for v in values]
        length(unique(vals)) == length(vals) ||
            throw(ArgumentError("axis $name contains duplicate values"))
        new{typeof(canonicalize)}(name, vals, canonicalize)
    end
end

"""Resource and execution policy owned by a benchmark tier."""
struct V2Tier
    name::Symbol
    lane::Symbol
    wall_seconds::Int
    memory_bytes::Int
    solve_policy::Symbol
    function V2Tier(name::Symbol, lane::Symbol, wall_seconds::Integer,
                    memory_bytes::Integer, solve_policy::Symbol=:local)
        name in (:small, :medium, :large, :extreme) ||
            throw(ArgumentError("unknown V2 tier $name"))
        wall_seconds > 0 || throw(ArgumentError("tier wall budget must be positive"))
        memory_bytes > 0 || throw(ArgumentError("tier memory budget must be positive"))
        new(name, lane, Int(wall_seconds), Int(memory_bytes), solve_policy)
    end
end

"""Arithmetic policy. Decimal tolerances remain strings until execution."""
struct V2Precision
    name::Symbol
    arithmetic::Any
    bits::Int
    solver_tolerance::String
    certificate_limit::String
    provider::Symbol
    function V2Precision(name::Symbol, arithmetic, bits::Integer,
                         solver_tolerance::AbstractString,
                         certificate_limit::AbstractString, provider::Symbol)
        bits >= 2 || throw(ArgumentError("precision bits must be at least two"))
        isempty(solver_tolerance) || isempty(certificate_limit) ?
            throw(ArgumentError("precision tolerances must be nonempty")) : nothing
        new(name, arithmetic, Int(bits), String(solver_tolerance),
            String(certificate_limit), provider)
    end
end

"""Independent reference contract; an absent oracle is explicitly build-only."""
struct V2Reference{O}
    status::Symbol
    expected_status::Symbol
    disposition::Symbol
    certificate_kind::Symbol
    objective_interval::Union{Nothing,Tuple{String,String}}
    oracle::O
    note::String
    # For XFAIL references, this records the known observed solver status
    # separately from the semantic expected status and never substitutes for it.
    prior_observed_status::Union{Nothing,Symbol}
    function V2Reference(status::Symbol, certificate_kind::Symbol,
                         objective_interval, oracle, note::AbstractString="";
                         expected_status::Symbol=status,
                         disposition::Symbol=(status === :xfail ? :XFAIL : :PASS),
                         prior_observed_status::Union{Nothing,Symbol}=nothing)
        status in (:optimal, :primal_infeasible, :dual_infeasible, :build_only,
                   :discretized, :xfail) ||
            throw(ArgumentError("unsupported reference status $status"))
        expected_status in (:optimal, :primal_infeasible, :dual_infeasible,
                            :iteration_limit, :numerical_breakdown, :build_only) ||
            throw(ArgumentError("unsupported expected solver status $expected_status"))
        disposition in (:PASS, :FAIL, :XFAIL, :XPASS, :RESOLVED) ||
            throw(ArgumentError("unsupported reference disposition $disposition"))
        status === :xfail && disposition ∉ (:XFAIL, :XPASS, :RESOLVED) &&
            throw(ArgumentError("xfail disposition must be XFAIL, XPASS, or RESOLVED"))
        status !== :xfail && disposition === :XFAIL &&
            throw(ArgumentError("only xfail references may use XFAIL disposition"))
        certificate_kind in (:optimal, :farkas, :ray, :build_only, :interval_or_bound) ||
            throw(ArgumentError("unsupported certificate kind $certificate_kind"))
        interval = if objective_interval === nothing
            nothing
        else
            length(objective_interval) == 2 || throw(ArgumentError("objective interval needs two endpoints"))
            (String(objective_interval[1]), String(objective_interval[2]))
        end
        status === :build_only && certificate_kind !== :build_only &&
            throw(ArgumentError("build-only reference requires certificate_kind=:build_only"))
        status !== :build_only && oracle === nothing &&
            throw(ArgumentError("non-build reference requires an independent oracle"))
        status === :optimal && (expected_status !== :optimal || certificate_kind !== :optimal || interval === nothing) &&
            throw(ArgumentError("optimal reference requires optimal status, certificate, and interval"))
        status === :primal_infeasible && (expected_status !== :primal_infeasible || certificate_kind !== :farkas) &&
            throw(ArgumentError("primal-infeasible reference requires a Farkas contract"))
        status === :xfail && isempty(note) &&
            throw(ArgumentError("xfail reference requires an issue note"))
        status === :build_only && expected_status !== :build_only &&
            throw(ArgumentError("build-only expected solver status must be build_only"))
        status === :xfail && expected_status === :xfail &&
            throw(ArgumentError("xfail must declare the expected solver status separately"))
        status === :xfail && expected_status === :build_only &&
            throw(ArgumentError("xfail must declare a concrete non-build solver status"))
        prior_observed_status === nothing || prior_observed_status in
            (:optimal, :primal_infeasible, :dual_infeasible, :iteration_limit,
             :numerical_breakdown, :build_only) || throw(ArgumentError(
                "invalid prior observed solver status $prior_observed_status"))
        new{typeof(oracle)}(status, expected_status, disposition, certificate_kind,
            interval, oracle, String(note), prior_observed_status)
    end
end

"""Provenance for a pluggable source-to-conic front-end transform.

The solver core never needs to know about polynomial or physics semantics.
`exactness` is deliberately an enum-like symbol so a finite grid or a
truncated SOS certificate cannot be reported as an exact source problem.
"""
struct V2Transform
    source_problem_type::Symbol
    target_cone_program::Symbol
    transform_id::Symbol
    version::Int
    exactness::Symbol
    positive_prefactor_factored::Bool
    positive_prefactor_proof::String
    lifting_dimensions::NamedTuple
    validation_receipts::NamedTuple
    fingerprint::String
    function V2Transform(source_problem_type::Symbol,
                         target_cone_program::Symbol,
                         transform_id::Symbol,
                         version::Integer,
                         exactness::Symbol;
                         positive_prefactor_factored::Bool=false,
                         positive_prefactor_proof::AbstractString="",
                         lifting_dimensions=(source=0, target=0, gram_blocks=0),
                         validation_receipts=(coefficient_match=false,
                                              source_reconstruction=false))
        exactness in (:identity, :exact_univariate_halfline,
                      :exact_univariate_matrix_halfline_if_proved,
                      :sos_relaxation, :finite_grid_surrogate) ||
            throw(ArgumentError("unsupported transform exactness $exactness"))
        version > 0 || throw(ArgumentError("transform version must be positive"))
        positive_prefactor_factored && isempty(positive_prefactor_proof) &&
            throw(ArgumentError("factored positive prefactor needs a proof/reference"))
        dims = (; lifting_dimensions...)
        receipts = (; validation_receipts...)
        payload = (source_problem_type, target_cone_program, transform_id,
                   Int(version), exactness, positive_prefactor_factored,
                   positive_prefactor_proof, dims, receipts)
        fp = bytes2hex(SHA.sha256(_canonical_bytes(payload)))
        new(source_problem_type, target_cone_program, transform_id, Int(version),
            exactness, positive_prefactor_factored, String(positive_prefactor_proof),
            dims, receipts, fp)
    end
end

"""A generic family: one builder, oracle, validator, and optional front-end transforms."""
struct V2Family{B,O,V,T}
    name::Symbol
    axes::Vector{V2Axis}
    build::B
    oracle::O
    validate::V
    transforms::T
end

"""Stable expanded instance. `payload` is front-end data, never solver state."""
struct V2Instance
    id::Symbol
    family::Symbol
    tier::V2Tier
    axis_values::NamedTuple
    split::Symbol
    source::String
    provenance::NamedTuple
    checksum::String
    resource::NamedTuple
    reference::V2Reference
    payload::Any
end

"""A validated family/instance/suite catalog."""
struct V2Catalog
    name::Symbol
    version::Int
    families::Vector{Any}
    instances::Vector{V2Instance}
    suites::NamedTuple
    function V2Catalog(name::Symbol, version::Integer, families, instances,
                       suites=(train=Symbol[], holdout=Symbol[], sentinel=Symbol[]))
        version >= 1 || throw(ArgumentError("catalog version must be positive"))
        cat = new(name, Int(version), Any[families...], V2Instance[instances...], suites)
        validate_catalog(cat)
        cat
    end
end

"""Result of a front-end build, retaining source and target facts."""
struct V2Built
    problem::Any
    oracle::Any
    source_artifact::Any
    input_fingerprint::String
    transform::V2Transform
    facts::NamedTuple
    resource::NamedTuple
end

struct V2Validation
    # `status` is the normalized validation status retained for compatibility;
    # `observed_status` is the actual solver status and `disposition` is the
    # explicit PASS/FAIL/XFAIL/XPASS/RESOLVED state-machine result.
    status::Symbol
    observed_status::Symbol
    disposition::Symbol
    certificate::Bool
    reference::Bool
    failures::Vector{Symbol}
end
V2Validation(status::Symbol, certificate::Bool, reference::Bool,
             failures::Vector{Symbol}) =
    V2Validation(status, status, status in (:optimal, :PASS) ? :PASS : :FAIL,
                 certificate, reference, failures)

"""Stable result with explicit unavailable lifecycle phases."""
struct V2RunResult
    id::Symbol
    family::Symbol
    tier::Symbol
    arithmetic::Symbol
    bits::Int
    status::Symbol
    certificate_valid::Bool
    objective::String
    dual_objective::String
    primal_residual::String
    dual_residual::String
    relative_gap::String
    iterations::Int
    setup_seconds::Union{Nothing,Float64}
    core_seconds::Union{Nothing,Float64}
    recovery_seconds::Union{Nothing,Float64}
    allocated_bytes::Union{Nothing,Int}
    input_fingerprint::String
    execution_fingerprint::String
    validation::V2Validation
    route_receipt::NamedTuple
end

const _TIERS = (
    V2Tier(:small, :local, 20, 4 * 1024^3, :full),
    V2Tier(:medium, :local, 600, 16 * 1024^3, :full),
    V2Tier(:large, :pbs, 6 * 3600, 128 * 1024^3, :selected),
    V2Tier(:extreme, :pbs, 24 * 3600, 256 * 1024^3, :selected),
)
resource_tiers() = _TIERS

"""The reviewed arithmetic policy matrix, represented as tier lane symbols."""
function precision_matrix()
    return (
        Float64=(small=:full, medium=:full, large=:full, extreme=:selected),
        Float64x2=(small=:full, medium=:full, large=:full, extreme=:sentinel),
        Float64x3=(small=:full, medium=:sentinel, large=:sentinel, extreme=:sentinel),
        Float64x4=(small=:full, medium=:full, large=:full, extreme=:selected),
        BigFloat256=(small=:full, medium=:full, large=:selected, extreme=:selected),
        BigFloat512=(small=:full, medium=:sentinel, large=:sentinel, extreme=:sentinel),
        BigFloat1024=(small=:full, medium=:sentinel, large=:sentinel, extreme=:sentinel),
    )
end

"""The reviewed seven-arithmetic precision declarations.

The non-BigFloat multifloat entries intentionally carry symbolic arithmetic
identifiers: this module does not import a provider merely to describe the
catalog.  A runner must resolve those identifiers to a concrete provider
before calling `build_instance`; an unavailable provider therefore fails
closed instead of silently falling back to Float64.
"""
function reviewed_precision_specs()
    return (
        V2Precision(:Float64, Float64, 53, "1e-8", "5e-7", :cholmod),
        V2Precision(:Float64x2, :Float64x2, 104, "1e-15", "5e-13", :multifloat_linear_algebra),
        V2Precision(:Float64x3, :Float64x3, 156, "1e-21", "5e-18", :multifloat_linear_algebra),
        V2Precision(:Float64x4, :Float64x4, 208, "1e-28", "5e-22", :multifloat_linear_algebra),
        V2Precision(:BigFloat256, BigFloat, 256, "1e-32", "5e-28", :bigfloat_linear_algebra),
        V2Precision(:BigFloat512, BigFloat, 512, "1e-50", "5e-46", :bigfloat_linear_algebra),
        V2Precision(:BigFloat1024, BigFloat, 1024, "1e-80", "5e-74", :bigfloat_linear_algebra),
    )
end

const _COMPATIBILITY_TIERS = Dict(
    :instant => :small, :small => :small, :medium => :medium,
    :heavy => :large, :large => :large, :extreme => :extreme,
)

"""Map the retired V1 tier names without changing their result schema."""
function compatibility_tier(tier::Symbol)
    haskey(_COMPATIBILITY_TIERS, tier) ||
        throw(ArgumentError("unknown compatibility tier $(repr(tier)); expected instant, small, medium, heavy, large, or extreme"))
    _COMPATIBILITY_TIERS[tier]
end

"""Fail-closed validation of the reviewed precision contract."""
function _validate_precision_spec(precision::V2Precision)
    expected = Dict(
        :Float64 => (53, "1e-8", "5e-7", :cholmod),
        :Float64x2 => (104, "1e-15", "5e-13", :multifloat_linear_algebra),
        :Float64x3 => (156, "1e-21", "5e-18", :multifloat_linear_algebra),
        :Float64x4 => (208, "1e-28", "5e-22", :multifloat_linear_algebra),
        :BigFloat256 => (256, "1e-32", "5e-28", :bigfloat_linear_algebra),
        :BigFloat512 => (512, "1e-50", "5e-46", :bigfloat_linear_algebra),
        :BigFloat1024 => (1024, "1e-80", "5e-74", :bigfloat_linear_algebra),
    )
    haskey(expected, precision.name) || throw(ArgumentError(
        "unsupported reviewed precision $(precision.name)"))
    bits, solver, cert, provider = expected[precision.name]
    precision.bits == bits || throw(ArgumentError(
        "precision $(precision.name) does not match the reviewed bit width"))
    # `:standard`, `:test`, and `:generic` are explicit local overrides used
    # by unit tests and exploratory callers.  They still retain the reviewed
    # arithmetic width, but are not mislabeled as a production provider.
    if precision.provider in (:standard, :test, :generic)
        return true
    end
    (precision.solver_tolerance, precision.certificate_limit,
     precision.provider) == (solver, cert, provider) || throw(ArgumentError(
        "precision $(precision.name) does not match the reviewed arithmetic matrix"))
    return true
end

for precision in reviewed_precision_specs()
    _validate_precision_spec(precision)
end

"""Check the original-coordinate residual fields against a V2 limit.

Missing fields fail closed.  Infeasibility certificates are checked by their
independent Farkas/ray oracle instead; this predicate is for optimal results.
"""
function certificate_gate(certificate, precision::V2Precision)
    _validate_precision_spec(precision)
    for field in (:primal_residual_scaled, :dual_residual_scaled, :relative_gap)
        hasproperty(certificate, field) || return false
        value = try BigFloat(getproperty(certificate, field)) catch; return false end
        isfinite(value) && value >= 0 || return false
        value <= BigFloat(precision.certificate_limit) || return false
    end
    return true
end

"""Check the normalized separation margin for an infeasibility ray.

The native result stores a primal-infeasibility Farkas pairing in
`dual_objective`, and a dual-infeasibility improving pairing in
`primal_objective`.  Require a 100x certificate-limit margin in addition to
the native validity bit; this prevents a numerically marginal ray from being
classified as a certified benchmark result.
"""
function ray_certificate_gate(certificate, precision::V2Precision,
                               expected_status::Symbol)
    _validate_precision_spec(precision)
    field = expected_status === :primal_infeasible ? :dual_objective :
        expected_status === :dual_infeasible ? :primal_objective : nothing
    field === nothing && return false
    hasproperty(certificate, field) || return false
    pairing = try BigFloat(getproperty(certificate, field)) catch; return false end
    isfinite(pairing) || return false
    margin = BigFloat(100) * BigFloat(precision.certificate_limit)
    expected_status === :primal_infeasible ? pairing >= margin : pairing <= -margin
end

"""Validate a fixed-format SHA256 manifest and all referenced files."""
function validate_manifest(path::AbstractString; root::AbstractString=dirname(path))
    isfile(path) || throw(ArgumentError("checksum manifest is missing: $path"))
    rows = Tuple{String,String}[]
    seen = Set{String}()
    for (line_number, raw) in enumerate(eachline(path))
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        match_result = match(r"^([0-9A-Fa-f]{64})[[:space:]]+(.+)$", line)
        match_result === nothing && throw(ArgumentError(
            "invalid checksum manifest line $line_number"))
        digest, relative = lowercase(match_result.captures[1]), match_result.captures[2]
        relative = normpath(relative)
        (!isabspath(relative) && relative != ".." && !startswith(relative, "../")) ||
            throw(ArgumentError("manifest path escapes root: $relative"))
        relative in seen && throw(ArgumentError("duplicate manifest path: $relative"))
        push!(seen, relative)
        file = joinpath(root, relative)
        isfile(file) || throw(ArgumentError("manifest file is missing: $relative"))
        actual = bytes2hex(SHA.sha256(read(file)))
        actual == digest || throw(ArgumentError("checksum mismatch for $relative"))
        push!(rows, (digest, relative))
    end
    isempty(rows) && throw(ArgumentError("checksum manifest is empty"))
    return rows
end

manifest_fingerprint(path::AbstractString; root::AbstractString=dirname(path)) =
    _hex((:sha256_manifest_v1, sort(validate_manifest(path; root))))

# Length-prefixed canonical bytes. All integer lengths are explicitly
# big-endian; no host-endian serialization participates in identity.
function _put_u64be!(io::IO, value::UInt64)
    for shift in (56, 48, 40, 32, 24, 16, 8, 0)
        write(io, UInt8((value >> shift) & 0xff))
    end
end
function _put_bytes!(io::IO, tag::UInt8, bytes)
    write(io, tag); _put_u64be!(io, UInt64(length(bytes))); write(io, bytes)
end
function _put_u16be!(io::IO, value::UInt16)
    write(io, UInt8(value >> 8), UInt8(value & 0xff))
end
function _put_u32be!(io::IO, value::UInt32)
    for shift in (24, 16, 8, 0); write(io, UInt8((value >> shift) & 0xff)); end
end
function _put!(io::IO, value)
    if value === nothing
        write(io, UInt8(0x00))
    elseif value isa Bool
        write(io, UInt8(value ? 0x02 : 0x01))
    elseif value isa Symbol
        _put_bytes!(io, 0x09, codeunits(String(value)))
    elseif value isa AbstractString
        _put_bytes!(io, 0x03, codeunits(String(value)))
    elseif value isa Rational
        write(io, UInt8(0x0a)); _put!(io, numerator(value)); _put!(io, denominator(value))
    elseif value isa Int8
        _put_bytes!(io, 0x40, codeunits(string(value)))
    elseif value isa Int16
        _put_bytes!(io, 0x41, codeunits(string(value)))
    elseif value isa Int32
        _put_bytes!(io, 0x42, codeunits(string(value)))
    elseif value isa Int64
        _put_bytes!(io, 0x43, codeunits(string(value)))
    elseif value isa Int128
        _put_bytes!(io, 0x44, codeunits(string(value)))
    elseif value isa UInt8
        _put_bytes!(io, 0x45, codeunits(string(value)))
    elseif value isa UInt16
        _put_bytes!(io, 0x46, codeunits(string(value)))
    elseif value isa UInt32
        _put_bytes!(io, 0x47, codeunits(string(value)))
    elseif value isa UInt64
        _put_bytes!(io, 0x48, codeunits(string(value)))
    elseif value isa UInt128
        _put_bytes!(io, 0x49, codeunits(string(value)))
    elseif value isa Float16
        write(io, UInt8(0x0c)); _put_u16be!(io, reinterpret(UInt16, value))
    elseif value isa Float32
        write(io, UInt8(0x0d)); _put_u32be!(io, reinterpret(UInt32, value))
    elseif value isa Float64
        write(io, UInt8(0x0e)); _put_u64be!(io, reinterpret(UInt64, value))
    elseif value isa BigFloat
        write(io, UInt8(0x0f)); _put!(io, precision(value)); _put_bytes!(io, 0x10, codeunits(string(value)))
    elseif value isa AbstractFloat
        _put_bytes!(io, 0x11, codeunits(string(typeof(value)) * ":" * repr(value)))
    elseif value isa Type
        _put_bytes!(io, 0x12, codeunits(string(value)))
    elseif value isa NamedTuple
        write(io, UInt8(0x05)); _put!(io, collect(keys(value))); _put!(io, collect(values(value)))
    elseif value isa Tuple
        write(io, UInt8(0x06)); _put_u64be!(io, UInt64(length(value)))
        for item in value; _put!(io, item); end
    elseif value isa AbstractArray
        # Arrays carry container type, rank and element type even when empty;
        # this prevents collisions between e.g. Vector{Int}[] and Matrix{Int}.
        write(io, UInt8(0x07)); _put!(io, string(typeof(value)))
        _put!(io, ndims(value)); _put!(io, string(eltype(value))); _put!(io, size(value))
        for item in value; _put!(io, item); end
    elseif value isa AbstractDict
        write(io, UInt8(0x08)); pairs_sorted = sort!(collect(value); by=x -> _canonical_bytes(first(x)) )
        _put!(io, length(pairs_sorted)); for (k, v) in pairs_sorted; _put!(io, k); _put!(io, v); end
    elseif value isa V2Tier
        _put!(io, (value.name, value.lane, value.wall_seconds, value.memory_bytes, value.solve_policy))
    elseif value isa V2Reference
        _put!(io, (value.status, value.expected_status, value.disposition,
                   value.certificate_kind, value.objective_interval,
                   value.oracle, value.note, value.prior_observed_status))
    elseif value isa V2Transform
        _put!(io, (value.source_problem_type, value.target_cone_program,
                   value.transform_id, value.version, value.exactness,
                   value.positive_prefactor_factored, value.positive_prefactor_proof,
                   value.lifting_dimensions, value.validation_receipts, value.fingerprint))
    else
        # Include a type tag and length-delimited fallback text. This is a
        # last-resort identity encoding, never a mathematical witness format.
        _put_bytes!(io, 0x13, codeunits(string(typeof(value))))
        _put_bytes!(io, 0x14, codeunits(string(value)))
    end
    return io
end

_canonical_bytes(value) = (io=IOBuffer(); _put!(io, value); take!(io))
_hex(value) = bytes2hex(SHA.sha256(_canonical_bytes(value)))

# Source-artifact metadata (generator IDs, issue/provenance notes) is not
# mathematical input. Native artifacts specialize this hook below. The V1
# compatibility payload is projected to its problem type and mathematical
# parameters, deliberately excluding BenchmarkSpec ID/source/reference fields.
function _math_payload(value)
    if hasproperty(value, :problem) && hasproperty(value, :params) &&
       hasproperty(value, :id) && hasproperty(value, :source)
        return (problem_type=string(typeof(getproperty(value, :problem))),
                params=getproperty(value, :params))
    end
    return value
end

# Pure mathematics deliberately excludes stable IDs, train/holdout/sentinel
# split labels and provenance. This lets cross-split duplicate checks compare
# the actual finite mathematical instance rather than metadata.
mathematical_fingerprint(instance::V2Instance) = _hex((
    :schema, V2_SCHEMA_VERSION, :family, instance.family,
    :axis_values, instance.axis_values,
    :payload, _math_payload(instance.payload),
))

input_fingerprint(instance::V2Instance) = _hex((
    :mathematical, mathematical_fingerprint(instance),
    :transform, get(instance.provenance, :transform, nothing),
))

catalog_fingerprint(catalog::V2Catalog) = _hex((
    :schema, V2_SCHEMA_VERSION, :catalog, catalog.name, :version, catalog.version,
    :families, [(f.name, f.axes, f.transforms) for f in catalog.families],
    :instances, [(x.id, x.family, x.tier, x.axis_values, x.split, x.source,
                  x.provenance, x.checksum, x.resource, x.reference, x.payload)
                 for x in catalog.instances],
    :suites, catalog.suites,
))

function execution_fingerprint(instance::V2Instance, precision::V2Precision;
                               route::Symbol=:auto, manifest::AbstractString="",
                               settings=nothing)
    return _hex((:input, input_fingerprint(instance), :precision, precision.name,
                 :arithmetic, precision.arithmetic, :bits, precision.bits,
                 :solver_tolerance, precision.solver_tolerance,
                 :certificate_limit, precision.certificate_limit,
                 :provider, precision.provider, :route, route, :settings, settings,
                 :manifest, manifest))
end

reference_interval(ref::V2Reference) = ref.objective_interval

"""Classify one observed run using the explicit five-state reference table.
A prior failure may only become RESOLVED after every current oracle, interval,
certificate-kind and residual gate passes; an unexpected xfail success is XPASS.
"""
function classify_disposition(expected_status::Symbol, prior_failure::Bool,
        observed_status::Symbol, oracle_ok::Bool, interval_ok::Bool,
        certificate_valid::Bool, certificate_kind_ok::Bool, failures=Symbol[];
        prior_observed_status::Union{Nothing,Symbol}=nothing)
    gates = oracle_ok && interval_ok && certificate_valid && certificate_kind_ok
    if expected_status === :build_only
        return observed_status === :build_only && oracle_ok && interval_ok &&
               certificate_kind_ok ? :PASS : :FAIL
    elseif prior_observed_status !== nothing
        # XFAIL applies only to the explicitly recorded prior failure.  A
        # certified result at the real semantic target is RESOLVED; an
        # unrelated (including contradictory optimal) result is FAIL.
        if observed_status === prior_observed_status && oracle_ok && interval_ok &&
           !certificate_valid && certificate_kind_ok && !isempty(failures)
            return :XFAIL
        elseif observed_status === expected_status && gates
            return :RESOLVED
        end
        return :FAIL
    elseif expected_status === :iteration_limit || expected_status === :numerical_breakdown
        if observed_status === expected_status && oracle_ok && !certificate_valid &&
           certificate_kind_ok && !isempty(failures)
            return :XFAIL
        elseif observed_status === :optimal && gates
            return prior_failure ? :RESOLVED : :XPASS
        end
        return :FAIL
    elseif observed_status === expected_status && gates
        return prior_failure ? :RESOLVED : :PASS
    end
    return :FAIL
end

function expand(axes::AbstractVector{<:V2Axis})
    ordered = sort(collect(axes); by=x -> String(x.name))
    result = NamedTuple[]
    function visit(index, values)
        if index > length(ordered)
            push!(result, (; values...)); return
        end
        axis = ordered[index]
        values_sorted = sort(axis.values; by=value -> string(value))
        for value in values_sorted
            visit(index + 1, [values..., axis.name => axis.canonicalize(value)])
        end
    end
    visit(1, Pair{Symbol,Any}[])
    return result
end

function _family_map(catalog::V2Catalog)
    map = Dict{Symbol,Any}()
    for family in catalog.families
        haskey(map, family.name) && throw(ArgumentError("duplicate V2 family $(family.name)"))
        map[family.name] = family
    end
    map
end

function validate_catalog(catalog::V2Catalog)
    families = _family_map(catalog)
    ids = Set{Symbol}()
    suite_sets = Dict(
        :train => Set(Symbol.(catalog.suites.train)),
        :holdout => Set(Symbol.(catalog.suites.holdout)),
        :sentinel => Set(Symbol.(catalog.suites.sentinel)),
    )
    sum(length, values(suite_sets)) == length(union(values(suite_sets)...)) ||
        throw(ArgumentError("V2 suites overlap"))
    for (name, ids_in_suite) in ((:train, catalog.suites.train),
                                  (:holdout, catalog.suites.holdout),
                                  (:sentinel, catalog.suites.sentinel))
        length(ids_in_suite) == length(suite_sets[name]) ||
            throw(ArgumentError("V2 suite $name contains duplicate IDs"))
    end
    math_fingerprints = Dict{String,Symbol}()
    for instance in catalog.instances
        instance.id in ids && throw(ArgumentError("duplicate V2 instance $(instance.id)"))
        push!(ids, instance.id)
        haskey(families, instance.family) || throw(ArgumentError("unknown family $(instance.family)"))
        if instance.payload isa AbstractV2SourceArtifact
            instance.payload.id == instance.id || throw(ArgumentError(
                "payload ID $(instance.payload.id) does not match instance $(instance.id)"))
            artifact_family = instance.payload isa LPArtifact ? :lp :
                instance.payload isa SOCPArtifact ? :soc :
                instance.payload isa RSOCArtifact ? :rsoc :
                instance.payload isa SDPArtifact ? :sdp :
                instance.payload isa IllConditionedArtifact ? :ill_conditioned :
                isdefined(@__MODULE__, :MixedArtifact) && instance.payload isa MixedArtifact ? :mixed :
                hasproperty(instance.payload, :family) ? getproperty(instance.payload, :family) : nothing
            artifact_family == instance.family || throw(ArgumentError(
                "payload family $(artifact_family) does not match instance $(instance.family)"))
        end
        isempty(instance.checksum) && throw(ArgumentError("missing checksum for $(instance.id)"))
        instance.split in (:train, :holdout, :sentinel) ||
            throw(ArgumentError("invalid split $(instance.split) for $(instance.id)"))
        instance.id in suite_sets[instance.split] ||
            throw(ArgumentError("instance $(instance.id) is absent from its suite"))
            (instance.reference.status === :build_only) ==
            (instance.reference.certificate_kind === :build_only) ||
            throw(ArgumentError("build-only reference mismatch for $(instance.id)"))
        instance.reference.status === :xfail && instance.reference.disposition !== :XFAIL &&
            throw(ArgumentError("xfail disposition mismatch for $(instance.id)"))
        instance.reference.status === :xfail && instance.reference.expected_status === :xfail &&
            throw(ArgumentError("xfail must retain a concrete observed solver status for $(instance.id)"))
        instance.reference.status === :xfail &&
            get(instance.provenance, :solve_eligible, false) === true &&
            throw(ArgumentError("XFAIL instances are optimizer-ineligible"))
        if instance.payload isa AbstractV2SourceArtifact
            instance.checksum == _hex(instance.payload) ||
                throw(ArgumentError("artifact checksum mismatch for $(instance.id)"))
            declared = get(instance.provenance, :transform, nothing)
            declared isa V2Transform || throw(ArgumentError(
                "source artifact instance $(instance.id) must declare its transform"))
            # Compare mathematical source data only.  Typed artifact
            # serializers intentionally retain IDs/generator metadata for
            # provenance, but those fields must not make duplicate math look
            # distinct across train/holdout splits.
            math_fp = _hex((instance.family, _math_payload(instance.payload)))
            haskey(math_fingerprints, math_fp) &&
                throw(ArgumentError("duplicate mathematical V2 artifact across splits: $(instance.id) and $(math_fingerprints[math_fp])"))
            math_fingerprints[math_fp] = instance.id
        end
    end
    union(values(suite_sets)...) == ids ||
        throw(ArgumentError("V2 suites must partition all instance IDs"))
    return true
end

function _declared_transform(instance::V2Instance)
    value = get(instance.provenance, :transform, nothing)
    return value isa V2Transform ? value : nothing
end

training_instances(catalog::V2Catalog) =
    filter(instance -> instance.split === :train &&
                       get(instance.provenance, :solve_eligible, false) === true &&
                       instance.reference.status !== :build_only &&
                       instance.reference.status !== :xfail,
           catalog.instances)

holdout_instances(catalog::V2Catalog) =
    filter(instance -> instance.split === :holdout, catalog.instances)

sentinel_instances(catalog::V2Catalog) =
    filter(instance -> instance.split === :sentinel, catalog.instances)

function build_instance(catalog::V2Catalog, instance::V2Instance, precision::V2Precision)
    # A direct BigFloat build must honor the requested precision rather than
    # inheriting the caller's ambient context. The recursive guard is reached
    # only after entering the requested precision scope.
    if precision.arithmetic === BigFloat && Base.precision(BigFloat) != precision.bits
        return setprecision(BigFloat, precision.bits) do
            build_instance(catalog, instance, precision)
        end
    end
    family = only(filter(f -> f.name === instance.family, catalog.families))
    started = time_ns()
    built = family.build(instance, precision)
    built isa V2Built || throw(ArgumentError("V2 builder must return V2Built"))
    built.input_fingerprint == input_fingerprint(instance) ||
        throw(ArgumentError("builder input fingerprint does not match instance"))
    built.transform.exactness in (:identity, :exact_univariate_halfline,
        :exact_univariate_matrix_halfline_if_proved, :sos_relaxation,
        :finite_grid_surrogate) || throw(ArgumentError("invalid transform metadata"))
    declared = _declared_transform(instance)
    declared === nothing && throw(ArgumentError("every V2 instance must own an explicit transform contract"))
    expected_transform = isdefined(@__MODULE__, :_expected_transform) ?
        _expected_transform(instance) : declared
    built.transform == expected_transform || throw(ArgumentError(
        "builder transform does not match independently derived instance contract"))
    built.transform == declared || throw(ArgumentError("builder transform does not match instance transform contract"))
    if instance.payload isa AbstractV2SourceArtifact
        hasproperty(built.facts, :artifact_fingerprint) &&
            built.facts.artifact_fingerprint == instance.checksum ||
            throw(ArgumentError("builder facts do not bind source artifact fingerprint"))
        hasproperty(built.facts, :model_fingerprint) ||
            throw(ArgumentError("builder must publish canonical generated-model fingerprint"))
        actual_model_fingerprint = isdefined(@__MODULE__, :_native_model_fingerprint) ?
            _native_model_fingerprint(built.problem) : nothing
        actual_model_fingerprint === nothing ||
            actual_model_fingerprint == built.facts.model_fingerprint ||
            throw(ArgumentError("builder model fingerprint does not match actual built model"))
        actual_model_fingerprint == "0"^64 &&
            throw(ArgumentError("zero generated-model fingerprint is forbidden"))
        occursin(r"^[0-9a-f]{64}$", String(built.facts.model_fingerprint)) ||
            throw(ArgumentError("invalid generated-model fingerprint"))
    end
    elapsed = (time_ns() - started) * 1.0e-9
    return built, elapsed
end

function _parse(::Type{T}, text::String) where {T}
    # Float64 has no string constructor (use Base.parse); BigFloat and the
    # MultiFloats backends construct exactly from decimal strings, which
    # Base.parse cannot do for MultiFloats (no tryparse method).
    T === Float64 ? parse(T, text) : T(text)
end

"""Solve an ordinary V2 instance via the existing public generic builder."""
function _run_instance_impl(catalog::V2Catalog, instance::V2Instance,
                      precision::V2Precision; settings=nothing, outputs=nothing)
    instance.reference.status === :build_only && throw(ArgumentError(
        "build-only instance $(instance.id) requires an explicit solve contract"))
    built, setup = build_instance(catalog, instance, precision)
    model = built.problem
    T = precision.arithmetic
    _validate_precision_spec(precision)
    if settings === nothing
        # Keep the requested solver tolerance distinct from the wider
        # certificate acceptance limit in the reviewed matrix.
        tol = _parse(T, precision.solver_tolerance)
        settings = SDPX.Settings{T}(
            tolerances=SDPX.Tolerances{T}(primal=tol, dual=tol, gap=tol),
            limits=SDPX.Limits(threads=1), verbosity=0, certification=true,
        )
    end
    measurement = if outputs === nothing
        @timed SDPX.optimize!(model; settings)
    else
        @timed SDPX.optimize!(model; settings, outputs)
    end
    solved = measurement.value
    certificate = SDPX.certificate(solved)
    objective = string(certificate.primal_objective)
    comparison_bits = max(256, 2 * precision.bits)
    interval_ok = setprecision(BigFloat, comparison_bits) do
        if instance.reference.objective_interval === nothing
            instance.reference.status !== :optimal
        else
            lower, upper = instance.reference.objective_interval
            value = try BigFloat(objective) catch; BigFloat(NaN) end
            isfinite(value) && BigFloat(lower) <= value <= BigFloat(upper)
        end
    end
    # Independent exact oracles must not inherit a low ambient BigFloat
    # precision from the caller.  This scope is also the required
    # max(256, 2*bits) reevaluation boundary for every optimal artifact.
    oracle_ok = setprecision(BigFloat, comparison_bits) do
        instance.reference.oracle === nothing ?
            instance.reference.status === :build_only : instance.reference.oracle(built, certificate)
    end
    # Infeasibility is independently certified by the exact Farkas oracle;
    # public certificate.valid is optimal-only.  Resolve an XFAIL only when
    # the observed status is the real semantic target and its exact oracle
    # passes; a prior numerical failure remains uncertified.
    observed_status = SDPX.status(solved)
    cert_ok = instance.reference.status === :build_only ||
        (instance.reference.expected_status in (:primal_infeasible, :dual_infeasible) ?
            (observed_status === instance.reference.expected_status &&
             certificate.valid &&
             ray_certificate_gate(certificate, precision,
                                  instance.reference.expected_status) && oracle_ok) :
            certificate.valid && certificate_gate(certificate, precision))
    failures = Symbol[]
    interval_ok || push!(failures, :objective_interval)
    oracle_ok || push!(failures, :oracle)
    cert_ok || push!(failures, :certificate)
    status_ok = observed_status === instance.reference.expected_status
    instance.reference.status === :build_only && (status_ok = true)
    validation_status = observed_status
    reference_ok = oracle_ok && interval_ok && status_ok && cert_ok
    semantic_ok = reference_ok
    cert_kind_ok = instance.reference.status === :build_only ||
        (instance.reference.expected_status in (:primal_infeasible, :dual_infeasible) ?
            ((instance.reference.expected_status === :primal_infeasible &&
              instance.reference.certificate_kind === :farkas) ||
             (instance.reference.expected_status === :dual_infeasible &&
              instance.reference.certificate_kind === :ray)) :
            instance.reference.certificate_kind === :optimal)
    prior_failure = instance.reference.disposition in (:XFAIL, :FAIL)
    disposition = classify_disposition(instance.reference.expected_status, prior_failure,
        observed_status, oracle_ok, interval_ok, cert_ok, cert_kind_ok, failures;
        prior_observed_status=instance.reference.prior_observed_status)
    if disposition === :XFAIL
        validation_status = :XFAIL
        semantic_ok = false
        reference_ok = true
        push!(failures, :xfail_expected_failure)
    elseif disposition === :XPASS
        validation_status = :XPASS
        semantic_ok = false
        reference_ok = false
        push!(failures, :xpass_unexpected_success)
    elseif disposition !== :PASS && disposition !== :RESOLVED
        semantic_ok = false
    end
    status_ok || push!(failures, :status)
    validation = V2Validation(validation_status, SDPX.status(solved), disposition,
        cert_ok, reference_ok, failures)
    diagnostics = try SDPX.diagnostics(solved) catch; nothing end
    timings = diagnostics === nothing ? nothing : getproperty(diagnostics, :timings)
    core_seconds = timings === nothing ? measurement.time : get(timings, :core, measurement.time)
    recovery_seconds = timings === nothing ? nothing : get(timings, :reconstruction, nothing)
    return V2RunResult(
        instance.id, instance.family, instance.tier.name, precision.name, precision.bits,
        SDPX.status(solved), certificate.valid, objective,
        string(certificate.dual_objective), string(certificate.primal_residual),
        string(certificate.dual_residual), string(certificate.relative_gap), solved.iterations,
        setup, core_seconds, recovery_seconds, measurement.bytes,
        input_fingerprint(instance), execution_fingerprint(instance, precision), validation,
        _route_receipt(diagnostics),
    )
end

function _route_receipt(diagnostics)
    names = (:requested_route, :planned_route, :executed_route,
             :requested_formulation, :planned_formulation, :executed_formulation,
             :requested_backend, :planned_backend, :executed_backend,
             :requested_provider, :planned_provider, :executed_provider,
             :requested_kernel, :planned_kernel, :executed_kernel, :reuse)
    selected = diagnostics === nothing ? nothing : try getproperty(diagnostics, :selected_algorithms) catch; nothing end
    aliases = Dict(
        :requested_route => (:requested_kkt_route, :requested_route),
        :planned_route => (:planned_kkt_route, :planned_route),
        :executed_route => (:executed_kkt_route, :executed_route),
        :requested_formulation => (:requested_kkt_formulation, :requested_formulation),
        :planned_formulation => (:planned_kkt_formulation, :planned_formulation),
        :executed_formulation => (:executed_kkt_formulation, :executed_formulation),
        :requested_backend => (:requested_backend,),
        :planned_backend => (:planned_backend, :planned_algorithm, :planned_backend),
        :executed_backend => (:executed_backend, :executed_algorithm, :executed_backend),
        :requested_provider => (:requested_provider,),
        :planned_provider => (:planned_la_provider, :planned_provider),
        :executed_provider => (:la_executed_provider, :executed_provider),
        :requested_kernel => (:requested_kernel,),
        :planned_kernel => (:planned_factorization_kernel, :planned_kernel),
        :executed_kernel => (:executed_factorization_kernel, :executed_kernel),
        :reuse => (:executed_factorization_reuse, :factorization_reuse, :reuse),
    )
    pick(name) = selected === nothing ? "not_declared_by_api" :
        try
            value = first((getproperty(selected, alias) for alias in aliases[name]
                           if hasproperty(selected, alias)), nothing)
            value === nothing ? "not_declared_by_api" : string(value)
        catch
            "not_declared_by_api"
        end
    return NamedTuple{names}(Tuple(pick(name) for name in names))
end

function run_instance(catalog::V2Catalog, instance::V2Instance,
                      precision::V2Precision; settings=nothing, outputs=nothing)
    T = precision.arithmetic
    T === BigFloat && return setprecision(BigFloat, precision.bits) do
        _run_instance_impl(catalog, instance, precision; settings, outputs)
    end
    return _run_instance_impl(catalog, instance, precision; settings, outputs)
end

"""Adapt existing generic specs without changing their V1 registry/API."""
function adapt_generic_specs(specs; source_prefix="generic-v1",
                              generic_module=Main.GenericConicBenchmark)
    isempty(specs) && return V2Catalog(:general_v2, V2_SCHEMA_VERSION, Any[], V2Instance[])
    family_names = sort(unique(Symbol[s.family for s in specs]); by=String)
    families = Any[]
    for name in family_names
        build_fn = function (instance, precision)
            spec = instance.payload
            params = precision.arithmetic === BigFloat ?
                merge(spec.params, (precision_bits=precision.bits,)) : spec.params
            model = generic_module.build(spec.problem, precision.arithmetic, params)
                transform = V2Transform(:generic_conic_model, :sdpx_cone_program,
                :identity, 1, :identity;
                validation_receipts=(coefficient_match=true,
                                     source_reconstruction=true))
            return V2Built(model, nothing, spec, input_fingerprint(instance), transform,
                (source_dimension=0, target_dimension=0, transform_exact=true),
                (setup_seconds=nothing,))
        end
        oracle_fn = (built, certificate) -> begin
            spec = built.source_artifact
            spec.known_objective === nothing || isapprox(certificate.primal_objective,
                spec.known_objective; atol=spec.objective_tolerance, rtol=spec.objective_tolerance)
        end
        validate_fn = (instance, result) -> result.validation
        push!(families, V2Family(name, V2Axis[], build_fn, oracle_fn, validate_fn, (:identity,)))
    end
    instances = V2Instance[]
    for spec in specs
        tier = only(filter(t -> t.name === spec.tier, _TIERS))
        # V1 remains compatibility-only: preserve its declared status as
        # metadata, but never promote a V1 finding/objective to a V2 oracle.
        v1_status = spec.expected_status
        ref = V2Reference(:build_only, :build_only, nothing, nothing,
            "V1 compatibility-only; original expected_status=$(v1_status) retained as metadata";
            expected_status=:build_only, disposition=:PASS)
        checksum = _hex((source=spec.source, id=spec.id, params=spec.params,
                         objective=spec.known_objective))
        compatibility_transform = V2Transform(:generic_conic_model,
            :sdpx_cone_program, :identity, 1, :identity;
            validation_receipts=(coefficient_match=true,
                                 source_reconstruction=true))
        push!(instances, V2Instance(spec.id, spec.family, tier, spec.params, :train,
            source_prefix * "/" * spec.source,
            (source=spec.source, v1_id=spec.id, compatibility_only=true,
             solve_eligible=false, v1_expected_status=v1_status,
             transform=compatibility_transform), checksum,
            (wall_seconds=tier.wall_seconds, memory_bytes=tier.memory_bytes), ref, spec))
    end
    return V2Catalog(:general_v2, V2_SCHEMA_VERSION, families, instances,
                     (train=Symbol[s.id for s in instances], holdout=Symbol[], sentinel=Symbol[]))
end

include(joinpath(@__DIR__, "native_catalog.jl"))

end # module
