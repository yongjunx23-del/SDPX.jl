module GeneralBenchmarkV2

using SHA
import SDPX

export V2_SCHEMA_VERSION, V2Axis, V2Tier, V2Precision, V2Reference,
    AbstractV2SourceArtifact, V2ConicArtifact, native_v2_catalog,
    V2Transform, V2Family, V2Instance, V2Catalog, V2Built, V2Validation,
    V2RunResult, expand, validate_catalog, catalog_fingerprint,
    input_fingerprint, execution_fingerprint, adapt_generic_specs,
    build_instance, run_instance, reference_interval, resource_tiers,
    precision_matrix

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
    certificate_kind::Symbol
    objective_interval::Union{Nothing,Tuple{String,String}}
    oracle::O
    note::String
    function V2Reference(status::Symbol, certificate_kind::Symbol,
                         objective_interval, oracle, note::AbstractString="")
        status in (:optimal, :primal_infeasible, :dual_infeasible, :build_only,
                   :discretized, :xfail) ||
            throw(ArgumentError("unsupported reference status $status"))
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
        status === :xfail && isempty(note) &&
            throw(ArgumentError("xfail reference requires an issue note"))
        new{typeof(oracle)}(status, certificate_kind, interval, oracle, String(note))
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
    status::Symbol
    certificate::Bool
    reference::Bool
    failures::Vector{Symbol}
end

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

# Length-prefixed canonical bytes. All integer lengths are explicitly
# big-endian; no host-endian serialization participates in identity.
function _put_u64be!(io::IO, value::UInt64)
    for shift in (56, 48, 40, 32, 24, 16, 8, 0)
        write(io, UInt8((value >> shift) & 0xff))
    end
end
function _put!(io::IO, value)
    if value === nothing
        write(io, UInt8(0x00))
    elseif value isa Bool
        write(io, UInt8(value ? 0x02 : 0x01))
    elseif value isa Symbol
        _put!(io, String(value))
    elseif value isa AbstractString
        bytes = codeunits(String(value)); write(io, UInt8(0x03));
        _put_u64be!(io, UInt64(length(bytes))); write(io, bytes)
    elseif value isa Integer
        text = string(value); write(io, UInt8(0x04)); _put_u64be!(io, UInt64(length(text))); write(io, codeunits(text))
    elseif value isa AbstractFloat
        text = sprint(show, value; context=:canonical=>true)
        _put!(io, text)
    elseif value isa Type
        _put!(io, string(value))
    elseif value isa NamedTuple
        write(io, UInt8(0x05)); _put!(io, collect(keys(value)))
        _put!(io, collect(values(value)))
    elseif value isa Tuple
        write(io, UInt8(0x06)); _put_u64be!(io, UInt64(length(value)))
        for item in value; _put!(io, item); end
    elseif value isa AbstractArray
        write(io, UInt8(0x07)); _put!(io, size(value))
        for item in value; _put!(io, item); end
    elseif value isa AbstractDict
        write(io, UInt8(0x08)); pairs_sorted = sort!(collect(value); by=x -> String(first(x)))
        _put!(io, length(pairs_sorted)); for (k, v) in pairs_sorted; _put!(io, k); _put!(io, v); end
    elseif value isa V2Tier
        _put!(io, (value.name, value.lane, value.wall_seconds, value.memory_bytes, value.solve_policy))
    elseif value isa V2Reference
        _put!(io, (value.status, value.certificate_kind, value.objective_interval, value.note))
    elseif value isa V2Transform
        _put!(io, (value.source_problem_type, value.target_cone_program,
                   value.transform_id, value.version, value.exactness,
                   value.positive_prefactor_factored, value.positive_prefactor_proof,
                   value.lifting_dimensions, value.validation_receipts, value.fingerprint))
    else
        _put!(io, string(typeof(value))); _put!(io, string(value))
    end
    return io
end

_canonical_bytes(value) = (io=IOBuffer(); _put!(io, value); take!(io))
_hex(value) = bytes2hex(SHA.sha256(_canonical_bytes(value)))

input_fingerprint(instance::V2Instance) = _hex((
    :schema, V2_SCHEMA_VERSION, :instance_id, instance.id, :family, instance.family,
    :tier, instance.tier, :axis_values, instance.axis_values, :source, instance.source,
    :provenance, instance.provenance, :checksum, instance.checksum,
    :reference, instance.reference,
    :transform, instance.payload isa V2Transform ? instance.payload : nothing,
    :payload, instance.payload,
    :resource, instance.resource, :split, instance.split,
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
    for instance in catalog.instances
        instance.id in ids && throw(ArgumentError("duplicate V2 instance $(instance.id)"))
        push!(ids, instance.id)
        haskey(families, instance.family) || throw(ArgumentError("unknown family $(instance.family)"))
        isempty(instance.checksum) && throw(ArgumentError("missing checksum for $(instance.id)"))
        instance.split in (:train, :holdout, :sentinel) ||
            throw(ArgumentError("invalid split $(instance.split) for $(instance.id)"))
            (instance.reference.status === :build_only) ==
            (instance.reference.certificate_kind === :build_only) ||
            throw(ArgumentError("build-only reference mismatch for $(instance.id)"))
    end
    return true
end

function build_instance(catalog::V2Catalog, instance::V2Instance, precision::V2Precision)
    family = only(filter(f -> f.name === instance.family, catalog.families))
    started = time_ns()
    built = family.build(instance, precision)
    built isa V2Built || throw(ArgumentError("V2 builder must return V2Built"))
    built.transform.exactness in (:identity, :exact_univariate_halfline,
        :exact_univariate_matrix_halfline_if_proved, :sos_relaxation,
        :finite_grid_surrogate) || throw(ArgumentError("invalid transform metadata"))
    elapsed = (time_ns() - started) * 1.0e-9
    return built, elapsed
end

function _parse(::Type{T}, text::String) where {T}
    T === BigFloat ? T(text) : parse(T, text)
end

"""Solve an ordinary V2 instance via the existing public generic builder."""
function _run_instance_impl(catalog::V2Catalog, instance::V2Instance,
                      precision::V2Precision; settings=nothing, outputs=nothing)
    instance.reference.status === :build_only && throw(ArgumentError(
        "build-only instance $(instance.id) requires an explicit solve contract"))
    built, setup = build_instance(catalog, instance, precision)
    model = built.problem
    T = precision.arithmetic
    if settings === nothing
        tol = _parse(T, precision.solver_tolerance)
        certtol = _parse(T, precision.certificate_limit)
        settings = SDPX.Settings{T}(
            tolerances=SDPX.Tolerances{T}(primal=certtol, dual=certtol, gap=certtol),
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
    oracle_ok = instance.reference.oracle === nothing ?
        instance.reference.status === :build_only : instance.reference.oracle(built, certificate)
    cert_ok = instance.reference.status === :build_only || certificate.valid
    failures = Symbol[]
    oracle_ok || push!(failures, :oracle)
    cert_ok || push!(failures, :certificate)
    status_ok = SDPX.status(solved) === instance.reference.status
    instance.reference.status === :build_only && (status_ok = true)
    status_ok || push!(failures, :status)
    validation = V2Validation(SDPX.status(solved), certificate.valid, oracle_ok && status_ok, failures)
    return V2RunResult(
        instance.id, instance.family, instance.tier.name, precision.name, precision.bits,
        SDPX.status(solved), certificate.valid, objective,
        string(certificate.dual_objective), string(certificate.primal_residual),
        string(certificate.dual_residual), string(certificate.relative_gap), solved.iterations,
        setup, measurement.time, nothing, measurement.bytes,
        input_fingerprint(instance), execution_fingerprint(instance, precision), validation,
    )
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
        ref_status = spec.expected_status === :known_solver_finding ? :xfail : spec.expected_status
        cert_kind = ref_status === :primal_infeasible ? :farkas :
            ref_status === :dual_infeasible ? :ray : ref_status === :xfail ? :interval_or_bound : :optimal
        interval = spec.known_objective === nothing ? nothing :
            (string(spec.known_objective - spec.objective_tolerance), string(spec.known_objective + spec.objective_tolerance))
        ref = V2Reference(ref_status, cert_kind, interval,
            (built, cert) -> begin
                spec.known_objective === nothing || isapprox(cert.primal_objective,
                    spec.known_objective; atol=spec.objective_tolerance, rtol=spec.objective_tolerance)
            end, "Adapted from V1; independent source metadata retained")
        checksum = _hex((source=spec.source, id=spec.id, params=spec.params,
                         objective=spec.known_objective))
        push!(instances, V2Instance(spec.id, spec.family, tier, spec.params, :train,
            source_prefix * "/" * spec.source, (source=spec.source, v1_id=spec.id), checksum,
            (wall_seconds=tier.wall_seconds, memory_bytes=tier.memory_bytes), ref, spec))
    end
    return V2Catalog(:general_v2, V2_SCHEMA_VERSION, families, instances,
                     (train=Symbol[s.id for s in instances], holdout=Symbol[], sentinel=Symbol[]))
end

include("native_catalog.jl")

end # module
