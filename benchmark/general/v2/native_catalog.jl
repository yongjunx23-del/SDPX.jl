# Native V2 corpus.  This file owns source artifacts and builders; it never
# stores or calls a GenericConicBenchmark BenchmarkSpec.
abstract type AbstractV2SourceArtifact end
struct V2ConicArtifact <: AbstractV2SourceArtifact
    family::Symbol
    id::Symbol
    coefficients::Vector{Rational{Int}}
    dimension::Int
    generator_id::Symbol
    generator_version::Int
end

# Explicit source-artifact encoding; never fall back to a host-dependent
# `string(::struct)` representation for benchmark identity.
function _put!(io::IO, artifact::V2ConicArtifact)
    _put!(io, (artifact.family, artifact.id, artifact.coefficients,
        artifact.dimension, artifact.generator_id, artifact.generator_version))
end

function _native_artifact(family, id; dimension=1, generator_id=family)
    V2ConicArtifact(family, id, Rational{Int}[0, 1, -1], dimension, generator_id, 1)
end

function _native_build(artifact::V2ConicArtifact, ::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T; name=String(artifact.id))
    f = artifact.family
    if f === :lp
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :fix, x[1] - one(T), SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
    elseif f === :soc
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[one(T), x[1], x[2]], SDPX.LorentzCone())
        SDPX.constraint!(model, :fix, x[1], SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(), x[2])
    elseif f === :rsoc
        x = SDPX.variable!(model, :x, 3; domain=SDPX.Reals())
        SDPX.constraint!(model, :rsoc, Any[one(T), x[1], x[2]], SDPX.RotatedLorentzCone())
        SDPX.constraint!(model, :fix, x[3], SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1] + x[2])
    elseif f === :sdp
        X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
        SDPX.constraint!(model, :diag1, X[1, 1] - one(T), SDPX.ZeroCone())
        SDPX.constraint!(model, :diag2, X[2, 2] - one(T), SDPX.ZeroCone())
        SDPX.constraint!(model, :offdiag, X[1, 2], SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(), zero(T))
    elseif f === :exp
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
        SDPX.constraint!(model, :exp, (zero(T), one(T), x[1]), SDPX.ExponentialCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
    elseif f === :power
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
        SDPX.constraint!(model, :power, (x[1], one(T), one(T)), SDPX.PowerCone(T(1) / T(2)))
        SDPX.objective!(model, SDPX.Minimize(), x[1])
    elseif f === :mixed
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[one(T), x[1], x[2]], SDPX.LorentzCone())
        SDPX.constraint!(model, :nonnegative, x[1], SDPX.Nonnegative())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
    else
        throw(ArgumentError("unknown native V2 family $(f)"))
    end
    transform = V2Transform(:native_conic_artifact, :sdpx_cone_program,
        :identity, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    return V2Built(model, (built, cert)->cert.valid, artifact,
        _hex(artifact), transform, (source_dimension=artifact.dimension,
        target_dimension=artifact.dimension, generator=artifact.generator_id),
        (setup_seconds=nothing,))
end

function native_v2_catalog()
    families = Any[]
    family_names = (:lp, :soc, :rsoc, :sdp, :exp, :power, :mixed)
    for family in family_names
        build = (instance, precision) -> _native_build(instance.payload, precision.arithmetic)
        oracle = (built, cert) -> cert.valid
        validate = (instance, result) -> result.validation
        push!(families, V2Family(family, V2Axis[], build, oracle, validate, (:identity,)))
    end
    tiers = Dict(t.name=>t for t in _TIERS)
    instances = V2Instance[]
    train = Symbol[]; holdout = Symbol[]; sentinel = Symbol[]
    for (idx, family) in enumerate(family_names)
        id = Symbol("v2_$(family)_train")
        artifact = _native_artifact(family, id)
        ref = V2Reference(:optimal, :optimal, ("0", "100"),
            (built, cert)->cert.valid, "analytic conic fixture")
        push!(instances, V2Instance(id, family, tiers[:small], (family=family,), :train,
            "general-v2/native/$(family)", (generator=artifact.generator_id, version=artifact.generator_version),
            _hex(artifact), (wall_seconds=20, memory_bytes=4*1024^3), ref, artifact)); push!(train,id)
        hid = Symbol("v2_$(family)_holdout")
        hart = _native_artifact(family, hid; generator_id=Symbol("$(family)_holdout"))
        href = V2Reference(:optimal, :optimal, ("0", "100"), (built,cert)->cert.valid, "independent generated holdout")
        push!(instances, V2Instance(hid, family, tiers[:small], (family=family, split=:holdout), :holdout,
            "general-v2/native/$(family)/holdout", (generator=hart.generator_id, version=1), _hex(hart),
            (wall_seconds=20, memory_bytes=4*1024^3), href, hart)); push!(holdout,hid)
        sid = Symbol("v2_$(family)_sentinel")
        sart = _native_artifact(family, sid; generator_id=Symbol("$(family)_sentinel"))
        sref = V2Reference(:xfail, :interval_or_bound, nothing, (built,cert)->false, "sentinel is expected to fail")
        push!(instances, V2Instance(sid, family, tiers[:small], (family=family, split=:sentinel), :sentinel,
            "general-v2/native/$(family)/sentinel", (generator=sart.generator_id, version=1), _hex(sart),
            (wall_seconds=20, memory_bytes=4*1024^3), sref, sart)); push!(sentinel,sid)
    end
    return V2Catalog(:general_v2_native, 2, families, instances,
        (train=train, holdout=holdout, sentinel=sentinel))
end
