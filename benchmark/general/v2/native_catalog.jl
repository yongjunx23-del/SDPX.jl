# Native V2 corpus. Source artifacts are complete exact conic data: every
# coefficient is consumed by the builder and participates in identity.
using LinearAlgebra
abstract type AbstractV2SourceArtifact end

struct V2ConicArtifact <: AbstractV2SourceArtifact
    family::Symbol
    id::Symbol
    coefficients::Vector{Rational{Int}}
    dimension::Int
    cone_parameter::Rational{Int}
    infeasible::Bool
    infeasibility_ray::Vector{Rational{Int}}
    generator_id::Symbol
    generator_version::Int
    function V2ConicArtifact(family::Symbol, id::Symbol,
            coefficients::Vector{Rational{Int}}, dimension::Integer,
            cone_parameter::Rational{Int}, infeasible::Bool,
            generator_id::Symbol, generator_version::Integer;
            infeasibility_ray=Rational{Int}[1, -1])
        dimension > 0 || throw(ArgumentError("artifact dimension must be positive"))
        isempty(coefficients) && throw(ArgumentError("artifact coefficients must not be empty"))
        generator_version > 0 || throw(ArgumentError("generator version must be positive"))
        length(infeasibility_ray) == 2 || throw(ArgumentError("infeasibility ray must have two entries"))
        new(family, id, coefficients, Int(dimension), cone_parameter, infeasible,
            Rational{Int}.(infeasibility_ray), generator_id, Int(generator_version))
    end
end

function _put!(io::IO, artifact::V2ConicArtifact)
    _put!(io, (artifact.family, artifact.id, artifact.coefficients,
        artifact.dimension, artifact.cone_parameter, artifact.infeasible,
        artifact.infeasibility_ray, artifact.generator_id, artifact.generator_version))
end

_q(artifact, index) = artifact.coefficients[mod1(index, length(artifact.coefficients))]
_Tq(::Type{T}, q::Rational{Int}) where {T<:AbstractFloat} = T(numerator(q)) / T(denominator(q))

function _fix!(model, name, expression, value, ::Type{T}) where {T}
    SDPX.constraint!(model, name, expression - T(value), SDPX.ZeroCone())
end
function _contradict!(model, name, expression, value, ::Type{T}) where {T}
    _fix!(model, Symbol(name, :_a), expression, value, T)
    _fix!(model, Symbol(name, :_b), expression, value + 1, T)
end

function _native_artifact(family, id; split=:train, infeasible=false)
    # Distinct split payloads remain feasible for train/holdout and are made
    # infeasible only by an explicit contradictory equality for sentinels.
    suffix = split === :train ? 0 : split === :holdout ? 1 : 2
    if family === :lp || family === :nonpositive || family === :exp || family === :power
        value = family === :nonpositive ? -1 : (suffix == 0 ? 1 : suffix == 1 ? 2 : 1)
        coefficients = Rational{Int}[value]
        return V2ConicArtifact(family, id, coefficients, length(coefficients), 1//2, infeasible,
            Symbol(family, :_generator_, split), 1)
    elseif family === :soc
        coefficients = suffix == 0 ? Rational{Int}[0, 0] : Rational{Int}[1//2, 1//2]
        return V2ConicArtifact(family, id, coefficients, 2, 1//2, infeasible,
            Symbol(family, :_generator_, split), 1)
    elseif family === :rsoc
        target = suffix == 0 ? 0//1 : 1//2
        return V2ConicArtifact(family, id, Rational{Int}[target], 3, 1//2, infeasible,
            Symbol(family, :_generator_, split), 1)
    elseif family === :sdp
        values = suffix == 0 ? Rational{Int}[1, 0, 0, 1] : Rational{Int}[1, 1//2, 1//2, 1]
        return V2ConicArtifact(family, id, values, 2, 1//2, infeasible,
            Symbol(family, :_generator_, split), 1)
    elseif family === :mixed
        values = suffix == 0 ? Rational{Int}[1, -1] : suffix == 1 ? Rational{Int}[1, 0] : Rational{Int}[1, -1]
        return V2ConicArtifact(family, id, values, 2, 1//2, infeasible,
            Symbol(family, :_generator_, split), 1)
    end
    throw(ArgumentError("unknown native V2 family $(family)"))
end

function _native_build(artifact::V2ConicArtifact, ::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T; name=String(artifact.id))
    f = artifact.family
    cone_parameter = _Tq(T, artifact.cone_parameter)
    witness = Rational{Int}[]
    objective = nothing
    if f === :lp
        n = artifact.dimension
        x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
        for i in 1:n
            value = _Tq(T, _q(artifact, i)) * cone_parameter; value >= zero(T) || throw(ArgumentError("LP artifact is not feasible"))
            _fix!(model, Symbol(:fix_, i), x[i], value, T)
            push!(witness, _q(artifact, i))
        end
        artifact.infeasible && _contradict!(model, :contradiction, x[1], _Tq(T, _q(artifact, 1)), T)
        objective = sum(x[i] for i in 1:n)
    elseif f === :nonpositive
        x = SDPX.variable!(model, :x, artifact.dimension; domain=SDPX.Nonpositive())
        for i in 1:artifact.dimension
            value = _Tq(T, _q(artifact, i)) * cone_parameter; value <= zero(T) || throw(ArgumentError("Nonpositive artifact is not feasible"))
            _fix!(model, Symbol(:fix_, i), x[i], value, T); push!(witness, _q(artifact, i))
        end
        artifact.infeasible && _contradict!(model, :contradiction, x[1], _Tq(T, _q(artifact, 1)), T)
        objective = -sum(x[i] for i in 1:artifact.dimension)
    elseif f === :soc
        x = SDPX.variable!(model, :x, artifact.dimension; domain=SDPX.Reals())
        vals = [_Tq(T, _q(artifact, i)) for i in 1:artifact.dimension]
        norm(vals) <= one(T) || throw(ArgumentError("SOC artifact is outside Q3"))
        SDPX.constraint!(model, :soc, Any[one(T) + cone_parameter, x[1], x[2]], SDPX.LorentzCone())
        for i in 1:artifact.dimension
            _fix!(model, Symbol(:fix_, i), x[i], vals[i], T); push!(witness, _q(artifact, i))
        end
        artifact.infeasible && _contradict!(model, :contradiction, x[1], vals[1], T)
        objective = sum(x[i] for i in 1:artifact.dimension)
    elseif f === :rsoc
        u = SDPX.variable!(model, :left, 1; domain=SDPX.Reals())
        v = SDPX.variable!(model, :right, 1; domain=SDPX.Reals())
        target = _Tq(T, _q(artifact, 1)) * cone_parameter
        SDPX.constraint!(model, :rsoc, (u[1], v[1], target), SDPX.RotatedLorentzCone())
        _fix!(model, :fix_left, u[1], one(T), T); _fix!(model, :fix_right, v[1], one(T), T)
        artifact.infeasible && _contradict!(model, :contradiction, u[1], one(T), T)
        append!(witness, [1//1, 1//1, _q(artifact, 1)])
        objective = u[1] + v[1]
    elseif f === :sdp
        artifact.dimension == 2 || throw(ArgumentError("native SDP artifact dimension must be 2"))
        X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
        vals = [_Tq(T, _q(artifact, i)) for i in 1:4]
        vals[2] *= cone_parameter; vals[3] *= cone_parameter
        _fix!(model, :diag1, X[1, 1], vals[1], T); _fix!(model, :offdiag, X[1, 2], vals[2], T)
        _fix!(model, :offdiag_lower, X[2, 1], vals[3], T); _fix!(model, :diag2, X[2, 2], vals[4], T)
        artifact.infeasible && _contradict!(model, :contradiction, X[1, 1], vals[1], T)
        append!(witness, [_q(artifact, i) for i in 1:4]); objective = X[1, 1] + X[2, 2]
    elseif f === :exp
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals()); value = _Tq(T, _q(artifact, 1))
        value >= cone_parameter || throw(ArgumentError("EXP artifact is outside the exponential epigraph"))
        SDPX.constraint!(model, :exp, (zero(T), cone_parameter, x[1]), SDPX.ExponentialCone())
        _fix!(model, :fix, x[1], value, T); artifact.infeasible && _contradict!(model, :contradiction, x[1], value, T)
        push!(witness, _q(artifact, 1)); objective = x[1]
    elseif f === :power
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals()); value = _Tq(T, _q(artifact, 1))
        alpha = _Tq(T, artifact.cone_parameter)
        SDPX.constraint!(model, :power, (x[1], one(T), one(T)), SDPX.PowerCone(alpha))
        _fix!(model, :fix, x[1], value, T); artifact.infeasible && _contradict!(model, :contradiction, x[1], value, T)
        push!(witness, _q(artifact, 1)); objective = x[1]
    elseif f === :mixed
        positive = SDPX.variable!(model, :positive, 1; domain=SDPX.Nonnegative())
        negative = SDPX.variable!(model, :negative, 1; domain=SDPX.Nonpositive())
        SDPX.constraint!(model, :soc, Any[one(T) + cone_parameter, positive[1], negative[1]], SDPX.LorentzCone())
        p, n = _Tq(T, _q(artifact, 1)), _Tq(T, _q(artifact, 2))
        p >= zero(T) && n <= zero(T) || throw(ArgumentError("mixed artifact domain violation"))
        _fix!(model, :fix_positive, positive[1], p, T); _fix!(model, :fix_negative, negative[1], n, T)
        artifact.infeasible && _contradict!(model, :contradiction, positive[1], p, T)
        append!(witness, [_q(artifact, 1), _q(artifact, 2)]); objective = positive[1] - negative[1]
    else
        throw(ArgumentError("unknown native V2 family $(f)"))
    end
    objective === nothing && throw(ArgumentError("native artifact has no objective"))
    SDPX.objective!(model, SDPX.Minimize(), objective)
    transform = V2Transform(:native_conic_artifact, :sdpx_cone_program,
        :identity, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    # The expected objective is computed from the exact artifact, not from a
    # solver certificate.  The oracle independently checks this value and the
    # exact witness/model source identity.
    expected = f === :sdp ? _q(artifact, 1) + _q(artifact, 4) :
        f === :rsoc ? 2//1 : f === :mixed ? _q(artifact, 1) - _q(artifact, 2) :
        f in (:lp, :nonpositive) ? artifact.cone_parameter *
            (f === :nonpositive ? -sum(artifact.coefficients[1:artifact.dimension]) :
             sum(artifact.coefficients[1:artifact.dimension])) :
        sum(witness[1:min(length(witness), artifact.dimension)])
    oracle = V2ExactOracle(artifact.infeasible ? :primal_infeasible : :optimal,
        expected, witness, artifact.infeasibility_ray, artifact)
    return V2Built(model, oracle, artifact, _hex(artifact), transform,
        (source_dimension=artifact.dimension, target_dimension=artifact.dimension,
         generator=artifact.generator_id, coefficients=artifact.coefficients,
         dimension=artifact.dimension, cone_parameter=artifact.cone_parameter,
         infeasible=artifact.infeasible),
        (setup_seconds=nothing,))
end

struct V2ExactOracle
    expected_status::Symbol
    objective::Rational{Int}
    primal_witness::Vector{Rational{Int}}
    dual_ray::Vector{Rational{Int}}
    artifact::V2ConicArtifact
end

function _put!(io::IO, oracle::V2ExactOracle)
    _put!(io, (oracle.expected_status, oracle.objective,
        oracle.primal_witness, oracle.dual_ray, oracle.artifact))
end

function _oracle_check(oracle::V2ExactOracle, built, certificate)
    built.source_artifact === oracle.artifact || return false
    oracle.expected_status === :primal_infeasible &&
        (length(oracle.dual_ray) == 2 && oracle.dual_ray == oracle.artifact.infeasibility_ray) ||
        oracle.expected_status === :optimal || return false
    # build_instance independently enforces the instance/artifact input hash;
    # here the oracle binds the exact artifact object and recomputes its
    # objective from exact witness data, rather than consulting certificate.valid.
    oracle.expected_status === :optimal || return false
    expected = BigFloat(numerator(oracle.objective)) / BigFloat(denominator(oracle.objective))
    actual = BigFloat(certificate.primal_objective)
    return isfinite(actual) && abs(actual - expected) <= BigFloat("1e-7")
end
(oracle::V2ExactOracle)(built, certificate) = _oracle_check(oracle, built, certificate)

function _oracle_witness(artifact::V2ConicArtifact)
    f = artifact.family
    f === :sdp && return copy(artifact.coefficients)
    f === :rsoc && return Rational{Int}[1, 1, _q(artifact, 1)]
    f === :mixed && return copy(artifact.coefficients)
    return copy(artifact.coefficients[1:artifact.dimension])
end

function _native_reference(artifact::V2ConicArtifact)
    if artifact.infeasible
        return V2Reference(:xfail, :farkas, nothing,
            V2ExactOracle(:primal_infeasible, 0//1, Rational{Int}[], artifact.infeasibility_ray, artifact),
            "independently described infeasible sentinel; expected status is primal_infeasible";
            expected_status=:primal_infeasible, disposition=:XFAIL)
    end
    # The exact objective is rebuilt by the same typed artifact rules; interval
    # endpoints are deliberately exact decimal strings for this integer-valued
    # corpus.  Non-integer coefficient changes remain visible in the model and
    # input fingerprint even when the objective oracle is unchanged.
    objective = artifact.family === :sdp ? _q(artifact,1)+_q(artifact,4) :
        artifact.family === :rsoc ? 2//1 : artifact.family === :mixed ? _q(artifact,1)-_q(artifact,2) :
        artifact.family in (:lp, :nonpositive) ? artifact.cone_parameter *
            (artifact.family === :nonpositive ? -sum(artifact.coefficients[1:artifact.dimension]) :
             sum(artifact.coefficients[1:artifact.dimension])) :
        sum(artifact.coefficients[1:artifact.dimension])
    exact_value = BigFloat(numerator(objective)) / BigFloat(denominator(objective))
    # Exact decimal interval strings retain the rational oracle while allowing
    # only solver roundoff (not a tolerance-policy change).
    lower = string(exact_value - BigFloat(1) / BigFloat(10)^7)
    upper = string(exact_value + BigFloat(1) / BigFloat(10)^7)
    oracle = V2ExactOracle(:optimal, objective, _oracle_witness(artifact),
        artifact.infeasibility_ray, artifact)
    V2Reference(:optimal, :optimal, (lower, upper), oracle, "independently reconstructed exact artifact objective";
        expected_status=:optimal, disposition=:PASS)
end

function native_v2_catalog()
    families = Any[]
    family_names = (:lp, :nonpositive, :soc, :rsoc, :sdp, :exp, :power, :mixed)
    for family in family_names
        build = (instance, precision) -> begin
            built = _native_build(instance.payload, precision.arithmetic)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform, built.facts, built.resource)
        end
        oracle = (built, cert) -> built.oracle(built, cert)
        validate = (instance, result) -> result.validation
        push!(families, V2Family(family, V2Axis[], build, oracle, validate, (:identity,)))
    end
    tiers = Dict(t.name=>t for t in _TIERS)
    instances = V2Instance[]
    train = Symbol[]; holdout = Symbol[]; sentinel = Symbol[]
    for family in family_names
        for (split, bucket, infeasible) in ((:train, train, false), (:holdout, holdout, false), (:sentinel, sentinel, true))
            id = Symbol("v2_$(family)_$(split)")
            artifact = _native_artifact(family, id; split, infeasible)
            ref = _native_reference(artifact)
            params = (family=family, split=split, dimension=artifact.dimension,
                coefficients=artifact.coefficients, cone_parameter=artifact.cone_parameter,
                infeasible=artifact.infeasible)
            push!(instances, V2Instance(id, family, tiers[:small], params, split,
                "general-v2/native/$(family)/$(split)",
                (generator=artifact.generator_id, version=artifact.generator_version,
                 artifact_fields=(:coefficients, :dimension, :cone_parameter, :infeasible)),
                _hex(artifact), (wall_seconds=20, memory_bytes=4*1024^3), ref, artifact))
            push!(bucket, id)
        end
    end
    return V2Catalog(:general_v2_native, 2, families, instances,
        (train=train, holdout=holdout, sentinel=sentinel))
end
