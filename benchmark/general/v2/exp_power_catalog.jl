# Typed EXP/Power small-tranche artifacts.  This file is included from
# native_catalog.jl after the common V2 build/fingerprint helpers.

struct ExpArtifact <: AbstractV2SmallArtifact
    family::Symbol
    id::Symbol
    kind::Symbol
    coefficients::Vector{Rational{Int}}
    witness::Vector{Rational{Int}}
    n::Int
    generator_id::Symbol
    generator_version::Int
    function ExpArtifact(id::Symbol, kind::Symbol, coefficients, witness;
                         n::Integer=length(coefficients),
                         generator_id::Symbol=:exp_small_v1,
                         generator_version::Integer=1)
        kind in (:unit_epigraph, :entropy, :logsumexp, :fitting) ||
            throw(ArgumentError("unsupported EXP small-tranche kind $kind"))
        cc = Rational{Int}.(coefficients)
        ww = Rational{Int}.(witness)
        n > 0 || throw(ArgumentError("EXP artifact size must be positive"))
        length(cc) == n || throw(ArgumentError("EXP coefficients must match n"))
        length(ww) > 0 || throw(ArgumentError("EXP witness must not be empty"))
        generator_version > 0 || throw(ArgumentError("generator version must be positive"))
        new(:exp, id, kind, cc, ww, Int(n), generator_id, Int(generator_version))
    end
end

struct PowerArtifact <: AbstractV2SmallArtifact
    family::Symbol
    id::Symbol
    kind::Symbol
    alphas::Vector{Rational{Int}}
    fixed_values::Vector{Rational{Int}}
    weighted_values::Vector{Rational{Int}}
    objective::Rational{Int}
    generator_id::Symbol
    generator_version::Int
    function PowerArtifact(id::Symbol, kind::Symbol, alphas, fixed_values,
                           weighted_values, objective::Rational{Int};
                           generator_id::Symbol=:power_small_v1,
                           generator_version::Integer=1)
        kind in (:separable_p_power, :weighted_mean, :alpha_sweep) ||
            throw(ArgumentError("unsupported Power small-tranche kind $kind"))
        aa = Rational{Int}.(alphas)
        ff = Rational{Int}.(fixed_values)
        ww = Rational{Int}.(weighted_values)
        isempty(aa) || all(a -> 0//1 < a < 1//1, aa) ||
            throw(ArgumentError("Power alphas must lie strictly between zero and one"))
        kind === :weighted_mean || length(ff) == length(aa) ||
            throw(ArgumentError("Power fixed values must match alphas"))
        kind === :weighted_mean && length(ww) == 2 ||
            kind !== :weighted_mean || throw(ArgumentError("weighted mean needs two fixed values"))
        generator_version > 0 || throw(ArgumentError("generator version must be positive"))
        new(:power, id, kind, aa, ff, ww, objective, generator_id, Int(generator_version))
    end
end

function _put!(io::IO, artifact::ExpArtifact)
    _put!(io, (:ExpArtifact, artifact.family, artifact.id, artifact.kind, artifact.coefficients,
        artifact.witness, artifact.n, artifact.generator_id, artifact.generator_version))
end
function _put!(io::IO, artifact::PowerArtifact)
    _put!(io, (:PowerArtifact, artifact.family, artifact.id, artifact.kind, artifact.alphas,
        artifact.fixed_values, artifact.weighted_values, artifact.objective,
        artifact.generator_id, artifact.generator_version))
end

struct V2ExpOracle
    artifact::ExpArtifact
    objective_interval::Tuple{String,String}
    witness_objective::String
end

struct V2PowerOracle
    artifact::PowerArtifact
end

function _exp_expected(artifact::ExpArtifact)
    setprecision(BigFloat, 256) do
        artifact.kind === :unit_epigraph && return BigFloat(1)
        artifact.kind === :entropy && return -log(BigFloat(artifact.n))
        artifact.kind === :logsumexp && begin
            values = BigFloat.(artifact.coefficients)
            return log(sum(exp(v) for v in values))
        end
        throw(ArgumentError("no EXP oracle for $(artifact.kind)"))
    end
end

function _interval(value::BigFloat)
    setprecision(BigFloat, 256) do
        δ = BigFloat("5e-7")
        (string(value - δ), string(value + δ))
    end
end

function _exp_expected_model(artifact::ExpArtifact, ::Type{T}; precision_bits=256) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits, name=String(artifact.id)) :
        SDPX.Model(T; name=String(artifact.id))
    if artifact.kind === :unit_epigraph
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
        SDPX.constraint!(model, :unit_exp, (zero(T), one(T), x[1]), SDPX.ExponentialCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
    elseif artifact.kind === :entropy
        p = SDPX.variable!(model, :p, artifact.n; domain=SDPX.Nonnegative())
        r = SDPX.variable!(model, :r, artifact.n; domain=SDPX.Reals())
        normalization = -one(T)
        for i in 1:artifact.n
            normalization += p[i]
            SDPX.constraint!(model, Symbol(:entropy_, i), (-r[i], p[i], one(T)), SDPX.ExponentialCone())
        end
        SDPX.constraint!(model, :normalization, normalization, SDPX.ZeroCone())
        objective = zero(T)
        for i in 1:artifact.n
            objective += r[i]
        end
        SDPX.objective!(model, SDPX.Minimize(), objective)
    elseif artifact.kind === :logsumexp
        t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())
        z = SDPX.variable!(model, :z, artifact.n; domain=SDPX.Nonnegative())
        for i in 1:artifact.n
            a = _Tq(T, artifact.coefficients[i])
            SDPX.constraint!(model, Symbol(:logsumexp_, i), (a - t[1], one(T), z[i]), SDPX.ExponentialCone())
        end
        sum_z = -one(T)
        for i in 1:artifact.n
            sum_z += z[i]
        end
        SDPX.constraint!(model, :normalization, sum_z, SDPX.Nonpositive())
        SDPX.objective!(model, SDPX.Minimize(), t[1])
    else
        throw(ArgumentError("EXP fitting lowering remains open"))
    end
    actual = _native_model_fingerprint(model)
    transform = V2Transform(:exp_small_artifact, :sdpx_cone_program,
        :exp_cone_lowering, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    expected_objective = _exp_expected(artifact)
    expected_model = _source_model_receipt(artifact, model)
    _actual_model_receipt(model) == expected_model || throw(ArgumentError(
        "EXP lowering differs from its independently reconstructed source contract"))
    V2Built(model, V2ExpOracle(artifact, _interval(expected_objective), string(expected_objective)),
        artifact, "", transform,
        (artifact_fingerprint=_hex(artifact), model_fingerprint=actual,
         model_contract_fingerprint=_hex(expected_model),
         model_source_receipt=expected_model, model_precision_bits=SDPX.precision_bits(model),
         source_dimension=artifact.n, target_dimension=artifact.n,
         generator=artifact.generator_id), (setup_seconds=nothing,))
end

function _exp_oracle_check(oracle::V2ExpOracle, built, certificate)
    artifact = oracle.artifact
    built.source_artifact === artifact || return false
    actual = _native_model_fingerprint(built.problem)
    expected_fp = _hex(_source_model_receipt(artifact, built.problem))
    actual == get(built.facts, :model_fingerprint, "") == expected_fp || return false
    get(built.facts, :model_contract_fingerprint, "") == expected_fp || return false
    _model_matches_source_receipt(artifact, built.problem) || return false
    # The witness is an independent analytic point.  EXP transcendental
    # objectives are intentionally checked against a high-precision interval.
    if artifact.kind === :unit_epigraph
        artifact.witness == Rational{Int}[1] || return false
    elseif artifact.kind === :entropy
        artifact.witness == fill(1//artifact.n, artifact.n) || return false
        sum(artifact.witness) == 1//1 || return false
    elseif artifact.kind === :logsumexp
        artifact.witness == fill(1//artifact.n, artifact.n) || return false
        sum(artifact.witness) == 1//1 || return false
    else
        return false
    end
    value = try BigFloat(certificate.primal_objective) catch; BigFloat(NaN) end
    lower, upper = oracle.objective_interval
    isfinite(value) && setprecision(BigFloat, 512) do
        BigFloat(lower) <= value <= BigFloat(upper)
    end
end

(oracle::V2ExpOracle)(built, certificate) = _exp_oracle_check(oracle, built, certificate)

function _power_build(artifact::PowerArtifact, ::Type{T}; precision_bits=256) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits, name=String(artifact.id)) :
        SDPX.Model(T; name=String(artifact.id))
    if artifact.kind in (:separable_p_power, :alpha_sweep)
        n = length(artifact.alphas)
        t = SDPX.variable!(model, :t, n; domain=SDPX.Nonnegative())
        x = SDPX.variable!(model, :x, n; domain=SDPX.Reals())
        for i in 1:n
            _fix!(model, Symbol(:fix_x_, i), x[i], _Tq(T, artifact.fixed_values[i]), T)
            SDPX.constraint!(model, Symbol(:power_, i), (t[i], one(T), x[i]),
                SDPX.PowerCone(_Tq(T, artifact.alphas[i])))
        end
        objective = zero(T)
        for i in 1:n
            objective += t[i]
        end
        SDPX.objective!(model, SDPX.Minimize(), objective)
    elseif artifact.kind === :weighted_mean
        left = SDPX.variable!(model, :left, 1; domain=SDPX.Nonnegative())
        right = SDPX.variable!(model, :right, 1; domain=SDPX.Nonnegative())
        z = SDPX.variable!(model, :z, 1; domain=SDPX.Reals())
        _fix!(model, :fix_left, left[1], _Tq(T, artifact.weighted_values[1]), T)
        _fix!(model, :fix_right, right[1], _Tq(T, artifact.weighted_values[2]), T)
        SDPX.constraint!(model, :weighted_mean,
            (left[1], right[1], z[1]), SDPX.PowerCone(_Tq(T, artifact.alphas[1])))
        SDPX.objective!(model, SDPX.Maximize(), z[1])
    end
    actual = _native_model_fingerprint(model)
    transform = V2Transform(:power_small_artifact, :sdpx_cone_program,
        :power_cone_lowering, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    oracle = V2PowerOracle(artifact)
    expected_model = _source_model_receipt(artifact, model)
    _actual_model_receipt(model) == expected_model || throw(ArgumentError(
        "Power lowering differs from its independently reconstructed source contract"))
    V2Built(model, oracle, artifact, "", transform,
        (artifact_fingerprint=_hex(artifact), model_fingerprint=actual,
         model_contract_fingerprint=_hex(expected_model),
         model_source_receipt=expected_model, model_precision_bits=SDPX.precision_bits(model),
         source_dimension=length(artifact.alphas), target_dimension=length(artifact.alphas),
         generator=artifact.generator_id), (setup_seconds=nothing,))
end

# The oracle checks source identity and the exact rational optimum.  The
# public certificate gate separately checks residuals in the requested backend.
_power_oracle(oracle::V2PowerOracle, built, certificate) = begin
    artifact = oracle.artifact
    built.source_artifact === artifact || return false
    actual = _native_model_fingerprint(built.problem)
    expected_fp = _hex(_source_model_receipt(artifact, built.problem))
    actual == get(built.facts, :model_fingerprint, "") == expected_fp || return false
    get(built.facts, :model_contract_fingerprint, "") == expected_fp || return false
    _model_matches_source_receipt(artifact, built.problem) || return false
    all(a -> 0//1 < a < 1//1, artifact.alphas) || return false
    artifact.kind === :weighted_mean ? artifact.weighted_values == Rational{Int}[1, 1] : all(==(1//1), artifact.fixed_values) || return false
    value = try BigFloat(certificate.primal_objective) catch; BigFloat(NaN) end
    isfinite(value) && abs(value - BigFloat(artifact.objective)) <= BigFloat("5e-7")
end

(oracle::V2PowerOracle)(built, certificate) = _power_oracle(oracle, built, certificate)

function _exp_instance(artifact, description, family, transform, oracle, objective_interval)
    V2Instance(artifact.id, family, only(filter(t -> t.name === :small, _TIERS)),
        (kind=artifact.kind,), :train, "general-v2/$(family)/$(artifact.kind)/small",
        (generator=artifact.generator_id, version=artifact.generator_version,
         transform=transform, solve_eligible=true), _hex(artifact),
        (wall_seconds=20, memory_bytes=4 * 1024^3),
        V2Reference(:optimal, :optimal, objective_interval, oracle, description), artifact)
end

function exp_tranche_catalog()
    # Only the unit epigraph is solve-eligible today.  Entropy and
    # log-sum-exp constructors remain typed candidates: the native route
    # currently reports numerical_breakdown on their boundary-sensitive
    # exponential blocks, so they are deliberately not registered.
    artifacts = [
        ExpArtifact(:v2_exp_unit_epigraph_small, :unit_epigraph, [0//1], [1//1]),
    ]
    transform = V2Transform(:exp_small_artifact, :sdpx_cone_program,
        :exp_cone_lowering, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    desc = [
        "unit epigraph: (0,1,x) in K_exp forces x>=exp(0)=1",
        "entropy epigraph: uniform p=(1/2,1/2) gives exact high-precision optimum -log(2)",
        "log-sum-exp epigraph: equal coefficients and z=(1/2,1/2) give exact high-precision optimum log(2)",
    ]
    instances = V2Instance[]
    for (artifact, text) in zip(artifacts, desc)
        expected = _exp_expected(artifact)
        interval = _interval(expected)
        push!(instances, _exp_instance(artifact, text, :exp, transform,
            V2ExpOracle(artifact, interval, string(expected)), interval))
    end
    family = V2Family(:exp, V2Axis[],
        (instance, precision) -> begin
            built = _exp_expected_model(instance.payload, precision.arithmetic; precision_bits=precision.bits)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform,
                merge(built.facts, (artifact_fingerprint=_hex(instance.payload),)), built.resource)
        end, (built, cert) -> built.oracle(built, cert),
        (instance, result) -> result.validation, (:identity,))
    V2Catalog(:general_v2_exp_tranche, 1, [family], instances,
        (train=[i.id for i in instances], holdout=Symbol[], sentinel=Symbol[]))
end

function power_tranche_catalog()
    alphas = Rational{Int}[1//2, 1//3, 2//3, 2//5, 7//10]
    separable = PowerArtifact(:v2_power_separable_small, :separable_p_power,
        alphas, fill(1//1, 5), Rational{Int}[], 5//1)
    weighted = PowerArtifact(:v2_power_weighted_mean_small, :weighted_mean,
        Rational{Int}[1//2], Rational{Int}[], Rational{Int}[1, 1], 1//1)
    sweep = PowerArtifact(:v2_power_alpha_sweep_small, :alpha_sweep,
        alphas, fill(1//1, 5), Rational{Int}[], 5//1)
    # The exact-alpha candidates are retained as typed source contracts, but
    # none is registered until the native PowerCone route supplies an
    # optimal, certificate-valid Float64 receipt.  Current probes return
    # numerical_breakdown/numerical_failure at the cone boundary.
    artifacts = PowerArtifact[]
    transform = V2Transform(:power_small_artifact, :sdpx_cone_program,
        :power_cone_lowering, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    desc = [
        "separable p-power: exact alpha list and x_i=1 force t_i=1, objective 5",
        "weighted mean: left=right=1 forces z<=1 for every exact alpha, objective 1",
        "alpha sweep: one exact-rational x=1 epigraph per reviewed alpha, objective 5",
    ]
    instances = V2Instance[]
    for (artifact, text) in zip(artifacts, desc)
        push!(instances, V2Instance(artifact.id, :power,
            only(filter(t -> t.name === :small, _TIERS)),
            (kind=artifact.kind, alphas=artifact.alphas), :train,
            "general-v2/power/$(artifact.kind)/small",
            (generator=artifact.generator_id, version=artifact.generator_version,
             transform=transform, solve_eligible=true), _hex(artifact),
            (wall_seconds=20, memory_bytes=4 * 1024^3),
            V2Reference(:optimal, :optimal,
                (string(BigFloat(artifact.objective)-BigFloat("5e-7")),
                 string(BigFloat(artifact.objective)+BigFloat("5e-7"))),
                V2PowerOracle(artifact), text), artifact))
    end
    family = V2Family(:power, V2Axis[],
        (instance, precision) -> begin
            built = _power_build(instance.payload, precision.arithmetic; precision_bits=precision.bits)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform,
                merge(built.facts, (artifact_fingerprint=_hex(instance.payload),)), built.resource)
        end, (built, cert) -> built.oracle(built, cert),
        (instance, result) -> result.validation, (:identity,))
    V2Catalog(:general_v2_power_tranche, 1, [family], instances,
        (train=[i.id for i in instances], holdout=Symbol[], sentinel=Symbol[]))
end

reviewed_power_alphas() = Rational{Int}[1//2, 1//3, 2//3, 2//5, 7//10]

export ExpArtifact, PowerArtifact, V2ExpOracle, V2PowerOracle,
    exp_tranche_catalog, power_tranche_catalog, reviewed_power_alphas
