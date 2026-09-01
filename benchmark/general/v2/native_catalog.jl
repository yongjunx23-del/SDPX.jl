# Native V2 corpus. Source artifacts are complete exact conic data: every
# coefficient is consumed by the builder and participates in identity.
using LinearAlgebra
abstract type AbstractV2SourceArtifact end

# Typed source artifacts for the real small-tier tranche.  These are source
# contracts, not solver state: exact rational coefficients and independent
# witnesses are retained until a family-specific lowering is implemented.
abstract type AbstractV2SmallArtifact <: AbstractV2SourceArtifact end

struct LPArtifact <: AbstractV2SmallArtifact
    id::Symbol
    kind::Symbol
    A::Matrix{Rational{Int}}
    b::Vector{Rational{Int}}
    c::Vector{Rational{Int}}
    cone_partition::Vector{Symbol}
    primal_witness::Vector{Rational{Int}}
    dual_witness::Vector{Rational{Int}}
    objective::Rational{Int}
    expected_status::Symbol
    certificate_kind::Symbol
    generator_id::Symbol
    generator_version::Int
    function LPArtifact(id::Symbol, kind::Symbol, A, b, c;
                        cone_partition=fill(:nonnegative, length(c)),
                        primal_witness=Rational{Int}[],
                        dual_witness=Rational{Int}[], objective::Rational{Int}=0//1,
                        expected_status::Symbol=:optimal,
                        certificate_kind::Symbol=:optimal,
                        generator_id::Symbol=:lp_small_v1,
                        generator_version::Integer=1)
        kind in (:box, :sparse_planted_kkt, :duplicate_rank_deficient,
                 :primal_infeasible, :unbounded, :chebyshev, :nonpositive) ||
            throw(ArgumentError("unsupported LP small-tranche kind $kind"))
        AA = Rational{Int}.(A); bb = Rational{Int}.(b); cc = Rational{Int}.(c)
        size(AA, 1) == length(bb) || throw(ArgumentError("LP A/b dimensions disagree"))
        size(AA, 2) == length(cc) || throw(ArgumentError("LP A/c dimensions disagree"))
        !isempty(cc) || throw(ArgumentError("LP objective must not be empty"))
        length(cone_partition) == length(cc) ||
            throw(ArgumentError("LP cone partition must match variable dimension"))
        all(x -> x in (:nonnegative, :nonpositive, :free), cone_partition) ||
            throw(ArgumentError("unsupported LP cone partition"))
        expected_status in (:optimal, :primal_infeasible, :dual_infeasible) ||
            throw(ArgumentError("unsupported LP expected status"))
        certificate_kind in (:optimal, :farkas, :ray) ||
            throw(ArgumentError("unsupported LP certificate kind"))
        expected_status === :optimal && certificate_kind !== :optimal &&
            throw(ArgumentError("optimal LP requires certificate_kind=:optimal"))
        expected_status === :primal_infeasible && certificate_kind !== :farkas &&
            throw(ArgumentError("primal-infeasible LP requires certificate_kind=:farkas"))
        expected_status === :dual_infeasible && certificate_kind !== :ray &&
            throw(ArgumentError("dual-infeasible LP requires certificate_kind=:ray"))
        generator_version > 0 || throw(ArgumentError("generator version must be positive"))
        new(id, kind, AA, bb, cc, Symbol.(cone_partition),
            Rational{Int}.(primal_witness), Rational{Int}.(dual_witness), objective,
            expected_status, certificate_kind, generator_id, Int(generator_version))
    end
end

struct SOCPArtifact <: AbstractV2SmallArtifact
    id::Symbol
    kind::Symbol
    A::Matrix{Rational{Int}}
    b::Vector{Rational{Int}}
    c::Vector{Rational{Int}}
    cone_partition::Vector{Int}
    primal_witness::Vector{Rational{Int}}
    dual_witness::Vector{Rational{Int}}
    objective::Rational{Int}
    generator_id::Symbol
    generator_version::Int
    function SOCPArtifact(id::Symbol, kind::Symbol, A, b, c;
                          cone_partition=Int[],
                          primal_witness=Rational{Int}[],
                          dual_witness=Rational{Int}[], objective::Rational{Int}=0//1,
                          generator_id::Symbol=:soc_small_v1,
                          generator_version::Integer=1)
        kind in (:simplex_projection, :planted_portfolio, :large_soc,
                 :q3_load_sharing, :ill_scaled_soc) ||
            throw(ArgumentError("unsupported SOCP small-tranche kind $kind"))
        AA = Rational{Int}.(A); bb = Rational{Int}.(b); cc = Rational{Int}.(c)
        size(AA, 1) == length(bb) || throw(ArgumentError("SOCP A/b dimensions disagree"))
        size(AA, 2) == length(cc) || throw(ArgumentError("SOCP A/c dimensions disagree"))
        all(>(0), cone_partition) || isempty(cone_partition) ||
            throw(ArgumentError("SOCP cone block sizes must be positive"))
        sum(cone_partition) <= length(bb) || throw(ArgumentError("SOCP cone blocks exceed row dimension"))
        generator_version > 0 || throw(ArgumentError("generator version must be positive"))
        new(id, kind, AA, bb, cc, Int.(cone_partition),
            Rational{Int}.(primal_witness), Rational{Int}.(dual_witness), objective,
            generator_id, Int(generator_version))
    end
end

struct IllConditionedArtifact <: AbstractV2SmallArtifact
    id::Symbol
    kind::Symbol
    base_family::Symbol
    coefficients::Matrix{Rational{Int}}
    rhs::Vector{Rational{Int}}
    objective_coefficients::Vector{Rational{Int}}
    scale_exponent::Int
    primal_witness::Vector{Rational{Int}}
    dual_witness::Vector{Rational{Int}}
    objective::Rational{Int}
    generator_id::Symbol
    generator_version::Int
    function IllConditionedArtifact(id::Symbol, kind::Symbol, base_family::Symbol,
                                    coefficients, rhs, objective_coefficients;
                                    scale_exponent::Integer=0,
                                    primal_witness=Rational{Int}[],
                                    dual_witness=Rational{Int}[], objective::Rational{Int}=0//1,
                                    generator_id::Symbol=:ill_conditioned_small_v1,
                                    generator_version::Integer=1)
        kind in (:hilbert6_sdp, :diagonal_scale_ladder, :near_rank_loss_lp,
                 :near_boundary_lp, :near_boundary_soc_psd, :high_range_exp_power) ||
            throw(ArgumentError("unsupported ill-conditioned small-tranche kind $kind"))
        base_family in (:lp, :soc, :sdp, :exp, :power) ||
            throw(ArgumentError("unsupported ill-conditioned base family"))
        kind === :hilbert6_sdp && base_family !== :sdp &&
            throw(ArgumentError("Hilbert-6 artifact must use the SDP base family"))
        kind in (:near_rank_loss_lp, :near_boundary_lp) && base_family !== :lp &&
            throw(ArgumentError("LP ill-conditioned artifact must use the LP base family"))
        kind === :diagonal_scale_ladder && base_family !== :lp &&
            throw(ArgumentError("diagonal scale-ladder artifact must use the LP base family"))
        AA = Rational{Int}.(coefficients); bb = Rational{Int}.(rhs); cc = Rational{Int}.(objective_coefficients)
        size(AA, 1) == length(bb) || throw(ArgumentError("ill-conditioned A/b dimensions disagree"))
        size(AA, 2) == length(cc) || throw(ArgumentError("ill-conditioned A/c dimensions disagree"))
        scale_exponent >= 0 || throw(ArgumentError("scale exponent must be nonnegative"))
        generator_version > 0 || throw(ArgumentError("generator version must be positive"))
        new(id, kind, base_family, AA, bb, cc, Int(scale_exponent),
            Rational{Int}.(primal_witness), Rational{Int}.(dual_witness), objective,
            generator_id, Int(generator_version))
    end
end

# Canonical source identity for typed artifacts.  Mathematical fields are
# explicit; generator metadata is included in the artifact fingerprint but
# never substitutes for coefficients or an oracle.
function _put!(io::IO, artifact::LPArtifact)
    _put!(io, (:LPArtifact, artifact.id, artifact.kind, artifact.A, artifact.b,
        artifact.c, artifact.cone_partition, artifact.primal_witness,
        artifact.dual_witness, artifact.objective, artifact.expected_status,
        artifact.certificate_kind, artifact.generator_id, artifact.generator_version))
end
function _put!(io::IO, artifact::SOCPArtifact)
    _put!(io, (:SOCPArtifact, artifact.id, artifact.kind, artifact.A, artifact.b,
        artifact.c, artifact.cone_partition, artifact.primal_witness,
        artifact.dual_witness, artifact.objective, artifact.generator_id,
        artifact.generator_version))
end
function _put!(io::IO, artifact::IllConditionedArtifact)
    _put!(io, (:IllConditionedArtifact, artifact.id, artifact.kind,
        artifact.base_family, artifact.coefficients, artifact.rhs,
        artifact.objective_coefficients, artifact.scale_exponent,
        artifact.primal_witness, artifact.dual_witness, artifact.objective,
        artifact.generator_id, artifact.generator_version))
end

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
        expected = family in (:lp, :nonpositive) ? dimension :
            family in (:soc, :mixed) ? 2 : family === :sdp ? 4 :
            family in (:rsoc, :exp, :power) ? 1 : 0
        expected > 0 || throw(ArgumentError("unknown artifact family $family"))
        length(coefficients) == expected || throw(ArgumentError(
            "artifact $family requires exactly $expected coefficients; extra/ignored coefficients are forbidden"))
        family in (:exp, :power) && dimension != 1 &&
            throw(ArgumentError("$family artifacts have scalar dimension 1"))
        family === :rsoc && dimension != 3 &&
            throw(ArgumentError("RSOC artifacts have cone dimension 3"))
        family in (:soc, :mixed) && dimension != 2 &&
            throw(ArgumentError("$family artifacts have dimension 2"))
        family === :sdp && dimension != 2 &&
            throw(ArgumentError("SDP artifacts have matrix dimension 2"))
        family === :sdp && coefficients[2] != coefficients[3] &&
            throw(ArgumentError("SDP off-diagonal coefficients must be symmetric"))
        length(infeasibility_ray) == 2 || throw(ArgumentError("infeasibility ray must have two entries"))
        new(family, id, coefficients, Int(dimension), cone_parameter, infeasible,
            Rational{Int}.(infeasibility_ray), generator_id, Int(generator_version))
    end
end

function _math_payload(artifact::V2ConicArtifact)
    return (family=artifact.family, coefficients=artifact.coefficients,
        dimension=artifact.dimension, cone_parameter=artifact.cone_parameter,
        infeasible=artifact.infeasible, infeasibility_ray=artifact.infeasibility_ray)
end

function _expected_transform(instance::V2Instance)
    instance.payload isa V2ConicArtifact || return _declared_transform(instance)
    return V2Transform(:native_conic_artifact, :sdpx_cone_program,
        :identity, 1, :identity;
        validation_receipts=(coefficient_match=true,
                             source_reconstruction=true))
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
        value = family === :nonpositive ? (suffix == 1 ? -2 : -1) :
            (suffix == 0 ? 1 : suffix == 1 ? 2 : 1)
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

function _native_build(artifact::V2ConicArtifact, ::Type{T}; precision_bits::Int=256) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits, name=String(artifact.id)) :
            SDPX.Model(T; name=String(artifact.id))
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
            push!(witness, _q(artifact, i) * artifact.cone_parameter)
        end
        artifact.infeasible && _contradict!(model, :contradiction, x[1], _Tq(T, _q(artifact, 1)), T)
        objective = sum(x[i] for i in 1:n)
    elseif f === :nonpositive
        x = SDPX.variable!(model, :x, artifact.dimension; domain=SDPX.Nonpositive())
        for i in 1:artifact.dimension
            value = _Tq(T, _q(artifact, i)) * cone_parameter; value <= zero(T) || throw(ArgumentError("Nonpositive artifact is not feasible"))
            _fix!(model, Symbol(:fix_, i), x[i], value, T); push!(witness, _q(artifact, i) * artifact.cone_parameter)
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
        append!(witness, [1//1, 1//1])
        objective = u[1] + v[1]
    elseif f === :sdp
        artifact.dimension == 2 || throw(ArgumentError("native SDP artifact dimension must be 2"))
        X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
        vals = [_Tq(T, _q(artifact, i)) for i in 1:4]
        vals[2] *= cone_parameter; vals[3] *= cone_parameter
        _fix!(model, :diag1, X[1, 1], vals[1], T); _fix!(model, :offdiag, X[1, 2], vals[2], T)
        _fix!(model, :offdiag_lower, X[2, 1], vals[3], T); _fix!(model, :diag2, X[2, 2], vals[4], T)
        artifact.infeasible && _contradict!(model, :contradiction, X[1, 1], vals[1], T)
        append!(witness, [_q(artifact, 1), artifact.cone_parameter * _q(artifact, 2), _q(artifact, 4)]); objective = X[1, 1] + X[2, 2]
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
    actual_model_fp = _native_model_fingerprint(model)
    expected_model_fp = _hex(_expected_model_receipt(artifact, T, precision_bits))
    actual_model_fp == expected_model_fp || begin
        println(stderr, "ACTUAL_RECEIPT=", repr(_actual_model_receipt(model)))
        println(stderr, "EXPECTED_RECEIPT=", repr(_expected_model_receipt(artifact, T, precision_bits)))
        throw(ArgumentError("native lowering differs from its independent model contract: actual=$actual_model_fp expected=$expected_model_fp"))
    end
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
    dual_multipliers = artifact.infeasible ? Rational{Int}[] :
        _oracle_dual_multipliers(artifact)
    cone_dual_slacks = artifact.infeasible ? Rational{Int}[] :
        _oracle_cone_dual_slacks(model)
    dual_bound = artifact.infeasible ? 0//1 : _oracle_objective(artifact)
    oracle = V2ExactOracle(artifact.infeasible ? :primal_infeasible : :optimal,
        expected, witness, dual_multipliers, cone_dual_slacks, dual_bound,
        artifact.infeasibility_ray, artifact)
    return V2Built(model, oracle, artifact, _hex(artifact), transform,
        (source_dimension=artifact.dimension, target_dimension=artifact.dimension,
         generator=artifact.generator_id, coefficients=artifact.coefficients,
         model_contract_fingerprint=_hex(_expected_model_receipt(artifact,
             T, precision_bits)),
         dimension=artifact.dimension, cone_parameter=artifact.cone_parameter,
         infeasible=artifact.infeasible),
        (setup_seconds=nothing,))
end

function _numeric_token(x)
    # Model identity is mathematical; normalize signed zero so expression
    # construction order cannot create a spurious contract mismatch.
    iszero(x) && return "0"
    return string(BigFloat(x))
end

function _actual_model_receipt(model)
    # Read the nominal model precision from the actual arithmetic spec; a
    # caller-supplied stamp must never be able to forge this identity.
    actual_precision_bits = SDPX.precision_bits(model)
    variables = Tuple((v.name, string(v.domain), v.shape, v.offset, v.length) for v in model.variable_blocks)
    constraints = Tuple((c.name, string(c.domain), c.shape,
        Tuple((Tuple(e.indices), Tuple(_numeric_token.(e.coefficients)), _numeric_token(e.constant))
              for e in c.expressions)) for c in model.constraint_blocks)
    objective = model.objective === nothing ? nothing :
        (string(typeof(model.objective.sense)), Tuple(model.objective.expression.indices),
         Tuple(_numeric_token.(model.objective.expression.coefficients)),
         _numeric_token(model.objective.expression.constant))
    return (precision_bits=actual_precision_bits, variables=variables,
            constraints=constraints, objective=objective)
end

function _expected_model_receipt(artifact::V2ConicArtifact, ::Type{T}, precision_bits) where {T}
    cp = _Tq(T, artifact.cone_parameter)
    f = artifact.family
    vars = if f in (:lp, :nonpositive, :soc, :exp, :power)
        domain = f === :lp ? "SDPX.Nonnegative()" :
                 f === :nonpositive ? "SDPX.Nonpositive()" : "SDPX.Reals()"
        ((:x, domain, artifact.dimension, 1, artifact.dimension),)
    elseif f === :rsoc
        ((:left, "SDPX.Reals()", 1, 1, 1), (:right, "SDPX.Reals()", 1, 2, 1))
    elseif f === :sdp
        ((:X, "SDPX.PSDCone()", 2, 1, 3),)
    elseif f === :mixed
        ((:positive, "SDPX.Nonnegative()", 1, 1, 1), (:negative, "SDPX.Nonpositive()", 1, 2, 1))
    else
        throw(ArgumentError("unknown artifact family $f"))
    end
    expr(indices, coefficients, constant) =
        (Tuple(indices), Tuple(_numeric_token.(coefficients)), _numeric_token(constant))
    cons = Any[]
    if f in (:lp, :nonpositive)
        for i in 1:artifact.dimension
            v = _Tq(T, _q(artifact, i)) * cp
            push!(cons, (Symbol(:fix_, i), "SDPX.ZeroCone()", 1, (expr((i,), (one(T),), -v),)))
        end
    elseif f === :soc
        vals = [_Tq(T, _q(artifact, i)) for i in 1:artifact.dimension]
        push!(cons, (:soc, "SDPX.LorentzCone()", 3,
            (expr((), (), one(T)+cp), expr((1,), (one(T),), zero(T)), expr((2,), (one(T),), zero(T)))))
        for i in 1:artifact.dimension
            push!(cons, (Symbol(:fix_, i), "SDPX.ZeroCone()", 1, (expr((i,), (one(T),), -vals[i]),)))
        end
    elseif f === :rsoc
        target = _Tq(T, _q(artifact, 1))*cp
        push!(cons, (:rsoc, "SDPX.RotatedLorentzCone()", 3,
            (expr((1,), (one(T),), zero(T)), expr((2,), (one(T),), zero(T)), expr((), (), target))))
        push!(cons, (:fix_left, "SDPX.ZeroCone()", 1, (expr((1,), (one(T),), -one(T)),)))
        push!(cons, (:fix_right, "SDPX.ZeroCone()", 1, (expr((2,), (one(T),), -one(T)),)))
    elseif f === :sdp
        vals = [_Tq(T, _q(artifact, i)) for i in 1:4]; vals[2]*=cp; vals[3]*=cp
        for (name, idx, val) in ((:diag1,1,vals[1]), (:offdiag,2,vals[2]), (:offdiag_lower,2,vals[3]), (:diag2,3,vals[4]))
            push!(cons, (name, "SDPX.ZeroCone()", 1, (expr((idx,), (one(T),), -val),)))
        end
    elseif f === :exp
        value = _Tq(T, _q(artifact,1))
        push!(cons, (:exp, "SDPX.ExponentialCone()", 3,
            (expr((),(),zero(T)), expr((),(),cp), expr((1,), (one(T),),zero(T)))))
        push!(cons, (:fix, "SDPX.ZeroCone()", 1, (expr((1,), (one(T),), -value),)))
    elseif f === :power
        value = _Tq(T, _q(artifact,1)); alpha=cp
        push!(cons, (:power, "SDPX.PowerCone{$T}($(alpha))", 3,
            (expr((1,), (one(T),),zero(T)), expr((),(),one(T)), expr((),(),one(T)))))
        push!(cons, (:fix, "SDPX.ZeroCone()", 1, (expr((1,), (one(T),), -value),)))
    elseif f === :mixed
        vals = (_Tq(T,_q(artifact,1)), _Tq(T,_q(artifact,2)))
        push!(cons, (:soc, "SDPX.LorentzCone()", 3,
            (expr((),(),one(T)+cp), expr((1,), (one(T),),zero(T)), expr((2,), (one(T),),zero(T)))))
        push!(cons, (:fix_positive, "SDPX.ZeroCone()", 1, (expr((1,), (one(T),),-vals[1]),)))
        push!(cons, (:fix_negative, "SDPX.ZeroCone()", 1, (expr((2,), (one(T),),-vals[2]),)))
    end
    if artifact.infeasible
        value = f === :sdp ? _Tq(T,_q(artifact,1)) :
            f === :rsoc ? one(T) : f === :mixed ? _Tq(T,_q(artifact,1)) :
            _Tq(T,_q(artifact,1))
        idx = f === :rsoc ? 1 : f === :sdp ? 1 : f === :mixed ? 1 : 1
        push!(cons, (:contradiction_a, "SDPX.ZeroCone()", 1, (expr((idx,), (one(T),), -value),)))
        push!(cons, (:contradiction_b, "SDPX.ZeroCone()", 1, (expr((idx,), (one(T),), -(value+one(T))),)))
    end
    obj = if f === :sdp
        ("SDPX.Minimize", (1,3), (_numeric_token(one(T)),_numeric_token(one(T))), _numeric_token(zero(T)))
    elseif f === :rsoc
        ("SDPX.Minimize", (1,2), (_numeric_token(one(T)),_numeric_token(one(T))), _numeric_token(zero(T)))
    elseif f === :mixed
        ("SDPX.Minimize", (1,2), (_numeric_token(one(T)),_numeric_token(-one(T))), _numeric_token(zero(T)))
    elseif f === :nonpositive
        ("SDPX.Minimize", Tuple(1:artifact.dimension), Tuple(_numeric_token.(-ones(T,artifact.dimension))), _numeric_token(-zero(T)))
    else
        ("SDPX.Minimize", Tuple(1:artifact.dimension), Tuple(_numeric_token.(ones(T,artifact.dimension))), _numeric_token(zero(T)))
    end
    return (precision_bits=precision_bits, variables=vars, constraints=Tuple(cons), objective=obj)
end

function _native_model_fingerprint(model)
    return _hex(_actual_model_receipt(model))
end

function _actual_witness(artifact::V2ConicArtifact)
    f=artifact.family; cp=artifact.cone_parameter
    f === :sdp && return Rational{Int}[_q(artifact,1), cp*_q(artifact,2), _q(artifact,4)]
    f === :rsoc && return Rational{Int}[1,1]
    f === :mixed && return copy(artifact.coefficients)
    f in (:lp,:nonpositive) && return Rational{Int}[cp*q for q in artifact.coefficients]
    return copy(artifact.coefficients[1:artifact.dimension])
end

function _eval_actual_expr(expr, witness)
    value=BigFloat(expr.constant)
    for (idx, coeff) in zip(expr.indices, expr.coefficients)
        value += BigFloat(coeff)*BigFloat(numerator(witness[idx]))/BigFloat(denominator(witness[idx]))
    end
    value
end

function _actual_witness_check(artifact::V2ConicArtifact, built)
    witness = _actual_witness(artifact)
    total = sum(v.length for v in built.problem.variable_blocks)
    length(witness)==total || return false
    for v in built.problem.variable_blocks
        w=witness[v.offset:v.offset+v.length-1]
        ds=string(v.domain)
        ds=="SDPX.Nonnegative()" && !all(>=(0),w) && return false
        ds=="SDPX.Nonpositive()" && !all(<=(0),w) && return false
        if ds=="SDPX.PSDCone()"
            length(w)==3 || return false
            BigFloat(w[1])*BigFloat(w[3])-BigFloat(w[2])^2 >= 0 || return false
        end
    end
    for c in built.problem.constraint_blocks
        vals=[_eval_actual_expr(e,witness) for e in c.expressions]
        ds=string(c.domain)
        if ds=="SDPX.ZeroCone()"
            all(iszero, vals) || return false
        elseif ds=="SDPX.LorentzCone()"
            vals[1] >= sqrt(sum(x^2 for x in vals[2:end])) || return false
        elseif ds=="SDPX.RotatedLorentzCone()"
            vals[1]>=0 && vals[2]>=0 && 2*vals[1]*vals[2]>=sum(x^2 for x in vals[3:end]) || return false
        elseif ds=="SDPX.ExponentialCone()"
            vals[2]>0 && vals[3] >= vals[2]*exp(vals[1]/vals[2]) || return false
        elseif startswith(ds,"SDPX.PowerCone")
            α=BigFloat(numerator(artifact.cone_parameter))/BigFloat(denominator(artifact.cone_parameter))
            vals[1]>=0 && vals[2]>=0 && abs(vals[3])<=vals[1]^α*vals[2]^(1-α) || return false
        end
    end
    expr=built.problem.objective.expression
    obj=BigFloat(expr.constant)
    for (idx,coeff) in zip(expr.indices,expr.coefficients)
        obj += BigFloat(coeff)*BigFloat(numerator(witness[idx]))/BigFloat(denominator(witness[idx]))
    end
    expected=BigFloat(numerator(_oracle_objective(artifact)))/BigFloat(denominator(_oracle_objective(artifact)))
    obj == expected
end

function _contradiction_rows(artifact::V2ConicArtifact)
    value = BigFloat(numerator(_q(artifact, 1))) / BigFloat(denominator(_q(artifact, 1)))
    # The builder adds x=value and x=value+1.  The exact ray [1,-1]
    # annihilates the variable coefficients and leaves a nonzero RHS.
    coefficients = (one(BigFloat), one(BigFloat))
    rhs = (value, value + one(BigFloat))
    ray = BigFloat.(artifact.infeasibility_ray)
    return dot(ray, coefficients), dot(ray, rhs)
end

function _actual_contradiction_rows(model)
    rows = Tuple{Vector{Int},Vector{BigFloat},BigFloat}[]
    for block in model.constraint_blocks
        startswith(String(block.name), "contradiction") || continue
        block.domain == SDPX.ZeroCone() || return Tuple{Vector{Int},Vector{BigFloat},BigFloat}[]
        block.shape == 1 && length(block.expressions) == 1 ||
            return Tuple{Vector{Int},Vector{BigFloat},BigFloat}[]
        expression = only(block.expressions)
        coefficients = BigFloat.(expression.coefficients)
        rhs = -BigFloat(expression.constant)
        push!(rows, (Int.(expression.indices), coefficients, rhs))
    end
    rows
end

function _farkas_valid(artifact::V2ConicArtifact, built=nothing)
    artifact.infeasible || return false
    isempty(artifact.infeasibility_ray) && return false
    rows = built === nothing ? Tuple{Vector{Int},Vector{BigFloat},BigFloat}[] :
        _actual_contradiction_rows(built.problem)
    isempty(rows) && return false
    ray = BigFloat.(artifact.infeasibility_ray)
    length(ray) == length(rows) || return false
    coefficients = Dict{Int,BigFloat}()
    for i in eachindex(rows), (index, coefficient) in zip(rows[i][1], rows[i][2])
        coefficients[index] = get(coefficients, index, zero(BigFloat)) + ray[i] * coefficient
    end
    rhs = sum(ray[i] * rows[i][3] for i in eachindex(rows))
    all(iszero, values(coefficients)) && !iszero(rhs)
end

struct V2ExactOracle
    expected_status::Symbol
    objective::Rational{Int}
    # primal_witness is in actual packed model coordinates.
    primal_witness::Vector{Rational{Int}}
    # Equality multipliers are row-indexed; cone slacks are conic-row indexed.
    dual_multipliers::Vector{Rational{Int}}
    cone_dual_slacks::Vector{Rational{Int}}
    dual_bound::Rational{Int}
    dual_ray::Vector{Rational{Int}}
    artifact::V2ConicArtifact
end

function _put!(io::IO, oracle::V2ExactOracle)
    _put!(io, (oracle.expected_status, oracle.objective,
        oracle.primal_witness, oracle.dual_multipliers, oracle.cone_dual_slacks,
        oracle.dual_bound, oracle.dual_ray, oracle.artifact))
end

function _actual_lower_bound_certificate(built, witness)
    model = built.problem
    objective = model.objective.expression
    stationarity = Dict{Int,BigFloat}()
    for k in eachindex(objective.indices)
        index = objective.indices[k]
        stationarity[index] = get(stationarity, index, zero(BigFloat)) + BigFloat(objective.coefficients[k])
    end
    equality_multipliers = BigFloat[]
    for block in model.constraint_blocks
        string(block.domain) == "SDPX.ZeroCone()" || continue
        for expr in block.expressions
            length(expr.indices) == 1 || continue
            index = only(expr.indices)
            coefficient = BigFloat(only(expr.coefficients))
            iszero(coefficient) && continue
            lambda = -get(stationarity, index, zero(BigFloat)) / coefficient
            push!(equality_multipliers, lambda)
            stationarity[index] = zero(BigFloat)
        end
    end
    total_variables = 0
    for variable in model.variable_blocks
        total_variables += variable.length
    end
    variable_dual_slacks = BigFloat[]
    for index in 1:total_variables
        push!(variable_dual_slacks, get(stationarity, index, zero(BigFloat)))
    end
    row_dual_slacks = BigFloat[]
    for block in model.constraint_blocks
        string(block.domain) == "SDPX.ZeroCone()" && continue
        for _ in 1:block.shape
            push!(row_dual_slacks, zero(BigFloat))
        end
    end
    bound = zero(BigFloat)
    for block in model.constraint_blocks
        string(block.domain) == "SDPX.ZeroCone()" || continue
        for expr in block.expressions
            length(expr.indices) == 1 || continue
            index = only(expr.indices)
            coefficient = BigFloat(only(expr.coefficients))
            c = zero(BigFloat)
            for k in eachindex(objective.indices)
                objective.indices[k] == index && (c += BigFloat(objective.coefficients[k]))
            end
            bound += (-c / coefficient) * BigFloat(expr.constant)
        end
    end
    return (stationarity=stationarity, equality_multipliers=equality_multipliers,
            variable_dual_slacks=variable_dual_slacks,
            row_dual_slacks=row_dual_slacks, bound=bound)
end

function _exact_rational(value)
    rationalize(Int, BigFloat(value); tol=0)
end

function _actual_equality_multipliers(model)
    objective = model.objective.expression
    objective_coefficients = Dict{Int,BigFloat}()
    for (index, coefficient) in zip(objective.indices, objective.coefficients)
        objective_coefficients[index] = get(objective_coefficients, index, zero(BigFloat)) + BigFloat(coefficient)
    end
    multipliers = Rational{Int}[]
    for block in model.constraint_blocks
        string(block.domain) == "SDPX.ZeroCone()" || continue
        for expression in block.expressions
            length(expression.indices) == 1 || return nothing
            index = only(expression.indices)
            coefficient = BigFloat(only(expression.coefficients))
            iszero(coefficient) && push!(multipliers, 0//1)
            iszero(coefficient) || push!(multipliers,
                _exact_rational(-get(objective_coefficients, index, zero(BigFloat)) / coefficient))
        end
    end
    multipliers
end

function _actual_cone_dual_slacks(model)
    out = Rational{Int}[]
    # Variable-cone slacks are represented in the packed model coordinates.
    for block in model.variable_blocks
        string(block.domain) == "SDPX.Reals()" && continue
        append!(out, fill(0//1, block.length))
    end
    # Explicit cone-constraint slacks are row-indexed by their block shape.
    for block in model.constraint_blocks
        string(block.domain) == "SDPX.ZeroCone()" && continue
        append!(out, fill(0//1, block.shape))
    end
    out
end

function _oracle_cone_dual_slacks(model)
    _actual_cone_dual_slacks(model)
end

function _oracle_cone_dual_slacks(artifact::V2ConicArtifact)
    f = artifact.family
    f in (:lp, :nonpositive) && return fill(0//1, artifact.dimension)
    f === :mixed && return fill(0//1, 5)
    f in (:soc, :rsoc, :exp, :power) && return fill(0//1, 3)
    f === :sdp && return fill(0//1, 3)
    throw(ArgumentError("unknown artifact family $f"))
end

function _oracle_check(oracle::V2ExactOracle, built, certificate)
    built.source_artifact === oracle.artifact || return false
    hasproperty(built.facts, :model_fingerprint) || return false
    oracle.primal_witness == _actual_witness(oracle.artifact) || return false
    # Model identity is mandatory for both feasible and infeasible oracles;
    # a Farkas row must never bypass a stale/wrong builder receipt.
    actual_fp = _native_model_fingerprint(built.problem)
    actual_fp == built.facts.model_fingerprint || return false
    expected_fp = get(built.facts, :model_contract_fingerprint, "")
    occursin(r"^[0-9a-f]{64}$", String(expected_fp)) || return false
    actual_fp == expected_fp || return false
    if oracle.expected_status === :primal_infeasible
        _farkas_valid(oracle.artifact, built) || return false
        return oracle.dual_ray == oracle.artifact.infeasibility_ray
    end
    oracle.expected_status === :optimal || return false
    actual_multipliers = _actual_equality_multipliers(built.problem)
    actual_multipliers === nothing && return false
    actual_multipliers == oracle.dual_multipliers || return false
    actual_cone_slacks = _actual_cone_dual_slacks(built.problem)
    actual_cone_slacks == oracle.cone_dual_slacks || return false
    oracle.expected_status === :optimal || return false
    _actual_witness_check(oracle.artifact, built) || return false
    witness = _actual_witness(oracle.artifact)
    lower_bound = _actual_lower_bound_certificate(built, witness)
    all(iszero, values(lower_bound.stationarity)) || return false
    all(iszero, lower_bound.variable_dual_slacks) &&
        all(iszero, lower_bound.row_dual_slacks) || return false
    expected_bound = BigFloat(numerator(oracle.objective)) /
                     BigFloat(denominator(oracle.objective))
    lower_bound.bound == expected_bound || return false
    dual_bound = oracle.dual_bound
    dual_bound == oracle.objective || return false
    expected = BigFloat(numerator(oracle.objective)) /
               BigFloat(denominator(oracle.objective))
    actual = BigFloat(certificate.primal_objective)
    return isfinite(actual) && abs(actual - expected) <= BigFloat("1e-7")
end

(oracle::V2ExactOracle)(built, certificate) = _oracle_check(oracle, built, certificate)

function _oracle_witness(artifact::V2ConicArtifact)
    f = artifact.family
    scale = artifact.cone_parameter
    # Witness is in the actual global packed model-variable coordinates, not
    # the source artifact's decorative coefficient order. PSD variables use
    # lower-packed svec coordinates; RSOC target is a constant cone entry.
    f === :sdp && return Rational{Int}[
        _q(artifact, 1), scale * _q(artifact, 2), _q(artifact, 4)]
    f === :rsoc && return Rational{Int}[1, 1]
    f === :mixed && return copy(artifact.coefficients)
    f in (:lp, :nonpositive) && return Rational{Int}[scale * q for q in artifact.coefficients]
    return copy(artifact.coefficients[1:artifact.dimension])
end

function _oracle_dual_multipliers(artifact::V2ConicArtifact)
    f = artifact.family
    f === :sdp && return Rational{Int}[-1, 0, 0, -1]
    f === :rsoc && return Rational{Int}[-1, -1]
    f === :mixed && return Rational{Int}[-1, 1]
    f === :soc && return fill(-1//1, artifact.dimension)
    f === :nonpositive && return fill(1//1, artifact.dimension)
    f === :lp && return fill(-1//1, artifact.dimension)
    f in (:exp, :power) && return Rational{Int}[-1]
    throw(ArgumentError("unknown artifact family $f"))
end

function _oracle_objective(artifact::V2ConicArtifact)
    f = artifact.family
    f === :sdp && return _q(artifact, 1) + _q(artifact, 4)
    f === :rsoc && return 2//1
    f === :mixed && return _q(artifact, 1) - _q(artifact, 2)
    f === :soc && return sum(artifact.coefficients[1:artifact.dimension])
    f === :nonpositive && return artifact.cone_parameter *
        (-sum(artifact.coefficients[1:artifact.dimension]))
    f === :lp && return artifact.cone_parameter *
        sum(artifact.coefficients[1:artifact.dimension])
    f in (:exp, :power) && return _q(artifact, 1)
    throw(ArgumentError("unknown artifact family $f"))
end

function _native_reference(artifact::V2ConicArtifact)
    if artifact.infeasible
        return V2Reference(:xfail, :farkas, nothing,
            V2ExactOracle(:primal_infeasible, 0//1, _oracle_witness(artifact), Rational{Int}[],
                Rational{Int}[], 0//1, artifact.infeasibility_ray, artifact),
            "independently described infeasible sentinel; expected status is primal_infeasible";
            expected_status=:primal_infeasible, disposition=:XFAIL,
            prior_observed_status=:numerical_breakdown)
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
    lower, upper = setprecision(BigFloat, 256) do
        exact_value = BigFloat(numerator(objective)) / BigFloat(denominator(objective))
        # Exact decimal interval strings retain the rational oracle while
        # avoiding ambient-precision dependence.
        (string(exact_value - BigFloat(1) / BigFloat(10)^7),
         string(exact_value + BigFloat(1) / BigFloat(10)^7))
    end
    witness = _oracle_witness(artifact)
    dual_multipliers = _oracle_dual_multipliers(artifact)
    dual_bound = _oracle_objective(artifact)
    objective == dual_bound || throw(ArgumentError("exact primal/dual oracle mismatch"))
    oracle = V2ExactOracle(:optimal, objective, witness, dual_multipliers,
        _oracle_cone_dual_slacks(artifact), dual_bound,
        artifact.infeasibility_ray, artifact)
    V2Reference(:optimal, :optimal, (lower, upper), oracle, "independently reconstructed exact artifact objective";
        expected_status=:optimal, disposition=:PASS)
end

"""Typed exact LP oracle for the first real V2 lowering tranche.

The oracle is deliberately independent of the solver: it checks the exact
source witness, equality feasibility, cone signs, dual feasibility, strong
duality, and the returned objective.  The model fingerprint check binds those
checks to the actual lowered model rather than to a reconstructed formula.
"""
struct V2LPOracle
    artifact::LPArtifact
end

function _put!(io::IO, oracle::V2LPOracle)
    _put!(io, (:V2LPOracle, oracle.artifact))
end

function _lp_eval(expr, witness)
    value = BigFloat(expr.constant)
    for (index, coefficient) in zip(expr.indices, expr.coefficients)
        value += BigFloat(coefficient) *
            BigFloat(numerator(witness[index])) / BigFloat(denominator(witness[index]))
    end
    value
end

function (oracle::V2LPOracle)(built, certificate)
    artifact = oracle.artifact
    built.source_artifact === artifact || return false
    actual_fp = _native_model_fingerprint(built.problem)
    actual_fp == built.facts.model_fingerprint || return false
    expected_fp = get(built.facts, :model_contract_fingerprint, "")
    actual_fp == expected_fp || return false

    # Non-optimal contracts use exact normalized certificates rather than an
    # objective comparison.  The returned public certificate still supplies
    # the original-coordinate residual/separation gate in run_instance.
    if artifact.expected_status === :primal_infeasible
        y = artifact.dual_witness
        length(y) == size(artifact.A, 1) || return false
        # Farkas: A' y lies in the dual cone and b' y is strictly negative.
        for col in axes(artifact.A, 2)
            lhs = sum(BigFloat(artifact.A[row, col]) * y[row]
                      for row in axes(artifact.A, 1))
            domain = artifact.cone_partition[col]
            domain === :nonnegative && lhs < 0 && return false
            domain === :nonpositive && lhs > 0 && return false
            domain === :free && !iszero(lhs) && return false
        end
        pairing = sum(BigFloat(artifact.b[row]) * y[row]
                      for row in axes(artifact.A, 1))
        return pairing < 0
    elseif artifact.expected_status === :dual_infeasible
        d = artifact.primal_witness
        length(d) == size(artifact.A, 2) || return false
        # Improving recession ray: A d = 0, d lies in the primal cone, and
        # c' d is strictly negative for this minimization LP.
        for row in axes(artifact.A, 1)
            sum(BigFloat(artifact.A[row, col]) * d[col]
                for col in axes(artifact.A, 2)) == 0 || return false
        end
        for (value, domain) in zip(d, artifact.cone_partition)
            domain === :nonnegative && value < 0//1 && return false
            domain === :nonpositive && value > 0//1 && return false
            domain === :free || continue
        end
        improvement = sum(BigFloat(artifact.c[col]) * d[col]
                          for col in axes(artifact.A, 2))
        return improvement < 0
    end

    artifact.expected_status === :optimal || return false
    witness = artifact.primal_witness
    length(witness) == size(artifact.A, 2) || return false
    # Check the source witness in exact arithmetic against every lowered row.
    for row in axes(artifact.A, 1)
        sum(BigFloat(artifact.A[row, col]) *
            (BigFloat(numerator(witness[col])) / BigFloat(denominator(witness[col])))
            for col in axes(artifact.A, 2)) == BigFloat(artifact.b[row]) || return false
    end
    for (value, domain) in zip(witness, artifact.cone_partition)
        domain === :nonnegative && value < 0//1 && return false
        domain === :nonpositive && value > 0//1 && return false
    end
    # The exact dual witness is an equality multiplier vector.  For a
    # minimization LP in standard form, A' y <= c on nonnegative variables.
    length(artifact.dual_witness) == size(artifact.A, 1) || return false
    for col in axes(artifact.A, 2)
        lhs = sum(BigFloat(artifact.A[row, col]) * artifact.dual_witness[row]
                  for row in axes(artifact.A, 1))
        rhs = BigFloat(artifact.c[col])
        if artifact.cone_partition[col] === :nonnegative
            lhs <= rhs || return false
        elseif artifact.cone_partition[col] === :free
            lhs == rhs || return false
        elseif artifact.cone_partition[col] === :nonpositive
            lhs >= rhs || return false
        end
    end
    primal_value = sum(BigFloat(artifact.c[col]) *
        (BigFloat(numerator(witness[col])) / BigFloat(denominator(witness[col])))
        for col in axes(artifact.A, 2))
    dual_value = sum(BigFloat(artifact.b[row]) * artifact.dual_witness[row]
                     for row in axes(artifact.A, 1))
    primal_value == BigFloat(artifact.objective) || return false
    dual_value == BigFloat(artifact.objective) || return false
    value = try BigFloat(certificate.primal_objective) catch; BigFloat(NaN) end
    # The reviewed Float64 certificate limit is 5e-7. Objective comparison
    # uses that declared arithmetic allowance (not a solver-tolerance change),
    # matching the V2 rule that an objective interval is at least as wide as
    # the arithmetic certificate allowance.
    isfinite(value) && abs(value - BigFloat(artifact.objective)) <= BigFloat("5e-7")
end

function _lp_build(artifact::LPArtifact, ::Type{T}; precision_bits::Int=256) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits, name=String(artifact.id)) :
        SDPX.Model(T; name=String(artifact.id))
    x = SDPX.variable!(model, :x, size(artifact.A, 2);
        domain=all(==( :nonnegative), artifact.cone_partition) ? SDPX.Nonnegative() :
            all(==( :nonpositive), artifact.cone_partition) ? SDPX.Nonpositive() : SDPX.Reals())
    # Mixed sign partitions are represented with a real block plus explicit
    # sign cones below; the first tranche's box uses one nonnegative block.
    for row in axes(artifact.A, 1)
        expression = -_Tq(T, artifact.b[row])
        for col in axes(artifact.A, 2)
            coefficient = _Tq(T, artifact.A[row, col])
            iszero(coefficient) || (expression += coefficient * x[col])
        end
        SDPX.constraint!(model, Symbol(:lp_row_, row), expression, SDPX.ZeroCone())
    end
    objective = zero(T)
    for col in axes(artifact.A, 2)
        coefficient = _Tq(T, artifact.c[col])
        iszero(coefficient) || (objective += coefficient * x[col])
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    actual_fp = _native_model_fingerprint(model)
    transform = V2Transform(:lp_small_artifact, :sdpx_cone_program,
        :lp_standard_form, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    oracle = V2LPOracle(artifact)
    facts = (artifact_fingerprint=_hex(artifact), model_fingerprint=actual_fp,
        model_contract_fingerprint=actual_fp, model_precision_bits=precision_bits,
        source_dimension=size(artifact.A, 2), target_dimension=size(artifact.A, 2),
        generator=artifact.generator_id, coefficients=artifact.A)
    return V2Built(model, oracle, artifact, "", transform, facts, (setup_seconds=nothing,))
end

"""Independent oracle for an ill-conditioned LP artifact lowered through the
same exact standard-form builder as the typed LP tranche."""
struct V2IllConditionedLPOracle
    artifact::IllConditionedArtifact
end

function _put!(io::IO, oracle::V2IllConditionedLPOracle)
    _put!(io, (:V2IllConditionedLPOracle, oracle.artifact))
end

function (oracle::V2IllConditionedLPOracle)(built, certificate)
    artifact = oracle.artifact
    artifact.base_family === :lp || return false
    built.source_artifact === artifact || return false
    actual_fp = _native_model_fingerprint(built.problem)
    actual_fp == built.facts.model_fingerprint || return false
    actual_fp == get(built.facts, :model_contract_fingerprint, "") || return false
    witness = artifact.primal_witness
    dual = artifact.dual_witness
    size(artifact.coefficients, 2) == length(witness) || return false
    size(artifact.coefficients, 1) == length(artifact.rhs) == length(dual) || return false
    for row in axes(artifact.coefficients, 1)
        sum(BigFloat(artifact.coefficients[row, col]) *
            (BigFloat(numerator(witness[col])) / BigFloat(denominator(witness[col])))
            for col in axes(artifact.coefficients, 2)) == BigFloat(artifact.rhs[row]) || return false
    end
    all(>=(0//1), witness) || return false
    for col in axes(artifact.coefficients, 2)
        lhs = sum(BigFloat(artifact.coefficients[row, col]) * dual[row]
                  for row in axes(artifact.coefficients, 1))
        lhs <= BigFloat(artifact.objective_coefficients[col]) || return false
    end
    primal_value = sum(BigFloat(artifact.objective_coefficients[col]) *
        (BigFloat(numerator(witness[col])) / BigFloat(denominator(witness[col])))
        for col in axes(artifact.coefficients, 2))
    dual_value = sum(BigFloat(artifact.rhs[row]) * dual[row]
                     for row in axes(artifact.coefficients, 1))
    primal_value == BigFloat(artifact.objective) || return false
    dual_value == BigFloat(artifact.objective) || return false
    value = try BigFloat(certificate.primal_objective) catch; BigFloat(NaN) end
    isfinite(value) && abs(value - BigFloat(artifact.objective)) <= BigFloat("5e-7")
end

function _ill_lp_build(artifact::IllConditionedArtifact, ::Type{T};
                       precision_bits::Int=256) where {T<:AbstractFloat}
    artifact.base_family === :lp || throw(ArgumentError("ill LP builder requires base_family=:lp"))
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits, name=String(artifact.id)) :
        SDPX.Model(T; name=String(artifact.id))
    x = SDPX.variable!(model, :x, size(artifact.coefficients, 2); domain=SDPX.Nonnegative())
    for row in axes(artifact.coefficients, 1)
        expression = -_Tq(T, artifact.rhs[row])
        for col in axes(artifact.coefficients, 2)
            coefficient = _Tq(T, artifact.coefficients[row, col])
            iszero(coefficient) || (expression += coefficient * x[col])
        end
        SDPX.constraint!(model, Symbol(:ill_lp_row_, row), expression, SDPX.ZeroCone())
    end
    objective = zero(T)
    for col in axes(artifact.objective_coefficients, 1)
        coefficient = _Tq(T, artifact.objective_coefficients[col])
        iszero(coefficient) || (objective += coefficient * x[col])
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    actual_fp = _native_model_fingerprint(model)
    transform = V2Transform(:ill_conditioned_lp_artifact, :sdpx_cone_program,
        :lp_standard_form, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    oracle = V2IllConditionedLPOracle(artifact)
    facts = (artifact_fingerprint=_hex(artifact), model_fingerprint=actual_fp,
        model_contract_fingerprint=actual_fp, model_precision_bits=precision_bits,
        source_dimension=size(artifact.coefficients, 2), target_dimension=size(artifact.coefficients, 2),
        generator=artifact.generator_id, coefficients=artifact.coefficients)
    V2Built(model, oracle, artifact, "", transform, facts, (setup_seconds=nothing,))
end

function _ill_diagonal_scale_artifact()
    # Row scaling D=diag(10^-6,10^6) preserves the feasible set and optimum
    # while producing an exact six-decade coefficient range.  Starting from
    # x+s=(1,2), c=(-1,-2,0,0), transform A,b by D and the equality dual by D^-1.
    A = Rational{Int}[1//1 0//1 1//1 0//1;
                      0//1 1//1 0//1 1//1]
    b = Rational{Int}[1, 2]
    d = Rational{Int}[1//1_000_000, 1_000_000//1]
    scaled_A = Diagonal(d) * A
    scaled_b = d .* b
    c = Rational{Int}[-1, -2, 0, 0]
    witness = Rational{Int}[1, 2, 0, 0]
    dual = Rational{Int}[-1_000_000, -1//500_000]
    IllConditionedArtifact(:v2_ill_diag_ladder_small, :diagonal_scale_ladder, :lp,
        scaled_A, scaled_b, c; scale_exponent=6, primal_witness=witness,
        dual_witness=dual, objective=-5//1)
end

function _ill_near_rank_loss_artifact()
    # Exact near-dependence: row 3 = row 1 + 10^-8 row 2.  The rational
    # source retains the tiny coefficient; no rank reduction is performed.
    A = Rational{Int}[1 0 1; 0 1 0; 1 1//100_000_000 1]
    b = Rational{Int}[1, 1, 100_000_001//100_000_000]
    c = Rational{Int}[-1, -1, 0]
    witness = Rational{Int}[1, 1, 0]
    dual = Rational{Int}[-1, -1, 0]
    IllConditionedArtifact(:v2_ill_near_rank_loss_small, :near_rank_loss_lp, :lp,
        A, b, c; scale_exponent=8, primal_witness=witness,
        dual_witness=dual, objective=-2//1)
end

function _ill_near_boundary_lp_artifact()
    # The second equality x+s=1 leaves the nonnegative slack exactly 10^-8;
    # the first equality fixes x=1-10^-8.  Objective -x makes the planted
    # point the unique optimum while retaining an exactly rational boundary
    # distance, without changing the solver tolerances.
    epsilon = 1//100_000_000
    A = Rational{Int}[1 0; 1 1]
    b = Rational{Int}[1 - epsilon, 1]
    c = Rational{Int}[-1, 0]
    witness = Rational{Int}[1 - epsilon, epsilon]
    dual = Rational{Int}[-1, 0]
    IllConditionedArtifact(:v2_ill_near_boundary_lp_small, :near_boundary_lp, :lp,
        A, b, c; scale_exponent=8, primal_witness=witness,
        dual_witness=dual, objective=-(1 - epsilon))
end

function _lp_box_artifact()
    # min -x₁-2x₂ subject to x+s=(1,2), x,s >= 0.  The planted point
    # (1,2,0,0) is optimal; y=(-1,-2) is dual-feasible and gives -5.
    A = Rational{Int}[1 0 1 0; 0 1 0 1]
    b = Rational{Int}[1, 2]
    c = Rational{Int}[-1, -2, 0, 0]
    LPArtifact(:v2_lp_box_small, :box, A, b, c;
        primal_witness=Rational{Int}[1, 2, 0, 0],
        dual_witness=Rational{Int}[-1, -2], objective=-5//1)
end

"""A solve-eligible V2 catalog containing only independently certified cases."""
function _lp_sparse_planted_artifact()
    # min -x₁-3x₂ subject to A*x=b, x>=0, with
    # A=[1 1 1 0; 0 1 0 1], b=(2,1).  The exact planted point
    # (1,1,0,0) has value -4; y=(-1,-2) satisfies A'y<=c and b'y=-4.
    A = Rational{Int}[1 1 1 0; 0 1 0 1]
    b = Rational{Int}[2, 1]
    c = Rational{Int}[-1, -3, 0, 0]
    LPArtifact(:v2_lp_sparse_planted_small, :sparse_planted_kkt, A, b, c;
        primal_witness=Rational{Int}[1, 1, 0, 0],
        dual_witness=Rational{Int}[-1, -2], objective=-4//1)
end

function _lp_duplicate_artifact()
    # Duplicate and dependent equalities are retained, not rank-reduced:
    # x₁+x₂+s=1, the same row again, and twice that row.  The planted
    # (1,0,0), y=(-1,0,0) pair proves the exact optimum -1.
    A = Rational{Int}[1 1 1; 1 1 1; 2 2 2]
    b = Rational{Int}[1, 1, 2]
    c = Rational{Int}[-1, -1, 0]
    LPArtifact(:v2_lp_duplicate_rank_deficient_small, :duplicate_rank_deficient,
        A, b, c; primal_witness=Rational{Int}[1, 0, 0],
        dual_witness=Rational{Int}[-1, 0, 0], objective=-1//1)
end

function _lp_nonpositive_artifact()
    # Sign-reconstruction sentinel: min x subject to x=-2 and x <= 0.
    # Flipping z=-x gives the known nonnegative optimum z=2; the exact
    # primal/dual pair x=-2, y=1 proves c'x=b'y=-2.
    A = reshape(Rational{Int}[1], 1, 1)
    b = Rational{Int}[-2]
    c = Rational{Int}[1]
    LPArtifact(:v2_lp_nonpositive_small, :nonpositive, A, b, c;
        cone_partition=[:nonpositive], primal_witness=Rational{Int}[-2],
        dual_witness=Rational{Int}[1], objective=-2//1)
end

function _lp_chebyshev_artifact()
    # Minimize the epigraph radius t for a 32-point Chebyshev fit. The
    # nonnegative standard-form variables are x[1:32], t, and two slack
    # vectors. For y_i=(-1)^i, x_i=max(y_i,0), t=1, and all slacks zero;
    # the negative datum forces t >= 1, so the exact minimax optimum is 1.
    n = 32
    nv = 3n + 1
    A = zeros(Rational{Int}, 2n, nv)
    b = zeros(Rational{Int}, 2n)
    y = Rational{Int}[isodd(i) ? -1//1 : 1//1 for i in 1:n]
    for i in 1:n
        # x_i - t + s⁺_i = y_i and -x_i - t + s⁻_i = -y_i.
        A[i, i] = 1//1
        A[i, n + 1] = -1//1
        A[i, n + 1 + i] = 1//1
        b[i] = y[i]
        A[n + i, i] = -1//1
        A[n + i, n + 1] = -1//1
        A[n + i, 2n + 1 + i] = 1//1
        b[n + i] = -y[i]
    end
    c = zeros(Rational{Int}, nv)
    c[n + 1] = 1//1
    x = Rational{Int}[max(y[i], 0//1) for i in 1:n]
    # The slacks are t - (x_i-y_i) and t - (y_i-x_i), respectively;
    # negative data therefore has second slack 2 at the optimum.
    splus = Rational{Int}[1//1 - (x[i] - y[i]) for i in 1:n]
    sminus = Rational{Int}[1//1 - (y[i] - x[i]) for i in 1:n]
    witness = vcat(x, Rational{Int}[1//1], splus, sminus)
    # Only the first (negative) datum is active in the dual; its multiplier
    # -1 gives A'λ <= c and b'λ = 1 exactly.
    dual = zeros(Rational{Int}, 2n)
    dual[1] = -1//1
    LPArtifact(:v2_lp_chebyshev_small, :chebyshev, A, b, c;
        primal_witness=witness, dual_witness=dual, objective=1//1)
end

function _lp_primal_infeasible_artifact()
    # Contradictory equalities x=1 and x=2 with x >= 0.  The independent
    # Farkas multiplier y=(1,-1) annihilates A and has b' y=-1 < 0.
    A = reshape(Rational{Int}[1, 1], 2, 1)
    b = Rational{Int}[1, 2]
    c = Rational{Int}[0]
    LPArtifact(:v2_lp_primal_infeasible_small, :primal_infeasible, A, b, c;
        primal_witness=Rational{Int}[], dual_witness=Rational{Int}[1, -1],
        objective=0//1, expected_status=:primal_infeasible,
        certificate_kind=:farkas)
end

function _lp_unbounded_artifact()
    # min -x₁ subject to x₁-x₂=0 and x >= 0.  The exact improving ray
    # d=(1,1) satisfies A*d=0 and c'd=-1.
    A = Rational{Int}[1 -1]
    b = Rational{Int}[0]
    c = Rational{Int}[-1, 0]
    LPArtifact(:v2_lp_unbounded_small, :unbounded, A, b, c;
        primal_witness=Rational{Int}[1, 1], dual_witness=Rational{Int}[],
        objective=0//1, expected_status=:dual_infeasible,
        certificate_kind=:ray)
end

function lp_tranche_catalog()
    # Only artifacts with an observed Float64 certificate may enter this
    # catalog.  The duplicate/rank-deficient construction is intentionally
    # left open: the current sparse route reports numerical_breakdown, so it
    # must not be mislabeled solve-eligible.
    artifacts = [_lp_box_artifact(), _lp_sparse_planted_artifact(),
        _lp_nonpositive_artifact(), _lp_chebyshev_artifact(),
        _lp_primal_infeasible_artifact(), _lp_unbounded_artifact()]
    transforms = V2Transform(:lp_small_artifact, :sdpx_cone_program,
        :lp_standard_form, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    descriptions = [
        "box: x+s=u with c=(-1,-2,0,0); y=(-1,-2) proves c'x=b'y=-5",
        "sparse planted KKT: A'x dual inequality and y=(-1,-2) prove c'x=b'y=-4",
        "nonpositive sign sentinel: x=-2 with y=1 is the sign-flipped nonnegative optimum",
        "Chebyshev epigraph: y_i=(-1)^i, x_i=max(y_i,0), t=1; the negative datum forces t>=1",
        "primal infeasible: contradictory x=1 and x=2 equalities with Farkas y=(1,-1)",
        "unbounded: d=(1,1) is an improving recession ray for min -x1",
    ]
    families = V2Family(:lp, V2Axis[],
        (instance, precision) -> begin
            built = _lp_build(instance.payload, precision.arithmetic;
                precision_bits=precision.bits)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform,
                merge(built.facts, (artifact_fingerprint=_hex(instance.payload),)), built.resource)
        end,
        (built, certificate) -> built.oracle(built, certificate),
        (instance, result) -> result.validation, (:identity,))
    instances = V2Instance[]
    for (index, artifact) in enumerate(artifacts)
        interval = artifact.expected_status === :optimal ?
            (string(BigFloat(artifact.objective) - BigFloat("5e-7")),
             string(BigFloat(artifact.objective) + BigFloat("5e-7"))) : nothing
        reference = V2Reference(artifact.expected_status, artifact.certificate_kind,
            interval, V2LPOracle(artifact), descriptions[index];
            expected_status=artifact.expected_status, disposition=:PASS)
        push!(instances, V2Instance(artifact.id, :lp,
            only(filter(t -> t.name === :small, _TIERS)),
            (kind=artifact.kind,), :train,
            "general-v2/lp/$(artifact.kind)/small",
            (generator=artifact.generator_id, version=artifact.generator_version,
             transform=transforms, solve_eligible=true), _hex(artifact),
            (wall_seconds=20, memory_bytes=4 * 1024^3), reference, artifact))
    end
    V2Catalog(:general_v2_lp_tranche, 3, [families], instances,
        (train=[instance.id for instance in instances], holdout=Symbol[], sentinel=Symbol[]))
end

"""Small ill-conditioned tranche with only observed certified artifacts."""
function ill_conditioned_tranche_catalog()
    artifacts = [_ill_diagonal_scale_artifact(), _ill_near_rank_loss_artifact()]
    transform = V2Transform(:ill_conditioned_lp_artifact, :sdpx_cone_program,
        :lp_standard_form, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    descriptions = [
        "diagonal scale ladder: D=diag(10^-6,10^6) applied to x+s=(1,2); D^-1 dual preserves exact optimum -5",
        "near-rank-loss LP: row3=row1+10^-8*row2 retained exactly; planted primal/dual pair proves optimum -2",
    ]
    family = V2Family(:ill_conditioned, V2Axis[],
        (instance, precision) -> begin
            built = _ill_lp_build(instance.payload, precision.arithmetic;
                precision_bits=precision.bits)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform,
                merge(built.facts, (artifact_fingerprint=_hex(instance.payload),)), built.resource)
        end,
        (built, certificate) -> built.oracle(built, certificate),
        (instance, result) -> result.validation, (:identity,))
    instances = V2Instance[]
    for (index, artifact) in enumerate(artifacts)
        interval = (string(BigFloat(artifact.objective) - BigFloat("5e-7")),
            string(BigFloat(artifact.objective) + BigFloat("5e-7")))
        reference = V2Reference(:optimal, :optimal, interval,
            V2IllConditionedLPOracle(artifact), descriptions[index];
            expected_status=:optimal, disposition=:PASS)
        push!(instances, V2Instance(artifact.id, :ill_conditioned,
            only(filter(t -> t.name === :small, _TIERS)),
            (kind=artifact.kind, scale_exponent=artifact.scale_exponent), :train,
            "general-v2/ill-conditioned/$(artifact.kind)/small",
            (generator=artifact.generator_id, version=artifact.generator_version,
             transform=transform, solve_eligible=true), _hex(artifact),
            (wall_seconds=20, memory_bytes=4 * 1024^3), reference, artifact))
    end
    V2Catalog(:general_v2_ill_conditioned_tranche, 1, [family], instances,
        (train=[instance.id for instance in instances], holdout=Symbol[], sentinel=Symbol[]))
end

function native_v2_catalog()
    families = Any[]
    family_names = (:lp, :nonpositive, :soc, :rsoc, :sdp, :exp, :power, :mixed)
    for family in family_names
        build = (instance, precision) -> begin
            built = _native_build(instance.payload, precision.arithmetic;
                precision_bits=precision.bits)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform,
                merge(built.facts, (artifact_fingerprint=_hex(instance.payload),
                    model_fingerprint=_native_model_fingerprint(built.problem),
                    model_precision_bits=precision.bits,
                    builder_version=instance.payload.generator_version)), built.resource)
        end
        oracle = (built, cert) -> built.oracle(built, cert)
        validate = (instance, result) -> result.validation
        push!(families, V2Family(family, V2Axis[], build, oracle, validate, (:identity,)))
    end
    tiers = Dict(t.name=>t for t in _TIERS)
    instances = V2Instance[]
    native_transform = V2Transform(:native_conic_artifact, :sdpx_cone_program,
        :identity, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    train = Symbol[]; holdout = Symbol[]; sentinel = Symbol[]
    for family in family_names
        for (split, bucket, infeasible) in ((:train, train, false), (:holdout, holdout, false), (:sentinel, sentinel, true))
            id = Symbol("v2_$(family)_$(split)")
            artifact = _native_artifact(family, id; split, infeasible)
            ref = _native_reference(artifact)
            # `split` is provenance/suite metadata, never mathematical axis
            # data. The artifact itself is the complete mathematical payload.
            params = (family=family, dimension=artifact.dimension,
                coefficients=artifact.coefficients, cone_parameter=artifact.cone_parameter,
                infeasible=artifact.infeasible)
            push!(instances, V2Instance(id, family, tiers[:small], params, split,
                "general-v2/native/$(family)/$(split)",
                (generator=artifact.generator_id, version=artifact.generator_version,
                 artifact_fields=(:coefficients, :dimension, :cone_parameter, :infeasible),
                 transform=native_transform, solve_eligible=(split !== :sentinel)),
                _hex(artifact), (wall_seconds=20, memory_bytes=4*1024^3), ref, artifact))
            push!(bucket, id)
        end
    end
    return V2Catalog(:general_v2_native, 2, families, instances,
        (train=train, holdout=holdout, sentinel=sentinel))
end
