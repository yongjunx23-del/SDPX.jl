module SDPXAnalyticBenchmarks

using LinearAlgebra
using SHA
using SparseArrays
using SDPX

export AnalyticCase, family_names, build_problem, problem, oracle
export analytic_objective, analytic_primal, analytic_dual
export oracle_objective, oracle_primal, oracle_dual

"""A deterministic analytic problem and its independent mathematical oracle."""
struct AnalyticCase{P,O}
    family::Symbol
    tier::Symbol
    kind::Symbol
    parameters::NamedTuple
    problem::P
    oracle::O
    input_fingerprint::String
end

Base.eltype(case::AnalyticCase) = eltype(case.problem)
problem(case::AnalyticCase) = case.problem
oracle(case::AnalyticCase) = case.oracle
oracle_objective(case::AnalyticCase) = get(case.oracle, :objective, nothing)
oracle_primal(case::AnalyticCase) = get(case.oracle, :primal, nothing)
oracle_dual(case::AnalyticCase) = get(case.oracle, :dual, nothing)

const _TIERS = (:tier1, :tier2, :tier3)

family_names() = (
    :chebyshev_lp,
    :weighted_minimum_norm_socp,
    :basel_soc_chain,
    :spectral_path_sdp,
    :odd_cycle_maxcut_sdp,
    :rational_moment_sdp,
)

function _check_tier(tier::Symbol)
    tier in _TIERS || throw(ArgumentError(
        "tier must be one of $(_TIERS), got $(repr(tier))",
    ))
    return tier
end

function _tier_value(tier::Symbol, values::NTuple{3,Int})
    tier === :tier1 && return values[1]
    tier === :tier2 && return values[2]
    return values[3]
end

function _typed(::Type{T}, value) where {T}
    value isa AbstractString || return T(value)
    value = parse(BigFloat, value)
    return T(value)
end

"""
Evaluate a transcendental function in BigFloat and convert to `T`.

Float64 and BigFloat keep their native library paths so existing benchmark
inputs are bit-for-bit unchanged.  Other supported arithmetics (notably
MultiFloat) may not implement every transcendental function; those inputs are
generated at more than twice the target precision and then converted
explicitly, rather than relying on a global MultiFloats switch.
"""
function _transcendental_as_t(::Type{T}, f::F, args::Vararg{Any,N}) where {T,F,N}
    if T === Float64 || T === BigFloat
        return f(args...)
    end
    bits = 2 * SDPX.sig_bits(T) + 64
    value = setprecision(BigFloat, bits) do
        f(map(BigFloat, args)...)
    end
    return T(value)
end

_cos_t(::Type{T}, args...) where {T} = _transcendental_as_t(T, cos, args...)
_sin_t(::Type{T}, args...) where {T} = _transcendental_as_t(T, sin, args...)
_acos_t(::Type{T}, args...) where {T} = _transcendental_as_t(T, acos, args...)
_log_t(::Type{T}, args...) where {T} = _transcendental_as_t(T, log, args...)
_exp2_t(::Type{T}, args...) where {T} = _transcendental_as_t(T, exp2, args...)

function _facts(p)
    if p isa SDPX.ConicProblem
        return (
            variables=p.variables,
            equalities=size(p.Aeq, 1),
            blocks=length(p.cones),
            block_sizes=Tuple(size(cone.A, 1) for cone in p.cones),
        )
    elseif p isa SDPX.SDPProblem
        return (
            variables=p.dims.m,
            equalities=p.dims.n,
            blocks=p.dims.L,
            block_sizes=Tuple(p.dims.k),
        )
    end
    throw(ArgumentError("unsupported analytic problem type $(typeof(p))"))
end

function _fingerprint(family, tier, parameters, p)
    facts = _facts(p)
    sizes = facts.block_sizes
    compact_facts = (
        variables=facts.variables,
        equalities=facts.equalities,
        blocks=facts.blocks,
        block_size_min=isempty(sizes) ? 0 : minimum(sizes),
        block_size_max=isempty(sizes) ? 0 : maximum(sizes),
        block_size_digest=_digest_values(sizes),
    )
    payload = sprint(show, (family=family, tier=tier, parameters=parameters,
                            facts=compact_facts))
    return bytes2hex(SHA.sha256(collect(codeunits(payload))))
end

"""Bounded-memory digest for a deterministic numeric parameter vector."""
function _digest_values(values)
    io = IOBuffer()
    for value in values
        print(io, repr(value), ';')
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _case(family, tier, kind, parameters, p, o)
    return AnalyticCase(
        family, tier, kind, parameters, p, o,
        _fingerprint(family, tier, parameters, p),
    )
end

"""Return monomial coefficients of `T_n(x)` in ascending order."""
function _chebyshev_monomial(::Type{T}, n::Int) where {T}
    n == 0 && return T[one(T)]
    t0 = T[one(T)]
    t1 = T[zero(T), one(T)]
    n == 1 && return t1
    for _ in 2:n
        next = zeros(T, length(t1) + 1)
        @inbounds for j in eachindex(t1)
            next[j + 1] += T(2) * t1[j]
        end
        @inbounds for j in eachindex(t0)
            next[j] -= t0[j]
        end
        t0, t1 = t1, next
    end
    return t1
end

"""Chebyshev alternation LP in either a stable or monomial basis.

The target is `x^n` and the approximating polynomial has degree `< n`.
The `n+1` alternation nodes are always present; optional extra nodes only add
constraints and therefore do not change the analytic optimum.
"""
function _chebyshev_lp(
    ::Type{T}, tier::Symbol;
    basis::Symbol=:chebyshev,
    degree::Union{Nothing,Integer}=nothing,
    extra_grid::Int=0,
) where {T}
    basis in (:chebyshev, :monomial) || throw(ArgumentError(
        "basis must be :chebyshev or :monomial",
    ))
    n = degree === nothing ? _tier_value(tier, (8, 32, 128)) : Int(degree)
    n >= 1 || throw(ArgumentError("degree must be positive"))
    extra_grid >= 0 || throw(ArgumentError("extra_grid must be nonnegative"))

    # Keep the exact alternation grid and append a deterministic dense grid.
    theta = T(pi) .* (T.(0:n) ./ T(n))
    xs = [_cos_t(T, value) for value in theta]
    if extra_grid > 0
        dense = [_cos_t(T, T(pi) * T(j) / T(extra_grid + 1))
                 for j in 1:extra_grid]
        append!(xs, dense)
    end
    rows_per_side = length(xs)
    variables = n + 1                         # n coefficients and t
    G = zeros(T, 2 * rows_per_side, variables)
    h = zeros(T, 2 * rows_per_side)
    @inbounds for (row, x) in enumerate(xs)
        target = x^n
        for k in 0:(n - 1)
            value = basis === :monomial ? x^k :
                    _cos_t(T, T(k) * _acos_t(T, x))
            G[row, k + 1] = value
            G[rows_per_side + row, k + 1] = -value
        end
        G[row, variables] = one(T)
        G[rows_per_side + row, variables] = one(T)
        h[row] = target
        h[rows_per_side + row] = -target
    end
    c = zeros(T, variables)
    c[variables] = one(T)
    p = SDPX.linear_program(c, G, h; T=T, sparse=false, verbosity=0)

    optimum = T(2)^(1 - n)
    # p*(x)=x^n - 2^(1-n) T_n(x).  The degree-n term cancels exactly.
    cheb_coefficients = zeros(T, n)
    residual = zeros(T, n + 1)
    residual[n + 1] = one(T)
    cheb_polynomials = Vector{Vector{T}}(undef, n + 1)
    for k in 0:n
        cheb_polynomials[k + 1] = _chebyshev_monomial(T, k)
    end
    # Expand x^n in the Chebyshev basis by descending leading degree.
    expansion = zeros(T, n + 1)
    for k in n:-1:0
        leading = k == 0 ? one(T) : T(2)^(k - 1)
        coefficient = residual[k + 1] / leading
        expansion[k + 1] = coefficient
        poly = cheb_polynomials[k + 1]
        @inbounds for j in eachindex(poly)
            residual[j] -= coefficient * poly[j]
        end
    end
    if basis === :chebyshev
        cheb_coefficients .= expansion[1:n]
    else
        poly = cheb_polynomials[n + 1]
        @inbounds for j in 1:n
            cheb_coefficients[j] = -optimum * poly[j]
        end
    end
    x_oracle = vcat(cheb_coefficients, optimum)
    parameters = (
        basis=basis,
        degree=n,
        alternation_nodes=n + 1,
        extra_grid=extra_grid,
    )
    o = (
        objective=optimum,
        primal=x_oracle,
        dual=nothing,
        physical_objective=x -> x[end],
        objective_kind=:exact,
        expected_status=:optimal,
    )
    return _case(:chebyshev_lp, tier, :lp, parameters, p, o)
end

function _normalized_weights(::Type{T}, n::Int, spread) where {T}
    s = _typed(T, spread)
    n == 1 && return T[one(T)]
    midpoint = T(1) / T(2)
    return [exp2(s * (T(i - 1) / T(n - 1) - midpoint)) for i in 1:n]
end

"""Weighted minimum norm with one large Lorentz cone."""
function _weighted_minimum_norm_socp(
    ::Type{T}, tier::Symbol;
    dimension::Union{Nothing,Integer}=nothing,
    spread=zero(T),
) where {T}
    n = dimension === nothing ? _tier_value(tier, (32, 512, 8192)) : Int(dimension)
    n >= 1 || throw(ArgumentError("dimension must be positive"))
    weights = _normalized_weights(T, n, spread)
    variables = n + 1
    t = variables
    A = zeros(T, n + 1, variables)
    A[1, t] = one(T)
    @inbounds for i in 1:n
        A[i + 1, i] = weights[i]
    end
    cone = SDPX.SOCConstraint(A, zeros(T, n + 1); T=T)
    Aeq = zeros(T, 1, variables)
    Aeq[1, 1:n] .= one(T)
    p = SDPX.second_order_program(
        [zeros(T, n); one(T)], [cone]; Aeq=Aeq, beq=T[one(T)], T=T,
    )
    invsq = [inv(weights[i]^2) for i in 1:n]
    denominator = sum(invsq)
    xstar = invsq ./ denominator
    tstar = inv(sqrt(denominator))
    parameters = (
        dimension=n,
        spread=_typed(T, spread),
        weight_min=first(weights),
        weight_max=last(weights),
        weight_digest=_digest_values(weights),
    )
    o = (
        objective=tstar,
        primal=vcat(xstar, tstar),
        x_star=xstar,
        dual=nothing,
        physical_objective=x -> x[end],
        objective_kind=:exact,
        expected_status=:optimal,
    )
    return _case(:weighted_minimum_norm_socp, tier, :socp, parameters, p, o)
end

function _basel_scales(::Type{T}, n::Int, spread) where {T}
    s = _typed(T, spread)
    n == 1 && return T[one(T)]
    return [exp2(s * (T(k - 1) / T(n - 1) - T(1) / T(2))) for k in 1:n]
end

"""Basel chain as native Q3 blocks or equivalent 2x2 PSD blocks."""
function _basel_soc_chain(
    ::Type{T}, tier::Symbol;
    representation::Symbol=:native,
    terms::Union{Nothing,Integer}=nothing,
    spread=zero(T),
) where {T}
    representation in (:native, :psd2) || throw(ArgumentError(
        "representation must be :native or :psd2",
    ))
    n = terms === nothing ? _tier_value(tier, (10, 1000, 10000)) : Int(terms)
    n >= 1 || throw(ArgumentError("terms must be positive"))
    scales = _basel_scales(T, n, spread)
    c = zeros(T, n)
    c[n] = one(T)
    if representation === :native
        cones = SDPX.SOCConstraint{T}[]
        for k in 1:n
            # Each Q3 block touches at most two chain variables.  Keeping the
            # affine map sparse avoids materialising an N×N grid of zeros when
            # the chain is scaled to thousands of cones.
            A = spzeros(T, 3, n)
            A[1, k] = scales[k]
            k > 1 && (A[2, k - 1] = scales[k])
            b = T[zero(T), zero(T), scales[k] / T(k)]
            push!(cones, SDPX.SOCConstraint(A, b; T=T))
        end
        p = SDPX.second_order_program(c, cones; T=T)
    else
        # Reuse SDPX's active-only sparse coefficient container.  Each 2×2
        # block touches at most two chain variables; materialising an n-entry
        # reference vector for every block would make setup quadratic before
        # the solver has a chance to expose its own scaling limit.
        coefficients = Vector{SDPX.ActiveSparseCoefficientVector{T}}(undef, n)
        constants = Vector{SparseMatrixCSC{T,Int}}(undef, n)
        for k in 1:n
            active = k == 1 ? [1] : [k - 1, k]
            block = SparseMatrixCSC{T,Int}[
                k == 1 ?
                    sparse(T[scales[k] 0; 0 scales[k]]) :
                    sparse(T[scales[k] 0; 0 -scales[k]]),
            ]
            k > 1 && push!(block, sparse(T[scales[k] 0; 0 scales[k]]))
            coefficients[k] = SDPX.ActiveSparseCoefficientVector(
                T, n, active, block, 2,
            )
            q = scales[k] / T(k)
            constants[k] = sparse(T[0 -q; -q 0])
        end
        p = SDPX.ingest(c, coefficients, constants,
                        spzeros(T, n, 0), T[];
                        T=T, sparse=true, verbosity=0)
    end
    tstar = Vector{T}(undef, n)
    running = zero(T)
    @inbounds for k in 1:n
        running += inv(T(k)^2)
        tstar[k] = sqrt(running)
    end
    objective = tstar[n]
    o = (
        objective=objective,
        primal=tstar,
        dual=nothing,
        physical_objective=x -> x[end],
        objective_kind=:exact,
        expected_status=:optimal,
    )
    parameters = (
        representation=representation,
        terms=n,
        spread=_typed(T, spread),
        scale_min=first(scales),
        scale_max=last(scales),
        scale_digest=_digest_values(scales),
    )
    return _case(:basel_soc_chain, tier,
                 representation === :native ? :socp : :sdp,
                 parameters, p, o)
end

function _path_matrix(::Type{T}, n::Int, delta, second::Bool) where {T}
    M = spzeros(T, n, n)
    one_t = one(T)
    for i in 1:(n - 1)
        M[i, i + 1] = one_t
        M[i + 1, i] = one_t
    end
    if second
        d = _typed(T, delta)
        for i in 1:n
            M[i, i] -= d
        end
    end
    return M
end

function _packed_pairs(n::Int)
    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, n * (n + 1) ÷ 2)
    for j in 1:n, i in j:n
        push!(pairs, (i, j))
    end
    return pairs
end

"""Trace-one spectral SDP in the original matrix variable X."""
function _spectral_path_sdp(
    ::Type{T}, tier::Symbol;
    path_length::Union{Nothing,Integer}=nothing,
    delta="0",
    delta_power::Union{Nothing,Integer}=nothing,
    near_degenerate::Bool=false,
) where {T}
    n = path_length === nothing ? _tier_value(tier, (8, 64, 256)) : Int(path_length)
    n >= 2 || throw(ArgumentError("path_length must be at least two"))
    delta_power === nothing || delta_power >= 0 || throw(ArgumentError(
        "delta_power must be nonnegative",
    ))
    δ = delta_power === nothing ? _typed(T, delta) : exp2(-Int(delta_power))
    δ >= zero(T) || throw(ArgumentError("delta must be nonnegative"))
    # Keep the primary formulation as one growing PSD block.  The optional
    # direct-sum variant deliberately keeps both blocks at δ=0 so exact
    # degeneracy is represented instead of being collapsed away.
    near_degenerate || iszero(δ) || (near_degenerate = true)
    total = near_degenerate ? 2n : n
    Q = spzeros(T, total, total)
    first = _path_matrix(T, n, zero(T), false)
    Q[1:n, 1:n] = first
    if total > n
        Q[(n + 1):total, (n + 1):total] = _path_matrix(T, n, δ, true)
    end
    pairs = _packed_pairs(total)
    m = length(pairs)
    coefficients = Vector{SparseMatrixCSC{T,Int}}(undef, m)
    c = zeros(T, m)
    B = spzeros(T, m, 1)
    for (index, (i, j)) in enumerate(pairs)
        if i == j
            coefficients[index] = sparse([i], [j], T[one(T)], total, total)
            B[index, 1] = one(T)
            c[index] = -Q[i, i]
        else
            coefficients[index] = sparse([i, j], [j, i],
                                         T[one(T), one(T)], total, total)
            c[index] = -(Q[i, j] + Q[j, i])
        end
    end
    p = SDPX.ingest(c, [coefficients], [spzeros(T, total, total)], B, T[one(T)];
                    T=T, sparse=true, verbosity=0)
    λ = T(2) * _cos_t(T, T(pi) / T(n + 1))
    vsmall = [sqrt(T(2) / T(n + 1)) *
              _sin_t(T, T(i) * T(pi) / T(n + 1)) for i in 1:n]
    v = total == n ? vsmall : vcat(vsmall, zeros(T, n))
    X = v * transpose(v)
    xstar = [X[i, j] for (i, j) in pairs]
    o = (
        objective=λ,
        primal=xstar,
        dual=nothing,
        X=X,
        physical_objective=x -> -dot(c, x),
        objective_kind=:exact,
        expected_status=:optimal,
    )
    parameters = (
        path_length=n,
        delta=δ,
        delta_power=delta_power,
        near_degenerate=near_degenerate,
        matrix_dimension=total,
        objective=:largest_eigenvalue,
    )
    return _case(:spectral_path_sdp, tier, :sdp, parameters, p, o)
end

"""Odd-cycle MaxCut SDP, optionally with a redundant trace equality."""
function _odd_cycle_maxcut_sdp(
    ::Type{T}, tier::Symbol;
    vertices::Union{Nothing,Integer}=nothing,
    redundant::Bool=false,
) where {T}
    n = vertices === nothing ? _tier_value(tier, (5, 31, 127)) : Int(vertices)
    n >= 3 && isodd(n) || throw(ArgumentError(
        "vertices must be an odd integer at least three",
    ))
    pairs = _packed_pairs(n)
    indices = Dict(pair => i for (i, pair) in enumerate(pairs))
    m = length(pairs)
    coefficients = Vector{SparseMatrixCSC{T,Int}}(undef, m)
    c = zeros(T, m)
    B = spzeros(T, m, n + (redundant ? 1 : 0))
    for (index, (i, j)) in enumerate(pairs)
        if i == j
            coefficients[index] = sparse([i], [j], T[one(T)], n, n)
            B[index, i] = one(T)
            redundant && (B[index, n + 1] = one(T))
        else
            coefficients[index] = sparse([i, j], [j, i],
                                         T[one(T), one(T)], n, n)
        end
    end
    for i in 1:n
        j = i == n ? 1 : i + 1
        pair = i >= j ? (i, j) : (j, i)
        # SDPX uses minimization internally.  Minimizing +1/2*sum_edge Xij
        # is exactly the standard MaxCut maximization n/2 - 1/2*sum_edge Xij.
        c[indices[pair]] = one(T) / T(2)
    end
    rhs = vcat(ones(T, n), redundant ? T[T(n)] : T[])
    p = SDPX.ingest(c, [coefficients], [spzeros(T, n, n)], B, rhs;
                    T=T, sparse=true, verbosity=0)
    angle = T(pi) - T(pi) / T(n)
    θ = [T(i - 1) * angle for i in 1:n]
    X = [_cos_t(T, θ[i] - θ[j]) for i in 1:n, j in 1:n]
    xstar = [X[i, j] for (i, j) in pairs]
    optimum = T(n) / T(2) * (one(T) + _cos_t(T, T(pi) / T(n)))
    o = (
        objective=optimum,
        primal=xstar,
        dual=nothing,
        X=X,
        physical_objective=x -> T(n) / T(2) - dot(c, x),
        objective_kind=:exact,
        expected_status=:optimal,
    )
    parameters = (
        vertices=n,
        redundant=redundant,
        edge_count=n,
        objective=:maxcut_value,
    )
    return _case(:odd_cycle_maxcut_sdp, tier, :sdp, parameters, p, o)
end

function _moment_block(
    ::Type{T}, dimension::Int, moment_count::Int, kind::Symbol,
) where {T}
    matrices = [spzeros(T, dimension, dimension) for _ in 1:moment_count]
    for i in 0:(dimension - 1), j in 0:(dimension - 1)
        if kind === :moment
            k = i + j
            matrices[k + 1][i + 1, j + 1] += one(T)
        elseif kind === :x
            k = i + j + 1
            matrices[k + 1][i + 1, j + 1] += one(T)
        elseif kind === :one_minus_x
            matrices[i + j + 1][i + 1, j + 1] += one(T)
            matrices[i + j + 2][i + 1, j + 1] -= one(T)
        else
            throw(ArgumentError("unknown moment block kind $kind"))
        end
    end
    return matrices
end

"""Moment SDP hierarchy for `∫₀¹ (1-rho*x)⁻¹ dx`.

The finite relaxation does not claim a closed-form optimum.  Its independent
oracle is the exact integral, and the harness checks lower/upper bound
direction and certificate validity instead of treating a hierarchy value as
an exact answer.
"""
function _rational_moment_sdp(
    ::Type{T}, tier::Symbol;
    order::Union{Nothing,Integer}=nothing,
    bound::Symbol=:lower,
    rho_power::Int=4,
    m::Union{Nothing,Integer}=nothing,
) where {T}
    bound in (:lower, :upper) || throw(ArgumentError("bound must be :lower or :upper"))
    d = order === nothing ? _tier_value(tier, (4, 16, 32)) : Int(order)
    d >= 1 || throw(ArgumentError("order must be positive"))
    m === nothing || (rho_power = Int(m))
    rho_power >= 1 || throw(ArgumentError("rho_power must be positive"))
    rho_gap = _exp2_t(T, T(-rho_power))
    zero(T) < rho_gap < one(T) ||
        throw(ArgumentError("rho is not representable strictly between zero and one in T"))
    rho = one(T) - rho_gap
    rho < one(T) ||
        throw(ArgumentError("rho rounds to one in T"))
    moment_count = 2d + 2
    c = zeros(T, moment_count)
    c[1] = bound === :lower ? one(T) : -one(T)
    blocks = Vector{Vector{SparseMatrixCSC{T,Int}}}()
    push!(blocks, _moment_block(T, d + 1, moment_count, :moment))
    push!(blocks, _moment_block(T, d, moment_count, :x))
    push!(blocks, _moment_block(T, d, moment_count, :one_minus_x))
    constants = SparseMatrixCSC{T,Int}[
        spzeros(T, d + 1, d + 1), spzeros(T, d, d), spzeros(T, d, d),
    ]
    recurrence_count = 2d + 1
    B = spzeros(T, moment_count, recurrence_count)
    rhs = zeros(T, recurrence_count)
    for k in 0:(recurrence_count - 1)
        B[k + 1, k + 1] = one(T)
        B[k + 2, k + 1] = -rho
        rhs[k + 1] = inv(T(k + 1))
    end
    p = SDPX.ingest(c, blocks, constants, B, rhs;
                    T=T, sparse=true, verbosity=0)
    # This equivalent form avoids evaluating log(rho_gap) through the rounded
    # value 1-rho and is stable for every registered precision ladder.
    exact = T(rho_power) * _log_t(T, T(2)) / (one(T) - rho_gap)
    isfinite(exact) || throw(ArgumentError("exact integral oracle is nonfinite"))
    moments = zeros(T, moment_count)
    moments[1] = exact
    for k in 0:(moment_count - 2)
        moments[k + 2] = (moments[k + 1] - inv(T(k + 1))) / rho
    end
    o = (
        objective=nothing,
        exact_integral=exact,
        primal=moments,
        dual=nothing,
        bound=bound,
        # The upper relaxation minimizes `-y₀` internally, but the benchmark
        # reports the physical moment y₀ so that both bounds are compared on
        # the same scale and in the same direction as the exact integral.
        physical_objective=x -> x[1],
        objective_kind=:bound,
        expected_status=:optimal,
    )
    parameters = (
        order=d,
        bound=bound,
        rho_power=rho_power,
        rho=rho,
        exact_integral=exact,
    )
    return _case(:rational_moment_sdp, tier, :sdp, parameters, p, o)
end

"""Exact objective oracle for a family without constructing the problem."""
function analytic_objective(family::Symbol; T::Type=Float64, kwargs...)
    tier = get(kwargs, :tier, :tier1)
    _check_tier(tier)
    if family === :chebyshev_lp
        n = Int(get(kwargs, :degree, _tier_value(tier, (8, 32, 128))))
        return T(2)^(1 - n)
    elseif family === :weighted_minimum_norm_socp
        n = Int(get(kwargs, :dimension, _tier_value(tier, (32, 512, 8192))))
        spread = _typed(T, get(kwargs, :spread, zero(T)))
        w = _normalized_weights(T, n, spread)
        return inv(sqrt(sum(inv(value^2) for value in w)))
    elseif family === :basel_soc_chain
        n = Int(get(kwargs, :terms, _tier_value(tier, (10, 1000, 10000))))
        return sqrt(sum(inv(T(k)^2) for k in 1:n))
    elseif family === :spectral_path_sdp
        n = Int(get(kwargs, :path_length, _tier_value(tier, (8, 64, 256))))
        return T(2) * _cos_t(T, T(pi) / T(n + 1))
    elseif family === :odd_cycle_maxcut_sdp
        n = Int(get(kwargs, :vertices, _tier_value(tier, (5, 31, 127))))
        return T(n) / T(2) * (one(T) + _cos_t(T, T(pi) / T(n)))
    elseif family === :rational_moment_sdp
        m = Int(get(kwargs, :m, get(kwargs, :rho_power, 4)))
        rho = one(T) - exp2(-m)
        return -log(one(T) - rho) / rho
    end
    throw(ArgumentError("unknown analytic family $(repr(family))"))
end

analytic_objective(case::AnalyticCase) =
    get(case.oracle, :exact_integral, oracle_objective(case))

analytic_primal(case::AnalyticCase) = oracle_primal(case)
analytic_dual(case::AnalyticCase) = oracle_dual(case)
analytic_primal(family::Symbol; kwargs...) = analytic_primal(build_problem(family; kwargs...))
analytic_dual(family::Symbol; kwargs...) = analytic_dual(build_problem(family; kwargs...))

function build_problem(
    family::Symbol;
    T::Type=Float64,
    tier::Symbol=:tier1,
    kwargs...,
)
    _check_tier(tier)
    SDPX.is_supported_arithmetic(T) || throw(ArgumentError(
        "unsupported arithmetic type $T",
    ))
    family === :chebyshev_lp && return _chebyshev_lp(T, tier; kwargs...)
    family === :weighted_minimum_norm_socp &&
        return _weighted_minimum_norm_socp(T, tier; kwargs...)
    family === :basel_soc_chain && return _basel_soc_chain(T, tier; kwargs...)
    family === :spectral_path_sdp && return _spectral_path_sdp(T, tier; kwargs...)
    family === :odd_cycle_maxcut_sdp &&
        return _odd_cycle_maxcut_sdp(T, tier; kwargs...)
    family === :rational_moment_sdp &&
        return _rational_moment_sdp(T, tier; kwargs...)
    throw(ArgumentError(
        "unknown analytic family $(repr(family)); choices=$(family_names())",
    ))
end

end # module
