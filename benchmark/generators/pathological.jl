"""
Typed, JuMP-free pathological conic benchmark builders.

The builders in this file are deliberately small analytic constructions.  They
are used for correctness and precision ladders, rather than as replacements
for the public Netlib/SDPLIB/DIMACS inputs.  Every decimal parameter enters
the requested arithmetic type with `parse(T, text)`; in particular, no
Float64 temporary is used when a BigFloat or MultiFloat problem is built.

Each builder returns a named tuple containing at least
`(problem, expected, kind, solve_settings)`.  The additional fields are
metadata consumed by the registry/tests and are not required by SDPX itself.
"""

using LinearAlgebra
using SparseArrays
using SDPX

const PATHOLOGICAL_CASES = (
    :lp_infeasible_margin,
    :lp_degenerate_scaled,
    :socp_near_tangent,
    :socp_near_infeasible,
    :socp_many_tiny_cones,
    :sdp_small_eigenvalue,
    :sdp_congruence_scaling,
    :sdp_infeasible_margin,
)

pathological_cases() = PATHOLOGICAL_CASES

"Parse a decimal directly in the benchmark arithmetic type."
@inline _path_decimal(::Type{T}, value::T) where {T} = value
@inline _path_decimal(::Type{T}, value::AbstractString) where {T} =
    parse(T, value)
@inline _path_decimal(::Type{T}, value::Real) where {T} = T(value)

function _path_solve_settings(
    ::Type{T};
    tolerance=nothing,
    maximum_iterations::Integer=200,
    specialization::Symbol=:auto,
) where {T}
    default_tolerance = T === Float64 ? "1e-8" : "1e-20"
    τ = tolerance === nothing ? default_tolerance : string(tolerance)
    return (
        tolerance=τ,
        maximum_iterations=maximum_iterations,
        specialization=specialization,
    )
end

function _path_result(
    problem,
    expected,
    kind::Symbol,
    case::Symbol,
    status::Symbol;
    solve_settings,
    kwargs...,
)
    return (
        problem=problem,
        expected=expected,
        kind=kind,
        solve_settings=solve_settings,
        case=case,
        expected_status=status,
        source_parameters=(; kwargs...),
    )
end

"""
    _path_lp_infeasible_margin(T; epsilon="1e-8")

The scalar LP has `x >= 0` and `-x >= epsilon`, hence is infeasible for every
positive epsilon.  The margin is represented as a typed scalar, so decreasing
it below Float64's resolution is meaningful in a higher precision run.
"""
function _path_lp_infeasible_margin(
    ::Type{T};
    epsilon="1e-8",
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    eps = _path_decimal(T, epsilon)
    eps > zero(T) || throw(ArgumentError("epsilon must be positive"))
    problem = SDPX.linear_program(
        T[zero(T)],
        reshape(T[one(T), -one(T)], 2, 1),
        T[zero(T), eps];
        T=T,
        sparse=true,
        verbosity=0,
    )
    return _path_result(
        problem,
        nothing,
        :sdp,
        :lp_infeasible_margin,
        :primal_infeasible;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        epsilon=eps,
    )
end

"""
    _path_lp_degenerate_scaled(T; n=8, decades=8)

Minimize `sum(x_i)` subject to two positively scaled copies of `x_i >= 1`
for every variable.  The optimizer is exactly `x*=1`, while the active
duplicate rows make the problem degenerate and the row scales span
`10^(2*decades)`.  This is intentionally a compact, analytically certified
Klee--Minty-style scaling/degeneracy stress case; it does not claim a
simplex-specific iteration count.
"""
function _path_lp_degenerate_scaled(
    ::Type{T};
    n::Integer=8,
    decades::Integer=8,
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    n > 0 || throw(ArgumentError("n must be positive"))
    decades >= 0 || throw(ArgumentError("decades must be nonnegative"))
    variables = Int(n)
    rows = 2 * variables
    G = zeros(T, rows, variables)
    h = zeros(T, rows)
    scales = Vector{T}(undef, variables)
    for i in 1:variables
        # Symmetric exponents give a broad but deterministic dynamic range.
        exponent = variables == 1 ? 0 : round(Int,
            -decades + 2 * decades * (i - 1) / (variables - 1))
        scale = _path_decimal(T, "1e$(exponent)")
        scales[i] = scale
        first_row = 2 * i - 1
        second_row = 2 * i
        G[first_row, i] = scale
        G[second_row, i] = scale + scale
        h[first_row] = scale
        h[second_row] = scale + scale
    end
    objective = ones(T, variables)
    problem = SDPX.linear_program(
        objective,
        G,
        h;
        T=T,
        sparse=true,
        verbosity=0,
    )
    return _path_result(
        problem,
        T(variables),
        :sdp,
        :lp_degenerate_scaled,
        :optimal;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        n=variables,
        decades=decades,
        scales=Tuple(scales),
    )
end

"Build `min s` subject to `(1+s, 1, epsilon) ∈ Q₃`."
function _path_socp_near_tangent(
    ::Type{T};
    epsilon="1e-4",
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    eps = _path_decimal(T, epsilon)
    eps >= zero(T) || throw(ArgumentError("epsilon must be nonnegative"))
    A = zeros(T, 3, 1)
    A[1, 1] = one(T)
    b = T[one(T), one(T), eps]
    cone = SDPX.SOCConstraint(A, b; T=T)
    problem = SDPX.second_order_program(
        T[one(T)],
        [cone];
        T=T,
    )
    expected = sqrt(one(T) + eps * eps) - one(T)
    return _path_result(
        problem,
        expected,
        :socp,
        :socp_near_tangent,
        :optimal;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        epsilon=eps,
    )
end

"Build an SOC constraint that is infeasible by a typed margin epsilon."
function _path_socp_near_infeasible(
    ::Type{T};
    epsilon="1e-4",
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    eps = _path_decimal(T, epsilon)
    zero(T) < eps < one(T) || throw(ArgumentError(
        "epsilon must lie in (0,1)",
    ))
    A = zeros(T, 3, 1)
    A[1, 1] = one(T)
    cone = SDPX.SOCConstraint(
        A,
        T[one(T) - eps, one(T), zero(T)];
        T=T,
    )
    problem = SDPX.second_order_program(
        T[zero(T)],
        [cone];
        Aeq=reshape(T[one(T)], 1, 1),
        beq=T[zero(T)],
        T=T,
    )
    return _path_result(
        problem,
        nothing,
        :socp,
        :socp_near_infeasible,
        :primal_infeasible;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        epsilon=eps,
    )
end

"""
    _path_socp_many_tiny_cones(T; ncones=100, epsilon="1e-4")

Separable SOCs with one scalar variable per cone.  Cone offsets are typed and
the coefficient matrices are sparse, exposing cone-count assembly and memory
scaling without introducing a modelling-layer dependency.
"""
function _path_socp_many_tiny_cones(
    ::Type{T};
    ncones::Integer=100,
    epsilon="1e-4",
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    ncones > 0 || throw(ArgumentError("ncones must be positive"))
    eps = _path_decimal(T, epsilon)
    eps >= zero(T) || throw(ArgumentError("epsilon must be nonnegative"))
    cones = SDPX.SOCConstraint{T}[]
    sizehint!(cones, ncones)
    for i in 1:ncones
        A = spzeros(T, 3, ncones)
        A[1, i] = one(T)
        push!(cones, SDPX.SOCConstraint(
            A,
            T[zero(T), one(T), eps];
            T=T,
        ))
    end
    problem = SDPX.second_order_program(
        ones(T, ncones),
        cones;
        T=T,
    )
    expected = T(ncones) * sqrt(one(T) + eps * eps)
    return _path_result(
        problem,
        expected,
        :socp,
        :socp_many_tiny_cones,
        :optimal;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        ncones=Int(ncones),
        epsilon=eps,
    )
end

"""
    _path_sdp_small_eigenvalue(T; dimension=4, epsilon="1e-8")

The PSD constraint is `diag(1,…,epsilon) - tI ⪰ 0`; minimizing `-t` gives the
exact objective `-epsilon`.  Keeping the matrix diagonal makes the oracle
independent of a target-arithmetic eigensolver while still exercising a small
eigenvalue, rank detection, and complementarity.
"""
function _path_sdp_small_eigenvalue(
    ::Type{T};
    dimension::Integer=4,
    epsilon="1e-8",
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    dimension >= 2 || throw(ArgumentError("dimension must be at least 2"))
    eps = _path_decimal(T, epsilon)
    zero(T) < eps <= one(T) || throw(ArgumentError(
        "epsilon must lie in (0,1]",
    ))
    diagonal = ones(T, dimension)
    diagonal[end] = eps
    identity = Matrix{T}(I, dimension, dimension)
    coefficients = zeros(T, 1, dimension, dimension)
    coefficients[1, :, :] .= -identity
    C = [-Matrix(Diagonal(diagonal))]
    problem = SDPX.ingest(
        T[-one(T)],
        [coefficients],
        C,
        spzeros(T, 1, 0),
        T[];
        T=T,
        sparse=false,
        symmetrize=false,
        verbosity=0,
    )
    return _path_result(
        problem,
        -eps,
        :sdp,
        :sdp_small_eigenvalue,
        :optimal;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        dimension=Int(dimension),
        epsilon=eps,
    )
end

"""
    _path_sdp_congruence_scaling(T; decades=4)

The matrix `[[x,s],[s,s²]]` is PSD iff `x >= 1` for positive `s`.  It is
encoded as `A₁*x - C` with `A₁=diag(1,0)` and
`C=[[0,-s],[-s,-s²]]`, preserving the objective while spanning a large
congruence scale.
"""
function _path_sdp_congruence_scaling(
    ::Type{T};
    decades::Integer=4,
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    decades >= 0 || throw(ArgumentError("decades must be nonnegative"))
    scale = _path_decimal(T, "1e$(decades)")
    coefficient = zeros(T, 2, 2)
    coefficient[1, 1] = one(T)
    constant = zeros(T, 2, 2)
    constant[2, 2] = -(scale * scale)
    constant[1, 2] = -scale
    constant[2, 1] = -scale
    problem = SDPX.ingest(
        T[one(T)],
        [reshape(coefficient, 1, 2, 2)],
        [constant],
        spzeros(T, 1, 0),
        T[];
        T=T,
        sparse=false,
        symmetrize=false,
        verbosity=0,
    )
    return _path_result(
        problem,
        one(T),
        :sdp,
        :sdp_congruence_scaling,
        :optimal;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        decades=Int(decades),
        scale=scale,
    )
end

"""
    _path_sdp_infeasible_margin(T; epsilon="1e-8")

The affine PSD matrix is `diag(x,-epsilon)`, which is infeasible for every
positive epsilon.  This is a strict infeasibility construction (not a claim
about weak infeasibility) and is suitable for status/certificate smoke tests.
"""
function _path_sdp_infeasible_margin(
    ::Type{T};
    epsilon="1e-8",
    tolerance=nothing,
    maximum_iterations::Integer=200,
) where {T}
    eps = _path_decimal(T, epsilon)
    eps > zero(T) || throw(ArgumentError("epsilon must be positive"))
    coefficient = zeros(T, 2, 2)
    coefficient[1, 1] = one(T)
    constant = zeros(T, 2, 2)
    constant[2, 2] = eps
    problem = SDPX.ingest(
        T[zero(T)],
        [reshape(coefficient, 1, 2, 2)],
        [constant],
        spzeros(T, 1, 0),
        T[];
        T=T,
        sparse=false,
        symmetrize=false,
        verbosity=0,
    )
    return _path_result(
        problem,
        nothing,
        :sdp,
        :sdp_infeasible_margin,
        :primal_infeasible;
        solve_settings=_path_solve_settings(
            T;
            tolerance,
            maximum_iterations,
        ),
        epsilon=eps,
    )
end

function build_pathological_problem(name::Symbol, ::Type{T}; kwargs...) where {T}
    # Registry loaders use a prefix so they cannot collide with the older
    # synthetic `:sdp_small_eigenvalue` loader.  Standalone callers may use
    # either the prefixed or the short case name.
    name = startswith(String(name), "pathological_") ?
           Symbol(replace(String(name), "pathological_" => ""; count=1)) : name
    name === :lp_infeasible_margin &&
        return _path_lp_infeasible_margin(T; kwargs...)
    name === :lp_degenerate_scaled &&
        return _path_lp_degenerate_scaled(T; kwargs...)
    name === :socp_near_tangent &&
        return _path_socp_near_tangent(T; kwargs...)
    name === :socp_near_infeasible &&
        return _path_socp_near_infeasible(T; kwargs...)
    name === :socp_many_tiny_cones &&
        return _path_socp_many_tiny_cones(T; kwargs...)
    name === :sdp_small_eigenvalue &&
        return _path_sdp_small_eigenvalue(T; kwargs...)
    name === :sdp_congruence_scaling &&
        return _path_sdp_congruence_scaling(T; kwargs...)
    name === :sdp_infeasible_margin &&
        return _path_sdp_infeasible_margin(T; kwargs...)
    throw(ArgumentError(
        "unknown pathological benchmark $name; choices=$(collect(PATHOLOGICAL_CASES))",
    ))
end

# A descriptive alias used by standalone campaign scripts.
build_pathological_case(name::Symbol, ::Type{T}; kwargs...) where {T} =
    build_pathological_problem(name, T; kwargs...)
