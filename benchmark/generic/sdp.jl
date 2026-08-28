# SDPLIB-style graph SDPs and seeded spectral SDPs with exact eigenvalue oracles.

struct SDPProblem <: AbstractGenericProblem end

function _trace_expression(X, n, ::Type{T}) where {T}
    expression = zero(T)
    for index in 1:n
        expression += X[index, index]
    end
    return expression
end

function _random_diagonal(seed, n)
    rng = Random.Xoshiro(seed)
    values = 0.5 .+ rand(rng, n)
    return values ./ sum(values)
end

function build(::SDPProblem, ::Type{T}, params) where {T<:AbstractFloat}
    n = params.n
    model = SDPX.Model(T; name="generic_$(params.name)")
    X = SDPX.variable!(model, :X, n, n; domain=SDPX.PSDCone())
    if params.kind === :weighted_trace
        diagonal = T.(_random_diagonal(params.seed, n))
        for index in 1:n
            SDPX.constraint!(model, Symbol(:random_diagonal_, index),
                X[index, index] - diagonal[index], SDPX.ZeroCone())
        end
        trace_X = _trace_expression(X, n, T)
        # Every feasible point is optimal. Both diag(diagonal) (interior) and
        # vv' with vᵢ=sqrt(diagonalᵢ) (rank-one boundary) are feasible.
        SDPX.objective!(model, SDPX.Minimize(), trace_X)
    elseif params.kind === :theta_complete
        for index in 1:n
            SDPX.constraint!(model, Symbol(:theta_diagonal_, index),
                X[index, index] - inv(T(n)), SDPX.ZeroCone())
        end
        edge = 0
        for row in 2:n, column in 1:(row - 1)
            edge += 1
            SDPX.constraint!(model, Symbol(:edge_, edge), X[row, column], SDPX.ZeroCone())
        end
        objective = zero(T)
        for row in 1:n
            objective += X[row, row]
            for column in 1:(row - 1)
                objective += T(2) * X[row, column]
            end
        end
        SDPX.objective!(model, SDPX.Maximize(), objective)
    elseif params.kind === :maxcut_complete
        for index in 1:n
            SDPX.constraint!(model, Symbol(:unit_diagonal_, index),
                X[index, index] - one(T), SDPX.ZeroCone())
        end
        objective = zero(T)
        quarter = inv(T(4))
        for row in 2:n, column in 1:(row - 1)
            objective += quarter * (X[row, row] + X[column, column] - T(2) * X[row, column])
        end
        SDPX.objective!(model, SDPX.Maximize(), objective)
    else
        throw(ArgumentError("unknown SDP benchmark kind $(params.kind)"))
    end
    return model
end

const _SDP_SOURCE = "SDPLIB graph relaxations: Lovasz theta and Goemans-Williamson Max-Cut; seeded trace SDP"
for (id, params, status, objective, tolerance) in (
    (:sdp_theta_k4, (kind=:theta_complete, name=:sdp_theta_k4, n=4),
        :known_solver_finding, 1.0, 2e-6),
    (:sdp_maxcut_k4, (kind=:maxcut_complete, name=:sdp_maxcut_k4, n=4),
        :optimal, 4.0, 2e-5),
    (:sdp_rank1_boundary,
        (kind=:weighted_trace, name=:sdp_rank1_boundary, seed=0x5d0001, n=2),
        :known_solver_finding, 1.0, 2e-6),
    (:sdp_random_small,
        (kind=:weighted_trace, name=:sdp_random_small, seed=0x5d0002, n=4),
        :known_solver_finding, 1.0, 2e-6),
)
    _register!(BenchmarkSpec(id, :sdp, :small, SDPProblem(), params,
        status, objective, tolerance, _SDP_SOURCE))
end
for (tier, seed, n, tol) in (
    (:medium, 0x5d0003, 14, 5e-6),
    (:large, 0x5d0004, 100, 5e-5),
)
    params = (kind=:weighted_trace, name=Symbol(:sdp_random_, tier), seed, n)
    _register!(BenchmarkSpec(Symbol(:sdp_random_, tier), :sdp, tier,
        SDPProblem(), params, :known_solver_finding, 1.0, tol, _SDP_SOURCE))
end
