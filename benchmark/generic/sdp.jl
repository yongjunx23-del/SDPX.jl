# SDPLIB-style graph SDPs and seeded spectral SDPs with exact eigenvalue oracles.

struct SDPProblem <: AbstractGenericProblem end

function _trace_expression(X, n, ::Type{T}) where {T}
    expression = zero(T)
    for index in 1:n
        expression += X[index, index]
    end
    return expression
end

function _spectral_values(seed, n)
    rng = Random.Xoshiro(seed)
    values = 0.5 .+ 1.5 .* rand(rng, n)
    # A unique minimum makes the exact optimum a rank-one boundary point.
    values[1] = 0.125
    return values
end

function build(::SDPProblem, ::Type{T}, params) where {T<:AbstractFloat}
    n = params.n
    model = SDPX.Model(T; name="generic_$(params.name)")
    X = SDPX.variable!(model, :X, n, n; domain=SDPX.PSDCone())
    if params.kind === :spectral
        lambda = T.(_spectral_values(params.seed, n))
        SDPX.constraint!(model, :trace_normalization,
            _trace_expression(X, n, T) - one(T), SDPX.ZeroCone())
        objective = zero(T)
        for index in 1:n
            objective += lambda[index] * X[index, index]
        end
        SDPX.objective!(model, SDPX.Minimize(), objective)
    elseif params.kind === :theta_cycle
        SDPX.constraint!(model, :trace_normalization,
            _trace_expression(X, n, T) - one(T), SDPX.ZeroCone())
        for vertex in 1:n
            neighbor = vertex == n ? 1 : vertex + 1
            SDPX.constraint!(model, Symbol(:edge_, vertex), X[vertex, neighbor], SDPX.ZeroCone())
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
for (id, params, objective, tolerance) in (
    (:sdp_theta_c5, (kind=:theta_cycle, name=:sdp_theta_c5, n=5), sqrt(5.0), 2e-5),
    (:sdp_maxcut_k4, (kind=:maxcut_complete, name=:sdp_maxcut_k4, n=4), 4.0, 2e-5),
    (:sdp_rank1_boundary,
        (kind=:spectral, name=:sdp_rank1_boundary, seed=0x5d0001, n=2), 0.125, 2e-6),
    (:sdp_random_small,
        (kind=:spectral, name=:sdp_random_small, seed=0x5d0002, n=4), 0.125, 2e-6),
)
    _register!(BenchmarkSpec(id, :sdp, :small, SDPProblem(), params,
        :optimal, objective, tolerance, _SDP_SOURCE))
end
for (tier, seed, n, tol) in (
    (:medium, 0x5d0003, 14, 5e-6),
    (:large, 0x5d0004, 100, 5e-5),
)
    params = (kind=:spectral, name=Symbol(:sdp_random_, tier), seed, n)
    _register!(BenchmarkSpec(Symbol(:sdp_random_, tier), :sdp, tier,
        SDPProblem(), params, :optimal, 0.125, tol, _SDP_SOURCE))
end
