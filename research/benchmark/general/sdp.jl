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
    model = _benchmark_model(T,params)
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
    elseif params.kind === :blockdiag_trace
        # Four independent 3x3 PSD blocks represented in one 12x12 PSD
        # variable. Fixing all entries to the block-diagonal identity makes
        # the feasible matrix positive definite and gives trace optimum 12.
        block = Int(params.block)
        n == 3 * block || throw(ArgumentError("blockdiag dimensions disagree"))
        for row in 1:n
            SDPX.constraint!(model, Symbol(:blockdiag_diag_, row),
                X[row, row] - one(T), SDPX.ZeroCone())
            for column in 1:(row - 1)
                SDPX.constraint!(model, Symbol(:blockdiag_offdiag_, row, :_, column),
                    X[row, column], SDPX.ZeroCone())
            end
        end
        SDPX.objective!(model, SDPX.Minimize(), _trace_expression(X, n, T))
    elseif params.kind === :ill_scaled_trace
        diagonal = T.(params.diagonal)
        length(diagonal) == n || throw(ArgumentError("ill-scaled diagonal length mismatch"))
        all(isfinite, diagonal) && all(>(zero(T)), diagonal) ||
            throw(ArgumentError("ill-scaled diagonal must be positive and finite"))
        # Keep the feasible PSD point strictly positive definite (I) while
        # injecting a six-decade positive diagonal scaling into the linear
        # objective. All matrix entries are fixed, so the optimum is the
        # closed-form sum(diagonal), and certification is not boundary-based.
        for row in 1:n
            SDPX.constraint!(model, Symbol(:ill_scaled_diag_, row),
                X[row, row] - one(T), SDPX.ZeroCone())
            for column in 1:(row - 1)
                SDPX.constraint!(model, Symbol(:ill_scaled_offdiag_, row, :_, column),
                    X[row, column], SDPX.ZeroCone())
            end
        end
        objective = zero(T)
        for row in 1:n
            objective += diagonal[row] * X[row, row]
        end
        SDPX.objective!(model, SDPX.Minimize(), objective)
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
    (:psd_blockdiag_small,
        (kind=:blockdiag_trace, name=:psd_blockdiag_small, block=4, n=12),
        :optimal, 12.0, 2e-5),
    (:psd_dense_maxcut_k5,
        (kind=:maxcut_complete, name=:psd_dense_maxcut_k5, n=5),
        :optimal, 6.25, 3e-5),
    (:psd_ill_scaled_small,
        (kind=:ill_scaled_trace, name=:psd_ill_scaled_small, n=7,
         diagonal=(1.0e-6, 1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4, 1.0e6)),
        :optimal, 1010101.010101, 3e-5),
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
