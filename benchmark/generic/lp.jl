# Deterministic LPs: hand-checkable NETLIB-style rows and planted-KKT random LPs.

struct LPProblem <: AbstractGenericProblem end

function _lp_sum(values)
    expression = values[1]
    for index in 2:length(values)
        expression += values[index]
    end
    return expression
end

function build(::LPProblem, ::Type{T}, params) where {T<:AbstractFloat}
    kind = params.kind
    model = SDPX.Model(T; name="generic_$(params.name)")
    if kind === :afiro_style
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
        s = SDPX.variable!(model, :slack, 2; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :capacity_1, x[1] + x[2] + s[1] - T(4), SDPX.ZeroCone())
        SDPX.constraint!(model, :capacity_2, T(2) * x[1] + x[2] + s[2] - T(5), SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Maximize(), T(3) * x[1] + T(2) * x[2])
    elseif kind === :degenerate
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
        s = SDPX.variable!(model, :duplicate_slack, 2; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :duplicate_1, x[1] + x[2] + s[1] - one(T), SDPX.ZeroCone())
        SDPX.constraint!(model, :duplicate_2, x[1] + x[2] + s[2] - one(T), SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Maximize(), x[1] + x[2])
    elseif kind === :infeasible
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :impossible, x[1] + one(T), SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
    elseif kind === :unbounded
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
        SDPX.objective!(model, SDPX.Maximize(), x[1])
    elseif kind === :planted
        rng = Random.Xoshiro(params.seed)
        m, n = params.m, params.n
        A = randn(rng, T, m, n) / sqrt(T(m))
        xstar = zeros(T, n)
        xstar[1:m] .= T(0.5) .+ rand(rng, T, m)
        ystar = randn(rng, T, m)
        slack = zeros(T, n)
        slack[(m + 1):n] .= T(0.25) .+ rand(rng, T, n - m)
        b = A * xstar
        c = transpose(A) * ystar + slack
        x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
        for row in 1:m
            expression = A[row, 1] * x[1]
            for column in 2:n
                expression += A[row, column] * x[column]
            end
            SDPX.constraint!(model, Symbol(:row_, row), expression - b[row], SDPX.ZeroCone())
        end
        objective = c[1] * x[1]
        for column in 2:n
            objective += c[column] * x[column]
        end
        SDPX.objective!(model, SDPX.Minimize(), objective)
    else
        throw(ArgumentError("unknown LP benchmark kind $kind"))
    end
    return model
end

function _planted_lp_objective(seed, m, n)
    rng = Random.Xoshiro(seed)
    A = randn(rng, Float64, m, n) / sqrt(Float64(m))
    xstar = zeros(n)
    xstar[1:m] .= 0.5 .+ rand(rng, m)
    ystar = randn(rng, m)
    slack = zeros(n)
    slack[(m + 1):n] .= 0.25 .+ rand(rng, n - m)
    c = transpose(A) * ystar + slack
    return dot(c, xstar)
end

const _LP_SOURCE = "NETLIB LP/Data conventions plus seeded primal-dual planted standard-form LPs"
for (id, params, status, objective, tolerance) in (
    (:lp_afiro_style, (kind=:afiro_style, name=:afiro_style), :optimal, 9.0, 1e-7),
    (:lp_degenerate, (kind=:degenerate, name=:degenerate), :optimal, 1.0, 1e-7),
    (:lp_infeasible, (kind=:infeasible, name=:infeasible), :primal_infeasible, nothing, 1e-7),
    (:lp_unbounded, (kind=:unbounded, name=:unbounded), :dual_infeasible, nothing, 1e-7),
    (:lp_random_small, (kind=:planted, name=:random_small, seed=0x4c5001, m=3, n=8),
        :optimal, _planted_lp_objective(0x4c5001, 3, 8), 2e-6),
)
    _register!(BenchmarkSpec(id, :lp, :small, LPProblem(), params, status, objective, tolerance, _LP_SOURCE))
end
for (tier, seed, m, n, tol) in (
    (:medium, 0x4c5002, 20, 60, 2e-6),
    (:large, 0x4c5003, 400, 1200, 2e-5),
)
    params = (kind=:planted, name=Symbol(:random_, tier), seed, m, n)
    _register!(BenchmarkSpec(Symbol(:lp_random_, tier), :lp, tier, LPProblem(), params,
        :optimal, _planted_lp_objective(seed, m, n), tol, _LP_SOURCE))
end
