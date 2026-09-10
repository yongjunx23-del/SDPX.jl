# PR-02: the SOC rank-2 mapping must match the operator SDPX actually applies.
#
# The plan's own warning is the reason this test exists: "must first establish
# the correspondence between SDPX apply_Theta! and this H; do not take a
# similarly-named `w` from the SDPX NT state and apply the formula directly."
#
# Clarabel's `(D, u, v)` formulas are stated for `Theta = Q_w = 2*w*w' - J` with
# `w` normalized so that `w0^2 - ||w_tail||^2 == 1`. SDPX's `theta_apply!`
# applies `quadratic_apply!`, whose (1,1) entry is `w0^2 + ||w_tail||^2` instead
# of `2*w0^2 - 1`. The two operators differ only there, but that is enough to
# make Clarabel's parameters wrong for SDPX by ~1e-1 -- not a rounding issue.
#
# This test pins the *measured* correspondence rather than the assumed one, and
# would fail loudly if either the scaling builder or the metric mapping drifted.
using Test
using LinearAlgebra
using SDPX

const SC = SDPX.SymmetricCones

# A strictly interior SOC pair: the leading entry strictly dominates the tail
# norm, which is what `nt_scaling!` requires.
function _soc_interior_pair(k::Int, seed::Int; scale::Float64=1.0)
    s = zeros(Float64, k)
    y = zeros(Float64, k)
    tail_s = [0.3 * sin(Float64(i * seed)) for i in 2:k]
    tail_y = [0.2 * cos(Float64(i * seed)) for i in 2:k]
    s[2:end] .= tail_s
    y[2:end] .= tail_y
    s[1] = 1.5 * norm(tail_s) + 1.0
    y[1] = 1.5 * norm(tail_y) + 1.0
    return s .* scale, y .* scale
end

"""The metric `theta_apply!` applies, materialized column by column."""
function _theta_matrix(cone, state, k::Int)
    theta = zeros(Float64, k, k)
    for j in 1:k
        e = zeros(Float64, k)
        e[j] = 1.0
        SC.theta_apply!(cone, view(theta, :, j), state, e)
    end
    return theta
end

@testset "SOC rank-2 mapping matches the production metric" begin
    @testset "Clarabel parameters do NOT fit SDPX's metric" begin
        # This is the negative control for the plan's warning. If someone
        # "fixes" the mapping back to Clarabel's formulas, this test documents
        # why that is wrong rather than letting it pass silently.
        k = 8
        cone = SC.SOCone(k)
        state = SC.SOCNTScaling{Float64}(k)
        s, y = _soc_interior_pair(k, 2)
        SC.nt_scaling!(cone, state, s, y)
        w = state.w
        theta = _theta_matrix(cone, state, k)

        # Clarabel's normalization does not hold for an SDPX scaling point.
        w0sq_minus_tail = w[1]^2 - sum(abs2, w[2:end])
        @test !isapprox(w0sq_minus_tail, 1.0; atol=1e-8)

        # And Clarabel's parameters therefore miss the metric by a lot.
        J = Matrix(Diagonal([1.0; fill(-1.0, k - 1)]))
        Qw = 2 * w * transpose(w) - J
        @test maximum(abs, theta - Qw) > 1e-2
    end

    @testset "derived mapping reproduces Theta to working precision" begin
        worst = 0.0
        worst_apply = 0.0
        for k in (2, 3, 5, 8, 16, 32, 64, 128, 512), seed in (1, 2, 7)
            cone = SC.SOCone(k)
            state = SC.SOCNTScaling{Float64}(k)
            s, y = _soc_interior_pair(k, seed)
            SC.nt_scaling!(cone, state, s, y)
            w = state.w

            D, u, v = SC.soc_rank2_parameters(w)
            @test length(D) == length(u) == length(v) == k
            # v must have a structurally zero leading entry: that is what keeps
            # the first column free of a rank-2 correction.
            @test v[1] == 0.0
            @test all(isfinite, D) && all(isfinite, u) && all(isfinite, v)

            theta = _theta_matrix(cone, state, k)
            rebuilt = Matrix(Diagonal(D)) + u * transpose(u) - v * transpose(v)
            metric_error = maximum(abs, theta - rebuilt)
            worst = max(worst, metric_error)
            @test metric_error <= 1e-12 * max(1.0, maximum(abs, theta))

            # The action form must agree too, not only the assembled matrix.
            x = [cos(Float64(i)) for i in 1:k]
            applied = zeros(Float64, k)
            SC.theta_apply!(cone, applied, state, x)
            expanded = D .* x .+ u .* dot(u, x) .- v .* dot(v, x)
            apply_error = maximum(abs, applied - expanded)
            worst_apply = max(worst_apply, apply_error)
            @test apply_error <= 1e-12 * max(1.0, maximum(abs, applied))
        end
        @test worst <= 1e-12
        @test worst_apply <= 1e-12
    end

    @testset "the metric stays a valid cone operator" begin
        for k in (3, 8, 32, 128)
            cone = SC.SOCone(k)
            state = SC.SOCNTScaling{Float64}(k)
            s, y = _soc_interior_pair(k, 3)
            SC.nt_scaling!(cone, state, s, y)
            D, u, v = SC.soc_rank2_parameters(state.w)
            # The tail diagonal must be positive; it is the strict-interiority
            # condition, and a non-positive value must have thrown instead.
            @test all(>(0.0), D)
        end
    end

    @testset "degenerate scaling points fail closed" begin
        @test_throws ArgumentError SC.soc_rank2_parameters([1.0])
        @test_throws ArgumentError SC.soc_rank2_parameters([NaN, 0.0, 0.0])
        @test_throws ArgumentError SC.soc_rank2_parameters([Inf, 0.0])
        # beta <= 0 means the point is not strictly interior: refuse it rather
        # than build an operator that is not a cone metric.
        @test_throws ArgumentError SC.soc_rank2_parameters([1.0, 1.0, 0.0])
        @test_throws ArgumentError SC.soc_rank2_parameters([1.0, 2.0])
    end

    @testset "storage preference uses the measured crossover k = 6" begin
        @test !SC.soc_rank2_prefer_expanded(2)
        @test !SC.soc_rank2_prefer_expanded(4)
        @test !SC.soc_rank2_prefer_expanded(5)
        @test SC.soc_rank2_prefer_expanded(6)
        @test SC.soc_rank2_prefer_expanded(4096)
        @test SC.soc_rank2_cone_slots(4096) == 12290
        # Exact agreement with the packed dense lower triangle.
        for k in 2:64
            packed = div(k * (k + 1), 2)
            @test SC.soc_rank2_prefer_expanded(k) == (3 * k + 2 < packed)
        end
    end
end
