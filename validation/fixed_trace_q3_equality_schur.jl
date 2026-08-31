using Test
using LinearAlgebra
using SDPX

function q3_reduction(::Type{T}) where {T}
    return SDPX.FixedTraceQ3Reduction(
        [1 3; 2 4],
        reshape(T[1, 0, 0, 1, 1, 0, 0, 1], 2, 2, 2),
        T[1, 1], zeros(T, 2, 2), Int[], :owned,
    )
end

function explicit_metric(metric::AbstractMatrix{T}) where {T}
    H = zeros(T, 4, 4)
    H[1:2, 1:2] .= T[metric[1,1] metric[2,1]; metric[2,1] metric[3,1]]
    H[3:4, 3:4] .= T[metric[1,2] metric[2,2]; metric[2,2] metric[3,2]]
    return H
end

function check_q3_equality_schur(::Type{T}) where {T}
    reduction = q3_reduction(T)
    panel = T[1 1//5 3//10 -1//10; 2//5 1 -1//5 1//2]
    panel_snapshot = deepcopy(panel)
    workspace = SDPX.FixedTraceQ3EqualitySchurWorkspace(reduction, panel)
    metric = T[4 3; 1//2 -1//4; 5 2]
    @test SDPX.prepare_fixed_trace_q3_equality_schur!(workspace, metric)
    H = explicit_metric(metric)
    explicit_schur = panel * (H \ transpose(panel))
    tolerance = T(2048) * eps(T) * max(one(T), maximum(abs, explicit_schur))
    @test maximum(abs, workspace.schur - explicit_schur) <= tolerance
    @test workspace.schur == transpose(workspace.schur)

    source = T[2//3, -1//4]
    action = zeros(T, 2)
    SDPX.fixed_trace_q3_equality_schur_action!(action, workspace, source)
    @test maximum(abs, action - explicit_schur * source) <= tolerance

    local_rhs = T[1, -2, 3//2, 1//3]
    equality_rhs = T[1//5, -2//7]
    schur_rhs = zeros(T, 2)
    SDPX.fixed_trace_q3_equality_rhs!(
        schur_rhs, workspace, local_rhs, equality_rhs,
    )
    equality_solution = workspace.schur \ schur_rhs
    local_solution = zeros(T, 4)
    SDPX.recover_fixed_trace_q3_local_direction!(
        local_solution, workspace, local_rhs, equality_solution,
    )
    direct = [H transpose(panel); panel zeros(T, 2, 2)] \
             vcat(local_rhs, equality_rhs)
    direct_local, direct_equality = direct[1:4], direct[5:6]
    scale = max(one(T), maximum(abs, direct))
    @test maximum(abs, local_solution - direct_local) <= T(4096) * eps(T) * scale
    @test maximum(abs, equality_solution - direct_equality) <= T(4096) * eps(T) * scale
    @test maximum(abs, H * local_solution + transpose(panel) * equality_solution - local_rhs) <=
          T(4096) * eps(T) * scale
    @test maximum(abs, panel * local_solution - equality_rhs) <=
          T(4096) * eps(T) * scale

    panel .= T(99)
    @test workspace.panel == panel_snapshot
    @test_throws DimensionMismatch SDPX.FixedTraceQ3EqualitySchurWorkspace(
        reduction, zeros(T, 2, 3),
    )
    bad = copy(panel_snapshot); bad[1,1] = T(Inf)
    @test_throws ArgumentError SDPX.FixedTraceQ3EqualitySchurWorkspace(reduction, bad)
end

@testset "fixed-trace Q3 equality Schur" begin
    check_q3_equality_schur(Float64)
    setprecision(BigFloat, 256) do
        check_q3_equality_schur(BigFloat)
    end
end
