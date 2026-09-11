# R5-A precision-controller targeted tests (quick qualification).
#
# Scope: the surviving planning shim in `src/kernels/mixed_precision_kkt.jl`
# (`_mixed_precision_storage_bytes`, `_mixed_precision_workspace_decision`).
# The dead factorization machinery and its pinned G1-G3 known-issue
# controls were removed with the source (the tests pinned UndefVarError
# branches, not behavior).
#
# Source is NOT modified by this file.

using Test
using SDPX
using LinearAlgebra

@testset "R5-A internal contract" begin
    @testset "storage-byte accounting matches the matrices counted" begin
        # _mixed_precision_storage_bytes(m, n): S (m x m) + Btil (m x n) +
        # Q (n x n) + r/dx (m) + p/dy (n) + equality_scale (n), all Float64.
        for (m, n) in ((1, 0), (3, 2), (4, 1), (16, 5), (300, 0))
            elements = m * m + m * n + n * n + 3m + 3n
            @test SDPX._mixed_precision_storage_bytes(m, n) ==
                  elements * sizeof(Float64)
        end
    end

    @testset "workspace decision refusal semantics" begin
        function _r5a_problem(::Type{T}, m::Int, n::Int) where {T}
            k1 = 2
            A = [zeros(T, m, k1, k1)]
            for i in 1:m
                A[1][i, 1, 1] = one(T)
            end
            C = [Matrix{T}(I, k1, k1)]
            B = ones(T, m, n)
            b = ones(T, n)
            c = ones(T, m)
            return SDPX.ingest(c, A, C, B, b; T=T, verbosity=0)
        end
        # Ordinary Float64 never takes the mixed route: refusal, not a silent
        # downgrade (required_bytes stays 0 so no budget is reserved).
        prob_f64 = _r5a_problem(Float64, 4, 1)
        decision_f64 = SDPX._mixed_precision_workspace_decision(
            prob_f64, :auto, 0.10,
        )
        @test decision_f64.enabled == false
        @test decision_f64.reason == :unsupported_arithmetic
        @test decision_f64.required_bytes == 0
        # Explicit :off refuses with :disabled.
        prob_big = setprecision(BigFloat, 256) do
            _r5a_problem(BigFloat, 4, 1)
        end
        decision_off = SDPX._mixed_precision_workspace_decision(
            prob_big, :off, 0.10,
        )
        @test decision_off.enabled == false
        @test decision_off.reason == :disabled
        # Automatic mode refuses small systems instead of silently engaging.
        decision_auto = SDPX._mixed_precision_workspace_decision(
            prob_big, :auto, 0.10,
        )
        @test decision_auto.enabled == false
        @test decision_auto.reason == :below_auto_dimension
        @test decision_auto.required_bytes ==
              SDPX._mixed_precision_storage_bytes(4, 1)
        # Expert :on engages a small well-formed system and the required-byte
        # figure equals the storage formula for exactly (m, n) = (4, 1).
        decision_on = SDPX._mixed_precision_workspace_decision(
            prob_big, :on, 0.10,
        )
        @test decision_on.enabled == true
        @test decision_on.reason == :selected
        @test decision_on.required_bytes == 36 * sizeof(Float64)
        # Memory-budget refusals are deterministic and independent of machine
        # free memory: unknown memory and an over-budget request both refuse.
        decision_unknown = SDPX._mixed_precision_workspace_decision(
            prob_big, :on, 0.10; available_memory_bytes=0,
        )
        @test decision_unknown.enabled == false
        @test decision_unknown.reason == :memory_unknown
        decision_budget = SDPX._mixed_precision_workspace_decision(
            prob_big, :on, 0.10; available_memory_bytes=10,
        )
        @test decision_budget.enabled == false
        @test decision_budget.reason == :memory_budget
    end
end
