# Independent dense-oracle tests for sparse three-dimensional nonsymmetric
# Schur assembly.  This file is intentionally standalone until the Phase-4
# asymmetric product-HSD integration gate is owned by the main test registry.

using LinearAlgebra
using SDPX
using SparseArrays
using Test

if !isdefined(SDPX, :NonsymmetricSchur3Workspace)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "hsd", "nonsymmetric_schur3.jl",
        ),
    )
end

const _NS3_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

function _ns3_metric!(metrics, block, values)
    @inbounds for column in 1:3, row in 1:3
        metrics[row, column, block] = values[row, column]
    end
    return metrics
end

function _ns3_fixture(::Type{T}) where {T<:AbstractFloat}
    rows = 11
    columns = 5
    dense_a = zeros(T, rows, columns)
    # Selected nonsymmetric blocks are deliberately unordered and separated by
    # rows owned by hypothetical symmetric blocks.
    dense_a[2, 1] = T(1) / T(2)
    dense_a[2, 4] = -T(1) / T(8)
    dense_a[3, 2] = T(3) / T(4)
    dense_a[3, 5] = T(1) / T(16)
    dense_a[4, 1] = -T(1) / T(4)
    dense_a[4, 3] = T(5) / T(8)
    dense_a[8, 2] = -T(3) / T(8)
    dense_a[8, 4] = T(7) / T(16)
    dense_a[9, 1] = T(1) / T(8)
    dense_a[9, 3] = -T(1) / T(2)
    dense_a[9, 5] = T(3) / T(16)
    dense_a[10, 2] = T(5) / T(16)
    dense_a[10, 5] = -T(1) / T(4)
    # These gap rows must not affect this partial assembler.
    dense_a[1, 1] = T(9)
    dense_a[5, 2] = -T(7)
    dense_a[7, 4] = T(11)
    dense_a[11, 5] = -T(13)
    A = sparse(dense_a)
    offsets = [8, 2]

    metrics = zeros(T, 3, 3, 2)
    _ns3_metric!(metrics, 1, T[
        T(2) T(1)/T(4) -T(1)/T(8)
        T(1)/T(4) T(3)/T(2) T(1)/T(16)
        -T(1)/T(8) T(1)/T(16) T(5)/T(4)
    ])
    _ns3_metric!(metrics, 2, T[
        T(3)/T(2) -T(1)/T(8) T(1)/T(4)
        -T(1)/T(8) T(7)/T(4) -T(1)/T(16)
        T(1)/T(4) -T(1)/T(16) T(9)/T(8)
    ])
    b = T[
        T(17), T(1)/T(2), -T(3)/T(4), T(5)/T(8), -T(19),
        T(23), -T(29), T(7)/T(16), T(3)/T(8), -T(1)/T(4), T(31),
    ]
    rhs = T[
        -T(37), -T(1)/T(8), T(5)/T(16), T(3)/T(4), T(41),
        -T(43), T(47), -T(3)/T(8), T(1)/T(4), T(7)/T(16), -T(53),
    ]
    workspace = SDPX.NonsymmetricSchur3Workspace(A, offsets)
    H = zeros(T, columns, columns)
    at_g_b = zeros(T, columns)
    bt_g_a = zeros(T, columns)
    at_g_rhs = zeros(T, columns)
    return (;
        A, dense_a, offsets, metrics, b, rhs, workspace, H, at_g_b,
        bt_g_a, at_g_rhs,
    )
end

# Independent reference: only this test helper materialises the global dense
# block-diagonal G.  It neither calls nor shares row/dyadic production code.
function _ns3_dense_oracle(A_dense, offsets, metrics, b, rhs)
    T = eltype(A_dense)
    rows, _ = size(A_dense)
    global_g = zeros(T, rows, rows)
    @inbounds for block in eachindex(offsets)
        offset = offsets[block]
        for column in 1:3, row in 1:3
            global_g[offset + row - 1, offset + column - 1] =
                metrics[row, column, block]
        end
    end
    ga = global_g * A_dense
    gb = global_g * b
    grhs = global_g * rhs
    return (
        H=transpose(A_dense) * ga,
        at_g_b=transpose(A_dense) * gb,
        bt_g_a=vec(transpose(b) * ga),
        b_g_b=dot(b, gb),
        at_g_rhs=transpose(A_dense) * grhs,
        b_g_rhs=dot(b, grhs),
    )
end

function _ns3_theta_dense_oracle(A_dense, offsets, thetas, b, rhs)
    T = eltype(A_dense)
    blocks = length(offsets)
    metrics = zeros(T, 3, 3, blocks)
    @inbounds for block in 1:blocks
        theta = Matrix{T}(undef, 3, 3)
        for column in 1:3, row in 1:3
            theta[row, column] = thetas[row, column, block]
        end
        metric = inv(Symmetric(theta))
        for column in 1:3, row in 1:3
            metrics[row, column, block] = metric[row, column]
        end
    end
    return _ns3_dense_oracle(A_dense, offsets, metrics, b, rhs)
end

function _ns3_effective_dense(A::SparseMatrixCSC{T}) where {T}
    dense = zeros(T, size(A))
    @inbounds for column in axes(A, 2)
        for pointer in nzrange(A, column)
            dense[A.rowval[pointer], column] += A.nzval[pointer]
        end
    end
    return dense
end

@inline function _ns3_tolerance(::Type{T}) where {T}
    return T(262144) * eps(one(T))
end

function _ns3_check_fixture(::Type{T}) where {T<:AbstractFloat}
    fixture = _ns3_fixture(T)
    result = SDPX.try_assemble_nonsymmetric_schur3!(
        fixture.workspace,
        fixture.H,
        fixture.at_g_b,
        fixture.bt_g_a,
        fixture.at_g_rhs,
        fixture.metrics,
        fixture.b,
        fixture.rhs,
    )
    oracle = _ns3_dense_oracle(
        fixture.dense_a, fixture.offsets, fixture.metrics,
        fixture.b, fixture.rhs,
    )
    tolerance = _ns3_tolerance(T)
    @test result.status === SDPX.NS_SCHUR3_ASSEMBLED
    @test result.reason === SDPX.NS_SCHUR3_CONVERGED
    @test fixture.workspace.setup_valid
    @test fixture.H == transpose(fixture.H)
    @test isapprox(fixture.H, oracle.H; atol=tolerance, rtol=tolerance)
    @test isapprox(
        fixture.at_g_b, oracle.at_g_b; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        fixture.bt_g_a, oracle.bt_g_a; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        fixture.at_g_rhs, oracle.at_g_rhs; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        result.b_g_b, oracle.b_g_b; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        result.b_g_rhs, oracle.b_g_rhs; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        fixture.at_g_b, fixture.bt_g_a; atol=tolerance, rtol=tolerance,
    )
    return fixture, result
end

@noinline function _ns3_compiled_call!(fixture)
    return SDPX.try_assemble_nonsymmetric_schur3!(
        fixture.workspace,
        fixture.H,
        fixture.at_g_b,
        fixture.bt_g_a,
        fixture.at_g_rhs,
        fixture.metrics,
        fixture.b,
        fixture.rhs,
    )
end

@noinline function _ns3_allocation_profile(fixture)
    _ns3_compiled_call!(fixture)
    total = 0
    maximum_call = 0
    @inbounds for _ in 1:10
        bytes = @allocated _ns3_compiled_call!(fixture)
        total += bytes
        maximum_call = max(maximum_call, bytes)
    end
    return total, maximum_call
end

@noinline function _ns3_theta_compiled_call!(fixture)
    return SDPX.try_assemble_nonsymmetric_schur3_theta!(
        fixture.workspace,
        fixture.H,
        fixture.at_g_b,
        fixture.bt_g_a,
        fixture.at_g_rhs,
        fixture.metrics,
        fixture.b,
        fixture.rhs,
    )
end

function _ns3_check_theta_fixture(::Type{T}) where {T<:AbstractFloat}
    fixture = _ns3_fixture(T)
    result = _ns3_theta_compiled_call!(fixture)
    oracle = _ns3_theta_dense_oracle(
        fixture.dense_a,
        fixture.offsets,
        fixture.metrics,
        fixture.b,
        fixture.rhs,
    )
    tolerance = T(1_048_576) * eps(one(T))
    @test result.status === SDPX.NS_SCHUR3_ASSEMBLED
    @test result.reason === SDPX.NS_SCHUR3_CONVERGED
    @test fixture.H == transpose(fixture.H)
    @test isapprox(fixture.H, oracle.H; atol=tolerance, rtol=tolerance)
    @test isapprox(
        fixture.at_g_b, oracle.at_g_b; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        fixture.bt_g_a, oracle.bt_g_a; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        fixture.at_g_rhs, oracle.at_g_rhs; atol=tolerance, rtol=tolerance,
    )
    @test isapprox(result.b_g_b, oracle.b_g_b; atol=tolerance, rtol=tolerance)
    @test isapprox(
        result.b_g_rhs, oracle.b_g_rhs; atol=tolerance, rtol=tolerance,
    )
    return fixture, result
end

@testset "nonsymmetric Schur3 typed ABI" begin
    @test isbitstype(SDPX.NonsymmetricSchur3Status)
    @test isbitstype(SDPX.NonsymmetricSchur3Reason)
    @test isbitstype(SDPX.NonsymmetricSchur3Result{Float64})
    @test fieldnames(SDPX.NonsymmetricSchur3Result{Float64}) ==
          (:status, :reason, :b_g_b, :b_g_rhs)
end

@testset "sparse 3D dyadic assembly vs independent dense oracle" begin
    for T in (
        Float64,
        _NS3_MF.Float64x2,
        _NS3_MF.Float64x3,
        _NS3_MF.Float64x4,
    )
        @testset "$T" begin
            fixture, result = _ns3_check_fixture(T)
            @test isbits(result)
            total, maximum_call = _ns3_allocation_profile(fixture)
            @test total == 0
            @test maximum_call == 0
        end
    end
    setprecision(BigFloat, 256) do
        fixture, _ = _ns3_check_fixture(BigFloat)
        @test all(precision(value) == 256 for value in fixture.H)
        @test all(precision(value) == 256 for value in fixture.at_g_b)
        @test all(precision(value) == 256 for value in fixture.at_g_rhs)
    end
end

@testset "factor-space Theta Schur matches independent inverse oracle" begin
    for T in (
        Float64,
        _NS3_MF.Float64x2,
        _NS3_MF.Float64x3,
        _NS3_MF.Float64x4,
    )
        @testset "$T" begin
            fixture, result = _ns3_check_theta_fixture(T)
            @test isbits(result)
            _ns3_theta_compiled_call!(fixture)
            total = 0
            maximum_call = 0
            for _ in 1:10
                bytes = @allocated _ns3_theta_compiled_call!(fixture)
                total += bytes
                maximum_call = max(maximum_call, bytes)
            end
            @test total == 0
            @test maximum_call == 0
        end
    end
    setprecision(BigFloat, 256) do
        fixture, _ = _ns3_check_theta_fixture(BigFloat)
        @test all(value -> precision(value) == 256, fixture.H)
    end

    # A structurally zero column remains exactly zero in every contraction.
    fixture = _ns3_fixture(Float64)
    zero_A = copy(fixture.A)
    zero_A[:, 5] .= 0.0
    dropzeros!(zero_A)
    workspace = SDPX.NonsymmetricSchur3Workspace(zero_A, fixture.offsets)
    result = SDPX.try_assemble_nonsymmetric_schur3_theta!(
        workspace,
        fixture.H,
        fixture.at_g_b,
        fixture.bt_g_a,
        fixture.at_g_rhs,
        fixture.metrics,
        fixture.b,
        fixture.rhs,
    )
    @test result.status === SDPX.NS_SCHUR3_ASSEMBLED
    @test all(iszero, fixture.H[:, 5])
    @test all(iszero, fixture.H[5, :])
    @test iszero(fixture.at_g_b[5])
    @test iszero(fixture.bt_g_a[5])
    @test iszero(fixture.at_g_rhs[5])
end

@testset "raw duplicate CSC rows preserve linear-operator semantics" begin
    T = Float64
    rows = 7
    columns = 3
    # Each column is allowed unsorted selected rows and duplicate row entries.
    colptr = [1, 4, 7, 9]
    rowval = [5, 3, 3, 4, 3, 4, 5, 4]
    values = T[1/2, 1/4, -1/8, 3/8, -1/4, 1/8, -1/8, 3/4]
    A = SparseMatrixCSC{T,Int}(rows, columns, colptr, rowval, values)
    offsets = [3]
    metrics = zeros(T, 3, 3, 1)
    _ns3_metric!(metrics, 1, T[2 1/4 -1/8; 1/4 3/2 1/16; -1/8 1/16 5/4])
    b = T[7, -11, 1/2, -3/4, 5/8, 13, -17]
    rhs = T[-19, 23, -1/8, 5/16, 3/4, -29, 31]
    workspace = SDPX.NonsymmetricSchur3Workspace(A, offsets)
    H = zeros(T, columns, columns)
    at_g_b = zeros(T, columns)
    bt_g_a = zeros(T, columns)
    at_g_rhs = zeros(T, columns)
    result = SDPX.try_assemble_nonsymmetric_schur3!(
        workspace, H, at_g_b, bt_g_a, at_g_rhs, metrics, b, rhs,
    )
    oracle = _ns3_dense_oracle(
        _ns3_effective_dense(A), offsets, metrics, b, rhs,
    )
    @test result.status === SDPX.NS_SCHUR3_ASSEMBLED
    @test H ≈ oracle.H
    @test at_g_b ≈ oracle.at_g_b
    @test bt_g_a ≈ oracle.bt_g_a
    @test at_g_rhs ≈ oracle.at_g_rhs
    @test result.b_g_b ≈ oracle.b_g_b
    @test result.b_g_rhs ≈ oracle.b_g_rhs
end

function _ns3_failure_call(fixture, metrics, b, rhs; H=fixture.H)
    fill!(H, 9)
    fill!(fixture.at_g_b, 9)
    fill!(fixture.bt_g_a, 9)
    fill!(fixture.at_g_rhs, 9)
    result = SDPX.try_assemble_nonsymmetric_schur3!(
        fixture.workspace,
        H,
        fixture.at_g_b,
        fixture.bt_g_a,
        fixture.at_g_rhs,
        metrics,
        b,
        rhs,
    )
    @test all(iszero, H)
    @test all(iszero, fixture.at_g_b)
    @test all(iszero, fixture.bt_g_a)
    @test all(iszero, fixture.at_g_rhs)
    @test iszero(result.b_g_b)
    @test iszero(result.b_g_rhs)
    return result
end

function _ns3_theta_failure_call(fixture, thetas, b, rhs)
    fill!(fixture.H, 9)
    fill!(fixture.at_g_b, 9)
    fill!(fixture.bt_g_a, 9)
    fill!(fixture.at_g_rhs, 9)
    result = SDPX.try_assemble_nonsymmetric_schur3_theta!(
        fixture.workspace,
        fixture.H,
        fixture.at_g_b,
        fixture.bt_g_a,
        fixture.at_g_rhs,
        thetas,
        b,
        rhs,
    )
    @test all(iszero, fixture.H)
    @test all(iszero, fixture.at_g_b)
    @test all(iszero, fixture.bt_g_a)
    @test all(iszero, fixture.at_g_rhs)
    @test iszero(result.b_g_b)
    @test iszero(result.b_g_rhs)
    return result
end

@testset "factor-space Theta Schur failures are typed and closed" begin
    fixture = _ns3_fixture(Float64)

    nonfinite = copy(fixture.metrics)
    nonfinite[1, 1, 1] = Inf
    result = _ns3_theta_failure_call(
        fixture, nonfinite, fixture.b, fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_NONFINITE_METRIC

    nonsymmetric = copy(fixture.metrics)
    nonsymmetric[1, 2, 1] += 1e-5
    result = _ns3_theta_failure_call(
        fixture, nonsymmetric, fixture.b, fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_NONSYMMETRIC_METRIC

    indefinite = copy(fixture.metrics)
    indefinite[:, :, 1] .= [1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 1.0]
    result = _ns3_theta_failure_call(
        fixture, indefinite, fixture.b, fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_METRIC_NOT_SPD

    huge_b = copy(fixture.b)
    huge_b[fixture.offsets[1]] = floatmax(Float64)
    result = _ns3_theta_failure_call(
        fixture, fixture.metrics, huge_b, fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_NONFINITE_RESULT
end

@testset "typed fail-closed numerical and structural gates" begin
    fixture = _ns3_fixture(Float64)

    nonfinite_metric = copy(fixture.metrics)
    nonfinite_metric[1, 1, 1] = Inf
    result = _ns3_failure_call(
        fixture, nonfinite_metric, fixture.b, fixture.rhs,
    )
    @test result.status === SDPX.NS_SCHUR3_FAILED
    @test result.reason === SDPX.NS_SCHUR3_NONFINITE_METRIC

    nonsymmetric_metric = copy(fixture.metrics)
    nonsymmetric_metric[2, 1, 1] += 1e-6
    result = _ns3_failure_call(
        fixture, nonsymmetric_metric, fixture.b, fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_NONSYMMETRIC_METRIC

    indefinite_metric = copy(fixture.metrics)
    indefinite_metric[:, :, 1] .= [1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 1.0]
    result = _ns3_failure_call(
        fixture, indefinite_metric, fixture.b, fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_METRIC_NOT_SPD

    nonfinite_b = copy(fixture.b)
    nonfinite_b[fixture.offsets[1]] = Inf
    result = _ns3_failure_call(
        fixture, fixture.metrics, nonfinite_b, fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_NONFINITE_VECTOR

    nonfinite_rhs = copy(fixture.rhs)
    nonfinite_rhs[fixture.offsets[2] + 1] = NaN
    result = _ns3_failure_call(
        fixture, fixture.metrics, fixture.b, nonfinite_rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_NONFINITE_VECTOR

    bad_H = zeros(6, 5)
    result = _ns3_failure_call(
        fixture, fixture.metrics, fixture.b, fixture.rhs; H=bad_H,
    )
    @test result.reason === SDPX.NS_SCHUR3_INVALID_DIMENSION

    # A selected non-finite frozen coefficient is recorded at setup and
    # reported through the same typed hot API.
    bad_A = copy(fixture.A)
    bad_A[fixture.offsets[2], 1] = Inf
    bad_fixture = merge(
        fixture,
        (workspace=SDPX.NonsymmetricSchur3Workspace(
            bad_A, fixture.offsets,
        ),),
    )
    result = _ns3_failure_call(
        bad_fixture, bad_fixture.metrics, bad_fixture.b, bad_fixture.rhs,
    )
    @test result.reason === SDPX.NS_SCHUR3_NONFINITE_A

    # Finite inputs whose dyadic contraction overflows are not mistaken for a
    # valid Schur epoch.
    overflow_A = sparse([1], [1], [floatmax(Float64) / 2], 3, 1)
    overflow_workspace = SDPX.NonsymmetricSchur3Workspace(overflow_A, [1])
    overflow_metric = zeros(3, 3, 1)
    overflow_metric[:, :, 1] .= Matrix{Float64}(I, 3, 3)
    overflow_H = zeros(1, 1)
    overflow_atgb = zeros(1)
    overflow_btga = zeros(1)
    overflow_atgrhs = zeros(1)
    overflow_result = SDPX.try_assemble_nonsymmetric_schur3!(
        overflow_workspace,
        overflow_H,
        overflow_atgb,
        overflow_btga,
        overflow_atgrhs,
        overflow_metric,
        ones(3),
        ones(3),
    )
    @test overflow_result.status === SDPX.NS_SCHUR3_FAILED
    @test overflow_result.reason === SDPX.NS_SCHUR3_NONFINITE_RESULT
    @test all(iszero, overflow_H)
    @test all(iszero, overflow_atgb)
    @test all(iszero, overflow_btga)
    @test all(iszero, overflow_atgrhs)

    # Structural ownership errors are setup bugs and remain hard failures.
    @test_throws ArgumentError SDPX.NonsymmetricSchur3Workspace(
        fixture.A, [2, 4],
    )
    @test_throws ArgumentError SDPX.NonsymmetricSchur3Workspace(
        fixture.A, [10],
    )

    # A previous rejected metric cannot poison a later valid epoch.
    recovered, _ = _ns3_check_fixture(Float64)
    @test SDPX.try_assemble_nonsymmetric_schur3!(
        recovered.workspace,
        recovered.H,
        recovered.at_g_b,
        recovered.bt_g_a,
        recovered.at_g_rhs,
        recovered.metrics,
        recovered.b,
        recovered.rhs,
    ).status === SDPX.NS_SCHUR3_ASSEMBLED
end
