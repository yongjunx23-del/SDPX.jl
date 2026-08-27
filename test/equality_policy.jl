# Phase 4 equality-policy selection tests (prepared infrastructure).
using SDPX
using Test
using LinearAlgebra
using SparseArrays

function _eqpolicy_canonical(::Type{T}; nz::Int, ncol::Int) where {T<:AbstractFloat}
    # nz zero rows, ncol variables; active rows = nz (nonnegative) so total m=2*nz.
    A = spzeros(T, 2 * nz, ncol)
    b = zeros(T, 2 * nz)
    c = zeros(T, ncol)
    identity_map = SDPX.CanonicalBlockMap{Float64}(:none, 0, 0, 1)
    blocks = SDPX.ConeBlockDescriptor{Float64}[]
    offset = 1
    for _ in 1:nz
        push!(blocks, SDPX.ConeBlockDescriptor(Float64, :zero, 1; offset=offset, reconstruction=identity_map)); offset += 1
    end
    for _ in 1:nz
        push!(blocks, SDPX.ConeBlockDescriptor(Float64, :nonnegative, 1; offset=offset, reconstruction=identity_map)); offset += 1
    end
    layout = SDPX.canonical_layout(blocks)
    bits = 53
    chain = SDPX.CanonicalReconstructionChain(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 1,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, A, b, layout, chain,
    )
end

@testset "equality policy selection" begin
    T = Float64
    # small dense equality system (nz=2, ncol=2 -> fill_estimate 4 <= 64^2)
    small = _eqpolicy_canonical(T; nz=2, ncol=2)
    p1, r1 = SDPX.select_equality_policy(small)
    @test p1 isa SDPX.EqualityPolicyRRQR
    @test r1 === :small_dense

    # no equalities
    none = _eqpolicy_canonical(T; nz=0, ncol=3)
    p2, r2 = SDPX.select_equality_policy(none)
    @test p2 isa SDPX.EqualityPolicyRetain
    @test r2 === :no_equalities

    # large equality system -> retain_mixed (me <= n and ma >= me)
    big = _eqpolicy_canonical(T; nz=100, ncol=200)
    p3, r3 = SDPX.select_equality_policy(big)
    @test p3 isa SDPX.EqualityPolicyRetain
    @test r3 === :retain_mixed

    # sparse-QR prepared marker surfaced when fill estimate large and
    # equality count exceeds variables (me > n) -> sparse_qr_prepared_not_executable
    sparse_marker = _eqpolicy_canonical(T; nz=200, ncol=10)
    p4, r4 = SDPX.select_equality_policy(sparse_marker)
    @test p4 isa SDPX.EqualityPolicySparseQR
    @test r4 === :sparse_qr_prepared_not_executable
end