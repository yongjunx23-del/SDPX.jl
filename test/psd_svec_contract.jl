# Phase 0-D: PSD raw-lower / MOI / HSD-svec coordinate contract.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using MultiFloats

function _psd_contract_values(::Type{T}, n::Int; precision_bits::Int=precision(T)) where {T<:AbstractFloat}
    values = Vector{T}(undef, SDPX.psd_packed_length(n))
    @inbounds for position in eachindex(values)
        row = SDPX.psd_packed_row(position, n)
        column = SDPX.psd_packed_column(position, n)
        # Keep the fixture deterministic and nonzero on every off-diagonal.
        value = T(row + 3 * column) / T(7)
        values[position] = T === BigFloat ?
            SDPX.owned_arithmetic_copy(T, value; precision_bits=precision_bits) : value
    end
    return values
end

@testset "PSD raw lower and svec maps" begin
    for n in (1, 2, 3, 4, 8)
        raw = _psd_contract_values(Float64, n)
        execution = zeros(Float64, length(raw))
        roundtrip = similar(raw)
        map = SDPX.PSDCoordinateMap(Float64, n)
        SDPX.matrix_raw_lower_to_svec!(execution, raw, map)
        SDPX.svec_to_matrix_raw_lower!(roundtrip, execution, map)
        @test roundtrip ≈ raw

        # The coordinate map is alias-safe for all four directions.
        alias = copy(raw)
        SDPX.matrix_raw_lower_to_svec!(alias, alias, map)
        SDPX.svec_to_matrix_raw_lower!(alias, alias, map)
        @test alias ≈ raw
        dual_raw = similar(raw)
        SDPX.svec_dual_to_raw!(dual_raw, execution, map)
        SDPX.raw_dual_to_svec!(alias, dual_raw, map)
        @test alias ≈ execution
    end
end

@testset "PSD svec is the trace inner product" begin
    S = [4.0 1.0; 1.0 2.0]
    Y = [1.0 0.5; 0.5 2.0]
    raw_s = [4.0, 1.0, 2.0]
    raw_y = [1.0, 0.5, 2.0]
    s = zeros(3)
    y = zeros(3)
    map = SDPX.PSDCoordinateMap(Float64, 2)
    SDPX.matrix_raw_lower_to_svec!(s, raw_s, map)
    SDPX.matrix_raw_lower_to_svec!(y, raw_y, map)
    @test dot(s, y) ≈ tr(S * Y)
    @test dot(raw_s, raw_y) ≈ 8.5
    @test dot(s, y) ≈ 9.0

    raw_multiplier = zeros(3)
    SDPX.svec_dual_to_raw!(raw_multiplier, y, map)
    @test raw_multiplier ≈ [1.0, 1.0, 2.0]
    recovered = zeros(2, 2)
    SDPX.reconstruct_psd_dual_matrix!(recovered, y, 2)
    @test recovered ≈ Y
end

@testset "MOI upper packed and SDPX raw lower round-trip" begin
    for n in (1, 2, 3, 4, 8)
        raw = _psd_contract_values(Float64, n)
        moi = similar(raw)
        recovered = similar(raw)
        SDPX.raw_lower_to_moi_upper!(moi, raw, n)
        SDPX.moi_upper_to_raw_lower!(recovered, moi, n)
        @test recovered ≈ raw
        execution = zeros(length(raw))
        SDPX.raw_lower_to_svec!(execution, recovered, n)
        SDPX.svec_to_raw_lower!(recovered, execution, n)
        @test recovered ≈ raw
    end
end

@testset "canonical PSD rows use D and reconstruction uses D inverse" begin
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    block = only(SDPX.layout_blocks(canonical.cone_layout))
    @test SDPX.block_execution_storage(block) === :svec
    @test canonical.A[1, 1] == -1.0
    @test canonical.A[2, 2] == -sqrt(2.0)
    @test canonical.A[3, 3] == -1.0

    x = [4.0, 1.0, 2.0]
    xc = zeros(3)
    sc = zeros(3)
    SDPX.primal_backward!(canonical, xc, sc, x)
    @test sc ≈ [4.0, sqrt(2.0), 2.0]
    xo = zeros(3)
    so = zeros(3)
    SDPX.primal_forward!(canonical, xo, so, xc, sc)
    @test xo == x
    @test so == x

    execution_dual = [1.0, sqrt(2.0) / 2, 2.0]
    raw_row = zeros(3)
    SDPX.dual_forward!(canonical, raw_row, execution_dual)
    @test raw_row ≈ [1.0, 1.0, 2.0]
    recovered_execution = zeros(3)
    SDPX.dual_backward!(canonical, recovered_execution, raw_row)
    @test recovered_execution ≈ execution_dual

    # Affine PSD rows use the same D map on both the sparse equality rows and
    # their rhs; this catches the common bug where only product PSD variables
    # are converted to svec.
    affine_model = SDPX.Model(Float64)
    z = SDPX.variable!(affine_model, :z, 3; domain=SDPX.Reals())
    SDPX.objective!(affine_model, SDPX.Minimize(), z[1])
    SDPX.constraint!(
        affine_model,
        :psd_affine,
        [z[1] - 4.0 z[2] - 1.0; z[2] - 1.0 z[3] - 2.0],
        SDPX.PSDCone(),
    )
    affine = SDPX.canonicalize(SDPX.compile_product_cone_model(affine_model))
    affine_block = only(SDPX.layout_blocks(affine.cone_layout))
    @test affine.A[1, 1] == -1.0
    @test affine.A[2, 2] == -sqrt(2.0)
    @test affine.A[3, 3] == -1.0
    @test affine.b ≈ [-4.0, -sqrt(2.0), -2.0]
    @test affine_block.reconstruction.coordinate_map isa SDPX.PSDCoordinateMap

    raw_A = Matrix{Float64}(I, 3, 3)
    raw_b = [4.0, 1.0, 2.0]
    SDPX.apply_psd_row_scaling!(raw_A, raw_b, affine_block)
    @test raw_A[2, 2] == sqrt(2.0)
    @test raw_b ≈ [4.0, sqrt(2.0), 2.0]
end

@testset "PSD precision ownership and warm caller-owned conversion" begin
    for T in (Float64, Float64x2, Float64x4)
        n = 4
        map = SDPX.PSDCoordinateMap(T, n)
        raw = _psd_contract_values(T, n)
        execution = zeros(T, length(raw))
        recovered = zeros(T, length(raw))
        SDPX.matrix_raw_lower_to_svec!(execution, raw, map)
        SDPX.svec_to_matrix_raw_lower!(recovered, execution, map)
        @test recovered ≈ raw
        # Compile and warm the fixed-width caller-owned path before measuring.
        for _ in 1:10
            SDPX.matrix_raw_lower_to_svec!(execution, raw, map)
            SDPX.svec_to_matrix_raw_lower!(recovered, execution, map)
        end
        @test @allocated(SDPX.matrix_raw_lower_to_svec!(execution, raw, map)) == 0
    end

    setprecision(BigFloat, 256) do
        n = 3
        map = SDPX.PSDCoordinateMap(BigFloat, n; precision_bits=256)
        raw = _psd_contract_values(BigFloat, n; precision_bits=256)
        execution = BigFloat[zero(BigFloat) for _ in raw]
        recovered = similar(execution)
        setprecision(BigFloat, 64) do
            SDPX.matrix_raw_lower_to_svec!(execution, raw, map)
            SDPX.svec_to_matrix_raw_lower!(recovered, execution, map)
        end
        @test all(precision(value) >= 256 for value in execution)
        @test all(precision(value) >= 256 for value in recovered)
        @test recovered == raw
    end
end
