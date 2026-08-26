using Test
using LinearAlgebra
using SparseArrays
using MultiFloats
using SDPX

function _pcapi_layout(::Type{T}, specs) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for (cone, dimension) in specs
        block = SDPX.ConeBlockDescriptor(T, cone, dimension; offset=offset)
        push!(blocks, block)
        offset += block.length
    end
    return SDPX.canonical_layout(blocks)
end

function _pcapi_identity(::Type{T}, layout) where {T}
    result = zeros(T, layout.dimension)
    for block in SDPX.layout_blocks(layout)
        if block.cone === :nonnegative
            result[block.offset:block.offset + block.length - 1] .= one(T)
        elseif block.cone === :soc
            result[block.offset] = one(T)
        elseif block.cone === :psd
            cursor = block.offset
            for column in 1:block.dimension
                for row in column:block.dimension
                    result[cursor] = row == column ? one(T) : zero(T)
                    cursor += 1
                end
            end
        end
    end
    return result
end

function _pcapi_warm_calls!(
    runtime, H, A, schur_scratch, h, s, y, ds_aff, dy_aff,
    ds, dy, theta_scratch, primal_rhs,
)
    SDPX.affine_shift!(runtime, h, s, y)
    SDPX.corrector_shift!(runtime, h, s, y, ds_aff, dy_aff, one(eltype(s)))
    SDPX.assemble_schur!(runtime, H, A, schur_scratch)
    SDPX.recover_direction!(
        runtime, ds, dy, h, primal_rhs, theta_scratch,
    )
    return nothing
end

@testset "frozen symmetric product runtime API and SOC central target" begin
    specs = [(:nonnegative, 2), (:soc, 3), (:psd, 2)]
    layout = _pcapi_layout(Float64, specs)
    runtime = SDPX.ProductConeRuntime(layout, Float64)
    s = zeros(layout.dimension)
    y = similar(s)
    SDPX.initialize_primal_dual!(runtime, s, y)
    e = _pcapi_identity(Float64, layout)
    @test s == y == e

    affine = similar(s)
    @test SDPX.affine_shift!(runtime, affine, s, y) === affine
    @test affine ≈ -e atol=3e-14 rtol=3e-14

    ds_aff = zeros(layout.dimension)
    dy_aff = zeros(layout.dimension)
    corrector = similar(s)
    @test SDPX.corrector_shift!(
        runtime, corrector, s, y, ds_aff, dy_aff, 1.0,
    ) === corrector
    expected = zeros(layout.dimension)
    soc = only(runtime.soc)
    expected[soc.offset] = 1.0
    # This independently distinguishes the ordinary-dot SOC target 2e from
    # the incorrect one-e target, which would yield an all-zero shift here.
    @test corrector ≈ expected atol=3e-14 rtol=3e-14

    scaled_primal = similar(s)
    scaled_dual = similar(s)
    compatibility = similar(s)
    SDPX.symmetric_corrector_shift!(
        runtime, compatibility, scaled_primal, scaled_dual,
        ds_aff, dy_aff, 1.0,
    )
    @test compatibility ≈ expected atol=3e-14 rtol=3e-14
    @test all(iszero, scaled_primal)
    @test all(iszero, scaled_dual)

    A = sparse([
        1.0  -0.2
        0.3   0.7
        -0.5  0.1
        0.4  -0.8
        0.2   0.6
        -0.7  0.9
        0.8  -0.4
        0.1   0.5
    ])
    H = zeros(2, 2)
    schur_scratch = SDPX.ProductSchurScratch(runtime)
    SDPX.assemble_schur!(runtime, H, A, schur_scratch)
    @test H ≈ Matrix(transpose(A) * A) atol=3e-13 rtol=3e-13

    primal_rhs = collect(range(-0.4, 0.3; length=layout.dimension))
    ds = similar(s)
    dy = similar(s)
    theta_scratch = similar(s)
    SDPX.recover_direction!(
        runtime, ds, dy, corrector, primal_rhs, theta_scratch,
    )
    @test dy ≈ primal_rhs atol=3e-13 rtol=3e-13
    @test ds ≈ corrector - primal_rhs atol=3e-13 rtol=3e-13

    @test_throws DimensionMismatch SDPX.assemble_schur!(
        runtime, zeros(3, 3), A, schur_scratch,
    )
    bad = copy(s)
    bad[1] = Inf
    @test_throws DomainError SDPX.affine_shift!(runtime, affine, bad, y)
end

@testset "frozen product runtime API fixed-width allocation gate" begin
    specs = [(:nonnegative, 2), (:soc, 3)]
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        layout = _pcapi_layout(T, specs)
        runtime = SDPX.ProductConeRuntime(layout, T)
        s = zeros(T, layout.dimension)
        y = similar(s)
        SDPX.initialize_primal_dual!(runtime, s, y)
        A = sparse(T[
            1  1 / 4
            1 / 3  -1 / 2
            2 / 5  3 / 7
            -1 / 5 4 / 9
            3 / 8  -2 / 7
        ])
        H = zeros(T, 2, 2)
        schur_scratch = SDPX.ProductSchurScratch(runtime)
        h = similar(s)
        ds_aff = zeros(T, layout.dimension)
        dy_aff = zeros(T, layout.dimension)
        ds = similar(s)
        dy = similar(s)
        theta_scratch = similar(s)
        primal_rhs = T[T(index) / T(17) for index in 1:layout.dimension]
        _pcapi_warm_calls!(
            runtime, H, A, schur_scratch, h, s, y, ds_aff, dy_aff,
            ds, dy, theta_scratch, primal_rhs,
        )
        samples = ntuple(10) do _
            @allocated _pcapi_warm_calls!(
                runtime, H, A, schur_scratch, h, s, y, ds_aff, dy_aff,
                ds, dy, theta_scratch, primal_rhs,
            )
        end
        @test all(iszero, samples)
    end
end
