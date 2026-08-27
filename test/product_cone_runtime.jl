# Native symmetric product-cone runtime (P1-A/P1-D).

using Test
using LinearAlgebra
using SDPX

const PCR_SC = SDPX.SymmetricCones

function _pcr_layout(::Type{T}, specs) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for (kind, dim) in specs
        push!(blocks, SDPX.ConeBlockDescriptor(T, kind, dim; offset=offset))
        offset += blocks[end].length
    end
    return SDPX.canonical_layout(blocks)
end

function _pcr_warm_alloc(runtime, s, y, src, dst, ds)
    return @allocated begin
        for _ in 1:10
            SDPX.update_scaling!(runtime, s, y, 1.0)
            SDPX.apply_Theta!(runtime, dst, src)
            SDPX.apply_G!(runtime, dst, src)
            SDPX.max_step_primal!(runtime, s, ds)
            SDPX.max_step_dual!(runtime, y, ds)
        end
    end
end

@testset "ProductConeRuntime setup and identity" begin
    layout = _pcr_layout(Float64, [(:nonnegative, 2), (:soc, 3), (:psd, 2)])
    runtime = SDPX.ProductConeRuntime(layout, Float64)
    @test runtime.dimension == 2 + 3 + 3
    @test length(runtime.orthant) == 1
    @test length(runtime.soc) == 1
    @test length(runtime.psd) == 1
    @test isempty(runtime.exp)
    @test isempty(runtime.power)

    s = zeros(8)
    y = zeros(8)
    SDPX.initialize_primal_dual!(runtime, s, y)
    @test runtime.valid
    @test s == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    @test y == s
    @test PCR_SC.membership(PCR_SC.NonnegativeCone(2), s[1:2])
    @test PCR_SC.membership(PCR_SC.SOCone(3), s[3:5])
    # PSD identity is encoded in svec coordinates.
    @test s[6:8] == [1.0, 0.0, 1.0]
    @test SDPX.cone_inner_product(runtime, s, y) == 5.0
end

@testset "ProductConeRuntime pair orientation and trace product" begin
    layout = _pcr_layout(Float64, [(:nonnegative, 2), (:soc, 3), (:psd, 2)])
    runtime = SDPX.ProductConeRuntime{Float64}(layout)
    s = [2.0, 3.0, 3.0, 1.0, 0.5, 2.0, 1.0, 1.5]
    y = [5.0, 7.0, 4.0, -0.5, 0.25, 1.0, -sqrt(2.0), 2.0]
    @test PCR_SC.membership(PCR_SC.SOCone(3), s[3:5])
    @test PCR_SC.membership(PCR_SC.SOCone(3), y[3:5])
    # PSD svec vectors represent [[d1, off/sqrt(2)], [off/sqrt(2), d2]].
    S0 = [s[6] s[7] / sqrt(2); s[7] / sqrt(2) s[8]]
    Y0 = [y[6] y[7] / sqrt(2); y[7] / sqrt(2) y[8]]
    @test minimum(eigvals(Symmetric(S0))) > 0
    @test minimum(eigvals(Symmetric(Y0))) > 0
    SDPX.update_scaling!(runtime, s, y, 1.0)
    theta_y = similar(s)
    g_s = similar(s)
    SDPX.apply_Theta!(runtime, theta_y, y)
    SDPX.apply_G!(runtime, g_s, s)
    @test theta_y ≈ s rtol=2e-10 atol=2e-10
    @test g_s ≈ y rtol=2e-10 atol=2e-10

    expected = dot(s[1:5], y[1:5]) + dot(s[6:8], y[6:8])
    @test SDPX.cone_inner_product(runtime, s, y) ≈ expected
    # In the PSD block this is exactly tr(SY), including the off-diagonal
    # factor represented by svec.
    S = [s[6] s[7] / sqrt(2); s[7] / sqrt(2) s[8]]
    Y = [y[6] y[7] / sqrt(2); y[7] / sqrt(2) y[8]]
    @test dot(s[6:8], y[6:8]) ≈ tr(S * Y)

    # Aliasing is safe: the source is copied to setup-owned block scratch.
    alias = copy(y)
    SDPX.apply_Theta!(runtime, alias, alias)
    @test alias ≈ s rtol=2e-10 atol=2e-10
end

@testset "ProductConeRuntime Cholesky-stable PSD pair update" begin
    layout = _pcr_layout(Float64, [(:psd, 3)])
    runtime = SDPX.ProductConeRuntime(layout, Float64)

    # Misaligned SPD eigenspaces with cond(Y)≈2e9 reproduce the old runtime
    # path's explicit Y^(-1/2) orientation rejection. The Cholesky congruence
    # remains fail-closed but certifies the pair in factorized coordinates.
    Q = Matrix(qr([1.0 2.0 3.0; 4.0 1.0 2.0; 2.0 5.0 1.0]).Q)
    R = Matrix(qr([2.0 1.0 4.0; 1.0 3.0 2.0; 5.0 2.0 1.0]).Q)
    Y = Q * Diagonal([1e-9, 0.7, 2.0]) * transpose(Q)
    S = R * Diagonal([3e-8, 0.4, 1.5]) * transpose(R)
    s = zeros(6)
    y = zeros(6)
    PCR_SC._pack_svec!(s, S, 3, sqrt(2.0))
    PCR_SC._pack_svec!(y, Y, 3, sqrt(2.0))

    @test cond(Y) > 1e9
    @test SDPX.try_update_scaling!(runtime, s, y, 1.0)
    theta_y = similar(s)
    g_s = similar(s)
    SDPX.apply_Theta!(runtime, theta_y, y)
    SDPX.apply_G!(runtime, g_s, s)
    @test theta_y ≈ s rtol=1e-9 atol=1e-9
    @test g_s ≈ y rtol=1e-9 atol=1e-9
end

@testset "ProductConeRuntime global boundary steps" begin
    layout = _pcr_layout(Float64, [(:nonnegative, 2), (:soc, 3), (:psd, 2)])
    runtime = SDPX.ProductConeRuntime(layout, Float64)
    s = zeros(8); y = zeros(8)
    SDPX.initialize_primal_dual!(runtime, s, y)

    ds = zeros(8)
    ds[1] = -1.0                 # orthant boundary at one
    ds[3] = 0.0; ds[4] = 1.0    # SOC boundary at one
    ds[6] = -1.0                 # PSD generalized eigenvalue -1
    @test SDPX.max_step_primal!(runtime, s, ds) ≈ 1.0
    @test SDPX.max_step_dual!(runtime, y, ds) ≈ 1.0
    @test SDPX.max_step_primal!(runtime, s, zeros(8)) == Inf

    # A current point outside the cone fails closed with a zero step.
    bad = copy(s); bad[1] = -1.0
    @test SDPX.max_step_primal!(runtime, bad, zeros(8)) == 0.0
end

@testset "ProductConeRuntime empty families and setup rejection" begin
    only_lp = _pcr_layout(Float64, [(:nonnegative, 3)])
    runtime = SDPX.ProductConeRuntime(only_lp, Float64)
    @test isempty(runtime.soc)
    @test isempty(runtime.psd)
    s = zeros(3); y = zeros(3)
    SDPX.initialize_primal_dual!(runtime, s, y)
    @test s == ones(3)
    out = zeros(3)
    SDPX.apply_Theta!(runtime, out, y)
    @test out == s

    # Exp/Power are now first-class runtime families: the constructor builds
    # their typed block storage and leaves the symmetric families empty.  It
    # still has no fallback for a family it cannot execute.
    exp_runtime = SDPX.ProductConeRuntime(
        _pcr_layout(Float64, [(:exp, 3)]), Float64,
    )
    @test length(exp_runtime.exp) == 1
    @test isempty(exp_runtime.power)
    @test isempty(exp_runtime.orthant) && isempty(exp_runtime.soc) &&
          isempty(exp_runtime.psd)

    power_runtime = SDPX.ProductConeRuntime(
        _pcr_layout(Float64, [(:power, 3)]), Float64,
    )
    @test length(power_runtime.power) == 1
    @test isempty(power_runtime.exp)
    @test isempty(power_runtime.orthant) && isempty(power_runtime.soc) &&
          isempty(power_runtime.psd)

    @test_throws ArgumentError SDPX.ProductConeRuntime(
        _pcr_layout(Float64, [(:zero, 1)]), Float64,
    )

    bad_desc = SDPX.ConeBlockDescriptor{Float64}(
        :psd, 1, 2, 4, :packed_lower, 0.0,
        SDPX.CanonicalBlockMap{Float64}(:none, 0, 0, 1),
    )
    bad_layout = SDPX.ConeProductLayout([bad_desc], 4, 2)
    @test_throws ArgumentError SDPX.ProductConeRuntime(bad_layout, Float64)

    noncontiguous = SDPX.ConeBlockDescriptor(Float64, :nonnegative, 1; offset=2)
    bad_offsets = SDPX.ConeProductLayout([noncontiguous], 1, 1)
    @test_throws ArgumentError SDPX.ProductConeRuntime(bad_offsets, Float64)
end

@testset "ProductConeRuntime invalid pair and warm fixed-width path" begin
    layout = _pcr_layout(Float64, [(:nonnegative, 4), (:soc, 4), (:psd, 2)])
    runtime = SDPX.ProductConeRuntime(layout, Float64)
    s = zeros(11); y = zeros(11)
    SDPX.initialize_primal_dual!(runtime, s, y)
    invalid = copy(s); invalid[1] = 0.0
    @test_throws DomainError SDPX.update_scaling!(runtime, invalid, y, 1.0)
    @test !runtime.valid
    @test_throws ArgumentError SDPX.update_scaling!(runtime, BigFloat.(s), BigFloat.(y), big"1")
    SDPX.update_scaling!(runtime, s, y, 1.0)
    src = copy(y); dst = similar(src); ds = zeros(11)
    SDPX.apply_Theta!(runtime, dst, src)
    SDPX.apply_G!(runtime, dst, src)
    SDPX.max_step_primal!(runtime, s, ds)
    SDPX.max_step_dual!(runtime, y, ds)
    warm_alloc = _pcr_warm_alloc(runtime, s, y, src, dst, ds)
    @test warm_alloc == 0
end

@testset "ProductConeRuntime BigFloat arithmetic" begin
    T = BigFloat
    layout = _pcr_layout(T, [(:soc, 3), (:psd, 2)])
    runtime = SDPX.ProductConeRuntime(layout, T)
    s = zeros(T, layout.dimension); y = zeros(T, layout.dimension)
    SDPX.initialize_primal_dual!(runtime, s, y)
    @test runtime.valid
    @test SDPX.max_step_primal!(runtime, s, zeros(T, layout.dimension)) == Inf
    s[1] = big"2"; s[2] = big"0.25"; s[3] = big"-0.125"
    y[1] = big"3"; y[2] = big"-0.125"; y[3] = big"0.25"
    # Keep the PSD pair strictly positive while exercising extended arithmetic.
    s[4:6] .= T[big"2", big"0.1", big"1.5"]
    y[4:6] .= T[big"1.5", big"-0.05", big"2"]
    SDPX.update_scaling!(runtime, s, y, one(T))
    theta_y = zeros(T, layout.dimension)
    SDPX.apply_Theta!(runtime, theta_y, y)
    @test theta_y ≈ s rtol=big"1e-45" atol=big"1e-45"
end
