# Canonical slack-cone IR (Subagent C).
#
# These tests run the FULL canonicalization pipeline: build a real
# frontend `Model`, compile it to `NativeConeProgram`, canonicalize to
# `CanonicalConicProgram`, and inspect the resulting canonical slack
# layout / reconstruction maps. They never hand-construct a
# `ConeBlockDescriptor`.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using SparseArrays
using LinearAlgebra

@testset "canonicalize keeps free variables free in x" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())  # v in R_+^2
    free = SDPX.variable!(model, :free, 3; domain=SDPX.Reals())  # free x
    SDPX.objective!(model, SDPX.Minimize(), x[1] + free[1])

    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)

    # x stays in frontend order and the objective is preserved.
    @test SDPX.canonical_num_variables(canon) == SDPX.program_num_variables(ncp)
    @test SDPX.canonical_num_variables(canon) == 5
    @test SDPX.canonical_objective(canon) == [1.0, 0.0, 1.0, 0.0, 0.0]

    # Only the nonnegative variable contributes a canonical slack block; the
    # free variables contribute NO slack rows.
    blocks = SDPX.layout_blocks(canon.cone_layout)
    @test length(blocks) == 1
    @test SDPX.block_cone(blocks[1]) === :nonnegative
    @test SDPX.block_length(blocks[1]) == 2
    @test SDPX.canonical_num_slack(canon) == 2
    @test SDPX.layout_barrier_degree(canon.cone_layout) == 2

    # Rectangular: A is m×n with m = 2 < n = 5.
    A = SDPX.canonical_equality(canon)
    @test size(A) == (2, 5)
    @test SDPX.canonical_rhs(canon) == zeros(2)

    # The slack rows only reference the nonnegative columns.
    @test Set(A.rowval) == Set([1, 2])
    # Identity rows: A = -I on the nonnegative columns, so slack s = v.
    @test A[1, 1] == -1.0
    @test A[2, 2] == -1.0
end

@testset "ZeroCone becomes a plain equality (no barrier block)" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    z = SDPX.variable!(model, :z, 1; domain=SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + z[1])
    SDPX.constraint!(model, :eq, (x[1],), SDPX.ZeroCone())

    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)

    blocks = SDPX.layout_blocks(canon.cone_layout)
    @test length(blocks) == 2
    @test SDPX.block_cone(blocks[1]) === :zero   # z == 0 variable block
    @test SDPX.block_cone(blocks[2]) === :zero   # affine ZeroCone equality
    # zero blocks contribute no barrier
    @test SDPX.layout_barrier_degree(canon.cone_layout) == 0
    @test SDPX.canonical_num_slack(canon) == 2
end

@testset "Nonpositive maps to Nonnegative via a -1 sign map" begin
    model = SDPX.Model(Float64)
    v = SDPX.variable!(model, :v, 3; domain=SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Minimize(), v[1])
    SDPX.constraint!(model, :c, (v[1],), SDPX.Nonpositive())

    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)

    blocks = SDPX.layout_blocks(canon.cone_layout)
    @test length(blocks) == 2
    @test SDPX.block_cone(blocks[1]) === :nonnegative
    @test SDPX.block_cone(blocks[2]) === :nonnegative
    @test SDPX.layout_barrier_degree(canon.cone_layout) == 4

    # The variable-in-cone block carries a -1 reconstruction sign.
    rmap = SDPX.block_reconstruction(blocks[1])
    @test rmap.source === :variable
    @test rmap.sign == -1
    @test rmap.linear === nothing
    # The affine Nonpositive constraint block also maps to nonnegative.
    rmap2 = SDPX.block_reconstruction(blocks[2])
    @test rmap2.source === :constraint
    @test rmap2.sign == -1

    # Round-trip: forward(backward(y)) == y through the sign map.
    m = SDPX.canonical_num_slack(canon)
    y = rand(m)
    orig = zeros(m)
    back = zeros(m)
    SDPX.dual_forward!(canon, orig, y)
    SDPX.dual_backward!(canon, back, orig)
    @test back ≈ y atol=1e-12
end

@testset "RotatedLorentzCone maps exactly to SOC" begin
    model = SDPX.Model(Float64)
    v = SDPX.variable!(model, :v, 3; domain=SDPX.RotatedLorentzCone())
    f = SDPX.variable!(model, :f, 1; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), v[1] + f[1])
    SDPX.constraint!(model, :rc, (f[1], v[2], v[1]), SDPX.RotatedLorentzCone())

    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)

    blocks = SDPX.layout_blocks(canon.cone_layout)
    @test length(blocks) == 2
    @test all(SDPX.block_cone(b) === :soc for b in blocks)
    @test SDPX.layout_barrier_degree(canon.cone_layout) == 4

    rmap = SDPX.block_reconstruction(blocks[1])
    @test rmap.linear !== nothing
    M = rmap.linear
    @test rmap.linear_adjoint ≈ M
    @test M * M ≈ Matrix(I, 3, 3)            # M is an involution
    @test M ≈ transpose(M)                    # M is symmetric
    # (u,v,w) in RSOC  <=>  M(u,v,w) in SOC
    rsoc_point = [4.0, 1.0, 0.5]              # 2uv = 8 >= 0.25
    msoc = M * rsoc_point
    @test msoc[1] >= norm(msoc[2:3])

    # Round-trip through the exact map.
    y = rand(6)
    orig = zeros(6); back = zeros(6)
    SDPX.dual_forward!(canon, orig, y)
    SDPX.dual_backward!(canon, back, orig)
    @test back ≈ y atol=1e-12
end

@testset "rectangular n != m program" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    s = SDPX.variable!(model, :s, 2; domain=SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    SDPX.constraint!(model, :c, (x[1], x[1]), SDPX.Nonnegative())

    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)
    n = SDPX.canonical_num_variables(canon)
    m = SDPX.canonical_num_slack(canon)
    @test n != m
    @test size(SDPX.canonical_equality(canon)) == (m, n)
    @test length(SDPX.canonical_rhs(canon)) == m
    @test length(SDPX.canonical_objective(canon)) == n
end

@testset "mixed LP + SOC is a first-class canonical program" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + y[1])
    SDPX.constraint!(model, :soc, (y[1], y[2], y[3]), SDPX.LorentzCone())

    ncp = SDPX.compile_product_cone_model(model)
    @test SDPX.classify_native_cone_program(ncp).route === :mixed_family
    canon = SDPX.canonicalize(ncp)
    blocks = SDPX.layout_blocks(canon.cone_layout)
    cones = [SDPX.block_cone(b) for b in blocks]
    @test (:nonnegative in cones) && (:soc in cones)
    @test SDPX.canonical_num_slack(canon) == 2 + 3
    @test SDPX.layout_barrier_degree(canon.cone_layout) == 2 + 2
end

@testset "mixed SOC + PSD is a first-class canonical program" begin
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())   # PSD 2x2
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    SDPX.constraint!(model, :c, (y[1], y[2], y[3]), SDPX.LorentzCone())

    ncp = SDPX.compile_product_cone_model(model)
    @test SDPX.classify_native_cone_program(ncp).route === :mixed_family
    canon = SDPX.canonicalize(ncp)
    blocks = SDPX.layout_blocks(canon.cone_layout)
    cones = [SDPX.block_cone(b) for b in blocks]
    @test (:soc in cones) && (:psd in cones)
    # PSD 2x2 -> packed slack length 3.
    @test SDPX.canonical_num_slack(canon) == 3 + 3
    @test SDPX.layout_barrier_degree(canon.cone_layout) == 2 + 2
end

@testset "Maximize negates canonical objective and records sign" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Maximize(), x[1])
    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)
    @test SDPX.canonical_objective(canon) == [-1.0]
    chain = SDPX.canonical_reconstruction_chain(canon)
    @test chain.objective_sign == -1
    @test chain.objective_constant == 0.0
end

@testset "primal forward/backward reconstruct original coordinates" begin
    model = SDPX.Model(Float64)
    v = SDPX.variable!(model, :v, 2; domain=SDPX.Nonnegative())
    f = SDPX.variable!(model, :f, 2; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), v[1] + f[1])
    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)

    n = SDPX.canonical_num_variables(canon)
    m = SDPX.canonical_num_slack(canon)
    @test m == 2
    x = rand(n)
    xc = zeros(n); sc = zeros(m)
    xo = zeros(n); so = zeros(m)
    SDPX.primal_backward!(canon, xc, sc, x)
    SDPX.primal_forward!(canon, xo, so, xc, sc)
    @test xo == x
    # canonical slack reconstruction returns the original variable slack
    # for the variable-in-cone block (s = v here).
    @test so[1:2] ≈ x[1:2] atol=1e-12
end

@testset "PowerCone parameters preserve source precision in the canonical program" begin
    model = SDPX.Model(BigFloat; precision_bits=128)
    p = SDPX.variable!(model, :p, 3; domain=SDPX.PowerCone(0.25))
    SDPX.objective!(model, SDPX.Minimize(), p[1])
    SDPX.constraint!(model, :pc, (p[1], p[2], p[3]), SDPX.PowerCone(0.5))
    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)
    @test eltype(canon.c) === BigFloat
    blocks = SDPX.layout_blocks(canon.cone_layout)
    @test SDPX.block_cone(blocks[1]) === :power
    @test SDPX.block_cone(blocks[2]) === :power
    @test SDPX.block_parameter(blocks[1]) isa BigFloat
    @test SDPX.block_parameter(blocks[1]) ≈ 0.25
    @test SDPX.block_parameter(blocks[2]) isa BigFloat
    @test SDPX.block_parameter(blocks[2]) ≈ 0.5
end

@testset "PowerCone source alpha converts only at the canonical boundary" begin
    decimal = "0.123456789012345678901234567890123456789"
    source_alpha = BigFloat(decimal; precision=1024)
    domain = SDPX.PowerCone(source_alpha)

    # The source-domain object and native IR retain the caller's arithmetic;
    # a Float64 model does not silently round alpha while registering it.
    float_model = SDPX.Model(Float64)
    SDPX.variable!(float_model, :p, 3; domain=domain)
    float_ncp = SDPX.compile_product_cone_model(float_model)
    @test typeof(float_ncp.blocks[1].domain.alpha) === BigFloat
    @test precision(float_ncp.blocks[1].domain.alpha) == 1024
    float_canonical = SDPX.canonicalize(float_ncp)
    float_parameter = SDPX.block_parameter(
        SDPX.layout_blocks(float_canonical.cone_layout)[1],
    )
    @test typeof(float_parameter) === Float64
    @test float_parameter == Float64(source_alpha)

    # BigFloat canonicalization owns the result at the model precision,
    # independent of the source alpha's larger precision.
    big_model = SDPX.Model(BigFloat; precision_bits=256)
    SDPX.variable!(big_model, :p, 3; domain=domain)
    big_ncp = SDPX.compile_product_cone_model(big_model)
    @test precision(big_ncp.blocks[1].domain.alpha) == 1024
    big_canonical = SDPX.canonicalize(big_ncp)
    big_parameter = SDPX.block_parameter(
        SDPX.layout_blocks(big_canonical.cone_layout)[1],
    )
    @test typeof(big_parameter) === BigFloat
    @test precision(big_parameter) == 256
    @test big_parameter == BigFloat(source_alpha; precision=256)
end

@testset "PowerCone preserves Float64, MultiFloat, and BigFloat parameters" begin
    function check_power_parameter(::Type{T}, alpha) where {T<:AbstractFloat}
        domain = SDPX.PowerCone(alpha)
        @test typeof(domain.alpha) === typeof(alpha)
        model = T === BigFloat ?
            SDPX.Model(BigFloat; precision_bits=precision(alpha)) : SDPX.Model(T)
        p = SDPX.variable!(model, :p, 3; domain=domain)
        SDPX.constraint!(model, :pc, (p[1], p[2], p[3]), domain)
        ncp = SDPX.compile_product_cone_model(model)
        @test typeof(ncp.blocks[1].domain.alpha) === typeof(alpha)
        canonical = SDPX.canonicalize(ncp)
        blocks = SDPX.layout_blocks(canonical.cone_layout)
        @test length(blocks) == 2
        for block in blocks
            parameter = SDPX.block_parameter(block)
            @test typeof(parameter) === T
            @test parameter == T(alpha)
        end
    end

    check_power_parameter(Float64, 0.25)

    # MultiFloats is an optional dependency. If present, exercise each of the
    # supported Float64xN arithmetic types without converting alpha to Float64.
    if Base.find_package("MultiFloats") !== nothing
        @eval using MultiFloats
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4)
            check_power_parameter(T, T("0.25"))
        end
    end

    decimal = "0.123456789012345678901234567890123456789"
    for bits in (256, 512, 1024)
        alpha = BigFloat(decimal; precision=bits)
        setprecision(BigFloat, bits) do
            check_power_parameter(BigFloat, alpha)
        end
        @test precision(alpha) == bits
    end
end

@testset "PowerCone validates alpha and fixed three-dimensional shapes" begin
    for alpha in (0.0, 1.0, -0.25, 1.25, Inf, NaN)
        @test_throws ArgumentError SDPX.PowerCone(alpha)
    end
    @test_throws ArgumentError SDPX.PowerCone(BigFloat("NaN"))

    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 3; domain=SDPX.Reals())
    @test_throws ArgumentError SDPX.variable!(model, :bad_exp, 2; domain=SDPX.ExponentialCone())
    @test_throws ArgumentError SDPX.variable!(model, :bad_power, 2; domain=SDPX.PowerCone(0.5))
    @test_throws ArgumentError SDPX.constraint!(
        model, :bad_exp_constraint, (x[1], x[2]), SDPX.ExponentialCone(),
    )
    @test_throws ArgumentError SDPX.constraint!(
        model, :bad_power_constraint, (x[1], x[2]), SDPX.PowerCone(0.5),
    )
    @test_throws ArgumentError SDPX.NativeBlock(SDPX.ExponentialCone(), 2, 1)
    @test_throws ArgumentError SDPX.NativeBlock(SDPX.PowerCone(0.5), 2, 1)
    @test_throws ArgumentError SDPX.RowBlock(SDPX.ExponentialCone(), 1, 2)
    @test_throws ArgumentError SDPX.RowBlock(SDPX.PowerCone(0.5), 1, 2)
    @test_throws ArgumentError SDPX.ConeBlockDescriptor(
        Float64, :exp, 2,
    )
    @test_throws ArgumentError SDPX.ConeBlockDescriptor(
        Float64, :power, 2; parameter=0.5,
    )
end

@testset "PowerCone canonical conversion rejects rounded endpoints" begin
    tiny = BigFloat("1e-1000"; precision=256)
    almost_one = prevfloat(BigFloat(1; precision=256))
    for alpha in (tiny, almost_one)
        variable_model = SDPX.Model(Float64)
        SDPX.variable!(
            variable_model, :p, 3; domain=SDPX.PowerCone(alpha),
        )
        @test_throws ArgumentError SDPX.canonicalize(
            SDPX.compile_product_cone_model(variable_model),
        )

        row_model = SDPX.Model(Float64)
        x = SDPX.variable!(row_model, :x, 3; domain=SDPX.Reals())
        SDPX.constraint!(
            row_model, :p, (x[1], x[2], x[3]), SDPX.PowerCone(alpha),
        )
        @test_throws ArgumentError SDPX.canonicalize(
            SDPX.compile_product_cone_model(row_model),
        )
    end
end
