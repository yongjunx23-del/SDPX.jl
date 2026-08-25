# Mixed symmetric-cone programs (LP+SOC, SOC+PSD, LP+PSD) solved through the
# universal PSD lifting (Subagent PSDLIFT).
#
# Every mixed native family is lifted to a single block-diagonal PSD cone and
# solved through the existing `algorithm=:sdp` path. These tests verify
# (a) the `lift_to_psd` canonical IR transformation, (b) that each mixed
# family reaches `Optimal` with a VALID original-coordinate certificate and
# the correct optimum, and (c) that pure SOC / pure PSD still solve (no
# regression).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra

@inline _mixed_outputs() = SDPX.Outputs(:all, :all, :all)

# ---------------------------------------------------------------------------
# lift_to_psd: the canonical-level universal PSD lifting
# ---------------------------------------------------------------------------

@testset "lift_to_psd maps mixed canonical to PSD-only layout" begin
    # LP + SOC
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + y[1])
    SDPX.constraint!(model, :soc, (y[1], y[2], y[3]), SDPX.LorentzCone())
    canon = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    lifted = SDPX.lift_to_psd(canon)
    blocks = SDPX.layout_blocks(lifted.cone_layout)
    cones = [SDPX.block_cone(b) for b in blocks]
    @test all(cone -> cone === :psd || cone === :zero, cones)
    # nonnegative(2) -> PSD(2) (3 packed); soc(3) -> PSD(3) (6 packed);
    # 5 original slack rows -> zero rows.
    @test count(==(:psd), cones) == 2
    @test count(==(:zero), cones) == 5
    @test SDPX.canonical_num_slack(lifted) == 3 + 6 + 5
    # variables become [x; s]: n + m.
    @test SDPX.canonical_num_variables(lifted) ==
          SDPX.canonical_num_variables(canon) + SDPX.canonical_num_slack(canon)
    @test SDPX.canonical_objective(lifted)[1:SDPX.canonical_num_variables(canon)] ==
          SDPX.canonical_objective(canon)
end

@testset "lift_to_psd SOC+PSD and LP+PSD" begin
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    SDPX.constraint!(model, :soc, (y[1], y[2], y[3]), SDPX.LorentzCone())
    canon = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    lifted = SDPX.lift_to_psd(canon)
    blocks = SDPX.layout_blocks(lifted.cone_layout)
    cones = [SDPX.block_cone(b) for b in blocks]
    @test count(cone -> cone === :psd, cones) == 2
    @test SDPX.canonical_num_slack(lifted) == 3 + 6 + 6

    model2 = SDPX.Model(Float64)
    a = SDPX.variable!(model2, :a, 2; domain=SDPX.Nonnegative())
    M = SDPX.variable!(model2, :M, 2, 2; domain=SDPX.PSDCone())
    SDPX.objective!(model2, SDPX.Minimize(), a[1])
    canon2 = SDPX.canonicalize(SDPX.compile_product_cone_model(model2))
    lifted2 = SDPX.lift_to_psd(canon2)
    cones2 = [SDPX.block_cone(b) for b in SDPX.layout_blocks(lifted2.cone_layout)]
    @test count(cone -> cone === :psd, cones2) == 2
end

# ---------------------------------------------------------------------------
# Mixed-family solves (optimum + valid certificate + reconstruction)
# ---------------------------------------------------------------------------

@testset "mixed LP+SOC solves via PSD lift" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + x[2] + y[1])
    SDPX.constraint!(model, :soc, (y[1], y[2], y[3]), SDPX.LorentzCone())
    SDPX.constraint!(model, :c1, (x[1] - 1.0,), SDPX.Nonnegative())
    SDPX.constraint!(model, :c2, (x[2] - 2.0,), SDPX.Nonnegative())
    result = SDPX.optimize!(model; outputs=_mixed_outputs())
    @test SDPX.status(result) === :optimal
    @test result.certificate.available && result.certificate.valid
    @test isapprox(Float64(SDPX.primal_objective(result)), 3.0; atol=1e-6)
    @test isapprox(Float64(SDPX.value(result, x[1])), 1.0; atol=1e-6)
    @test isapprox(Float64(SDPX.value(result, x[2])), 2.0; atol=1e-6)
    @test isapprox(Float64(SDPX.value(result, y[1])), 0.0; atol=1e-6)
end

@testset "mixed SOC+PSD solves via PSD lift" begin
    model = SDPX.Model(Float64)
    M = SDPX.variable!(model, :M, 2, 2; domain=SDPX.PSDCone())
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), M[1, 1] + M[2, 2] + y[1])
    SDPX.constraint!(model, :soc, (y[1], y[2], y[3]), SDPX.LorentzCone())
    SDPX.constraint!(model, :c1, (M[1, 1] - 2.0,), SDPX.Nonnegative())
    SDPX.constraint!(model, :c2, (M[2, 2] - 2.0,), SDPX.Nonnegative())
    result = SDPX.optimize!(model; outputs=_mixed_outputs())
    @test SDPX.status(result) === :optimal
    @test result.certificate.available && result.certificate.valid
    @test isapprox(Float64(SDPX.primal_objective(result)), 4.0; atol=1e-6)
    # reconstructed PSD variable: M[1,1] -> 2, M[2,2] -> 2
    @test isapprox(Float64(SDPX.value(result, M[1, 1])), 2.0; atol=1e-6)
    @test isapprox(Float64(SDPX.value(result, M[2, 2])), 2.0; atol=1e-6)
    @test isapprox(Float64(SDPX.value(result, y[1])), 0.0; atol=1e-6)
end

@testset "mixed LP+PSD solves via PSD lift" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    M = SDPX.variable!(model, :M, 2, 2; domain=SDPX.PSDCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + x[2] + M[1, 1])
    SDPX.constraint!(model, :c1, (x[1] - 1.0,), SDPX.Nonnegative())
    SDPX.constraint!(model, :c2, (x[2] - 1.0,), SDPX.Nonnegative())
    SDPX.constraint!(model, :c3, (M[1, 1] - 3.0,), SDPX.Nonnegative())
    result = SDPX.optimize!(model; outputs=_mixed_outputs())
    @test SDPX.status(result) === :optimal
    @test result.certificate.available && result.certificate.valid
    @test isapprox(Float64(SDPX.primal_objective(result)), 5.0; atol=1e-6)
    @test isapprox(Float64(SDPX.value(result, M[1, 1])), 3.0; atol=1e-6)
end

@testset "mixed product-SOC + affine nonnegative" begin
    # Exercise a SOC *product* block (lifted from a variable-in-cone) and an
    # affine nonnegative row.  min -y1 s.t. (y1,y2) in SOC (y1>=|y2|), plus
    # an affine row forcing y1 <= 2.
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Maximize(), y[1])
    SDPX.constraint!(model, :cap, (2.0 - y[1],), SDPX.Nonnegative())
    result = SDPX.optimize!(model; outputs=_mixed_outputs())
    @test SDPX.status(result) === :optimal
    @test result.certificate.available && result.certificate.valid
    # maximize y1 subject to y1 <= 2 and y1 >= |y2| -> optimum 2.
    @test isapprox(Float64(SDPX.primal_objective(result)), 2.0; atol=1e-6)
    @test isapprox(Float64(SDPX.value(result, y[1])), 2.0; atol=1e-6)
end

# ---------------------------------------------------------------------------
# Regression: pure SOC and pure PSD still solve through their own routes
# ---------------------------------------------------------------------------

@testset "pure SOC and PSD routes still solve" begin
    # Pure SOC (with an equality zero row, still SOC family)
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), y[1])
    SDPX.constraint!(model, :soc, (y[1], y[2], y[3]), SDPX.LorentzCone())
    SDPX.constraint!(model, :eq, (y[1] - 3.0,), SDPX.ZeroCone())
    r = SDPX.optimize!(model; outputs=_mixed_outputs())
    @test SDPX.classify_native_cone_program(
        SDPX.compile_product_cone_model(model)).route === :soc_family
    @test SDPX.status(r) === :optimal
    @test isapprox(Float64(SDPX.primal_objective(r)), 3.0; atol=1e-6)

    # Pure PSD (product PSD block + affine PSD row M - I ⪰ 0)
    model2 = SDPX.Model(Float64)
    M = SDPX.variable!(model2, :M, 2, 2; domain=SDPX.PSDCone())
    SDPX.objective!(model2, SDPX.Minimize(), M[1, 1] + M[2, 2])
    SDPX.constraint!(model2, :c1,
        [M[1, 1] - 1.0 M[1, 2]; M[1, 2] M[2, 2] - 1.0], SDPX.PSDCone())
    r2 = SDPX.optimize!(model2; outputs=_mixed_outputs())
    @test SDPX.classify_native_cone_program(
        SDPX.compile_product_cone_model(model2)).route === :sdp_family
    @test SDPX.status(r2) === :optimal
    @test isapprox(Float64(SDPX.primal_objective(r2)), 2.0; atol=1e-6)
end
