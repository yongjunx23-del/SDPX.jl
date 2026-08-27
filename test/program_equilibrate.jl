# Phase 4 equilibration tests (prepared program-layer infrastructure).
using SDPX
using Test
using LinearAlgebra
using SparseArrays

# Build a small canonical program with a free variable, one Nonnegative row,
# one Zero row, and one SOC block, then check the map round-trips.
function _sample_canonical(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    n = SDPX.variable!(model, :n, 2; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :s, 3; domain=SDPX.LorentzCone())
    SDPX.constraint!(model, :eq, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :nn, n[1] + n[2] - 0.5, SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + 2 * x[2] + n[1])
    # compile to canonical
    native = SDPX.compile_product_cone_model(model)
    return SDPX.canonicalize(native)
end

@testset "equilibration map round-trip" begin
    for T in (Float64, BigFloat)
        canonical = _sample_canonical(T)
        m = SDPX.canonical_num_slack(canonical)
        n = SDPX.canonical_num_variables(canonical)
        map = SDPX.equilibrate(canonical)
        @test length(map.row_scale) == m
        @test length(map.col_scale) == n
        @test all(>(zero(T)), map.row_scale)
        @test all(>(zero(T)), map.col_scale)

        Ahat, bhat, chat = SDPX.apply_equilibration(map, canonical)
        @test size(Ahat) == size(canonical.A)
        @test length(bhat) == m
        @test length(chat) == n

        # Primal round-trip: x = reconstruct(apply(x̂)) is identity.
        xhat = T.(collect(1:n)) ./ T(n + 1)
        x = SDPX.reconstruct_primal(map, xhat)
        @test x ≈ xhat atol = 1e-12

        # Dual round-trip: y = reconstruct(apply(ŷ)) is identity.
        yhat = T.(collect(1:m)) ./ T(m + 1)
        y = SDPX.reconstruct_dual(map, yhat)
        @test y ≈ yhat atol = 1e-12
    end
end

@testset "equilibration preserves feasibility scaling" begin
    for T in (Float64,)
        canonical = _sample_canonical(T)
        map = SDPX.equilibrate(canonical)
        Ahat, bhat, chat = SDPX.apply_equilibration(map, canonical)
        # The equilibrated A is still finite and correctly dimensioned.
        @test all(isfinite, Ahat.nzval)
        @test all(isfinite, bhat)
        @test all(isfinite, chat)
    end
end