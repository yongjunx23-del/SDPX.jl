# hsd_border_failure.jl -- deterministic fail-closed LP-HSD border injection.

using SDPX
using Test
using SparseArrays
using MultiFloats

function _p0b_border_canonical(::Type{T}, A, b, c) where {T<:AbstractFloat}
    m, _ = size(A)
    descriptor = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
    layout = SDPX.canonical_layout([descriptor])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1,
        zero(T),
        SDPX.VariableRef[],
        SDPX.ConstraintRef[],
        SDPX.VariableRef[],
        0,
    )
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(c), sparse(A), Vector{T}(b),
        layout, chain,
    )
end

function _p0b_border_state(::Type{T}) where {T<:AbstractFloat}
    A = reshape(T[1], 1, 1)
    canonical = _p0b_border_canonical(T, A, T[0], T[1])
    state = SDPX.HSDState(canonical)
    fill!(state.x, zero(T))
    fill!(state.s, one(T))
    fill!(state.y, one(T))
    state.tau = one(T)
    state.kappa = one(T)
    return state
end

function _p0b_all_directions_finite(state)
    return all(isfinite, state.dx) && all(isfinite, state.dy) &&
           all(isfinite, state.ds) && isfinite(state.dtau) &&
           isfinite(state.dkappa) && all(isfinite, state.dx_a) &&
           all(isfinite, state.dy_a) && all(isfinite, state.ds_a) &&
           isfinite(state.dtau_a) && isfinite(state.dkappa_a) &&
           all(isfinite, state.dxr)
end

function _p0b_iterate_snapshot(state)
    return (
        x=copy(state.x),
        s=copy(state.s),
        y=copy(state.y),
        tau=state.tau,
        kappa=state.kappa,
    )
end

function _p0b_iterate_unchanged(state, snapshot)
    return state.x == snapshot.x && state.s == snapshot.s &&
           state.y == snapshot.y && state.tau == snapshot.tau &&
           state.kappa == snapshot.kappa
end

@inline function _p0b_border_call_noreturn!(
    output::Vector{Tuple{Bool,T}},
    state::SDPX.HSDState{T},
    d::T,
    rho::T,
) where {T}
    output[1] = SDPX._hsd_border_solve!(state, d, rho, state.dxr)
    return nothing
end

@testset "P0-B unsafe border fails closed without NaN" begin
    @testset "exact full-step denominator fault" begin
        # At this point G=H=q=r=d=u=1, hence delta=d-r'u=0 exactly.
        # The predictor numerator is 2, so this is not a benign 0/0 case.
        # The native product caller classifies the singular full border:
        # the base HSDState driver is never factored (the bordered route owns
        # its own pivoted-LU driver), and one numeric factor attempt is
        # observed on the product route before the typed failure.
        base_state = _p0b_border_state(Float64)
        state = SDPX.ProductConeHSDState(base_state.canonical)
        SDPX.product_hsd_cold_start!(state)
        base = state.base
        snapshot = _p0b_iterate_snapshot(base)
        @test SDPX.product_hsd_factor_count(state) == 0
        @test SDPX.kkt_factor_count(base.driver) == 0

        code = SDPX.product_hsd_step!(state)

        @test code === SDPX.HSDStepSingularKKT
        @test SDPX.product_hsd_factor_count(state) == 1
        @test SDPX.kkt_factor_count(base.driver) == 0
        @test base.record.iterations == 0
        @test base.record.step_size == 0.0
        @test _p0b_iterate_unchanged(base, snapshot)
        @test _p0b_all_directions_finite(base)
        @test base.dtau == 0.0
        @test all(iszero, base.dxr)
    end

    @testset "product caller classifies the singular full border" begin
        base_state = _p0b_border_state(Float64)
        product = SDPX.ProductConeHSDState(base_state.canonical)
        SDPX.product_hsd_cold_start!(product)
        base = product.base
        snapshot = _p0b_iterate_snapshot(base)

        code = SDPX.product_hsd_step!(product)

        @test code === SDPX.HSDStepSingularKKT
        @test SDPX.product_hsd_factor_count(product) == 1
        @test SDPX.kkt_factor_count(base.driver) == 0
        @test base.record.iterations == 0
        @test _p0b_iterate_unchanged(base, snapshot)
        @test _p0b_all_directions_finite(base)
        @test base.dtau == 0.0
    end

    @testset "low-level cancellation, nonfinite, and normal gates" begin
        state = _p0b_border_state(Float64)

        # Huge equal terms form a zero denominator.  The scale gate must use
        # the terms being subtracted, not the already-cancelled result.
        state.rvec[1] = 1.0e300
        state.u[1] = 1.0
        state.w[1] = 0.0
        state.dxr[1] = 7.0
        ok, dtau = SDPX._hsd_border_solve!(state, 1.0e300, 1.0, state.dxr)
        @test !ok
        @test dtau == 0.0
        @test state.dxr == [7.0]

        # Neither a non-finite scalar input nor a non-finite dot-product input
        # may poison the returned scalar or partially overwrite dx.
        for (d, rho, rvalue) in (
            (2.0, Inf, 1.0),
            (Inf, 1.0, 1.0),
            (2.0, 1.0, Inf),
        )
            state.rvec[1] = rvalue
            state.u[1] = 0.5
            state.w[1] = 0.25
            state.dxr[1] = 7.0
            ok, dtau = SDPX._hsd_border_solve!(state, d, rho, state.dxr)
            @test !ok
            @test isfinite(dtau) && iszero(dtau)
            @test state.dxr == [7.0]
        end

        # A finite quotient whose reconstructed dx would overflow is rejected
        # before the destination is touched.
        state.rvec[1] = 0.0
        state.u[1] = 2.0
        state.w[1] = floatmax(Float64)
        state.dxr[1] = 7.0
        ok, dtau = SDPX._hsd_border_solve!(
            state, 2.0, -floatmax(Float64), state.dxr,
        )
        @test !ok
        @test isfinite(dtau) && iszero(dtau)
        @test state.dxr == [7.0]

        # A normal, comfortably separated denominator still takes the original
        # algebraic path: delta=3/2, numerator=11/4, dtau=11/6, dx=-2/3.
        state.rvec[1] = 1.0
        state.u[1] = 0.5
        state.w[1] = 0.25
        state.dxr[1] = 7.0
        ok, dtau = SDPX._hsd_border_solve!(state, 2.0, 3.0, state.dxr)
        @test ok
        @test dtau == 11 / 6
        @test state.dxr[1] == -2 / 3
    end

    @testset "fixed-width successful border remains allocation-free" begin
        for T in (Float64, Float64x2, Float64x3, Float64x4)
            state = _p0b_border_state(T)
            state.rvec[1] = one(T)
            state.u[1] = one(T) / T(2)
            state.w[1] = one(T) / T(4)
            output = Vector{Tuple{Bool,T}}(undef, 1)
            _p0b_border_call_noreturn!(output, state, T(2), T(3))
            @test output[1][1]
            @test output[1] isa Tuple{Bool,T}

            bytes = @allocated _p0b_border_call_noreturn!(
                output, state, T(2), T(3),
            )
            @test bytes == 0
            @test output[1][1]
            @test isfinite(output[1][2])
        end
    end
end
