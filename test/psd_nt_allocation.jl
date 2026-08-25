using SDPX
using Test
import MultiFloats

const NTALLOC_SC = SDPX.SymmetricCones

function _ntalloc_ten(f)
    f()
    f()
    f()
    return ntuple(_ -> @allocated(f()), 10)
end

function _ntalloc_gate(label::AbstractString, f)
    samples = _ntalloc_ten(f)
    @testset "$label: 10/10 zero allocation" begin
        @test samples == ntuple(_ -> 0, 10)
    end
    return nothing
end

function _ntalloc_orthant(::Type{T}) where {T}
    cone = NTALLOC_SC.NonnegativeCone(8)
    state = NTALLOC_SC.OrthantNTScaling{T}(8)
    s = T[T(i + 2) for i in 1:8]
    y = T[T(i + 3) / T(2) for i in 1:8]
    z = T[T(i) / T(5) for i in 1:8]
    out = zeros(T, 8)
    _ntalloc_gate("orthant $T update", () -> NTALLOC_SC.nt_scaling!(cone, state, s, y))
    _ntalloc_gate("orthant $T Theta", () -> NTALLOC_SC.theta_apply!(cone, out, state, z))
    _ntalloc_gate("orthant $T G", () -> NTALLOC_SC.g_apply!(cone, out, state, z))
    _ntalloc_gate("orthant $T R", () -> NTALLOC_SC.r_apply!(cone, out, state, z))
    _ntalloc_gate("orthant $T R inverse", () -> NTALLOC_SC.r_inverse_apply!(cone, out, state, z))
    _ntalloc_gate("orthant $T solve Llambda", () -> NTALLOC_SC.solve_Llambda!(cone, out, state, z))
    _ntalloc_gate("orthant $T third order", () -> NTALLOC_SC.third_order_correction!(cone, out, s, y, z))
    return nothing
end

function _ntalloc_soc(::Type{T}) where {T}
    n = 8
    cone = NTALLOC_SC.SOCone(n)
    state = NTALLOC_SC.SOCNTScaling{T}(n)
    s = zeros(T, n)
    y = zeros(T, n)
    z = zeros(T, n)
    s[1], y[1], z[1] = T(7), T(6), T(2)
    @inbounds for i in 2:n
        s[i] = T(i) / T(40)
        y[i] = -T(i) / T(50)
        z[i] = T(i) / T(60)
    end
    out = zeros(T, n)
    _ntalloc_gate("SOC $T update", () -> NTALLOC_SC.nt_scaling!(cone, state, s, y))
    _ntalloc_gate("SOC $T Theta", () -> NTALLOC_SC.theta_apply!(cone, out, state, z))
    _ntalloc_gate("SOC $T G", () -> NTALLOC_SC.g_apply!(cone, out, state, z))
    _ntalloc_gate("SOC $T R", () -> NTALLOC_SC.r_apply!(cone, out, state, z))
    _ntalloc_gate("SOC $T R inverse", () -> NTALLOC_SC.r_inverse_apply!(cone, out, state, z))
    _ntalloc_gate("SOC $T solve Llambda", () -> NTALLOC_SC.solve_Llambda!(cone, out, state, z))
    _ntalloc_gate("SOC $T quadratic", () -> NTALLOC_SC.quadratic_apply!(cone, out, state.w, z))
    _ntalloc_gate("SOC $T spectral basis", () -> NTALLOC_SC.spectral_basis!(cone, state, z))
    # This is deliberately the state overload: its scratch is part of the NT state.
    _ntalloc_gate("SOC $T state third order", () -> NTALLOC_SC.third_order_correction!(cone, state, out, s, y, z))
    return nothing
end

function _ntalloc_svec2(a11::T, a21::T, a22::T) where {T}
    return T[a11, sqrt(T(2)) * a21, a22]
end

function _ntalloc_psd(::Type{T}) where {T}
    cone = NTALLOC_SC.PSDTriangleCone{T}(2)
    state = NTALLOC_SC.PSDNTScaling{T}(2)
    s = _ntalloc_svec2(T(4), T(1) / T(5), T(3))
    y = _ntalloc_svec2(T(3), -T(1) / T(7), T(5))
    z = _ntalloc_svec2(T(2), T(1) / T(9), T(4))
    out = zeros(T, 3)
    _ntalloc_gate("PSD $T update", () -> NTALLOC_SC.nt_scaling!(cone, state, s, y))
    _ntalloc_gate("PSD $T Theta", () -> NTALLOC_SC.theta_apply!(cone, out, state, z))
    _ntalloc_gate("PSD $T G", () -> NTALLOC_SC.g_apply!(cone, out, state, z))
    _ntalloc_gate("PSD $T R", () -> NTALLOC_SC.r_apply!(cone, out, state, z))
    _ntalloc_gate("PSD $T R inverse", () -> NTALLOC_SC.r_inverse_apply!(cone, out, state, z))
    _ntalloc_gate("PSD $T solve Llambda", () -> NTALLOC_SC.solve_Llambda!(cone, out, state, z))
    # This is deliberately the state overload: it must not allocate a temporary.
    _ntalloc_gate("PSD $T state third order", () -> NTALLOC_SC.third_order_correction!(cone, state, out, s, y, z))
    return nothing
end

@testset "pair-dependent symmetric-cone NT allocation gates" begin
    for T in (
        Float64,
        MultiFloats.Float64x2,
        MultiFloats.Float64x3,
        MultiFloats.Float64x4,
    )
        @testset "$T" begin
            _ntalloc_orthant(T)
            _ntalloc_soc(T)
            _ntalloc_psd(T)
        end
    end
end
