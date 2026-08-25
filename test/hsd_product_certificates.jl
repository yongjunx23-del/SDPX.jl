# Original-coordinate certificate gates for a mixed orthant/RSOC/PSD product.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using SparseArrays
using LinearAlgebra

function _mixed_certificate_layout(::Type{T}) where {T<:AbstractFloat}
    bits = T === BigFloat ? precision(BigFloat) : precision(T)
    orthant = SDPX.ConeBlockDescriptor(T, :nonnegative, 1; offset=1)

    rsoc_map = SDPX._rsoc_to_soc_map(T, 3, bits)
    rsoc_reconstruction = SDPX.CanonicalBlockMap(
        :constraint, 1, 1, 1;
        linear=rsoc_map,
        linear_adjoint=rsoc_map,
    )
    rsoc = SDPX.ConeBlockDescriptor(
        T, :soc, 3; offset=2, reconstruction=rsoc_reconstruction,
    )

    psd_map = SDPX.PSDCoordinateMap(T, 2; precision_bits=bits)
    psd_reconstruction = SDPX.CanonicalBlockMap(
        :constraint, 2, 1, 1; coordinate_map=psd_map,
    )
    psd = SDPX.ConeBlockDescriptor(
        T, :psd, 2; offset=5, reconstruction=psd_reconstruction,
    )
    return SDPX.canonical_layout([orthant, rsoc, psd])
end

function _mixed_certificate_pair(::Type{T}) where {T<:AbstractFloat}
    root2 = sqrt(T(2))
    # Orthant: (s,y)=(1,0).  SOC: (1,+1,0) and (1,-1,0).
    # PSD: vv' and ww' for v=(1,1), w=(1,-1).  All pairs are in their
    # respective self-dual cones and have exactly zero trace pairing.
    s = T[1, 1, 1, 0, 1, root2, 1]
    y = T[0, 1, -1, 0, 1, -root2, 1]
    return s, y
end

function _mixed_certificate_program(layout, A, b, c, ::Type{T}) where {T<:AbstractFloat}
    bits = T === BigFloat ? precision(BigFloat) : precision(T)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(c), sparse(T.(A)), Vector{T}(b),
        layout, chain,
    )
end

function _exercise_mixed_certificates(::Type{T}) where {T<:AbstractFloat}
    layout = _mixed_certificate_layout(T)
    s, y = _mixed_certificate_pair(T)
    m = length(s)
    tolerance = T === Float64 ? T(1e-12) : parse(T, "1e-40")

    # Exact optimal certificate with no free variables.  The original-space
    # recovery must undo both the RSOC map and the PSD svec coordinate map.
    optimal = _mixed_certificate_program(
        layout, zeros(T, m, 0), s, T[], T,
    )
    state = SDPX.HSDState(optimal)
    copyto!(state.s, s)
    copyto!(state.y, y)
    state.tau = one(T)
    state.kappa = zero(T)
    x_original = T[]
    s_original = zeros(T, m)
    y_original = zeros(T, m)
    @test SDPX.verify_optimal!(
        optimal, state, x_original, s_original, y_original; tol=tolerance,
    )
    root2 = sqrt(T(2))
    @test s_original ≈ T[1, root2, 0, 0, 1, 1, 1]
    @test y_original ≈ T[0, 0, root2, 0, 1, -2, 1]

    # A primal-infeasibility ray is normalized and then reconstructed through
    # the same mixed block maps.  No orthant-only projected-gradient helper is
    # involved in this direct certificate gate.
    ray_norm2 = dot(y, y)
    primal_infeasible = _mixed_certificate_program(
        layout, zeros(T, m, 0), -y / ray_norm2, T[], T,
    )
    primal_state = SDPX.HSDState(primal_infeasible)
    copyto!(primal_state.y, y)
    y_ray_original = zeros(T, m)
    @test SDPX.verify_primal_infeasibility!(
        primal_infeasible, primal_state, y_ray_original; tol=tolerance,
    )
    @test dot(primal_infeasible.b, primal_state.yt) ≈ -one(T)
    @test y_ray_original[6] ≈ -T(2)

    # A dual-infeasibility ray x=1 has -A*x=s in the complete product cone.
    dual_infeasible = _mixed_certificate_program(
        layout, reshape(-s, m, 1), zeros(T, m), T[-1], T,
    )
    dual_state = SDPX.HSDState(dual_infeasible)
    dual_state.x[1] = one(T)
    x_ray_original = zeros(T, 1)
    s_ray_original = zeros(T, m)
    @test SDPX.verify_dual_infeasibility!(
        dual_infeasible, dual_state, x_ray_original, s_ray_original;
        tol=tolerance,
    )
    @test x_ray_original == T[1]
    @test s_ray_original ≈ s_original

    # The same affine ray must fail when its PSD svec coordinate is changed
    # to an indefinite matrix.  This specifically exercises the svec-aware
    # certificate membership path.
    bad_A = copy(Matrix(dual_infeasible.A))
    bad_A[6, 1] = -T(3) * root2
    bad_program = _mixed_certificate_program(
        layout, bad_A, zeros(T, m), T[-1], T,
    )
    bad_state = SDPX.HSDState(bad_program)
    bad_state.x[1] = one(T)
    @test !SDPX.verify_dual_infeasibility!(
        bad_program, bad_state, x_ray_original, s_ray_original;
        tol=tolerance,
    )
end

@testset "mixed symmetric HSD certificates in original coordinates" begin
    _exercise_mixed_certificates(Float64)
    setprecision(BigFloat, 256) do
        _exercise_mixed_certificates(BigFloat)
    end
end
