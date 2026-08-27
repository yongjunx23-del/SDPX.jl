# Native LP/SOC/PSD product-cone HSD core (Phase 2, self-contained gate).

using SDPX
using Test
using LinearAlgebra
using SparseArrays

function _pch_layout(::Type{T}, specs) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for (kind, dim) in specs
        block = SDPX.ConeBlockDescriptor(T, kind, dim; offset=offset)
        push!(blocks, block)
        offset += block.length
    end
    return SDPX.canonical_layout(blocks)
end

function _pch_svec(X::AbstractMatrix{T}) where {T}
    n = size(X, 1)
    out = Vector{T}(undef, div(n * (n + 1), 2))
    root2 = sqrt(T(2))
    k = 1
    @inbounds for j in 1:n, i in j:n
        out[k] = i == j ? X[i, j] : root2 * X[i, j]
        k += 1
    end
    return out
end

function _pch_unsvec(x::AbstractVector{T}, n::Int) where {T}
    X = zeros(T, n, n)
    invroot2 = one(T) / sqrt(T(2))
    k = 1
    @inbounds for j in 1:n, i in j:n
        value = i == j ? x[k] : invroot2 * x[k]
        X[i, j] = value
        X[j, i] = value
        k += 1
    end
    return X
end

function _pch_canonical(::Type{T}, specs) where {T<:AbstractFloat}
    layout = _pch_layout(T, specs)
    m = layout.dimension
    n = 2
    A = zeros(T, m, n)
    b = zeros(T, m)
    @inbounds for k in 1:m
        A[k, 1] = one(T) + T(k) / T(13)
        A[k, 2] = (isodd(k) ? -one(T) : one(T)) *
                  (T(2) / T(5) + T(k) / T(29))
        b[k] = (isodd(k) ? -one(T) : one(T)) * T(k + 2) / T(17)
    end
    c = T[T(7) / T(20), -T(11) / T(50)]
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    bits = T === BigFloat ? precision(BigFloat) : 53
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout, chain,
    )
end

function _pch_set_pair!(state::SDPX.ProductConeHSDState{T}) where {T}
    base = state.base
    fill!(base.s, zero(T))
    fill!(base.y, zero(T))
    for block in SDPX.layout_blocks(base.canonical.cone_layout)
        o = block.offset
        if block.cone === :nonnegative
            @inbounds for i in 1:block.length
                base.s[o + i - 1] = T(6 + i) / T(5)
                base.y[o + i - 1] = T(9 + 2i) / T(7)
            end
        elseif block.cone === :soc
            base.s[o] = T(5) / T(2)
            base.s[o + 1] = T(1) / T(5)
            base.s[o + 2] = -T(1) / T(10)
            base.y[o] = T(9) / T(5)
            base.y[o + 1] = -T(3) / T(20)
            base.y[o + 2] = T(3) / T(25)
        elseif block.cone === :psd
            n = block.dimension
            S = Matrix{T}(I, n, n)
            Y = Matrix{T}(I, n, n)
            @inbounds for i in 1:n
                S[i, i] = T(i + 2)
                Y[i, i] = T(2i + 3) / T(2)
            end
            if n >= 2
                S[1, 2] = S[2, 1] = T(3) / T(20)
                Y[1, 2] = Y[2, 1] = -T(1) / T(10)
            end
            sv = _pch_svec(S)
            yv = _pch_svec(Y)
            @inbounds for i in 1:block.length
                base.s[o + i - 1] = sv[i]
                base.y[o + i - 1] = yv[i]
            end
        end
    end
    base.x[1] = T(3) / T(20)
    base.x[2] = -T(2) / T(25)
    base.tau = T(11) / T(10)
    base.kappa = T(13) / T(10)
    return state
end

function _pch_state(::Type{T}, specs) where {T<:AbstractFloat}
    state = SDPX.ProductConeHSDState(_pch_canonical(T, specs))
    _pch_set_pair!(state)
    return state
end

function _pch_qmatrix(w::AbstractVector{T}) where {T}
    n = length(w)
    Q = zeros(T, n, n)
    tail2 = zero(T)
    @inbounds for i in 2:n
        tail2 += w[i] * w[i]
    end
    Q[1, 1] = w[1] * w[1] + tail2
    @inbounds for i in 2:n
        Q[1, i] = T(2) * w[1] * w[i]
        Q[i, 1] = Q[1, i]
        for j in 2:n
            Q[i, j] = T(2) * w[i] * w[j]
        end
        Q[i, i] += w[1] * w[1] - tail2
    end
    return Q
end

"""Independent dense Theta assembled from frozen NT data, never HSD assembly."""
function _pch_dense_theta(state::SDPX.ProductConeHSDState{T}) where {T}
    runtime = state.runtime
    m = state.base.m
    Theta = zeros(T, m, m)
    for block in runtime.orthant
        @inbounds for i in 1:block.dim
            Theta[block.offset + i - 1, block.offset + i - 1] = block.state.theta[i]
        end
    end
    for block in runtime.soc
        Q = _pch_qmatrix(block.state.w)
        r = block.offset:block.offset + block.dim - 1
        Theta[r, r] .= Q
    end
    for block in runtime.psd
        n = block.dim
        len = block.len
        r = block.offset:block.offset + len - 1
        for j in 1:len
            ej = zeros(T, len)
            ej[j] = one(T)
            Z = _pch_unsvec(ej, n)
            Theta[r, r[j]] .= _pch_svec(block.state.P * Z * block.state.P)
        end
    end
    return Theta
end

"""
Independent central target `-∇F(e)` in canonical dot coordinates.

For the Lorentz barrier `F(t,u)=-log(t^2-‖u‖^2)`, `ν_Q=2` and
`-∇F(e)=2e`.  Orthant and PSD/svec blocks have the usual identity target.
Thus `dot(e, target) == ν` for every block, including an RSOC after its
orthogonal canonical map to SOC.
"""
function _pch_central_target(::Type{T}, layout) where {T}
    e = zeros(T, layout.dimension)
    for block in SDPX.layout_blocks(layout)
        if block.cone === :nonnegative
            e[block.offset:block.offset + block.length - 1] .= one(T)
        elseif block.cone === :soc
            e[block.offset] = T(2)
        elseif block.cone === :psd
            @inbounds for j in 1:block.dimension
                # packed-lower column-major diagonal index
                idx = block.offset - 1 +
                      div((j - 1) * (2 * block.dimension - j + 2), 2) + 1
                e[idx] = one(T)
            end
        end
    end
    return e
end


function _pch_jordan_identity(::Type{T}, layout) where {T}
    e = _pch_central_target(T, layout)
    for block in SDPX.layout_blocks(layout)
        block.cone === :soc && (e[block.offset] /= T(2))
    end
    return e
end

function _pch_jordan(layout, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    z = zeros(T, layout.dimension)
    for block in SDPX.layout_blocks(layout)
        o = block.offset
        if block.cone === :nonnegative
            @inbounds for i in 1:block.length
                z[o + i - 1] = x[o + i - 1] * y[o + i - 1]
            end
        elseif block.cone === :soc
            head = x[o] * y[o]
            @inbounds for i in 2:block.dimension
                head += x[o + i - 1] * y[o + i - 1]
            end
            z[o] = head
            @inbounds for i in 2:block.dimension
                z[o + i - 1] = x[o] * y[o + i - 1] +
                               y[o] * x[o + i - 1]
            end
        elseif block.cone === :psd
            r = o:o + block.length - 1
            X = _pch_unsvec(x[r], block.dimension)
            Y = _pch_unsvec(y[r], block.dimension)
            z[r] .= _pch_svec((X * Y + Y * X) / T(2))
        end
    end
    return z
end

function _pch_trace_inner(layout, s::AbstractVector{T}, y::AbstractVector{T}) where {T}
    value = zero(T)
    for block in SDPX.layout_blocks(layout)
        r = block.offset:block.offset + block.length - 1
        if block.cone === :psd
            value += tr(
                _pch_unsvec(s[r], block.dimension) *
                _pch_unsvec(y[r], block.dimension),
            )
        else
            @inbounds for i in r
                value += s[i] * y[i]
            end
        end
    end
    return value
end

function _pch_Lmatrix(layout, lambda::AbstractVector{T}) where {T}
    m = layout.dimension
    L = zeros(T, m, m)
    for j in 1:m
        ej = zeros(T, m)
        ej[j] = one(T)
        L[:, j] .= _pch_jordan(layout, lambda, ej)
    end
    return L
end

function _pch_full_newton(base, Theta, h, scalar_rhs)
    T = eltype(base.c)
    A = Matrix(base.A)
    m, n = size(A)
    N = n + 2m + 2
    ix = 1:n
    iy = n + 1:n + m
    is = n + m + 1:n + 2m
    it = n + 2m + 1
    ik = N
    rp = 1:m
    rd = m + 1:m + n
    rg = m + n + 1
    rc = m + n + 2:m + n + 1 + m
    rk = N
    J = zeros(T, N, N)
    rhs = zeros(T, N)
    J[rp, ix] .= A
    J[rp, is] .= Matrix{T}(I, m, m)
    J[rp, it] .= -base.b
    J[rd, iy] .= transpose(A)
    J[rd, it] .= base.c
    J[rg, ix] .= -base.c
    J[rg, iy] .= -base.b
    J[rg, ik] = one(T)
    J[rc, iy] .= Theta
    J[rc, is] .= Matrix{T}(I, m, m)
    J[rk, it] = base.kappa
    J[rk, ik] = base.tau
    rhs[rp] .= -base.rP
    rhs[rd] .= -base.rD
    rhs[rg] = -base.rG
    rhs[rc] .= h
    rhs[rk] = scalar_rhs
    sol = J \ rhs
    return sol, J, rhs
end

function _pch_direction_vector(base; affine=false)
    if affine
        return [base.dx_a; base.dy_a; base.ds_a; base.dtau_a; base.dkappa_a]
    end
    return [base.dx; base.dy; base.ds; base.dtau; base.dkappa]
end

function _pch_five_residuals(base, Theta, h, scalar_rhs, direction)
    m = base.m
    n = base.n
    dx = direction[1:n]
    dy = direction[n + 1:n + m]
    ds = direction[n + m + 1:n + 2m]
    dtau = direction[n + 2m + 1]
    dkappa = direction[end]
    p = base.A * dx + ds - base.b * dtau + base.rP
    d = transpose(base.A) * dy + base.c * dtau + base.rD
    g = -dot(base.c, dx) - dot(base.b, dy) + dkappa + base.rG
    comp = ds + Theta * dy - h
    scalar = base.tau * dkappa + base.kappa * dtau - scalar_rhs
    return maximum(abs.(p)), maximum(abs.(d)), abs(g), maximum(abs.(comp)), abs(scalar)
end

const PCH_CASES = (
    ("LP", [(:nonnegative, 3)]),
    ("SOC", [(:soc, 3)]),
    ("PSD", [(:psd, 2)]),
    ("LP+SOC", [(:nonnegative, 2), (:soc, 3)]),
    ("LP+PSD", [(:nonnegative, 2), (:psd, 2)]),
    ("SOC+PSD", [(:soc, 3), (:psd, 2)]),
    ("LP+SOC+PSD", [(:nonnegative, 2), (:soc, 3), (:psd, 2)]),
)

@testset "product HSD barrier-metric central target" begin
    mu = 3.0 / 7.0
    for (label, specs) in PCH_CASES
        @testset "$label" begin
            layout = _pch_layout(Float64, specs)
            e = _pch_jordan_identity(Float64, layout)
            minus_gradient = _pch_central_target(Float64, layout)
            nu = SDPX.layout_barrier_degree(layout)
            # With SDPX's ordinary coordinate dot (and svec for PSD), the
            # barrier-gradient target contributes exactly nu*mu.
            @test dot(e, minus_gradient) ≈ nu atol=2e-14
            tau = 1.0
            kappa = mu
            y = mu .* minus_gradient
            @test (dot(e, y) + tau * kappa) / (nu + 1) ≈ mu atol=2e-14
        end
    end

    # The exact RSOC -> SOC canonicalization is orthogonal.  It therefore
    # transports the raw rotated-cone identity and its `-∇F=2e` target to
    # the same canonical Lorentz target without changing the dot pairing.
    M = SDPX._rsoc_to_soc_map(Float64, 3, 53)
    e_soc = [1.0, 0.0, 0.0]
    e_rsoc = M * e_soc
    minus_gradient_rsoc = 2.0 .* e_rsoc
    @test transpose(M) * M ≈ Matrix{Float64}(I, 3, 3) atol=2e-15
    @test M * minus_gradient_rsoc ≈ [2.0, 0.0, 0.0] atol=2e-15
    @test dot(e_rsoc, minus_gradient_rsoc) ≈ 2.0 atol=2e-15
end

@testset "native product HSD independent full-Newton directions" begin
    for (label, specs) in PCH_CASES
        @testset "$label" begin
            state = _pch_state(Float64, specs)
            base = state.base
            SDPX.hsd_residual!(base)
            rP0, rD0, rG0 = copy(base.rP), copy(base.rD), base.rG
            s0, y0 = copy(base.s), copy(base.y)
            tau0, kappa0, mu0 = base.tau, base.kappa, base.mu
            layout = base.canonical.cone_layout
            trace_inner = _pch_trace_inner(layout, s0, y0)
            @test dot(s0, y0) ≈ trace_inner rtol=2e-14 atol=2e-14
            @test mu0 ≈ (trace_inner + tau0 * kappa0) / (base.nu + 1) rtol=2e-14 atol=2e-14
            SDPX.update_scaling!(state.runtime, base.s, base.y, base.mu)
            Theta = _pch_dense_theta(state)
            theta_y = Theta * y0
            @test theta_y ≈ s0 rtol=2e-10 atol=2e-10
            @test Theta \ s0 ≈ y0 rtol=2e-10 atol=2e-10

            # R is the unique Euclidean SPD square root of the self-adjoint
            # NT operator Theta.  This is independent dense test algebra.
            F = eigen(Symmetric(Theta))
            R = F.vectors * Diagonal(sqrt.(F.values)) * transpose(F.vectors)
            lambda = R * y0
            Llambda = _pch_Lmatrix(layout, lambda)
            rc_aff = -_pch_jordan(layout, lambda, lambda)
            h_aff = R * (Llambda \ rc_aff)
            scalar_aff = -tau0 * kappa0
            affine_ref, Jaff, rhsaff = _pch_full_newton(base, Theta, h_aff, scalar_aff)

            fact0 = SDPX.product_hsd_factor_count(state)
            old_h_fact0 = SDPX.kkt_factor_count(base.driver)
            @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
            fact1 = SDPX.product_hsd_factor_count(state)
            @test fact1 - fact0 == 1
            @test SDPX.kkt_factor_count(base.driver) == old_h_fact0 == 0
            affine = _pch_direction_vector(base; affine=true)
            @test affine ≈ affine_ref rtol=3e-8 atol=3e-9
            @test maximum(abs.(Jaff * affine - rhsaff)) < 3e-8

            # The equation check uses a lightweight pre-step snapshot because
            # the accepted global step has already updated the base iterate.
            snap = (
                A=base.A, b=base.b, c=base.c, m=base.m, n=base.n,
                s=s0, y=y0, tau=tau0, kappa=kappa0,
                rP=rP0, rD=rD0, rG=rG0,
            )
            @test all(r -> r < 3e-8, _pch_five_residuals(
                snap, Theta, h_aff, scalar_aff, affine,
            ))

            ratio = max(base.mu_aff / mu0, 0.0)
            sigma = min(ratio^3, 1.0)
            ds_hat = R \ base.ds_a
            dy_hat = R * base.dy_a
            rc = sigma * mu0 .* _pch_central_target(Float64, layout) .-
                 _pch_jordan(layout, lambda, lambda) .-
                 _pch_jordan(layout, ds_hat, dy_hat)
            h = R * (Llambda \ rc)
            scalar_rhs = sigma * mu0 - tau0 * kappa0 -
                         base.dtau_a * base.dkappa_a
            corrector_ref, J, rhs = _pch_full_newton(snap, Theta, h, scalar_rhs)
            corrector = _pch_direction_vector(base)
            @test state.ds_hat ≈ ds_hat rtol=3e-8 atol=3e-9
            @test state.dy_hat ≈ dy_hat rtol=3e-8 atol=3e-9
            @test state.h ≈ h rtol=3e-8 atol=3e-9
            @test corrector ≈ corrector_ref rtol=5e-8 atol=5e-9
            @test maximum(abs.(J * corrector - rhs)) < 5e-8
            @test all(r -> r < 5e-8, _pch_five_residuals(
                snap, Theta, h, scalar_rhs, corrector,
            ))

            alpha = base.record.step_size
            @test base.record.primal_step == base.record.dual_step == alpha
            @test base.rP ≈ (1 - alpha) .* rP0 rtol=2e-10 atol=2e-11
            @test base.rD ≈ (1 - alpha) .* rD0 rtol=2e-10 atol=2e-11
            @test base.rG ≈ (1 - alpha) * rG0 rtol=2e-10 atol=2e-11
            @test state.runtime.valid
            @test state.runtime.last_mu == base.mu
            theta_post = similar(base.s)
            SDPX.apply_Theta!(state.runtime, theta_post, base.y)
            @test theta_post ≈ base.s rtol=3e-9 atol=3e-10
        end
    end
end


# Store the enum in caller-owned memory so the top-level allocation macro does
# not count a 16-byte boxed return value.  This is the same compiled-wrapper
# protocol as the legacy HSD hot-step gate.
@inline function _pch_step_noreturn!(codes, index::Int, state)
    codes[index] = SDPX.product_hsd_step!(state)
    return nothing
end

function _pch_prime_roundtrip!(state)
    base = state.base
    SDPX.hsd_residual!(base)
    SDPX.try_update_scaling!(state.runtime, base.s, base.y, base.mu) ||
        return false
    @inbounds for k in 1:base.m
        state.g_input[k] = (isodd(k) ? -1.0 : 1.0) * (k + 2) / 17
    end
    SDPX.apply_G!(state.runtime, base.dy, state.g_input)
    SDPX.apply_Theta!(state.runtime, base.e, base.dy)
    return true
end

"""Prepare a deterministic pre-line-search terminal-authority direction.

The ordinary public step refreshes `base.rD` after line search, so its
post-step iterate no longer carries the Newton RHS that belongs to the saved
bordered solve.  The strict bordered authority is intentionally exercised at
the same point as the production caller: after the two bounded corrections,
before the accepted pair is committed by line search.
"""
function _pch_authority_direction!()
    state = _pch_state(
        Float64, [(:nonnegative, 2), (:soc, 3), (:psd, 2)],
    )
    for _ in 1:7
        SDPX.product_hsd_step!(state) === SDPX.HSDStepOK || return nothing
    end
    base = state.base
    SDPX.hsd_residual!(base)
    SDPX.try_update_scaling!(state.runtime, base.s, base.y, base.mu) ||
        return nothing
    base.epoch += 1
    border_scalar = SDPX._product_hsd_form_schur_border!(state)
    isfinite(border_scalar) || return nothing
    SDPX._product_hsd_assemble_bordered!(state, border_scalar) ||
        return nothing
    SDPX._product_hsd_factor_bordered!(state) || return nothing
    SDPX._product_hsd_direction!(state) || return nothing
    ratio = base.mu_aff / base.mu
    sigma = min(ratio * ratio * ratio, 1.0)
    scalar_rhs = sigma * base.mu - base.tau * base.kappa -
                 base.dtau_a * base.dkappa_a
    return state, scalar_rhs
end

@testset "symmetric bordered step-8 factor and certificate regression" begin
    state = _pch_state(
        Float64, [(:nonnegative, 2), (:soc, 3), (:psd, 2)],
    )
    @test all(1:7) do _
        SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
    end
    workspace = state.symmetric_bordered
    factors_before = SDPX.product_hsd_factor_count(state)
    old_h_factors = SDPX.kkt_factor_count(state.base.driver)

    @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
    @test SDPX.product_hsd_factor_count(state) == factors_before + 1
    @test SDPX.kkt_factor_count(state.base.driver) == old_h_factors == 0
    @test workspace.factor_epoch == workspace.assembly_epoch == state.base.epoch
    @test workspace.factor_certified
    # Stable scalar recovery makes both predictor and corrector pass directly;
    # each reuses the single epoch factor and no bounded correction solve is
    # needed on this formerly ill-conditioned step-8 regression.
    @test workspace.solves == 2
    @test workspace.refinements == 0
    @test workspace.accumulations == 0
    @test cond(workspace.matrix) > 1e6
    @test cond(workspace.factor_matrix) < 1e6
    @test SDPX._product_bordered_transform_matrix_ok(workspace)
    @test SDPX._product_bordered_factor_certificate!(workspace)
    @test SDPX._product_bordered_factor_solution_ok!(workspace)
    @test SDPX._product_bordered_original_solution_ok!(workspace)
    @test all(abs.(workspace.residual) .<= workspace.bound)

    # A further solve with the retained corrector RHS must consume no factor.
    retained_factor_count = SDPX.product_hsd_factor_count(state)
    @test SDPX._product_bordered_staged_solve!(workspace)
    @test SDPX.product_hsd_factor_count(state) == retained_factor_count
    @test SDPX._product_bordered_factor_solution_ok!(workspace)
    @test SDPX._product_bordered_original_solution_ok!(workspace)
end

@testset "symmetric bordered transform and corruption gates" begin
    state = _pch_state(
        Float64, [(:nonnegative, 2), (:soc, 3), (:psd, 2)],
    )
    @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
    workspace = state.symmetric_bordered
    factor_count = SDPX.product_hsd_factor_count(state)

    saved_matrix_entry = workspace.factor_matrix[1, 1]
    workspace.factor_matrix[1, 1] = nextfloat(saved_matrix_entry)
    @test !SDPX._product_bordered_transform_matrix_ok(workspace)
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.factor_matrix[1, 1] = saved_matrix_entry

    saved_scale = workspace.row_scale[1]
    workspace.row_scale[1] = 0.0
    @test !SDPX._product_bordered_transform_matrix_ok(workspace)
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.row_scale[1] = saved_scale
    workspace.row_scale[1] = ldexp(saved_scale, 10)
    @test isfinite(workspace.row_scale[1])
    @test !SDPX._product_bordered_transform_matrix_ok(workspace)
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.row_scale[1] = saved_scale

    saved_original_matrix_entry = workspace.matrix[1, 1]
    workspace.matrix[1, 1] = saved_original_matrix_entry +
        1024 * max(1.0, abs(saved_original_matrix_entry))
    @test isfinite(workspace.matrix[1, 1])
    @test !SDPX._product_bordered_transform_matrix_ok(workspace)
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.matrix[1, 1] = saved_original_matrix_entry

    saved_exponent = workspace.row_exponent[1]
    workspace.row_exponent[1] = saved_exponent + 1
    @test !SDPX._product_bordered_transform_matrix_ok(workspace)
    workspace.row_exponent[1] = saved_exponent

    saved_order = workspace.transform_order
    workspace.transform_order = 0x00
    @test !SDPX._product_bordered_transform_matrix_ok(workspace)
    workspace.transform_order = saved_order

    saved_rhs = workspace.factor_rhs[1]
    workspace.factor_rhs[1] = nextfloat(saved_rhs)
    @test !SDPX._product_bordered_transform_rhs_ok(workspace)
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.factor_rhs[1] = saved_rhs

    workspace.factor_matrix[1, 1] = NaN
    @test !SDPX._product_bordered_transform_matrix_ok(workspace)
    workspace.factor_matrix[1, 1] = saved_matrix_entry

    factors = workspace.driver.route.factors
    saved_factor_entry = factors[1, 1]
    factors[1, 1] = saved_factor_entry +
        1024 * max(1.0, abs(saved_factor_entry))
    @test isfinite(factors[1, 1])
    @test !SDPX._product_bordered_factor_certificate!(workspace)
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    factors[1, 1] = saved_factor_entry
    @test SDPX._product_bordered_factor_certificate!(workspace)

    saved_lower_entry = factors[2, 1]
    factors[2, 1] = saved_lower_entry +
        1024 * max(1.0, abs(saved_lower_entry))
    @test isfinite(factors[2, 1])
    @test !SDPX._product_bordered_factor_certificate!(workspace)
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    factors[2, 1] = saved_lower_entry
    @test SDPX._product_bordered_factor_certificate!(workspace)

    saved_factor_epoch = workspace.factor_epoch
    workspace.factor_epoch += 1
    @test !SDPX._product_hsd_prepare_bordered_rhs!(state, -1.0)
    @test workspace.last_reason === SDPX.SYMMETRIC_BORDERED_EPOCH_MISMATCH
    workspace.factor_epoch = saved_factor_epoch

    saved_solution = copy(workspace.solution)
    workspace.solution[1] += 1024 * max(1.0, abs(workspace.solution[1]))
    @test !SDPX._product_bordered_triangular_solution_ok!(
        workspace, workspace.solution, workspace.factor_rhs,
        8 * workspace.dimension,
    )
    corrupted_factor_ok =
        SDPX._product_bordered_factor_solution_ok!(workspace)
    corrupted_original_ok =
        SDPX._product_bordered_original_solution_ok!(workspace)
    @test !(corrupted_factor_ok && corrupted_original_ok)
    copyto!(workspace.solution, saved_solution)
    @test SDPX._product_bordered_factor_solution_ok!(workspace)
    @test SDPX._product_bordered_original_solution_ok!(workspace)

    saved_snapshot = workspace.certified_solution[1]
    workspace.certified_solution[1] = saved_snapshot +
        1024 * max(1.0, abs(saved_snapshot))
    @test isfinite(workspace.certified_solution[1])
    @test !SDPX._product_bordered_factor_solution_ok!(workspace)
    workspace.certified_solution[1] = saved_snapshot
    @test SDPX._product_bordered_factor_solution_ok!(workspace)

    saved_bound = workspace.bound[1]
    saved_certified_bound = workspace.certified_factor_bound[1]
    workspace.bound[1] = saved_bound + 1024 * max(1.0, abs(saved_bound))
    @test isfinite(workspace.bound[1])
    @test !SDPX._product_bordered_original_solution_ok!(workspace)
    workspace.bound[1] = saved_bound
    workspace.certified_factor_bound[1] = saved_certified_bound +
        1024 * max(1.0, abs(saved_certified_bound))
    @test isfinite(workspace.certified_factor_bound[1])
    @test !SDPX._product_bordered_original_solution_ok!(workspace)
    workspace.certified_factor_bound[1] = saved_certified_bound
    @test SDPX._product_bordered_original_solution_ok!(workspace)
    @test SDPX._product_bordered_physical_snapshot_ok(workspace)

    saved_physical_bound = workspace.bound[1]
    saved_physical_snapshot = workspace.certified_physical_bound[1]
    workspace.bound[1] = saved_physical_bound +
        1024 * max(1.0, abs(saved_physical_bound))
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.bound[1] = saved_physical_bound
    workspace.certified_physical_bound[1] = saved_physical_snapshot +
        1024 * max(1.0, abs(saved_physical_snapshot))
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.certified_physical_bound[1] = saved_physical_snapshot
    workspace.original_solution_certified = false
    @test !SDPX._product_bordered_physical_snapshot_ok(workspace)
    workspace.original_solution_certified = true
    @test SDPX._product_bordered_physical_snapshot_ok(workspace)

    prepared = _pch_authority_direction!()
    @test prepared !== nothing
    authority_state, authority_scalar_rhs = prepared
    authority_workspace = authority_state.symmetric_bordered
    @test authority_workspace.solves == 2
    @test authority_workspace.refinements == 0
    @test SDPX._product_bordered_physical_snapshot_ok(
        authority_workspace,
    )
    # A direct strict five-equation certificate is authoritative. Merely
    # spoofing one refinement cannot activate the separate conditioned route.
    authority_workspace.refinements = 1
    @test !SDPX._product_hsd_symmetric_dual_residual_ok(authority_state)
    @test !SDPX._product_hsd_symmetric_scalar_residual_ok(
        authority_state, authority_scalar_rhs,
    )
    authority_workspace.refinements = 0
    @test SDPX._product_hsd_newton_residual_ok(
        authority_state, authority_scalar_rhs,
    )
    saved_dy = authority_state.base.dy[1]
    authority_state.base.dy[1] = saved_dy +
        1024 * max(1.0, abs(saved_dy))
    @test isfinite(authority_state.base.dy[1])
    @test !SDPX._product_hsd_newton_residual_ok(
        authority_state, authority_scalar_rhs,
    )
    authority_state.base.dy[1] = saved_dy
    @test SDPX._product_hsd_newton_residual_ok(
        authority_state, authority_scalar_rhs,
    )

    saved_dkappa = authority_state.base.dkappa
    authority_state.base.dkappa = saved_dkappa +
        1024 * max(1.0, abs(saved_dkappa))
    @test isfinite(authority_state.base.dkappa)
    @test !SDPX._product_hsd_newton_residual_ok(
        authority_state, authority_scalar_rhs,
    )
    authority_state.base.dkappa = saved_dkappa
    @test SDPX._product_hsd_newton_residual_ok(
        authority_state, authority_scalar_rhs,
    )

    # A retained accumulated candidate is accepted only while it exactly
    # matches the preallocated sum snapshot made from certified raw solves.
    copyto!(workspace.previous_solution, workspace.solution)
    fill!(workspace.correction_solution, 0.0)
    copyto!(workspace.certified_solution, workspace.solution)
    workspace.accumulated_candidate = true
    workspace.candidate_epoch = workspace.factor_epoch
    @test SDPX._product_bordered_factor_solution_ok!(workspace)
    workspace.solution[1] += 1024 * max(1.0, abs(workspace.solution[1]))
    @test !SDPX._product_bordered_factor_solution_ok!(workspace)
    copyto!(workspace.solution, workspace.certified_solution)

    saved_previous = workspace.previous_solution[1]
    workspace.previous_solution[1] = saved_previous +
        1024 * max(1.0, abs(saved_previous))
    @test !SDPX._product_bordered_factor_solution_ok!(workspace)
    workspace.previous_solution[1] = saved_previous

    saved_correction = workspace.correction_solution[1]
    workspace.correction_solution[1] = saved_correction + 1024.0
    @test !SDPX._product_bordered_factor_solution_ok!(workspace)
    workspace.correction_solution[1] = saved_correction
    @test SDPX._product_bordered_factor_solution_ok!(workspace)

    workspace.accumulated_candidate = false
    @test SDPX._product_bordered_factor_solution_ok!(workspace)
    @test SDPX._product_bordered_original_solution_ok!(workspace)
    @test SDPX.product_hsd_factor_count(state) == factor_count

    # A zero-work row has an exact certificate: no nonzero residual is
    # admitted through a relative or absolute residual floor.
    @test SDPX._product_bordered_zero_safe_close(0.0, 0.0)
    @test !SDPX._product_bordered_zero_safe_close(nextfloat(0.0), 0.0)

    # Exact binary equilibration must reject, rather than erase, a nonzero
    # coefficient that would underflow after scaling a floatmax-sized row.
    loss = _pch_state(Float64, [(:nonnegative, 3)])
    loss.base.epoch = 1
    fill!(loss.base.H, 0.0)
    @inbounds for i in 1:loss.base.nr
        loss.base.H[i, i] = 1.0
        loss.base.qr[i] = 1.0
        loss.base.rvec[i] = 1.0
    end
    loss.base.H[1, 1] = floatmax(Float64) / 2
    loss.base.qr[1] = nextfloat(0.0)
    @test !SDPX._product_hsd_assemble_bordered!(loss, 1.0)
    @test loss.symmetric_bordered.last_reason ===
          SDPX.SYMMETRIC_BORDERED_TRANSFORM_FAILED
    @test SDPX.product_hsd_factor_count(loss) == 0

    zero_row = _pch_state(Float64, [(:nonnegative, 3)])
    zero_row.base.epoch = 1
    fill!(zero_row.base.H, 0.0)
    fill!(zero_row.base.qr, 1.0)
    fill!(zero_row.base.rvec, 1.0)
    zero_row.base.qr[1] = 0.0
    @test !SDPX._product_hsd_assemble_bordered!(zero_row, 1.0)
    @test zero_row.symmetric_bordered.last_reason ===
          SDPX.SYMMETRIC_BORDERED_ZERO_ROW
    @test SDPX.product_hsd_factor_count(zero_row) == 0
end

@testset "roundtrip PSD-inconclusive fallback is family-specific" begin
    orthant = _pch_state(Float64, [(:nonnegative, 3)])
    @test _pch_prime_roundtrip!(orthant)
    @test SDPX._product_hsd_roundtrip_backward_status(orthant) ==
          (true, false)
    orthant.runtime.orthant[1].output[1] += 1.0
    @test SDPX._product_hsd_roundtrip_backward_status(orthant) ==
          (false, false)
    @test !SDPX._product_hsd_psd_cone_newton_residual_ok(orthant)

    soc = _pch_state(Float64, [(:soc, 3)])
    @test _pch_prime_roundtrip!(soc)
    @test SDPX._product_hsd_roundtrip_backward_status(soc) == (true, false)
    soc_block = soc.runtime.soc[1]
    soc_budget = SDPX._product_hsd_soc_condition_budget(
        soc_block.state.w, soc_block.dim,
    )
    @test isfinite(soc_budget) && 0 < soc_budget < 0.01
    dot_work = sum(abs(
        soc_block.state.w[i] * soc_block.input[i],
    ) for i in 1:soc_block.dim)
    radius = hypot(soc_block.state.w[2], soc_block.state.w[3])
    determinant = (soc_block.state.w[1] - radius) *
                  (soc_block.state.w[1] + radius)
    row_work = 2 * abs(soc_block.state.w[1]) * dot_work +
               abs(determinant) * abs(soc_block.input[1]) +
               abs(soc.g_input[1])
    allowance = soc_budget * row_work
    @test isfinite(allowance) && allowance > 0
    saved_soc_output = soc_block.output[1]
    soc_block.output[1] = soc.g_input[1] + 1024 * allowance
    @test isfinite(soc_block.output[1])
    @test abs(soc_block.output[1] - soc.g_input[1]) > allowance
    @test SDPX._product_hsd_roundtrip_backward_status(soc) == (false, false)
    soc_block.output[1] = saved_soc_output

    saved_w = copy(soc_block.state.w)
    soc_block.state.w[1] = 1.0
    soc_block.state.w[2] = 1.0 - 1e-8
    soc_block.state.w[3] = 0.0
    capped_budget = SDPX._product_hsd_soc_condition_budget(
        soc_block.state.w, soc_block.dim,
    )
    @test isfinite(capped_budget) && capped_budget >= 0.01
    @test !SDPX.SymmetricCones._soc_q_condition_reliable(
        soc_block.state.w, soc_block.dim,
    )
    @test SDPX._product_hsd_roundtrip_backward_status(soc) == (false, false)

    soc_block.state.w[1] = Inf
    @test !isfinite(SDPX._product_hsd_soc_condition_budget(
        soc_block.state.w, soc_block.dim,
    ))
    @test SDPX._product_hsd_roundtrip_backward_status(soc) == (false, false)
    copyto!(soc_block.state.w, saved_w)
    @test SDPX._product_hsd_roundtrip_backward_status(soc) == (true, false)

    soc.runtime.soc[1].output[1] = NaN
    @test SDPX._product_hsd_roundtrip_backward_status(soc) == (false, false)
    @test !SDPX._product_hsd_psd_cone_newton_residual_ok(soc)

    exp_state = SDPX.ProductConeHSDState(_pch_canonical(Float64, [(:exp, 3)]))
    SDPX.product_hsd_cold_start!(exp_state)
    @test _pch_prime_roundtrip!(exp_state)
    @test SDPX._product_hsd_roundtrip_backward_status(exp_state) ==
          (true, false)
    exp_state.runtime.exp[1].output[1] += 1.0
    @test SDPX._product_hsd_roundtrip_backward_status(exp_state) ==
          (false, false)
    @test !SDPX._product_hsd_psd_cone_newton_residual_ok(exp_state)
end

@testset "symmetric bordered nr=0 and custom base-driver semantics" begin
    T = Float64
    layout = _pch_layout(T, [(:nonnegative, 1)])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, 0.0, SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    canonical = SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), 53, T[], sparse(zeros(T, 1, 0)), T[0],
        layout, chain,
    )
    nr0 = SDPX.ProductConeHSDState(canonical)
    SDPX.product_hsd_cold_start!(nr0)
    @test nr0.base.nr == 0
    @test nr0.symmetric_bordered.dimension == 1
    @test SDPX.product_hsd_step!(nr0) === SDPX.HSDStepOK
    @test SDPX.product_hsd_factor_count(nr0) == 1
    @test SDPX.kkt_factor_count(nr0.base.driver) == 0

    ordinary = _pch_canonical(
        Float64, [(:nonnegative, 2), (:soc, 3), (:psd, 2)],
    )
    custom_base = SDPX.HotRouteCache(SDPX.LPLUCache{Float64}(2); n=2)
    custom = SDPX.ProductConeHSDState(ordinary, custom_base)
    _pch_set_pair!(custom)
    @test custom.base.nr == 2
    @test SDPX.product_hsd_step!(custom) === SDPX.HSDStepOK
    @test SDPX.kkt_factor_count(custom_base) == 0
    @test SDPX.product_hsd_factor_count(custom) == 1
    @test custom.symmetric_bordered.driver !== custom_base
end

@testset "product HSD warm Float64 steps allocate zero bytes" begin
    allocation_cases = (
        ("LP", [(:nonnegative, 3)]),
        ("SOC", [(:soc, 3)]),
        ("PSD", [(:psd, 2)]),
        ("mixed", [(:nonnegative, 2), (:soc, 3), (:psd, 2)]),
    )
    for (label, specs) in allocation_cases
        @testset "$label" begin
            state = _pch_state(Float64, specs)
            warm = Vector{SDPX.HSDStepCode}(undef, 1)
            _pch_step_noreturn!(warm, 1, state)
            @test warm[1] === SDPX.HSDStepOK
            # The mixed seven-sample lane includes the formerly failing step
            # 8.  The pure-SOC synthetic lane reaches its deliberate
            # near-machine-singular continuation at step 6, so its measured
            # allocation window stops at step 5.
            sample_count = label == "SOC" ? 4 : 7
            codes = Vector{SDPX.HSDStepCode}(undef, sample_count)
            samples = Vector{Int}(undef, sample_count)
            factors0 = SDPX.product_hsd_factor_count(state)
            old_h_factors = SDPX.kkt_factor_count(state.base.driver)
            @inbounds for sample in 1:sample_count
                samples[sample] = @allocated _pch_step_noreturn!(
                    codes, sample, state,
                )
            end
            @test all(==(SDPX.HSDStepOK), codes)
            @test samples == zeros(Int, sample_count)
            @test SDPX.product_hsd_factor_count(state) - factors0 ==
                  sample_count
            @test SDPX.kkt_factor_count(state.base.driver) == old_h_factors == 0
        end
    end
end

@testset "product HSD boundary and failure injection" begin
    state = _pch_state(Float64, [(:soc, 3), (:psd, 2)])
    # A near-boundary but strict SOC pair must either take a finite safe step
    # or fail closed with an isbits code; it must never emit a non-finite state.
    state.base.s[1] = hypot(state.base.s[2], state.base.s[3]) + 1e-10
    code = SDPX.product_hsd_step!(state)
    @test code isa SDPX.HSDStepCode
    @test all(isfinite, state.base.s)
    @test all(isfinite, state.base.y)

    outside = _pch_state(Float64, [(:nonnegative, 2), (:soc, 3)])
    outside.base.s[1] = 0.0
    @test SDPX.product_hsd_step!(outside) === SDPX.HSDStepDirectionFailed
    @test SDPX.product_hsd_factor_count(outside) == 0
    @test SDPX.kkt_factor_count(outside.base.driver) == 0

    nonfinite = _pch_state(Float64, [(:nonnegative, 3)])
    nonfinite.base.Ar.nzval[1] = NaN
    @test SDPX.product_hsd_step!(nonfinite) === SDPX.HSDStepDirectionFailed
    @test SDPX.product_hsd_factor_count(nonfinite) == 0
    @test SDPX.kkt_factor_count(nonfinite.base.driver) == 0

    singular = _pch_state(Float64, [(:nonnegative, 3)])
    fill!(singular.base.Ar.nzval, 0.0) # inject a zero Schur after setup RRQR
    @test SDPX.product_hsd_step!(singular) === SDPX.HSDStepSingularKKT
end

@testset "product HSD BigFloat direction math" begin
    setprecision(BigFloat, 192) do
        state = _pch_state(BigFloat, [(:soc, 3)])
        base = state.base
        SDPX.hsd_residual!(base)
        snap = (
            A=base.A, b=base.b, c=base.c, m=base.m, n=base.n,
            s=copy(base.s), y=copy(base.y), tau=base.tau, kappa=base.kappa,
            rP=copy(base.rP), rD=copy(base.rD), rG=base.rG,
        )
        SDPX.update_scaling!(state.runtime, base.s, base.y, base.mu)
        Theta = _pch_dense_theta(state)
        h = -copy(base.s) # predictor identity, independently exact for every symmetric cone
        scalar_rhs = -base.tau * base.kappa
        reference, J, rhs = _pch_full_newton(snap, Theta, h, scalar_rhs)
        @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
        affine = _pch_direction_vector(base; affine=true)
        @test affine ≈ reference rtol=big"1e-45" atol=big"1e-45"
        @test maximum(abs.(J * affine - rhs)) < big"1e-45"
        @test SDPX.product_hsd_factor_count(state) == 1
        @test SDPX.kkt_factor_count(base.driver) == 0
    end
end
