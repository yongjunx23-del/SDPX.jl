# Native mixed symmetric/nonsymmetric product-HSD integration gate.
#
# This file is deliberately standalone until the Phase-5 public/MOI route is
# enabled.  The dense coupled oracle below assembles and solves the complete
# Newton system directly; it never calls the production Schur, bordered solve,
# direction recovery, or line-search helpers.

using SDPX
using Test
using LinearAlgebra
using SparseArrays
using MultiFloats

const NPH_CASES = (
    ("Exp", ((:exp, 3),)),
    ("Power-0.1", ((:power, 3, "0.1"),)),
    ("Power-0.5", ((:power, 3, "0.5"),)),
    ("Power-0.9", ((:power, 3, "0.9"),)),
    ("LP+Exp", ((:nonnegative, 2), (:exp, 3))),
    ("SOC+Exp", ((:soc, 3), (:exp, 3))),
    ("LP+Power", ((:nonnegative, 2), (:power, 3, "0.5"))),
    ("SOC+Power", ((:soc, 3), (:power, 3, "0.5"))),
    ("PSD+Exp", ((:psd, 2), (:exp, 3))),
    ("LP+SOC+PSD+Exp+Power", (
        (:nonnegative, 2), (:soc, 3), (:psd, 2),
        (:exp, 3), (:power, 3, "0.5"),
    )),
)

@inline _nph_parameter(::Type{T}, value) where {T} = parse(T, value)

function _nph_layout(::Type{T}, specs) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for spec in specs
        kind = spec[1]
        dim = spec[2]
        block = if kind === :power
            SDPX.ConeBlockDescriptor(
                T, kind, dim; offset=offset,
                parameter=_nph_parameter(T, spec[3]),
            )
        else
            SDPX.ConeBlockDescriptor(T, kind, dim; offset=offset)
        end
        push!(blocks, block)
        offset += block.length
    end
    return SDPX.canonical_layout(blocks)
end

function _nph_program(::Type{T}, specs; n::Int=2) where {T<:AbstractFloat}
    layout = _nph_layout(T, specs)
    m = layout.dimension
    A = Matrix{T}(undef, m, n)
    @inbounds for j in 1:n, k in 1:m
        sign = isodd(k + 2j) ? -one(T) : one(T)
        A[k, j] = sign * (T(j + 1) / T(7) + T(k + j) / T(31))
    end
    b = Vector{T}(undef, m)
    @inbounds for k in 1:m
        b[k] = (isodd(k) ? -one(T) : one(T)) * T(k + 2) / T(19)
    end
    c = Vector{T}(undef, n)
    @inbounds for j in 1:n
        c[j] = (isodd(j) ? one(T) : -one(T)) * T(2j + 3) / T(17)
    end
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout, chain,
    )
end

function _nph_state(::Type{T}, specs) where {T<:AbstractFloat}
    state = SDPX.ProductConeHSDState(_nph_program(T, specs))
    SDPX.product_hsd_cold_start!(state)
    base = state.base
    @inbounds for j in 1:base.n
        base.x[j] = (isodd(j) ? one(T) : -one(T)) * T(j + 1) / T(23)
    end
    base.tau = T(11) / T(10)
    base.kappa = T(13) / T(10)
    return state
end

function _nph_svec(X::AbstractMatrix{T}) where {T}
    n = size(X, 1)
    output = Vector{T}(undef, div(n * (n + 1), 2))
    root2 = sqrt(T(2))
    index = 1
    @inbounds for j in 1:n, i in j:n
        output[index] = i == j ? X[i, j] : root2 * X[i, j]
        index += 1
    end
    return output
end

function _nph_unsvec(x::AbstractVector{T}, n::Int) where {T}
    output = zeros(T, n, n)
    invroot2 = inv(sqrt(T(2)))
    index = 1
    @inbounds for j in 1:n, i in j:n
        value = i == j ? x[index] : invroot2 * x[index]
        output[i, j] = value
        output[j, i] = value
        index += 1
    end
    return output
end

function _nph_soc_q(w::AbstractVector{T}) where {T}
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

"""Densify the frozen block metric without using production apply_G/Theta."""
function _nph_dense_theta(state::SDPX.ProductConeHSDState{T}) where {T}
    runtime = state.runtime
    theta = zeros(T, state.base.m, state.base.m)
    for block in runtime.orthant
        @inbounds for i in 1:block.dim
            theta[block.offset + i - 1, block.offset + i - 1] =
                block.state.theta[i]
        end
    end
    for block in runtime.soc
        r = block.offset:block.offset + block.dim - 1
        theta[r, r] .= _nph_soc_q(block.state.w)
    end
    for block in runtime.psd
        r = block.offset:block.offset + block.len - 1
        @inbounds for j in 1:block.len
            basis = zeros(T, block.len)
            basis[j] = one(T)
            X = _nph_unsvec(basis, block.dim)
            theta[r, r[j]] .= _nph_svec(block.state.P * X * block.state.P)
        end
    end
    for block in runtime.exp
        r = block.offset:block.offset + 2
        theta[r, r] .= block.scaling.theta
    end
    for block in runtime.power
        r = block.offset:block.offset + 2
        theta[r, r] .= block.scaling.theta
    end
    return theta
end

function _nph_jordan(block, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    if block.cone === :nonnegative
        return x .* y
    elseif block.cone === :soc
        output = zeros(T, block.length)
        output[1] = dot(x, y)
        @inbounds for i in 2:block.length
            output[i] = x[1] * y[i] + y[1] * x[i]
        end
        return output
    elseif block.cone === :psd
        X = _nph_unsvec(x, block.dimension)
        Y = _nph_unsvec(y, block.dimension)
        return _nph_svec((X * Y + Y * X) / T(2))
    end
    throw(ArgumentError("Jordan product requested for $(block.cone)"))
end

function _nph_central_target(block, ::Type{T}) where {T}
    if block.cone === :nonnegative
        return ones(T, block.length)
    elseif block.cone === :soc
        target = zeros(T, block.length)
        target[1] = T(2)
        return target
    elseif block.cone === :psd
        return _nph_svec(Matrix{T}(I, block.dimension, block.dimension))
    end
    throw(ArgumentError("symmetric central target requested for $(block.cone)"))
end

function _nph_symmetric_corrector_block(
    block,
    theta::AbstractMatrix{T},
    s::AbstractVector{T},
    y::AbstractVector{T},
    ds_aff::AbstractVector{T},
    dy_aff::AbstractVector{T},
    sigma_mu::T,
) where {T}
    eig = eigen(Symmetric(Matrix(theta)))
    R = eig.vectors * Diagonal(sqrt.(eig.values)) * transpose(eig.vectors)
    lambda = R * y
    ds_hat = R \ ds_aff
    dy_hat = R * dy_aff
    len = length(lambda)
    L = zeros(T, len, len)
    @inbounds for j in 1:len
        basis = zeros(T, len)
        basis[j] = one(T)
        L[:, j] .= _nph_jordan(block, lambda, basis)
    end
    rhs = sigma_mu .* _nph_central_target(block, T) .-
          _nph_jordan(block, lambda, lambda) .-
          _nph_jordan(block, ds_hat, dy_hat)
    return R * (L \ rhs)
end

"""Independent barrier-jet finite difference of the asymmetric chi term."""
function _nph_nonsymmetric_corrector_block(
    block,
    theta::AbstractMatrix{T},
    s::AbstractVector{T},
    y::AbstractVector{T},
    ds_aff::AbstractVector{T},
    dy_aff::AbstractVector{T},
    sigma_mu::T,
) where {T}
    return setprecision(BigFloat, 256) do
        sb = BigFloat.(s)
        yb = BigFloat.(y)
        dsb = BigFloat.(ds_aff)
        dyb = BigFloat.(dy_aff)
        oracle_block = block.cone === :exp ?
            SDPX.NewtonExpBlock() :
            SDPX.NewtonPowerBlock(BigFloat(block.parameter))
        gradient, hessian = SDPX._ns_newton_barrier_data(
            oracle_block, sb[1], sb[2], sb[3],
        )
        u = hessian \ dyb
        delta = big"1e-20" / max(one(BigFloat), norm(dsb))
        sp = sb + delta * dsb
        sm = sb - delta * dsb
        _, hp = SDPX._ns_newton_barrier_data(
            oracle_block, sp[1], sp[2], sp[3],
        )
        _, hm = SDPX._ns_newton_barrier_data(
            oracle_block, sm[1], sm[2], sm[3],
        )
        third_on_u = ((hp - hm) / (delta + delta)) * u
        chi = -third_on_u / BigFloat(2)
        rho = BigFloat(sigma_mu) .* (-gradient) .- yb .- chi
        h = BigFloat.(theta) * rho
        return T.(h)
    end
end

function _nph_corrector_h(
    layout,
    theta::AbstractMatrix{T},
    s::AbstractVector{T},
    y::AbstractVector{T},
    ds_aff::AbstractVector{T},
    dy_aff::AbstractVector{T},
    sigma_mu::T,
) where {T}
    h = zeros(T, layout.dimension)
    for block in SDPX.layout_blocks(layout)
        r = block.offset:block.offset + block.length - 1
        if block.cone === :exp || block.cone === :power
            h[r] .= _nph_nonsymmetric_corrector_block(
                block, theta[r, r], s[r], y[r], ds_aff[r], dy_aff[r],
                sigma_mu,
            )
        else
            h[r] .= _nph_symmetric_corrector_block(
                block, theta[r, r], s[r], y[r], ds_aff[r], dy_aff[r],
                sigma_mu,
            )
        end
    end
    return h
end

"""Independent complete Newton matrix in (dx,dy,ds,dtau,dkappa) order."""
function _nph_full_newton(snapshot, theta, h, scalar_rhs)
    T = eltype(snapshot.c)
    A = Matrix(snapshot.A)
    m, n = size(A)
    dimension = n + 2m + 2
    ix = 1:n
    iy = n + 1:n + m
    is = n + m + 1:n + 2m
    itau = n + 2m + 1
    ikappa = dimension
    rp = 1:m
    rd = m + 1:m + n
    rg = m + n + 1
    rc = m + n + 2:m + n + m + 1
    rtau = dimension
    J = zeros(T, dimension, dimension)
    rhs = zeros(T, dimension)
    J[rp, ix] .= A
    J[rp, is] .= Matrix{T}(I, m, m)
    J[rp, itau] .= -snapshot.b
    J[rd, iy] .= transpose(A)
    J[rd, itau] .= snapshot.c
    J[rg, ix] .= -snapshot.c
    J[rg, iy] .= -snapshot.b
    J[rg, ikappa] = one(T)
    J[rc, iy] .= theta
    J[rc, is] .= Matrix{T}(I, m, m)
    J[rtau, itau] = snapshot.kappa
    J[rtau, ikappa] = snapshot.tau
    rhs[rp] .= -snapshot.rP
    rhs[rd] .= -snapshot.rD
    rhs[rg] = -snapshot.rG
    rhs[rc] .= h
    rhs[rtau] = scalar_rhs
    return J \ rhs, J, rhs
end

function _nph_direction(base; affine::Bool=false)
    if affine
        return [base.dx_a; base.dy_a; base.ds_a; base.dtau_a; base.dkappa_a]
    end
    return [base.dx; base.dy; base.ds; base.dtau; base.dkappa]
end

function _nph_five_residuals(snapshot, theta, h, scalar_rhs, direction)
    m = length(snapshot.b)
    n = length(snapshot.c)
    dx = direction[1:n]
    dy = direction[n + 1:n + m]
    ds = direction[n + m + 1:n + 2m]
    dtau = direction[n + 2m + 1]
    dkappa = direction[end]
    primal = snapshot.A * dx + ds - snapshot.b * dtau + snapshot.rP
    dual = transpose(snapshot.A) * dy + snapshot.c * dtau + snapshot.rD
    gap = -dot(snapshot.c, dx) - dot(snapshot.b, dy) + dkappa + snapshot.rG
    complementarity = ds + theta * dy - h
    scalar = snapshot.kappa * dtau + snapshot.tau * dkappa - scalar_rhs
    return (
        maximum(abs, primal), maximum(abs, dual), abs(gap),
        maximum(abs, complementarity), abs(scalar),
    )
end

@testset "hybrid coupled pivoted LU has an explicit factor/solve certificate" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        workspace = SDPX.NonsymmetricCoupledWorkspace(
            spzeros(T, 3, 1), 1, [1],
        )
        workspace.matrix .= T[
            4 1 0 0 0 1
            1 5 1 0 1 0
            0 1 4 1 0 0
            0 0 1 3 1 0
            1 1 0 1 6 1
            0 0 0 0 1 2
        ]
        workspace.rhs .= T[1, 2, 3, 4, 5, 6]
        @test SDPX._product_coupled_factorize!(workspace, 7)
        @test workspace.factor_certified
        @test SDPX.factor_epoch(workspace.cache) == 1
        ok, merit = SDPX._product_coupled_solve!(
            workspace, workspace.solution, workspace.rhs,
        )
        @test ok
        @test zero(T) <= merit <= one(T)
        @test workspace.last_reason === SDPX.COUPLED_READY
        @test workspace.matrix * workspace.solution ≈ workspace.rhs
        permuted_residual = workspace.residual[workspace.permutation]
        @test all(
            abs(permuted_residual[i] - workspace.identity_rhs[i]) <=
            workspace.bound[workspace.permutation[i]]
            for i in eachindex(permuted_residual)
        )
        @test all(isfinite, workspace.forward_residual)
        @test all(isfinite, workspace.backward_residual)

        # A certificate for a refined total candidate must rebuild staged
        # y/f/u for that exact `(z,rhs)` pair.  Corrupting the residual scratch
        # therefore cannot be hidden by stale data from a prior correction
        # RHS.
        fill!(workspace.forward_residual, T(NaN))
        fill!(workspace.backward_residual, T(NaN))
        recertified, recertified_merit =
            SDPX._product_coupled_solution_certificate!(
                workspace, workspace.solution, workspace.rhs,
            )
        @test recertified
        @test zero(T) <= recertified_merit <= one(T)
        @test all(isfinite, workspace.forward_residual)
        @test all(isfinite, workspace.backward_residual)

        initial_residual = copy(workspace.residual)
        initial_norm = maximum(abs, initial_residual)
        workspace.correction_rhs .= -initial_residual
        correction_ok, _ = SDPX._product_coupled_solve!(
            workspace, workspace.correction, workspace.correction_rhs,
        )
        @test correction_ok
        workspace.solution .+= workspace.correction
        fill!(workspace.forward_residual, T(NaN))
        fill!(workspace.backward_residual, T(NaN))
        total_ok, total_merit =
            SDPX._product_coupled_solution_certificate!(
                workspace, workspace.solution, workspace.rhs,
            )
        @test total_ok
        @test zero(T) <= total_merit <= one(T)
        @test maximum(abs, workspace.residual) <= initial_norm
    end

    invalid = SDPX.NonsymmetricCoupledWorkspace(
        spzeros(Float64, 3, 1), 1, [1],
    )
    invalid.matrix .= Matrix{Float64}(I, 6, 6)
    invalid.matrix[1, 1] = NaN
    @test !SDPX._product_coupled_factorize!(invalid, 1)
    @test invalid.last_reason === SDPX.COUPLED_ASSEMBLY_NONFINITE
end

@testset "factor-coordinate coupled KKT preserves physical certificates" begin
    specs = (
        (:nonnegative, 2), (:soc, 3), (:psd, 2),
        (:exp, 3), (:power, 3, "0.5"),
    )
    state = _nph_state(Float64, specs)
    base = state.base
    SDPX.hsd_residual!(base)
    @test SDPX.try_update_scaling!(state.runtime, base.s, base.y, base.mu)
    base.epoch = 1
    workspace = state.coupled
    @test SDPX._product_hsd_form_coupled_matrix!(state)
    @test workspace.transform_valid
    @test workspace.transform_order == SDPX._PRODUCT_COUPLED_TRANSFORM_ORDER
    @test SDPX._product_coupled_factor_coordinate_matrix_certificate!(workspace)
    @test all(isfinite, workspace.factor_coordinate_matrix)
    @test all(isfinite, workspace.factor_coordinate_factor)
    physical_condition = cond(workspace.matrix)
    factor_condition = cond(workspace.factor_coordinate_matrix)
    @test isfinite(physical_condition)
    @test isfinite(factor_condition)
    @test factor_condition != physical_condition
    @test any(
        workspace.factor_coordinate_matrix[i, j] != workspace.matrix[i, j]
        for i in axes(workspace.matrix, 1), j in axes(workspace.matrix, 2)
    )

    @test SDPX._product_coupled_factorize!(workspace, base.epoch)
    @test SDPX.factor_epoch(workspace.cache) == 1
    @test SDPX._product_coupled_factorize!(workspace, base.epoch)
    @test SDPX.factor_epoch(workspace.cache) == 1

    SDPX.affine_shift!(state.runtime, state.h, base.s, base.y)
    scalar_rhs = -base.tau * base.kappa
    @test SDPX._product_hsd_coupled_rhs!(state, scalar_rhs)
    @test workspace.factor_coordinate_rhs_valid
    @test SDPX._product_coupled_factor_coordinate_rhs_certificate!(workspace)

    solved, _ = SDPX._product_coupled_solve!(
        workspace, workspace.solution, workspace.rhs,
    )
    @test solved
    @test workspace.factor_coordinate_matrix * workspace.solution ≈
          workspace.factor_coordinate_rhs rtol=2e-12 atol=2e-12
    @test SDPX._product_coupled_recover_physical!(
        workspace, workspace.solution,
    )
    @test SDPX._product_coupled_original_solution_certificate!(
        workspace, workspace.physical_solution,
    )
    @test workspace.matrix * workspace.physical_solution ≈
          workspace.rhs atol=2e-12 rtol=2e-12

    # Every mutable factor-coordinate component is independently fail-closed.
    saved_order = workspace.transform_order
    workspace.transform_order = UInt8(0x02)
    @test !SDPX._product_coupled_factor_coordinate_matrix_certificate!(workspace)
    workspace.transform_order = saved_order
    saved_epoch = workspace.transform_epoch
    workspace.transform_epoch = 0
    @test !SDPX._product_coupled_solution_certificate!(
        workspace, workspace.solution, workspace.factor_coordinate_rhs,
    )[1]
    workspace.transform_epoch = saved_epoch

    saved_khat = workspace.factor_coordinate_matrix[1, 1]
    workspace.factor_coordinate_matrix[1, 1] = NaN
    @test !SDPX._product_coupled_factor_coordinate_matrix_certificate!(workspace)
    workspace.factor_coordinate_matrix[1, 1] = saved_khat
    finite_khat_delta = 16.0 * max(1.0, abs(saved_khat))
    workspace.factor_coordinate_matrix[1, 1] = saved_khat + finite_khat_delta
    @test !SDPX._product_coupled_factor_coordinate_matrix_certificate!(workspace)
    workspace.factor_coordinate_matrix[1, 1] = saved_khat

    saved_l = workspace.factor_coordinate_factor[1, 1]
    workspace.factor_coordinate_factor[1, 1] = NaN
    @test !SDPX._product_coupled_factor_coordinate_matrix_certificate!(workspace)
    workspace.factor_coordinate_factor[1, 1] = saved_l
    workspace.factor_coordinate_factor[1, 1] = 0.0
    @test !SDPX._product_coupled_factor_coordinate_matrix_certificate!(workspace)
    workspace.factor_coordinate_factor[1, 1] = saved_l

    saved_qhat = workspace.factor_coordinate_rhs[1]
    workspace.factor_coordinate_rhs[1] = NaN
    @test !SDPX._product_coupled_factor_coordinate_rhs_certificate!(workspace)
    workspace.factor_coordinate_rhs[1] = saved_qhat
    finite_qhat_delta = 16.0 * max(1.0, abs(saved_qhat))
    workspace.factor_coordinate_rhs[1] = saved_qhat + finite_qhat_delta
    @test !SDPX._product_coupled_factor_coordinate_rhs_certificate!(workspace)
    workspace.factor_coordinate_rhs[1] = saved_qhat

    saved_rhs_checkpoint = workspace.factor_coordinate_rhs_valid
    workspace.factor_coordinate_rhs_valid = false
    @test !SDPX._product_coupled_solution_certificate!(
        workspace, workspace.solution, workspace.factor_coordinate_rhs,
    )[1]
    workspace.factor_coordinate_rhs_valid = saved_rhs_checkpoint

    saved_v = workspace.solution[base.nr + 1]
    workspace.solution[base.nr + 1] = NaN
    @test !SDPX._product_coupled_solution_certificate!(
        workspace, workspace.solution, workspace.factor_coordinate_rhs,
    )[1]
    workspace.solution[base.nr + 1] = saved_v
    finite_v_delta = 1024.0 * max(1.0, maximum(abs, workspace.solution))
    workspace.solution[base.nr + 1] = saved_v + finite_v_delta
    @test SDPX._product_coupled_recover_physical!(
        workspace, workspace.solution,
    )
    @test !SDPX._product_coupled_original_solution_certificate!(
        workspace, workspace.physical_solution,
    )
    workspace.solution[base.nr + 1] = saved_v
    @test SDPX._product_coupled_recover_physical!(
        workspace, workspace.solution,
    )
    @test SDPX._product_coupled_original_solution_certificate!(
        workspace, workspace.physical_solution,
    )

    workspace.transform_epoch = base.epoch + 1
    @test !SDPX._product_coupled_factorize!(workspace, base.epoch)
    @test workspace.last_reason === SDPX.COUPLED_TRANSFORM_EPOCH_MISMATCH
    workspace.transform_epoch = base.epoch
    @test SDPX._product_coupled_factorize!(workspace, base.epoch)
    warm_ok, _ = SDPX._product_coupled_solve!(
        workspace, workspace.solution, workspace.rhs,
    )
    @test warm_ok
    warm_alloc = @allocated SDPX._product_coupled_solve!(
        workspace, workspace.solution, workspace.rhs,
    )
    @test warm_alloc == 0
end

@testset "factor-coordinate transform precision smoke" begin
    specs = ((:exp, 3), (:power, 3, "0.5"))
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            state = _nph_state(T, specs)
            base = state.base
            SDPX.hsd_residual!(base)
            @test SDPX.try_update_scaling!(
                state.runtime, base.s, base.y, base.mu,
            )
            base.epoch = 1
            workspace = state.coupled
            @test SDPX._product_hsd_form_coupled_matrix!(state)
            @test SDPX._product_coupled_factor_coordinate_matrix_certificate!(
                workspace,
            )
            @test SDPX._product_coupled_factorize!(workspace, base.epoch)
            @test SDPX.factor_epoch(workspace.cache) == 1
            SDPX.affine_shift!(state.runtime, state.h, base.s, base.y)
            @test SDPX._product_hsd_coupled_rhs!(state, -base.tau * base.kappa)
            solved, _ = SDPX._product_coupled_solve!(
                workspace, workspace.solution, workspace.rhs,
            )
            @test solved
            @test SDPX._product_coupled_recover_physical!(
                workspace, workspace.solution,
            )
            @test SDPX._product_coupled_original_solution_certificate!(
                workspace, workspace.physical_solution,
            )
        end
    end
    setprecision(BigFloat, 256) do
        @testset "BigFloat256" begin
            state = _nph_state(BigFloat, specs)
            base = state.base
            SDPX.hsd_residual!(base)
            @test SDPX.try_update_scaling!(
                state.runtime, base.s, base.y, base.mu,
            )
            base.epoch = 1
            workspace = state.coupled
            @test SDPX._product_hsd_form_coupled_matrix!(state)
            @test SDPX._product_coupled_factorize!(workspace, base.epoch)
            @test SDPX.factor_epoch(workspace.cache) == 1
            SDPX.affine_shift!(state.runtime, state.h, base.s, base.y)
            @test SDPX._product_hsd_coupled_rhs!(state, -base.tau * base.kappa)
            solved, _ = SDPX._product_coupled_solve!(
                workspace, workspace.solution, workspace.rhs,
            )
            @test solved
            @test SDPX._product_coupled_recover_physical!(
                workspace, workspace.solution,
            )
            @test SDPX._product_coupled_original_solution_certificate!(
                workspace, workspace.physical_solution,
            )
        end
    end
end

@testset "mixed sparse nonsymmetric Schur matches independent dense oracle" begin
    for (label, specs) in NPH_CASES
        @testset "$label" begin
            state = _nph_state(Float64, specs)
            base = state.base
            SDPX.hsd_residual!(base)
            @test SDPX.try_update_scaling!(
                state.runtime, base.s, base.y, base.mu,
            )

            theta = _nph_dense_theta(state)
            dense_g = inv(theta)
            nonsymmetric_g = zeros(Float64, base.m, base.m)
            for block in state.runtime.exp
                rows = block.offset:block.offset + 2
                nonsymmetric_g[rows, rows] .=
                    inv(Symmetric(Matrix(block.scaling.theta)))
            end
            for block in state.runtime.power
                rows = block.offset:block.offset + 2
                nonsymmetric_g[rows, rows] .=
                    inv(Symmetric(Matrix(block.scaling.theta)))
            end
            Ar = Matrix(base.Ar)
            factors_before = SDPX.kkt_factor_count(base.driver)
            border = SDPX._product_hsd_form_schur_border!(state)

            expected_h = transpose(Ar) * dense_g * Ar
            expected_at_g_b = transpose(Ar) * dense_g * base.b
            expected_b_g_a = vec(transpose(base.b) * dense_g * Ar)
            expected_b_g_b = dot(base.b, dense_g * base.b)
            @test base.H ≈ expected_h rtol=2e-10 atol=2e-11
            @test base.qr ≈ base.cr - expected_at_g_b rtol=2e-10 atol=2e-11
            @test isapprox(
                base.rvec, base.tau .* (base.cr + expected_b_g_a);
                rtol=2e-10, atol=2e-11,
            )
            @test isapprox(
                border, base.kappa - base.tau * expected_b_g_b;
                rtol=2e-10, atol=2e-11,
            )

            expected_ns_h = transpose(Ar) * nonsymmetric_g * Ar
            expected_ns_at_g_b = transpose(Ar) * nonsymmetric_g * base.b
            expected_ns_b_g_a = vec(
                transpose(base.b) * nonsymmetric_g * Ar,
            )
            @test state.ns_H ≈ expected_ns_h rtol=2e-10 atol=2e-11
            @test isapprox(
                state.ns_at_g_b, expected_ns_at_g_b;
                rtol=2e-10, atol=2e-11,
            )
            @test isapprox(
                state.ns_bt_g_a, expected_ns_b_g_a;
                rtol=2e-10, atol=2e-11,
            )

            @inbounds for index in 1:base.m
                state.h[index] = (isodd(index) ? 1.0 : -1.0) *
                                 (index + 3) / 29
            end
            rhs = state.h + base.rP
            expected_at_g_rhs = transpose(Ar) * dense_g * rhs
            expected_b_g_rhs = dot(base.b, dense_g * rhs)
            b_g_rhs = SDPX._product_hsd_rhs!(state)
            @test isapprox(
                base.rhs, -base.rDr - expected_at_g_rhs;
                rtol=2e-10, atol=2e-11,
            )
            @test b_g_rhs ≈ expected_b_g_rhs rtol=2e-10 atol=2e-11
            @test isapprox(
                state.ns_at_g_rhs,
                transpose(Ar) * nonsymmetric_g * rhs;
                rtol=2e-10, atol=2e-11,
            )
            @test SDPX.kkt_factor_count(base.driver) == factors_before
        end
    end
end

@testset "mixed asymmetric product-HSD coupled directions" begin
    for (label, specs) in NPH_CASES
        @testset "$label" begin
            state = _nph_state(Float64, specs)
            base = state.base
            SDPX.hsd_residual!(base)
            @test SDPX.try_update_scaling!(
                state.runtime, base.s, base.y, base.mu,
            )
            theta = _nph_dense_theta(state)
            s0 = copy(base.s)
            y0 = copy(base.y)
            mu0 = base.mu
            rP0 = copy(base.rP)
            rD0 = copy(base.rD)
            rG0 = base.rG
            snapshot = (
                A=base.A, b=base.b, c=base.c,
                rP=rP0, rD=rD0, rG=rG0,
                tau=base.tau, kappa=base.kappa,
            )

            # Theta*y=s is the frozen pair orientation for every block.
            @test theta * y0 ≈ s0 rtol=2e-9 atol=2e-10
            affine_h = -s0
            affine_scalar = -base.tau * base.kappa
            affine_reference, J_aff, rhs_aff = _nph_full_newton(
                snapshot, theta, affine_h, affine_scalar,
            )
            factors_before = SDPX.product_hsd_factor_count(state)
            @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
            @test SDPX.product_hsd_factor_count(state) - factors_before == 1
            affine = _nph_direction(base; affine=true)
            @test affine ≈ affine_reference rtol=3e-8 atol=3e-9
            @test maximum(abs, J_aff * affine - rhs_aff) < 3e-8
            @test all(value -> value < 3e-8, _nph_five_residuals(
                snapshot, theta, affine_h, affine_scalar, affine,
            ))

            ratio = base.mu_aff / mu0
            @test isfinite(ratio) && ratio >= 0
            sigma = min(ratio^3, 1.0)
            sigma_mu = sigma * mu0
            corrector_h = _nph_corrector_h(
                base.canonical.cone_layout, theta, s0, y0,
                base.ds_a, base.dy_a, sigma_mu,
            )
            corrector_scalar = sigma_mu - snapshot.tau * snapshot.kappa -
                               base.dtau_a * base.dkappa_a
            corrector_reference, J, rhs = _nph_full_newton(
                snapshot, theta, corrector_h, corrector_scalar,
            )
            corrector = _nph_direction(base)
            @test state.h ≈ corrector_h rtol=3e-6 atol=3e-7
            @test corrector ≈ corrector_reference rtol=4e-6 atol=4e-7
            @test maximum(abs, J * corrector - rhs) < 4e-6
            @test all(value -> value < 4e-6, _nph_five_residuals(
                snapshot, theta, corrector_h, corrector_scalar, corrector,
            ))

            alpha = base.record.step_size
            @test base.record.primal_step == base.record.dual_step == alpha
            @test base.rP ≈ (1 - alpha) .* rP0 rtol=3e-10 atol=3e-11
            @test base.rD ≈ (1 - alpha) .* rD0 rtol=3e-10 atol=3e-11
            @test base.rG ≈ (1 - alpha) * rG0 rtol=3e-10 atol=3e-11
            @test SDPX.product_strictly_interior(
                state.runtime, base.s, base.y,
            )
            @test state.runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_READY
            @test all(block -> !block.scaling.used_fallback, state.runtime.exp)
            @test all(block -> !block.scaling.used_fallback, state.runtime.power)
        end
    end
end

function _nph_exact_program(::Type{T}, specs, A, b, c) where {T<:AbstractFloat}
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(c), sparse(T.(A)),
        Vector{T}(b), _nph_layout(T, specs), chain,
    )
end

function _nph_initial_pair(::Type{T}, specs) where {T<:AbstractFloat}
    state = SDPX.ProductConeHSDState(_nph_program(T, specs))
    SDPX.product_hsd_cold_start!(state)
    return copy(state.base.s), copy(state.base.y)
end

function _nph_promote_exact(state, tol)
    base = state.base
    return SDPX._product_hsd_verified_result(
        state,
        zeros(eltype(base.x), base.n),
        zeros(eltype(base.s), base.m),
        zeros(eltype(base.y), base.m),
        tol,
        SDPX.ProductHSDVerifiedInitialPoint,
        SDPX.HSDStepOK,
    )
end

@testset "Exp/Power original-coordinate certificate-only statuses" begin
    for (label, specs) in NPH_CASES
        @testset "$label" begin
            T = Float64
            s, y = _nph_initial_pair(T, specs)
            m = length(s)
            tol = 1e-10

            optimal_program = _nph_exact_program(
                T, specs, zeros(T, m, 1), s, T[0],
            )
            optimal_state = SDPX.ProductConeHSDState(optimal_program)
            fill!(optimal_state.base.x, 0)
            copyto!(optimal_state.base.s, s)
            fill!(optimal_state.base.y, 0)
            optimal_state.base.tau = 1
            optimal_state.base.kappa = 0
            optimal = _nph_promote_exact(optimal_state, tol)
            @test optimal !== nothing
            @test optimal.status === SDPX.ProductHSDOptimal
            @test SDPX.verify_optimal!(
                optimal_program, optimal_state.base,
                zeros(T, 1), zeros(T, m), zeros(T, m); tol=tol,
            )

            primal_program = _nph_exact_program(
                T, specs, zeros(T, m, 1), -y, T[0],
            )
            primal_state = SDPX.ProductConeHSDState(primal_program)
            fill!(primal_state.base.x, 0)
            fill!(primal_state.base.s, 0)
            copyto!(primal_state.base.y, y)
            primal_state.base.tau = 0
            primal_state.base.kappa = 0
            primal = _nph_promote_exact(primal_state, tol)
            @test primal !== nothing
            @test primal.status === SDPX.ProductHSDPrimalInfeasible
            @test SDPX.verify_primal_infeasibility!(
                primal_program, primal_state.base, zeros(T, m); tol=tol,
            )

            dual_program = _nph_exact_program(
                T, specs, reshape(-s, m, 1), zeros(T, m), T[-1],
            )
            dual_state = SDPX.ProductConeHSDState(dual_program)
            dual_state.base.x[1] = 1
            fill!(dual_state.base.s, 0)
            fill!(dual_state.base.y, 0)
            dual_state.base.tau = 0
            dual_state.base.kappa = 0
            dual = _nph_promote_exact(dual_state, tol)
            @test dual !== nothing
            @test dual.status === SDPX.ProductHSDDualInfeasible
            @test SDPX.verify_dual_infeasibility!(
                dual_program, dual_state.base, zeros(T, 1), zeros(T, m);
                tol=tol,
            )
        end
    end
end

function _nph_solver_fixture(::Type{T}, specs, kind::Symbol) where {T<:AbstractFloat}
    s, y = _nph_initial_pair(T, specs)
    m = length(s)
    if kind === :optimal
        # No free variables: the unique primal slack is b=3s/2 and the
        # analytic objective is zero. The scale avoids the homogeneous-border
        # cancellation at b=s for the Power initialization.
        return _nph_exact_program(
            T, specs, zeros(T, m, 0), T(3) / T(2) .* s, T[],
        )
    elseif kind === :primal_infeasible
        # The initialized dual-interior point is an exact Farkas ray:
        # A'y=0 and b'y=-||y||^2<0.
        return _nph_exact_program(
            T, specs, zeros(T, m, 0), -y, T[],
        )
    elseif kind === :dual_infeasible
        # x=1 gives -A*x=2s in the full primal product cone and c'x=-1.
        return _nph_exact_program(
            T, specs, reshape(-T(2) .* s, m, 1), zeros(T, m), T[-1],
        )
    end
    throw(ArgumentError("unknown solver fixture $kind"))
end

function _nph_reverify_result(program, result, tol)
    state = SDPX.ProductConeHSDState(program)
    copyto!(state.base.x, result.hsd_x)
    copyto!(state.base.s, result.hsd_s)
    copyto!(state.base.y, result.hsd_y)
    state.base.tau = result.tau
    state.base.kappa = result.kappa
    if result.status === SDPX.ProductHSDOptimal
        return SDPX.verify_optimal!(
            program, state.base,
            zeros(eltype(result.x), state.base.n),
            zeros(eltype(result.s), state.base.m),
            zeros(eltype(result.y), state.base.m);
            tol=tol,
        )
    elseif result.status === SDPX.ProductHSDPrimalInfeasible
        return SDPX.verify_primal_infeasibility!(
            program, state.base, zeros(eltype(result.y), state.base.m);
            tol=tol,
        )
    elseif result.status === SDPX.ProductHSDDualInfeasible
        return SDPX.verify_dual_infeasibility!(
            program, state.base,
            zeros(eltype(result.x), state.base.n),
            zeros(eltype(result.s), state.base.m);
            tol=tol,
        )
    end
    return false
end

@testset "factor-coordinate whitening controls an ill-conditioned Exp release fixture" begin
    # Reproduce the accepted production Exp metric at release iteration 55,
    # then use the exact [C_N;D;G;T] five-row coupled layout with zero b_N.
    # This deterministic equivalent fixture isolates the block whitening
    # effect while retaining the production metric and accepted factor.
    release_program = _nph_exact_program(
        Float64, ((:exp, 3),), reshape(Float64[0, 0, -1], 3, 1),
        Float64[0, 1, 0], Float64[1],
    )
    release_state = SDPX.ProductConeHSDState(release_program)
    SDPX.product_hsd_cold_start!(release_state)
    for _ in 1:55
        @test SDPX.product_hsd_step!(release_state) === SDPX.HSDStepOK
    end
    block = release_state.runtime.exp[1]
    workspace = SDPX.NonsymmetricCoupledWorkspace(
        spzeros(Float64, 3, 0), 0, [1],
    )
    workspace.matrix[1:3, 1:3] .= block.scaling.theta
    workspace.matrix[4, 5] = 1.0
    workspace.matrix[5, 4] = 1.0
    workspace.matrix[5, 5] = 1.0
    workspace.rhs .= [0.1, -0.2, 0.3, 0.4, -0.5]
    @test SDPX._product_coupled_copy_factor_block!(
        workspace, 0, block.scaling.factor, block.scaling.theta,
    )
    @test SDPX._product_coupled_prepare_factor_coordinates!(workspace, 1)
    physical_condition = cond(workspace.matrix)
    factor_condition = cond(workspace.factor_coordinate_matrix)
    @test physical_condition > 1.0e8
    @test physical_condition > 1.0e6 * factor_condition
    @test factor_condition < 1.0e5
    @test SDPX._product_coupled_factor_coordinate_matrix_certificate!(workspace)
    @test SDPX._product_coupled_factorize!(workspace, 1)
    @test SDPX._product_coupled_transform_rhs!(workspace)
    solved, _ = SDPX._product_coupled_solve!(
        workspace, workspace.solution, workspace.rhs,
    )
    @test solved
    @test SDPX._product_coupled_recover_physical!(
        workspace, workspace.solution,
    )
    @test SDPX._product_coupled_original_solution_certificate!(
        workspace, workspace.physical_solution,
    )
    @test all(
        abs(workspace.recovery_residual[i]) <= workspace.recovery_bound[i]
        for i in eachindex(workspace.recovery_residual)
    )
end

@testset "cold-start asymmetric product solver earns certificate statuses" begin
    expected = (
        :optimal => SDPX.ProductHSDOptimal,
        :primal_infeasible => SDPX.ProductHSDPrimalInfeasible,
        :dual_infeasible => SDPX.ProductHSDDualInfeasible,
    )
    for (label, specs) in NPH_CASES
        @testset "$label" begin
            for (kind, status) in expected
                program = _nph_solver_fixture(Float64, specs, kind)
                state = SDPX.ProductConeHSDState(program)
                result = SDPX.product_hsd_solve!(
                    state; max_iterations=100, tol=1e-8,
                )
                @test result.status === status
                @test _nph_reverify_result(program, result, 1e-8)
                @test result.reason in (
                    SDPX.ProductHSDVerifiedInitialPoint,
                    SDPX.ProductHSDVerifiedAcceptedStep,
                    SDPX.ProductHSDVerifiedTerminalNewtonTrial,
                )
                @test all(isfinite, result.hsd_x)
                @test all(isfinite, result.hsd_s)
                @test all(isfinite, result.hsd_y)
                @test result.factorizations >= result.iterations
                @test all(
                    block -> block.policy isa
                             SDPX.DoubleSecantWithDualHessianFallback,
                    state.runtime.exp,
                )
                @test all(
                    block -> block.policy isa
                             SDPX.DoubleSecantWithDualHessianFallback,
                    state.runtime.power,
                )
            end
        end
    end
end

function _nph_release_program(::Type{T}, kind::Symbol) where {T<:AbstractFloat}
    specs = kind === :exp ? ((:exp, 3),) :
            kind === :power ? ((:power, 3, "0.5"),) :
            throw(ArgumentError("unknown release fixture $kind"))
    A = reshape(T[0, 0, -1], 3, 1)
    b = kind === :exp ? T[0, 1, 0] : T[1, 1, 0]
    return _nph_exact_program(T, specs, A, b, T[1])
end

function _nph_release_trace(program, iterations::Int, kind::Symbol)
    state = SDPX.ProductConeHSDState(program)
    SDPX.product_hsd_cold_start!(state)
    saw_shadow_only_strict = false
    saw_strict_after_long_backtrack = false
    maximum_backtracking = 0
    for _ in 1:iterations
        code = SDPX.product_hsd_step!(state)
        code === SDPX.HSDStepOK || return (
            state, false, false, maximum_backtracking, code,
        )
        maximum_backtracking = max(
            maximum_backtracking, state.base.record.backtracking,
        )
        block = kind === :exp ? state.runtime.exp[1] : state.runtime.power[1]
        saw_shadow_only_strict |=
            block.last_scaling_status === SDPX.NS_SCALING_DOUBLE_SECANT &&
            block.last_scaling_reason === SDPX.NS_SCALING_CONVERGED &&
            !block.scaling.conjugate.inverse_valid &&
            !block.scaling.conjugate.accepted_inverse_valid
        saw_strict_after_long_backtrack |=
            state.base.record.backtracking > 16 &&
            block.last_scaling_status === SDPX.NS_SCALING_DOUBLE_SECANT &&
            block.last_scaling_reason === SDPX.NS_SCALING_CONVERGED
    end
    return (
        state, saw_shadow_only_strict, saw_strict_after_long_backtrack,
        maximum_backtracking, SDPX.HSDStepOK,
    )
end

@testset "Exp/Power release models converge from the real cold solver" begin
    expected_objective = (:exp => 1.0, :power => -1.0)
    for (kind, optimum) in expected_objective
        @testset "$kind" begin
            program = _nph_release_program(Float64, kind)
            state = SDPX.ProductConeHSDState(program)
            result = SDPX.product_hsd_solve!(
                state; max_iterations=400, tol=1e-8,
            )
            @test result.status === SDPX.ProductHSDOptimal
            @test result.reason in (
                SDPX.ProductHSDVerifiedAcceptedStep,
                SDPX.ProductHSDVerifiedTerminalNewtonTrial,
            )
            @test _nph_reverify_result(program, result, 1e-8)
            @test result.x[1] ≈ optimum rtol=2e-6 atol=2e-6
            @test result.normalized_residual <= 1e-8

            trace_state, saw_shadow_only_strict, saw_long_strict,
            max_backtracking,
            trace_code = _nph_release_trace(
                program, result.iterations, kind,
            )
            @test trace_code === SDPX.HSDStepOK
            @test SDPX.product_hsd_factor_count(trace_state) ==
                  result.iterations
            if kind === :exp
                @test saw_shadow_only_strict
            else
                # The hardened Power scaling accepts all seven convergent
                # fraction-to-boundary trials directly (normalized residual
                # 6.14e-9).  The former `> 16`/long-strict expectations
                # described an older scaling-rejection path, not a convergence
                # requirement: strict double-secant now succeeds before the
                # later certified fallback epochs without forced backtracking.
                @test saw_shadow_only_strict
                @test max_backtracking == 0
                @test !saw_long_strict
            end
        end
    end
end

@testset "mixed asymmetric failure injection is typed and closed" begin
    outside = _nph_state(Float64, ((:exp, 3),))
    outside.base.s[2] = 0.0
    @test SDPX.product_hsd_step!(outside) === SDPX.HSDStepDirectionFailed
    @test SDPX.product_hsd_factor_count(outside) == 0
    @test outside.runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_FAILED

    nonfinite = _nph_state(Float64, ((:power, 3, "0.5"),))
    nonfinite.base.Ar.nzval[1] = NaN
    @test SDPX.product_hsd_step!(nonfinite) === SDPX.HSDStepDirectionFailed
    @test SDPX.product_hsd_factor_count(nonfinite) == 0
    @test all(isfinite, nonfinite.base.s)
    @test all(isfinite, nonfinite.base.y)

    invalid_layout = _nph_layout(Float64, ((:power, 3, "1.0"),))
    @test_throws ArgumentError SDPX.ProductConeRuntime(invalid_layout, Float64)

    # Nonfinite implicit-map work must remain a typed three-field statistic;
    # callers may destructure it but must never reach a Bool-only early return.
    for specs in (((:soc, 3),), ((:psd, 2),))
        invalid_work = _nph_state(Float64, specs)
        fill!(invalid_work.base.dy, Inf)
        stats = SDPX._product_hsd_cone_newton_stats(invalid_work)
        # Typed four-field statistic: (componentwise_ok, group_residual,
        # group_work, conditioned_family_failed).  Non-finite work must fail
        # on the three core fields on every cone family; the 4th Bool marks
        # whether the failing family was itself conditioned (varies per cone).
        @test stats isa NTuple{4, Union{Bool,Float64}}
        @test stats[1:3] == (false, Inf, 0.0)
        @test stats[4] isa Bool
        @test !SDPX._product_hsd_cone_newton_residual_ok(invalid_work)
    end
end

@testset "mixed asymmetric precision smoke" begin
    smoke = (
        ((:exp, 3),),
        ((:power, 3, "0.5"),),
        ((:nonnegative, 2), (:soc, 3), (:exp, 3),
         (:power, 3, "0.5")),
    )
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            for specs in smoke
                state = _nph_state(T, specs)
                @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
                @test SDPX.product_hsd_factor_count(state) == 1
                @test SDPX.product_strictly_interior(
                    state.runtime, state.base.s, state.base.y,
                )
            end
        end
    end
    precision_bits = if isdefined(@__MODULE__, :TEST_PROFILE) &&
                        TEST_PROFILE === :quick
        (256,)
    else
        (256, 512, 1024)
    end
    for bits in precision_bits
        setprecision(BigFloat, bits) do
            bigfloat_smoke = bits == 256 ? smoke : (smoke[1], smoke[2])
            for specs in bigfloat_smoke
                state = _nph_state(BigFloat, specs)
                @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
                @test SDPX.product_hsd_factor_count(state) == 1
                @test SDPX.product_strictly_interior(
                    state.runtime, state.base.s, state.base.y,
                )
            end
        end
    end
end

struct _NPHStepSeed{T}
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    tau::T
    kappa::T
end

function _nph_step_seed(state::SDPX.ProductConeHSDState{T}) where {T}
    base = state.base
    return _NPHStepSeed{T}(
        copy(base.x), copy(base.s), copy(base.y), base.tau, base.kappa,
    )
end

@inline function _nph_reset_step_noreturn!(codes, index::Int, state, seed)
    base = state.base
    copyto!(base.x, seed.x)
    copyto!(base.s, seed.s)
    copyto!(base.y, seed.y)
    base.tau = seed.tau
    base.kappa = seed.kappa
    codes[index] = SDPX.product_hsd_step!(state)
    return nothing
end

@testset "mixed asymmetric warmed accepted steps allocate zero bytes" begin
    allocation_cases = (
        ("Exp", ((:exp, 3),)),
        ("Power", ((:power, 3, "0.5"),)),
        ("LP+Exp", ((:nonnegative, 2), (:exp, 3))),
        ("SOC+Power", ((:soc, 3), (:power, 3, "0.5"))),
        ("full", (
            (:nonnegative, 2), (:soc, 3), (:psd, 2),
            (:exp, 3), (:power, 3, "0.5"),
        )),
    )
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        selected = T === Float64 ? allocation_cases : (allocation_cases[end],)
        @testset "$T" begin
            for (label, specs) in selected
                @testset "$label" begin
                    state = _nph_state(T, specs)
                    seed = _nph_step_seed(state)
                    warm = Vector{SDPX.HSDStepCode}(undef, 1)
                    _nph_reset_step_noreturn!(warm, 1, state, seed)
                    @test warm[1] === SDPX.HSDStepOK
                    codes = Vector{SDPX.HSDStepCode}(undef, 10)
                    samples = Vector{Int}(undef, 10)
                    factors_before = SDPX.product_hsd_factor_count(state)
                    @inbounds for sample in 1:10
                        samples[sample] = @allocated _nph_reset_step_noreturn!(
                            codes, sample, state, seed,
                        )
                    end
                    @test all(==(SDPX.HSDStepOK), codes)
                    @test samples == zeros(Int, 10)
                    @test SDPX.product_hsd_factor_count(state) -
                          factors_before == 10
                end
            end
        end
    end
end
