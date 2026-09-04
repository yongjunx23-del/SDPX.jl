# Setup and validation for the native symmetric product runtime.

@inline _runtime_psd_length(n::Int) = (n * (n + 1)) >>> 1

"""Allocation-free authoritative solve of `Theta*(G*q) = q` for a 3D block.

The accepted scaling's `theta` is the numerical authority.  The separately
stored explicit inverse remains available for scaling diagnostics, but is not
used by product-runtime `G` actions.  Factoring from scalar temporaries on
every call keeps the solve tied to the current accepted/checkpointed `theta`
without introducing another mutable factor epoch.
"""
@inline function _runtime_nonsymmetric_theta_factor3!(
    factor::Matrix{T}, theta::Matrix{T},
) where {T<:AbstractFloat}
    @inbounds for j in 1:3, i in 1:3
        isfinite(theta[i, j]) || return false
    end
    theta[1, 2] == theta[2, 1] &&
    theta[1, 3] == theta[3, 1] &&
    theta[2, 3] == theta[3, 2] || return false
    h11 = theta[1, 1]
    isfinite(h11) && h11 > zero(T) || return false
    l11 = sqrt(h11)
    l21 = theta[2, 1] / l11
    l31 = theta[3, 1] / l11

    pivot2 = theta[2, 2] - l21 * l21
    isfinite(pivot2) && pivot2 > zero(T) || return false
    l22 = sqrt(pivot2)
    l32 = (theta[3, 2] - l31 * l21) / l22

    pivot3 = theta[3, 3] - l31 * l31 - l32 * l32
    isfinite(pivot3) && pivot3 > zero(T) || return false
    l33 = sqrt(pivot3)

    z = zero(T)
    factor[1, 1] = l11
    factor[1, 2] = z
    factor[1, 3] = z
    factor[2, 1] = l21
    factor[2, 2] = l22
    factor[2, 3] = z
    factor[3, 1] = l31
    factor[3, 2] = l32
    factor[3, 3] = l33

    three_eps = T(3) * eps(T)
    three_eps < one(T) || return false
    forcing = T(128) * three_eps / (one(T) - three_eps)
    @inbounds for j in 1:3, i in j:3
        reconstructed = zero(T)
        product_work = zero(T)
        for k in 1:j
            term = factor[i, k] * factor[j, k]
            reconstructed += term
            product_work += abs(term)
        end
        work = abs(theta[i, j]) + product_work
        residual = reconstructed - theta[i, j]
        isfinite(residual) && isfinite(work) || return false
        if iszero(work)
            iszero(residual) || return false
        elseif abs(residual) > forcing * work
            return false
        end
    end
    return true
end

@inline function _runtime_nonsymmetric_forward_solve3!(
    destination::Vector{T}, factor::Matrix{T}, rhs::Vector{T},
) where {T<:AbstractFloat}
    l11 = factor[1, 1]
    l21 = factor[2, 1]
    l22 = factor[2, 2]
    l31 = factor[3, 1]
    l32 = factor[3, 2]
    l33 = factor[3, 3]

    # Read the full right-hand side before writing the destination so the
    # primitive remains alias-safe.
    b1, b2, b3 = rhs[1], rhs[2], rhs[3]
    isfinite(b1) && isfinite(b2) && isfinite(b3) || return false
    q1 = b1 / l11
    q2 = (b2 - l21 * q1) / l22
    q3 = (b3 - l31 * q1 - l32 * q2) / l33
    isfinite(q1) && isfinite(q2) && isfinite(q3) || return false
    destination[1] = q1
    destination[2] = q2
    destination[3] = q3

    three_eps = T(3) * eps(T)
    three_eps < one(T) || return false
    forcing = T(128) * three_eps / (one(T) - three_eps)
    @inbounds for i in 1:3
        action = zero(T)
        work = abs(rhs[i])
        for j in 1:i
            term = factor[i, j] * destination[j]
            action += term
            work += abs(term)
        end
        residual = action - rhs[i]
        if iszero(work)
            iszero(residual) || return false
        elseif !(isfinite(residual) && isfinite(work) &&
                 abs(residual) <= forcing * work)
            return false
        end
    end
    return true
end

@inline function _runtime_nonsymmetric_backward_solve3!(
    destination::Vector{T}, factor::Matrix{T}, rhs::Vector{T},
) where {T<:AbstractFloat}
    # `rhs` may alias `destination`; capture it before the backward solve.
    b1, b2, b3 = rhs[1], rhs[2], rhs[3]
    l11 = factor[1, 1]
    l21 = factor[2, 1]
    l22 = factor[2, 2]
    l31 = factor[3, 1]
    l32 = factor[3, 2]
    l33 = factor[3, 3]
    x3 = b3 / l33
    x2 = (b2 - l32 * x3) / l22
    x1 = (b1 - l21 * x2 - l31 * x3) / l11
    isfinite(x1) && isfinite(x2) && isfinite(x3) || return false
    destination[1] = x1
    destination[2] = x2
    destination[3] = x3
    return true
end

@inline function _runtime_nonsymmetric_theta_solve3!(
    destination::Vector{T}, theta::Matrix{T}, rhs::Vector{T},
    factor::Matrix{T}, forward::Vector{T},
) where {T<:AbstractFloat}
    _runtime_nonsymmetric_theta_factor3!(factor, theta) || return false
    _runtime_nonsymmetric_forward_solve3!(forward, factor, rhs) || return false
    return _runtime_nonsymmetric_backward_solve3!(
        destination, factor, forward,
    )
end

@inline function _runtime_nonsymmetric_theta_solve3_backward_ok(
    theta::Matrix{T}, solution::Vector{T}, rhs::Vector{T},
) where {T<:AbstractFloat}
    three_eps = T(3) * eps(T)
    three_eps < one(T) || return false
    gamma3 = three_eps / (one(T) - three_eps)
    forcing = T(128) * gamma3
    isfinite(forcing) || return false
    @inbounds for i in 1:3
        action = theta[i, 1] * solution[1] +
                 theta[i, 2] * solution[2] +
                 theta[i, 3] * solution[3]
        work = abs(rhs[i]) +
               abs(theta[i, 1]) * abs(solution[1]) +
               abs(theta[i, 2]) * abs(solution[2]) +
               abs(theta[i, 3]) * abs(solution[3])
        residual = action - rhs[i]
        isfinite(residual) && isfinite(work) || return false
        if iszero(work)
            iszero(residual) || return false
        elseif abs(residual) > forcing * work
            return false
        end
    end
    return true
end

"""No-throw authoritative nonsymmetric `G` action used by the HSD hot path."""
@inline function try_apply_nonsymmetric_G_reason!(
    destination::Vector{T},
    workspace::NonsymmetricScalingWorkspace{T},
    source::Vector{T},
) where {T<:AbstractFloat}
    workspace.valid || return NS_SCALING_INVALID_PARAMETER
    length(destination) == length(source) == 3 ||
        return NS_SCALING_INVALID_PARAMETER
    isfinite(source[1]) && isfinite(source[2]) && isfinite(source[3]) ||
        return NS_SCALING_NONFINITE_INPUT
    _runtime_nonsymmetric_theta_factor3!(
        workspace.factor, workspace.theta,
    ) || return NS_SCALING_METRIC_NOT_SPD
    _runtime_nonsymmetric_forward_solve3!(
        workspace.work3, workspace.factor, source,
    ) || return NS_SCALING_INVERSE_MISMATCH
    _runtime_nonsymmetric_backward_solve3!(
        destination, workspace.factor, workspace.work3,
    ) || return NS_SCALING_INVERSE_MISMATCH
    _runtime_nonsymmetric_theta_solve3_backward_ok(
        workspace.theta, destination, source,
    ) || return NS_SCALING_INVERSE_MISMATCH
    return NS_SCALING_CONVERGED
end

@inline function try_apply_nonsymmetric_G!(
    destination::Vector{T},
    workspace::NonsymmetricScalingWorkspace{T},
    source::Vector{T},
) where {T<:AbstractFloat}
    return try_apply_nonsymmetric_G_reason!(
        destination, workspace, source,
    ) === NS_SCALING_CONVERGED
end

function _runtime_validate_block(block, expected::Int)
    block.offset == expected || throw(ArgumentError(
        "ProductConeRuntime requires contiguous offsets: block offset $(block.offset), " *
        "expected $(expected)",
    ))
    block.dimension >= 1 || throw(ArgumentError("cone block dimension must be positive"))
    if block.cone === :psd
        block.storage === :packed_lower || throw(ArgumentError(
            "PSD runtime requires :packed_lower storage, got $(block.storage)",
        ))
        block.length == _runtime_psd_length(block.dimension) || throw(ArgumentError(
            "PSD packed length $(block.length) does not match dimension $(block.dimension)",
        ))
    elseif block.cone === :zero
        block.storage === :vector || throw(ArgumentError(
            "zero-cone runtime requires :vector storage, got $(block.storage)",
        ))
        block.length == block.dimension || throw(ArgumentError(
            "zero-cone block length must equal dimension",
        ))
    elseif block.cone === :nonnegative || block.cone === :soc
        block.storage === :vector || throw(ArgumentError(
            "$(block.cone) runtime requires :vector storage, got $(block.storage)",
        ))
        block.length == block.dimension || throw(ArgumentError(
            "$(block.cone) block length must equal dimension",
        ))
    elseif block.cone === :exp || block.cone === :power
        block.storage === :vector || throw(ArgumentError(
            "$(block.cone) runtime requires :vector storage, got $(block.storage)",
        ))
        block.dimension == 3 && block.length == 3 || throw(ArgumentError(
            "$(block.cone) runtime blocks must have dimension and length three",
        ))
    else
        throw(ArgumentError(
            "ProductConeRuntime supports :zero, :nonnegative, :soc, :psd, :exp and :power; " *
            "unsupported block $(repr(block.cone))",
        ))
    end
    return expected + block.length
end

function _runtime_empty_vectors(::Type{T}) where {T}
    return Vector{Nothing}()
end

@inline function _runtime_initial_step_result(::Type{T}) where {T}
    return NonsymmetricStepResult{T}(
        NS_STEP_FULL_LIMIT, zero(T), zero(T), one(T), 0,
    )
end

function _runtime_make_line_search(
    ::Type{T}, primal_tag, dual_tag,
) where {T}
    # A representable interior retreat is required near curved boundaries;
    # `1-O(eps)` can round back onto the boundary after the final multiply.
    safety = T(995) / T(1000)
    return NonsymmetricRuntimeLineSearchWorkspace{
        T,typeof(primal_tag),typeof(dual_tag),
    }(
        primal_tag,
        dual_tag,
        safety,
        one(T),
        64,
        _ns_initialization_default_bisections(T),
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        _runtime_initial_step_result(T),
        _runtime_initial_step_result(T),
    )
end

function _runtime_make_scaling_checkpoint(::Type{T}) where {T}
    return NonsymmetricRuntimeScalingCheckpoint{T}(
        false,
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        alloc_zeros(T, 3, 3),
        alloc_zeros(T, 3, 3),
        alloc_zeros(T, 3, 3),
        zero(T),
        false,
        false,
        NS_SCALING_FAILED,
        NS_SCALING_INVALID_PARAMETER,
        NS_SCALING_NO_FALLBACK,
        NS_CONJUGATE_INVALID_PARAMETER,
        alloc_zeros(T, 3),
        alloc_zeros(T, 3, 3),
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        alloc_zeros(T, 3, 3),
        alloc_zeros(T, 3, 3),
        alloc_zeros(T, 3, 3),
        alloc_zeros(T, 3, 3),
        alloc_zeros(T, 3, 3),
        T(Inf),
        T(Inf),
        zero(T),
        zero(T),
        false,
        false,
        false,
        false,
        false,
        false,
        NS_CONJUGATE_MAPPED_COLD_SEED,
    )
end

function _runtime_make_orthant(::Type{T}, block) where {T}
    dim = block.dimension
    cone = SymmetricCones.NonnegativeCone(dim)
    state = SymmetricCones.OrthantNTScaling{T}(dim)
    return OrthantRuntimeBlock{T}(
        block.offset, dim, cone, state,
        alloc_zeros(T, dim), alloc_zeros(T, dim), alloc_zeros(T, dim),
        alloc_zeros(T, dim), alloc_zeros(T, dim), Ref{T}(zero(T)),
    )
end

function _runtime_make_soc(::Type{T}, block) where {T}
    dim = block.dimension
    dim >= 2 || throw(ArgumentError("SOC block dimension must be at least 2"))
    cone = SymmetricCones.SOCone(dim)
    state = SymmetricCones.SOCNTScaling{T}(dim)
    return SOCRuntimeBlock{T}(
        block.offset, dim, cone, state,
        alloc_zeros(T, dim), alloc_zeros(T, dim), alloc_zeros(T, dim),
        alloc_zeros(T, dim), alloc_zeros(T, dim), Ref{T}(zero(T)),
    )
end

function _runtime_make_psd(::Type{T}, block) where {T}
    n = block.dimension
    len = _runtime_psd_length(n)
    block.length == len || throw(ArgumentError(
        "PSD packed length $(block.length) does not match dimension $(n)",
    ))
    cone = SymmetricCones.PSDTriangleCone{T}(n)
    state = SymmetricCones.PSDNTScaling{T}(n)
    return PSDRuntimeBlock{T}(
        block.offset, n, len, cone, state,
        alloc_zeros(T, len), alloc_zeros(T, len), alloc_zeros(T, len),
        alloc_zeros(T, len), alloc_zeros(T, len), alloc_zeros(T, len),
        alloc_zeros(T, len), Ref{T}(zero(T)),
    )
end

function _runtime_make_exp(::Type{T}, block) where {T}
    initialization = NonsymmetricInitializationWorkspace(T)
    scaling = initialization.scaling
    corrector = NonsymmetricCorrectorWorkspace(T)
    line_search = _runtime_make_line_search(
        T, ExpPrimalStepTag(), ExpDualStepTag(),
    )
    return ExpRuntimeBlock{T}(
        block.offset,
        3,
        ExpConjugateTag(),
        DoubleSecantWithDualHessianFallback(),
        initialization,
        scaling,
        corrector,
        line_search,
        scaling.primal,
        scaling.dual,
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        _runtime_make_scaling_checkpoint(T),
        NS_SCALING_FAILED,
        NS_SCALING_INVALID_PARAMETER,
        NS_SCALING_NO_FALLBACK,
        NS_CONJUGATE_INVALID_PARAMETER,
    )
end

function _runtime_make_power(::Type{T}, block) where {T}
    block.parameter isa T || throw(ArgumentError(
        "power runtime parameter must already have target type $T, got " *
        "$(typeof(block.parameter))",
    ))
    alpha = block.parameter
    isfinite(alpha) && zero(T) < alpha < one(T) || throw(ArgumentError(
        "power runtime alpha must be finite and strictly between zero and one",
    ))
    tag = PowerConjugateTag(alpha)
    initialization = NonsymmetricInitializationWorkspace(T)
    scaling = initialization.scaling
    corrector = NonsymmetricCorrectorWorkspace(T)
    line_search = _runtime_make_line_search(
        T, PowerPrimalStepTag(alpha), PowerDualStepTag(alpha),
    )
    return PowerRuntimeBlock{T}(
        block.offset,
        3,
        tag,
        DoubleSecantWithDualHessianFallback(),
        initialization,
        scaling,
        corrector,
        line_search,
        scaling.primal,
        scaling.dual,
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        alloc_zeros(T, 3),
        _runtime_make_scaling_checkpoint(T),
        false,
        0,
        NS_SCALING_FAILED,
        NS_SCALING_INVALID_PARAMETER,
        NS_SCALING_NO_FALLBACK,
        NS_CONJUGATE_INVALID_PARAMETER,
    )
end

"""Construct a setup-built product runtime at arithmetic type `T`.

Symmetric blocks own NT state. Exp/Power blocks own a frozen explicit
double-secant-with-recorded-dual-Hessian-fallback policy and all nonsymmetric
workspaces. Free/zero/RSOC and all other noncanonical families fail closed.
"""
function ProductConeRuntime(layout::ConeProductLayout, ::Type{T}) where {T<:AbstractFloat}
    blocks = layout.blocks
    has_nonsymmetric = any(
        block -> block.cone === :exp || block.cone === :power,
        blocks,
    )
    has_nonsymmetric &&
        !isdefined(@__MODULE__, :_NONSYMMETRIC_RUNTIME_API_LOADED) &&
        throw(ArgumentError(
            "nonsymmetric runtime API must be included before constructing Exp/Power blocks",
        ))
    orthant = OrthantRuntimeBlock{T}[]
    soc = SOCRuntimeBlock{T}[]
    psd = PSDRuntimeBlock{T}[]
    exp_blocks = ExpRuntimeBlock{T}[]
    power_blocks = PowerRuntimeBlock{T}[]
    zero_ranges = UnitRange{Int}[]
    expected = 1
    for block in blocks
        expected = _runtime_validate_block(block, expected)
        if block.cone === :zero
            push!(zero_ranges, block.offset:(block.offset + block.length - 1))
        elseif block.cone === :nonnegative
            push!(orthant, _runtime_make_orthant(T, block))
        elseif block.cone === :soc
            push!(soc, _runtime_make_soc(T, block))
        elseif block.cone === :psd
            push!(psd, _runtime_make_psd(T, block))
        elseif block.cone === :exp
            push!(exp_blocks, _runtime_make_exp(T, block))
        elseif block.cone === :power
            push!(power_blocks, _runtime_make_power(T, block))
        end
    end
    expected - 1 == layout.dimension || throw(ArgumentError(
        "layout dimension $(layout.dimension) does not equal block storage $(expected - 1)",
    ))
    return ProductConeRuntime{
        T,typeof(orthant),typeof(soc),typeof(psd),
        typeof(exp_blocks),typeof(power_blocks),
    }(
        orthant,
        soc,
        psd,
        exp_blocks,
        power_blocks,
        zero_ranges,
        layout.dimension,
        false,
        zero(T),
        _runtime_nonsymmetric_default_result(T),
        zero(T),
        _runtime_nonsymmetric_default_result(T),
        false,
    )
end

ProductConeRuntime{T}(layout::ConeProductLayout) where {T<:AbstractFloat} =
    ProductConeRuntime(layout, T)

ProductConeRuntime(layout::ConeProductLayout{B}, sample::AbstractArray{T}) where {B,T<:AbstractFloat} =
    ProductConeRuntime(layout, T)

@inline function _runtime_authoritative_g_failure!(
    runtime::ProductConeRuntime{T},
    block_offset::Int,
    scaling_reason::NonsymmetricScalingReason=NS_SCALING_METRIC_NOT_SPD,
) where {T}
    runtime_reason = scaling_reason === NS_SCALING_NONFINITE_INPUT ?
                     NS_RUNTIME_NONFINITE_INPUT :
                     NS_RUNTIME_SCALING_FAILED
    runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
        NS_RUNTIME_FAILED,
        runtime_reason,
        block_offset,
        NS_INITIALIZATION_CONVERGED,
        NS_SCALING_FAILED,
        scaling_reason,
        NS_SCALING_NO_FALLBACK,
        NS_CONJUGATE_INVALID_PARAMETER,
        NS_CORRECTOR_CONVERGED,
        NS_STEP_INVALID_PARAMETER,
        zero(T),
    )
    runtime.valid = false
    return false
end
