"""Immutable, owned data needed by the FixedTraceQ3 reduced dual."""
struct FixedTraceQ3DualLayout{T}
    active_ids::Matrix{Int}
    inverse_map::Array{T,3}
    inverse_transpose::Array{T,3}
    fixed_head::Vector{T}
    offset::Matrix{T}
    objective::Matrix{T}
    equality_panel::Matrix{T}
    equality_rhs::Vector{T}
    problem_fingerprint::NTuple{32,UInt8}
    reduction_fingerprint::NTuple{32,UInt8}
    arithmetic::Symbol
    precision_bits::Int
    ownership::Symbol
end

mutable struct FixedTraceDualEvalStats
    transpose_gemv_seconds::Float64
    block_support_seconds::Float64
    forward_gemv_seconds::Float64
    objective_evaluations::Int
    gradient_evaluations::Int
    transpose_gemv_calls::Int
    forward_gemv_calls::Int
end

FixedTraceDualEvalStats() = FixedTraceDualEvalStats(0, 0, 0, 0, 0, 0, 0)

@inline function _fixed_trace_digest_token!(context, token::AbstractString)
    bytes = codeunits(token)
    SHA.update!(context, codeunits(string(length(bytes), ':')))
    SHA.update!(context, bytes)
    return nothing
end

function _fixed_trace_digest_value!(context, value::AbstractArray)
    _fixed_trace_digest_token!(
        context,
        string("array:", typeof(value), ':', join(size(value), ',')),
    )
    if isbitstype(eltype(value)) && value isa Array
        bytes = reinterpret(UInt8, vec(value))
        _fixed_trace_digest_token!(context, string("raw:", length(bytes)))
        SHA.update!(context, bytes)
        return nothing
    end
    @inbounds for element in value
        _fixed_trace_digest_value!(context, element)
    end
    return nothing
end

function _fixed_trace_digest_value!(context, value::BigFloat)
    _fixed_trace_digest_token!(
        context,
        string("BigFloat:", precision(value), ':', repr(value)),
    )
    return nothing
end

function _fixed_trace_digest_value!(context, value)
    _fixed_trace_digest_token!(
        context,
        string(typeof(value), ':', repr(value)),
    )
    return nothing
end

function _fixed_trace_dual_digest(tag::Symbol, values...)
    context = SHA.SHA2_256_CTX()
    _fixed_trace_digest_value!(context, tag)
    for value in values
        _fixed_trace_digest_value!(context, value)
    end
    return Tuple(SHA.digest!(context))
end

@inline function _fixed_trace_norm2(first, second)
    scale = max(abs(first), abs(second))
    iszero(scale) && return zero(scale)
    return scale * sqrt((first / scale)^2 + (second / scale)^2)
end

@inline function _fixed_trace_rho(first, second, tau)
    scale = max(abs(first), abs(second), abs(tau))
    iszero(scale) && return zero(scale)
    return scale * sqrt(
        (first / scale)^2 + (second / scale)^2 + (tau / scale)^2,
    )
end

function _compile_fixed_trace_q3_dual(
    problem::ConicProblem{T};
    timing::Union{Nothing,Base.RefValue}=nothing,
) where {T}
    compile_started = time_ns()
    reduction = _fixed_trace_q3_reduction(problem)
    reduction === nothing && throw(ArgumentError(
        "reduced-dual L-BFGS requires the verified FixedTraceQ3 structure",
    ))
    all(isfinite, problem.c) || throw(ArgumentError("objective must be finite"))
    all(isfinite, problem.Aeq) || throw(ArgumentError("equalities must be finite"))
    all(isfinite, problem.beq) || throw(ArgumentError("equality RHS must be finite"))

    blocks = length(reduction.blocks)
    inverse_map = alloc_zeros(T, 2, 2, blocks)
    inverse_transpose = alloc_zeros(T, 2, 2, blocks)
    objective = alloc_zeros(T, 2, blocks)
    @inbounds for block in 1:blocks
        a = reduction.tail_map[1, 1, block]
        b = reduction.tail_map[1, 2, block]
        c = reduction.tail_map[2, 1, block]
        d = reduction.tail_map[2, 2, block]
        determinant = a * d - b * c
        isfinite(determinant) && !iszero(determinant) || throw(ArgumentError(
            "FixedTraceQ3 block $block has a singular tail map",
        ))
        inverse_map[1, 1, block] = d / determinant
        inverse_map[1, 2, block] = -b / determinant
        inverse_map[2, 1, block] = -c / determinant
        inverse_map[2, 2, block] = a / determinant
        inverse_transpose[1, 1, block] =
            _ingest_owned_scalar(T, inverse_map[1, 1, block])
        inverse_transpose[1, 2, block] =
            _ingest_owned_scalar(T, inverse_map[2, 1, block])
        inverse_transpose[2, 1, block] =
            _ingest_owned_scalar(T, inverse_map[1, 2, block])
        inverse_transpose[2, 2, block] =
            _ingest_owned_scalar(T, inverse_map[2, 2, block])
        first = reduction.active_ids[1, block]
        second = reduction.active_ids[2, block]
        objective[1, block] = _ingest_owned_scalar(T, problem.c[first])
        objective[2, block] = _ingest_owned_scalar(T, problem.c[second])
    end

    panel_started = time_ns()
    panel = _owned_array_copy(T, Matrix(problem.Aeq))
    rhs = _owned_array_copy(T, problem.beq)
    panel_seconds = (time_ns() - panel_started) / 1.0e9
    active_ids = copy(reduction.active_ids)
    heads = _owned_array_copy(T, reduction.fixed_head)
    offsets = _owned_array_copy(T, reduction.offset)
    reduction_fingerprint = _fixed_trace_dual_digest(
        :fixed_trace_q3_reduction_v1,
        active_ids, reduction.tail_map, heads, offsets,
    )
    # For an eligible model the compiled reduction is a lossless description
    # of every cone affine map.  Hashing the original 3×m SparseMatrixCSC per
    # block would serialize L copies of an O(m) colptr array and reintroduce
    # the very O(L*m) setup cost this representation removes.
    problem_fingerprint = _fixed_trace_dual_digest(
        :fixed_trace_q3_problem_v2,
        string(T), problem.c, panel, rhs,
        reduction_fingerprint,
    )
    precision_bits = T === BigFloat ? Base.precision(BigFloat) : sig_bits(T)
    layout = FixedTraceQ3DualLayout{T}(
        active_ids,
        inverse_map,
        inverse_transpose,
        heads,
        offsets,
        objective,
        panel,
        rhs,
        problem_fingerprint,
        reduction_fingerprint,
        _la_arithmetic_symbol(T),
        precision_bits,
        :owned,
    )
    if timing !== nothing
        total = (time_ns() - compile_started) / 1.0e9
        timing[] = (
            fixed_trace_compile=total - panel_seconds,
            equality_panel_conversion=panel_seconds,
            reduced_dual_setup=total,
        )
    end
    return layout
end

function _fixed_trace_dual_workspace(layout::FixedTraceQ3DualLayout{T}) where {T}
    blocks = size(layout.active_ids, 2)
    variables = size(layout.equality_panel, 2)
    equalities = size(layout.equality_panel, 1)
    return (
        u=alloc_zeros(T, variables),
        x=alloc_zeros(T, variables),
        gradient=alloc_zeros(T, equalities),
        w=alloc_zeros(T, 2, blocks),
        rho=alloc_zeros(T, blocks),
        wnorm=alloc_zeros(T, blocks),
    )
end

"""
Evaluate the smoothed reduced dual and its analytic gradient in-place.

The only dense operations are `E' * y` and `E * x`; every cone block is a
fixed-size scalar kernel and the original sparse `cone.A` objects are never
visited.
"""
function _fixed_trace_dual_evaluate!(
    layout::FixedTraceQ3DualLayout{T},
    backend::AbstractLABackend,
    y::AbstractVector{T},
    tau::T,
    u::AbstractVector{T},
    x::AbstractVector{T},
    gradient::AbstractVector{T},
    w::AbstractMatrix{T},
    rho::AbstractVector{T},
    wnorm::AbstractVector{T};
    stats::Union{Nothing,FixedTraceDualEvalStats}=nothing,
) where {T}
    tau > zero(T) && isfinite(tau) || throw(ArgumentError(
        "reduced-dual smoothing must be finite and positive",
    ))
    length(y) == length(layout.equality_rhs) || throw(DimensionMismatch(
        "reduced-dual y has the wrong dimension",
    ))
    length(u) == length(x) == size(layout.equality_panel, 2) ||
        throw(DimensionMismatch("reduced-dual primal workspace is invalid"))
    size(w) == (2, size(layout.active_ids, 2)) || throw(DimensionMismatch(
        "reduced-dual block workspace is invalid",
    ))

    started = time_ns()
    la_mul_owned!(
        backend, u, transpose(layout.equality_panel), y, one(T), zero(T),
    )
    transpose_seconds = (time_ns() - started) / 1.0e9

    block_started = time_ns()
    value = zero(T)
    @inbounds for block in axes(layout.active_ids, 2)
        first = layout.active_ids[1, block]
        second = layout.active_ids[2, block]
        q1 = layout.objective[1, block] - u[first]
        q2 = layout.objective[2, block] - u[second]
        w1 = layout.inverse_transpose[1, 1, block] * q1 +
             layout.inverse_transpose[1, 2, block] * q2
        w2 = layout.inverse_transpose[2, 1, block] * q1 +
             layout.inverse_transpose[2, 2, block] * q2
        radius = _fixed_trace_norm2(w1, w2)
        smooth_radius = _fixed_trace_rho(w1, w2, tau)
        w[1, block] = w1
        w[2, block] = w2
        wnorm[block] = radius
        rho[block] = smooth_radius
        tail1 = -layout.offset[1, block] -
                layout.fixed_head[block] * w1 / smooth_radius
        tail2 = -layout.offset[2, block] -
                layout.fixed_head[block] * w2 / smooth_radius
        x[first] = layout.inverse_map[1, 1, block] * tail1 +
                   layout.inverse_map[1, 2, block] * tail2
        x[second] = layout.inverse_map[2, 1, block] * tail1 +
                    layout.inverse_map[2, 2, block] * tail2
        value += layout.fixed_head[block] * smooth_radius +
                 layout.offset[1, block] * w1 +
                 layout.offset[2, block] * w2
    end
    @inbounds for index in eachindex(y, layout.equality_rhs)
        value -= layout.equality_rhs[index] * y[index]
    end
    block_seconds = (time_ns() - block_started) / 1.0e9

    copy_owned!(gradient, layout.equality_rhs)
    @inbounds for index in eachindex(gradient)
        gradient[index] = -gradient[index]
    end
    forward_started = time_ns()
    la_mul_owned!(
        backend, gradient, layout.equality_panel, x, one(T), one(T),
    )
    forward_seconds = (time_ns() - forward_started) / 1.0e9
    if stats !== nothing
        stats.transpose_gemv_seconds += transpose_seconds
        stats.block_support_seconds += block_seconds
        stats.forward_gemv_seconds += forward_seconds
        stats.objective_evaluations += 1
        stats.gradient_evaluations += 1
        stats.transpose_gemv_calls += 1
        stats.forward_gemv_calls += 1
    end
    # A non-finite line-search trial is a rejected numerical candidate, not a
    # malformed API call.  Returning Inf lets the typed L-BFGS core backtrack
    # deterministically; shape/tau contract violations still throw above.
    return isfinite(value) && all(isfinite, gradient) ? value : T(Inf)
end

"""Reconstruct original Lorentz coordinates for cold certification."""
function _fixed_trace_dual_reconstruct(
    layout::FixedTraceQ3DualLayout{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractMatrix{T},
    wnorm::AbstractVector{T},
) where {T}
    blocks = size(layout.active_ids, 2)
    slack = [alloc_zeros(T, 3) for _ in 1:blocks]
    dual = [alloc_zeros(T, 3) for _ in 1:blocks]
    @inbounds for block in 1:blocks
        first = layout.active_ids[1, block]
        second = layout.active_ids[2, block]
        # Recover M from inv(M) without consulting the original sparse cone.
        i11 = layout.inverse_map[1, 1, block]
        i12 = layout.inverse_map[1, 2, block]
        i21 = layout.inverse_map[2, 1, block]
        i22 = layout.inverse_map[2, 2, block]
        deti = i11 * i22 - i12 * i21
        m11 = i22 / deti
        m12 = -i12 / deti
        m21 = -i21 / deti
        m22 = i11 / deti
        slack[block][1] = layout.fixed_head[block]
        slack[block][2] = m11 * x[first] + m12 * x[second] +
                          layout.offset[1, block]
        slack[block][3] = m21 * x[first] + m22 * x[second] +
                          layout.offset[2, block]
        # The certificate is for the original nonsmoothed dual cone.  Its head
        # is ||w||, not the smoothing radius rho.
        dual[block][1] = _ingest_owned_scalar(T, wnorm[block])
        dual[block][2] = _ingest_owned_scalar(T, w[1, block])
        dual[block][3] = _ingest_owned_scalar(T, w[2, block])
    end
    return (
        x=_owned_array_copy(T, x),
        slack,
        dual,
        equality_dual=_owned_array_copy(T, y),
    )
end
