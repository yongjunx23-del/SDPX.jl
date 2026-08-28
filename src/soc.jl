"""
    SOCConstraint(A, b)

One affine Lorentz-cone constraint `A * x + b in Q`, where
`Q = {(t, u): t >= norm(u)}`. The rows of `A` and entries of `b` use the
physical SOC coordinates directly; no PSD arrow matrix is stored here.
"""
struct SOCConstraint{T}
    A::Union{Matrix{T},SparseMatrixCSC{T,Int}}
    b::Vector{T}
end

function SOCConstraint(
    A::AbstractMatrix,
    b::AbstractVector;
    T::Union{Nothing,Type}=nothing,
)
    size(A, 1) == length(b) || throw(DimensionMismatch(
        "the row count of A must equal length(b)",
    ))
    length(b) > 0 || throw(ArgumentError("an SOC block must be nonempty"))
    ET = T === nothing ?
         _require_supported_arithmetic_type(float(promote_type(eltype(A), eltype(b)))) :
         _require_supported_arithmetic_type(T)
    matrix = A isa SparseMatrixCSC ?
             _ingest_owned_sparse(ET, A) : _ingest_owned_array(ET, A)
    offset = _ingest_owned_array(ET, b)
    _check_finite(matrix, "SOC A")
    _check_finite(offset, "SOC b")
    return SOCConstraint{ET}(matrix, offset)
end

"""Compact standard-form LP+SOC model used by the native conic frontend."""
struct ConicProblem{T}
    c::Vector{T}
    cones::Vector{SOCConstraint{T}}
    Aeq::Union{Matrix{T},SparseMatrixCSC{T,Int}}
    beq::Vector{T}
    variables::Int
end

Base.eltype(::ConicProblem{T}) where {T} = T

"""
Compatibility result in original Lorentz coordinates. Qualified ConicProblem
entrypoints adapt product-HSD results into this layout; no PSD lift or separate
SOC solver result is stored.
"""
struct ConicResult{T,D} <: AbstractCoreResult{T}
    status::SolveStatus
    message::String
    x::Vector{T}
    slack::Vector{Vector{T}}
    dual::Vector{Vector{T}}
    equality_dual::Vector{T}
    pObj::T
    dObj::T
    gap_rel::T
    p_res::T
    d_res::T
    iterations::Int
    diagnostics::D
end

function ConicResult{T}(
    status::SolveStatus,
    message::String,
    x::Vector{T},
    slack::Vector{Vector{T}},
    dual::Vector{Vector{T}},
    equality_dual::Vector{T},
    pObj::T,
    dObj::T,
    gap_rel::T,
    p_res::T,
    d_res::T,
    iterations::Int,
    diagnostics::D,
) where {T,D}
    return ConicResult{T,D}(
        status,
        message,
        x,
        slack,
        dual,
        equality_dual,
        pObj,
        dObj,
        gap_rel,
        p_res,
        d_res,
        iterations,
        diagnostics,
    )
end

"""Native SOCP compatibility aliases for common result-inspection fields."""
function Base.getproperty(result::ConicResult, name::Symbol)
    name === :y && return getfield(result, :equality_dual)
    if name in (:termination, :timings, :restarts, :regularizations,
                :parameter_history)
        diagnostics = getfield(result, :diagnostics)
        name === :termination && return diagnostics === nothing ?
                                      (reason=:unavailable,) :
                                      diagnostics.termination
        name === :timings && return diagnostics === nothing ?
                                  nothing : diagnostics.timings
        name === :restarts && return 0
        name === :regularizations && return diagnostics === nothing ? 0 :
            get(diagnostics.termination, :regularizations, 0)
        return NamedTuple[]
    end
    return getfield(result, name)
end

function Base.propertynames(result::ConicResult, private::Bool=false)
    fields = fieldnames(typeof(result))
    aliases = (
        :y, :termination, :timings, :restarts, :regularizations,
        :parameter_history,
    )
    return private ? (fields..., aliases...) : (fields..., aliases...)
end

"""Lorentz Jordan product `(t,u) o (s,v)`."""
function _soc_jordan!(destination, left, right)
    length(destination) == length(left) == length(right) ||
        throw(DimensionMismatch("Lorentz vectors must have equal dimensions"))
    # Cache both heads before writing.  This makes the kernel safe when the
    # destination aliases either input, which is useful for scaled Newton
    # corrections and avoids an otherwise silent tail corruption.
    left_head = left[1]
    right_head = right[1]
    value = left_head * right_head
    @inbounds for index in 2:length(left)
        value += left[index] * right[index]
    end
    destination[1] = value
    @inbounds for index in 2:length(left)
        destination[index] =
            left_head * right[index] + right_head * left[index]
    end
    return destination
end

@inline function _soc_determinant(vector)
    value = vector[1] * vector[1]
    @inbounds for index in 2:length(vector)
        value -= vector[index] * vector[index]
    end
    return value
end

@inline _soc_is_interior(vector) =
    vector[1] > zero(eltype(vector)) && _soc_margin(vector) > zero(eltype(vector))

"""
    _soc_fraction_to_boundary(s, ds)

Largest step in `[0,1]` for which `s + alpha*ds` remains in the closed
Lorentz cone. The determinant is quadratic, so no backtracking loop or matrix
factorization is required.
"""
function _soc_fraction_to_boundary(s, ds)
    length(s) == length(ds) || throw(DimensionMismatch())
    T = promote_type(eltype(s), eltype(ds))
    z = zero(T)
    o = one(T)
    two = o + o
    four = two + two

    # Normalize before forming determinant coefficients.  Cone membership is
    # homogeneous, so the roots are unchanged while representable large/small
    # inputs avoid overflow and underflow.
    scale = z
    @inbounds for index in eachindex(s, ds)
        scale = max(scale, abs(T(s[index])), abs(T(ds[index])))
    end
    iszero(scale) && return o

    s0 = T(s[1]) / scale
    d0 = T(ds[1]) / scale
    c0 = s0 * s0
    c1 = two * s0 * d0
    c2 = d0 * d0
    full_first = s0 + d0
    full_det = full_first * full_first
    @inbounds for index in 2:length(s)
        si = T(s[index]) / scale
        di = T(ds[index]) / scale
        c0 -= si * si
        c1 -= two * si * di
        c2 -= di * di
        full_value = si + di
        full_det -= full_value * full_value
    end

    # Preserve the literal closed-cone contract for boundary diagnostics.
    if s0 == z && d0 < z
        return z
    elseif c0 == z && (c1 < z || (iszero(c1) && c2 < z))
        return z
    end
    full_first >= z && full_det >= z && return o

    root = o
    if iszero(c2)
        c1 < z && (root = min(root, -c0 / c1))
    else
        discriminant = max(c1 * c1 - four * c2 * c0, z)
        square_root = sqrt(discriminant)
        # Stable q-formula: form the well-separated numerator and recover the
        # second root from their product c0/c2.
        q = c1 >= z ? -(c1 + square_root) / two :
            -(c1 - square_root) / two
        if !iszero(q)
            first = q / c2
            second = c0 / q
            z < first <= root && (root = first)
            z < second <= root && (root = second)
        else
            repeated = -c1 / (two * c2)
            z < repeated <= root && (root = repeated)
        end
    end
    if d0 < z
        root = min(root, -s0 / d0)
    end
    return clamp(root, z, o)
end

function _convert_soc_constraint(::Type{T}, cone::SOCConstraint) where {T}
    return SOCConstraint(cone.A, cone.b; T=T)
end

"""
    second_order_program(c, cones; Aeq=nothing, beq=nothing, T=nothing)

Build a compact LP+SOC model. Each element of `cones` is an
[`SOCConstraint`](@ref). Equalities use the conventional row-oriented form
`Aeq*x = beq`.
"""
function second_order_program(
    c::AbstractVector,
    cones::AbstractVector{<:SOCConstraint};
    Aeq=nothing,
    beq=nothing,
    T::Union{Nothing,Type}=nothing,
)
    isempty(cones) && throw(ArgumentError("at least one SOC block is required"))
    equality_source = Aeq === nothing ? spzeros(Float64, 0, length(c)) : Aeq
    rhs_source = beq === nothing ? Float64[] : beq
    ET = T === nothing ? _require_supported_arithmetic_type(float(promote_type(
        eltype(c),
        mapreduce(cone -> eltype(cone.A), promote_type, cones),
        eltype(equality_source),
        eltype(rhs_source),
    ))) : _require_supported_arithmetic_type(T)
    variables = length(c)
    converted = SOCConstraint{ET}[
        _convert_soc_constraint(ET, cone) for cone in cones
    ]
    all(size(cone.A, 2) == variables for cone in converted) ||
        throw(DimensionMismatch("every SOC matrix must have $variables columns"))
    equality_rows = size(equality_source, 1)
    size(equality_source, 2) == variables ||
        throw(DimensionMismatch("Aeq must have $variables columns"))
    length(rhs_source) == equality_rows ||
        throw(DimensionMismatch("length(beq) must equal the row count of Aeq"))
    equality = equality_source isa SparseMatrixCSC ?
               _ingest_owned_sparse(ET, equality_source) :
               _ingest_owned_array(ET, equality_source)
    rhs = _ingest_owned_array(ET, rhs_source)
    return ConicProblem{ET}(
        _ingest_owned_array(ET, c),
        converted,
        equality,
        rhs,
        variables,
    )
end

"""Stacked-matrix convenience form for `second_order_program`."""
function second_order_program(
    c::AbstractVector,
    G::AbstractMatrix,
    h::AbstractVector;
    cone_dims::AbstractVector{<:Integer}=[size(G, 1)],
    kwargs...,
)
    sum(cone_dims) == size(G, 1) == length(h) || throw(DimensionMismatch(
        "sum(cone_dims), size(G,1), and length(h) must agree",
    ))
    all(>(0), cone_dims) || throw(ArgumentError("cone dimensions must be positive"))
    cones = SOCConstraint[]
    first_row = 1
    for dimension in cone_dims
        last_row = first_row + dimension - 1
        push!(cones, SOCConstraint(G[first_row:last_row, :], h[first_row:last_row]))
        first_row = last_row + 1
    end
    return second_order_program(c, cones; kwargs...)
end

solve(problem::ConicProblem; kwargs...) = solve_socp(problem; kwargs...)

"""Fixed-trace classification for one PSD block."""
struct FixedTraceBlock{T}
    block::Int
    dimension::Int
    kind::Symbol
    trace::T
    source::Symbol
    relation_residual::T
    equality_coefficients::Vector{T}
end

"""Result of conservative fixed-trace analysis in original coordinates."""
struct FixedTraceAnalysis{T}
    blocks::Vector{FixedTraceBlock{T}}
    fixed_blocks::Int
    soc_blocks::Int
    traceless_sdp_blocks::Int
    zero_blocks::Int
    infeasible_blocks::Vector{Int}
end

function _block_trace_coefficients(prob::SDPProblem{T}, block::Int) where {T}
    coefficients = alloc_zeros(T, prob.dims.m)
    dimension = prob.dims.k[block]
    if prob.cons isa DenseCons{T}
        panel = (prob.cons::DenseCons{T}).Av[block]
        @inbounds for variable in 1:prob.dims.m
            value = zero(T)
            for diagonal in 1:dimension
                value += panel[diagonal + (diagonal - 1) * dimension, variable]
            end
            coefficients[variable] = value
        end
    else
        sparse_cons = prob.cons::SparseCons{T}
        @inbounds for variable in sparse_cons.active[block]
            coefficients[variable] = tr(sparse_cons.Asp[block][variable])
        end
    end
    return coefficients
end

"""
Inspect a block's trace coefficients without materialising the historical
dense length-`m` vector. Large fixed-trace bootstrap models normally have only
two active variables per block, so this keeps direct-trace detection O(nnz)
instead of O(Lm).

For packed 2x2 blocks the exact test is `a11 == -a22`. Besides avoiding an
MPFR temporary, this recognises the structural traceless pair in the original
arithmetic.
"""
function _block_trace_summary(prob::SDPProblem{T}, block::Int) where {T}
    dimension = prob.dims.k[block]
    largest = zero(T)
    exact = true
    if prob.cons isa SparseCons{T}
        sparse_cons = prob.cons::SparseCons{T}
        active = sparse_cons.active[block]
        packed = sparse_cons.packed2[block]
        if dimension == 2 && size(packed, 1) == 3 &&
           size(packed, 2) == length(active)
            @inbounds for position in eachindex(active)
                a11 = packed[1, position]
                a22 = packed[3, position]
                a11 == -a22 && continue
                value = a11 + a22
                exact = false
                largest = max(largest, abs(value))
            end
        else
            @inbounds for variable in active
                value = tr(sparse_cons.Asp[block][variable])
                exact &= iszero(value)
                largest = max(largest, abs(value))
            end
        end
    else
        panel = (prob.cons::DenseCons{T}).Av[block]
        @inbounds for variable in 1:prob.dims.m
            value = zero(T)
            for diagonal in 1:dimension
                value += panel[
                    diagonal + (diagonal - 1) * dimension,
                    variable,
                ]
            end
            exact &= iszero(value)
            largest = max(largest, abs(value))
        end
    end
    return exact, largest
end

function _fixed_trace_tolerance(::Type{T}, scale, rows::Int, columns::Int) where {T}
    return T(128) * eps(T) * T(max(rows, columns, 1)) * max(scale, one(T))
end

"""
    analyze_fixed_trace(problem; tolerance=nothing)

Detect constant PSD-block traces directly or through `B' * x = b`. Only a
verified original-arithmetic relation is classified as fixed. A caller may
provide a tighter or looser absolute residual tolerance; the default is a
dimension-scaled machine-precision guard.
"""
function analyze_fixed_trace(
    prob::SDPProblem{T};
    tolerance::Union{Nothing,Real}=nothing,
) where {T}
    blocks = FixedTraceBlock{T}[]
    infeasible = Int[]
    equality_factor = nothing
    # Equality-implied trace detection is useful for small PSD blocks and
    # modest analysis problems. A dense QR plus one solve per large block is
    # not a conservative preprocessing cost on Task_Low08-scale inputs, and
    # no larger-block basis reduction is promoted yet. Keep those cases
    # analysis-only until a batched sparse relation solver is selected.
    equality_relation_work =
        prob.dims.m * prob.dims.n * max(prob.dims.L, 1)
    equality_relation_budget = 2_000_000
    for block in 1:prob.dims.L
        direct, largest_trace_coefficient =
            _block_trace_summary(prob, block)
        scale = max(
            largest_trace_coefficient,
            abs(tr(prob.C[block])),
            one(T),
        )
        threshold = tolerance === nothing ?
                    _fixed_trace_tolerance(T, scale, prob.dims.m, prob.dims.n) :
                    T(tolerance)
        # Direct blocks need no equality relation. Retaining an `n`-vector for
        # each of J80's 32,800 blocks consumed hundreds of MiB without being
        # used by the solve.
        relation = T[]
        residual = zero(T)
        source = :none
        # A direct fixed trace is structural only when every trace coefficient
        # is exactly zero in the ingested arithmetic. Tiny nonzero coefficients
        # can multiply unbounded variables, so treating them as zero would be
        # an unsound infeasibility reduction. Record them as near-direct
        # diagnostics but leave the model unchanged.
        fixed = direct
        near_direct = !fixed && largest_trace_coefficient <= threshold
        trace_value = -tr(prob.C[block])
        if fixed
            source = :direct
        elseif prob.dims.n > 0 &&
               equality_relation_work <= equality_relation_budget
            try
                equality_factor === nothing &&
                    (equality_factor = qr(Matrix(prob.B), ColumnNorm()))
                coefficients = _block_trace_coefficients(prob, block)
                relation = equality_factor \ coefficients
                reconstructed = prob.B * relation
                residual = knrmInf(reconstructed - coefficients)
                fixed = residual <= threshold
                if fixed
                    source = :equalities
                    trace_value = dot(relation, prob.b) - tr(prob.C[block])
                elseif near_direct
                    source = :near_direct
                end
            catch exception
                _recoverable(exception) || rethrow()
                fixed = false
                residual = T(Inf)
            end
        elseif near_direct
            source = :near_direct
            residual = largest_trace_coefficient
        elseif prob.dims.n > 0
            source = :equality_scan_skipped
        end
        kind = :variable_trace
        if fixed
            if trace_value < -threshold
                kind = :infeasible
                push!(infeasible, block)
            elseif abs(trace_value) <= threshold
                kind = :zero
            elseif prob.dims.k[block] == 1
                kind = :fixed_scalar
            elseif prob.dims.k[block] == 2
                kind = :soc
            else
                kind = :traceless_sdp
            end
        end
        push!(blocks, FixedTraceBlock{T}(
            block,
            prob.dims.k[block],
            kind,
            trace_value,
            source,
            residual,
            relation,
        ))
    end
    return FixedTraceAnalysis{T}(
        blocks,
        count(block -> block.kind != :variable_trace, blocks),
        count(block -> block.kind === :soc, blocks),
        count(block -> block.kind === :traceless_sdp, blocks),
        count(block -> block.kind === :zero, blocks),
        infeasible,
    )
end
