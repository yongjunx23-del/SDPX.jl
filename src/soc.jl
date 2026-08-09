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
Result in original Lorentz coordinates. `lifted` is retained as a compatibility
and validation reference while the compact Newton backend is introduced.
"""
struct ConicResult{T}
    status::SolveStatus
    message::String
    x::Vector{T}
    slack::Vector{Vector{T}}
    dual::Vector{Vector{T}}
    pObj::T
    dObj::T
    gap_rel::T
    p_res::T
    d_res::T
    iterations::Int
    diagnostics::Union{Nothing,SolveDiagnostics}
    lifted::SDPResult{T}
end

"""Lorentz Jordan product `(t,u) o (s,v)`."""
function _soc_jordan!(destination, left, right)
    length(destination) == length(left) == length(right) ||
        throw(DimensionMismatch("Lorentz vectors must have equal dimensions"))
    value = left[1] * right[1]
    @inbounds for index in 2:length(left)
        value += left[index] * right[index]
    end
    destination[1] = value
    @inbounds for index in 2:length(left)
        destination[index] =
            left[1] * right[index] + right[1] * left[index]
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
    vector[1] > zero(eltype(vector)) && _soc_determinant(vector) > zero(eltype(vector))

"""Exact inverse in the Lorentz Jordan algebra."""
function _soc_inverse!(destination, source)
    determinant = _soc_determinant(source)
    determinant > zero(determinant) ||
        throw(ArgumentError("the Lorentz vector is not in the cone interior"))
    destination[1] = source[1] / determinant
    @inbounds for index in 2:length(source)
        destination[index] = -source[index] / determinant
    end
    return destination
end

"""
    _soc_fraction_to_boundary(s, ds)

Largest step in `[0,1]` for which `s + alpha*ds` remains in the closed
Lorentz cone. The determinant is quadratic, so no backtracking loop or matrix
factorization is required.
"""
function _soc_fraction_to_boundary(s, ds)
    length(s) == length(ds) || throw(DimensionMismatch())
    T = eltype(s)
    full_first = s[1] + ds[1]
    full_det = full_first * full_first
    @inbounds for index in 2:length(s)
        value = s[index] + ds[index]
        full_det -= value * value
    end
    full_first >= zero(T) && full_det >= zero(T) && return one(T)

    c0 = _soc_determinant(s)
    c1 = (one(T) + one(T)) * s[1] * ds[1]
    c2 = ds[1] * ds[1]
    @inbounds for index in 2:length(s)
        c1 -= (one(T) + one(T)) * s[index] * ds[index]
        c2 -= ds[index] * ds[index]
    end
    root = one(T)
    if iszero(c2)
        c1 < zero(T) && (root = min(root, -c0 / c1))
    else
        discriminant = max(c1 * c1 - T(4) * c2 * c0, zero(T))
        square_root = sqrt(discriminant)
        denominator = T(2) * c2
        first = (-c1 - square_root) / denominator
        second = (-c1 + square_root) / denominator
        zero(T) < first <= root && (root = first)
        zero(T) < second <= root && (root = second)
    end
    if ds[1] < zero(T)
        root = min(root, -s[1] / ds[1])
    end
    return clamp(root, zero(T), one(T))
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

function _soc_arrow_matrix(vector::AbstractVector{T}) where {T}
    dimension = length(vector)
    if dimension == 3
        matrix = alloc_zeros(T, 2, 2)
        matrix[1, 1] = vector[1] + vector[2]
        matrix[2, 2] = vector[1] - vector[2]
        matrix[1, 2] = vector[3]
        matrix[2, 1] = vector[3]
        return matrix
    end
    matrix = alloc_zeros(T, dimension, dimension)
    @inbounds begin
        for index in 1:dimension
            matrix[index, index] = vector[1]
        end
        for index in 2:dimension
            matrix[1, index] = vector[index]
            matrix[index, 1] = vector[index]
        end
    end
    return matrix
end

function _soc_coefficient_matrix(::Type{T}, cone::SOCConstraint{T}, variable::Int) where {T}
    dimension = length(cone.b)
    if dimension == 3
        first = cone.A[1, variable]
        second = cone.A[2, variable]
        third = cone.A[3, variable]
        rows = Int[]
        columns = Int[]
        values = T[]
        first_diagonal = first + second
        second_diagonal = first - second
        if !iszero(first_diagonal)
            push!(rows, 1); push!(columns, 1); push!(values, first_diagonal)
        end
        if !iszero(second_diagonal)
            push!(rows, 2); push!(columns, 2); push!(values, second_diagonal)
        end
        if !iszero(third)
            push!(rows, 1); push!(columns, 2); push!(values, third)
            push!(rows, 2); push!(columns, 1); push!(values, third)
        end
        return sparse(rows, columns, values, 2, 2)
    end
    rows = Int[]
    columns = Int[]
    values = T[]
    head = cone.A[1, variable]
    if !iszero(head)
        for index in 1:dimension
            push!(rows, index); push!(columns, index); push!(values, head)
        end
    end
    @inbounds for index in 2:dimension
        value = cone.A[index, variable]
        iszero(value) && continue
        push!(rows, 1); push!(columns, index); push!(values, value)
        push!(rows, index); push!(columns, 1); push!(values, value)
    end
    return sparse(rows, columns, values, dimension, dimension)
end

"""Compile the compact SOC model to the exact PSD-arrow reference model."""
function _soc_psd_lift(problem::ConicProblem{T}; sparse=:auto, verbosity=1) where {T}
    blocks = SparseCoefficientVector{T}[]
    constants = Matrix{T}[]
    sizehint!(blocks, length(problem.cones))
    sizehint!(constants, length(problem.cones))
    for cone in problem.cones
        dimension = length(cone.b)
        side = dimension == 3 ? 2 : dimension
        ids = Int[]
        coefficients = SparseMatrixCSC{T,Int}[]
        for variable in 1:problem.variables
            matrix = _soc_coefficient_matrix(T, cone, variable)
            nnz(matrix) == 0 && continue
            push!(ids, variable)
            push!(coefficients, matrix)
        end
        push!(blocks, ActiveSparseCoefficientVector(
            T,
            problem.variables,
            ids,
            coefficients,
            side,
        ))
        push!(constants, -_soc_arrow_matrix(cone.b))
    end
    return ingest(
        problem.c,
        blocks,
        constants,
        transpose(problem.Aeq),
        problem.beq;
        T=T,
        sparse=sparse,
        validate=true,
        symmetrize=false,
        verbosity=verbosity,
    )
end

function _conic_result(problem::ConicProblem{T}, result::SDPResult{T}) where {T}
    slack = Vector{Vector{T}}(undef, length(problem.cones))
    dual = Vector{Vector{T}}(undef, length(problem.cones))
    for block in eachindex(problem.cones)
        dimension = length(problem.cones[block].b)
        primal_matrix = result.X[block]
        dual_matrix = result.Y[block]
        primal = Vector{T}(undef, dimension)
        dual_vector = Vector{T}(undef, dimension)
        if dimension == 3
            two = one(T) + one(T)
            primal[1] = (primal_matrix[1, 1] + primal_matrix[2, 2]) / two
            primal[2] = (primal_matrix[1, 1] - primal_matrix[2, 2]) / two
            primal[3] = (primal_matrix[1, 2] + primal_matrix[2, 1]) / two
            dual_vector[1] = dual_matrix[1, 1] + dual_matrix[2, 2]
            dual_vector[2] = dual_matrix[1, 1] - dual_matrix[2, 2]
            dual_vector[3] = dual_matrix[1, 2] + dual_matrix[2, 1]
        else
            primal[1] = primal_matrix[1, 1]
            dual_vector[1] = tr(dual_matrix)
            @inbounds for index in 2:dimension
                primal[index] = primal_matrix[index, 1]
                dual_vector[index] = T(2) * dual_matrix[1, index]
            end
        end
        slack[block] = primal
        dual[block] = dual_vector
    end
    return ConicResult{T}(
        result.status,
        result.message,
        result.x,
        slack,
        dual,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.diagnostics,
        result,
    )
end

"""
    solve_socp(problem; kwargs...)

Solve a compact LP+SOC model. The current reference implementation compiles
the model once to an exact PSD arrow; the separate representation and result
boundary allow the native Lorentz Newton backend to replace this reference
without changing the public API.
"""
function solve_socp(problem::ConicProblem{T}; sparse=:auto, verbosity::Int=1, kwargs...) where {T}
    lifted = _soc_psd_lift(problem; sparse=sparse, verbosity=verbosity)
    result = solve(lifted; verbosity=verbosity, kwargs...)
    return _conic_result(problem, result)
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
