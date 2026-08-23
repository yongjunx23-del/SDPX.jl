const _DEFAULT_REFERENCE = BenchmarkReference(
    :optimal,
    nothing,
    1.0e-7,
    1.0e-7,
    "status/residual/certificate regression",
)

function _sdp_ingest(c, A, C, B, b; sparse=false)
    return SDPX.ingest(c, A, C, B, b; sparse=sparse, verbosity=0)
end

_typed_decimal(::Type{BigFloat}, value::AbstractString) = parse(BigFloat, value)
_typed_decimal(::Type{T}, value::AbstractString) where {T} =
    T(parse(BigFloat, value))

function _build_lp_box(::Type{T}; variables::Int=3) where {T}
    objective = T.(1:variables)
    identity = Matrix{T}(I, variables, variables)
    G = vcat(identity, -identity)
    h = vcat(ones(T, variables), fill(T(-3), variables))
    return (
        problem=SDPX.linear_program(
            objective, G, h; T=T, sparse=false, verbosity=0,
        ),
        expected=sum(objective),
        kind=:sdp,
    )
end

function _build_lp_sparse(::Type{T}) where {T}
    G = sparse([1, 2, 3, 3], [1, 2, 1, 2], T[1, 1, 1, 1], 3, 2)
    Aeq = sparse([1, 1], [1, 2], T[1, 1], 1, 2)
    return (
        problem=SDPX.linear_program(
            T[1, 2], G, T[1, 1, 3];
            Aeq=Aeq, beq=T[3], T=T, sparse=true, verbosity=0,
        ),
        expected=T(4),
        kind=:sdp,
    )
end

function _build_lp_equality(::Type{T}; delta="0.001", rank=:full) where {T}
    δ = _typed_decimal(T, delta)
    if rank === :exact_deficient
        δ = zero(T)
    end
    G = T[1 0; 0 1; -1 0; 0 -1]
    Aeq = T[1 1; 1 1 + δ]
    objective = rank === :exact_deficient ? T[1, 1] : T[1, 2]
    return (
        problem=SDPX.linear_program(
            objective, G, T[0, 0, -2, -2];
            Aeq=Aeq, beq=T[2, 2 + δ], T=T, verbosity=0,
        ),
        # Float64's default equality-rank policy deliberately drops the most
        # nearly dependent row. Extended arithmetic keeps it at the tighter
        # benchmark tolerance and therefore retains the mathematical optimum.
        expected=rank === :exact_deficient ? T(2) :
                 (T === Float64 && δ <= T(1e-8) ? T(2) : T(3)),
        kind=:sdp,
    )
end

function _build_lp_scaling(::Type{T}; decades::Int=4) where {T}
    scale = T(10)^decades
    G = T[scale 0; 0 inv(scale); -scale 0; 0 -inv(scale)]
    h = T[scale, inv(scale), -T(3)*scale, -T(3)*inv(scale)]
    return (
        problem=SDPX.linear_program(
            T[1, 2], G, h; T=T, sparse=false, verbosity=0,
        ),
        expected=T(3),
        kind=:sdp,
    )
end

function _build_soc_dimension(::Type{T}; dimension::Int=3) where {T}
    objective = zeros(T, dimension)
    objective[1] = one(T)
    Aeq = zeros(T, dimension - 1, dimension)
    @inbounds for row in axes(Aeq, 1)
        Aeq[row, row + 1] = one(T)
    end
    return (
        problem=SDPX.second_order_program(
            objective,
            Matrix{T}(I, dimension, dimension),
            zeros(T, dimension);
            Aeq=Aeq,
            beq=ones(T, dimension - 1),
        ),
        expected=sqrt(T(dimension - 1)),
        kind=:socp,
    )
end

function _build_soc_mixed(::Type{T}) where {T}
    disk = SDPX.SOCConstraint(Matrix{T}(I, 3, 3), zeros(T, 3); T=T)
    nonnegative = SDPX.SOCConstraint(
        reshape(T[0, 1, 0], 1, 3), T[0]; T=T,
    )
    return (
        problem=SDPX.second_order_program(
            T[1, 0, 0], [disk, nonnegative];
            Aeq=T[0 1 0; 0 0 1], beq=T[3, 4], T=T,
        ),
        expected=T(5),
        kind=:socp,
    )
end

function _build_soc_scaling(::Type{T}; decades::Int=4) where {T}
    scale = T(10)^decades
    # Preserve the same Lorentz geometry while scaling only the equality rows.
    # This separates frontend/equality scaling stress from a changed optimum.
    A = Matrix{T}(I, 3, 3)
    Aeq = T[0 scale 0; 0 0 inv(scale)]
    return (
        problem=SDPX.second_order_program(
            T[1, 0, 0], A, zeros(T, 3);
            Aeq=Aeq, beq=T[T(3)*scale, T(4)*inv(scale)],
        ),
        expected=T(5),
        kind=:socp,
    )
end

function _build_soc_many_blocks(
    ::Type{T};
    blocks::Int=64,
    dimension::Int=8,
) where {T}
    # A block-diagonal second-order program: `blocks` independent
    # `dimension`-dim Lorentz cones sharing a dense objective. This is the
    # many-small-cones shape that dominates structured SOCP applications,
    # as opposed to the single-large-cone shape of the soc_dimension ladder.
    cones = Vector{SDPX.SOCConstraint{T}}(undef, blocks)
    for b in 1:blocks
        # Each cone's selection matrix spans the full variable axis: rows
        # (b-1)*dimension+1 : b*dimension form an identity slice.
        A_b = zeros(T, dimension, blocks * dimension)
        for i in 1:dimension
            A_b[i, (b-1)*dimension+i] = one(T)
        end
        cones[b] = SDPX.SOCConstraint(
            A_b, zeros(T, dimension); T=T,
        )
    end
    variables = blocks * dimension
    objective = zeros(T, variables)
    return (
        problem=SDPX.second_order_program(objective, cones; T=T),
        expected=zero(T),
        kind=:socp,
    )
end

function _build_sdp_dense(::Type{T}) where {T}
    coefficients = zeros(T, 2, 2, 2)
    coefficients[1, 1, 1] = one(T)
    coefficients[2, 2, 2] = one(T)
    return (
        problem=_sdp_ingest(
            T[2, 3], [coefficients], [T[0 1; 1 0]],
            Matrix{T}(undef, 2, 0), T[],
        ),
        expected=T(2) * sqrt(T(6)),
        kind=:sdp,
    )
end

function _build_sdp_many_small(::Type{T}; blocks::Int=6) where {T}
    variables = blocks
    coefficients = [zeros(T, variables, 2, 2) for _ in 1:blocks]
    @inbounds for block in 1:blocks
        coefficients[block][block, :, :] .= Matrix{T}(I, 2, 2)
    end
    return (
        problem=_sdp_ingest(
            ones(T, variables), coefficients,
            [Matrix{T}(I, 2, 2) for _ in 1:blocks],
            Matrix{T}(undef, variables, 0), T[],
        ),
        expected=T(variables),
        kind=:sdp,
    )
end

function _build_sdp_sparse(::Type{T}; dimension::Int=4) where {T}
    identity = sparse(Matrix{T}(I, dimension, dimension))
    return (
        problem=_sdp_ingest(
            T[1], [[identity]], [Matrix{T}(I, dimension, dimension)],
            spzeros(T, 1, 0), T[]; sparse=true,
        ),
        expected=one(T),
        kind=:sdp,
    )
end

function _build_sdp_equality(::Type{T}; variables::Int=5) where {T}
    variables >= 3 || throw(ArgumentError("variables must be at least 3"))
    coefficients = zeros(T, variables, variables, variables)
    @inbounds for variable in 1:variables
        coefficients[variable, variable, variable] = one(T)
    end
    Aeq = zeros(T, variables - 1, variables)
    @inbounds for row in axes(Aeq, 1)
        Aeq[row, row] = one(T)
        Aeq[row, row + 1] = one(T)
    end
    return (
        problem=_sdp_ingest(
            ones(T, variables), [coefficients],
            [Matrix{T}(I, variables, variables)],
            transpose(Aeq), fill(T(2), variables - 1),
        ),
        expected=T(variables),
        kind=:sdp,
    )
end

function _build_sdp_block_arrow(::Type{T}; blocks::Int=5) where {T}
    variables = blocks + 1
    coefficients = Vector{Vector{SparseMatrixCSC{T,Int}}}(undef, blocks)
    @inbounds for block in 1:blocks
        local_blocks = [spzeros(T, 2, 2) for _ in 1:variables]
        local_blocks[1] = sparse(Matrix{T}(I, 2, 2))
        local_blocks[block + 1] = sparse([1, 2], [2, 1], T[1, 1], 2, 2)
        coefficients[block] = local_blocks
    end
    objective = zeros(T, variables)
    objective[1] = one(T)
    return (
        problem=_sdp_ingest(
            objective, coefficients,
            [Matrix{T}(I, 2, 2) for _ in 1:blocks],
            spzeros(T, variables, 0), T[]; sparse=true,
        ),
        expected=one(T),
        kind=:sdp,
    )
end

function _build_sdp_small_eigenvalue(
    ::Type{T}; dimension::Int=4, epsilon="1e-8", seed::Int=71,
) where {T}
    epsilon_value = _typed_decimal(T, epsilon)
    rng = MersenneTwister(seed)
    diagonal = ones(T, dimension)
    diagonal[end] = epsilon_value
    permutation = randperm(rng, dimension)
    diagonal = diagonal[permutation]
    coefficients = zeros(T, 1, dimension, dimension)
    @inbounds for index in 1:dimension
        coefficients[1, index, index] = -one(T)
    end
    offset = -Matrix(Diagonal(diagonal))
    return (
        problem=_sdp_ingest(
            T[-1], [coefficients], [offset],
            Matrix{T}(undef, 1, 0), T[],
        ),
        expected=-epsilon_value,
        kind=:sdp,
    )
end

function build_generated_problem(loader::Symbol, ::Type{T}; kwargs...) where {T}
    loader === :lp_box && return _build_lp_box(T; kwargs...)
    loader === :lp_sparse && return _build_lp_sparse(T; kwargs...)
    loader === :lp_equality && return _build_lp_equality(T; kwargs...)
    loader === :lp_scaling && return _build_lp_scaling(T; kwargs...)
    loader === :soc_dimension && return _build_soc_dimension(T; kwargs...)
    loader === :soc_many_blocks &&
        return _build_soc_many_blocks(T; kwargs...)
    loader === :soc_mixed && return _build_soc_mixed(T; kwargs...)
    loader === :soc_scaling && return _build_soc_scaling(T; kwargs...)
    loader === :sdp_dense && return _build_sdp_dense(T; kwargs...)
    loader === :sdp_many_small && return _build_sdp_many_small(T; kwargs...)
    loader === :sdp_sparse && return _build_sdp_sparse(T; kwargs...)
    loader === :sdp_equality && return _build_sdp_equality(T; kwargs...)
    loader === :sdp_block_arrow && return _build_sdp_block_arrow(T; kwargs...)
    loader === :sdp_small_eigenvalue &&
        return _build_sdp_small_eigenvalue(T; kwargs...)
    throw(ArgumentError("unsupported generated benchmark loader $loader"))
end
