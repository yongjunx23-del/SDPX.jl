#=====================================================================#
#    Universal PSD lifting for MIXED symmetric-cone programs (Subagent
#    PSDLIFT).
#
#    Makes mixed LP+SOC / SOC+PSD / LP+PSD conic programs first-class
#    executable by lifting every native cone block to a single
#    block-diagonal PSD cone ("universal PSD lifting") and solving the
#    resulting pure-SDP program through the existing, working
#    `ingest` + `solve(algorithm=:sdp)` SDP path.
#
#    The lifting is exact (a bijection onto the structured subcone):
#
#      * :nonnegative  s in R_+^k  <->  X = diag(s) in PSD(k).
#      * :soc          (t,u) in SOC_n  <->  X = [[t, u'], [u, t*I]] in PSD(n)
#        (Schur complement: t - u'u/t >= 0 <-> t >= ||u||).
#      * :psd          stays as its packed slack matrix.
#
#    Concatenating the blocks into ONE block-diagonal PSD cone turns the
#    canonical primal `A x + s = b, s in K` into a pure SDP program where
#    the slack `s` is replaced by the lifted PSD matrices and the affine
#    equalities enforce the block structure.
#
#    The family lowerer here mirrors `lower_sdp_native`
#    (src/ir/lower_sdp.jl) for the *mixed* native family: each cone block
#    (product or affine) contributes one PSD block in the geometric
#    SDPProblem form
#        min c'x  s.t.  sum_i x_i A_i[l] - C[l] in PSD,  B'x = b
#    and every original-coordinate certificate maps back through the lift.
#
#    Include order: after src/ir/lower_sdp.jl (uses `SDPProblem`, `ingest`,
#    `ActiveSparseCoefficientVector`, `SDPBlockOrigin`, `_sdp_packed_basis`,
#    `_sdp_affine_psd_coefficients`, `_sdp_unpack_packed`).
#=====================================================================#

using SparseArrays: SparseMatrixCSC, sparse, dropzeros!

# ---------------------------------------------------------------------------
# Typed per-block PSD-lift origin
# ---------------------------------------------------------------------------

"""
    SDPX.PSDBlockOrigin

Typed map for ONE core PSD block in a mixed lifted program. `origin` is
`:product` when the block enforces a frontend product-variable cone and
`:affine` when it comes from an affine row block. `block` is the source
product/row-block number, `core_block` its ordered core PSD-block number,
`shape` the lifted PSD matrix dimension, `cone` the native cone kind
(`:nonnegative`, `:soc` or `:psd`) and `dimension` the source cone
dimension (vector length, or matrix dimension for `:psd`).
"""
struct PSDBlockOrigin
    origin::Symbol        # :product | :affine
    cone::Symbol          # :nonnegative | :soc | :psd
    block::Int
    core_block::Int
    shape::Int
    dimension::Int
end

"""
    SDPX.MixedPSDLowering{T<:AbstractFloat}

Immutable, fully typed result of [`lower_mixed_psd_native`](@ref). The
`core` is an owned `SDPProblem{T}` (the block-diagonal PSD lift), ready for
the `ingest`-based `solve(..., algorithm=:sdp)` path. Reconstruction
vectors and ordered origin maps are fresh vectors; mutating a lowering
result cannot mutate the source NCP's maps or arrays.
"""
struct MixedPSDLowering{T<:AbstractFloat}
    core::SDPProblem{T}
    route::NativeConeRoute
    objective_sign::Int
    objective_constant::T
    primal_refs::Vector{VariableRef}
    psd_origins::Vector{PSDBlockOrigin}
    equality_origins::Vector{SDPEqualityOrigin}
    constraint_dual_reconstruction::Vector{ConstraintRef}
    variable_dual_slack_reconstruction::Vector{VariableRef}
end

# ---------------------------------------------------------------------------
# PSD-lift matrix builders (the universal lift, sparse basis form)
# ---------------------------------------------------------------------------

"""
    _lift_nonnegative_basis(::Type{T}, dimension, bits)

The `k` diagonal basis matrices `E_jj` for the `:nonnegative` lift
`diag(s) in PSD(k)`. The j-th basis carries a `1` on `(j,j)`; this is the
coefficient matrix for the j-th source coordinate.
"""
function _lift_nonnegative_basis(::Type{T}, dimension::Int, bits::Int) where {T<:AbstractFloat}
    one_value = owned_arithmetic_copy(T, 1; precision_bits=bits)
    return [
        SparseMatrixCSC{T,Int}(
            sparse([j], [j], [owned_arithmetic_copy(T, 1; precision_bits=bits)], dimension, dimension),
        )
        for j in 1:dimension
    ]
end

"""
    _lift_soc_basis(::Type{T}, n, bits)

The `n` coefficient matrices for the `:soc` lift
`X(t,u) = [[t, u'],[u, t*I]] in PSD(n)`:
- coordinate 1 (head `t`) -> `I_n` (t appears on every diagonal);
- coordinate j (tail `u_{j-1}`, `j >= 2`) -> `E_{1,j} + E_{j,1}`.
"""
function _lift_soc_basis(::Type{T}, n::Int, bits::Int) where {T<:AbstractFloat}
    head = SparseMatrixCSC{T,Int}(sparse(
        collect(1:n), collect(1:n),
        [owned_arithmetic_copy(T, 1; precision_bits=bits) for _ in 1:n],
        n, n,
    ))
    basis = Vector{SparseMatrixCSC{T,Int}}(undef, n)
    basis[1] = head
    for j in 2:n
        basis[j] = SparseMatrixCSC{T,Int}(sparse(
            [1, j], [j, 1],
            [owned_arithmetic_copy(T, 1; precision_bits=bits),
             owned_arithmetic_copy(T, 1; precision_bits=bits)],
            n, n,
        ))
    end
    return basis
end

"""
    _lift_source_vector(::Type{T}, cone, vector, shape, bits) -> Matrix{T}

The constant (or a source column) lifted into the PSD block: `diag` for
`:nonnegative`, the arrow matrix for `:soc`, and the unpacked symmetric
matrix for `:psd`. Used to build the affine constant matrix `C = lift(b)`.
"""
function _lift_source_vector(::Type{T}, cone::Symbol, vector, shape::Int, bits::Int) where {T<:AbstractFloat}
    matrix = alloc_zeros(T, shape, shape)
    if cone === :nonnegative
        @inbounds for j in 1:shape
            matrix[j, j] = owned_arithmetic_copy(T, vector[j]; precision_bits=bits)
        end
    elseif cone === :soc
        @inbounds for d in 1:shape
            matrix[d, d] = owned_arithmetic_copy(T, vector[1]; precision_bits=bits)
        end
        @inbounds for j in 2:shape
            value = owned_arithmetic_copy(T, vector[j]; precision_bits=bits)
            matrix[1, j] = value
            matrix[j, 1] = value
        end
    else # :psd
        @inbounds for position in 1:length(vector)
            row = psd_packed_row(position, shape)
            column = psd_packed_column(position, shape)
            value = owned_arithmetic_copy(T, vector[position]; precision_bits=bits)
            matrix[row, column] = value
            matrix[column, row] = value
        end
    end
    return matrix
end

"""
    _sdp_affine_lifted_coefficients(::Type{T}, cone, equality_matrix,
                                    row_block, variables, total_rows, bits)

Build the sparse affine coefficient map `A_i` for a NONNEGATIVE or SOC
affine row block in the lifted PSD program. Each source row of the block
contributes the lift of that row's coefficients to the PSD matrix; off
diagonal entries of the SOC arrow are mirrored symmetrically.
"""
function _sdp_affine_lifted_coefficients(
    ::Type{T},
    cone::Symbol,
    equality_matrix::SparseMatrixCSC{T,Int},
    row_block::RowBlock,
    variables::Int,
    total_rows::Int,
    bits::Int,
) where {T<:AbstractFloat}
    shape = row_block.shape
    source_rows = row_block.rows
    packed = row_block.length
    length(source_rows) == packed || throw(SDPLoweringError(
        :row_map_violation,
        "SDPLoweringError: affine lifted row source-map length mismatch",
    ))
    row_major = SparseArrays.sparse(transpose(equality_matrix))
    row_indices = rowvals(row_major)
    row_values = nonzeros(row_major)

    # For :nonnegative the matrix is diagonal; for :soc the head fills the
    # diagonal and each tail row feeds the (1,j)/(j,1) slots. Accumulate
    # per-variable dense buffers then sparsify.
    row_buffers = [Int[] for _ in 1:variables]
    column_buffers = [Int[] for _ in 1:variables]
    value_buffers = [T[] for _ in 1:variables]

    local_rhs = nothing
    @inbounds for position in 1:packed
        source_row = source_rows[position]
        1 <= source_row <= total_rows || throw(SDPLoweringError(
            :row_map_violation,
            "SDPLoweringError: affine lifted source row $source_row out of range 1:$total_rows",
        ))
        row = cone === :soc && position >= 2 ? 1 : position
        column = cone === :soc && position >= 2 ? position : position
        if cone === :nonnegative
            row = position
            column = position
        elseif cone === :soc
            row = position == 1 ? 1 : 1
            column = position == 1 ? 1 : position
        end
        # row,column are the single entry this source coordinate populates
        # (SOC head -> diagonal (1,1); but head also fills the other diagonals).
        # For SOC the head (position 1) writes to ALL diagonals.
        for stored in nzrange(row_major, source_row)
            variable = row_indices[stored]
            coefficient = row_values[stored]
            iszero(coefficient) && continue
            owned = owned_arithmetic_copy(T, coefficient; precision_bits=bits)
            if cone === :nonnegative
                push!(row_buffers[variable], position)
                push!(column_buffers[variable], position)
                push!(value_buffers[variable], owned)
            elseif cone === :soc
                if position == 1
                    for d in 1:shape
                        push!(row_buffers[variable], d)
                        push!(column_buffers[variable], d)
                        push!(value_buffers[variable], owned_arithmetic_copy(T, owned; precision_bits=bits))
                    end
                else
                    push!(row_buffers[variable], 1)
                    push!(column_buffers[variable], position)
                    push!(value_buffers[variable], owned_arithmetic_copy(T, owned; precision_bits=bits))
                    push!(row_buffers[variable], position)
                    push!(column_buffers[variable], 1)
                    push!(value_buffers[variable], owned_arithmetic_copy(T, owned; precision_bits=bits))
                end
            end
        end
    end

    active = findall(buffer -> !isempty(buffer), row_buffers)
    matrices = SparseMatrixCSC{T,Int}[]
    sizehint!(matrices, length(active))
    @inbounds for variable in active
        matrix = SparseMatrixCSC{T,Int}(sparse(
            row_buffers[variable], column_buffers[variable], value_buffers[variable],
            shape, shape,
        ))
        dropzeros!(matrix)
        push!(matrices, matrix)
    end
    return ActiveSparseCoefficientVector(T, variables, active, matrices, shape)
end

"""
    _lift_product_origin_coefficients(::Type{T}, cone, dimension, block,
                                      variables, bits)

Coefficient matrices for a PRODUCT cone block in the PSD lift. The block's
source coordinates are the block's own variable slots; the coefficient for
source coordinate `j` is the j-th lift basis.
"""
function _lift_product_origin_coefficients(
    ::Type{T}, cone::Symbol, dimension::Int, block::NativeBlock,
    variables::Int, bits::Int,
) where {T<:AbstractFloat}
    if cone === :nonnegative
        basis = _lift_nonnegative_basis(T, dimension, bits)
    elseif cone === :soc
        basis = _lift_soc_basis(T, dimension, bits)
    else
        basis = [
            _sdp_packed_basis(T, position, dimension, bits)
            for position in 1:block.length
        ]
    end
    ids = collect(block.offset:(block.offset + length(basis) - 1))
    return ActiveSparseCoefficientVector(T, variables, ids, basis, dimension)
end

# ---------------------------------------------------------------------------
# PSD-lift lowering (NativeConeProgram -> SDPProblem)
# ---------------------------------------------------------------------------

"""
    lower_mixed_psd_native(program::NativeConeProgram{T}; sparse=:auto,
                           verbosity=1) -> MixedPSDLowering{T}

Lower a program classified as exactly `:mixed_family` to a single
block-diagonal PSD SDPProblem via the universal PSD lifting, ready for the
existing `algorithm=:sdp` solve path. Pure LP/SOC/SDP programs must use
their dedicated lowerers; this lowerer accepts only the mixed family
(LP+SOC, SOC+PSD, LP+PSD) and rejects exponential/power cones (no
symmetric lift).
"""
function _lower_mixed_psd_native_impl(
    program::NativeConeProgram{T};
    sparse::Union{Bool,Symbol}=:auto,
    verbosity::Int=1,
) where {T<:AbstractFloat}
    route = classify_native_cone_program(program)
    route.route === :mixed_family || throw(SDPLoweringError(
        :non_mixed_route,
        "SDPLoweringError: lower_mixed_psd_native requires a :mixed_family program, got route $(route.route)",
    ))
    variables = program_num_variables(program)
    variables > 0 || throw(SDPLoweringError(
        :no_variables,
        "SDPLoweringError: mixed PSD lifting requires at least one scalar variable",
    ))
    program.arithmetic isa ArithmeticSpec{T} || throw(SDPLoweringError(
        :arithmetic_mismatch,
        "SDPLoweringError: program arithmetic is not ArithmeticSpec{$T}",
    ))
    bits = program.precision_bits

    A_blocks = SparseCoefficientVector{T}[]
    C_blocks = Matrix{T}[]
    psd_origins = PSDBlockOrigin[]
    core_psd_block = 0

    # ---- 1. product variable blocks (nonnegative / soc / psd) ----
    for (block_number, block) in enumerate(program.blocks)
        cone = _domain_cone(block.domain)
        if cone === :free || cone === :zero
            continue            # free: omitted; zero: equality column below
        elseif cone === :nonnegative || cone === :soc || cone === :psd
            core_psd_block += 1
            push!(A_blocks, _lift_product_origin_coefficients(
                T, cone, block.shape, block, variables, bits,
            ))
            push!(C_blocks, alloc_zeros(T, block.shape, block.shape))
            push!(psd_origins, PSDBlockOrigin(
                :product, cone, block_number, core_psd_block, block.shape, block.shape,
            ))
        else
            throw(SDPLoweringError(
                :unexpected_mixed_product_cone,
                "SDPLoweringError: mixed PSD lifting supports nonnegative/soc/psd product cones, got $(repr(cone))",
            ))
        end
    end

    equality_matrix = program.equality_matrix
    rhs = program.rhs
    total_rows = program_num_rows(program)
    row_major = SparseArrays.sparse(transpose(equality_matrix))
    row_indices = rowvals(row_major)
    row_values = nonzeros(row_major)

    # ---- 2. affine row blocks (nonnegative / soc / psd) ----
    for (row_number, row_block) in enumerate(program.row_blocks)
        cone = _domain_cone(row_block.domain)
        if cone === :free || cone === :zero
            continue            # zero handled as equality columns below
        elseif cone === :nonnegative || cone === :soc
            shape = row_block.shape
            core_psd_block += 1
            source_rows = row_block.rows
            length(source_rows) == row_block.length || throw(SDPLoweringError(
                :row_map_violation,
                "SDPLoweringError: affine cone row source-map length mismatch",
            ))
            push!(A_blocks, _sdp_affine_lifted_coefficients(
                T, cone, equality_matrix, row_block, variables, total_rows, bits,
            ))
            rhs_vector = Vector{T}(undef, row_block.length)
            @inbounds for position in 1:row_block.length
                source_row = source_rows[position]
                rhs_vector[position] = owned_arithmetic_copy(T, rhs[source_row]; precision_bits=bits)
            end
            push!(C_blocks, _lift_source_vector(T, cone, rhs_vector, shape, bits))
            push!(psd_origins, PSDBlockOrigin(
                :affine, cone, row_number, core_psd_block, shape, row_block.length,
            ))
        elseif cone === :psd
            core_psd_block += 1
            push!(A_blocks, _sdp_affine_psd_coefficients(
                T, equality_matrix, row_block, variables, total_rows, bits,
            ))
            local_rhs = Vector{T}(undef, row_block.length)
            @inbounds for position in 1:row_block.length
                source_row = row_block.rows[position]
                local_rhs[position] = owned_arithmetic_copy(T, rhs[source_row]; precision_bits=bits)
            end
            push!(C_blocks, _sdp_unpack_packed(T, local_rhs, row_block.shape, bits))
            push!(psd_origins, PSDBlockOrigin(
                :affine, :psd, row_number, core_psd_block, row_block.shape, row_block.shape,
            ))
        else
            throw(SDPLoweringError(
                :unexpected_mixed_affine_cone,
                "SDPLoweringError: mixed PSD lifting supports nonnegative/soc/psd affine rows, got $(repr(cone))",
            ))
        end
    end

    # ---- 3. zero blocks / zero rows -> equality columns (as in lower_sdp) ----
    B_rows = Int[]
    B_columns = Int[]
    B_values = T[]
    beq = T[]
    equality_origins = SDPEqualityOrigin[]
    core_equality_column = 0
    for (block_number, block) in enumerate(program.blocks)
        block.domain isa ZeroCone || continue
        for position in 1:block.shape
            core_equality_column += 1
            push!(B_rows, block.offset + position - 1)
            push!(B_columns, core_equality_column)
            push!(B_values, owned_arithmetic_copy(T, 1; precision_bits=bits))
            push!(beq, owned_arithmetic_copy(T, 0; precision_bits=bits))
            push!(equality_origins, SDPEqualityOrigin(:product_zero, block_number, position, 0, core_equality_column))
        end
    end
    for (row_number, row_block) in enumerate(program.row_blocks)
        row_block.domain isa ZeroCone || continue
        for position in 1:row_block.length
            source_row = row_block.rows[position]
            core_equality_column += 1
            for stored in nzrange(row_major, source_row)
                variable = row_indices[stored]
                coefficient = row_values[stored]
                iszero(coefficient) && continue
                push!(B_rows, variable)
                push!(B_columns, core_equality_column)
                push!(B_values, owned_arithmetic_copy(T, coefficient; precision_bits=bits))
            end
            push!(beq, owned_arithmetic_copy(T, rhs[source_row]; precision_bits=bits))
            push!(equality_origins, SDPEqualityOrigin(:affine_zero, row_number, position, source_row, core_equality_column))
        end
    end
    B = SparseArrays.sparse(B_rows, B_columns, B_values, variables, core_equality_column)

    objective_sign = program.objective_sense isa Maximize ? -1 : 1
    c_core = Vector{T}(undef, variables)
    @inbounds for index in 1:variables
        value = objective_sign == -1 ? -program.objective_vector[index] : program.objective_vector[index]
        c_core[index] = owned_arithmetic_copy(T, value; precision_bits=bits)
    end
    objective_constant = owned_arithmetic_copy(T, program.objective_constant; precision_bits=bits)

    isempty(A_blocks) && throw(SDPLoweringError(
        :no_psd_blocks,
        "SDPLoweringError: mixed PSD lifting produced no PSD block",
    ))
    core = ingest(
        c_core, A_blocks, C_blocks, B, beq;
        T=T, sparse=sparse, validate=true, symmetrize=false, verbosity=verbosity,
    )
    return MixedPSDLowering{T}(
        core, route, objective_sign, objective_constant,
        copy(program.primal_reconstruction),
        psd_origins, equality_origins,
        copy(program.constraint_dual_reconstruction),
        copy(program.variable_dual_slack_reconstruction),
    )
end

"""
    lower_mixed_psd_native(program; sparse=:auto, verbosity=1)

Precision-owning public wrapper around the pure lifting implementation.
"""
function lower_mixed_psd_native(
    program::NativeConeProgram{T};
    sparse::Union{Bool,Symbol}=:auto,
    verbosity::Int=1,
) where {T<:AbstractFloat}
    if T === BigFloat
        bits = program.precision_bits
        return setprecision(BigFloat, bits) do
            _lower_mixed_psd_native_impl(program; sparse=sparse, verbosity=verbosity)
        end
    end
    return _lower_mixed_psd_native_impl(program; sparse=sparse, verbosity=verbosity)
end

# ---------------------------------------------------------------------------
# Canonical-level PSD lift (the universal PSD lifting, IR form)
# ---------------------------------------------------------------------------

# Classify the packed positions of one PSD block into source coordinates,
# redundant-zero entries and SOC tail entries. Returns three vectors indexed
# by packed position: `source` (0 when the position is not a source), plus
# for non-source positions whether they must equal zero (`redundant_zero`) or
# equal the SOC head (`redundant_head`).
function _psd_lift_positions(cone::Symbol, dimension::Int)
    packed = psd_packed_length(dimension)
    source = zeros(Int, packed)
    redundant_zero = falses(packed)
    redundant_head = falses(packed)
    for p in 1:packed
        row = psd_packed_row(p, dimension)
        col = psd_packed_column(p, dimension)
        if cone === :nonnegative
            if row == col
                source[p] = row
            else
                redundant_zero[p] = true
            end
        elseif cone === :soc
            if row == 1 && col == 1
                source[p] = 1
            elseif min(row, col) == 1 && max(row, col) >= 2
                source[p] = max(row, col)
            elseif row == col  # tail diagonal must equal the head t
                redundant_head[p] = true
            else
                redundant_zero[p] = true
            end
        else # :psd — every packed position is a source coordinate, in order
            source[p] = p
        end
    end
    return source, redundant_zero, redundant_head
end

"""
    lift_to_psd(canonical::CanonicalConicProgram{T}) -> CanonicalConicProgram{T}

The **universal PSD lifting** of a mixed symmetric-cone canonical program

    min c'x   s.t.   A x + s = b,   s in K = K_1 x ... x K_q

into a pure PSD program. Each cone block is lifted to one block-diagonal PSD
matrix (nonnegative -> `diag(s)`, SOC -> `[[t, u'],[u, t*I]]`, PSD -> itself),
the original slack `s` becomes free variables, and the affine equalities
`A x + s = b` (carried by `:zero` slack rows) plus the per-block structure
equalities (the redundant matrix entries equal their constrained values)
enforce the block structure exactly. The lift is a bijection onto the
structured subcone of the block-diagonal PSD cone.
"""
function lift_to_psd(canonical::CanonicalConicProgram{T}) where {T<:AbstractFloat}
    n = canonical_num_variables(canonical)
    m = canonical_num_slack(canonical)
    bits = canonical.precision_bits
    A = canonical.A
    b = canonical.b
    blocks = canonical.cone_layout.blocks

    # Variables: [x (n); s (m)] — original free x plus the original slack s.
    variables = n + m
    c_new = Vector{T}(undef, variables)
    @inbounds for i in 1:n
        c_new[i] = owned_arithmetic_copy(T, canonical.c[i]; precision_bits=bits)
    end
    @inbounds for i in 1:m
        c_new[n + i] = owned_arithmetic_copy(T, 0; precision_bits=bits)
    end

    A_rows = Int[]
    A_cols = Int[]
    A_vals = T[]
    b_vals = T[]
    descriptors = ConeBlockDescriptor{T}[]
    rowcount = 0

    # ---- PSD structure blocks: slack_p = L(source) or a fixed value ----
    for block in blocks
        cone = block.cone
        cone in (:nonnegative, :soc, :psd) || throw(ArgumentError(
            "PSD lift supports :nonnegative/:soc/:psd canonical blocks, got $(repr(cone))",
        ))
        nd = block.dimension
        source, redundant_zero, redundant_head = _psd_lift_positions(cone, nd)
        push!(descriptors, ConeBlockDescriptor(
            T, :psd, nd; offset=rowcount + 1,
            reconstruction=CanonicalBlockMap{T}(:lift, 0, 0, 1),
        ))
        packed = length(source)
        for p in 1:packed
            newrow = rowcount + 1
            if source[p] != 0
                # slack_p = s_{block.offset + source[p] - 1}
                svar = n + (block.offset + source[p] - 1)
                push!(A_rows, newrow); push!(A_cols, svar)
                push!(A_vals, _owned_arithmetic_eval(T, () -> -one(T); precision_bits=bits))
                push!(b_vals, owned_arithmetic_copy(T, 0; precision_bits=bits))
            elseif redundant_head[p]
                # slack_p = t (head = block.offset source)
                svar = n + block.offset
                push!(A_rows, newrow); push!(A_cols, svar)
                push!(A_vals, _owned_arithmetic_eval(T, () -> -one(T); precision_bits=bits))
                push!(b_vals, owned_arithmetic_copy(T, 0; precision_bits=bits))
            else
                # redundant_zero[p]: slack_p = 0
                push!(b_vals, owned_arithmetic_copy(T, 0; precision_bits=bits))
            end
            rowcount += 1
        end
    end

    # ---- original equality rows: A x + s = b, carried by a zero slack ----
    for i in 1:m
        push!(descriptors, ConeBlockDescriptor(
            T, :zero, 1; offset=rowcount + 1,
            reconstruction=CanonicalBlockMap{T}(:lift, 0, 0, 1),
        ))
        newrow = rowcount + 1
        for stored in nzrange(A, i)   # A is m x n; iterate row i's columns
            col = A.rowval[stored]
            val = A.nzval[stored]
            iszero(val) && continue
            push!(A_rows, newrow); push!(A_cols, col)
            push!(A_vals, owned_arithmetic_copy(T, val; precision_bits=bits))
        end
        push!(A_rows, newrow); push!(A_cols, n + i)
        push!(A_vals, owned_arithmetic_copy(T, 1; precision_bits=bits))
        push!(b_vals, owned_arithmetic_copy(T, b[i]; precision_bits=bits))
        rowcount += 1
    end

    A_new = SparseArrays.sparse(A_rows, A_cols, A_vals, rowcount, variables)
    layout = canonical_layout(descriptors)
    return CanonicalConicProgram(
        canonical.arithmetic, bits, c_new, A_new, b_vals, layout,
        canonical.reconstruction_chain,
    )
end

# ---------------------------------------------------------------------------
# Reconstruction maps: solved PSD matrices -> original cone coordinates
# ---------------------------------------------------------------------------

"""
    _lift_primal_slack(matrix, cone, dimension) -> Vector{T}

Extract the original-cone slack vector `s` from a solved primal PSD matrix:
- `:nonnegative` -> `s_i = X[i,i]`;
- `:soc` -> `t = X[1,1]`, `u_i = X[i,1]`;
- `:psd` -> packed-lower entries.
"""
function _lift_primal_slack(::Type{T}, matrix, cone::Symbol, dimension::Int) where {T<:AbstractFloat}
    if cone === :nonnegative
        return [matrix[i, i] for i in 1:dimension]
    elseif cone === :soc
        out = Vector{T}(undef, dimension)
        out[1] = matrix[1, 1]
        @inbounds for i in 2:dimension
            out[i] = matrix[i, 1]
        end
        return out
    else
        packed = variable_length(PSDCone(), dimension)
        out = Vector{T}(undef, packed)
        @inbounds for position in 1:packed
            out[position] = matrix[psd_packed_row(position, dimension), psd_packed_column(position, dimension)]
        end
        return out
    end
end

"""
    _lift_dual_slack(::Type{T}, matrix, cone, dimension) -> Vector{T}

Extract the original-cone dual vector from a lifted PSD dual matrix `Y`:
- `:nonnegative` -> `z_i = Y[i,i]`;
- `:soc` -> `lambda = tr(Y)`, `mu_i = 2*Y[1,i]`;
- `:psd` -> packed-lower entries (matching `pack_psd_dual`).
"""
function _lift_dual_slack(::Type{T}, matrix, cone::Symbol, dimension::Int) where {T<:AbstractFloat}
    if cone === :nonnegative
        return [matrix[i, i] for i in 1:dimension]
    elseif cone === :soc
        out = Vector{T}(undef, dimension)
        trace = zero(T)
        @inbounds for i in 1:dimension
            trace = _owned_arithmetic_eval(T, () -> trace + matrix[i, i]; precision_bits=precision(T))
        end
        out[1] = trace
        @inbounds for i in 2:dimension
            out[i] = _owned_arithmetic_eval(
                T, () -> T(2) * matrix[1, i]; precision_bits=precision(T),
            )
        end
        return out
    else
        return pack_psd_dual(matrix; precision_bits=precision(T))
    end
end
