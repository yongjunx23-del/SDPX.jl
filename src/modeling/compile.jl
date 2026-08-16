#=====================================================================#
#    Model -> NativeConeProgram compiler (v0.5).
#
#    Contract: `validate_model(model)` then
#    `compile_product_cone_model(model)` turns ONE authoritative
#    `SDPX.Model{T}` into ONE private
#    `SDPX.NativeConeProgram{T}`. No numerical lowering occurs: there
#    is no solver route, no scalarization, no free/± split, and no
#    SOC→PSD lift. Each model product block is preserved as one native
#    product block (free / orthants / zero / SOC / RSOC / PSD
#    lower-packed); each affine constraint record is preserved as one
#    native affine row block (an affine PSD record stays one packed
#    PSD row block).
#
#    Row semantics: NCP rows are `A*x - rhs in domain`, where an
#    `AffineConstraintRecord` expression `E` with constant `c` means
#    `E + c in domain`; the compiler stores coefficient rows of `E`
#    and `rhs[r] = -c` so the native row is `A*x - rhs = E + c`.
#    Objective: the model sense, coefficient vector, and constant are
#    retained verbatim in the program at the model arithmetic. Only a
#    lowerer (e.g. the LP lowering consumer) flips `Maximize` to the
#    core minimization convention by negating the NCP coefficient
#    vector; the compiler itself never changes signs. Reconstruction
#    maps are typed per ref; packed SDP shapes are validated
#    lower-column-major (see `src/ir/storage.jl`).
#
#    NO orientation, primal/dual labels, dualization metadata,
#    provenance, provider choice, or KKT/plan information is created.
#
#    Include order: after src/modeling/domains.jl, refs.jl, types.jl
#    and src/ir/types.jl. Uses src/ir/storage.jl helpers and the
#    reconstruction helpers in src/ir/reconstruction.jl.
#=====================================================================#

using SparseArrays: sparse, spzeros, dropzeros!

"""
    validate_model(model::Model{T}) -> Model{T}

Validate every boundary of an authoritative `Model{T}` before
compilation:

- arithmetic and precision identity (`precision_bits` matches the
  model), and objective identity;
- variable registries / block records / names agree in count, order,
  offsets, shapes, packed lengths, and model identity;
- constraint registries / records agree in count, order, shapes,
  row lengths, refs, and model identity;
- affine PSD records are packed-lower-column-major of exactly the
  required shape; every expression row and coefficient is finite;
- objective expression (when present) is finite and every coefficient
  is owned by the model's variable range.

Returns `model` unchanged. Throws `ArgumentError` on the first
violation. Never mutates the model.
"""
function validate_model(model::Model{T}) where {T<:AbstractFloat}
    expected_identity = model_identity(model)
    arithmetic(model) === model.arithmetic ||
        throw(ArgumentError("model arithmetic accessor mismatch"))
    precision_bits(model) == model.arithmetic.precision_bits ||
        throw(ArgumentError("model precision metadata mismatch"))

    # --- variable blocks and registry ---
    next = 1
    for (block_index, record) in enumerate(model.variable_blocks)
        record.primal_start === nothing ||
            length(record.primal_start) == record.length ||
            throw(ArgumentError(
                "variable block $block_index primal_start length " *
                "$(length(record.primal_start)) != block length $(record.length)",
            ))
        record.dual_slack_start === nothing ||
            length(record.dual_slack_start) == record.length ||
            throw(ArgumentError(
                "variable block $block_index dual_slack_start length " *
                "$(length(record.dual_slack_start)) != block length $(record.length)",
            ))
        record.offset == next ||
            throw(ArgumentError(
                "variable block $block_index offset $(record.offset) != expected $next",
            ))
        record.length == variable_length(record.domain, record.shape) ||
            throw(ArgumentError(
                "variable block $block_index length $(record.length) != " *
                "variable_length($(record.domain), $(record.shape)) = " *
                "$(variable_length(record.domain, record.shape))",
            ))
        for start in (record.primal_start, record.dual_slack_start)
            start === nothing && continue
            all(isfinite, start) ||
                throw(ArgumentError(
                    "variable block $block_index start values contain NaN or Inf",
                ))
        end
        if record.domain isa PSDCone
            is_stored_psd_metadata(PSDStorageMetadata(record.shape), :psd) ||
                throw(ArgumentError(
                    "variable block $block_index PSD storage metadata is not " *
                    "lower-column-major packed",
                ))
        end
        get(model.block_names, record.name, 0) == block_index ||
            throw(ArgumentError(
                "variable block $block_index name map does not point to it",
            ))
        next += record.length
    end
    num_variables(model) == next - 1 ||
        throw(ArgumentError(
            "model reports $(num_variables(model)) variables but blocks total $next-1",
        ))
    num_variables(model) == length(model.variables) ||
        throw(ArgumentError("model variable registry length mismatch"))
    for (position, ref) in enumerate(model.variables)
        ref.model == expected_identity ||
            throw(ArgumentError(
                "variable ref $position belongs to a different model identity",
            ))
        ref.block == variable_block_of(model, position) ||
            throw(ArgumentError(
                "variable ref $position block $(ref.block) != compiled block " *
                "$(variable_block_of(model, position))",
            ))
        ref.index == index_in_block_of(model, position) ||
            throw(ArgumentError(
                "variable ref $position index $(ref.index) != compiled index " *
                "$(index_in_block_of(model, position))",
            ))
    end

    # --- constraint records and registry ---
    next_row = 1
    for (block_index, record) in enumerate(model.constraint_blocks)
        expected_rows = variable_length(record.domain, record.shape)
        length(record.expressions) == expected_rows ||
            throw(ArgumentError(
                "constraint block $block_index expression count " *
                "$(length(record.expressions)) != expected rows $expected_rows",
            ))
        length(record.refs) == expected_rows ||
            throw(ArgumentError(
                "constraint block $block_index ref count " *
                "$(length(record.refs)) != expected rows $expected_rows",
            ))
        record.dual_start === nothing ||
            length(record.dual_start) == expected_rows ||
            throw(ArgumentError(
                "constraint block $block_index dual_start length " *
                "$(length(record.dual_start)) != expected rows $expected_rows",
            ))
        record.dual_start === nothing || all(isfinite, record.dual_start) ||
            throw(ArgumentError(
                "constraint block $block_index dual_start contains NaN or Inf",
            ))
        if record.domain isa PSDCone
            is_stored_psd_metadata(PSDStorageMetadata(record.shape), :psd) ||
                throw(ArgumentError(
                    "constraint block $block_index PSD storage metadata is not " *
                    "lower-column-major packed",
                ))
        end
        for (row, expression) in enumerate(record.expressions)
            expression.model == expected_identity ||
                throw(ArgumentError(
                    "constraint block $block_index row $row belongs to a different model identity",
                ))
            expression.precision_bits == precision_bits(model) ||
                throw(ArgumentError(
                    "constraint block $block_index row $row precision " *
                    "$(expression.precision_bits) != model precision " *
                    "$(precision_bits(model))",
                ))
            isfinite(expression.constant) ||
                throw(ArgumentError(
                    "constraint block $block_index row $row constant is NaN or Inf",
                ))
            length(expression.indices) == length(expression.coefficients) ||
                throw(ArgumentError(
                    "constraint block $block_index row $row index/coefficient mismatch",
                ))
            issorted(expression.indices) && allunique(expression.indices) ||
                throw(ArgumentError(
                    "constraint block $block_index row $row affine indices must be sorted and unique",
                ))
            all(isfinite, expression.coefficients) ||
                throw(ArgumentError(
                    "constraint block $block_index row $row coefficients contain NaN or Inf",
                ))
            for index in expression.indices
                index >= 1 && index <= num_variables(model) ||
                    throw(ArgumentError(
                        "constraint block $block_index row $row references variable $index " *
                        "out of range 1:$(num_variables(model))",
                    ))
            end
            ref = record.refs[row]
            ref.model == expected_identity ||
                throw(ArgumentError(
                    "constraint block $block_index ref $row belongs to a different model identity",
                ))
            ref.block == block_index ||
                throw(ArgumentError(
                    "constraint block $block_index ref $row block $(ref.block) != $block_index",
                ))
            ref.index == row ||
                throw(ArgumentError(
                    "constraint block $block_index ref $row index $(ref.index) != $row",
                ))
        end
        get(model.constraint_names, record.name, 0) == block_index ||
            throw(ArgumentError(
                "constraint block $block_index name map does not point to it",
            ))
        next_row += expected_rows
    end
    num_constraints(model) == next_row - 1 ||
        throw(ArgumentError(
            "model reports $(num_constraints(model)) constraints but records total $next_row-1",
        ))
    num_constraints(model) == length(model.constraints) ||
        throw(ArgumentError("model constraint registry length mismatch"))

    # --- objective ---
    if model.objective !== nothing
        expression = model.objective.expression
        expression.model == expected_identity ||
            throw(ArgumentError("objective expression belongs to a different model identity"))
        expression.precision_bits == precision_bits(model) ||
            throw(ArgumentError("objective precision does not match model precision"))
        isfinite(expression.constant) ||
            throw(ArgumentError("objective constant is NaN or Inf"))
        length(expression.indices) == length(expression.coefficients) ||
            throw(ArgumentError("objective index/coefficient mismatch"))
        issorted(expression.indices) && allunique(expression.indices) ||
            throw(ArgumentError("objective affine indices must be sorted and unique"))
        all(isfinite, expression.coefficients) ||
            throw(ArgumentError("objective coefficients contain NaN or Inf"))
        for index in expression.indices
            index >= 1 && index <= num_variables(model) ||
                throw(ArgumentError(
                    "objective references variable $index out of range 1:$(num_variables(model))",
                ))
        end
    end
    return model
end

"""1-based product-block number owning global packed position in the model."""
function variable_block_of(model::Model{T}, position::Integer) where {T<:AbstractFloat}
    position >= 1 || throw(ArgumentError("variable position must be >= 1"))
    expected = position
    for (index, record) in enumerate(model.variable_blocks)
        block_length = record.length
        if expected <= block_length
            return index
        end
        expected -= block_length
    end
    throw(ArgumentError("variable position $position out of model range"))
end

"""1-based within-block index of global packed position in the model."""
function index_in_block_of(model::Model{T}, position::Integer) where {T<:AbstractFloat}
    remaining = position
    for record in model.variable_blocks
        if remaining <= record.length
            return remaining
        end
        remaining -= record.length
    end
    throw(ArgumentError("variable position $position out of model range"))
end

"""
    compile_product_cone_model(model::Model{T}) -> NativeConeProgram{T}

Compile the authoritative model to the private native product-cone
program, preserving:

- every variable block as ONE `NativeBlock` in model order (free /
  orthant / zero / SOC / RSOC / PSD lower-packed);
- every affine constraint record as ONE `RowBlock` in model order
  (`A*x - rhs in domain`, `rhs = -constant`);
- the sparse equality map as a `SparseMatrixCSC{T,Int}` with exactly
  the stored affine coefficients (zero coefficients are dropped);
- objective sense, coefficient vector, and constant verbatim (a
  lowerer, never this compiler, flips `Maximize` when it needs a core
  minimization vector);
- typed reconstruction maps: primal per variable, constraint dual per
  row, and variable dual-slack per variable.

All scalar copies are owned at `T` / model precision (BigFloat copies
use `BigFloat(value; precision=precision_bits)`, never ambient
precision); model-owned warm-start vectors stay Model-owned until a
lowerer maps them. Fails with `ArgumentError` on any dimension,
offset, identity, finiteness, or PSD packed-shape violation.
"""
function compile_product_cone_model(model::Model{T}) where {T<:AbstractFloat}
    validate_model(model)
    bits = precision_bits(model)
    identity = model_identity(model)

    # ---- native product blocks ----
    blocks = NativeBlock[]
    offset = 1
    for record in model.variable_blocks
        push!(
            blocks,
            NativeBlock(record.domain, record.shape, offset),
        )
        offset += block_length(blocks[end])
    end
    num_variable_slots = offset - 1

    # ---- native affine row blocks and scalar affine rows ----
    row_blocks = RowBlock[]
    rhs = Vector{T}()
    equality_rows = Int[]
    equality_columns = Int[]
    equality_values = Vector{T}()
    next_row = 1
    for record in model.constraint_blocks
        block_row_count = expected_rows(record)
        block_first_row = next_row
        block_rows = collect(block_first_row:(block_first_row + block_row_count - 1))
        push!(
            row_blocks,
            RowBlock(record.domain, block_first_row, record.shape, block_rows),
        )
        for (local_row, expression) in enumerate(record.expressions)
            global_row = block_first_row + local_row - 1
            push!(rhs, _owned_arithmetic_eval(
                T,
                () -> -expression.constant;
                precision_bits=bits,
            ))
            for (slot, index) in enumerate(expression.indices)
                coefficient = expression.coefficients[slot]
                iszero(coefficient) && continue
                push!(equality_rows, global_row)
                push!(equality_columns, index)
                push!(
                    equality_values,
                    owned_arithmetic_copy(T, coefficient; precision_bits=bits),
                )
            end
        end
        next_row += block_row_count
    end
    if length(equality_values) > 0
        matrix =
            sparse(equality_rows, equality_columns, equality_values, next_row - 1, num_variable_slots)
        matrix = owned_sparse_copy(T, matrix; precision_bits=bits)
        dropzeros!(matrix)
    else
        matrix = spzeros(T, next_row - 1, num_variable_slots)
    end

    # ---- objective ----
    objective_vector = Vector{T}(undef, num_variable_slots)
    owned_zero = owned_arithmetic_copy(T, 0; precision_bits=bits)
    for index in eachindex(objective_vector)
        objective_vector[index] =
            owned_arithmetic_copy(T, owned_zero; precision_bits=bits)
    end
    objective_constant = owned_arithmetic_copy(T, owned_zero; precision_bits=bits)
    sense = Minimize()
    if model.objective !== nothing
        record = model.objective
        sense = record.sense
        objective_constant =
            owned_arithmetic_copy(T, record.expression.constant; precision_bits=bits)
        for (slot, index) in enumerate(record.expression.indices)
            coefficient = record.expression.coefficients[slot]
            iszero(coefficient) && continue
            copied = owned_arithmetic_copy(T, coefficient; precision_bits=bits)
            objective_vector[index] = _owned_arithmetic_eval(
                T,
                () -> objective_vector[index] + copied;
                precision_bits=bits,
            )
        end
    end

    return NativeConeProgram(
        arithmetic(model),
        sense,
        objective_vector,
        objective_constant,
        matrix,
        rhs,
        blocks,
        row_blocks,
        primal_reconstruction(model),
        constraint_dual_reconstruction(model),
        variable_dual_slack_reconstruction(model),
        UInt64(identity),
    )
end

"""Block row count of an affine record at its frozen shape."""
expected_rows(record::AffineConstraintRecord) =
    variable_length(record.domain, record.shape)
