#=====================================================================#
#    Pure LP-family NCP numerical lowering (v0.5).
#
#    Contract: a route-classified `NativeConeProgram` whose route is
#    exactly `:lp_family` is lowered, without any split/lift/
#    scalarization, to the existing dense or sparse active-row
#    `linear_program` path. The core program remains one block per
#    native product/row block; nothing here creates additional
#    `NativeBlock`s or synthesizes cone rows.
#
#    Mathematical semantics (all rows in the NCP are `A*x - rhs in
#    domain`):
#      * Reals product block: no inequality row; dual slack is zero.
#      * Nonnegative product block: one inequality `+I*x >= 0`.
#      * Nonpositive product block: one inequality `-I*x >= 0`, with a
#        -1 sign on the reconstructed variable-dual slack.
#      * Zero product block: one equality row `+I*x == 0`. Under LP
#        stationarity `c - A'·y - I·q = 0`, the equality multiplier is
#        exactly the original variable dual slack `q`, so the core
#        equality dual maps back as that variable dual slack with
#        sign +1 (`mapped=false` marks the variable-slack kind).
#      * Row constraints: Nonnegative -> `G=A, h=rhs`; Nonpositive ->
#        `G=-A, h=-rhs` plus a -1 sign on that reconstructed core
#        inequality dual; Zero -> equality `Aeq=A, beq=rhs`; Reals is
#        vacuous and its reconstructed dual is zero.
#    Objective: Minimize is unchanged; Maximize negates the core
#    objective and records `objective_sign = -1` with the original
#    constant for reconstruction. No orientation labels, no dual-model
#    strings, no provider/precision decisions, no README/docs edits.
#
#    Fail-closed rules:
#      * non-LP routes fail immediately through
#        `classify_native_cone_program` (before any lowering work);
#      * a program whose core LP would need a dummy cone row (no
#        inequality row at all: all-free/equality-only) fails with the
#        typed `LPLoweringError` instead of synthesizing a cone.
#=====================================================================#

# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

"""
    SDPX.LPLoweringError <: Exception

Typed failure raised by [`lower_lp_native`](@ref) when a
route-classified `:lp_family` program cannot be represented by the
existing `linear_program` boundary without inventing a dummy cone.
The structured fields describe the blocker.
"""
struct LPLoweringError <: Exception
    reason::Symbol
    message::String
end

Base.showerror(io::IO, err::LPLoweringError) = print(io, err.message)

"""
    SDPX.CoreLPRecordOrigin

Small typed descriptor of where one core LP inequality/equality row
map came from in the original NCP.

- `kind = :inequality` — a core inequality row reconstructed from an
  original affine row block (`block`/`index` = source `RowBlock`
  number / within-block position).
- `kind = :variable_dual_slack` — a coordinate reconstructed from an
  original product `NativeBlock` (`block`/`index` = source block
  number / within-block position), either a built-in nonnegative/
  nonpositive variable dual slack (core inequality) or a Zero
  product-block variable dual slack (core equality).
"""
struct CoreLPRecordOrigin
    kind::Symbol
    block::Int
    index::Int
    sign::Int
end

"""
    SDPX.CoreLPInequalityDual

Reconstruction record for a core LP inequality coordinate:
`mapped` is `true` when the coordinate is a real inequality dual,
`false` when it is a built-in nonnegative-variable dual slack. `sign`
is the multiplier to apply to the core dual when recovering the
original (dual) coordinate.
"""
struct CoreLPInequalityDual
    mapped::Bool
    sign::Int
end

"""
    SDPX.CoreLPEqualityDual

Reconstruction record for a core LP equality coordinate: `mapped` is
`true` for a real core equality multiplier and `false` for a Zero
product-block variable-dual-slack coordinate. `sign` is the
reconstruction multiplier (`+1` for Zero product blocks).
"""
struct CoreLPEqualityDual
    mapped::Bool
    sign::Int
end

# ---------------------------------------------------------------------------
# Typed immutable lowering result
# ---------------------------------------------------------------------------

"""
    SDPX.LPLowering{T<:AbstractFloat}

Immutable, fully typed result of [`lower_lp_native`](@ref): the core
`SDPProblem{T}`, the route record, objective sign/constant metadata,
the original primal refs in program order, and ordered typed maps for
core inequality/equality dual coordinates back to a constraint dual or
variable dual slack (including sign).

`inequality_dual_origins[i]` describes, in core inequality-row order:
- `kind = :inequality`, `block`/`index` = source `RowBlock` /
  within-block position, `mapped = true`, `sign = ±1`;
- `kind = :variable_dual_slack`, `block`/`index` = source
  `NativeBlock` / within-block position, `mapped = false` (built-in
  dual slack of an original nonnegative/nonpositive variable),
  `sign = ±1` (`-1` for Nonpositive).

`equality_dual_origins[j]` describes, in core equality-row order:
- `kind = :equality`, `block`/`index` = source `RowBlock` /
  within-block position, `mapped = true`, `sign = 1`;
- `kind = :variable_dual_slack`, `block`/`index` = source
  `NativeBlock` / within-block position, `mapped = false` (Zero
  product-block variable dual slack), `sign = +1`.

No `Any` fields and no orientation/dual-model strings.
"""
struct LPLowering{T<:AbstractFloat}
    core::SDPProblem{T}
    route::NativeConeRoute
    objective_sign::Int
    objective_constant::T
    primal_refs::Vector{VariableRef}
    inequality_dual_origins::Vector{CoreLPRecordOrigin}
    equality_dual_origins::Vector{CoreLPRecordOrigin}
    inequality_duals::Vector{CoreLPInequalityDual}
    equality_duals::Vector{CoreLPEqualityDual}
end

"""
    lower_lp_native(program::NativeConeProgram{T};
                    sparse::Union{Bool,Symbol}=:auto,
                    verbosity::Int=1) -> LPLowering{T}

Lower a `classify_native_cone_program(program).route === :lp_family`
program with the semantics above. Throws [`LPLoweringError`](@ref)
when the pure-LP core has no inequality coordinate that can populate
the existing active-row LP vector.
"""
function lower_lp_native(
    program::NativeConeProgram{T};
    sparse::Union{Bool,Symbol}=:auto,
    verbosity::Int=1,
) where {T<:AbstractFloat}
    route = classify_native_cone_program(program)
    route.route === :lp_family ||
        throw(LPLoweringError(
            :non_lp_route,
            "LPLoweringError: lower_lp_native requires an :lp_family program, got route $(route.route)",
        ))

    variables = program_num_variables(program)
    variables > 0 || throw(LPLoweringError(
        :no_variables,
        "LPLoweringError: pure LP lowering requires at least one variable",
    ))
    program.arithmetic isa ArithmeticSpec{T} ||
        throw(LPLoweringError(
            :arithmetic_mismatch,
            "LPLoweringError: program arithmetic is not ArithmeticSpec{$T}",
        ))
    bits = program.precision_bits
    # Every scalar entering the LP core is copied at the program-owned
    # precision.  In particular, `_ingest_owned_scalar(BigFloat, x)` keeps a
    # mutable copy at `precision(x)`, which can differ from the NCP's declared
    # precision; using this helper prevents ambient/source precision from
    # leaking across the lowering boundary.
    function owned_scalar(value)
        converted = owned_arithmetic_copy(T, value; precision_bits=bits)
        isfinite(converted) || throw(LPLoweringError(
            :nonfinite_data,
            "LPLoweringError: non-finite scalar encountered while lowering",
        ))
        return converted
    end
    owned_zero() = owned_scalar(0)
    owned_one() = owned_scalar(1)
    owned_neg(value) = _owned_arithmetic_eval(
        T,
        () -> -value;
        precision_bits=bits,
    )

    # --- ordered row/block accumulation (no dummy rows) --------------------
    G_rows = Int[]
    G_columns = Int[]
    G_values = T[]
    h_values = T[]
    Aeq_rows = Int[]
    Aeq_columns = Int[]
    Aeq_values = T[]
    beq_values = T[]

    inequality_origins = CoreLPRecordOrigin[]
    equality_origins = CoreLPRecordOrigin[]
    inequality_duals = CoreLPInequalityDual[]
    equality_duals = CoreLPEqualityDual[]

    # Product blocks first, in program order.
    for (block_number, block) in enumerate(program.blocks)
        domain = block.domain
        shape = block.shape
        offset = block.offset
        if domain isa Reals
            continue
        elseif domain isa Nonnegative
            for position in 1:shape
                col = offset + position - 1
                push!(G_rows, length(h_values) + 1)
                push!(G_columns, col)
                push!(G_values, owned_one())
                push!(h_values, owned_zero())
                push!(inequality_origins, CoreLPRecordOrigin(
                    :variable_dual_slack,
                    block_number,
                    position,
                    1,
                ))
                push!(inequality_duals, CoreLPInequalityDual(false, 1))
            end
        elseif domain isa Nonpositive
            for position in 1:shape
                col = offset + position - 1
                push!(G_rows, length(h_values) + 1)
                push!(G_columns, col)
                push!(G_values, owned_neg(owned_one()))
                push!(h_values, owned_zero())
                push!(inequality_origins, CoreLPRecordOrigin(
                    :variable_dual_slack,
                    block_number,
                    position,
                    -1,
                ))
                push!(inequality_duals, CoreLPInequalityDual(false, -1))
            end
        elseif domain isa ZeroCone
            for position in 1:shape
                col = offset + position - 1
                push!(Aeq_rows, length(beq_values) + 1)
                push!(Aeq_columns, col)
                push!(Aeq_values, owned_one())
                push!(beq_values, owned_zero())
                push!(equality_origins, CoreLPRecordOrigin(
                    :variable_dual_slack,
                    block_number,
                    position,
                    1,
                ))
                push!(equality_duals, CoreLPEqualityDual(false, 1))
            end
        else
            throw(LPLoweringError(
                :unexpected_lp_block_domain,
                "LPLoweringError: unexpected :lp_family product domain $(domain)",
            ))
        end
    end

    # Affine row blocks, in program order. Iterate every stored nonzero of the
    # equality matrix through a transposed CSC view (row-contiguous), never
    # stopping after the first coefficient.
    row_blocks = program.row_blocks
    equality_matrix = program.equality_matrix
    rhs = program.rhs
    n_rows = program_num_rows(program)
    row_major = SparseArrays.sparse(transpose(equality_matrix))
    row_indices = rowvals(row_major)
    row_values = nonzeros(row_major)
    for (row_index, row_block) in enumerate(row_blocks)
        domain = row_block.domain
        shape = row_block.shape
        source_rows = row_block.rows
        if domain isa Reals
            continue
        elseif domain isa Nonnegative
            for position in 1:shape
                source_row = source_rows[position]
                1 <= source_row <= n_rows || throw(LPLoweringError(
                    :row_map_violation,
                    "LPLoweringError: row block $(row_index) source row out of range",
                ))
                for position_in_row in nzrange(row_major, source_row)
                    variable = row_indices[position_in_row]
                    coefficient = row_values[position_in_row]
                    iszero(coefficient) && continue
                    push!(G_rows, length(h_values) + 1)
                    push!(G_columns, variable)
                    push!(G_values, owned_scalar(coefficient))
                end
                push!(h_values, owned_scalar(rhs[source_row]))
                push!(inequality_origins, CoreLPRecordOrigin(
                    :inequality,
                    row_index,
                    position,
                    1,
                ))
                push!(inequality_duals, CoreLPInequalityDual(true, 1))
            end
        elseif domain isa Nonpositive
            for position in 1:shape
                source_row = source_rows[position]
                1 <= source_row <= n_rows || throw(LPLoweringError(
                    :row_map_violation,
                    "LPLoweringError: row block $(row_index) source row out of range",
                ))
                for position_in_row in nzrange(row_major, source_row)
                    variable = row_indices[position_in_row]
                    coefficient = row_values[position_in_row]
                    iszero(coefficient) && continue
                    push!(G_rows, length(h_values) + 1)
                    push!(G_columns, variable)
                    push!(G_values, owned_neg(coefficient))
                end
                push!(h_values, owned_neg(rhs[source_row]))
                push!(inequality_origins, CoreLPRecordOrigin(
                    :inequality,
                    row_index,
                    position,
                    -1,
                ))
                push!(inequality_duals, CoreLPInequalityDual(true, -1))
            end
        elseif domain isa ZeroCone
            for position in 1:shape
                source_row = source_rows[position]
                1 <= source_row <= n_rows || throw(LPLoweringError(
                    :row_map_violation,
                    "LPLoweringError: row block $(row_index) source row out of range",
                ))
                for position_in_row in nzrange(row_major, source_row)
                    variable = row_indices[position_in_row]
                    coefficient = row_values[position_in_row]
                    iszero(coefficient) && continue
                    push!(Aeq_rows, length(beq_values) + 1)
                    push!(Aeq_columns, variable)
                    push!(Aeq_values, owned_scalar(coefficient))
                end
                push!(beq_values, owned_scalar(rhs[source_row]))
                push!(equality_origins, CoreLPRecordOrigin(
                    :equality,
                    row_index,
                    position,
                    1,
                ))
                push!(equality_duals, CoreLPEqualityDual(true, 1))
            end
        else
            throw(LPLoweringError(
                :unexpected_lp_row_domain,
                "LPLoweringError: unexpected :lp_family row domain $(domain)",
            ))
        end
    end

    # The active-row LP vector needs at least one inequality coordinate.
    isempty(h_values) && throw(LPLoweringError(
        :no_dummy_cone,
        "LPLoweringError: pure-LP lowering requires at least one inequality row; " *
        "an all-free or equality-only NCP cannot be represented without a dummy cone",
    ))

    # Assemble sparse CSC with every stored entry. `linear_program` then
    # builds its active-row storage from these matrices.
    G = SparseArrays.sparse(G_rows, G_columns, G_values, length(h_values), variables)
    h = h_values
    Aeq = SparseArrays.sparse(Aeq_rows, Aeq_columns, Aeq_values, length(beq_values), variables)
    beq = beq_values

    objective_vector = program.objective_vector
    objective_constant = owned_scalar(program.objective_constant)
    if program.objective_sense isa Maximize
        objective_sign = -1
    else
        objective_sign = 1
    end
    # Owned, precision-safe copy of the (possibly negated) objective.
    c_core = Vector{T}(undef, variables)
    for index in eachindex(c_core, objective_vector)
        value = objective_sign == -1 ? owned_neg(objective_vector[index]) : objective_vector[index]
        c_core[index] = owned_scalar(value)
    end

    core = linear_program(
        c_core,
        G,
        h;
        Aeq=Aeq,
        beq=beq,
        T=T,
        sparse=sparse,
        validate=true,
        verbosity=verbosity,
    )
    return LPLowering{T}(
        core,
        route,
        objective_sign,
        objective_constant,
        copy(program.primal_reconstruction),
        inequality_origins,
        equality_origins,
        inequality_duals,
        equality_duals,
    )
end
