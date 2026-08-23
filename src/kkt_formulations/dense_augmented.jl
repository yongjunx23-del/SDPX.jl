#=
    Explicit dense augmented KKT formulation.

    SDPX's reduced Newton equations are

        S*dx - B*dy = r
        B'*dx        = p.

    Negating the equality row gives the symmetric system

        [ S   -B ] [dx] = [ r]
        [-B'   0 ] [dy]   [-p].

    Only the lower triangle is assembled.  The provider owns pivoted LDLT
    factor/solve; SDPX owns assembly, the RHS sign, residuals, correction
    policy, acceptance, and all failure semantics.
=#

@inline function _augmented_workspace(ws::Workspace{T}) where {T}
    workspace = ws.augmented
    workspace === nothing && error(
        "dense augmented KKT backend has no allocated workspace",
    )
    return workspace::DenseAugmentedKKTWorkspace{T}
end
@inline function _augmented_store!(
    destination::AbstractMatrix{T},
    row::Int,
    column::Int,
    value::T,
) where {T}
    destination[row, column] = value
    return destination
end

@inline function _augmented_store!(
    destination::AbstractMatrix{BigFloat},
    row::Int,
    column::Int,
    value::BigFloat,
)
    MA.operate_to!(destination[row, column], copy, value)
    return destination
end

@inline function _augmented_store_negative!(
    destination::AbstractArray{T},
    index,
    value::T,
) where {T}
    destination[index] = -value
    return destination
end

@inline function _augmented_store_negative!(
    destination::AbstractArray{BigFloat},
    index,
    value::BigFloat,
)
    MA.operate_to!(destination[index], -, value)
    return destination
end

"""Assemble the authoritative lower triangle of `[S -B; -B' 0]`."""
function assemble_dense_augmented_kkt!(
    workspace::DenseAugmentedKKTWorkspace{T},
    S::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T}
    m = size(S, 1)
    n = size(B, 2)
    size(S, 2) == m || throw(DimensionMismatch("Schur matrix must be square"))
    size(B, 1) == m || throw(DimensionMismatch(
        "equality matrix row count must match the Schur matrix",
    ))
    size(workspace.matrix) == (m + n, m + n) || throw(DimensionMismatch(
        "augmented workspace dimensions do not match the Newton system",
    ))

    # Every authoritative nonzero block is overwritten below.  Only the
    # equality/equality lower block must be reset to structural zero; the
    # inactive upper triangle is deliberately left untouched.
    n > 0 && zero_owned!(view(
        workspace.matrix,
        (m + 1):(m + n),
        (m + 1):(m + n),
    ))
    @inbounds for column in 1:m
        for row in column:m
            _augmented_store!(workspace.matrix, row, column, S[row, column])
        end
    end
    # Bottom-left is authoritative: K[m+j,i] = -B[i,j].
    @inbounds for equality in 1:n
        row = m + equality
        for variable in 1:m
            _augmented_store_negative!(
                workspace.matrix,
                CartesianIndex(row, variable),
                B[variable, equality],
            )
        end
    end
    return workspace.matrix
end

"""Factor one explicit augmented Newton matrix; failure never changes route."""
function factor_dense_augmented_kkt!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    ::SolverOptions{T},
) where {T}
    workspace = _augmented_workspace(ws)
    started = time_ns()
    assemble_dense_augmented_kkt!(workspace, ws.S, prob.B)
    assembly_seconds = _elapsed_seconds(started)

    # Map the existing Schur regularization exactly onto the augmented primal
    # block.  The unfactored `matrix` remains the original Newton system for
    # residual/refinement; only the provider input receives the deterministic
    # diagonal shift. Provider perturbation or a retry through Normal
    # Equations is never authorized.
    # A provider factor may borrow `factor_buffer`; release the previous
    # handle before this iteration overwrites that storage.
    workspace.factor = nothing
    workspace.regularization = zero(T)
    workspace.factor_diagnostics = nothing
    workspace.inertia = nothing
    workspace.pivot_blocks = nothing
    workspace.permutation = nothing
    workspace.factor_precision = nothing
    workspace.rank_deficient = false
    factorization_seconds = 0.0
    reg_attempts = 0
    reg = zero(T)
    m = prob.dims.m
    n = prob.dims.n
    factor = nothing
    last_inertia = nothing
    rejection = :factor_failed
    while true
        _copy_lower_triangle!(workspace.factor_buffer, workspace.matrix)
        if reg_attempts > 0
            @inbounds for index in 1:m
                workspace.factor_buffer[index, index] +=
                    reg * max(abs(ws.S[index, index]), one(T))
            end
        end
        started = time_ns()
        factor = la_ldlt_factor!(ws.la_backend, workspace.factor_buffer)
        factorization_seconds += _elapsed_seconds(started)
        if factor !== nothing
            inertia = la_ldlt_inertia(factor)
            last_inertia = inertia
            inertia_class = _ldlt_inertia_class(inertia, m, n)
            if inertia_class === :accepted
                workspace.rank_deficient = false
                workspace.factor = factor
                workspace.inertia = inertia
                workspace.pivot_blocks = la_ldlt_blocks(factor)
                workspace.permutation = la_ldlt_permutation(factor)
                workspace.factor_precision = la_factor_precision(factor)
                workspace.factor_diagnostics = la_factor_diagnostics(factor)
                break
            end
            workspace.factor = nothing
            workspace.rank_deficient = inertia_class === :rank_deficient
            rejection = inertia_class
            if reg_attempts == 6
                # Snapshot the terminal candidate while its borrowed factor
                # storage is still valid.  No provider accessor may be called
                # after the next overwrite of `factor_buffer`.
                workspace.inertia = inertia_class === :invalid ? nothing : inertia
                if inertia_class !== :invalid
                    workspace.pivot_blocks = la_ldlt_blocks(factor)
                    workspace.permutation = la_ldlt_permutation(factor)
                    workspace.factor_precision = la_factor_precision(factor)
                end
                workspace.factor_diagnostics = la_factor_diagnostics(factor)
            end
            factor = nothing
        end
        reg_attempts == 6 && break
        reg_attempts += 1
        reg = reg_attempts == 1 ? sqrt(eps(T)) : reg * T(10)
    end
    workspace.regularization = reg
    if factor === nothing
        workspace.factor = nothing
        if workspace.inertia === nothing &&
           last_inertia !== nothing &&
           _ldlt_inertia_class(last_inertia, m, n) !== :invalid
            # A terminal provider failure has no live factor metadata. Keep
            # only the last lightweight immutable inertia evidence; blocks,
            # permutation, precision, and diagnostics remain unavailable.
            workspace.inertia = last_inertia
        end
        ws.la_fallback_reason = if rejection === :rank_deficient
            :la_equality_rank_deficient
        elseif rejection === :mismatch
            :la_equality_inertia_mismatch
        elseif rejection === :invalid
            :la_provider_inertia_invalid
        else
            :la_factor_failed
        end
        return (
            ok=false,
            reg_attempts=reg_attempts,
            q_pivoted=false,
            q_rank_deficient=rejection === :rank_deficient,
            equality_solver=:augmented_ldlt,
            phase_times=(
                schur_copy=assembly_seconds,
                schur_factorization=factorization_seconds,
                constraint_triangular_solve=0.0,
                equality_gram=0.0,
                equality_factorization=0.0,
            ),
        )
    end
    return (
        ok=true,
        reg_attempts=reg_attempts,
        # `q_pivoted` refers to the normal-equation equality Gram.  The full
        # augmented factor is pivoted, but no Q matrix exists on this route.
        q_pivoted=false,
        q_rank_deficient=false,
        equality_solver=:augmented_ldlt,
        phase_times=(
            schur_copy=assembly_seconds,
            schur_factorization=factorization_seconds,
            constraint_triangular_solve=0.0,
            equality_gram=0.0,
            equality_factorization=0.0,
        ),
    )
end

function _assemble_dense_augmented_rhs!(
    workspace::DenseAugmentedKKTWorkspace{T},
    r::AbstractVector{T},
    p::AbstractVector{T},
) where {T}
    m = length(r)
    length(workspace.rhs) == m + length(p) || throw(DimensionMismatch(
        "augmented RHS dimensions do not match the Newton system",
    ))
    copy_owned!(view(workspace.rhs, 1:m), r)
    @inbounds for equality in eachindex(p)
        _augmented_store_negative!(
            workspace.rhs,
            m + equality,
            p[equality],
        )
    end
    return workspace.rhs
end

"""Solve and recover `[dx;dy]` from the provider-owned LDLT handle."""
function solve_dense_augmented_kkt!(
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    workspace = _augmented_workspace(ws)
    factor = workspace.factor
    factor === nothing && error(
        "dense augmented KKT solve requested without a successful LDLT factor",
    )
    length(p) == n || throw(DimensionMismatch(
        "equality RHS length does not match the augmented solve request",
    ))
    _assemble_dense_augmented_rhs!(workspace, r, p)
    copy_owned!(workspace.solution, workspace.rhs)
    la_ldlt_factor_solve!(factor, workspace.solution)
    m = length(r)
    copy_owned!(dx_out, view(workspace.solution, 1:m))
    n > 0 && copy_owned!(dy_out, view(workspace.solution, (m + 1):(m + n)))
    return dx_out, dy_out
end

"""
Apply exactly one provider correction solve to the explicit augmented system.

SDPX has already formed the structured KKT residual and remains responsible
for deciding whether the correction is needed, whether it improves the
residual, and whether another pass is allowed.  On this formulation the
provider's correction primitive is mathematically identical to one solve with
the retained full augmented LDLT factor, so it can reuse provider-owned solve
scratch without replacing SDPX's refinement policy.
"""
function correct_dense_augmented_kkt!(
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    workspace = _augmented_workspace(ws)
    factor = workspace.factor
    factor === nothing && error(
        "dense augmented KKT correction requested without a successful LDLT factor",
    )
    length(p) == n || throw(DimensionMismatch(
        "equality residual length does not match the augmented correction request",
    ))
    _assemble_dense_augmented_rhs!(workspace, r, p)
    la_refinement_correction!(factor, workspace.rhs, workspace.solution)
    m = length(r)
    copy_owned!(dx_out, view(workspace.solution, 1:m))
    n > 0 && copy_owned!(dy_out, view(workspace.solution, (m + 1):(m + n)))
    return dx_out, dy_out
end

"""
Evaluate the complete symmetric augmented residual.  This is intentionally
independent of the provider factor storage and uses the unfactored SDPX-owned
lower triangle through `Symmetric(..., :L)`.
"""
function reduced_augmented_kkt_residual!(
    workspace::DenseAugmentedKKTWorkspace{T},
    r::AbstractVector{T},
    p::AbstractVector{T},
    dx::AbstractVector{T},
    dy::AbstractVector{T},
) where {T}
    _assemble_dense_augmented_rhs!(workspace, r, p)
    m = length(dx)
    copy_owned!(view(workspace.solution, 1:m), dx)
    !isempty(dy) && copy_owned!(view(workspace.solution, (m + 1):(m + length(dy))), dy)
    copy_owned!(workspace.residual, workspace.rhs)
    kmul_owned!(
        workspace.residual,
        Symmetric(workspace.matrix, :L),
        workspace.solution,
        -one(T),
        one(T),
    )
    return knrmInf(workspace.residual)
end
