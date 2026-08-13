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

    zero_owned!(workspace.matrix)
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
    workspace.regularization = zero(T)
    workspace.factor_diagnostics = nothing
    workspace.inertia = nothing
    workspace.rank_deficient = false
    copy_owned!(workspace.factor_buffer, workspace.matrix)
    started = time_ns()
    factor = la_ldlt_factor!(ws.la_backend, workspace.factor_buffer)
    factorization_seconds = _elapsed_seconds(started)
    reg_attempts = 0
    reg = zero(T)
    m = prob.dims.m
    while factor === nothing && reg_attempts < 6
        reg_attempts += 1
        reg = reg_attempts == 1 ? sqrt(eps(T)) : reg * T(10)
        copy_owned!(workspace.factor_buffer, workspace.matrix)
        @inbounds for index in 1:m
            workspace.factor_buffer[index, index] +=
                reg * max(abs(ws.S[index, index]), one(T))
        end
        started = time_ns()
        factor = la_ldlt_factor!(ws.la_backend, workspace.factor_buffer)
        factorization_seconds += _elapsed_seconds(started)
    end
    workspace.regularization = reg
    if factor === nothing
        workspace.factor = nothing
        workspace.factor_diagnostics = nothing
        workspace.inertia = nothing
        ws.la_fallback_reason = :la_factor_failed
        return (
            ok=false,
            reg_attempts=reg_attempts,
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
    workspace.factor = factor
    workspace.factor_diagnostics = la_factor_diagnostics(factor)
    inertia = la_ldlt_inertia(factor)
    workspace.inertia = inertia
    # Equality presolve, not LDLT pivoting, owns rank reduction. A successful
    # factor with structural zero inertia would otherwise silently accept a
    # dependent equality basis when presolve is disabled.
    Int(inertia[3]) == 0 || begin
        workspace.factor = nothing
        workspace.rank_deficient = true
        ws.la_fallback_reason = :la_equality_rank_deficient
        return (
            ok=false,
            reg_attempts=reg_attempts,
            q_pivoted=false,
            q_rank_deficient=true,
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

# Compatibility/internal spelling retained for callers added during Round 3.
dense_augmented_kkt_residual!(args...) =
    reduced_augmented_kkt_residual!(args...)
