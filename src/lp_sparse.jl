#=====================================================================
    Sparse linear system for the dedicated LP path (plan §11.1, §12.6)

    The LP solver forms `H = GᵀDG` with `D = diag(z/s)` and factors the
    Newton system every iteration. Both steps are dense: `G` is read out
    of the problem into a dense `L × m` array, so a constraint matrix
    with a handful of entries per row still pays `O(m³)` per
    factorization.

    When `G` really is sparse that is the dominant cost and it is
    avoidable. The gate is measured rather than assumed — see
    `select_lp_formulation` — because a sparse factorization of a matrix
    that is not sparse enough is *slower* than the dense one, and the
    crossover is a property of the fill-in of `GᵀDG` rather than of `G`.
=====================================================================#

"""
    LPSparseSystem{T}

Sparse Newton system for the LP path, together with the symbolic
factorization that is reused across iterations.

Only the numeric values of `K` change from iteration to iteration — the pattern
is fixed by the patterns of `G` and `B` — so the fill-reducing ordering and
elimination tree are computed once. That reuse is the larger part of the win:
measured separately on this backend, refactorizing into an existing symbolic
factorization is 2.45x / 2.37x / 1.17x faster at `n = 500 / 2000 / 5000`.

`formulation` is `:sparse_normal` when there are no equality rows, in which case
the system is positive definite and the provider-neutral sparse Cholesky path
applies. Sparse equality KKT systems are unsupported and fail closed before
execution.
"""
mutable struct LPSparseSystem{T}
    G::SparseMatrixCSC{T,Int}
    B::SparseMatrixCSC{T,Int}
    K::SparseMatrixCSC{T,Int}
    backend::KKTBackend
    formulation::Symbol
    variables::Int
    equalities::Int
    analyzed::Bool
    # First-class sparse storage is optional for the historical Float64
    # augmented route.  Generic normal-equation systems populate these fields
    # once at setup and update only `matrix.nzval` in later iterations.
    storage::Any
    assembly_map::Any
end

# Compatibility constructor for the pre-Round-6 eight-field value layout.
function LPSparseSystem{T}(
    G::SparseMatrixCSC{T,Int},
    B::SparseMatrixCSC{T,Int},
    K::SparseMatrixCSC{T,Int},
    backend::KKTBackend,
    formulation::Symbol,
    variables::Int,
    equalities::Int,
    analyzed::Bool,
) where {T}
    return LPSparseSystem{T}(
        G, B, K, backend, formulation, variables, equalities, analyzed,
        nothing, nothing,
    )
end

"""
    lp_sparse_candidate(G, B, arithmetic) -> Union{Nothing,LPSparseSystem}

Build the sparse system if the measured gate says it will pay, and return
`nothing` otherwise so the caller keeps the dense path.

The gate is evaluated on the *assembled* augmented matrix rather than on `G`,
because `GᵀDG` can be far denser than `G` — a single dense column of `G` fills
`H` completely. Assembling once to find out costs one sparse triple product,
which is negligible against the per-iteration factorizations it decides.
"""
function lp_sparse_candidate(
    G::SparseMatrixCSC{T,Int},
    B::SparseMatrixCSC{T,Int},
    ::Type{T};
    storage::Union{Bool,Symbol}=:auto,
) where {T}
    variables = size(G, 2)
    equalities = size(B, 2)
    requested_storage = storage isa Bool ? (storage ? :sparse : :dense) : storage
    requested_storage in (:auto, :dense, :sparse) || throw(ArgumentError(
        "sparse storage must be :auto, :dense, or :sparse",
    ))
    requested_storage === :dense && return nothing

    # Sparse providers intentionally implement only SPD normal equations.
    # Explicit sparse requests fail closed rather than silently routing to a
    # dense or indefinite solver.
    generic = supports_sparse_generic(T)
    if generic && equalities > 0
        requested_storage === :sparse && throw(ArgumentError(
            "generic sparse provider currently supports only normal equations " *
            "(equalities=0); sparse augmented KKT is not implemented",
        ))
        return nothing
    end
    supports_sparse_execution(T) || return nothing

    # Probe the pattern with unit weights: the fill-in of `GᵀDG` does not
    # depend on the weight values, only on the pattern of `G`.
    probe = transpose(G) * G
    kkt = _lp_sparse_assemble(probe, B, one(T))
    formulation = select_lp_formulation(;
        dimension=size(kkt, 1),
        nonzeros=nnz(kkt),
        equalities=equalities,
        arithmetic=T,
        storage=requested_storage,
    )
    formulation === :dense_lu && return nothing

    if generic
        lower = sparse_lower_csc(probe)
        frozen = freeze_sparse_csc(lower; provider=GenericSparseProvider{T}())
        assembly_map = sparse_gram_assembly_map(G, frozen)
        backend = GenericSparseCholeskyBackend(T)
        return LPSparseSystem{T}(
            G,
            B,
            frozen.matrix,
            backend,
            :sparse_normal,
            variables,
            equalities,
            false,
            frozen,
            assembly_map,
        )
    end

    # Both explicit and authoritative auto sparse normal equations use one
    # frozen CSC pattern and refactor the provider's numeric values in place.
    if T === Float64 && equalities == 0 && formulation === :sparse_normal
        lower = sparse_lower_csc(probe)
        frozen = freeze_sparse_csc(
            lower;
            provider=CHOLMODSparseProvider(),
        )
        assembly_map = sparse_gram_assembly_map(G, frozen)
        backend = CHOLMODSparseCholeskyBackend()
        return LPSparseSystem{T}(
            G,
            B,
            frozen.matrix,
            backend,
            :sparse_normal,
            variables,
            equalities,
            false,
            frozen,
            assembly_map,
        )
    end

    return LPSparseSystem{T}(
        G,
        B,
        kkt,
        formulation_backend(formulation),
        formulation,
        variables,
        equalities,
        false,
        nothing,
        nothing,
    )
end

"""
    _lp_sparse_assemble(H, B, regularization) -> SparseMatrixCSC

The augmented matrix in the **symmetric** convention

```text
K = [ H + δI    B   ]
    [   Bᵀ    -δI   ]
```

which is factorized only for equality-free normal equations. With no equality
rows this degenerates to `H + δI`, which is positive definite; the explicit
sparse equality route is rejected before this assembly is reached.

This is deliberately *not* the *shape* of the dense `_lp_populate_kkt!`, which
is unsymmetric and therefore needs `lu!`. Solving this form yields `-y` where
the dense form yields `y`, and [`lp_sparse_solve!`](@ref) is the single place
that converts, so no caller has to know.

The two do now solve the **same regularized system**. They previously did not:
the dense equality block carried `-δ`, which differs from this one by `O(δ)` in
the multiplier — invisible at the default regularization and worth 0.22 in the
direction once `δ` reached `1e-2`, which the LP loop's escalation reaches. The
dense side was changed to match this form rather than the reverse, because this
one is quasi-definite and that is what makes LDL stable on it.
"""
function _lp_sparse_assemble(H::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                             regularization::T) where {T}
    variables = size(H, 1)
    equalities = size(B, 2)
    shifted = H + regularization * sparse(one(T) * LinearAlgebra.I, variables, variables)
    equalities == 0 && return shifted
    lower = -regularization * sparse(one(T) * LinearAlgebra.I, equalities, equalities)
    return [shifted B; transpose(B) lower]
end

"""
    lp_sparse_factor!(system, weights, regularization) -> Bool

Assemble `K` for the current barrier weights and factor it, reusing the symbolic
analysis. Returns whether the factorization succeeded, matching the `issuccess`
contract the dense path uses.
"""
function lp_sparse_factor!(system::LPSparseSystem{T}, weights::AbstractVector{T},
                           regularization::T) where {T}
    if system.storage isa SparseKKTStorage{T}
        storage = system.storage::SparseKKTStorage{T}
        assembly_map = system.assembly_map::SparseAssemblyMap{T}
        system.K = assemble_sparse_gram!(
            storage,
            assembly_map,
            weights;
            regularization=regularization,
        )
        ok = factorize!(system.backend, system.K)
        system.analyzed = ok
        return ok
    end
    weighted = transpose(system.G) * (Diagonal(weights) * system.G)
    system.K = _lp_sparse_assemble(weighted, system.B, regularization)
    # `factorize!` analyses on its own whenever the pattern is new, so calling
    # `analyze!` first here factored the same matrix twice. That cost a full
    # extra numeric factorization on the first iteration of every sparse LP
    # solve, and showed up as `factorizations = 2` after one call.
    ok = factorize!(system.backend, system.K)
    system.analyzed = ok
    return ok
end

"""
    lp_sparse_solve!(rhs, system) -> rhs

Solve in place, returning the direction in the **dense path's** sign convention
so the caller cannot tell the two apart.

The equality block is negated on the way out: the symmetric form yields `-y`
(see [`_lp_sparse_assemble`](@ref)). This is the only place that conversion
happens, and getting it wrong is silent rather than loud, which is why it is
covered directly by a regression test against the dense factorization.
"""
function lp_sparse_solve!(rhs::AbstractVector{T}, system::LPSparseSystem{T}) where {T}
    solution = if system.backend isa GenericSparseCholeskyBackend{T}
        factor = (system.backend::GenericSparseCholeskyBackend{T}).factor
        bits = factor === nothing ? 0 : (factor::GenericSparseCholeskyFactor{T}).precision_bits
        _sparse_factor_zeros(T, length(rhs), bits)
    else
        similar(rhs)
    end
    solve!(solution, system.backend, rhs)
    copyto!(rhs, solution)
    @inbounds for index in (system.variables + 1):(system.variables + system.equalities)
        rhs[index] = -rhs[index]
    end
    return rhs
end
