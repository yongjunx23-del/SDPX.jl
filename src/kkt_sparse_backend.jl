#=====================================================================
    Sparse KKT backend with symbolic reuse (plan §15.1, §15.2)

    An interior-point method factorizes a matrix whose *values* change
    every iteration but whose *sparsity pattern* does not. Recomputing
    the elimination ordering and symbolic factorization each time is
    pure waste, and §15.2 asks that the ordering, symbolic
    factorization, permutation, and numeric buffers all be reused,
    reanalysing only when the pattern actually changes.

    CHOLMOD already separates the two phases: `cholesky(A)` analyses and
    factorizes, `cholesky!(F, A)` refactorizes into an existing `F`,
    reusing its analysis. This backend makes that separation explicit,
    detects pattern changes so reuse is never silently wrong, and counts
    both phases so the reuse can be verified rather than assumed.

    CHOLMOD is `Float64`-only. Extended-precision types keep the dense
    path; `supports_sparse_backend` is the guard.
=====================================================================#

"""
    SparseCholeskyBackend

Sparse Cholesky with a cached symbolic factorization.

Mutable because the whole point is state carried across iterations: the
factorization object, the pattern it was analysed for, and counters proving the
analysis is being reused.
"""
mutable struct SparseCholeskyBackend <: KKTBackend
    factorization::Any          # CHOLMOD factor, or nothing before the first analyse
    pattern::UInt64             # fingerprint of the structure analysed
    dimension::Int
    analyses::Int               # how many times the symbolic phase ran
    factorizations::Int         # how many numeric factorizations ran
    failures::Int
end

SparseCholeskyBackend() = SparseCholeskyBackend(nothing, UInt64(0), 0, 0, 0, 0)

backend_name(::SparseCholeskyBackend) = :sparse_cholesky

"""
    supports_sparse_backend(T) -> Bool

CHOLMOD is compiled for `Float64` only, so extended-precision solves keep the
dense path. Reporting this explicitly is better than discovering it through a
method error deep in a solve.
"""
supports_sparse_backend(::Type{Float64}) = true
supports_sparse_backend(::Type) = false

"""
    pattern_fingerprint(A) -> UInt64

Cheap identifier for a sparsity pattern: the structure arrays only, never the
values. Two matrices with the same fingerprint may share a symbolic
factorization.

Hashing `colptr` and `rowval` is O(nnz) but touches no floating point, so it is
negligible against a numeric factorization and — unlike comparing the matrices
themselves — cannot be fooled by values that happen to coincide.
"""
pattern_fingerprint(A::SparseMatrixCSC) =
    hash(A.colptr, hash(A.rowval, hash(size(A))))

"""
    analyze!(backend, A) -> Bool

Compute and cache the symbolic factorization for `A`'s pattern. Returns whether
the analysis succeeded.

Called by `factorize!` when the pattern is new; a caller does not normally
invoke it directly.

The fill-reducing ordering is left to CHOLMOD rather than chosen here (plan
§15.3). Measured on the augmented systems this backend actually receives from
the LP path, the library default beats the natural ordering on every case, in
both fill and time:

```text
  vars  nnz(K)   default nnz(L)   natural nnz(L)   default   natural
   800    7960           114138           216964   0.0020 s  0.0039 s
  1200   11966           233600           475255   0.0037 s  0.0102 s
  1000   24702           320095           441718   0.0051 s  0.0050 s
```

A selector layered on top would have to beat that to earn its cost, and
nothing measured here suggests it would.
"""
function analyze!(backend::SparseCholeskyBackend, A::SparseMatrixCSC{Float64})
    factor = try
        cholesky(Symmetric(A, :L); check=false)
    catch exception
        _recoverable(exception) || rethrow()
        nothing
    end
    if factor === nothing || !issuccess(factor)
        backend.failures += 1
        backend.factorization = nothing
        return false
    end
    backend.factorization = factor
    backend.pattern = pattern_fingerprint(A)
    backend.dimension = size(A, 1)
    backend.analyses += 1
    backend.factorizations += 1        # `cholesky` also produced numeric values
    return true
end

"""
    factorize!(backend, A) -> Bool

Numeric factorization of `A`, reusing the cached symbolic analysis whenever the
pattern is unchanged.

The pattern check is what makes the reuse safe. Refactorizing into a factor that
was analysed for a *different* structure is the classic way symbolic reuse
produces silent corruption, so the fingerprint is compared every time rather
than assumed stable.
"""
function factorize!(backend::SparseCholeskyBackend, A::SparseMatrixCSC{Float64})
    reusable = backend.factorization !== nothing &&
               backend.dimension == size(A, 1) &&
               backend.pattern == pattern_fingerprint(A)
    reusable || return analyze!(backend, A)
    try
        cholesky!(backend.factorization, Symmetric(A, :L); check=false)
    catch exception
        _recoverable(exception) || rethrow()
        # A pattern-compatible refactorization can still fail numerically (loss
        # of definiteness). Fall back to a fresh analysis rather than leaving a
        # stale factor in place.
        return analyze!(backend, A)
    end
    if !issuccess(backend.factorization)
        return analyze!(backend, A)
    end
    backend.factorizations += 1
    return true
end

"""
    _cholmod_solve!(x, factorization, rhs) -> x

Solve with a CHOLMOD factorization, portably across supported Julia versions.

`ldiv!` is deliberately not used. On Julia 1.10 -- this package's declared
minimum -- CHOLMOD provides neither `ldiv!(F, b)` nor `ldiv!(x, F, b)` for a
`Vector` right-hand side, and the generic three-argument fallback dispatches to
the missing two-argument method. Only `\` is available there. Julia 1.12 does
supply the three-argument form, which is why this went unnoticed: every local
run was on 1.12 and every sparse-backend test errored on 1.10.

The cost is one allocation of the solution vector per solve, against an
`O(nnz(L))` triangular pair, and correctness on the minimum supported version
is worth more than avoiding it.
"""
function _cholmod_solve!(x::AbstractVector{Float64}, factorization,
                         rhs::AbstractVector{Float64})
    copyto!(x, factorization \ rhs)
    return x
end

"""
    solve!(x, backend, rhs) -> x

Solve with the current factorization. Errors if no successful factorization
exists, rather than returning an unspecified vector.
"""
function solve!(x::AbstractVector{Float64}, backend::SparseCholeskyBackend,
                rhs::AbstractVector{Float64})
    backend.factorization === nothing &&
        error("sparse KKT backend has no valid factorization; call factorize! first")
    return _cholmod_solve!(x, backend.factorization, rhs)
end

"""
    statistics(backend) -> NamedTuple

Reuse accounting. `symbolic_reuse_ratio` is the fraction of numeric
factorizations that avoided a fresh analysis — the number that says whether
§15.2 is actually doing anything on a given model.
"""
function statistics(backend::SparseCholeskyBackend)
    reused = max(backend.factorizations - backend.analyses, 0)
    return (
        backend=backend_name(backend),
        analyses=backend.analyses,
        factorizations=backend.factorizations,
        reused=reused,
        symbolic_reuse_ratio=backend.factorizations == 0 ? 0.0 :
                             reused / backend.factorizations,
        failures=backend.failures,
    )
end

"""
    analyze(backend, prob) -> NamedTuple

Structural summary, reporting that this backend does carry a reusable symbolic
factorization (unlike the dense and arrow paths).
"""
function analyze(backend::SparseCholeskyBackend, prob::SDPProblem)
    L, m, n, k = prob.dims
    return (
        backend=backend_name(backend),
        variables=m,
        equalities=n,
        blocks=L,
        symbolic_reuse=true,
    )
end

#---------------------------------------------------------------------
#   Sparse LDL for the augmented (symmetric indefinite) KKT system
#---------------------------------------------------------------------

"""
    SparseLDLBackend

Sparse LDLᵀ with a cached symbolic factorization, for the augmented KKT system

```text
K = [ H + δI   -B  ]
    [   Bᵀ    -δI  ]
```

which is symmetric **indefinite** — the `-δI` block puts eigenvalues on both
sides of zero — so Cholesky does not apply and `SparseCholeskyBackend` cannot be
reused for it.

The augmented form matters for sparse LPs precisely because it avoids forming
`GᵀDG`: the normal-equations Hessian fills in and can be far denser than `G`,
whereas the augmented matrix keeps `G`'s own sparsity.
"""
mutable struct SparseLDLBackend <: KKTBackend
    factorization::Any
    pattern::UInt64
    dimension::Int
    analyses::Int
    factorizations::Int
    failures::Int
end

SparseLDLBackend() = SparseLDLBackend(nothing, UInt64(0), 0, 0, 0, 0)

backend_name(::SparseLDLBackend) = :sparse_ldl

"""
    analyze!(backend::SparseLDLBackend, K) -> Bool

Compute and cache the symbolic factorization for `K`'s pattern.
"""
function analyze!(backend::SparseLDLBackend, K::SparseMatrixCSC{Float64})
    factor = try
        ldlt(Symmetric(K, :L); check=false)
    catch exception
        _recoverable(exception) || rethrow()
        nothing
    end
    if factor === nothing || !issuccess(factor)
        backend.failures += 1
        backend.factorization = nothing
        return false
    end
    backend.factorization = factor
    backend.pattern = pattern_fingerprint(K)
    backend.dimension = size(K, 1)
    backend.analyses += 1
    backend.factorizations += 1
    return true
end

"""
    factorize!(backend::SparseLDLBackend, K) -> Bool

Numeric LDLᵀ of `K`, reusing the cached analysis when the pattern is unchanged.
The pattern is compared every call for the same reason as in the Cholesky
backend: refactorizing into an analysis computed for a different structure is a
silent-corruption bug, not a loud one.
"""
function factorize!(backend::SparseLDLBackend, K::SparseMatrixCSC{Float64})
    reusable = backend.factorization !== nothing &&
               backend.dimension == size(K, 1) &&
               backend.pattern == pattern_fingerprint(K)
    reusable || return analyze!(backend, K)
    try
        ldlt!(backend.factorization, Symmetric(K, :L); check=false)
    catch exception
        _recoverable(exception) || rethrow()
        return analyze!(backend, K)
    end
    issuccess(backend.factorization) || return analyze!(backend, K)
    backend.factorizations += 1
    return true
end

function solve!(x::AbstractVector{Float64}, backend::SparseLDLBackend,
                rhs::AbstractVector{Float64})
    backend.factorization === nothing &&
        error("sparse LDL backend has no valid factorization; call factorize! first")
    return _cholmod_solve!(x, backend.factorization, rhs)
end

function statistics(backend::SparseLDLBackend)
    reused = max(backend.factorizations - backend.analyses, 0)
    return (
        backend=backend_name(backend),
        analyses=backend.analyses,
        factorizations=backend.factorizations,
        reused=reused,
        symbolic_reuse_ratio=backend.factorizations == 0 ? 0.0 :
                             reused / backend.factorizations,
        failures=backend.failures,
    )
end

function analyze(backend::SparseLDLBackend, prob::SDPProblem)
    L, m, n, k = prob.dims
    return (
        backend=backend_name(backend),
        variables=m,
        equalities=n,
        blocks=L,
        symbolic_reuse=true,
    )
end

"""
    augmented_kkt(H, B, regularization) -> SparseMatrixCSC

Assemble the **symmetric** augmented KKT matrix that `SparseLDLBackend`
factorizes:

```text
K = [ H + δI    B   ]
    [   Bᵀ    -δI   ]
```

Note the sign convention differs from the dense `_lp_populate_kkt!`, which
writes `-B` above the diagonal and `+Bᵀ` below and is therefore **unsymmetric**
— which is why that path uses `lu!` rather than a symmetric factorization.

The two forms are equivalent up to the sign of the equality multiplier: solving
this system yields `-y` where the dense form yields `y`. Any caller that swaps
one for the other must negate the equality block of the solution. Getting this
wrong does not fail loudly — it silently returns a dual direction of the wrong
sign — so it is stated here rather than left to be rediscovered.
"""
function augmented_kkt(H::SparseMatrixCSC{Float64}, B::SparseMatrixCSC{Float64},
                       regularization::Float64)
    variables = size(H, 1)
    equalities = size(B, 2)
    equalities == 0 &&
        return sparse(Symmetric(H + regularization * I, :L))
    upper = hcat(H + regularization * I, B)
    lower = hcat(transpose(B), -regularization * sparse(I, equalities, equalities))
    # Build the full matrix and symmetrize explicitly rather than wrapping in
    # `Symmetric(_, :L)`, which would discard the upper blocks and mirror the
    # lower ones — the bug that made an earlier version emit `+B` where `-B`
    # was intended.
    return sparse(vcat(upper, lower))
end

#---------------------------------------------------------------------
#   Formulation selection (plan §12.6)
#---------------------------------------------------------------------

"""Measured crossover, in nonzeros per row of the augmented KKT matrix, below
which sparse factorization beats dense LU.

Calibrated on an IPM-like loop (10 factorize+solve, fixed pattern, 98% symbolic
reuse, Apple M4, one thread), sweeping density at n=1000, m=200, dim=1200:

| nnz/row | dense | sparse | sparse/dense |
|---|---|---|---|
| 2.2 | 0.127 s | 0.001 s | **133x faster** |
| 3.5 | 0.139 s | 0.001 s | 103x |
| 5.4 | 0.108 s | 0.008 s | 13.4x |
| 10.0 | 0.116 s | 0.058 s | 2.0x |
| 15.9 | 0.109 s | 0.130 s | 0.84x — dense wins |
| 32.7 | 0.108 s | 0.301 s | 0.36x — dense wins by 2.8x |

The gain grows with size at low density (21.7x at n=2000, 24.3x at n=4000, and
1477x at n=4000 with density 2e-4), so this is a threshold worth getting right:
choosing wrongly is expensive in *both* directions."""
const LP_SPARSE_NNZ_PER_ROW = 13.0

"""Below this dimension the dense path wins regardless of sparsity — BLAS is
efficient enough on small matrices that sparse bookkeeping dominates."""
const LP_SPARSE_MINIMUM_DIMENSION = 200

"""
    LPFormulation

Which linear system the LP path should form and factorize.

* `:dense_lu` — today's behaviour: form the dense augmented `K`, factorize with
  LU. Correct for every problem, and fastest once the KKT matrix is not sparse.
* `:sparse_ldl` — keep the augmented system sparse and factorize with LDLᵀ,
  reusing the symbolic analysis across iterations.
* `:sparse_normal` — sparse Cholesky of the normal equations. Only valid with no
  equality rows, where the system is positive definite.
"""
const LPFormulation = Symbol

"""
    select_lp_formulation(; dimension, nonzeros, equalities, arithmetic) -> Symbol

Choose the LP linear-system formulation from measured structure.

Returns `:dense_lu` unless the augmented matrix is sparse enough to pay for
sparse factorization, because that is the choice that is never wrong — only
slower. `nonzeros` is the nonzero count of the augmented KKT matrix, not of the
constraint matrix, since it is the fill-in of `GᵀDG` that decides this and not
the sparsity of `G` alone.

Non-`Float64` arithmetic always gets `:dense_lu`: CHOLMOD is `Float64`-only.
"""
function select_lp_formulation(; dimension::Integer, nonzeros::Integer,
                               equalities::Integer, arithmetic::Type=Float64)
    supports_sparse_backend(arithmetic) || return :dense_lu
    dimension >= LP_SPARSE_MINIMUM_DIMENSION || return :dense_lu
    dimension > 0 || return :dense_lu
    nonzeros / dimension < LP_SPARSE_NNZ_PER_ROW || return :dense_lu
    # With no equality rows the system is positive definite, so Cholesky applies
    # and is cheaper than LDL.
    return equalities == 0 ? :sparse_normal : :sparse_ldl
end

"""
    formulation_backend(formulation) -> KKTBackend

The backend implementing a chosen formulation. `:dense_lu` maps to the LP LU
backend that the solver already uses.
"""
function formulation_backend(formulation::Symbol)
    formulation === :sparse_normal && return SparseCholeskyBackend()
    formulation === :sparse_ldl && return SparseLDLBackend()
    formulation === :dense_lu && return LPLUBackend()
    throw(ArgumentError("unknown LP formulation $(formulation)"))
end

#---------------------------------------------------------------------
#   Iterative refinement in the original KKT equations (plan §12.3)
#---------------------------------------------------------------------

"""Refinement stops once a pass fails to cut the residual to at most this
fraction of the previous one — past that it only accumulates rounding noise."""
const SPARSE_REFINE_MIN_DECREASE = 0.5

"""
    refine_solution!(x, backend, K, rhs; max_steps, tolerance) -> (steps, residual)

Iterative refinement of `x` against the **original** system `K x = rhs`, not
against the regularized or factored form.

This matters for the augmented KKT specifically. That system is deliberately
regularized with a small `δ` on the equality diagonal, and it is ill-conditioned
by construction: measured directly, a 75x75 augmented system at `δ = 1e-8` gives
a direct-solve residual of 2.5e-7 — nine orders worse than the 1e-15 a
well-conditioned solve of the same size returns. Refining against the original
equations recovers that accuracy, and §12.3 asks for exactly this because the
formulation is chosen when *certificate quality* matters.

Refinement is stopped on stagnation rather than run to a fixed count: once a
pass no longer reduces the residual meaningfully, further passes add noise. The
previous iterate is restored if a pass makes things worse, so the result is
never worse than stopping earlier would have been.
"""
function refine_solution!(x::AbstractVector{Float64}, backend::KKTBackend,
                          K::SparseMatrixCSC{Float64}, rhs::AbstractVector{Float64};
                          max_steps::Int=4, tolerance::Float64=0.0)
    residual = similar(x)
    correction = similar(x)
    previous_best = similar(x)
    scale = max(maximum(abs, rhs), 1.0)
    target = tolerance > 0 ? tolerance : 64 * eps(Float64) * scale

    mul!(residual, K, x)
    residual .= rhs .- residual
    best = maximum(abs, residual)
    steps = 0

    for _ in 1:max_steps
        best <= target && break
        copyto!(previous_best, x)
        solve!(correction, backend, residual)
        x .+= correction
        mul!(residual, K, x)
        residual .= rhs .- residual
        current = maximum(abs, residual)
        if !isfinite(current) || current > best
            # This pass made it worse: undo it and stop.
            copyto!(x, previous_best)
            break
        end
        steps += 1
        improved = current <= best * SPARSE_REFINE_MIN_DECREASE
        best = current
        improved || break
    end
    return (steps, best)
end

"""
    inertia_available(backend) -> Bool

Whether the backend can report the inertia of the factored system.

`false` for the CHOLMOD-backed sparse factorizations: Julia's interface exposes
`F.D` only as a `FactorComponent`, which cannot be materialized (`sparse`,
`Matrix` and `diag` all raise `CanonicalIndexError`), so the sign counts needed
for an inertia check are unreachable. §12.3 asks for inertia monitoring "where
available"; here it is not, and this reports that honestly rather than
substituting a proxy that would not detect the same failures.

A wrong inertia in an augmented KKT means the regularization is insufficient and
the point is not a saddle point. Without inertia, the practical substitute is
`refine_solution!`: a system with wrong inertia will not refine to a small
residual.
"""
inertia_available(::KKTBackend) = false
