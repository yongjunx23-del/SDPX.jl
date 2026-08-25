#=
    Wave A-2: LinearSolve A/B spike.

Compares, for a small dense SPD matrix (50x50):

  * native SDPX dense Cholesky solve (LinearAlgebra.cholesky, which is exactly
    what SDPX's bundled legacy LA backend delegates to via `kchol!`), and
  * `SDPXLinearSolveExt.LinearSolveFactorCache` with an explicit
    `LinearSolve.LUFactorization()`.

Measures: first-factorization time, same-A RHS-only solve time, `@allocated`
for the solve, and correctness (relative residual norm). Prints a comparison
table and exits 0.
=#
using SDPX
using LinearAlgebra
using LinearSolve
using SciMLBase
using Random
using Printf: @sprintf

# Load the optional extension explicitly so the wrapper is available.
# Extensions are not packages, so access the loaded module via get_extension.
const SDPXLinearSolveExt = Base.get_extension(SDPX, :SDPXLinearSolveExt)
@assert SDPXLinearSolveExt !== nothing "SDPXLinearSolveExt failed to load"
# Import the wrapper bindings explicitly: `solve!`/`factorize!` are also
# exported by LinearSolve/SciMLBase, so an explicit import pins the binding.
import .SDPXLinearSolveExt:
    LinearSolveFactorCache, prepare!, factorize!, solve!, solve_multi!,
    invalidate!, factor_status, factor_matrix_epoch

Random.seed!(0xA2)

# --- problem setup ----------------------------------------------------------
n = 50
B = randn(n, n)
A = Symmetric(B * B' + n * I)   # dense SPD
Amat = Matrix(A)                 # plain dense matrix for the solvers
rhs1 = randn(n)
rhs2 = randn(n)                  # a different RHS for the same-A solve

# --- native SDPX dense Cholesky baseline ------------------------------------
function native_cholesky_ab(A, rhs1, rhs2)
    # first factorization
    t_fact = @elapsed F = cholesky(A)
    # same-A RHS-only solve (reuses F)
    t_solve = @elapsed x1 = F \ rhs1
    alloc_solve = @allocated x1 = F \ rhs1
    x2 = F \ rhs2
    res = norm(A * x1 - rhs1) / norm(rhs1)
    return (t_fact=t_fact, t_solve=t_solve, alloc_solve=alloc_solve, residual=res)
end

# --- LinearSolveFactorCache (explicit LU) -----------------------------------
function linearsolve_ab(A, rhs1, rhs2)
    cache = LinearSolveFactorCache(LUFactorization())
    # first factorization
    t_fact = @elapsed factorize!(cache, A)
    # same-A RHS-only solve (reuses factorization)
    dest1 = zeros(eltype(A), size(A, 1))
    t_solve = @elapsed solve!(cache, dest1, rhs1)
    alloc_solve = @allocated solve!(cache, dest1, rhs1)
    dest2 = zeros(eltype(A), size(A, 1))
    solve!(cache, dest2, rhs2)
    res = norm(A * dest1 - rhs1) / norm(rhs1)
    return (t_fact=t_fact, t_solve=t_solve, alloc_solve=alloc_solve,
            residual=res, cache=cache)
end

# --- run --------------------------------------------------------------------
native = native_cholesky_ab(Amat, rhs1, rhs2)
ls = linearsolve_ab(Amat, rhs1, rhs2)

# --- report -----------------------------------------------------------------
println("="^72)
println("Wave A-2 LinearSolve A/B spike  (n = $n, dense SPD)")
println("="^72)
println(rpad("metric", 22) * rpad("native Cholesky", 24) * rpad("LinearSolve LU", 24))
println("-"^72)
println(rpad("first factorization (s)", 22) *
        rpad(@sprintf("%.6f", native.t_fact), 24) *
        rpad(@sprintf("%.6f", ls.t_fact), 24))
println(rpad("same-A solve (s)", 22) *
        rpad(@sprintf("%.6f", native.t_solve), 24) *
        rpad(@sprintf("%.6f", ls.t_solve), 24))
println(rpad("solve allocations (bytes)", 22) *
        rpad(@sprintf("%12d", native.alloc_solve), 24) *
        rpad(@sprintf("%12d", ls.alloc_solve), 24))
println(rpad("relative residual", 22) *
        rpad(@sprintf("%.3e", native.residual), 24) *
        rpad(@sprintf("%.3e", ls.residual), 24))
println("-"^72)
println("factor_status(ls cache) = ", factor_status(ls.cache),
        "  factor_matrix_epoch = ", factor_matrix_epoch(ls.cache))
println("="^72)

# correctness gate: both solvers must be accurate
@assert native.residual < 1e-10 "native Cholesky residual too large: $(native.residual)"
@assert ls.residual < 1e-10 "LinearSolve LU residual too large: $(ls.residual)"
println("PASS: both solvers accurate (residual < 1e-10)")
exit(0)
