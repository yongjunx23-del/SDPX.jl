# Review of `e786a7c` matrix-free PCG experiment

Target reviewed: `codex/review-v05-dirty-methods` at `2cdf566`.

## Decision

Do not cherry-pick `e786a7c` or its prerequisite chain
`b962657 -> 69c55f0 -> 1964eb2` into the v0.5 solver.  Preserve the PCG
operator as a design candidate, but integrate it only through the existing
NativeSOC/`Model` route after the numerical and memory gates below pass.

## Keep as design inputs

- The Hessian-vector product
  `N' * A * blockdiag(H_block) * A' * N`.
- The separation between computing block curvature and assembling the dense
  reduced Hessian.  Matrix-free mode must compute `H_block` while skipping
  only the dense assembly.
- An explicit, opt-in direct-versus-PCG selector and a deterministic test that
  compares directions, objectives, residuals, and certificates.

## Do not change in v0.5

- Keep the `_EXECUTION_ROUTE_TOKEN` guard.  Removing it is unrelated to PCG
  and permits planner routes to be constructed outside the planner.
- Keep the single public `Model -> optimize! -> Result` interface.  The
  experiment's parallel `GeneralSOCPProblem` and result API is incompatible
  with the frozen v0.5 surface and uses different affine-cone semantics.
- Keep the current Q3/fixed-trace and `solve.jl` architecture.  The experiment's
  prerequisite purge is broad, independently risky, and unnecessary for PCG.
- Do not enable automatic PCG routing before a production-scale A/B benchmark.

## Blocking defects found

1. In committed `e786a7c`, matrix-free mode called `eval_dual` with
   `compute_hessian=false`; every block Hessian was therefore zero.  The
   method reduced to a damping-scaled gradient step rather than Newton-PCG.
2. It is not memory-scalable: `N_mat = Matrix(I, m, m)` is `O(m^2)`, and the
   free-variable path rebuilds dense `A' * N_mat` every Newton iteration.
3. The Lorentz objective and gradient disagree (approximately a factor of two
   in a finite-difference check), and Lorentz feasibility is evaluated as a
   Euclidean ball.
4. `:optimal` checks only equality residual.  It ignores gradient, gap, cone
   violation, and certificate validity; the reported dual objective can also
   be stale after the final accepted step.
5. The claimed Steihaug-Toint routine has no trust-region boundary handling,
   and the line-search fallback uses the square root of the Jacobi diagonal
   instead of the stated preconditioner.

## Evidence

- The original focused suite passed `18/18`, but did not detect the zero-block
  Hessian defect.
- After computing block Hessians without assembling the dense reduced Hessian,
  a deterministic multi-block case matched the direct objective to
  `3.1e-14` and reduced PCG-path iterations from `56` to `15`.
- The follow-up focused suite passed `25/25`; it remains only small-case
  backward evidence, not a production acceptance benchmark.

Reconsider integration only after the nullspace/projector is implicit, no
dense `m x m` or `A' * N` object is formed, Lorentz derivatives are corrected,
and a real CSDR case passes direct-versus-PCG certificate and memory gates.
