# Linear-algebra providers

The SDPX solver owns all problem semantics; linear algebra is a replaceable
arithmetic seam underneath it. The dependency direction is:

```text
SDPX planner / sparse symbolic structure / KKT assembly / certification
    -> unified LA API (plan, instantiate, la_* dispatchers)
       -> StandardLinearAlgebra (LinearAlgebra/BLAS/LAPACK)
       -> bundled SDPXLegacyLA provider (owned legacy k* kernels)
       -> optional MFLA provider (MultiFloatLinearAlgebra extension)
       -> optional BFLA provider (BigFloatLinearAlgebra extension)
          -> arithmetic libraries
```

## Planning and execution

`plan_la_backend` and `instantiate_la_backend` are the only authorized entry
points. Solver code requests `:auto`, `:standard`, `:bfla`, `:multifloat`, or
`:legacy`; the planner resolves once while the `ExecutionPlan` is built.
Numerical execution never retries another provider implicitly. Executed
diagnostics record the provider, ownership contract, planned fallback reason,
and runtime fallback reason; they never infer the backend from the presence of
a type or a global.

Provider availability may satisfy a selected formulation, but it never selects
the mathematical formulation itself. The formulation planner consumes
mathematical input facts and equality-basis evidence only. A capability is
advertised by the SDPX extension only when both the upstream capability and
the corresponding semantic adapter are present.

Unsupported provider/formulation pairs fail during planning. Runtime failure
policy is explicit: a provider failure is not permission to execute a
different provider unless that fallback is already part of the plan.

## Standard and legacy providers

Julia's `LinearAlgebra` (BLAS/LAPACK) covers ordinary Float64 dense work and
is the reference route. The bundled `SDPXLegacyLA` provider owns the
historical `k*` kernels behind the `la_*` dispatcher seam for generic and
BigFloat arithmetic. It advertises exactly
`SDPX_LEGACY_LA_CAPABILITIES`: `:dot`, `:norminf`, `:mul`, `:mul_owned`,
`:syrk`, `:chol`, `:cholesky_factor!`, `:solve`, `:cholsolve_owned`, `:trsm`,
`:trsv_lower`, `:trsv_transpose`, `:axpby`, and `:axpby_owned`.

The legacy provider's `ksyrk!` contract mirrors both triangles after the
kernel runs, even when an upstream arithmetic provider is lower-triangle-only.
That mirroring note is load-bearing: some SDPX call sites still read the upper
triangle, so a provider advertising `:syrk` must either mirror both triangles
itself or the provider layer must mirror the authoritative lower triangle
before returning.

Callers use `la_*` dispatchers, never `k*` directly. `kdot!`,
`kdot_columns!`, `alloc_zeros`, and `copy_owned!` are implementation helpers
or shared ownership infrastructure today, not provider capabilities.

## MultiFloatLinearAlgebra

When the complete provider is loaded, fixed `Float64x2`, `Float64x3`, and
`Float64x4` dense-Cholesky plans may resolve to the MFLA extension. The
`ExecutionPlan` remains the only selection authority; a failed MFLA operation
is never retried through generic `LinearAlgebra` or the bundled legacy
provider unless a structural fallback is already present in the plan.

The migrated ordinary dense path uses MFLA for dot, GEMV/GEMM, lower SYRK,
TRSV/TRSM, Cholesky factorization, vector/multiple-RHS solves, and equality
rank-revealing QR. The provider owns one reusable `MFWorkspace` for sequential
factorizations. MFLA factors snapshot their LU/LDLT/RRQR metadata, so
workspace reuse or growth does not invalidate live factors; only the
destructively factorized matrix remains borrowed. Concurrent factorization
must still use distinct solver/provider workspaces. Packed GEMM calls may
share the provider workspace because MFLA serializes that scratch internally.

MFLA also supplies ordinary residuals, normwise backward error, explicit
`x2 -> x3`, `x2 -> x4`, and `x3 -> x4` residuals, plus exactly one requested
factor correction. SDPX still decides whether to request a promoted residual,
whether to accept a correction, and whether to repeat refinement. Structured
KKT residuals, cone mappings, stopping policy, and certification stay in SDPX.

LU and LDLT are provider capabilities and tested internal seams. The
mathematical formulation is still chosen first: an explicit augmented request,
or the provider-neutral static planner, may require LDLT; only then may
backend planning select MFLA. MFLA availability never causes an augmented
choice by itself.

## BigFloatLinearAlgebra

The optional BFLA extension supplies provider-owned Cholesky, LU, LDLT, QR,
factor-owned solve, residual/refinement primitives, triangle/precision/
ownership validation, and Native/Generic backend provenance. BFLA QR no
longer has an exact-zero default rank policy; SDPX determines equality rank
from its explicit relative tolerance and the packed `R` diagonal.

The following routes remain SDPX-specific formulations rather than ordinary
dense provider work: native LP and fixed-trace Q3/SOCP kernels, sparse and
block-arrow Schur assembly and global-ID scatter, packed block-local
reduction/merge paths, BigFloat paths served by BFLA or explicit legacy
rollback, and solver-local AXPBY, stopping norms, fraction-to-boundary, and
iterate-update loops.

## Sparse providers

The sparse layer is a separate provider-neutral seam. Float64 uses CHOLMOD
through Julia's SuiteSparse. `Float64x2`/`Float64x3`/`Float64x4` and BigFloat
use the arithmetic-generic simplicial sparse Cholesky. Capabilities and
ownership are explicit (`CHOLMOD_SPARSE_CAPABILITIES`,
`GENERIC_SPARSE_CAPABILITIES`); a changed CSC pattern fails closed, while
unchanged patterns support numeric refactorization without repeating symbolic
ordering. See [sparse-execution.md](sparse-execution.md).
