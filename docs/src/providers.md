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

The plan records the selected provider/formulation pair and any authorized
runtime fallback. A provider failure does not implicitly change that plan.

## Standard and legacy providers

Julia's `LinearAlgebra` (BLAS/LAPACK) covers ordinary Float64 dense work and
is the reference route. The bundled `SDPXLegacyLA` provider owns the
historical `k*` kernels behind the `la_*` dispatcher seam for generic and
BigFloat arithmetic. It advertises exactly
`SDPX_LEGACY_LA_CAPABILITIES`: `:dot`, `:norminf`, `:mul`, `:mul_owned`,
`:syrk`, `:chol`, `:cholesky_factor!`, `:solve`, `:cholsolve_owned`, `:trsm`,
`:trsv_lower`, `:trsv_transpose`, `:axpby`, and `:axpby_owned`.

The solver-facing `la_syrk!` contract is lower-triangle-authoritative.  A
provider may leave the upper triangle unchanged, poisoned, or materialized as
an implementation detail; dense equality Cholesky, pivoted compatibility, and
augmented assembly select the lower triangle explicitly.  The bundled legacy
kernel may still mirror both triangles, but no migrated solver path relies on
that extra work.  Provider tests deliberately poison the inactive upper
triangle to keep this boundary auditable.

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

Pivoted LDLT status, block layout, final permutation, and inertia are queried
through MFLA's lightweight public accessors.  Full `factor_diagnostics` is a
cold observability operation: an augmented factorization records it once for
the accepted factor or final rejected candidate, rather than rescanning every
retry for each metadata field.

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

One precision-matched `BFLAWorkspace` is created lazily per instantiated SDPX
provider and reused by sequential factorizations and predictor, corrector, and
correction solves.  Because SDPX owns the factor buffer and does not mutate it
between those solves, the adapter uses BFLA's explicit trusted solve boundary;
the checked public BFLA semantics remain unchanged for every other caller.
The workspace is never shared concurrently.  BFLA's one-step correction is
used only when it is the same retained factor equation (for example the full
augmented LDLT); SDPX still forms the KKT residual and owns acceptance,
stopping, precision escalation, and fallback.

A successful BFLA Cholesky is a factorization fact, not an equality-rank
decision.  SDPX compares the public lower-factor diagonal quality against its
requested-accuracy and explicit factor/problem precision, then either accepts
normal equations, invokes only a plan-authorized RRQR/pivoted route, or fails
closed.  Ambient MPFR precision is not part of that decision.

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
