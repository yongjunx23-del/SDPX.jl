# MultiFloatLinearAlgebra provider status

SDPX `0.5.0-DEV` resolves fixed `Float64x2`, `Float64x3`, and `Float64x4`
dense-Cholesky plans to the optional MultiFloatLinearAlgebra (MFLA) extension
when that complete provider is loaded. The `ExecutionPlan` remains the only
selection authority; a failed MFLA operation is never retried through generic
`LinearAlgebra` or the bundled legacy provider unless a structural fallback is
already present in the plan.

## Production boundary

The migrated ordinary dense path uses MFLA for dot, GEMV/GEMM, lower SYRK,
TRSV/TRSM, Cholesky factorization, vector/multiple-RHS solves, and equality
rank-revealing QR. RRQR retains an independent `MFWorkspace` with every live
factor payload. The same lifetime rule is used by the internal LU and pivoted
LDLT seams: two concurrently live borrowed factors never share an
`MFWorkspace`.

MFLA also supplies ordinary residuals, normwise backward error, explicit
`x2 -> x3`, `x2 -> x4`, and `x3 -> x4` residuals, plus exactly one requested
factor correction. SDPX still decides whether to request a promoted residual,
whether to accept a correction, and whether to repeat refinement. Structured
KKT residuals, cone mappings, stopping policy, and certification stay in SDPX.

LU and LDLT are provider capabilities and tested internal seams. They are not
automatically chosen: SDPX has not introduced a new augmented-KKT formulation
solely because MFLA exposes LDLT. Enabling that production route requires an
explicit formulation in the Midend/ExecutionPlan first.

## Deliberate remaining routes

The following paths still use SDPX-specific or legacy numerical code because
their semantics are not an ordinary dense factor/solve:

- native LP and fixed-trace Q3/SOCP kernels;
- sparse and block-arrow Schur assembly and global-ID scatter;
- packed block-local reduction/merge paths whose deterministic ownership is
  part of the solver;
- BigFloat paths (served separately by BFLA or the explicit legacy rollback);
- solver-local AXPBY, stopping norms, fraction-to-boundary, and iterate update
  loops.

The unused fixed-MultiFloat EPBLAS GEMM and column-packing helpers were removed.
Live EPBLAS SYRK/packed/scatter code remains until each production caller has a
matching MFLA local-kernel contract; global scatter will remain in SDPX.

GenericLinearAlgebra remains an explicit/reference and optional-provider-
unavailable route. It is not the normal production provider when a complete
MFLA extension is loaded, but it cannot yet be removed because supported
explicit `:standard` plans and unloaded-extension operation still depend on
Julia's generic `LinearAlgebra` method set.

