# MultiFloatLinearAlgebra (SDPX prototype)

This directory is an SDPX-first prototype of a future `MultiFloatLinearAlgebra.jl` backend.

Goal:

```
SDPX kernel API
      |
      v
MultiFloat backend
      |
      +-- MultiFloatVec microkernels
      +-- SYRK
      +-- TRSM
      +-- DOT/GEMV
      +-- blocked Cholesky
```

This is intentionally not a full BLAS replacement. The first target is high-precision conic optimization workloads:

- SDP Schur assembly
- KKT factorization
- predictor/corrector solves
- LP/SOCP extensions later

Design rules:

1. Solver code never depends on `Float64xN` details.
2. Backend selection happens through `ExecutionPlan`.
3. Numerical order must remain deterministic when required by certification.
4. SIMD is applied across independent outputs, not by changing reduction order.

Planned modules:

- `backend.jl` - backend interface
- `traits.jl` - Float64x2/x3/x4 traits
- `packing.jl` - SIMD-friendly panel layouts
- `microkernels/` - dot/gemm/syrk/trsm kernels
- `factorization/` - blocked Cholesky and refinement helpers
