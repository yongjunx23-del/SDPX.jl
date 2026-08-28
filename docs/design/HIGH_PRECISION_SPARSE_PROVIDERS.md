# High-precision sparse factor provider decision

Status: accepted design direction; implementation and performance promotion remain gated.

## Decision

Do not add sparse symbolic analysis or sparse LU/LDLT implementations to
MultiFloatLinearAlgebra or BigFloatLinearAlgebra. Those libraries remain
independent dense/fixed-width numerical providers. SDPX should consume existing
pure-Julia sparse packages through narrow optional provider extensions:

| KKT structure | First candidate | Role |
|---|---|---|
| symmetric quasi-definite expanded KKT | QDLDL.jl | generic sparse LDLᵀ, inertia/sign regularization, value update + refactor |
| general/nonsymmetric reduced Schur | PureKLU.jl | generic sparse LU, BTF/AMD, symbolic/numeric phases and value refactor |
| symmetric no-pivot reference | LDLFactorizations.jl | validation/benchmark reference, not first production fallback |
| budget-approved dense fallback | MFLA/BFLA | dense factorization only |

Adapters must call these packages directly; they must not reintroduce
LinearSolve/SciMLBase.

## Evidence

Sources:

- QDLDL.jl: <https://github.com/osqp/QDLDL.jl>
- PureKLU.jl: <https://github.com/SciML/PureKLU.jl>
- LDLFactorizations.jl: <https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl>
- Clarabel arbitrary precision example: <https://clarabel.org/stable/literate/build/arbitrary_precision/>

A local Julia 1.12 probe used registered QDLDL 0.4.1, PureKLU 1.4.1,
LDLFactorizations 0.10.2, and MultiFloats 3.3.0. All three preserved the
input scalar type for BigFloat, Float64x2, and Float64x4 on small sparse
systems. Representative relative residuals were:

| scalar | QDLDL | LDLFactorizations | PureKLU |
|---|---:|---:|---:|
| BigFloat | 0 | 0 | 1.2e-77 |
| Float64x2 | 0 | 0 | 1.6e-32 |
| Float64x4 | 7.9e-67 | 7.9e-67 | 8.1e-65 |

This proves basic generic-scalar compatibility, not production robustness or
performance.

## Production routing

```text
Float64 sparse reduced Schur       -> SuiteSparse/UMFPACK
BigFloat/MultiFloat quasi-definite -> QDLDL provider
BigFloat/MultiFloat general LU     -> PureKLU provider
factor/provider rejection          -> explicit memory-budget test
budget permits dense               -> BFLA/MFLA dense fallback
budget rejects dense               -> typed InsufficientPrecision/factor failure
```

No route may silently convert values to Float64. Every accepted direction must
pass the unregularized semantic Newton residual in the original scalar type.
Terminal status still requires an original-coordinate certificate.

## Densification gate

Densification is a planned route, never an exception-driven fallback. The route
planner must estimate dense bytes, predicted factor fill, workspace bytes, and
precision-dependent scalar storage. It may select MFLA/BFLA only when the
configured memory budget admits the full factor and refinement workspaces.

## Promotion gates

Before either package becomes an SDPX weak dependency/provider:

1. manufactured Newton systems across Float64x2, Float64x4, BigFloat256;
2. symbolic pattern fixed once and numeric refactor once per epoch;
3. multi-RHS predictor/corrector/refinement reuse;
4. expected inertia/sign checks for QDLDL;
5. factor and unregularized direction residuals;
6. no scalar downcast and no global precision mutation;
7. fill, peak RSS, Julia/MPFR allocation, and wall-time comparison against dense MFLA/BFLA;
8. generic conic medium tier plus at least one bootstrap SDP;
9. failure ladder preserves the same accepted HSD iterate;
10. independent review before default or capability promotion.

LDLFactorizations is retained initially as an independent reference because it
supports generic types and symbolic/numeric separation, but its own documentation
warns that the unpivoted method is less robust and less performant than advanced
sparse solvers. It should not be the first production fallback for difficult
indefinite KKT systems.
