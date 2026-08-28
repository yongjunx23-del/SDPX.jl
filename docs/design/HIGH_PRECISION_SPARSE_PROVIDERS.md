# High-precision sparse factor provider decision

Status: accepted design direction; implementation and performance promotion remain gated.

## Exact SDPX matrix structure

The frozen HSD signs produce the exact expanded operator

```text
[  0    A'     c   ]
[  A    -H     -b  ]
[ -c'   -b'   -k/t ]
```

The `(x,tau)` blocks are skew adjoints, so this matrix is genuinely
nonsymmetric. The exact reduced-Schur route is also nonsymmetric (`c-g` versus
`c+g`). A separate signed-regularized symmetric quasi-definite companion is
used to certify expected inertia; it is not the exact Newton operator.

Consequently a symmetric QDLDL factor alone cannot solve the frozen exact
Newton equations. Forcing the companion signs into the exact system would
change the homogeneous-gap equation and is forbidden.

## Decision

Do not add sparse symbolic analysis or sparse LU/LDLT implementations to
MultiFloatLinearAlgebra or BigFloatLinearAlgebra. Those libraries remain
independent dense/fixed-width providers. SDPX should evaluate two narrow,
optional sparse adapters whose responsibilities do not overlap:

| Responsibility | Candidate | Role |
|---|---|---|
| exact nonsymmetric sparse Newton solve | PureKLU.jl | generic sparse LU, BTF/AMD, symbolic/numeric phases, value refactor |
| symmetric companion inertia certification | QDLDL.jl | generic sparse LDLT, inertia/sign regularization, value update + refactor |
| generic no-pivot cross-check | LDLFactorizations.jl | benchmark-only reference, not a runtime dependency |
| budget-approved dense exact solve | MFLA/BFLA | dense LU/factorization only |

PureKLU and QDLDL are therefore complementary in SDPX: PureKLU supplies the
physical Newton direction; QDLDL supplies sparse inertia evidence. Adapters
must call both packages directly and must not reintroduce LinearSolve/SciMLBase.

If policy absolutely permits only one new package, retain PureKLU because
solving the exact equations is mandatory. That reduced configuration must mark
sparse inertia certification unsupported and may not claim the full expanded
KKT robustness contract. It is not the preferred production design.

## Evidence

Sources:

- QDLDL.jl: <https://github.com/osqp/QDLDL.jl>
- PureKLU.jl: <https://github.com/SciML/PureKLU.jl>
- LDLFactorizations.jl: <https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl>
- Clarabel arbitrary precision example: <https://clarabel.org/stable/literate/build/arbitrary_precision/>

A local Julia 1.12 probe used registered QDLDL 0.4.1, PureKLU 1.4.1,
LDLFactorizations 0.10.2, and MultiFloats 3.3.0. All three preserved the input
scalar type for BigFloat, Float64x2, and Float64x4 on small sparse systems.
Representative relative residuals were:

| scalar | QDLDL | LDLFactorizations | PureKLU |
|---|---:|---:|---:|
| BigFloat | 0 | 0 | 1.2e-77 |
| Float64x2 | 0 | 0 | 1.6e-32 |
| Float64x4 | 7.9e-67 | 7.9e-67 | 8.1e-65 |

This proves basic generic-scalar compatibility, not production robustness or
performance.

## Production routing

```text
Float64 exact nonsymmetric sparse solve       -> SuiteSparse/UMFPACK
BigFloat/MultiFloat exact nonsymmetric solve  -> PureKLU provider
symmetric companion inertia                   -> QDLDL provider
local 3x3 Exp/Power nonsymmetric blocks       -> MFLA/BFLA dense block LU
factor/provider rejection                     -> explicit memory-budget test
budget permits dense exact solve              -> BFLA/MFLA dense fallback
budget rejects dense                          -> typed InsufficientPrecision/factor failure
```

No route may silently convert values to Float64. Every accepted direction must
pass the unregularized semantic Newton residual in the original scalar type.
Terminal status still requires an original-coordinate certificate. A QDLDL
companion factor can reject a direction/regularization policy, but can never
replace the exact PureKLU/UMFPACK direction solve.

## Densification gate

Densification is a planned route, never an exception-driven fallback. The route
planner must estimate dense bytes, predicted factor fill, workspace bytes, and
precision-dependent scalar storage. It may select MFLA/BFLA only when the
configured memory budget admits the full exact factor and refinement workspaces.

## Promotion gates

Before either package becomes an SDPX weak dependency/provider:

1. manufactured exact Newton systems across Float64x2, Float64x4, BigFloat256;
2. symbolic pattern fixed once and numeric refactor once per epoch;
3. multi-RHS predictor/corrector/refinement reuse;
4. expected companion inertia/sign checks from QDLDL;
5. PureKLU exact-factor and unregularized five-equation direction residuals;
6. no scalar downcast and no global precision mutation;
7. fill, peak RSS, Julia/MPFR allocation, and wall-time comparison against dense MFLA/BFLA;
8. generic conic medium tier plus at least one bootstrap SDP;
9. failure ladder preserves the same accepted HSD iterate;
10. cross-check that companion certification never substitutes for the exact solve;
11. independent review before default or capability promotion.

LDLFactorizations remains an independent development reference because it
supports generic types and symbolic/numeric separation, but its own documentation
warns that the unpivoted method is less robust and less performant than advanced
sparse solvers. It is not a planned runtime dependency.
