# Automatic pipeline

Every public solve runs a conservative preparation pipeline:

1. classify the cone, storage, arithmetic, size, and expected Schur density;
2. remove exact zero, duplicate, and verified dependent equalities;
3. extract scalar bounds and eliminate exactly fixed variables;
4. select scaling or Ruiz equilibration;
5. estimate formulation, kernel, factorization, memory, and scheduling costs;
6. solve the reduced model; and
7. reconstruct and certify the original-coordinate result.

Transformations that are not fully implemented or not numerically justified,
including general primal-to-dual conversion and chordal decomposition, remain
analysis-only.

```julia
result = solve(
    problem;
    presolve=:auto,
    scaling=:auto,
    algorithm=:auto,
    threads=8,
    diagnostics=true,
)
```

Inspect decisions after the solve:

```julia
result.diagnostics.classification
result.diagnostics.plan
result.diagnostics.presolve
result.diagnostics.selected_algorithms
result.diagnostics.timings
result.diagnostics.memory
result.diagnostics.warnings
```

Sparse coefficient matrices do not necessarily imply a sparse Schur matrix.
The selector uses active incidence density and predicted Schur density
separately, which is important for lattice-bootstrap models with sparse
individual coefficients but dense aggregate coupling.

See the
[preprocessing design](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/preprocessing.md)
and
[automatic pipeline design](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/automatic-optimization-pipeline.md)
for stage invariants and reconstruction details.
