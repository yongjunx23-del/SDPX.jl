# Finite-support LP benchmark

This benchmark exercises SDPX's dedicated LP path on the finite-energy-support
bootstrap model

```text
minimize or maximize  c' rho
subject to            B' rho = b
                      rho >= 0.
```

The primary case has 2,002 nonnegative variables and 31 equality constraints.
The 1,001-by-17 and 91-by-17 models are correctness and crossover controls.
All benchmark solves must run in a cluster batch allocation; the login node is
used only to stage immutable source and input data.

`benchmark.jl` reports model construction, solve, and validation separately.
It also records allocation, peak RSS, Julia and BLAS thread counts, the executed
Newton backend, objective values, residuals, and the scalar PSD margin.  The
independent equality audit uses

```text
abs(B[:,j]'rho - b[j]) /
max(1, abs(b[j]) + sum(abs.(B[:,j]) .* abs.(rho)))
```

because the unscaled moment rows span roughly 52 decimal orders of magnitude.
The raw absolute residual is retained and reported; it is not a meaningful
cross-precision accuracy measure by itself.

The source model is stored as round-trip `Float64` decimals.  Float64x4 and
BigFloat256 convert the parsed binary64 coefficients exactly.  These runs test
solver arithmetic and robustness on identical data; they do not claim that the
model coefficients contain more than binary64 input accuracy.

Recommended cluster matrix:

| Case | Arithmetic | Julia threads | BLAS threads | Repetitions |
|---|---|---:|---:|---:|
| 91 x 17 | all | 1 | 1 | correctness gate |
| 1,001 x 17 | all | 1, 2, 4, 8 | 1 except Float64 |
| 2,002 x 31 | Float64 | 1, 2, 4, 8, 16 | match Julia/BLAS sweep |
| 2,002 x 31 | Float64x4 | 1, 2, 4, 8, 16 | 1 |
| 2,002 x 31 | BigFloat256 | 1, 2, 4, 8 | 1 |

For Float64, use BLAS threads and one Julia packing worker.  For Float64x4 and
BigFloat, use one BLAS thread; the reduced Schur kernel assigns complete,
disjoint lower-triangular tiles to Julia tasks.  Do not use nested BLAS and
Julia parallelism.
