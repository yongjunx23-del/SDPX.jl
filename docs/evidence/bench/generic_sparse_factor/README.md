# Generic sparse-factor microbenchmark

`benchmark.jl` records the baseline behavior of SDPX's arithmetic-generic
sparse Cholesky numeric refactorization.  It uses one deterministic symmetric
strictly diagonally dominant tridiagonal CSC pattern per arithmetic mode:

| arithmetic | dimension | precision |
| --- | ---: | ---: |
| `Float64` | 1024 | binary64 |
| `Float64x4` | 128 | four-component MultiFloats |
| `BigFloat256` | 64 | 256-bit MPFR |

The script warms factorization/refactorization/solve in one Julia process, then
collects at least nine independent samples for numeric refactorization and
solve.  Each sample changes only `nzval` while retaining the exact CSC
`colptr`/`rowval` pattern; the reported residual is computed against that
sample's final matrix.  `summary.toml` and `summary.tsv` include source and
input fingerprints, timing median/min/max/MAD, warm allocation bytes,
factorization counts, and a high-precision residual string.  A missing
MultiFloats installation is represented as a structured `status = "skip"` row.

Run from the repository root:

```sh
julia --project=. bench/generic_sparse_factor/benchmark.jl \
  --output=work/baseline/generic_sparse_factor
```

Comparison rules: compare rows only when arithmetic, dimensions, pattern and
value-schedule fingerprints, Julia/BLAS thread settings, package/manifest
identity, and the driver/source hashes match.  Solver source hashes are
expected to differ for a candidate optimization; the other fields must remain
identical.  The timings are in-process and therefore exclude process startup,
JIT compilation, and operating-system scheduling noise.  The script records no
RSS/peak-memory measurement; warm `@allocated` covers Julia allocation counts
for the isolated calls but not allocator arenas or external library memory.
