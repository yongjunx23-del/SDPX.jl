# BigFloat sparse and block-arrow results

Measured on Julia 1.12.6, Apple M4, macOS, at 256-bit precision with one Julia
thread and one BLAS thread. The real CSDR `s15` artifact is the small
representative sparse problem. All optimized results were compared with the
allocation-heavy reference formula in the same process.

| Stage | Reference | Optimized | Speedup | Allocated bytes, reference → optimized |
|---|---:|---:|---:|---:|
| Packed `buildP!` | 1.839 ms | 0.609 ms | 3.02× | 3.06 MB → 0 |
| Packed accumulation | 1.636 ms | 0.501 ms | 3.27× | 3.02 MB → 0.121 MB |
| Fused-arrow Schur | 16.604 ms | 4.869 ms | 3.41× | 30.44 MB → 0.282 MB |
| Arrow KKT factorization | 6.233 ms | 2.989 ms | 2.09× | 10.69 MB → 0.209 MB |
| Arrow KKT solve | 1.607 ms | 0.508 ms | 3.16× | 2.86 MB → 0.042 MB |
| General sparse COO Schur | 2.718 ms | 0.866 ms | 3.14× | 6.87 MB → 0.013 MB |

For a synthetic block with 385 active variables, fused Schur assembly changed
from 32.20 ms, 50.97 MB, and 910,160 allocations to 14.81 ms, 784 bytes, and
14 allocations: a 2.17× runtime improvement.

The first three rows were rerun after the final fused-arrow selector change;
the KKT optimized rows were rerun after the owned-output reset change. The
reference KKT values are the matched pre-optimization measurements.

The general sparse COO row is a same-process synthetic kernel benchmark with
96 active variables, one `10x10` PSD block, and four structural entries per
constraint. It measures the non-`2x2`, non-fused COO contraction path and is
therefore separate from the CSDR block-arrow rows above. The optimized result
used 99.81% fewer allocated bytes and matched the reference Schur entries
exactly.

The relative Schur error was exactly zero in both checked cases. Dedicated
tests also verified an arrow KKT residual below `1e-65`, no destination
aliasing, and unchanged source Schur blocks, right-hand sides, coupling
matrices, and problem data. BigFloat remains serial.

See [README.md](README.md) for the reproducible commands and input-generation
instructions.
