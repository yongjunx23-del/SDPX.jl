# BigFloat kernel results

Measured on Julia 1.12.6, Apple M4, macOS, with one Julia thread and one BLAS
thread. Each comparison uses identical 256-bit inputs, warms both paths, and
reports the median of repeated samples from
`benchmark_hot_kernels.jl`. These are kernel measurements, not claims about
unrelated SDP shapes or hardware.

| Kernel | Reference | Optimized | Speedup | Allocated bytes, reference → optimized |
|---|---:|---:|---:|---:|
| Cholesky solve, `n=32` | allocation-based generic path | owned MPFR path | 2.56× | 3,725,824 → 57,680 |
| Lower triangular solve | generic path | buffered MPFR path | 2.67× | about 30× fewer |
| Transposed triangular solve | generic path | buffered MPFR path | 2.54× | about 30× fewer |
| `kaxpby!` | non-buffered scalar loop | buffered MPFR loop | 3.95× | substantially reduced |
| Infinity norm | allocating reduction | scalar buffered reduction | 7.05× | 2,240,000 → 224 |
| Independent zero initialization | `x + zero(BigFloat)` | `mutable_copy` | 5.45× | reduced |
| Owned matrix multiplication | alias-safe public path | owned destination path | approximately neutral | 98,528 → 224 |

At 1024 bits and `n=32`, the optimized Cholesky solve was 1.59× faster with
64.6× fewer allocated bytes; the transposed triangular solve was 1.85× faster
with 29.3× fewer bytes. The infinity norm improved by 9.97×.

On a matched small BigFloat SDP workload, the complete solve changed from
4.1976 s and 894.3 MB allocated to 3.6867 s and 588.7 MB allocated: a 1.14×
runtime improvement and a 34.2% allocation reduction. GC time fell from
0.1871 s to 0.0377 s. Status, six-iteration count, residuals, and the final
gap agreed.

Run the benchmark with:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
julia --project=. --startup-file=no \
  bench/bigfloat_kernels/benchmark_hot_kernels.jl
```
