# BigFloat sparse block-arrow benchmark

This benchmark isolates the high-precision kernels used by sparse CSDR
problems made of `2x2` PSD blocks:

- packed primal-block construction;
- packed dual contraction;
- fused block-arrow Schur assembly.

It compares the allocation-heavy scalar reference implementations with the
owned-buffer MPFR kernels, checks the relative contraction and Schur errors,
and reports median runtime, speedup, allocated bytes, and allocation counts.
BigFloat is intentionally serial.

Generate a CSDR artifact with `bench/csdr_psd_dual/prepare_problem.jl`, then
run:

```bash
JULIA_NUM_THREADS=1 \
SPARSE_INPUT=/path/to/problem.bin \
BIGFLOAT_BITS=256 \
BENCH_SAMPLES=7 \
julia --project=. bench/bigfloat_sparse_schur/benchmark.jl
```

The default output is
`bench/bigfloat_sparse_schur/results/bigfloat-sparse-schur.csv`. Benchmark
inputs and results are ignored by Git because they are machine-specific and
can be regenerated.

Representative validated measurements are summarized in
[RESULTS.md](RESULTS.md).

To stress the same inner loop at the 385-active-variable size of the large
CSDR blocks without loading a multi-gigabyte artifact, run the self-contained
single-block case:

```bash
JULIA_NUM_THREADS=1 \
SYNTHETIC_ACTIVE=385 \
BIGFLOAT_BITS=256 \
BENCH_SAMPLES=7 \
julia --project=. bench/bigfloat_sparse_schur/benchmark.jl
```
