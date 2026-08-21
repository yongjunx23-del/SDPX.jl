# NativeSOC metric-assembly microbenchmark

`benchmark.jl` measures the existing production `_native_soc_add_metric!`
kernel through `NativeSOCWorkspace` and `build_execution_plan`, with
`specialization=:off` explicitly forcing `general_lorentz`.  Each arithmetic
mode runs two deterministic profiles:

| profile | matrix data | purpose |
| --- | --- | --- |
| `sparse_active3` | exactly three scattered nonzero columns per block | primary sparse-aware bottleneck fixture |
| `dense_as_csc` | every column nonzero, stored as CSC in the sparse lane | dense-as-CSC regression guard |

The bounded cases are Float64 (16 blocks × 3 coordinates × 256 variables),
Float64x4 (8 × 3 × 64), and BigFloat256 (4 × 3 × 32).  Dense and CSC lanes use
the same exact values, offsets, and deterministic strictly interior primal/dual
states.  NT scaling is initialized through the production `_native_soc_scaling!`
routine before the warm-up and at least nine independent timed samples.  TOML
and TSV rows include timing median/min/max/MAD, warm allocations, full
problem/state fingerprints, metric fingerprints, high-precision norm and
dense-vs-sparse difference strings, exact/approximate parity, active-column
counts, and analytic coordinate-iteration counts.

The CBLIB `nql30.cbf.gz` row is structural only: it verifies the canonical
compressed SHA-256 and parses VAR/CON/OBJACOORD/ACOORD/BCOORD sections without
allocating a `NativeSOCWorkspace`.  It records 4501 variables, 900 Q3 blocks,
ACOORD nnz, Q3 coefficient nnz, equality coefficient nnz, equality offsets,
active-column distribution/fingerprint, and current-versus-active coordinate
iteration estimates.  A missing or checksum-mismatched cache becomes a
machine-readable skip row.

Run from the repository root:

```sh
julia --project=. bench/soc_metric_assembly/benchmark.jl \
  --output=work/baseline/soc_metric_assembly
```

Compare candidate rows only when arithmetic/profile, dimensions, input/state
fingerprints, Julia/BLAS thread counts, source/driver/project/manifest hashes,
and environment metadata match.  A solver-source hash is expected to change
after an optimization; the benchmark driver and inputs must not.  Timings are
in-process and exclude process startup/JIT.  Warm `@allocated` reports Julia
allocation counts for the isolated assembly calls, not RSS, allocator arenas,
or external-library memory.
