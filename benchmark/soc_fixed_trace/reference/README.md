# Optional Float64 direct-Q3 references

`benchmark_float64.jl` is a bounded reference harness for the immutable
reduced J40 CSDR payload.  It uses the same loader as
`../benchmark.jl`, requires the source payload SHA-256, and records both that
hash and the hash of the reduced serialized `SDPProblem` (as well as the
loaded SDPX source-tree hash).  Geometry is gated at
4,200 PSD2 blocks, 8,400 variables, and 170 equalities.

The harness builds the exact fixed-trace isomorphism

```text
X = [[t + u, v], [v, t - u]]  <=>  (t, u, v) in SecondOrderCone(3)
```

from `SDPX._compile_fixed_trace_q3`.  It first checks
`SDPX._fixed_trace_q3_rejection(problem) == :eligible` and verifies the
coefficient identity against the original sparse blocks.  If that predicate
rejects the reduced representation, the script stops and reports the precise
blocker; it never invents a lifted or approximate model.

Clarabel.jl and MosekTools.jl/Mosek.jl are optional.  Select one or both with
`--solver=clarabel|mosek|both`; an unavailable requested package is a hard,
actionable error.  Clarabel receives the equalities as one MOI `Zeros` vector,
while the current MosekTools wrapper receives equivalent scalar `EqualTo`
constraints (its vector-affine wrapper does not advertise `Zeros`); the
reduced data and direct Q3 cone map are otherwise identical.  This is only an
MOI encoding difference, not a model relaxation.  `--preflight-only` performs
the hash, exact reduction, geometry, and Q3 checks without requiring either
solver.  Each solver record
separates model build, MOI-to-solver setup, solve, validation, and end-to-end
time.  External primal and SOC-dual values are mapped back to original 2x2
blocks, then checked with `SDPX.result_certificate` in the reduced model's
coordinates.  The TOML keeps per-solver records under `solvers` and also
provides a flattened `runs` array for simple table ingestion.
Every measured repetition constructs a fresh optimizer and copies the same
MOI model; warm-up instances are not reused as timed instances. The report
records Julia, active Project/Manifest, external package version/source,
iteration/time/thread controls, and both benchmark-driver hashes. Any status
other than `OPTIMAL`, including `SLOW_PROGRESS`, is rejected and cannot appear
as a certified reference row.

Example:

```bash
julia --project=. bench/soc_fixed_trace/reference/benchmark_float64.jl \
  --solver=clarabel --model=/data/J40.bin --release=/data/CSDR \
  --expected-hash="$J40_SHA" --reps=1 --warmup=0 \
  --output=/results/j40-clarabel-float64.toml
```

The default certificate tolerance is `1e-8`, appropriate for a Float64
external reference; pass `--tolerance` explicitly when matching another
campaign row.  The output is not a replacement for the fixed-trace SDPX
benchmark and should not be compared across source or reduced-model hashes.
