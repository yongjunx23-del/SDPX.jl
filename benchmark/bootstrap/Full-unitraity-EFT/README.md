# CSDR FixedTrace NativeSOCP benchmark

## Reduced-dual L-BFGS path

`run_reduced_dual.jl` benchmarks the explicit analytic FixedTraceQ3 reduced
dual. It records a warm objective/gradient microbenchmark (at least two
warmups and ten measured repetitions), then runs the requested 1/5/20/full
L-BFGS gate. The runner never requests an IPM polish, PSD lift, Hessian, Gram
matrix, HVP, or CG solve.

```sh
SDPX_BENCH_ARITHMETIC=float64 \
SDPX_BENCH_MAX_ITERATIONS=20 \
SDPX_BENCH_REQUIRE_OPTIMAL=false \
julia -t 1 --project=<provider-env> run_reduced_dual.jl \
  /tmp/csdr-fixedtrace-reduced-neutral.bin /tmp/eft-rd-f64-i20
```

Supported selectors are `float64`, `float64x2`, `float64x3`, and
`float64x4`. Float64 uses the planned Standard/BLAS GEMV route; MultiFloat
requests MFLA explicitly and fails during planning if it is unavailable. The
reported interval is numerical at the requested arithmetic and tolerance
(`rigorous_interval=false`), not an outward-rounded real bound.

### Current reduced-dual evidence

On the final one-thread worktree, warm objective/gradient medians were about
`0.000198 s` (Float64), `0.00263 s` (Float64x2), and `0.02084 s`
(Float64x4), with zero measured steady-state allocation. The 200-step runs
took `0.109 s`, `0.736 s`, and `5.654 s`, respectively, and all preserved the
planned provider, `fixed_trace_q3`, no-PSD, and no-fallback invariants.

Those fast runs did **not** pass the numerical gate: all ended at `IterLimit`
with an invalid original-coordinate certificate. A diagnostic Float64 run
with 10,000 accepted steps took `2.604 s` but also failed certification
(`relative_gap=0.525`). The low-`tau` support Hessian becomes increasingly
ill-conditioned near nonsmooth active blocks, so cheap evaluations do not
translate into high-accuracy L-BFGS convergence on this model. The runner and
API therefore remain experimental; they do not replace the certified IPM.

This benchmark converts the archived CSDR model

```text
Gamma = [q r; r 2-q] >= 0
```

exactly into the Lorentz constraint `(1, q-1, r) in Q3`. It preserves the
BigFloat256-generated Float64x4 coefficients, the low-energy elimination, and
the 90-to-84 equality reduction from the archived study. Production solving
uses NativeSOC with the verified `fixed_trace_q3` specialization and the local
MFLA provider; it does not construct PSD matrices. The specialization uses
the historical HKM-Q3 direction in direct Lorentz coordinates. General
NativeSOC retains its Nesterov--Todd direction.

The script expects an environment containing the local SDPX, MFLA,
MultiFloats, and the dependencies of the archived `CSDRBootstrap` source.
The archived model path can be supplied explicitly:

```bash
julia --project=<prepared-environment> \
  benchmark/bootstrap/Full-unitraity-EFT/run.jl \
  /path/to/study/results/generate/model.bin \
  benchmark/out/bootstrap_full_unitraity_eft
```

When the archived serialized `SDPProblem` predates the current SDPX layout,
export the reduced arithmetic-neutral payload once with the archived vendored
environment:

```bash
julia --project=/path/to/archived/share/sdpx_c00_mac_singlecore \
  benchmark/bootstrap/Full-unitraity-EFT/export_reduced.jl \
  /path/to/study/results/generate/model.bin \
  /tmp/csdr-fixedtrace-reduced-neutral.bin
```

The production benchmark can consume that neutral payload directly. It
contains only reduced Float64x4 coefficients and the physical-objective map;
it contains no archived SDPX object.

Set `CSDR_STUDY_ROOT` when the archived source tree is not at the historical
default location. Results are written as `result.toml` and `RESULTS.md` under
the selected output directory. Benchmark outputs are intentionally ignored by
Git.

For profiling or a bounded diagnostic run, set
`SDPX_BENCH_MAX_ITERATIONS` and `SDPX_BENCH_MAX_TIME`. The first attempted
one-thread run of the Round 5 NativeSOC implementation was intentionally
stopped after it exceeded the historical solve time by a large margin; see
`NEGATIVE_OPTIMIZATION.md` before launching another full solve.

The numerical gate requires `Optimal`, a valid original-coordinate Lorentz
certificate, the `fixed_trace_q3` specialization, the MFLA backend, and
agreement with the archived physical objective.

The default arithmetic is `Float64x4`. Set
`SDPX_BENCH_ARITHMETIC=float64x2` to convert the same reduced coefficients to
`Float64x2` before problem construction and run the identical NativeSOC/HKM
path. The arithmetic choice is recorded in `result.toml`.

The real transformed `8400 x 84` equality panel can be compared without a
solver run:

```bash
julia -t 1 --project=<prepared-environment> \
  benchmark/bootstrap/Full-unitraity-EFT/gram_ab.jl \
  /tmp/csdr-fixedtrace-reduced-neutral.bin /tmp/gram-t1.toml
```

Repeat with `-t 2` and `-t 4`. The script compares MFLA SYRK with the retained
output-tiled SDPX kernel and checks the complete lower triangle. Threaded
kernel measurements are not compared directly with the historical one-thread
end-to-end solve.

## Validated regression fix

On the same one-Julia-thread/one-BLAS-thread Mac configuration and final
worktree, the Float64x4 reference run completed in 120 iterations with
`solve_seconds=57.2905` and `timing_total=56.3282`. The original-coordinate
certificate was valid, the
relative gap was `9.27e-14`, and the physical objective was inside the archived
certificate interval. Each iteration performed exactly one equality-panel
transform, one Gram SYRK, one equality factorization, and two KKT RHS solves.

The corresponding Float64x2 run also completed in 120 iterations, with
`solve_seconds=10.4755` and `timing_total=10.2199`. Its relative gap was
`9.51e-14`, the original-coordinate certificate was valid, and the physical
objective remained inside the same archived interval. Equality Gram assembly
remained the largest measured phase at `5.9825` seconds. These are separate
single-thread measurements; they do not imply a cross-thread speedup.
