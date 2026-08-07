# Native SDPX versus Convex.jl frontend benchmark

This benchmark measures the cost and numerical result of two ways to express
the same conic problem:

- `native`: construct SDPX block arrays directly and call `SDPX.solve!`;
- `convex`: construct a DCP model with Convex.jl and call
  `Convex.solve!(problem, SDPX.Optimizer)`.

The three deterministic examples cover a nonnegative simplex LP, a minimum
Euclidean-norm SOCP, and a MaxCut-style SDP relaxation. Each process runs only
one frontend so its peak RSS remains attributable to that frontend. Data
generation and Julia compilation are separated from the warmed repetitions.
The default sizes are 512 LP variables, SOCP dimension 24, and SDP side 6;
override any of them with `--size=N`.

The reported boundaries are:

- `build`: native array ingestion or Convex expression construction;
- `frontend overhead`: Convex conic canonicalization, MOI bridges, copy to
  SDPX, and result recovery, estimated as the Convex `solve!` wall time minus
  `MOI.SolveTimeSec`;
- `core_solver`: the SDPX optimizer time reported through MOI;
- `validation`: independent objective, equality, and cone checks;
- `end_to_end`: build plus frontend solve plus validation.

Both frontends use the same requested validation tolerance. The CSV also
records the solver's primal and dual objectives, relative gap, primal and dual
residuals, and certificate decision; a non-optimal or uncertified run exits
with an error after preserving its CSV for diagnosis.

Install the isolated environment without changing the package environment:

```bash
julia --project=bench/convex_frontend -e '
  using Pkg
  Pkg.develop(path=".")
  Pkg.instantiate()'
```

Example matched runs:

```bash
for case in lp socp sdp; do
  julia -t 1 --project=bench/convex_frontend \
    bench/convex_frontend/benchmark.jl \
    --case=$case --frontend=native --arithmetic=Float64 \
    --repetitions=5 --threads=1 --output=/tmp/${case}_native.csv

  julia -t 1 --project=bench/convex_frontend \
    bench/convex_frontend/benchmark.jl \
    --case=$case --frontend=convex --arithmetic=Float64 \
    --repetitions=5 --threads=1 --output=/tmp/${case}_convex.csv
done
```

Use a separate process for each arithmetic type. `Float64x4` and
`BigFloat256` are supported by the harness, but frontend-overhead comparisons
should first be established with Float64. Set `SDPX_BLAS_THREADS` explicitly
and avoid nested Julia/BLAS oversubscription.

Convex emits square PSD cones. MathOptInterface's standard square-to-triangle
bridge enforces symmetry and converts them to the triangle representation
accepted by SDPX. Symbolically symmetric expressions, including ordinary
`Convex.Semidefinite` variables, follow the intended bridge path. A native
upper-triangle model avoids the full-matrix variables, symmetry equalities, and
bridge/copy work. The benchmark therefore reports model construction
separately rather than presenting only a single solve time.

`process_peak_rss_raw` is the process high-water mark and therefore includes
package loading, compilation, and warm-up. The deterministic SDP weights are
generated as Float64 and then converted; extended-precision runs compare
frontend and solver arithmetic on identical binary64-origin data, not the
accuracy of independently generated high-precision coefficients.
