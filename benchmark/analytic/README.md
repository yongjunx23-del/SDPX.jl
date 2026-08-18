# Analytic benchmark suite

This directory supplies deterministic mathematical oracles to SDPX's
canonical benchmark registry. There is one runner, one result schema and one
comparison path:

```sh
julia --project=. benchmark/runner.jl analytic_fast \
  --allow-semantic-failures --output=/tmp/sdpx-analytic-fast.toml

julia --project=. benchmark/runner.jl analytic_numerical \
  --samples=3 --allow-semantic-failures \
  --output=/tmp/sdpx-analytic-numerical.toml
```

`benchmark/analytic/run.jl` is only a compatibility wrapper around that
runner. Problem construction, JIT warm-up, provider selection, ExecutionPlan,
sampling, original-coordinate certification, TOML/TSV output and comparison
all remain in `SDPXBenchmarkRegistry`.

## Families

| family | oracle | registered scale/pathology axes | primary component |
| --- | --- | --- | --- |
| Chebyshev LP | `t*=2^(1-n)` | `n=8,16,24,32,48,64,96,128`; Chebyshev/monomial basis | LP precision, conditioning, scaling |
| weighted minimum-norm SOCP | `t*=1/sqrt(sum d_i^-2)` | `n=32,128,512,2048,8192`; normalized base-2 spreads `0,10,20,40` | one large Lorentz cone |
| Basel chain | `t_N*=sqrt(sum(k^-2))` | `N=10,100,1000,10000`; native Q3/PSD2; spreads `0,10,20,40` | many tiny cones and homogeneous scaling |
| spectral SDP | `2cos(pi/(n+1))` | `n=8,16,32,64,128,256`; single/direct sum; `delta=0,2^-10,...,2^-160` | one equality, growing PSD, degeneracy |
| odd-cycle MaxCut SDP | `n/2*(1+cos(pi/n))` | `n=5,7,15,31,63,127,255`; clean/redundant equality | rank detection and low-rank optimum |
| rational moment SDP | `-log(1-rho)/rho` | `m=1,2,4,8,16,32`; orders `4,8,12,16,24,32`; lower/upper | moment conditioning and high precision |

The registry contains the full requested grid. The executable suites sample it
instead of taking a costly Cartesian product:

- `analytic_fast`: small Float64 correctness and formulation pairs.
- `analytic_numerical`: moderate conditioning plus selected MFLA/BFLA rows.
- `analytic_stress`: large/extreme cases, gated by `--allow-large`.

## Correctness gate

Every row is classified `PASS`, `FAIL` or `UNRESOLVED`. `PASS` requires an
appropriate solver status, the typed analytic objective/bound tolerance,
finite residuals and gap, cone feasibility, and an independently recomputed
original-coordinate certificate. An `Optimal` row with a wrong oracle value
is `FAIL`; a limit, stall or numerical breakdown is `UNRESOLVED`.

Rows also carry group gates:

- stable versus monomial Chebyshev;
- native SOC versus PSD2 and rescaled Basel;
- single versus degenerate spectral formulations;
- clean versus redundant MaxCut;
- moment lower/upper bracketing and monotonic tightening across orders.

Any failed group is excluded from performance comparison. The comparator emits
time ratios only when both baseline and candidate rows are eligible. Use
`--allow-semantic-failures` to retain a failure-map artifact while locating a
current solver boundary; omitting it keeps strict regression behavior.

## Result contract

Schema 4 records dimensions, cone composition, equality and nonzero counts,
arithmetic/provider/precision, planned and executed route, status and
classification, physical/analytic objectives, absolute/relative error,
`b_correct`, residuals, gap, cone violations, complementarity, certificate
kind/failures/provenance, iterations, factorization/refinement counts, setup
and kernel phases, internal and independent certification time, total wall
time, allocations, workspace/peak-memory facts, fingerprints and sample
median/MAD. At least three samples are required for timing claims.

## Local baseline, 2026-08-18

On the one-thread Apple local run, weighted SOCP `n=32`, all three Basel
`N=10` representations/scales, spectral `n=8` single/exact-degenerate, and
MaxCut `n=5` clean/redundant passed their individual and group gates.
Chebyshev `n=8` returned `Optimal` with valid certificates but only about 23.4
correct objective bits (`~9.08e-8` relative error versus a requested `1e-8`),
so both basis rows are correctly classified `FAIL`. Float64 moment orders 4
and 8 were `UNRESOLVED` (`NumericalBreakdown`/`Stalled`); a BigFloat-256 order-4
lower solve reached `Optimal` with a valid certificate, confirming a precision
wall rather than an invalid oracle.

The first benchmark-driven kernel fix was generic sparse CSC traversal in the
native SOC metric assembly. On Basel `N=1000`, the warmed solve improved from
31.5 s / 2.00 GB to 0.405 s / 36.9 MB (about 77.8x faster and 54x fewer
allocations), with identical objective/status/iterations. The remaining sparse
SOC loop still scans inactive variable columns and is a documented scaling
target; it is not hidden by a Basel-specific branch.
