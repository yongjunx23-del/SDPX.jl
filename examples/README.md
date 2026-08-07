# SDPX examples

Runnable examples, ordered so that each one builds on the last. They are
executed by the test suite (`test/examples.jl`), so if they are here they run.

## Setup

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run any example:

```bash
julia --project=examples examples/01_basic_sdp.jl
```

## The examples

| File | What it shows |
| --- | --- |
| [`01_basic_sdp.jl`](01_basic_sdp.jl) | The problem format, on an SDP whose optimum is `2√6` in closed form |
| [`02_extended_precision.jl`](02_extended_precision.jl) | Where `Float64` runs out and `Float64x4` keeps going — the reason this solver exists |
| [`03_sparse_lp.jl`](03_sparse_lp.jl) | The dedicated LP path, and how it decides between a sparse and a dense factorization |
| [`04_certificates.jl`](04_certificates.jl) | Verifying a solution independently, including a solve the certificate refuses to accept |
| [`05_jump.jl`](05_jump.jl) | The same problem through JuMP, via MathOptInterface |
| [`07_convex.jl`](07_convex.jl) | LP, SOCP, and SDP modeling through Convex.jl, including SDPX's packed-triangle PSD frontend |

## What `02` is really about

The headline result, measured by the example itself:

| arithmetic | tolerance | status | error |
| --- | --- | --- | --- |
| `Float64` | 1e-8 | Optimal | 2.107e-08 |
| `Float64x4` | 1e-8 | Optimal | 2.107e-08 |
| `Float64` | 1e-14 | **Stalled** | 1.781e-13 |
| `Float64x4` | 1e-14 | Optimal | 2.107e-14 |
| `Float64x4` | 1e-30 | Optimal | 2.107e-30 |

Two things follow. Extended precision buys nothing until `Float64` actually
runs out — at 1e-8 the two agree to every digit, and the wider type is pure
cost. And once it does run out, no amount of iterating recovers it: `Float64`
stalls rather than converging slowly. Reach for `Float64x4` when a solve
stalls short of your tolerance, not before.

These numbers are from one small, well-conditioned problem. Real bootstrap
programs exhaust `Float64` far earlier, which is the case the solver is built
for; the mechanism is the same, the crossover is not.
