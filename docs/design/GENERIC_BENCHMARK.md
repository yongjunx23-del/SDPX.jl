# Generic conic benchmark suite

Status: implemented on `agent/bench-generic` (2026-08-28)
Scope: general-purpose LP/SOCP/SDP/EXP/POWER behavior, deliberately separate from
`benchmark/bootstrap/` and `benchmark/bootstrap/physics/`.

## Why these sources

The corpus follows formats and model families used to compare mature conic
solvers rather than inventing physics-specific surrogates:

| family | recognized source | use in this suite |
|---|---|---|
| LP | [NETLIB LP](https://www.netlib.org/lp/data/) | checksum-pinned AFIRO (small) and ADLITTLE (medium), published reference objectives; native MPS reader |
| SOCP / continuous conic | [CBLIB](https://github.com/HFriberg/cblib-base) and its CBF 2.x specification | native CBF section reader; deterministic nearest-point and Markowitz generators fill local/cluster size tiers |
| SDP | [SDPLIB 1.2](https://github.com/vsdp/SDPLIB) | checksum-pinned `control1` and `mcp100`, published reference objectives; sparse SDPA reader; theta/Max-Cut/seeded PSD generators |
| large sparse SDP | Mittelmann/Gondzio sparse SDP collections | documented source for a future cluster expansion; not silently claimed as cached |
| EXP | MOSEK Modeling Cookbook, [exponential cone](https://docs.mosek.com/modeling-cookbook/expo.html) | entropy and log-sum-exp epigraph generators |
| POWER | MOSEK Modeling Cookbook, [power cone](https://docs.mosek.com/modeling-cookbook/powo.html) | weighted geometric mean and p-power epigraph generators; no comparably broad public power-cone library is known |

The CBLIB fixture currently cached is a reader-only mixed-integer instance from
Hypatia's CBLIB campaign. SDPX does not claim MIP support. Continuous CBLIB
solve cases should only be promoted after a checksum-pinned upstream pack is
available and the native CBF-to-`Model` lowering is reviewed.

## Reproducible public-data cache

Run once:

```bash
benchmark/general/scripts/fetch_generic_benchmarks.sh
```

Downloads go under the gitignored `benchmark/general/data/` cache. The committed
`data/MANIFEST.sha256` is authoritative and the script fails on any byte change.
The current manifest contains:

| file | SHA-256 | documented objective |
|---|---|---:|
| NETLIB `afiro.mps` | `9cd304f02717cbd6f85068cb777b69d28539b22a4868ae0f0fb425f514f0eea5` | `-4.6475314286e2` |
| NETLIB `adlittle.mps` | `ed99da009e35279828219ff7f04a2cd4f170692bedf58f704b622338e4adc1f9` | `2.2549496316e5` |
| SDPLIB `control1.dat-s` | `482528bb128e64dad102fab88e4e8b7074efdfa22e396ebec586d832b1545bcb` | `1.778463e1` |
| SDPLIB `mcp100.dat-s` | `a33665823d81f4ba1285272b355cefc2d3307a1f5fb8bb933edee58b3615a9b8` | `2.261574e2` |
| CBF reader fixture | `5b0aadaf1c121f6e32ca1c6d33d9f3d51aa8dfedcfed660ecb1cfc02e7aff0bf` | reader only (contains integer declarations) |

`src/mps.jl`, `src/sdpa.jl`, and `src/cbf.jl` are dependency-free native
readers. Missing data produces an instruction naming the fetch script. They are
fail-closed format boundaries: MPS integer markers/bounds, RANGES, and unknown
bound/extension sections are rejected; CBF accepts exactly version 2 with exact
`MIN`/`MAX` sense and rejects `INT`; SDPA checks every dimension, index,
upper-triangle coordinate, diagonal-block coordinate, duplicate, and finite
numeric value. The checksum-pinned CBLIB fixture intentionally exercises the
`INT` rejection and is never relaxed into a continuous model. The MPS reader
also lowers the curated standard-form continuous subset through SDPX's public
`Model` API. External reference values live in `src/external.jl`; use
`reference_matches`, which applies a scaled absolute/relative check instead of
a self-recorded baseline.

## Generated tiers

Every random generator uses `Random.Xoshiro` with its seed stored in the
problem's `params` named tuple. No generator reads global RNG state.

| tier | intended lane | LP | SOCP | SDP | EXP | POWER |
|---|---|---|---|---|---|---|
| small | local; each solve <30 s | AFIRO-style, degenerate, infeasible ray, unbounded ray, boxed random `n=8` | nearest simplex `n=3`, Markowitz `n=4`, 100:1 ill-scaled risk map | theta K4, Max-Cut K4, random/rank-one-face `n=2,4` | entropy/log-sum-exp `n=3` | geometric mean, p-power `n=3` |
| medium | opt-in; seconds to minutes | boxed random `n=60` + NETLIB ADLITTLE | nearest simplex `n=24` | random PSD `n=14` + SDPLIB control1 | log-sum-exp `n=12` | p-power `n=12` |
| large | PBS only; generation-ready | boxed random `n=1200` | one SOC of order 1501 | PSD order 100 + SDPLIB mcp100 | 256 Exp blocks | 256 Power blocks |

Large tiers intentionally use dense dimensions that can require hours with the
current dense KKT route. They are guarded by `SDPX_GENERIC_ALLOW_LARGE=1` and
`pbs/run_large.pbs` requests 16 CPUs, 128 GiB, and 24 hours.

## Strict-feasibility and analytic checks

Generated cases are not allowed to fail merely because the generator omitted
an interior point.

- Box LP: `0 < x < u`, `s=u-x`; the positive profit proves `x=u` optimal.
- Infeasible/unbounded LP: exact original-coordinate Farkas/recession rays.
- Nearest SOCP: positive simplex point and sufficiently large `t` is strict;
  symmetry proves the uniform projection and its distance.
- Portfolio SOCP: uniform positive holding and a loose risk bound is strict;
  the budget makes the equal-return objective exactly one.
- Random/rank-one-face SDP: fixed positive diagonal admits the positive-definite
  diagonal matrix; `vv'` with `v_i=sqrt(d_i)` proves rank-one boundary points.
- Theta K4: `I/4` is positive definite and satisfies all affine rows.
- Entropy/EXP: uniform positive probabilities and epigraph values strictly above
  `p log p` give an interior point; Jensen proves `-log(n)`.
- Log-sum-exp: choose `t` above the analytic log-sum-exp and positive Exp slacks
  whose sum is below one.
- Power epigraph: fixed nonzero signals and `t_i>|x_i|^(1/alpha)` are strict;
  equality at optimum gives the analytic objective.
- Geometric mean: `|z| < left^alpha right^(1-alpha)` is strict and the boundary
  gives the analytic optimum.

## Honest solver findings (nonblocking, listed as `FINDING`)

A finding is accepted only when the model above has a strict feasible point and
a finite analytic optimum, while SDPX returns a fail-closed numerical status
without a valid certificate. It is not printed as PASS and its objective is not
promoted to a reference. These are benchmark discoveries, not generator fixes:

| case | seed/size | observed status | iterations | observed objective / note |
|---|---|---:|---:|---|
| `sdp_theta_k4` | n=4 | `insufficient_precision` | 17 | 0, no certificate |
| `sdp_random_small` | `0x5d0002`, n=4 | `insufficient_precision` | 17 | 0, no certificate |
| `sdp_rank1_boundary` | `0x5d0001`, n=2 | `insufficient_precision` | 17 | 0, no certificate |
| `sdp_random_medium` | `0x5d0003`, n=14 | `insufficient_precision` | 17 | 0, 0.040 s, no certificate |
| NETLIB AFIRO | published bounded LP | `numerical_failure` | 15 | 0, no certificate (native MPS lowering) |
| `exp_entropy_small` | n=3 | `numerical_failure` | 25 | `-1.098612280`, residuals `3.54e-9/2.69e-8`, certificate rejected |
| `exp_logsumexp_small` | `0x0e0002`, n=3 | `numerical_failure` | 49 | 0, no certificate |
| `exp_logsumexp_medium` | `0x0e0003`, n=12 | `numerical_breakdown` | 103 | 0, 0.047 s, no certificate |
| `power_geomean_small` | alpha=.35 | `numerical_breakdown` | 5 | 0, no certificate |
| `power_epigraph_medium` | `0x900002`, n=12 | `numerical_breakdown` | 66 | 0, 0.022 s, no certificate |

In contrast, generated Max-Cut K4, all five local LP cases, all three local SOCP
cases, and the small power epigraph return certificate-backed expected results:
11 certificate-backed cases and 6 explicit findings in the 17-case small tier.
On the validation host, a fresh Julia process took 74.6 s wall time including
compilation; cumulative measured solver calls were about 1.46 s and the slowest
individual solve was 1.38 s. This is why the seconds gate is per solve.

## Running

```bash
# Local default (small only, large guard active)
julia --project=. benchmark/general/GenericConicBenchmark.jl

# One family
julia --project=. benchmark/general/GenericConicBenchmark.jl small socp

# Medium report lane
julia --project=. benchmark/general/GenericConicBenchmark.jl medium

# Tests (including PASS/FINDING counts and reader/reference contracts)
julia --project=. benchmark/general/test_small.jl

# Cluster
qsub benchmark/general/pbs/run_large.pbs
```

The runner reports status, original-coordinate certificate validity, primal and
dual residuals, relative gap, objective, iterations, solve time, GC time, and
allocation. The 30-second small threshold applies per solver call and excludes
one-time Julia compilation; the suite also reports total wall time.
