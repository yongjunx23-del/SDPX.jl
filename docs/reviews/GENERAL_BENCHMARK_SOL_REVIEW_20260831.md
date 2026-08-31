# Addendum: concrete scalable corpus blueprint

This addendum extends the prior review. All prior P0/P1 findings and the **BLOCK** verdict remain in force, especially the BigFloat precision bug, false exception accounting, infeasible mixed tiers, fabricated metrics, and overfitted `known_solver_finding` policy.

## Review

- **Correct:** SDPX publicly supports free/equality, nonnegative, nonpositive, Lorentz, rotated Lorentz, PSD, exponential, and power domains (`src/SDPX.jl:145-146`).
- **Finding: P1 — missing cone coverage.** Neither `benchmark/general_benchmark.jl:40-42` nor `benchmark/general/` has a rotated-Lorentz family, despite `RotatedLorentzCone` being public. Nonpositive orientation also lacks a dedicated sign/reconstruction case.
- **Finding: P1 — incompatible tier taxonomies.** The monolith uses `instant/medium/heavy/extreme` (`benchmark/general_benchmark.jl:40-42`); the maintained suite documents only `small/medium/large` (`docs/design/GENERAL_BENCHMARK.md:59-73`). Adopt exactly `small/medium/large/extreme`.
- **Finding: P0 — declared precision support is not real.** The seven names already appear at `benchmark/general_benchmark.jl:42`, but prior-review evidence shows BigFloat bits are not propagated and metrics narrow to Float64.
- **Merge verdict:** **BLOCK** until the P0 fixes and catalog migration below land.

## 1. Mandatory problem inventory

Free variables and `ZeroCone` equalities are exercised inside every family. Add explicit nonpositive and rotated-SOC reconstruction sentinels.

| Family | Required kinds | Small | Medium | Large | Extreme |
|---|---|---:|---:|---:|---:|
| LP/orthant | box, sparse planted KKT, duplicate/rank-deficient, primal infeasible, unbounded, Chebyshev, explicit Nonpositive sign case | `n=32`, sparse `m=16` | `n=512,m=256` | `n=8192,m=4096` | `n=131072,m=32768` |
| SOCP | simplex projection, planted portfolio, one large SOC, many Q3 load-sharing blocks, ill-scaled SOC | Q order 33; 16 Q3 | Q order 257; 256 Q3 | Q order 4097; 4096 Q3 | Q order 32769; 65536 Q3 |
| Rotated SOCP | quadratic epigraph, perspective least squares, many QR3 blocks | 16 QR3 | 256 QR3 | 4096 QR3 | 65536 QR3 |
| SDP | weighted trace, complete-graph theta/Max-Cut, sparse graph SDP, multi-block PSD, linear elliptope | orders 4 and 8; `8×PSD(3)` | order 32; `64×PSD(5)` | order 128; `512×PSD(5)` | order 512; `4096×PSD(5)` |
| EXP | unit epigraph, entropy, log-sum-exp, exponential fitting, corrected coercive GP | 8 blocks | 128 | 2048 | 32768 |
| Power | separable p-power, fixed weighted mean, budgeted mean, alpha sweep | 8 blocks | 128 | 2048 | 32768 |
| Mixed | planted direct product; planted active cross-cone coupling; sparse many-block mixed | `(NN=32,Q=4,QR=4,PSD=4,EXP=8,POW=8)` | `(512,32,32,24,128,128)` | `(8192,512,512,96,2048,2048)` | `(65536,4096,4096,384,32768,32768)` |
| Ill-conditioned | Hilbert SDP, diagonal scaling ladder, near-rank-loss LP, near-boundary SOC/PSD, high-range EXP/Power | Hilbert 6; scale `1e6` | Hilbert 12; `1e12` | Hilbert 20; `1e24` | Hilbert 32; `1e48` |

Power alphas must include exact source values `1/2`, `1/3`, `2/3`, `2/5`, and `7/10`. “p-norm” must be renamed unless a true norm epigraph is implemented. “Nearest correlation” must be renamed or given its missing quadratic term, as identified in the prior review.

Every generated optimal case must return an oracle object containing:

- frozen coefficient fingerprint;
- strict primal point where applicable;
- dual point or analytic lower bound;
- analytic objective;
- expected status and certificate kind.

The mixed generator must be replaced rather than rescaled: plant a complementary primal-dual pair and derive the coupling RHS from the planted strict point.

## 2. Precision matrix

Definitions:

- **Full:** all generated kinds assigned to that tier.
- **Sentinel:** at least one analytic case for every public cone signature, including Nonpositive and rotated SOC.
- **Holdout:** public external rows in addition to generated cases.

| Arithmetic | Small | Medium | Large, UCAS | Extreme, UCAS |
|---|---|---|---|---|
| Float64 | Full | Full + holdout | Full + holdout | Full selected kinds |
| Float64x2 | Full | Full | Full | Sentinels |
| Float64x3 | Full | Sentinels | Sentinels | Sentinels |
| Float64x4 | Full | Full + holdout | Full + holdout | Full selected kinds |
| BigFloat256 | Full | Full + holdout | Full selected kinds | Full selected kinds |
| BigFloat512 | Full | Sentinels | Sentinels + ill-conditioned | Sentinels + ill-conditioned |
| BigFloat1024 | Full | Sentinels | Sentinels + ill-conditioned | Sentinels + ill-conditioned |

“Sentinel” still runs every cone at that tier’s dimensions; it reduces kind multiplicity, not size. This avoids a seven-precision Cartesian product over every structurally similar case.

Required implementation:

```text
PrecisionSpec(name, T, bits, solver_tolerance, certificate_limit, provider)
```

BigFloat construction and solve must occur inside `setprecision(BigFloat, bits) do`; every model must receive `precision_bits=bits`. Results remain strings or arithmetic-typed values until comparison.

## 3. Analytic train versus public holdout

### Train

Use roughly 70% of generated instances and only a few public calibration cases:

- LP: box/KKT/rays plus NETLIB AFIRO;
- SOCP/RSOC: analytic generated families;
- SDP: complete graphs, weighted trace, SDPLIB control1;
- EXP: unit, entropy, log-sum-exp;
- Power: exact epigraph/geometric-mean formulas;
- mixed: planted KKT constructions;
- ill-conditioned: disclosed scale exponents.

Seeds must be frozen from `SHA256("sdpx-general-v1/<family>/train/<index>")`.

### Holdout

Do not run during inner optimization loops:

- **NETLIB:** ADLITTLE, SC50A, SHARE2B, RECIPE.
- **SDPLIB:** control5, theta1/theta3, maxG11/maxG32, qap5, hinf2, nqlp2, truss1, mcp100; mcp250-1 for extreme.
- **SOCP/RSOC:** checksum-pinned continuous CBLIB cases selected as one large SOC, many small SOCs, and LP/Q/QR mixed structure.
- **EXP/Power/mixed:** continuous CBF or Hypatia public instances with archived independent reference intervals.
- **Large sparse SDP:** one Mittelmann/Gondzio case only after format, license, checksum, and workspace review.

The current expanded inventory is not sufficient: `benchmark/general/src/external.jl:56-88` declares files that are absent or unpinned in `benchmark/general/data/MANIFEST.sha256:1-5`.

A public case enters the solve corpus only with:

1. immutable URL and license/provenance;
2. SHA-256;
3. parsed and canonical dimension/cone fingerprint;
4. official status and objective interval;
5. successful parity against an independent solver/checker.

## 4. Certificate and accuracy gates

All gates are original-coordinate gates for **every** cone, including EXP and Power.

| Arithmetic | Requested solver tolerance | Maximum scaled residual/gap/complementarity |
|---|---:|---:|
| Float64 | `1e-8` | `5e-7` |
| Float64x2 | `1e-15` | `5e-13` |
| Float64x3 | `1e-21` | `5e-18` |
| Float64x4 | `1e-28` | `5e-22` |
| BigFloat256 | `1e-32` | `5e-28` |
| BigFloat512 | `1e-50` | `5e-46` |
| BigFloat1024 | `1e-80` | `5e-74` |

An optimal case passes only with:

- exact expected status;
- valid certificate and matching kind;
- empty certificate failure list;
- finite primal/dual objectives;
- primal affine/cone, dual affine/cone, relative gap, and relative complementarity below the table limit;
- primal and dual objectives inside the analytic/public reference allowance;
- independent reevaluation at `max(256, 2*bits)` precision.

For published objectives with limited digits, use the published interval rather than pretending it is exact. The objective allowance is the larger of the public interval half-width and arithmetic tolerance.

Infeasible and unbounded cases retain the normalized ray gates from the prior review: valid matching certificate, cone membership, homogeneous equation residual below the arithmetic limit, and separating/improving inner product at least `100×` that limit.

Current solver failures remain `XFAIL`, not pass. A newly certified optimum becomes `PASS/RESOLVED`.

## 5. Runtime and memory lanes

| Tier | Authoritative lane | Per-case wall budget | Memory budget |
|---|---|---:|---:|
| Small | local default | 20 s solve; 180 s whole default matrix | 4 GiB |
| Medium | local opt-in/nightly | 10 min | 16 GiB |
| Large | local preflight; UCAS PBS authoritative | 2 h local selected; 6 h PBS | 64 GiB local; 128 GiB PBS |
| Extreme | UCAS PBS only | 24 h; 48 h for BigFloat1024 | 256 GiB; 512 GiB for approved shards |

Required memory fields:

- input and canonical matrix bytes/nnz;
- build, canonicalization, presolve, setup, core, and certification allocations/times;
- predicted and actual workspace;
- factorization bytes and factor nonzeros;
- fresh-process peak RSS;
- GC bytes/time;
- declared memory limit and preflight decision;
- BigFloat repeated-batch RSS drift;
- terminal `memory_limit`/preflight status rather than an omitted exception.

Delete guessed GFLOPS and hard-coded zero setup/workspace fields from `benchmark/general_benchmark.jl:738-790`.

Performance release gates remain:

- five fresh samples, three nightly;
- every sample semantically certified;
- MAD/median ≤5%;
- runtime ratio ≤1.10;
- allocation/workspace ratio ≤1.05 plus 1 MiB;
- RSS ratio ≤1.10 plus 32 MiB;
- otherwise inconclusive, not pass.

## 6. Sequential optimization workflow

### Local: small → medium → large

1. **Small Float64, ≤3 minutes:** all families and cone-sign sentinels.
2. **Small precision qualification:** all seven arithmetic types.
3. **Medium train:** run Float64 → Float64x2 → Float64x4 → BigFloat256, then Float64x3/BigFloat512/1024 sentinels.
4. **Freeze candidate:** no further tuning from holdout output.
5. **Medium holdout:** one fresh correctness sample; three only for designated performance rows.
6. **Local large:** build, canonical fingerprint, workspace preflight, and selected Float64/Float64x4 solve only when estimate is below 64 GiB and two hours.
7. Produce immutable PBS shard manifests from the same catalog fingerprints.

### UCAS PBS: large → extreme

Use one problem/arithmetic pair per PBS array element, BLAS fixed to one thread.

- **Large template:** `select=1:ncpus=16:mem=128gb`, `walltime=06:00:00`.
- **Extreme template:** `select=1:ncpus=32:mem=256gb`, `walltime=24:00:00`.
- **BigFloat1024 exception:** approved `mem=512gb`, `walltime=48:00:00`.

Run baseline/candidate in alternating same-node order where scheduling permits. Persist failure rows before nonzero exit; the current sequential `benchmark/general/pbs/run_large.pbs:13-14` is insufficient because one failure aborts the entire family batch.

### CSDR/physics placement

Until the parent provides the authoritative catalog, the standalone F3L builder can define build/report tiers:

- small `J=10,N_mu=32`: 192 Lorentz blocks;
- medium `J=20,N_mu=64`: 704 blocks;
- large `J=40,N_mu=120`: 2520 blocks;
- extreme `J=80,N_mu=240`: 9840 blocks.

No objective gate should be claimed without an independent CSDR interval. Existing physics catalogs remain build-only under `benchmark/bootstrap/README.md:33-43`; include their scaling/stress builds in nightly/weekly memory matrices, not solver-optimality counts.

## 7. Migration from `benchmark/general_benchmark.jl`

1. **Freeze the monolith:** mark it legacy and accept no new generators.
2. **Create one shared benchmark-core package/module:** extract fingerprint, result schema, fresh-process, and comparison machinery from `benchmark/bootstrap/`.
3. **Introduce `GeneralBenchmarkSpec`:** source, split, tier, precision spec, cone signature, dimensions, resource budget, oracle, checksum, and fingerprint.
4. **Port corrected generators by family:** LP, SOCP/RSOC, SDP, EXP, Power, mixed, ill-conditioned. Do not port the existing mixed, nearest-correlation, p-norm, or probabilistic-status contracts unchanged.
5. **Add exact coefficient artifacts:** one artifact feeds every arithmetic build and its oracle.
6. **Integrate external instances into normal inventory:** parsing alone is not a benchmark solve.
7. **Add `:extreme` to `benchmark/general/`** and resource/preflight guards.
8. **Provide a compatibility CLI:** map `instant→small`, `medium→medium`, `heavy→large`, `extreme→extreme`; warn that the old result schema is retired.
9. **Cut over tests/docs/PBS:** inventory, oracle, precision metadata, checksum, exception accounting, certificate, holdout isolation, and PBS shard tests.
10. **Remove the monolith after two releases** once compatibility output proves catalog parity.