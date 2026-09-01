# V2 provider-backed precision qualification

Generated 2026-09-02 by three fresh per-tier processes (combined in-process runs
triggered the known Julia 1.12 + MultiFloats parallel-GC corruption even with
`--gcthreads=1`; per-tier process isolation is the documented safe mode).
The qualification runner now enumerates the 16 certified optimal-path cases (including the mixed planted case); the two ray cases are skipped. The matrix below predates the mixed row and must be regenerated before release.

Provider status: `MultiFloats` + `MultiFloatLinearAlgebra` and
`BigFloatLinearAlgebra` loaded per tier. BigFloat runs are inside
`setprecision(BigFloat, bits)`. All rows qualify ONLY through the existing V2
certificate/oracle gates with the STRICT reviewed solver tolerances
(Float64x2 1e-15, Float64x4 1e-28, BigFloat256 1e-32) and objective allowances
(5e-13 / 5e-22 / 5e-28). No solver tolerance or formulation is changed.

The prior 15-case matrix is retained as historical evidence only; a release qualification must regenerate all 16 optimal-path rows from the live catalog.
Key findings:
- MultiFloats x2/x4 tiers: most cases return `numerical_breakdown` at iteration 0
  under the reviewed tolerances - a genuine provider/solver limitation at these
  settings (Phase-4 mixed-precision ladder work item), not a benchmark defect.
- `log(::MultiFloat)` is not implemented upstream (EXP rows fail closed).
- BigFloat256 with the BFLA provider is the strongest high-precision path today
  (all SDP/RSOC/SOCP/LP planted cases converge within 5e-28 except the two
  ill-conditioned rows and EXP numerical_breakdown).
- Tolerance parsing fix: `MultiFloats` lacks `tryparse`; decimal strings are now
  constructed via the MultiFloat string constructor (`T(text)`), `Float64` via
  `parse`, `BigFloat` via constructor.
## Float64x2 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `5.0` | `0` | `8.332602375` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_lp_sparse_planted_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `4.0` | `0` | `0.00023004200000000002` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_lp_nonpositive_small` | `Float64x2` | `optimal` | true | `-2.0` | `0.0` | `0` | `2.357348458` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `-1.0` | `0` | `0.006466958` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_ill_diag_ladder_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `5.0` | `0` | `0.00019283400000000002` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_ill_near_rank_loss_small` | `Float64x2` | `optimal` | true | `-1.99999999999999988897771141527038` | `1.110222885847296200000000000000000000000000000000000000000000058476497021426618e-16` | `0` | `0.000187834` | **qualified** | status/certificate/oracle gate |
| `v2_soc_one_large_small` | `Float64x2` | `optimal` | true | `2.0` | `0.0` | `0` | `0.021929209` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `-16.0` | `0` | `0.001291209` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_rsoc_quadratic_epigraph_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `-1.5` | `0` | `0.00022616700000000002` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_rsoc_perspective_ls_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `-0.5` | `0` | `0.00021637500000000001` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_rsoc_many_qr3_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `-24.0` | `0` | `0.000996125` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_sdp_weighted_trace_small` | `Float64x2` | `optimal` | true | `1.0` | `0.0` | `0` | `0.0066767910000000005` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `Float64x2` | `numerical_breakdown` | false | `0.0` | `4.0` | `0` | `0.000301542` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_sdp_multiblock_small` | `Float64x2` | `optimal` | true | `8.0000000000000035527132347113471989067` | `3.5527132347113471989066999999999999999999999999999999999999999474271188150322e-15` | `0` | `0.001876292` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `Float64x2` | `exception` | false | `n/a` | `n/a` | `n/a` | `n/a` | **not-qualified** | ErrorException: log(::MultiFloat) is not yet implemented. For a workaround,
call MultiFloats.use_bigfloat_transcendentals() after importing
MultiFloats. This will use the BigFloat implementation of log,
which will not be as fast as a pure-MultiFloat implementation.
 |

## Float64x4 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `5.0` | `0` | `33.10475` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_lp_sparse_planted_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `4.0` | `0` | `0.00030875` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_lp_nonpositive_small` | `Float64x4` | `optimal` | true | `-2.0` | `0.0` | `0` | `6.880555209000001` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `-1.0` | `0` | `0.019774292000000002` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_ill_diag_ladder_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `5.0` | `0` | `0.000282084` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_ill_near_rank_loss_small` | `Float64x4` | `optimal` | true | `-1.99999999999999999999999999999500000062499992187501143095789990040043425046616075` | `4.999999375000078124988569042100099599565749533839250000000000000000000000000019e-30` | `0` | `0.000262208` | **not-qualified** | oracle |
| `v2_soc_one_large_small` | `Float64x4` | `optimal` | true | `2.0` | `0.0` | `0` | `0.122364834` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `-16.0` | `0` | `0.006257250000000001` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_rsoc_quadratic_epigraph_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `-1.5` | `0` | `0.00045120800000000003` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_rsoc_perspective_ls_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `-0.5` | `0` | `0.000437167` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_rsoc_many_qr3_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `-24.0` | `0` | `0.0050755` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_sdp_weighted_trace_small` | `Float64x4` | `optimal` | true | `1.0000000000000000000000000000000000000000000000000000000000000000000000035608374595552517272380067878629017708117871808115` | `3.56083745955525172723800678786290177081178718081150000022619984858901642130687e-72` | `0` | `0.014062708` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `Float64x4` | `numerical_breakdown` | false | `0.0` | `4.0` | `0` | `0.001013292` | **not-qualified** | objective_interval,oracle,certificate,status |
| `v2_sdp_multiblock_small` | `Float64x4` | `optimal` | true | `8.00000000000000000000000000015999998000000249999963420934720316483911531402882` | `1.599999800000024999996342093472031648391153140288199999999999999999999999999993e-28` | `0` | `0.009792333` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `Float64x4` | `exception` | false | `n/a` | `n/a` | `n/a` | `n/a` | **not-qualified** | ErrorException: log(::MultiFloat) is not yet implemented. For a workaround,
call MultiFloats.use_bigfloat_transcendentals() after importing
MultiFloats. This will use the BigFloat implementation of log,
which will not be as fast as a pure-MultiFloat implementation.
 |

## BigFloat256 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `BigFloat256` | `optimal` | true | `-4.999999999999999999999999999999989509378492577511102906652244244790516095174084` | `1.049062150742248889709334775575520948390482591599999999999999999999999999999999e-32` | `34` | `32.51055575` | **qualified** | status/certificate/oracle gate |
| `v2_lp_sparse_planted_small` | `BigFloat256` | `optimal` | true | `-3.999999999999999999999999999999997419161944207210751416401498169394207313601762` | `2.580838055792789248583598501830605792686398237999999999999999999999999999999998e-33` | `34` | `0.008037625000000001` | **qualified** | status/certificate/oracle gate |
| `v2_lp_nonpositive_small` | `BigFloat256` | `optimal` | true | `-2.0` | `0.0` | `51` | `1.9812092090000002` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `BigFloat256` | `optimal` | true | `1.000000000000000000000000000000084379548154000455742739870384769037145447259115` | `8.437954815400045574273987038476903714544725911499999999999999999999999999999998e-32` | `35` | `3.3948357920000003` | **qualified** | status/certificate/oracle gate |
| `v2_ill_diag_ladder_small` | `BigFloat256` | `optimal` | true | `-4.999999999999999999999999999999989509378492577511102906652244244790516095173877` | `1.049062150742248889709334775575520948390482612299999999999999999999999999999998e-32` | `34` | `0.006931416` | **not-qualified** | oracle |
| `v2_ill_near_rank_loss_small` | `BigFloat256` | `optimal` | true | `-1.999999999999999999999999999999997862992802744340439072638259736991532520158354` | `2.137007197255659560927361740263008467479841645999999999999999999999999999999992e-33` | `33` | `0.005069959000000001` | **not-qualified** | oracle |
| `v2_soc_one_large_small` | `BigFloat256` | `optimal` | true | `2.0` | `0.0` | `35` | `17.479109792000003` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `BigFloat256` | `optimal` | true | `16.0` | `0.0` | `17` | `0.254372916` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_quadratic_epigraph_small` | `BigFloat256` | `optimal` | true | `1.500000000000000000000000000000002866348827448569542427407256452883219796156846` | `2.866348827448569542427407256452883219796156845999999999999999999999999999999994e-33` | `36` | `0.01760375` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_perspective_ls_small` | `BigFloat256` | `optimal` | true | `0.500000000000000000000000000000002866348827448569542427407256452883219796156846` | `2.866348827448569542427407256452883219796156845999999999999999999999999999999994e-33` | `36` | `0.018055792` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_many_qr3_small` | `BigFloat256` | `optimal` | true | `23.99999999999999999999999999999999359327577703412941216793285088030958802956624` | `-6.40672422296587058783206714911969041197043375999999999999999999999999999999995e-33` | `34` | `0.46662945800000005` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_weighted_trace_small` | `BigFloat256` | `optimal` | true | `1.0` | `0.0` | `34` | `0.078007083` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `BigFloat256` | `optimal` | true | `-3.999999999999999999999999999999999999999999939661317465760391435695836087313574` | `6.033868253423960856430416391268642600000000000000000000000000000000000000000014e-44` | `42` | `0.23334183400000003` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_multiblock_small` | `BigFloat256` | `optimal` | true | `8.000000000000000000000000000000068384230312181105949675575688416270959354933493` | `6.838423031218110594967557568841627095935493349299999999999999999999999999999987e-32` | `33` | `0.7085042500000001` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `BigFloat256` | `numerical_breakdown` | false | `0.0` | `n/a` | `34` | `0.22379695800000002` | **not-qualified** | objective_interval,oracle,certificate,status |


## Contract

A row is **qualified** only when the requested arithmetic provider is loaded, status is `:optimal`, the public certificate is valid, the original-coordinate certificate gate passes, the independent oracle passes at `max(256,2*bits)`, and the objective is within the reviewed allowance. Failures record their honest terminal status and failure reasons; no row is qualified by default.
