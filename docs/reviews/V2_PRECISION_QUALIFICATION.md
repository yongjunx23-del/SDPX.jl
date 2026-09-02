# V2 provider-backed precision qualification

Executable source commit: `b17cc69` (`fix/general-benchmark-v2-review`).
Generated 2026-09-02 with Julia 1.12, `--gcthreads=1`, `JULIA_NUM_THREADS=4`,
MultiFloats 3.3.1, MultiFloatLinearAlgebra 0.4.0, and
BigFloatLinearAlgebra 0.3.0. All 17 live certified optimal-path cases run; the
two Farkas/ray status cases are intentionally outside this optimal-path report.

Qualification is fail-closed through status, public certificate, reconstructed
original-coordinate variable AND constraint-row cone metrics (including row
complementarity), an independent analytic oracle at `max(256,2*bits)`, and the
central precision-specific objective allowance. No solver tolerance,
formulation, or solver source is changed by this report.

## Summary

| Precision | Process isolation | Qualified | Honest residual finding |
|---|---|---:|---|
| Float64x2 | one fresh tier process | **17/17** | none |
| Float64x4 | one fresh process **per case**, max three attempts | **16/17** | `v2_lp_chebyshev_small` exited 139 in 3/3 fresh processes before any solver row/status was emitted |
| BigFloat256 | one fresh tier process under `setprecision(BigFloat,256)` | **16/17** | `v2_exp_unit_epigraph_small` terminates `numerical_breakdown` at iteration 34 with invalid certificate |

Float64x4 provider/runtime stability remains an explicit blocker. On this exact
source, Chebyshev crashed after 892,390,043 / 891,182,664 / 891,017,930
allocations (GC counts 88 / 89 / 84). Perspective-RSOC and weighted-trace SDP
each crashed once before a fresh retry certified. A process crash is never
converted into a solver status or silently qualified; finite retries stop at
three. Per-case isolation prevents one crash from erasing unrelated receipts.

These receipts supersede every earlier 15/16/17-case matrix. The prior x4
17/17 result remains historical evidence of intermittency, not a current
qualification: the stricter round-3 source must use the results above.

## Float64x2 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `Float64x2` | `optimal` | true | `-4.999999999999999503608309318307` | `4.963916906816929999999999999999999999999999999999999999999999841892918361795326e-16` | `17` | `24.581260500000003` | **qualified** | status/certificate/oracle gate |
| `v2_lp_sparse_planted_small` | `Float64x2` | `optimal` | true | `-3.99999999999999987788030486853662` | `1.221196951314633800000000000000000000000000000000000000000000126242063105505657e-16` | `17` | `0.000655417` | **qualified** | status/certificate/oracle gate |
| `v2_lp_nonpositive_small` | `Float64x2` | `optimal` | true | `-2.0` | `0.0` | `24` | `1.896464208` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `Float64x2` | `optimal` | true | `1.0000000000000039925969389862234205` | `3.992596938986223420499999999999999999999999999999999999999999991524335633810499e-15` | `18` | `0.176540375` | **qualified** | status/certificate/oracle gate |
| `v2_ill_diag_ladder_small` | `Float64x2` | `optimal` | true | `-4.99999999999999950360830931830717` | `4.963916906816928299999999999999999999999999999999999999999999941977120515936364e-16` | `17` | `0.000462375` | **qualified** | status/certificate/oracle gate |
| `v2_ill_near_rank_loss_small` | `Float64x2` | `optimal` | true | `-1.9999999999999990323604031830587` | `9.676395968169412999999999999999999999999999999999999999999999968246461916011881e-16` | `15` | `0.000333458` | **qualified** | status/certificate/oracle gate |
| `v2_soc_one_large_small` | `Float64x2` | `optimal` | true | `2.0` | `0.0` | `17` | `0.354346208` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `Float64x2` | `optimal` | true | `16.0000000000000000000000000000000000000020203809154758067` | `2.020380915475806700000000000000000000045265940947713031276741716159640893126369e-39` | `9` | `0.009472334` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_quadratic_epigraph_small` | `Float64x2` | `optimal` | true | `1.500000000000000000000000008838372930192928` | `8.838372930192927999999999999999999999999999999999997116229197827562904246191422e-27` | `16` | `0.0005551250000000001` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_perspective_ls_small` | `Float64x2` | `optimal` | true | `0.500000000000000000000000008838372930192928` | `8.838372930192927999999999999999999999999999999999997116229197827562904246191422e-27` | `16` | `0.0005295` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_many_qr3_small` | `Float64x2` | `optimal` | true | `24.000000000000000000000000015651394790037485` | `1.565139479003748499999999999999999999999999999999990982635266281754754060449361e-26` | `15` | `0.010580292` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_weighted_trace_small` | `Float64x2` | `optimal` | true | `1.00000000000000000000000000000000000000000000007459932733670201` | `7.459932733670200999999999999999689867417189746527847112605005192227837356260245e-47` | `16` | `0.007373959` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `Float64x2` | `optimal` | true | `-3.9999999999999999999999999941759385443664225` | `5.824061455633577500000000000000000000000000000000004874113981523411268046768993e-27` | `20` | `0.004250334000000001` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_multiblock_small` | `Float64x2` | `optimal` | true | `8.0000000000000032357864495500059` | `3.235786449550005900000000000000000000000000000000000000000000024663913581217719e-15` | `16` | `0.018212792000000002` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `Float64x2` | `optimal` | true | `1.0` | `n/a` | `21` | `0.023295583` | **qualified** | status/certificate/oracle gate |
| `v2_power_interior_epigraph_small` | `Float64x2` | `optimal` | true | `0.249999999999999508946782625814278` | `-4.910532173741857219999999999999999999999999999999999999999999990523949515953268e-16` | `52` | `0.008434958000000001` | **qualified** | status/certificate/oracle gate |
| `v2_mixed_planted_cross_cone_small` | `Float64x2` | `optimal` | true | `7.99999999999999999999999999999992046013630265932` | `-7.953986369734068000000000000000000000000000003028028444624961995212996418802165e-32` | `17` | `0.003423333` | **qualified** | status/certificate/oracle gate |

## Float64x4 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `Float64x4` | `optimal` | true | `-4.99999999999999999999999999991202969274093344266811082730338130522396450932006` | `8.797030725906655733188917269661869477603549067994000000000000000000000000000026e-29` | `30` | `113.42299308300001` | **qualified** | status/certificate/oracle gate |
| `v2_lp_sparse_planted_small` | `Float64x4` | `optimal` | true | `-3.99999999999999999999999999997835806750361425130610170594350438073742930859205` | `2.164193249638574869389829405649561926257069140794999999999999999999999999999993e-29` | `30` | `113.98215204200001` | **qualified** | status/certificate/oracle gate |
| `v2_lp_nonpositive_small` | `Float64x4` | `optimal` | true | `-2.0` | `0.0` | `45` | `113.624392916` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `Float64x4` | `process_crash` | false | `n/a` | `n/a` | `n/a` | `n/a` | **not-qualified** | SIGSEGV exit 139 in 3/3 fresh processes before row emission; allocations 892390043 / 891182664 / 891017930 |
| `v2_ill_diag_ladder_small` | `Float64x4` | `optimal` | true | `-4.99999999999999999999999999991202969274093344266811082730338130397054956687977` | `8.797030725906655733188917269661869602945043312023000000000000000000000000000034e-29` | `30` | `113.351832583` | **qualified** | status/certificate/oracle gate |
| `v2_ill_near_rank_loss_small` | `Float64x4` | `optimal` | true | `-1.999999999999999999999999999982079875710275827105757496112294138662593113889765` | `1.79201242897241728942425038877058613374068861102349999999999999999999999999999e-29` | `29` | `113.71652620900001` | **qualified** | status/certificate/oracle gate |
| `v2_soc_one_large_small` | `Float64x4` | `optimal` | true | `2.0` | `0.0` | `30` | `116.86747037500001` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `Float64x4` | `optimal` | true | `16.0` | `0.0` | `15` | `113.59518308300001` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_quadratic_epigraph_small` | `Float64x4` | `optimal` | true | `1.500000000000000000000000000024036470739680064410661340720411633683243705478365` | `2.403647073968006441066134072041163368324370547836499999999999999999999999999991e-29` | `32` | `113.93592116600001` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_perspective_ls_small` | `Float64x4` | `optimal` | true | `0.500000000000000000000000000024036470739680064410661340720411633683243705478365` | `2.403647073968006441066134072041163368324370547836499999999999999999999999999991e-29` | `32` | `113.851824792` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_many_qr3_small` | `Float64x4` | `optimal` | true | `24.000000000000000000000000000207836584990478129538276269685272212096403285072587` | `2.078365849904781295382762696852722120964032850725870000000000000000000000000011e-28` | `30` | `113.86893145900001` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_weighted_trace_small` | `Float64x4` | `optimal` | true | `0.9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999993710658859023460710049820701097706165710832914076` | `-6.289341140976539289950179298902293847098203332877174721727841500290329223357213e-91` | `30` | `114.05199054200001` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `Float64x4` | `optimal` | true | `-3.999999999999999999999999999999999999999646114007155330472307234800212004109596889425013407014231819746832306094772243668684025487462055692710889583986841202399002615758380647925651629748244886286556720733642578125` | `3.538859928446695276927651997879958904031105749865929857681802531676939052277563e-40` | `35` | `113.426183291` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_multiblock_small` | `Float64x4` | `optimal` | true | `8.0000000000000000000000000005734439772711735326157601244065875582458674044093` | `5.734439772711735326157601244065875582458674044092999999999999999999999999999992e-28` | `29` | `113.601465875` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `Float64x4` | `optimal` | true | `0.9999999999999999999999999999999999999999999999999999999999999999620177290169608050101070309217521713831161366655202201348808800366839670774207553638675242729143455996278894680473481315630124299786984920501708984375` | `n/a` | `59` | `114.22455625` | **qualified** | status/certificate/oracle gate |
| `v2_power_interior_epigraph_small` | `Float64x4` | `optimal` | true | `0.249999999999999999999999999957378846238172243331700041829118705894423945102793` | `-4.262115376182775666829995817088129410557605489720700000000000000000000000000001e-29` | `85` | `113.619106333` | **qualified** | status/certificate/oracle gate |
| `v2_mixed_planted_cross_cone_small` | `Float64x4` | `optimal` | true | `8.000000000000000000000000000000000000000000000000000000000000018195392656755189391331561974145255980725173558348` | `1.819539265675518939133156197414525598072517355834799999999999998251900768933423e-62` | `30` | `114.322381542` | **qualified** | status/certificate/oracle gate |

## BigFloat256 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `BigFloat256` | `optimal` | true | `-4.999999999999999999999999999999989509378492577511102906652244244790516095174084` | `1.049062150742248889709334775575520948390482591599999999999999999999999999999999e-32` | `34` | `32.733260167000005` | **qualified** | status/certificate/oracle gate |
| `v2_lp_sparse_planted_small` | `BigFloat256` | `optimal` | true | `-3.999999999999999999999999999999997419161944207210751416401498169394207313601762` | `2.580838055792789248583598501830605792686398237999999999999999999999999999999998e-33` | `34` | `0.007960458` | **qualified** | status/certificate/oracle gate |
| `v2_lp_nonpositive_small` | `BigFloat256` | `optimal` | true | `-2.0` | `0.0` | `51` | `2.000207916` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `BigFloat256` | `optimal` | true | `1.000000000000000000000000000000084379548154000455742739870384769037145447259115` | `8.437954815400045574273987038476903714544725911499999999999999999999999999999998e-32` | `35` | `3.422835833` | **qualified** | status/certificate/oracle gate |
| `v2_ill_diag_ladder_small` | `BigFloat256` | `optimal` | true | `-4.999999999999999999999999999999989509378492577511102906652244244790516095173877` | `1.049062150742248889709334775575520948390482612299999999999999999999999999999998e-32` | `34` | `0.0068870830000000004` | **qualified** | status/certificate/oracle gate |
| `v2_ill_near_rank_loss_small` | `BigFloat256` | `optimal` | true | `-1.999999999999999999999999999999997862992802744340439072638259736991532520158354` | `2.137007197255659560927361740263008467479841645999999999999999999999999999999992e-33` | `33` | `0.0052427080000000004` | **qualified** | status/certificate/oracle gate |
| `v2_soc_one_large_small` | `BigFloat256` | `optimal` | true | `2.0` | `0.0` | `35` | `17.133746792` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `BigFloat256` | `optimal` | true | `16.0` | `0.0` | `17` | `0.252260417` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_quadratic_epigraph_small` | `BigFloat256` | `optimal` | true | `1.500000000000000000000000000000002866348827448569542427407256452883219796156846` | `2.866348827448569542427407256452883219796156845999999999999999999999999999999994e-33` | `36` | `0.018219708` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_perspective_ls_small` | `BigFloat256` | `optimal` | true | `0.500000000000000000000000000000002866348827448569542427407256452883219796156846` | `2.866348827448569542427407256452883219796156845999999999999999999999999999999994e-33` | `36` | `0.035059041` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_many_qr3_small` | `BigFloat256` | `optimal` | true | `23.99999999999999999999999999999999359327577703412941216793285088030958802956624` | `-6.40672422296587058783206714911969041197043375999999999999999999999999999999995e-33` | `34` | `0.45995679100000003` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_weighted_trace_small` | `BigFloat256` | `optimal` | true | `1.0` | `0.0` | `34` | `0.06360133400000001` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `BigFloat256` | `optimal` | true | `-3.999999999999999999999999999999999999999999939661317465760391435695836087313574` | `6.033868253423960856430416391268642600000000000000000000000000000000000000000014e-44` | `42` | `0.21282762500000002` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_multiblock_small` | `BigFloat256` | `optimal` | true | `8.000000000000000000000000000000068384230312181105949675575688416270959354933493` | `6.838423031218110594967557568841627095935493349299999999999999999999999999999987e-32` | `33` | `0.710869458` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `BigFloat256` | `numerical_breakdown` | false | `0.0` | `n/a` | `34` | `0.34327216600000005` | **not-qualified** | objective_interval,certificate,status |
| `v2_power_interior_epigraph_small` | `BigFloat256` | `optimal` | true | `0.2499999999999999999999999999999901769872649288084573830818723938549297080935641` | `-9.823012735071191542616918127606145070291906435899999999999999999999999999999952e-33` | `99` | `0.6246489590000001` | **qualified** | status/certificate/oracle gate |
| `v2_mixed_planted_cross_cone_small` | `BigFloat256` | `optimal` | true | `8.0` | `0.0` | `34` | `0.193534917` | **qualified** | status/certificate/oracle gate |

## Contract

A row is **qualified** only when the requested provider is loaded, status is
`:optimal`, the public certificate is valid, every required original-coordinate
variable/row metric is finite and within its gate, the independent source/oracle
proof passes, and the objective is within `max(public_interval_half_width,
precision.certificate_limit)`. Reviewed solver tolerances / certificate limits
are Float64x2 `1e-15 / 5e-13`, Float64x4 `1e-28 / 5e-22`, and BigFloat256
`1e-32 / 5e-28`. Process crashes are orchestration/provider findings and never
solver statuses.
