# V2 provider-backed precision qualification

Source solver/benchmark commit: `576dabc` (`fix/general-benchmark-v2-review`).
Generated 2026-09-02 with Julia 1.12, `--gcthreads=1`, `JULIA_NUM_THREADS=4`,
MultiFloats 3.3.1, MultiFloatLinearAlgebra 0.4.0, and
BigFloatLinearAlgebra 0.3.0. All 17 live certified optimal-path cases run; the
two Farkas/ray status cases are intentionally outside this optimal-path report.

Qualification is fail-closed through status, public certificate, the full
original-coordinate metric gate, an independent analytic oracle at
`max(256,2*bits)`, and the central precision-specific objective allowance.
No tolerance, formulation, or solver source is changed by this report.

## Summary

| Precision | Process isolation | Qualified | Honest residual finding |
|---|---|---:|---|
| Float64x2 | one fresh tier process | **17/17** | none |
| Float64x4 | one fresh process **per case** | **17/17** | provider/runtime instability: `v2_ill_near_rank_loss_small` first final attempt exited 139 after 878,765,757 allocations / 85 GCs; the fresh retry certified |
| BigFloat256 | one fresh tier process under `setprecision(BigFloat,256)` | **16/17** | `v2_exp_unit_epigraph_small` terminates `numerical_breakdown` at iteration 34 with invalid certificate |

A whole-tier Float64x4 process is not an accepted orchestration mode: three
qualification-development attempts exited with SIGSEGV before emitting their
first row (about 914 million allocations). Per-case fresh-process isolation
prevents one provider/runtime crash from erasing unrelated receipts. A crashed
process is never converted into a solver status or silently qualified.

These final receipts supersede the earlier 5/15 x2 and 4/15 x4 matrices. Those
counts were benchmark-layer false negatives caused by non-fail-closed precision
construction and ambient-BigFloat-dependent model fingerprints, not evidence
that the corresponding solvers failed. The fixed model token uses each
arithmetic's stable representation (canonical MultiFloat limb bits / stored
BigFloat precision), and exact rational source oracles no longer demand
bit-exact certificate equality outside the central allowance gate.

## Float64x2 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `Float64x2` | `optimal` | true | `-4.999999999999999503608309318307` | `4.963916906816929999999999999999999999999999999999999999999999841892918361795326e-16` | `17` | `24.67789425` | **qualified** | status/certificate/oracle gate |
| `v2_lp_sparse_planted_small` | `Float64x2` | `optimal` | true | `-3.99999999999999987788030486853662` | `1.221196951314633800000000000000000000000000000000000000000000126242063105505657e-16` | `17` | `0.000602375` | **qualified** | status/certificate/oracle gate |
| `v2_lp_nonpositive_small` | `Float64x2` | `optimal` | true | `-2.0` | `0.0` | `24` | `1.92967075` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `Float64x2` | `optimal` | true | `1.0000000000000039925969389862234205` | `3.992596938986223420499999999999999999999999999999999999999999991524335633810499e-15` | `18` | `0.204493208` | **qualified** | status/certificate/oracle gate |
| `v2_ill_diag_ladder_small` | `Float64x2` | `optimal` | true | `-4.99999999999999950360830931830717` | `4.963916906816928299999999999999999999999999999999999999999999941977120515936364e-16` | `17` | `0.000501584` | **qualified** | status/certificate/oracle gate |
| `v2_ill_near_rank_loss_small` | `Float64x2` | `optimal` | true | `-1.9999999999999990323604031830587` | `9.676395968169412999999999999999999999999999999999999999999999968246461916011881e-16` | `15` | `0.000387959` | **qualified** | status/certificate/oracle gate |
| `v2_soc_one_large_small` | `Float64x2` | `optimal` | true | `2.0` | `0.0` | `17` | `0.36400075000000004` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `Float64x2` | `optimal` | true | `16.0000000000000000000000000000000000000020203809154758067` | `2.020380915475806700000000000000000000045265940947713031276741716159640893126369e-39` | `9` | `0.009986167` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_quadratic_epigraph_small` | `Float64x2` | `optimal` | true | `1.500000000000000000000000008838372930192928` | `8.838372930192927999999999999999999999999999999999997116229197827562904246191422e-27` | `16` | `0.0005478330000000001` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_perspective_ls_small` | `Float64x2` | `optimal` | true | `0.500000000000000000000000008838372930192928` | `8.838372930192927999999999999999999999999999999999997116229197827562904246191422e-27` | `16` | `0.0005242910000000001` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_many_qr3_small` | `Float64x2` | `optimal` | true | `24.000000000000000000000000015651394790037485` | `1.565139479003748499999999999999999999999999999999990982635266281754754060449361e-26` | `15` | `0.010826959` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_weighted_trace_small` | `Float64x2` | `optimal` | true | `1.00000000000000000000000000000000000000000000007459932733670201` | `7.459932733670200999999999999999689867417189746527847112605005192227837356260245e-47` | `16` | `0.00802825` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `Float64x2` | `optimal` | true | `-3.9999999999999999999999999941759385443664225` | `5.824061455633577500000000000000000000000000000000004874113981523411268046768993e-27` | `20` | `0.0041595420000000004` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_multiblock_small` | `Float64x2` | `optimal` | true | `8.0000000000000032357864495500059` | `3.235786449550005900000000000000000000000000000000000000000000024663913581217719e-15` | `16` | `0.017793042000000002` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `Float64x2` | `optimal` | true | `1.0` | `n/a` | `21` | `0.022804167` | **qualified** | status/certificate/oracle gate |
| `v2_power_interior_epigraph_small` | `Float64x2` | `optimal` | true | `0.249999999999999508946782625814278` | `-4.910532173741857219999999999999999999999999999999999999999999990523949515953268e-16` | `52` | `0.008646333` | **qualified** | status/certificate/oracle gate |
| `v2_mixed_planted_cross_cone_small` | `Float64x2` | `optimal` | true | `7.99999999999999999999999999999992046013630265932` | `-7.953986369734068000000000000000000000000000003028028444624961995212996418802165e-32` | `17` | `0.003391625` | **qualified** | status/certificate/oracle gate |

## Float64x4 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `Float64x4` | `optimal` | true | `-4.99999999999999999999999999991202969274093344266811082730338130522396450932006` | `8.797030725906655733188917269661869477603549067994000000000000000000000000000026e-29` | `30` | `113.93314466700001` | **qualified** | status/certificate/oracle gate |
| `v2_lp_sparse_planted_small` | `Float64x4` | `optimal` | true | `-3.99999999999999999999999999997835806750361425130610170594350438073742930859205` | `2.164193249638574869389829405649561926257069140794999999999999999999999999999993e-29` | `30` | `113.198757208` | **qualified** | status/certificate/oracle gate |
| `v2_lp_nonpositive_small` | `Float64x4` | `optimal` | true | `-2.0` | `0.0` | `45` | `113.59408987500001` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `Float64x4` | `optimal` | true | `1.000000000000000000000000000707572437083697979700306628107304091241974260758` | `7.075724370836979797003066281073040912419742607580000000000000000000000000000023e-28` | `31` | `114.38824208400001` | **qualified** | status/certificate/oracle gate |
| `v2_ill_diag_ladder_small` | `Float64x4` | `optimal` | true | `-4.99999999999999999999999999991202969274093344266811082730338130397054956687977` | `8.797030725906655733188917269661869602945043312023000000000000000000000000000034e-29` | `30` | `113.36757875` | **qualified** | status/certificate/oracle gate |
| `v2_ill_near_rank_loss_small` | `Float64x4` | `optimal` | true | `-1.999999999999999999999999999982079875710275827105757496112294138662593113889765` | `1.79201242897241728942425038877058613374068861102349999999999999999999999999999e-29` | `29` | `113.696122375` | **qualified** | status/certificate/oracle gate |
| `v2_soc_one_large_small` | `Float64x4` | `optimal` | true | `2.0` | `0.0` | `30` | `117.404879042` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `Float64x4` | `optimal` | true | `16.0` | `0.0` | `15` | `113.91122158300001` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_quadratic_epigraph_small` | `Float64x4` | `optimal` | true | `1.500000000000000000000000000024036470739680064410661340720411633683243705478365` | `2.403647073968006441066134072041163368324370547836499999999999999999999999999991e-29` | `32` | `113.87844454200001` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_perspective_ls_small` | `Float64x4` | `optimal` | true | `0.500000000000000000000000000024036470739680064410661340720411633683243705478365` | `2.403647073968006441066134072041163368324370547836499999999999999999999999999991e-29` | `32` | `113.208680208` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_many_qr3_small` | `Float64x4` | `optimal` | true | `24.000000000000000000000000000207836584990478129538276269685272212096403285072587` | `2.078365849904781295382762696852722120964032850725870000000000000000000000000011e-28` | `30` | `113.66629129100001` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_weighted_trace_small` | `Float64x4` | `optimal` | true | `0.9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999993710658859023460710049820701097706165710832914076` | `-6.289341140976539289950179298902293847098203332877174721727841500290329223357213e-91` | `30` | `113.567127708` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `Float64x4` | `optimal` | true | `-3.999999999999999999999999999999999999999646114007155330472307234800212004109596889425013407014231819746832306094772243668684025487462055692710889583986841202399002615758380647925651629748244886286556720733642578125` | `3.538859928446695276927651997879958904031105749865929857681802531676939052277563e-40` | `35` | `113.64342291700001` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_multiblock_small` | `Float64x4` | `optimal` | true | `8.0000000000000000000000000005734439772711735326157601244065875582458674044093` | `5.734439772711735326157601244065875582458674044092999999999999999999999999999992e-28` | `29` | `113.74587008300001` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `Float64x4` | `optimal` | true | `0.9999999999999999999999999999999999999999999999999999999999999999620177290169608050101070309217521713831161366655202201348808800366839670774207553638675242729143455996278894680473481315630124299786984920501708984375` | `n/a` | `59` | `114.26689787500001` | **qualified** | status/certificate/oracle gate |
| `v2_power_interior_epigraph_small` | `Float64x4` | `optimal` | true | `0.249999999999999999999999999957378846238172243331700041829118705894423945102793` | `-4.262115376182775666829995817088129410557605489720700000000000000000000000000001e-29` | `85` | `113.876660791` | **qualified** | status/certificate/oracle gate |
| `v2_mixed_planted_cross_cone_small` | `Float64x4` | `optimal` | true | `8.000000000000000000000000000000000000000000000000000000000000018195392656755189391331561974145255980725173558348` | `1.819539265675518939133156197414525598072517355834799999999999998251900768933423e-62` | `30` | `113.858680417` | **qualified** | status/certificate/oracle gate |

## BigFloat256 matrix

| Case | Precision | Status | Certificate | Objective | Error vs exact | Iterations | Core s | Classification | Reason |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| `v2_lp_box_small` | `BigFloat256` | `optimal` | true | `-4.999999999999999999999999999999989509378492577511102906652244244790516095174084` | `1.049062150742248889709334775575520948390482591599999999999999999999999999999999e-32` | `34` | `32.689731459` | **qualified** | status/certificate/oracle gate |
| `v2_lp_sparse_planted_small` | `BigFloat256` | `optimal` | true | `-3.999999999999999999999999999999997419161944207210751416401498169394207313601762` | `2.580838055792789248583598501830605792686398237999999999999999999999999999999998e-33` | `34` | `0.008004416` | **qualified** | status/certificate/oracle gate |
| `v2_lp_nonpositive_small` | `BigFloat256` | `optimal` | true | `-2.0` | `0.0` | `51` | `1.984854375` | **qualified** | status/certificate/oracle gate |
| `v2_lp_chebyshev_small` | `BigFloat256` | `optimal` | true | `1.000000000000000000000000000000084379548154000455742739870384769037145447259115` | `8.437954815400045574273987038476903714544725911499999999999999999999999999999998e-32` | `35` | `3.399348042` | **qualified** | status/certificate/oracle gate |
| `v2_ill_diag_ladder_small` | `BigFloat256` | `optimal` | true | `-4.999999999999999999999999999999989509378492577511102906652244244790516095173877` | `1.049062150742248889709334775575520948390482612299999999999999999999999999999998e-32` | `34` | `0.007069375` | **qualified** | status/certificate/oracle gate |
| `v2_ill_near_rank_loss_small` | `BigFloat256` | `optimal` | true | `-1.999999999999999999999999999999997862992802744340439072638259736991532520158354` | `2.137007197255659560927361740263008467479841645999999999999999999999999999999992e-33` | `33` | `0.005162375` | **qualified** | status/certificate/oracle gate |
| `v2_soc_one_large_small` | `BigFloat256` | `optimal` | true | `2.0` | `0.0` | `35` | `17.306772166000002` | **qualified** | status/certificate/oracle gate |
| `v2_soc_q3_load_sharing_small` | `BigFloat256` | `optimal` | true | `16.0` | `0.0` | `17` | `0.25501329100000003` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_quadratic_epigraph_small` | `BigFloat256` | `optimal` | true | `1.500000000000000000000000000000002866348827448569542427407256452883219796156846` | `2.866348827448569542427407256452883219796156845999999999999999999999999999999994e-33` | `36` | `0.017786041000000002` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_perspective_ls_small` | `BigFloat256` | `optimal` | true | `0.500000000000000000000000000000002866348827448569542427407256452883219796156846` | `2.866348827448569542427407256452883219796156845999999999999999999999999999999994e-33` | `36` | `0.017677250000000002` | **qualified** | status/certificate/oracle gate |
| `v2_rsoc_many_qr3_small` | `BigFloat256` | `optimal` | true | `23.99999999999999999999999999999999359327577703412941216793285088030958802956624` | `-6.40672422296587058783206714911969041197043375999999999999999999999999999999995e-33` | `34` | `0.46074425` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_weighted_trace_small` | `BigFloat256` | `optimal` | true | `1.0` | `0.0` | `34` | `0.07855445800000001` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_maxcut_k4_small` | `BigFloat256` | `optimal` | true | `-3.999999999999999999999999999999999999999999939661317465760391435695836087313574` | `6.033868253423960856430416391268642600000000000000000000000000000000000000000014e-44` | `42` | `0.23382091700000002` | **qualified** | status/certificate/oracle gate |
| `v2_sdp_multiblock_small` | `BigFloat256` | `optimal` | true | `8.000000000000000000000000000000068384230312181105949675575688416270959354933493` | `6.838423031218110594967557568841627095935493349299999999999999999999999999999987e-32` | `33` | `0.8160248750000001` | **qualified** | status/certificate/oracle gate |
| `v2_exp_unit_epigraph_small` | `BigFloat256` | `numerical_breakdown` | false | `0.0` | `n/a` | `34` | `0.23754708400000002` | **not-qualified** | objective_interval,certificate,status |
| `v2_power_interior_epigraph_small` | `BigFloat256` | `optimal` | true | `0.2499999999999999999999999999999901769872649288084573830818723938549297080935641` | `-9.823012735071191542616918127606145070291906435899999999999999999999999999999952e-33` | `99` | `0.619307708` | **qualified** | status/certificate/oracle gate |
| `v2_mixed_planted_cross_cone_small` | `BigFloat256` | `optimal` | true | `8.0` | `0.0` | `34` | `0.19672191700000002` | **qualified** | status/certificate/oracle gate |

## Contract

A row is **qualified** only when the requested provider is loaded, status is
`:optimal`, the public certificate is valid, every required original-coordinate
metric is finite and within its gate, the independent source/oracle proof
passes, and the objective is within `max(public_interval_half_width,
precision.certificate_limit)`. Reviewed solver tolerances / certificate limits
are Float64x2 `1e-15 / 5e-13`, Float64x4 `1e-28 / 5e-22`, and BigFloat256
`1e-32 / 5e-28`. Process crashes are orchestration/provider findings and never
solver statuses.
