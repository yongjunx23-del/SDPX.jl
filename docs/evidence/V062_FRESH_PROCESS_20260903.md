# v0.6.2-dev fresh-process representative matrix — 2026-09-03

Source commit: `48f5620ec04dcbcc7e1afa41bb4b048d3a68e9e4`  
Tree fingerprint: `5d4cddccc4f115cce3522e3d2723590d9bdd5493`  
Host: Apple M4, Julia 1.12.6, Float64, one Julia/BLAS/OMP thread.  
Protocol: schema-v9; one excluded warmup plus three distinct measured processes per row. Every row passed its independent reference and original-coordinate certificate.

| Family | Case | Iterations | Median total | Median core | Median setup | Median recovery | Peak RSS | Certificate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LP | `v2_lp_box_small` | 9 | 22.888 s | 11.538 s | 0.621 s | 0.032 s | 1.79 GiB | pass |
| Ill-conditioned LP | `v2_ill_diag_ladder_small` | 9 | 22.867 s | 11.546 s | 0.584 s | 0.033 s | 1.75 GiB | pass |
| SOCP | `v2_soc_one_large_small` | 13 | 29.395 s | 17.023 s | 1.805 s | 0.033 s | 1.97 GiB | pass |
| RSOC | `v2_rsoc_quadratic_epigraph_small` | 23 | 23.187 s | 11.614 s | 0.807 s | 0.033 s | 1.79 GiB | pass |
| SDP | `v2_sdp_weighted_trace_small` | 9 | 23.327 s | 11.549 s | 0.679 s | 0.035 s | 1.71 GiB | pass |
| EXP | `v2_exp_unit_epigraph_small` | 14 | 23.535 s | 11.792 s | 0.768 s | 0.033 s | 1.76 GiB | pass |
| Power | `v2_power_interior_epigraph_small` | 31 | 22.632 s | 11.354 s | 0.750 s | 0.032 s | 1.75 GiB | pass |
| Mixed six-cone | `v2_mixed_planted_cross_cone_small` | 10 | 36.474 s | 23.991 s | 1.706 s | 0.036 s | 2.21 GiB | pass |

These are intentionally cold fresh-process values. The nearly constant ~11.5 s core floor is compilation, not the warmed numerical kernel; sysimage receipts must be reported separately.

Completion marker SHA-256 values:

```text
cd1b749b891895ff08fb4381506ed7e987e417a1389b44a44d99c1b8614a3b0f  v2_exp_unit_epigraph_small.complete.toml
15c828dfc33db27cf667a12341d398a355f40a2dd45600724925f90ff4a6a8ad  v2_ill_diag_ladder_small.complete.toml
4e0efdcbddd557d531e88abe6ea79ea61196825093e9794d61fcd2b59477997f  v2_lp_box_small.complete.toml
0ae97893809a4b93f5bc77533e9df601ba46bb3ac34c5fb725c7fad9fed25c84  v2_mixed_planted_cross_cone_small.complete.toml
e25b9b548c87c439a0cd62f4278eef2ee7adf6ab10a98a1c25708828ca5a6646  v2_power_interior_epigraph_small.complete.toml
de1c46ec176a8d0d8ccdfd7c3bffe5ed6c6eaadcd78e689ae0079d4acc226780  v2_rsoc_quadratic_epigraph_small.complete.toml
8bcc429f92a1cd9a52f1cda91e9d9c4d041bda5fbaba0cbec308193e12934ca1  v2_sdp_weighted_trace_small.complete.toml
261468f7513f5a93241f5252f723f19b7679c6820af132e17bdc3fb306e51b9b  v2_soc_one_large_small.complete.toml
```

Raw private bundles: `/tmp/sdpx-v062-fresh-48f5620/`.
