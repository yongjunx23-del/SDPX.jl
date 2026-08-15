# Manifest notes

- `downloads.toml`: direct public URLs where pinned reliably.
- `pathological_suites.toml`: parameter grids for generated numerical stress tests.
- `lp_netlib_core20.csv`: classic LP regression set.
- `../references/mittelmann_lp_modern20_2026.csv`: modern large LP reference subset.
- `../references/mittelmann_network_lp25_2026-07-01.csv`: network LP stress set.
- `../references/mittelmann_sdp_core30_2026-04-25.csv`: curated SDP core.
- `../references/mittelmann_socp_full18_2026-08-10.csv`: SOCP full 18, with `core12`.
- `../references/mittelmann_infeasible_sdp_8x100_2025-09-21.csv`: reported infeasibility counts.

Reference timing/status strings must stay strings because flags such as `f`,
`m`, `t`, and `a` are semantically meaningful.
