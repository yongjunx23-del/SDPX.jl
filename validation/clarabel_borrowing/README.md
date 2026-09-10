# `validation/clarabel_borrowing/`

Acceptance gates for the SDPX x Clarabel engineering plan
(`SDPX_Clarabel_engineering_plan_2ab596f.md`).

Created by PR-00. These gates are **independent references**: they must not
reuse the production assembly, RHS helpers, or factor cache they are checking.
A copied formula that agrees with itself is not evidence.

## Files

| File | Purpose | Last result |
|---|---|---|
| `soc_rank2_gate.jl` | Independent SOC rank-2 expansion algebra: metric identity, auxiliary elimination, direction agreement, storage accounting. Calls no SDPX production code. | 375 pass, 0 fail |
| `baseline_receipt.jl` | Deterministic per-case cost/status record over five analytic cases, with executed-route accounting. Writes `baseline_receipt.toml`. | 5 cases, 0 failures |

Both are runnable and were executed on 2026-09-11:

```bash
julia --startup-file=no --project=. validation/clarabel_borrowing/soc_rank2_gate.jl
julia --startup-file=no --project=. validation/clarabel_borrowing/baseline_receipt.jl
```

A gate that has never run is not evidence, so each file records what it last
produced. Neither is wired into `test/runtests.jl`; they are specialist gates,
and `baseline_receipt.toml` is a measurement record rather than an assertion.

Evidence notes for these gates live in `docs/evidence/`.
