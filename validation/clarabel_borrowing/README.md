# `validation/clarabel_borrowing/`

Acceptance gates for the SDPX x Clarabel engineering plan
(`SDPX_Clarabel_engineering_plan_2ab596f.md`).

Created by PR-00. These gates are **independent references**: they must not
reuse the production assembly, RHS helpers, or factor cache they are checking.
A copied formula that agrees with itself is not evidence.

## Files

| File | Purpose |
|---|---|
| `full_jacobian_oracle.jl` | Direct five-equation Jacobian oracle for the product-cone HSD system, built without production residual helpers. |
| `contract_smoke.jl` | Frozen-contract smoke: gap normalization, cone layout storage, scratch ownership, start counting. |

Run one gate directly:

```bash
julia --startup-file=no --project=. validation/clarabel_borrowing/contract_smoke.jl
```

These entry points were created by PR-00 and have been executed at least once
in the development environment; a gate that has never run is not evidence.
