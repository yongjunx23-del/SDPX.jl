# PR-04B evidence: start-point policy comparison

Date: 2026-09-11. Baseline HEAD `a9e2935`. **Default unchanged.**

```
julia --startup-file=no --project=. benchmark/clarabel_borrowing/start_point_comparison.jl
```

9 timed repeats plus one untimed warmup per (case, policy). 6/6 cases solved
under both policies with a valid original-coordinate certificate.

## How the two starts were reached

`product_cone_solve.jl` binds `initialization === :auto` to the route:

| route | start |
|---|---|
| `:bordered`, `:sparse_augmented` | `:identity` |
| `:expanded`, `:sparse_schur` | `:kkt` |

So the comparison is `kkt_route=:bordered` (identity start) against
`kkt_route=:expanded` (KKT start) through the **public** API.

**A rejected first attempt, recorded because it nearly produced a false result.**
Driving `product_hsd_solve!` directly with `initialization=:kkt` bypasses
equilibration and formulation selection. Both policies then *fail* on these
cases — LP: `line_search_breakdown` at 194 iterations (objective −214 vs −21)
for identity, `direction_breakdown` at 16 for kkt. Taken at face value that
would have read as "the KKT start is 12× faster on the LP"; it was a comparison
of two failures. Only the public routes actually solve.

## Results

| case | identity | kkt | kkt/identity | Δiterations |
|---|---|---|---|---|
| `lp_afiro_style` | 10 it | 9 it | 1.744 | −1 |
| `soc_disk` | 24 it | 21 it | 2.341 | −3 |
| `soc_k64` | 24 it | 26 it | **0.107** | +2 |
| `soc_many_small` | 15 it | 14 it | 0.704 | −1 |
| `psd_2x2` | 8 it | 9 it | 2.343 | +1 |
| `psd_blockdiag` | 19 it | 19 it | 0.900 | 0 |

`start_seconds` / `iteration_seconds` / `cert_seconds` are recorded per case in
`start_point_comparison.toml`, as the plan's acceptance criterion requires.

## Reading the result honestly

**No consistent winner.** The KKT start is faster on 3 cases and slower on 3.
The iteration counts barely move (−3 to +2). Per the plan's own rule — *"只有总
成本/稳健性有代表性改善后才更新 auto；未获胜的候选不进入默认"* — the `auto`
policy is **not** updated.

**The one large ratio is not a start effect.** `soc_k64` shows 0.107×, but the
KKT side also does 2 *more* iterations. The gain is therefore per-iteration cost,
i.e. the route change (`:bordered` sparse core vs `:expanded` dense), not the
start. The same confound applies in the other direction to `soc_disk` and
`psd_2x2`, where the KKT side is ~2.3× slower at *fewer* iterations.

**Confound, stated rather than hidden.** Route and start are not independently
selectable on the public surface today, so this measures the route-and-start
pair — which is exactly what `auto` decides — but it cannot attribute a
difference to the start alone. Isolating the start would need a new public knob,
and PR-04B must not add public surface while comparing.

## What this establishes and what it does not

Establishes: both starts are correct on all six cases (optimal + certified);
the accounting split the plan asks for is measurable; the current `auto`
binding is not demonstrably improvable by a uniform change.

Does **not** establish: any statement about Exp/Power or mixed cones (the plan
restricts PR-04B to pure symmetric cones first, and that is what ran); anything
at production scale (these are small cases); or that a *structure-aware* start
choice could not win — the pattern above (helps a large single cone, hurts small
ones) is a hypothesis for a follow-up, not a result.
