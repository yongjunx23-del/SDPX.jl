# Benchmark and measurement protocol

**Established by Q01, 2026-09-11.** Binding on every task that reports a number.
Implements ADR-003 §7 (measurement discipline) and the packet's Q01 card.

Baseline for every figure in this document:
`docs/rebuild/baseline.md`.

---

## 1. The three phases are never averaged

| Phase | What it is | May be quoted? |
|---|---|---|
| `first_compile` | the first solve of a case in a process, including LLVM compilation | **no** — reported for diagnosis only |
| `warm_fresh_setup` | later solves that still rebuild state | with the failure counts |
| `prepared_solve` | steady-state repeats | with the failure counts |

Why this is not pedantry: measured on this baseline the gap is four orders of
magnitude. `lp_afiro_style` costs **22.6 s** to first solve and **0.00031 s** warm.
A single averaged number would describe neither and would move with the order the
cases happen to run in.

**A true `prepared_solve` is NOT reachable at this baseline.** It requires the
prepared-update API from S04/S07, which does not exist yet. The harness therefore
reports repeated fresh solves under that label **and says so in the payload**
(`prepared_solve_note`). It does not claim to have measured a prepared path.
Per ADR-003 §3, an unreachable phase is `not_run` with a reason — not estimated.

## 2. Timing medians are over successes, with failures reported beside them

```julia
success(row) = !row.threw && row.status == "optimal" && row.certificate_valid === true
```

Medians are taken over `warm_success` only, and the payload always carries
`solved` and `failed` alongside. A run that fails fast is therefore impossible to
present as a speed win. This rule was already implemented in
`benchmark/clarabel_borrowing/start_point_comparison.jl`; the same discipline is
now mandatory here.

## 3. Unmeasured is `not_run`, never `0`

ADR-003 §3. A missing measurement and a measured zero are different facts, and
serialisation must preserve the difference. `_tomlify` maps `nothing` to the
literal string `"not_run"`, so a receipt reader can always tell them apart.

A case with no independent optimum records `objective_error = "not_run"`, never
`0.0`.

## 4. Requested and executed threads are recorded separately

They genuinely differ on this host:

```
requested=1  julia=1  blas=4  cpu=4
```

Quoting only `requested` would hide that BLAS is running 4 threads underneath a
1-thread solve. Both are recorded, and `cpu_threads` is recorded so a reader can
tell how much of the machine was available.

Packet tiers 16 and 64 are **not measurable here** and are recorded as
`unsupported` with the host fact. Per `AGENTS.md`, `unsupported` may satisfy an
explicit-refusal negative test but **cannot** satisfy a required release
capability.

## 5. No name-based dispatch

A benchmark **name** may be recorded in a result and used to look up a fixture.
It may **not** choose a route, provider, formulation or settings value.

`test/rebuild/dependency_rules.jl` enforces this **statically**. Two earlier
formulations of the rule were tried and rejected, and the rejections are recorded
in the test file because they are the reason the final rule is shaped as it is:

1. *"does the tree compare a spec id?"* — flagged nine legitimate
   `filter(spec -> spec.id === :name, inventory)` fixture lookups.
2. *that, plus "is strategy vocabulary nearby?"* — flagged
   `spec.family === :lp ? :lp_native : :sdp_native` in
   `benchmark/bootstrap/runner_impl.jl`, which assigns a **descriptive**
   `conic_formulation` field on a catalog record. No strategy is selected.

The final rule asks the question the packet actually poses: does a name
comparison branch to a symbol that is a real **route or provider identifier**?
The vocabulary is taken from the public route validator so it cannot drift from
the product silently. Result at this baseline: **0 violations**, 13 informational
name comparisons that only label records.

## 6. Static checks declare themselves static

A static scan can find a forbidden dependency that exists. It **cannot** prove
that an allowed path is exercised, that two methods are never both loaded, or
that some code is unreachable. Every check in `dependency_rules.jl` carries
`static_only = true` and `limits_note`, and the first testset asserts those
declarations exist — so a reader cannot mistake the scan for a reachability
proof. Where a rule would need dynamic evidence it is reported informationally
rather than asserted.

The same discipline applies to the duplicate-definition scan: it is
**informational**, because Julia overloading legitimately reuses a name across
files. Asserting it would be a static check masquerading as a semantic one.

## 7. Reproducibility

`benchmark/rebuild/manifest.jl` produces a `cases_fingerprint`: a stable hash over
the input shape trace only (dimensions, cone signature, id, family). It
deliberately excludes the `build` closure, which is not stable across revisions.
Two runs with the same fingerprint and the same SDPX SHA describe the same
inputs, so a rerun is comparable — Q01 acceptance item 1.

`measure.jl` records `sdpx_head` and `worktree_dirty` in every payload. A receipt
from a dirty tree is marked as such; it is not silently presented as a revision
measurement.

## 8. Functional tests and performance measurement do not share a machine

Packet acceptance: "支持独占性能runner；并行功能测试与性能测量分开." The
measurement harness is a standalone entry point, not part of `Pkg.test()`:

```bash
julia --startup-file=no --project=. benchmark/rebuild/measure.jl --repeats=5
```

`test/rebuild/dependency_rules.jl` is a functional check and is safe to run in
parallel with anything. `measure.jl` is not: it must own the machine.

## 9. What this protocol does NOT establish

- No provider (MF/BF) measurement: the provider environment does not exist in the
  default project (ADR-002 §1). MF/BF rows are `not_run`.
- No allocation disaggregation for BigFloat. ADR-003 §7 requires Julia heap / cell
  identity / native allocator / RSS to be distinguished; that is only meaningful
  once a BF provider is loadable, and it is not attempted here.
- No prepared-solve measurement, per §1.
- The allocation figure is a **whole-public-call** figure including result
  construction, certificate assembly and recovery. It is not an inner-loop
  allocation figure and must never be reported as one.
