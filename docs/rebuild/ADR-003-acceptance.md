# ADR-003: Acceptance and certification

**Status: ACCEPTED (A00, 2026-09-11).** Binding on every task card, on Q01/Q02,
on V01, and on I01/I02/I03.

This ADR answers A00's third acceptance item — "任务图、文件白名单无冲突；上一轮
性能和数值门槛成为继承gate" — and fixes what "done" is allowed to mean.

---

## 1. The three certification layers are not merged

Per packet §3.4. A single `check=true` flag is forbidden.

| Layer | Checks | Runs when |
|---|---|---|
| L1 boundary/structure | dimensions, indices, ownership, precision, pattern and mutation protocol | setup, data import, capability/policy change, after untrusted access |
| L2 numeric step | finiteness, interiority, scaling map, true Newton residual, required error bounds | every candidate direction / accepted point |
| L3 result | original-coordinate primal/dual/cone/gap/ray against declared tolerances | termination, checkpoint re-adoption, result publication |

**Only L1 may be skipped** via exclusive ownership plus a generation lease.
L2 and L3 may never be replaced by a lease. In particular:

> `trusted`只意味着已有独占admission/lease，不意味着可以跳过数值
> finite/backward/原始坐标检查。

## 2. Production gates do not vary with verbosity

Packet §3.4: "生产 correctness gate 不得因为 `verbose=false`、计时关闭或
benchmark mode 而改变." A gate that can be switched off by a logging option is
not a gate. Debug depth may be *added*; nothing may be *removed*.

## 3. Status vocabulary

`AGENTS.md` fixes two orthogonal vocabularies; conflating them is a defect.

- Work status: `not_started` | `in_progress` | `blocked` | `needs_review` | `accepted`
- Test status: `pass` | `fail` | `not_run` | `unsupported`

Rules carried from the packet:

- `unsupported` **may** satisfy an explicit-refusal negative test.
- `unsupported` **may not** satisfy a required release capability.
- Unrun numbers must be `null`. **Never `0`.** A missing measurement and a
  measured zero are different facts.
- A single micro-kernel result is never an end-to-end speed conclusion.
- A missing script or a missing dependency is an **infrastructure** problem, not
  a numerical-algorithm failure. It must be reported as such and must not be
  recorded as a numeric `fail`.

## 4. Evidence required from every task

Per `AGENTS.md`: full base and candidate SHA, changed paths, command, cwd, exit
code, log, input hash, arithmetic/precision/rounding/tolerance, actual threads,
raw samples, unresolved items, and the rollback commit. Reports use
`templates/agent_report.json`.

An implementation's self-report is **not** evidence. That is V01's entire purpose:
"独立对抗审查：不由实现者批准自己的正确性".

## 5. Inherited numeric and performance gates

These predate the packet and become inheritance gates (see
`docs/rebuild/baseline.md` §3 for their exact locations and current status):

1. Every public route must terminate with a **valid original-coordinate
   certificate**. A solver status alone is never acceptance.
2. `Pkg.test()` green in the default environment.
3. The platform/thread matrix (`test.yml`) green.
4. A gate that is not executed is not inherited — it is an artefact. Three
   existing artefacts are named in `baseline.md` §3.

## 6. Forbidden acceptance shortcuts

From `AGENTS.md` and the packet's §5, restated as one list because they are the
failure modes that have actually occurred in this project:

- no implicit precision downgrade;
- no unauthorized tolerance relaxation;
- no benchmark-name-based dispatch;
- no deleting a failing sample;
- no labelling a prototype as production;
- no second production HSD loop;
- no widening a lease instead of adding a gate;
- no treating a *retained physical factor* as a *valid logical factor*.

## 7. Measurement discipline

The packet's §7.4-style rules, which this ADR adopts:

- Measurement must account for packing, setup, recovery and certification;
  compilation is listed separately and never folded into a warm number.
- BigFloat allocation must distinguish Julia heap, cell identity, native
  allocator, and RSS. Reporting one as another is a defect.
- A higher-precision reference must be **independent of the optimization path
  under test** — "同一个错误内核不能既生成答案又验证答案".
- Timing medians are taken over **successful** runs only, and the success/failure
  counts are reported alongside, so a fast failure cannot masquerade as a win.
  (This rule is already implemented in
  `benchmark/clarabel_borrowing/start_point_comparison.jl`.)
- One change category per transition: pure responsibility move, data
  representation, numeric kernel, policy default, and compatibility cleanup must
  not be combined into one unattributable patch.

## 8. Consequences for the packet's own claims

`packet_validation.json` reports the packet's self-check as `pass` with
`julia_tests_run: false` and `numerical_benchmarks_run: false`. That is the
correct and honest status: it means the packet is internally consistent, and
**nothing about the source, numerics, or performance**. No task may cite packet
validation as evidence of implementation.
