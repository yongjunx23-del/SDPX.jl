# Benchmarks and evidence

SDPX keeps measured decisions close to the code or benchmark that produced
them. This page records where the evidence lives and how to read it; it does
not repeat every table.

## Benchmark locations

- [`bench/soc_fixed_trace/`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/benchmark/soc_fixed_trace/README.md) — the native
  fixed-trace Q3 versus PSD2 campaign, including models, launcher policy,
  certificate checks, and provenance.
- [`bench/RESULTS.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/evidence/bench/RESULTS.md) — retained historical context and
  the small smoke tier index.
- [`bench/automatic_pipeline/PARAMETER_SELECTION.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/evidence/bench/automatic_pipeline/PARAMETER_SELECTION.md)
  — LP cold-start crossover and distant-start sweeps.
- [`bench/threading/RESULTS.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/evidence/bench/threading/RESULTS.md) — Schur
  scheduling kernel measurements.
- [`bench/mixed_precision_kkt/RESULTS.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/evidence/bench/mixed_precision_kkt/RESULTS.md)
  — mixed-precision KKT promotion evidence.
- [`bench/opt2026/`](https://github.com/yongjunx23-del/SDPX.jl/tree/main/bench/opt2026) — native BigFloat optimization,
  preprocessing, and cluster campaign reports.

## Reporting policy

Performance claims must state the model, arithmetic, tolerances, hardware,
thread configuration, warm-up, timing boundary, numerical result, and
repeated-run statistic. The independent original-coordinate certificate is
the validation boundary: a row without a valid certificate is not a solve
claim. Benchmark results are workload-specific evidence and must not be
generalized to other solvers, hardware, tolerances, or problem families.

Measurements that gate a production default are recorded in the code at the
constant they gate. Changing one requires re-running the sweep, not one
benchmark. The fixed-trace campaign, for example, records that the native Q3
specialization is now selected by structure whenever its exact reduction
verifies; its matched certificate/performance gates remain documented with
the benchmark, not encoded as size gates in the planner.

## Native SOC evidence

The lightweight scoreboard in the SOC reports warms each route once and then
measures fresh solves. The native fixed-trace specialization is primarily a
dimensional reduction plus local elimination: for many independent fixed-head
Q3 blocks it uses a fraction of the GeneralLorentz workspace and is
substantially faster, while the PSD reference is retained only as a
correctness and comparison route. At very small block counts the
specialization's dispatch overhead can outweigh its structural saving, which
is why the automatic selection is structural rather than a blanket
performance preference.

The Q3 HKM direction remains the validated default after matched gates; the
Nesterov--Todd direction is an explicit research option. Historical PSD2-
versus-Q3 campaign evidence is arithmetic-dependent: in the retained matched
BigFloat measurements the PSD2 reference won on total runtime, but that is
campaign evidence, not a current route policy. NativeSOC BigFloat general and
fixed-trace execution are production and tested today.

## Adaptive parameter evidence

The adaptive parameter controller and its fixed fallback are documented in
[`docs/adaptive-parameter-policy.md`](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/src/adaptive-parameter-policy.md), with
equations, exact bounds, and arithmetic-specific behavior. Measured
fixed-versus-adaptive decisions appear in the benchmark locations above.
