# Qualified internals

The public API is the typed model/result surface described in
[Quick start](quickstart.md). The names on this page are implementation details
for solver development and diagnostics; use explicit `SDPX.name` qualification
and expect them to change before 1.0.

## Canonical layer

The principal internal records are:

- `CanonicalConicProgram` — typed `A*x+s=b`, objective, cone layout, and
  reconstruction ownership;
- `ConeProductLayout` and cone block descriptors — ordered canonical blocks;
- canonical transform/reconstruction records — primal, dual, slack, objective,
  and ray inverse maps;
- equality, presolve, and equilibration maps.

The canonical layer is the semantic boundary. A provider or KKT route may not
reinterpret its coordinates.

## Product-HSD runtime

`ProductConeHSDState` owns the five homogeneous variables, cone runtime,
accepted iterate, Newton workspaces, route sessions, receipts, and performance
trace. The iteration entry points are internal state-machine operations, not a
second public solve API.

Termination helpers produce candidates only. Original-coordinate verification
remains the authority for `Optimal`, `PrimalInfeasible`, and
`DualInfeasible`.

## Newton and KKT layer

`NewtonSystem` and its right-hand side freeze the five equations. Internal
sessions implement bordered, expanded, and sparse-Schur realizations.

`FactorReceipt` records factor-wide evidence and generations. Direction checks
still evaluate the exact operator and all five equations for each right-hand
side.

The canonical local-Q3 specialization is named `disjoint_fixed_head_q3`.
Eligibility requires a fixed positive Q3 head, exactly two local tail variables
per block, and disjoint tail-variable pairs across blocks; a fixed head alone is
not sufficient. Dense shared-variable partial-wave SOCPs therefore use the
general symmetric core. Legacy implementation symbols and the source filename
retain `fixed_trace_q3` as a compatibility spelling, but diagnostics use the
canonical name and generic-core failures use `symmetric_core_*` reasons.

## Cone runtime

The product runtime contains block-specific operations for nonnegative,
Lorentz, PSD, exponential, and power cones. Exact canonical transforms handle
nonpositive and rotated Lorentz coordinates. Local PSD panels and fixed-size
3x3/Q3 kernels are specializations of these operations, not public solver
families.

## Planning and diagnostics

Internal plan/trace records expose:

- planned and executed KKT routes;
- provider and specialization;
- symbolic/operator/factor generations;
- fallback attempts;
- memory and fill estimates; and
- phase timings.

These records explain execution but cannot override certificate status.

## Provider extensions

Provider extensions register capabilities and factor/solve hooks. They must
preserve arithmetic and obey invalidation, multi-RHS, transpose, alias, and
thread-ownership contracts. See [Providers](providers.md).

## Development rule

Do not build application code against internal field layouts. If a diagnostic
or control is broadly useful, add a typed public accessor rather than exporting
an HSD, cone, KKT, or provider workspace.
