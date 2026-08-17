# Diagnostics and certificates

`optimize!` returns the single typed v0.5 result. Terminal state and the
compact original-coordinate certificate are always available through public
accessors:

```julia
result = optimize!(model; settings=settings, outputs=outputs)
status(result)

cert = certificate(result)
cert.available
cert.valid
cert.reason
cert.primal_objective
cert.dual_objective
cert.primal_residual
cert.dual_residual
cert.relative_gap
```

An optimal core status is exposed as `:optimal` only when the independent
original-coordinate certificate is valid. The certificate checks affine
equations, cone membership, dual stationarity, objectives, and gap after all
presolve, scaling, and reconstruction maps have been undone. Non-finite or
uncertified results fail closed rather than acquiring an optimal public
status.

A stopped solve keeps its non-optimal core status. Absence of an optimal
certificate is never interpreted as feasibility or infeasibility.

## Retention policy

`Outputs` controls the potentially large result payloads. Requested primal,
constraint-dual, and dual-slack entries are read with `value`, `dual`, and
`dual_slack`. Objectives use `primal_objective` and `dual_objective`. Accessing
a payload that was not retained raises `ResultFieldNotRetained` instead of
re-solving or returning a placeholder.

Retention applies to the returned `Result`; it does not shrink the numerical
workspace used during the solve. Iteration history is route-dependent and may
be empty even when retained if the selected core does not publish iteration
records.

```julia
outputs = Outputs(
    :all,
    :all,
    :all;
    objectives=true,
    certificate=:summary,
    diagnostics=:full,
    history=true,
    trace=true,
)
```

## Execution provenance

`execution_plan(result)` returns the immutable route/formulation/provider plan
used by the solve. When diagnostics were retained, `diagnostics(result)` adds
classification, presolve facts, planned-versus-executed algorithms, workspace
and memory estimates, phase timings, warnings, and fallback provenance.
`iteration_history(result)` and `performance_trace(result)` return their
retained payloads.

Automatic BigFloat precision staging is one predeclared same-route ladder, not
an alternate cone formulation. Full diagnostics retain its executed rungs and
each rung's child execution plan. See [architecture](architecture.md),
[parameters](parameters.md), and [precision](precision.md) for the planning and
arithmetic invariants.
