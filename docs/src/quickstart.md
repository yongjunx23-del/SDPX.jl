# Quick start

## Install

Until SDPX is registered, install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/SDPX.jl")
```

## The public model route

The v0.5 public interface has one modeling path. Create a typed `Model`, add
product-cone variables with `variable!`, add affine cone constraints with
`constraint!`, set one objective with `objective!`, and solve with
`optimize!`. `Settings` selects policy and `Outputs` selects retained result
fields; the result is read through accessors rather than by reaching into
solver structs.

```julia
using SDPX
using LinearAlgebra

model = Model(Float64)
w = variable!(model, :w, 3; domain=Reals())
constraint!(model, :normalization, w[1] - 1, ZeroCone())
constraint!(model, :quartic_recurrence, w[1] - w[2] - w[3], ZeroCone())
constraint!(
    model,
    :moment_matrix,
    [w[1] w[2]; w[2] w[3]],
    PSDCone(),
)
objective!(model, Maximize(), w[2])

settings = Settings(
    model;
    limits=Limits(iterations=200, time=60.0, threads=1),
    verbosity=0,
)
outputs = Outputs(:all, :all, :all; objectives=true, certificate=:summary)
result = optimize!(model; settings=settings, outputs=outputs)

@assert status(result) == :optimal
moments = value(result, w)  # approximately [1, 0.618034, 0.381966]
H = [moments[1] moments[2]; moments[2] moments[3]]
eigvals(Symmetric(H))       # nonnegative up to solver tolerance
primal_objective(result)    # approximately 0.618034
certificate(result).valid  # independent original-coordinate check
```

This is the `g=1` two-by-two moment truncation of the quartic integral:
`W0=1`, `W0-W2-W4=0`, and `[W0 W2; W2 W4]` is positive semidefinite.

`Model(Float64)` is the default arithmetic. `Model(BigFloat;
precision_bits=256)` and loaded `MultiFloats` types provide the other typed
arithmetic choices; all model data is converted and owned in that arithmetic.

## Native product-cone blocks

The canonical program stores an ordered product of supported blocks. A model
may combine supported families; `Reals` and `ZeroCone` provide free coordinates
and equalities:

| Block family | Product/constraint domains |
|---|---|
| Scalar | `Nonnegative`, `Nonpositive`, `ZeroCone` |
| Lorentz | `LorentzCone`, `RotatedLorentzCone` |
| Semidefinite | `PSDCone` |
| Nonsymmetric | `ExponentialCone`, `PowerCone` |

All combinations use the same product-HSD state and certificate path.

For a native Lorentz model, keep the cone block in Lorentz coordinates:

```julia
soc = Model(Float64)
q = variable!(soc, :q, 3; domain=LorentzCone())
constraint!(soc, :fix_tail, [q[2] - 3, q[3] - 4], ZeroCone())
objective!(soc, Minimize(), q[1])
soc_result = optimize!(
    soc;
    settings=Settings(soc; verbosity=0),
    outputs=Outputs(:all, :all, :all; objectives=true, certificate=:summary),
)
@assert status(soc_result) == :optimal
```

`RotatedLorentzCone()` is available for native RSOC blocks. A PSD variable is
declared with two shape arguments and remains one matrix block:

```julia
sdp = Model(Float64)
X = variable!(sdp, :X, 2, 2; domain=PSDCone())
constraint!(
    sdp,
    :lower,
    [X[1, 1] X[1, 2] - 1; X[2, 1] - 1 X[2, 2]],
    PSDCone(),
)
objective!(sdp, Minimize(), 2 * X[1, 1] + 3 * X[2, 2])
sdp_result = optimize!(sdp; settings=Settings(sdp; verbosity=0))
```

The default `Settings(; algorithm=:auto)` (the only accepted value)
uses native product HSD and selects only implementation details. Algorithm-family selectors
(`algorithm=:lp`, `:socp`, `:sdp`) are rejected with a migration error: native product HSD is the only public
engine, and `algorithm` is now a read-only diagnostic label that never
changes the correctness path.

## Nearby SDP continuation

For a sequence of native SDP models with the same ordered variable and cone
layout but nearby numerical data, pass the previous certified `Result` as a
non-mutating continuation start:

```julia
function nearby_sdp(lambda)
    model = Model(Float64)
    X = variable!(model, :X, 2, 2; domain=PSDCone())
    constraint!(
        model,
        :upper,
        [1 - X[1, 1] -X[1, 2]; -X[1, 2] lambda - X[2, 2]],
        PSDCone(),
    )
    objective!(model, Maximize(), X[1, 2])
    return model
end

continuation_outputs = Outputs(
    :all,
    :all,
    :all;
    objectives=true,
    certificate=:summary,
    diagnostics=:summary,
)
base_result = optimize!(
    nearby_sdp(1.0);
    outputs=continuation_outputs,
)
next_result = optimize!(
    nearby_sdp(1.001);
    outputs=continuation_outputs,
    warm_start=base_result,
)
```

The source must be optimal with a valid certificate and must retain all
primal, constraint-dual, and dual-slack components. SDPX maps those original
coordinates through the target model's fresh preprocessing and scaling,
rebuilds the target primal PSD slack, and repairs the iterate to the strict
interior. It does not reuse the old presolve map, Schur matrix, or
factorization. An incompatible source safely uses ordinary cold
initialization; when diagnostics are retained, the warning and initialization
record contain the rejection reason. Result continuation is currently limited
to the native SDP route and requires target Ruiz equilibration (the default
`:auto` scaling route); `scaling=:none` safely falls back to cold
initialization. It cannot be combined with explicit starts attached through
`set_start!`, `set_dual_start!`, or `set_dual_slack_start!`.

## Settings and retained outputs

`Settings{T}` is the typed policy boundary. The most useful fields are
`tolerances=Tolerances{T}(; primal=..., dual=..., gap=...)`,
`limits=Limits(; iterations=..., time=..., threads=...)`, `engine`,
`presolve`, `scaling`, `sparse`, `formulation`, `provider`,
`equality_solver`, `working_precision_policy`, `diagnostics`, `verbosity`,
`timing`, `certification`, and `blas_threads`. `engine` accepts only
`:auto` or `:native_hsd`; `algorithm` is a read-only diagnostic label whose
only accepted value is `:auto`. Public formulation names are `:auto`,
`:variable_space_schur`, and `:dense_augmented_kkt`; unsupported
combinations fail closed.

`Outputs` controls `:all`/`:none` (or typed reference vectors) for `primal`,
`constraint_dual`, and `dual_slack`, plus `objectives`, `certificate`,
`diagnostics`, `history`, and `trace`. Accessing a field that was not retained
raises `ResultFieldNotRetained` instead of silently recomputing it.

The stable result accessors are `status`, `value`, `dual`, `dual_slack`,
`primal_objective`, `dual_objective`, `certificate`, `diagnostics`,
`iteration_history`, `performance_trace`, and `execution_plan`.

## Qualified compatibility internals

`SDPX.ingest`, `SDPX.solve!`, `SDPX.linear_program`,
`SDPX.second_order_program`, `SDPX.solve_socp`, and
`SDPX.SolverOptions` remain mature, qualified implementation interfaces used
by algorithm tests and compatibility integrations. They are intentionally not
the public v0.5 quickstart surface; new application code should use the typed
`Model` route above.
