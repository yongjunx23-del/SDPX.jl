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
    algorithm=:sdp,
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

## The three pure routes

The route classifier looks only at the non-free cone families in the model.
Use one family per model, with `Reals` and `ZeroCone` allowed as auxiliary
blocks:

| Route | Product/constraint domains | `Settings.algorithm` |
|---|---|---|
| LP | `Nonnegative`, `Nonpositive`, `ZeroCone` | `:auto` or `:lp` |
| native SOC/RSOC | `LorentzCone`, `RotatedLorentzCone`, `ZeroCone` | `:auto` or `:socp` |
| SDP | `PSDCone`, `ZeroCone` | `:auto` or `:sdp` |

For a native Lorentz model, keep the cone block in Lorentz coordinates:

```julia
soc = Model(Float64)
q = variable!(soc, :q, 3; domain=LorentzCone())
constraint!(soc, :fix_tail, [q[2] - 3, q[3] - 4], ZeroCone())
objective!(soc, Minimize(), q[1])
soc_result = optimize!(
    soc;
    settings=Settings(soc; algorithm=:socp, verbosity=0),
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
sdp_result = optimize!(sdp; settings=Settings(sdp; algorithm=:sdp, verbosity=0))
```

Choose `algorithm=:lp`, `:socp`, or `:sdp` to match the model family shown in
the table. `:auto` asks SDPX to select the matching implemented route.

## Settings and retained outputs

`Settings{T}` is the typed policy boundary. The most useful fields are
`tolerances=Tolerances{T}(; primal=..., dual=..., gap=...)`,
`limits=Limits(; iterations=..., time=..., threads=...)`, `algorithm`,
`presolve`, `scaling`, `sparse`, `formulation`, `provider`,
`equality_solver`, `working_precision_policy`, `diagnostics`, `verbosity`,
`timing`, `certification`, and `blas_threads`. Public formulation names are
`:auto`, `:variable_space_schur`, and `:dense_augmented_kkt`; unsupported
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
