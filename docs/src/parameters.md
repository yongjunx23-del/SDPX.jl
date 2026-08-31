# Parameters

Public solve policy is typed and attached to the model passed to `optimize!`.
Unknown keywords fail; there is no catch-all option sink.

```julia
using SDPX

model = Model(Float64)
x = variable!(model, :x, 1; domain=Nonnegative())
constraint!(model, :lower, x[1] - 1, Nonnegative())
objective!(model, Minimize(), x[1])

settings = Settings(
    model;
    tolerances=Tolerances(Float64; primal=1e-8, dual=1e-8, gap=1e-8),
    limits=Limits(iterations=200, time=60.0, threads=1),
    kkt_route=:bordered,
    diagnostics=:summary,
    certification=true,
    verbosity=0,
)

result = optimize!(model; settings=settings)
```

## Tolerances

`Tolerances{T}` stores optional primal, dual, and gap targets in arithmetic `T`.
An omitted field requests the deterministic automatic target. Explicit values
must be finite and strictly positive.

Tolerance values do not bypass certification. Nonfinite data, overflow-hidden
residuals, or invalid tolerances fail closed.

## Limits

`Limits` contains:

| Field | Meaning |
|---|---|
| `iterations` | maximum outer iterations; `0`/omitted selects the automatic default |
| `time` | solve-phase wall-clock limit in seconds; `Inf` means unlimited |
| `threads` | maximum Julia threads requested by one solve |

A time or iteration limit produces an exhaustion status unless an independent
terminal certificate has already passed.

## Settings fields

| Field | Accepted public values | Purpose |
|---|---|---|
| `engine` | `:auto`, `:native_hsd` | native product HSD only; `:legacy` is rejected |
| `algorithm` | `:auto` | diagnostic label; family selectors are rejected |
| `kkt_route` | `:bordered`, `:expanded`, `:sparse_schur` | implementation of the frozen Newton system |
| `scaling` | `:auto`, `:none`, `:equilibrate` | scaling/equilibration policy |
| `equilibration` keyword | `:off`, `:ruiz` | public view of reversible Ruiz equilibration |
| `formulation` | `:auto`, `:variable_space_schur`, `:dense_augmented_kkt`, `:psd_lift` | formulation policy; `:psd_lift` is explicit, never hidden |
| `provider` | `:auto`, `:standard`, `:bfla`, `:multifloat`, `:legacy` | dense LA provider namespace; `:legacy` here is not a solver engine |
| `presolve` | `:auto`, `:on`, `:off` | structural presolve |
| `sparse` | `:auto`, `:on`, `:off` | sparse storage preference |
| `equality_solver` | `:auto`, `:normal_equations`, `:qr` | equality policy |
| `working_precision_policy` | `:auto`, `:fixed` | precision policy without arithmetic narrowing |
| `diagnostics` | `:none`, `:summary`, `:full` | retained diagnostic detail |
| `verbosity` | nonnegative integer | textual output level |
| `timing` | `Bool` | retain phase timings |
| `certification` | `Bool` | run/retain certificate verification |
| `blas_threads` | `nothing` or positive integer | requested BLAS thread metadata |

The provider value `:legacy` denotes the bundled LA-backend compatibility
namespace. It cannot select the removed public legacy solver.

The typed constructor records the complete policy namespace for qualified
compatibility integrations, but the direct native product-HSD route currently
executes only the following settings: `formulation=:auto`,
`provider=:auto`/`:standard`, `presolve=:auto`/`:off`, `sparse=:auto`/`:off`,
and `equality_solver=:auto`/`:qr`. `blas_threads` must remain `nothing` because
the native route is serial. Other combinations are rejected before numerical
setup with a structured unsupported-policy error; they do not silently fall
back to another engine. `provider=:auto` selects the arithmetic-matched
provider when its optional extension is loaded.

## KKT route policy

`:bordered` is the conservative default. `:expanded` and `:sparse_schur` are
expert implementation choices for the same five-equation `NewtonSystem`.

An explicit route request fails if its structural, provider, arithmetic, or
memory contract is unavailable. Authorized fallback occurs on the same iterate
and is recorded; it cannot silently change arithmetic or solver engine.

## Scaling and reconstruction

Ruiz equilibration is reversible. Model data, warm starts, returned values, and
certificates remain in original coordinates. Equilibration is currently
opt-in while physical benchmark regressions remain under evaluation.

## Precision

The model element type selects Float64, MultiFloat, or BigFloat arithmetic.
`working_precision_policy` may control supported staged precision behavior, but
a missing provider never triggers silent Float64 execution.

BigFloat input conversion cannot recover digits already lost before model
construction. Use strings or exact typed values when loading high-precision
coefficients.

## Thread ownership

`Limits.threads` and `blas_threads` are requests consumed by the pipeline
thread budget. Julia outer workers, BLAS, and provider threads must not all own
the same parallel region.

## Diagnostics and outputs

`Settings` controls solve policy. `Outputs` controls retained result data. Use
result accessors rather than depending on internal field layouts:

```julia
status(result)
primal_objective(result)
value(result, x)
certificate(result)
iteration_history(result)
```

Detailed execution metadata explains planning, provider selection, fallback,
and timing. It cannot override the original-coordinate certificate.

## Low-level compatibility options

Qualified records such as `SolverOptions` remain implementation compatibility
surfaces while legacy source is retired. They are not additional public solve
engines. New applications should use `Settings`, `Tolerances`, and `Limits`.

See [Adaptive predictor-corrector policy](adaptive-parameter-policy.md) for the
shared controller and [Architecture](architecture.md) for route ownership.
