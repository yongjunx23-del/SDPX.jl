# BigFloat256 ownership and nonsymmetric fallback receipt (2026-09-04)

## Source identity

- SDPX commit: `40b6380aa5e148b72db4eab67fa16d5f40296bc3`
- Branch/remote: `main`, `origin/main`
- Julia: 1.12.6
- Execution policy: `--gcthreads=1 --threads=1`
- BigFloat precision: 256 bits
- No tolerance widening, precision downgrade, model-name dispatch, or certificate bypass was used.

## Defect and repair

Two independent mutable-storage failures remained after the first HSD ownership pass:

1. one-hot RHS and symmetric inverse workspaces could retain paired `BigFloat` objects from earlier shallow publication, so an owned in-place store to one coordinate changed another coordinate;
2. averaging independently rounded off-diagonal inverse-column solves can invalidate the most sensitive column near an exponential-cone face even when one exact symmetric upper/lower selection passes the unchanged componentwise solve and SPD certificates.

The repair rebuilds potentially aliased storage with `zero_distinct!`, uses owned stores/copies across HSD and nonsymmetric workspaces, preserves the historical midpoint for fixed-width arithmetic, and permits BigFloat-only bounded enumeration of the eight exact symmetric upper/lower selections. Every accepted candidate must pass the existing inverse-column and SPD/factor certificates.

A regression point that previously returned `NS_CONJUGATE_INVERSE_HESSIAN_FAILED` now returns:

- status: `NS_SCALING_DUAL_HESSIAN_FALLBACK`
- reason: `NS_SCALING_CONVERGED`
- recorded primary reason: `NS_SCALING_BFGS_DENOMINATOR`
- fallback secant error: `3.5903040394419528553e-78`
- fallback inverse error: `9.9178645213593524290e-78`

## BigFloat256 eight-family matrix

Command:

```bash
JULIA_NUM_THREADS=1 SDPX_PRECISION=BigFloat256 \
julia --startup-file=no --gcthreads=1 --threads=1 \
  --project=/Users/xuyongjun/Desktop/project/SDPX/CSDR/reproduce_env \
  benchmark/general/precision_matrix.jl
```

| Case | Status | Certificate | Iterations | Original primal residual | Original dual residual | Relative gap |
|---|---:|---:|---:|---:|---:|---:|
| `lp_afiro_style` | optimal | true | 34 | `2.764e-77` | `3.775e-34` | `9.257e-34` |
| `socp_portfolio_small` | optimal | true | 33 | `0` | `9.961e-33` | `5.757e-78` |
| `socp_ill_scaled_small` | optimal | true | 35 | `0` | `1.026e-33` | `0` |
| `rsoc_epigraph_small` | optimal | true | 95 | `0` | `9.141e-73` | `7.969e-34` |
| `sdp_maxcut_k4` | optimal | true | 42 | `0` | `2.211e-75` | `0` |
| `exp_unit_small` | optimal | true | 39 | `0` | `0` | `0` |
| `power_epigraph_small` | optimal | true | 91 | `1.162e-33` | `5.532e-34` | `6.256e-34` |
| `mixed_orthant_exp_small` | optimal | true | 86 | `0` | `4.956e-33` | `0` |

Result: **8/8 pass**.

## Allocation and memory gates

- Float64 warmed HSD steps: ten `HSDStepOK`, `0` bytes each.
- Float64x4 warmed HSD steps: ten `HSDStepOK`, `0` bytes each.
- BigFloat256 warmed-step allocation samples:
  `[6896848, 6897072, 6899760, 6895056, 6898192, 6897072, 6897408, 6897520, 6897632, 6897408]`.
- BigFloat256 60-step RSS audit: seven checkpoints identical; measured drift `0` bytes.

## Validation gates

- Complete `test/runtests.jl`: exit 0.
- `validation/providers/multifloat_linear_algebra_integration.jl`: 168/168.
- `benchmark/robustness/test_route_guard.jl`: 43/43.

## PackageCompiler incident and safe profile

The legacy full execution workload on immutable release `fe5ac58...`, PBS job `209911.node220`, emitted `GC error (probable corruption)` during PackageCompiler precompile execution and recursively printed `Core.CodeInstance`, producing a 32 GB log. This is recorded as provider/runtime evidence, not a solver status. The job was cancelled and the unbounded log removed after retaining a bounded diagnostic head and TOML receipt.

Commit `e8cc367` replaces that build route with `safe_float64` only: no BigFloat/MultiFloat provider state, no dynamic provider `@eval`, generic CPU target by default, and a one-GiB image-size bound. Extended-precision methods JIT in the consuming process.

## N14 pilot status

The post-`@noinline` N14 SOC pilot on immutable release `fe5ac58...`, PBS job `209931.node220`, completed all 500 solver iterations without the earlier LLVM compile crash, but returned `iteration_limit` with no certificate. It then encountered a glibc/Julia GC abort at the driver's explicit post-solve `GC.gc()`. The process abort is provider/runtime evidence and does not alter the solver result. N14 remains unqualified until an optimal original-coordinate certificate is produced.
