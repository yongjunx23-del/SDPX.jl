# SDPX examples

These three self-checking examples use only the public v0.5 API and run in the
test suite.

## Setup

From the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## Examples

| File | Purpose |
| --- | --- |
| [`moment_lp.jl`](moment_lp.jl) | Finite-grid moment LP with minimization and maximization objectives |
| [`l2_integral_socp.jl`](l2_integral_socp.jl) | Native Lorentz-cone L2 integral bound |
| [`quartic_integral_sdp.jl`](quartic_integral_sdp.jl) | Native PSD quartic moment bounds with arithmetic and order options |

```bash
julia --project=examples examples/moment_lp.jl 17
julia --project=examples examples/l2_integral_socp.jl 16
julia --project=examples examples/quartic_integral_sdp.jl --order 8 --bound both
```

Each example runs a `Float64` problem and a smaller `BigFloat` smoke check.
The quartic example also supports optional `MultiFloats` arithmetic; run it
with `--help` for the available flags.

The JSON command-line bridge and fixtures are documented in
[`docs/src/cli.md`](../docs/src/cli.md).
