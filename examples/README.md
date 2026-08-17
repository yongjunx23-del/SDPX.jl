# SDPX worked examples

These examples use only the public `Model`/`optimize!` API. Each script builds
its model from first principles, solves it, reconstructs the relevant original
quantities, and rejects an invalid result certificate.

## Setup

From the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## Quartic integral SDP

[`quartic_integral_sdp.jl`](quartic_integral_sdp.jl) bounds the normalized
second moment of

```math
\rho(x)\propto\exp(-x^2/2-gx^4/4).
```

With `W_k=⟨x^k⟩`, normalization and integration by parts impose

```math
W_0=1,\qquad (2n+1)W_{2n}-W_{2n+2}-gW_{2n+4}=0.
```

The decision vector contains `W₀,W₂,…,W₂ₙ`. The script assembles a symmetric
Hankel matrix of affine expressions and enters it with

```julia
constraint!(model, :hankel_psd, hankel, PSDCone())
```

It solves independent minimization and maximization models for `W₂`, then
reconstructs the numerical Hankel matrix, computes its eigenvalues with
`GenericLinearAlgebra`, and reports certificate residuals.

```bash
# Float64, both bounds
julia --project=examples examples/quartic_integral_sdp.jl \
  --g 1 --order 8 --bound both --arithmetic f64

# 256-bit and 512-bit BigFloat
julia --project=examples examples/quartic_integral_sdp.jl \
  --order 8 --arithmetic bf256
julia --project=examples examples/quartic_integral_sdp.jl \
  --order 8 --arithmetic bf512

# Fixed-width extended precision
julia --project=examples examples/quartic_integral_sdp.jl \
  --order 8 --arithmetic f64x4

# Eight SDPX workers and one BLAS thread
julia -t 8 --project=examples examples/quartic_integral_sdp.jl \
  --order 8 --threads 8 --blas-threads 1
```

Useful flags:

- `--bound lower|upper|both`
- `--g VALUE`
- `--order N` with `N ≥ 2`
- `--arithmetic f64|f64x2|f64x4|bf256|bf512`
- `--threads N`, `--blas-threads N`, `--max-iterations N`

The returned lower/upper records contain `moments`, `hankel`, `spectrum`,
`certificate`, and the public `result`.

## Finite-grid moment LP

[`moment_lp.jl`](moment_lp.jl) optimizes a probability distribution on an
odd uniform grid containing `0`, `1/2`, and `1`:

```math
p_i\ge0,\qquad \sum_i p_i=1,\qquad
\sum_i t_i p_i=\frac12.
```

The objective is the second moment `Σtᵢ²pᵢ`. The minimum is `1/4` (mass at the
midpoint) and the maximum is `1/2` (equal mass at the endpoints). The example
solves both senses, checks these values, and reports mass and mean residuals.

```bash
julia --project=examples examples/moment_lp.jl 17
```

The script also runs a smaller `BigFloat(256)` model. Its result records expose
the optimized weights, objective, residuals, and certificate.

## Native Lorentz SOCP

[`l2_integral_socp.jl`](l2_integral_socp.jl) maximizes a centered linear
functional over

```math
\sum_i u_i=0,\qquad \lVert u\rVert_2\le1.
```

The Euclidean ball is entered directly as one Lorentz cone:

```julia
constraint!(model, :l2_ball, tuple(one(T), u...), LorentzCone())
```

After solving, the example checks the closed-form finite-grid value and
reports both the equality residual and the cone margin `1-‖u‖₂`.

```bash
julia --project=examples examples/l2_integral_socp.jl 16
```

It also runs a smaller `BigFloat(256)` smoke model.

## Reading results

All three examples use the same public result boundary:

```julia
status(result)
primal_objective(result)
dual_objective(result)
certificate(result)
execution_plan(result)
```

Variable values are retrieved with `value(result, ref)`. For an affine PSD
constraint, rebuild the numerical matrix from those values and inspect
`eigvals(Symmetric(matrix))`. High-precision spectra require a generic
eigensolver; the examples environment includes `GenericLinearAlgebra`.

The JSON bridge is documented in [`docs/src/cli.md`](../docs/src/cli.md).
