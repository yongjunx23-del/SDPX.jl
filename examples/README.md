# Learning SDPX through a quartic bootstrap problem

This tutorial develops one mathematical problem in three coordinate systems:

- a moment-space semidefinite program (SDP);
- a spectrum-space linear program (LP);
- an exact Lorentz-cone form of the smallest `2×2` SDP (SOCP).

The three executable files are:

```text
examples/
├── quartic_bootstrap_sdp.jl
├── quartic_discrete_lp.jl
└── quartic_2x2_socp.jl
```

Set up their environment once from the repository root:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## 1. The quartic integral

For `g>0`, define

```math
Z(g)=
\int_{-\infty}^{\infty}
\exp\!\left(-\frac{x^2}{2}-\frac{g x^4}{4}\right)\,dx
```

and the normalized moments

```math
W_n(g)=\frac{1}{Z(g)}
\int_{-\infty}^{\infty}
x^n\exp\!\left(-\frac{x^2}{2}-\frac{g x^4}{4}\right)\,dx.
```

We will not evaluate this integral directly. The target is to bound

```math
W_2=\langle x^2\rangle
```

using exact identities and positivity. For `g=1`, the independently known
numerical value

```text
W2 ≈ 0.4679199169736651886
```

is used only to judge convergence. It is never supplied to an optimization
model.

## 2. Replace integration by a convex search

The density is even, so every odd moment vanishes:

```math
W_{2n+1}=0.
```

For the even moments, integrate a total derivative:

```math
0=\int_{-\infty}^{\infty}
\frac{d}{dx}\left[x^{2n+1}e^{-V(x)}\right]dx,
\qquad
V(x)=\frac{x^2}{2}+\frac{g x^4}{4}.
```

The boundary term is zero and the remaining terms give the exact linear
recurrence

```math
(2n+1)W_{2n}-W_{2n+2}-gW_{2n+4}=0,
\qquad n=0,1,\ldots.
```

Positivity supplies the convex constraint. For every polynomial

```math
p(x)=\sum_{i=0}^{K}c_i x^i,
```

we have

```math
\langle p(x)^2\rangle
=c^{\mathsf T}M_Kc\ge0,
\qquad
(M_K)_{ij}=W_{i+j}.
```

Therefore the moment matrix is positive semidefinite:

```math
M_K\succeq0.
```

The bootstrap idea is now simple:

> Replace direct integration by a convex search over all moment sequences
> consistent with exact identities and positivity.

At finite order, solve

```math
L_K(g)=\min W_2,
\qquad
U_K(g)=\max W_2
```

subject to `W0=1`, every recurrence available within the truncation, and the
retained moment positivity conditions. The true integral is feasible, hence

```math
L_K(g)\le W_2(g)\le U_K(g).
```

Increasing `K` shrinks the allowed moment region and can tighten the interval.

### Parity and the Stieltjes moment matrices

It is useful to write `lambda=x^2>=0`. The even moments are moments of a
positive measure `nu` on the half-line:

```math
W_{2n}=\int_0^\infty \lambda^n\,d\nu(\lambda).
```

The full moment matrix separates into even and odd parity sectors. In terms of
`lambda`, these are the two Stieltjes families

```math
H^{(0)}_{ij}=W_{2(i+j)},
\qquad
H^{(1)}_{ij}=W_{2(i+j+1)}.
```

The SDP example stores them as one parity-aware matrix: entries with odd
`i+j` are zero, so a simultaneous row/column permutation exposes exactly the
`H^(0)` and `H^(1)` blocks.

## 3. Start with an exact `2×2` SDP

For `g=1`, keep only `W0`, `W2`, and `W4`. Normalization and the first
recurrence give

```math
W_0=1,
\qquad
W_4=1-W_2.
```

The first moment matrix is

```math
\begin{bmatrix}W_0&W_2\\W_2&W_4\end{bmatrix}\succeq0.
```

Its determinant must be nonnegative, so

```math
1-W_2-W_2^2\ge0.
```

Consequently the exact maximum is

```math
W_2^{\max}=\frac{\sqrt5-1}{2}
\approx0.61803398875.
```

This is a small but complete test of modeling, solving, matrix reconstruction,
and certification.

```julia
using SDPX
using LinearAlgebra

model = Model(Float64)
w = variable!(model, :w, 3; domain=Reals()) # [W0, W2, W4]

normalization = constraint!(model, :normalization, w[1] - 1, ZeroCone())
recurrence = constraint!(model, :recurrence, w[1] - w[2] - w[3], ZeroCone())
moment_matrix = constraint!(
    model,
    :moment_matrix,
    [w[1] w[2]; w[2] w[3]],
    PSDCone(),
)
objective!(model, Maximize(), w[2])
```

`Model(Float64)` fixes the arithmetic type. `variable!` creates the three
free scalar moments. An affine expression in `ZeroCone()` is a linear
equality; a symmetric matrix of affine expressions in `PSDCone()` is one PSD
constraint. `objective!` fixes the objective sense and expression.

The mathematical model and the numerical execution policy are separate:

```julia
settings = Settings(
    model;
    algorithm=:sdp,
    formulation=:auto,
    provider=:auto,
    sparse=:auto,
    scaling=:auto,
    limits=Limits(iterations=200, time=60.0, threads=1),
    verbosity=0,
    timing=true,
    diagnostics=:summary,
    certification=true,
)

outputs = Outputs(
    :all,
    :all,
    :all;
    objectives=true,
    certificate=:summary,
    diagnostics=:summary,
    trace=true,
)

result = optimize!(model; settings=settings, outputs=outputs)
```

The `Model` defines what is optimized. `Settings` define how SDPX executes
that model. `Outputs` define which result data are retained.

Automatic stopping targets are appropriate for this example. To request
explicit targets, construct a typed policy and pass it into `Settings`:

```julia
tolerances = Tolerances(
    Float64;
    primal=1e-8,
    dual=1e-8,
    gap=1e-8,
)
settings = Settings(model; algorithm=:sdp, tolerances=tolerances)
```

### Values, matrix spectrum, duals, and certification

```julia
moments = value(result, w)
H = [moments[1] moments[2]; moments[2] moments[3]]

status(result)                    # :optimal
moments                           # approximately [1, 0.618034, 0.381966]
primal_objective(result)          # approximately 0.618034
dual_objective(result)            # approximately the same value
eigvals(Symmetric(H))             # nonnegative up to solver tolerance

dual(result, moment_matrix)       # dual PSD block
dual_slack(result, w)             # dual slack for the moment variables
```

Certification evaluates the returned point in the original model
coordinates:

```julia
cert = certificate(result)
cert.valid
cert.primal_residual
cert.dual_residual
cert.relative_gap
```

Solver status and certification answer different questions: status describes
how numerical execution ended, while `cert.valid` checks whether the retained
solution meets the requested original-coordinate conditions.

The public provenance and timing accessors are:

```julia
plan = execution_plan(result)
diag = diagnostics(result)
trace = performance_trace(result)
```

`execution_plan(result)` is the immutable plan that resolved formulation,
arithmetic, storage, and linear algebra before numerical execution. Detailed
architecture information belongs in the main
[diagnostics documentation](../docs/src/diagnostics.md); the example only
makes the plan visible.

## 4. Extend the same model to the quartic bootstrap SDP

At order `K`, store

```math
w=[W_0,W_2,\ldots,W_{2K}].
```

The core of
[`quartic_bootstrap_sdp.jl`](quartic_bootstrap_sdp.jl) follows the equations
directly:

```julia
model = Model(T)
w = variable!(model, :w, order + 1; domain=Reals())

constraint!(model, :normalization, w[1] - one(T), ZeroCone())
for n in 0:(order - 2)
    recurrence = (2n + 1) * w[n + 1] - w[n + 2] - g * w[n + 3]
    constraint!(model, Symbol("recurrence_", n), recurrence, ZeroCone())
end

hankel = [
    iseven(i + j) ? w[(i + j) ÷ 2 + 1] : zero(T)
    for i in 0:order, j in 0:order
]
constraint!(model, :moment_matrix, hankel, PSDCone())
objective!(model, sense, w[2])
```

Run both bounds at `g=1`, order 8:

```bash
julia --project=examples examples/quartic_bootstrap_sdp.jl \
  --g 1 --order 8 --bound both --arithmetic f64
```

Typical Float64 output is

```text
lower W2 ≈ 0.46781157267
upper W2 ≈ 0.46794579859
```

so the known reference lies inside the certified finite-order interval. The
script also reconstructs the full numerical moment matrix, computes its
spectrum, and prints the original-coordinate certificate residuals.

## 5. Arithmetic, spectra, and providers

The arithmetic is a property of the model:

```julia
using MultiFloats

model64 = Model(Float64)
model2 = Model(Float64x2)
model4 = Model(Float64x4)
modelbig = Model(BigFloat; precision_bits=256)
```

- `Float64` is ordinary double precision.
- `Float64x2` is a fixed-width extended type useful for moderately
  ill-conditioned moment equations.
- `Float64x4` provides more fixed-width precision and is often a practical
  bootstrap choice.
- `BigFloat` provides arbitrary precision when a fixed-width type is not
  enough.

BigFloat constants must be created inside the intended precision scope:

```julia
setprecision(BigFloat, 256) do
    model = Model(BigFloat; precision_bits=256)
    g = BigFloat("1.0")
    w = variable!(model, :w, 9; domain=Reals())
    # Build every coefficient and constant in this scope.
end
```

Increasing solver precision later cannot restore digits already rounded out
of the model data.

The SDP script accepts:

```bash
julia --project=examples examples/quartic_bootstrap_sdp.jl --order 8 --arithmetic f64x2
julia --project=examples examples/quartic_bootstrap_sdp.jl --order 8 --arithmetic f64x4
julia --project=examples examples/quartic_bootstrap_sdp.jl --order 8 --arithmetic bf256
julia --project=examples examples/quartic_bootstrap_sdp.jl --order 4 --bound upper --arithmetic bf512
```

The last command is a shorter 512-bit smoke test; raise `--order` when the
extra precision is needed for a larger truncation.

For a high-precision matrix `H`, inspect its spectrum without converting it
to Float64:

```julia
using GenericLinearAlgebra, LinearAlgebra
lambda = eigvals(Symmetric(H))
minimum(lambda)
```

### Recommended extended-precision linear algebra

[MultiFloatLinearAlgebra](https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl)
is recommended for `Float64x2`/`Float64x4` models, and
[BigFloatLinearAlgebra](https://github.com/yongjunx23-del/BigFloatLinearAlgebra.jl)
is recommended for `BigFloat` models. Both packages are optional and currently
installed directly from GitHub. The following pins are the latest reviewed
`main` revisions as of 2026-08-17:

```julia
using Pkg
Pkg.add(PackageSpec(
    url="https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl",
    rev="c0c78b365036a28b9dabb9c31a10cb60370a5bae",
))
Pkg.add(PackageSpec(
    url="https://github.com/yongjunx23-del/BigFloatLinearAlgebra.jl",
    rev="a76093dc43a285632c396c83ed0bb6e78894d987",
))
```

Load the matching package and request its provider explicitly:

```julia
using MultiFloats, MultiFloatLinearAlgebra
model = Model(Float64x4)
settings = Settings(model; algorithm=:sdp, provider=:multifloat)
```

or

```julia
using BigFloatLinearAlgebra
setprecision(BigFloat, 256) do
    model = Model(BigFloat; precision_bits=256)
    settings = Settings(model; algorithm=:sdp, provider=:bfla)
    # Build all BigFloat data and call optimize! in this scope.
end
```

The SDP example performs the imports for explicit provider requests, so the
same recommendation can be exercised directly:

```bash
julia --project=examples examples/quartic_bootstrap_sdp.jl \
  --order 4 --bound upper --arithmetic f64x4 --provider multifloat
julia --project=examples examples/quartic_bootstrap_sdp.jl \
  --order 4 --bound upper --arithmetic bf256 --provider bfla
```

Each run prints the executed provider, the outer `optimize!` wall time, the
solver-recorded core time, the iteration count, the spectrum, and certificate
residuals. These timings describe that one run and include machine- and
compilation-dependent effects; they are diagnostics, not a provider ranking.
`GenericLinearAlgebra` remains a useful generic implementation and supplies
high-precision eigensolvers used by this tutorial. Provider choice changes
linear algebra, not the mathematical optimization problem.

The tutorial fixes `formulation=:variable_space_schur`, `sparse=:off`, and
`presolve=:off` because its equality system is already explicit and its moment
matrix is small and dense. The two specialized providers implement that
high-precision route. Large sparse models should choose preprocessing,
formulation, and storage independently from this example.

### Parallel execution

Start Julia with the desired thread count, then give SDPX no more threads than
Julia owns:

```bash
OPENBLAS_NUM_THREADS=1 julia -t 8 --project=examples \
  examples/quartic_bootstrap_sdp.jl \
  --order 8 --threads 8 --arithmetic f64x4 --provider multifloat
```

`Limits(threads=8)` controls SDPX work scheduling. The process-level BLAS
setting keeps one BLAS thread per SDPX process and avoids accidental nested
oversubscription. Use the equivalent setting for a non-OpenBLAS installation.

## 6. LP: discretize the same positive measure

[`quartic_discrete_lp.jl`](quartic_discrete_lp.jl) works in the spectral
coordinate `lambda=x^2` while keeping the same normalization and exact
recurrences.

The half-line measure formulation is already linear:

```math
\begin{aligned}
\min/\max_{\nu\ge0}\quad&
\int_0^\infty \lambda\,d\nu(\lambda)\\
\text{s.t.}\quad&
\int_0^\infty d\nu=1,\\
&\int_0^\infty q_n(\lambda)\,d\nu(\lambda)=0,
\end{aligned}
```

where

```math
q_n(\lambda)=(2n+1)\lambda^n-\lambda^{n+1}-g\lambda^{n+2}.
```

Approximate the positive measure by atoms at fixed spectral nodes:

```math
d\nu(\lambda)\approx
\sum_{i=1}^{N}p_i\,\delta(\lambda-\lambda_i),
\qquad p_i\ge0.
```

Then

```math
W_{2n}\approx\sum_i p_i\lambda_i^n
```

and the finite problem is the LP

```math
\begin{aligned}
\min/\max\quad&\sum_i p_i\lambda_i\\
\text{s.t.}\quad&p_i\ge0,\\
&\sum_i p_i=1,\\
&\sum_i q_n(\lambda_i)p_i=0.
\end{aligned}
```

It is linear because the nodes are fixed; only the nonnegative masses are
unknown.

### Chebyshev nodes without a hard support cutoff

Use interior Chebyshev points

```math
t_i=\frac12\left[1-\cos\left(\frac{(2i-1)\pi}{2N}\right)\right]
\in(0,1)
```

and the rational compactification

```math
\boxed{\lambda_i=\lambda_{\mathrm{scale}}\frac{t_i}{1-t_i}}.
```

Equivalently, the implementation evaluates

```math
\lambda_i=\lambda_{\mathrm{scale}}
\tan^2\left(\frac{(2i-1)\pi}{4N}\right),
```

which avoids cancellation near `t=1`. The map covers `(0,infinity)` rather
than imposing a finite cutoff. Increasing `N` refines the compactified
spectrum; `lambda_scale` controls where resolution is concentrated. The
example uses `lambda_scale=1` for `g=1`.

High nodes and high moments create a large coefficient range. The executable
uses the positive diagonal change of variables

```math
z_i=(1+\lambda_i^{r+1})p_i,
```

where `r` is the number of retained recurrences and `r+1` is the largest
power appearing in those recurrence polynomials. Replacing `p_i` by
`z_i/(1+lambda_i^(r+1))` preserves the LP exactly and keeps the scaled
high-node coefficients bounded. The returned `z` values are converted back
to physical masses `p` before mass and recurrence residuals are checked.

The corresponding SDPX modeling core is still a literal LP:

```julia
model = Model(T)
z = variable!(model, :scaled_mass, length(nodes); domain=Nonnegative())
q(n, lambda) = lambda^n * ((2n + 1) - lambda - g * lambda^2)
denominators = [one(T) + lambda^(recurrence_count + 1) for lambda in nodes]

mass = sum((one(T) / denominators[i]) * z[i] for i in eachindex(nodes))
constraint!(model, :normalization, mass - one(T), ZeroCone())
for n in 0:(recurrence_count - 1)
    row = sum(
        (q(n, nodes[i]) / denominators[i]) * z[i]
        for i in eachindex(nodes)
    )
    constraint!(model, Symbol("recurrence_", n), row, ZeroCone())
end

w2 = sum((nodes[i] / denominators[i]) * z[i] for i in eachindex(nodes))
objective!(model, sense, w2) # sense is Minimize() or Maximize()
```

Run the seven-recurrence convergence table with fixed-width extended
precision:

```bash
julia --project=examples examples/quartic_discrete_lp.jl \
  --nodes 64,128,256,512,1024 \
  --recurrences 7 --lambda-scale 1 --arithmetic f64x2
```

This command performs ten extended-precision solves and can take several
minutes. Use `--nodes 64,128` for a shorter convergence check.

The current run gives:

| Nodes | LP minimum | LP maximum |
|---:|---:|---:|
| 64 | 0.467816598131 | 0.467944212991 |
| 128 | 0.467812989509 | 0.467945463818 |
| 256 | 0.467811885837 | 0.467945694652 |
| 512 | 0.467811609746 | 0.467945784585 |
| 1024 | 0.467811589469 | 0.467945791885 |

For comparison:

| Quantity | Value |
|:--|--:|
| order-8 SDP lower bound | 0.46781157267 |
| external integral reference | 0.46791991697 |
| order-8 SDP upper bound | 0.46794579859 |

The finite-order SDP and finite discrete LP are not identical.

- The SDP is a **moment-space bootstrap relaxation**: it permits every moment
  sequence satisfying the retained identities and positivity conditions.
- The LP is a **spectrum-space discretization**: it restricts the measure to
  a chosen finite node set.

At the complete infinite-dimensional level, a positive measure and its full
moment/positivity data describe the same object. At finite size, the
approximations differ. For the matched recurrences above, each positive
discrete measure is feasible for the SDP, so the grid interval sits inside
the SDP interval. The displayed sequence approaches that interval as the
spectral grid is refined, but these Chebyshev grids are not nested, so
monotonic convergence is not guaranteed. The LP numbers are grid-dependent
and are not automatically rigorous bounds on the original integral.

For a faster Float64 demonstration with fewer retained identities:

```bash
julia --project=examples examples/quartic_discrete_lp.jl \
  --nodes 64,128,256 --recurrences 5 --arithmetic f64
```

## 7. SOCP: the exact Lorentz form of the `2×2` block

For a symmetric matrix

```math
X=\begin{bmatrix}a&b\\b&c\end{bmatrix},
```

positive semidefiniteness is equivalent to

```math
a+c\ge\sqrt{(a-c)^2+4b^2},
```

or

```math
(a+c,\ a-c,\ 2b)\in\mathcal Q_3.
```

Therefore the small quartic moment constraint has the exact Lorentz form

```math
(W_0+W_4,\ W_0-W_4,\ 2W_2)\in\mathcal Q_3.
```

[`quartic_2x2_socp.jl`](quartic_2x2_socp.jl) changes only the cone constraint:

```julia
using SDPX

model = Model(Float64)
w = variable!(model, :w, 3; domain=Reals())
constraint!(model, :normalization, w[1] - 1, ZeroCone())
constraint!(model, :recurrence, w[1] - w[2] - w[3], ZeroCone())
constraint!(
    model,
    :moment_lorentz,
    (w[1] + w[3], w[1] - w[3], 2 * w[2]),
    LorentzCone(),
)
objective!(model, Maximize(), w[2])
```

Run it with

```bash
julia --project=examples examples/quartic_2x2_socp.jl
```

Both the SDP and SOCP formulations recover

```text
W2 ≈ 0.61803398875 = (sqrt(5)-1)/2.
```

Some `2×2` PSD constraints admit this exact Lorentz representation. When a
model consists of such structure, the native SOCP formulation is usually the
simpler representation; no semidefinite lift is required.

`RotatedLorentzCone()` is also available for models naturally written as
rotated quadratic inequalities. This example uses `LorentzCone()` because the
ordinary `Q3` map above is direct.

## 8. What the three examples teach

The mathematical chain is now continuous:

```text
quartic integral
    -> exact moment recurrences + positivity
    -> moment-space SDP hierarchy
    -> fixed-node positive-measure LP
    -> exact Q3 representation of the first 2x2 PSD block
```

The same public workflow appears throughout:

```text
Model -> variable! / constraint! / objective!
      -> Settings + Outputs
      -> optimize!
      -> status / value / dual / dual_slack
      -> objectives / certificate / execution_plan / diagnostics / trace
```

For API details beyond the tutorial, continue with the
[SDPX documentation](../docs/src/index.md).
