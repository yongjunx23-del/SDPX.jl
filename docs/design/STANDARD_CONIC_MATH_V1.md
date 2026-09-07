# Standard conic mathematics — version1

This is a mathematical contract, not a handoff or a production qualification.
The old public HSD/Exp algorithms have **not** been silently switched. Their
historical receipts and CSDR fingerprint remain versioned legacy evidence.
The new exponential primitives are internal migration targets; the bounded
symmetric reference is only a validation oracle.

## Standard embedding and residual homotopy

For `min c'x`, `Ax+s=b`, `s∈K`, the dual is `max -b'y`,
`A'y+c=0`, `y∈K*`. With `u=(x,y,τ)`, `v=(0,s,κ)`, use

```
Q = [ 0    A'    c
     -A     0    b
     -c'   -b'   0 ],       Qu=v.
```

The residuals are `rp=Ax+s-bτ`, `rd=A'y+cτ`,
`rg=c'x+b'y+κ`. They satisfy the exact identity

```
s'y + τκ = -x'rd + y'rp + τrg.
```

There is no strictly interior exact homogeneous solution. Interior iterations
contract nonzero residuals: a direction solving the three affine residual
rows gives `r(z+αd)=(1-α)r(z)`. Cone and scalar complementarity supply the
remaining Newton rows. This is an infeasible-start residual path, not a claim
that `Qu=v` holds at every interior iterate.

With `ds+Θdy=h`, and right sides `p,d,g,t` for primal, dual, gap and scalar
rows respectively, solve

```
K = [0 A'; A -Θ]
w = K⁻¹[d; p-h],       u = K⁻¹[-c; b]
ηw = c'wx+b'wy,         ηu = c'ux+b'uy
D = κ-τηu
Δτ = (t-τg+τηw)/D
Δx = wx+uxΔτ,          Δy = wy+uyΔτ
Δs = p-AΔx+bΔτ,        Δκ = g-c'Δx-b'Δy.
```

For an invertible core and positive definite Θ,
`A'uy=-c`, `Aux-Θuy=b`, hence **`ηu=-uy'Θuy`** and
**`D=κ+τuy'Θuy>0`** for positive τ,κ. This explains why the former
`κ+τηu` closure is not an interchangeable sign convention. Singular cores,
zero-cone rows and nonsymmetric operators require their own analysis; this
positive-definiteness argument must not be generalized without its hypotheses.

`validation/scientific_core/StandardConicMath.jl` assembles the full five-row
Newton system independently and compares its solution against this condensation.
The symmetric reference reuses cone geometry, but not production HSD residuals,
closure, factor caches or termination. Its Newton dimension is capped at256.
It is not an independent cone implementation or a performance implementation.

## A global proof for the logarithmic Exp barrier

On `y>0,z>0, ψ=y log(z/y)-x>0`, use

```
F(x,y,z) = -log ψ -log y -log z.
```

This is not the former exp-gap barrier. For any fixed interior point, a positive
scaling followed by the linear cone automorphism
`(x,y,z)↦(x+t y,y,exp(t)z)` takes it to `(-δ,1,1)`, where
`δ=ψ/y>0`. These transformations change F only by a constant, so derivatives
and the self-concordance inequality transform covariantly.

At that normalized point, for direction `h`, define

```
a = -hx-hy+hz,  b=hy,  c=hz,
u = a/δ,        v=(b-c)/sqrt(δ).
```

Direct differentiation gives

```
D²F[h,h]   = u²+v²+b²+c²,
-D³F[h³]   = 2u³+3uv²+(b+2c)v²+2b³+2c³,
DF[h]      = -u-b-c.
```

Now construct the real symmetric matrix

```
M = [u          v/sqrt(6)  v/sqrt(3)
     v/sqrt(6)  b          0
     v/sqrt(3)  0          c].
```

Then `D²F[h,h]=tr(M²)` and `-D³F[h³]=2tr(M³)`. Its real eigenvalues imply

```
|tr(M³)| ≤ sum |λi|³ ≤ (sum λi²)^(3/2),
```

which proves the global self-concordance inequality. The Hessian is positive
definite because the quadratic form vanishes only for `h=0`.
Cauchy–Schwarz gives `(DF[h])²≤3D²F[h,h]`. Finally,
`F(αs)=F(s)-3log α`, and F diverges at any finite boundary point of the
interior domain. Thus this is a degree3 logarithmically homogeneous
self-concordant barrier, with an explicit proof rather than only derivative
or sampling agreement.

`src/cones/exp_logarithmic.jl` implements value, gradient, Hessian and third
contraction. Outputs use owned scalar stores. The caller supplies the arithmetic
precision context, as for the existing cone primitives. Nonfinite results and
unrepresentable positive Hessian diagonals fail closed. This does not promise
uniform numerical stability for every extreme dynamic range.

The tests compare scalarized kernels with a separate matrix-calculus reference,
finite differences, logarithmic homogeneity, the trace proof identities, mutable
ownership and representative dynamic-range cases. The old exp-gap barrier's
self-concordance counterexample remains a negative control during migration.

**Still required before switching the public Exp path:** derive its actual
Fenchel conjugate and domain, dual inversion, initialization, scaling and
corrector consistently. A cone isomorphism is not a Fenchel-conjugacy proof.

## Remaining integration gates

- Migrate every production HSD residual/closure/Newton/line-search/certificate
  consumer together, including nonsymmetric and specialized routes. A one-line
  scalar sign change is not sufficient.
- Keep original-coordinate primal/dual/objective and ray checks authoritative.
- Preserve and diagnose the reference PSD NT rejections; do not increase
  arithmetic tolerances to make them disappear.
- Establish a separately named standard-algorithm CSDR baseline only after
  original-coordinate certification; retain the old101-iteration receipt.
- Only after local gates pass, run bounded PBS qualification and N14 comparisons.

Background: [SCS embedding, §2.3](https://web.stanford.edu/~boyd/papers/pdf/scs.pdf),
[Clarabel exponential cone](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/src/cones/coneops_expcone.jl).
The proof above is explicit and does not depend on those implementations' tests.
