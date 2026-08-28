# Lorentz and rotated-Lorentz cones

SOC and RSOC constraints are native product-cone blocks in the same HSD engine
used by LP, PSD, exponential, and power models. They are not dispatched to a
separate SOC solver.

## Direct model example

```julia
using SDPX

# Minimize t subject to (t, x, y) in Q3 and (x,y)=(3,4).
model = Model(Float64)
q = variable!(model, :q, 3; domain=Reals())
constraint!(model, :lorentz, (q[1], q[2], q[3]), LorentzCone())
constraint!(model, :tail, (q[2] - 3, q[3] - 4), ZeroCone())
objective!(model, Minimize(), q[1])

result = optimize!(model; settings=Settings(model; verbosity=0))
status(result)                  # :optimal
primal_objective(result)       # approximately 5
certificate(result).valid      # true
```

Affine Lorentz expressions are stored as one canonical cone block. Reals and
ZeroCone coordinates may appear in the same model, as may other supported cone
blocks; the product runtime preserves their block boundaries.

## Rotated Lorentz cones

RSOC uses the exact linear isomorphism

```text
(u, v, w) in Qr  <=>  (u+v, u-v, sqrt(2)w) in Q.
```

The transform is part of canonicalization. Primal values, dual values, slacks,
and rays are mapped back to the original rotated coordinates before a terminal
status is certified. The transform does not introduce a PSD lift.

## Cone algebra

For `x=(t,u)` and `z=(s,v)`, SDPX uses

```text
<x,z>       = t*s + dot(u,v)
det(x)      = t^2 - dot(u,u)
margin(x)   = t - norm(u)
x o z       = (t*s + dot(u,v), t*v + s*u)
```

Boundary steps use scale-normalized quadratic roots followed by a strict
interior check on the actual trial point. Lorentz blocks contribute their
barrier degree to the shared product-cone complementarity measure.

## Scaling and Newton equations

Lorentz blocks provide local Nesterov--Todd metric operations to the common
`NewtonSystem`. Predictor and corrector directions are verified against the
same five HSD equations as every other cone family. Cone-local specialization
may change how a metric or local solve is evaluated, but it cannot change the
HSD residuals or terminal certificate.

## Q3 and fixed-trace specialization

For

```math
X=\begin{bmatrix}a&b\\b&c\end{bmatrix},\qquad a+c=\tau,
```

PSD feasibility is exactly equivalent to

```math
(\tau,\ a-c,\ 2b)\in\mathcal Q_3.
```

SDPX may use this identity for a verified fixed-trace/Q3 structural
specialization. The specialization is not a second solver: it shares the
canonical problem, product-HSD state, Newton equations, line search, provider
policy, reconstruction, and certificate.

Promotion requires the exact structural conditions expected by the local
kernel. If they are not proved, execution remains on the general block path.
No automatic family conversion or hidden fallback is allowed.

## Sparse and high-precision execution

Sparse Schur assembly records Lorentz block incidence and reuses a frozen CSC
pattern when eligible. Float64, MultiFloat, and BigFloat retain identical cone
semantics. Provider selection may not narrow arithmetic.

BigFloat scratch values must have independent MPFR ownership. Threaded block
work writes disjoint destinations and follows the pipeline thread budget.

## Certification

The final verifier recomputes in original model coordinates:

- affine and equality residuals;
- primal and dual Lorentz margins;
- objectives and relative gap;
- complementarity; and
- normalized infeasibility rays when applicable.

MOI getters for rotated cones return rotated coordinates, not the internal
standard-Lorentz image. A provider or factor success cannot promote a status
when the independent certificate fails.
