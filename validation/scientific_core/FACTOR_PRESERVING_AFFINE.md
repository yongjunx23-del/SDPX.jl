# Factor-preserving affine epoch — unpromoted research

This is an explicit dual-Hessian one-secant representation experiment. It is
not selected by SDPX, does not update iterates, and always reports
production_admitted=false. Strict double-secant, default routes, correctors,
regularization and all old production acceptance flags remain unchanged.

## Input authority

The two fixtures preserve actual trial17/19 values from capture
1fac09c8d7ec310532596add88c5daa1ae78d00d96dd46c8955fd627a39a573b:
A/b/c, trial x/s/y/tau/kappa, and the mu passed to the first attempted power
scaling. The base/runtime mu is stale at that failure point and is not used.
The documented builder supplies three orthant rows followed by three half-Power
blocks, with A12x3 and six stored nonzeros. The method is structurally bounded
(n<=16,m<=32), never selected by a model name. All coefficient words and source
record identities are retained.

For each Power block, the reviewed root qualifier reconstructs NEW scratch at
the actual trial dual using its captured warm gap/settings. The actual stored
L is copied. The old production root and all native reconstruction/factor/
Cartesian/inverse results remain in root_reports. In particular, trial19's third
native inverse publication fails. This is visible and prevents any production
admission; forming an unpromoted research direction is not a waiver of that gate.

## Same formula, distinct stored representation

Let HL=L L', C=sqrt(mu)L^-T, a=C'y, b=C^-1 s, p=s'y. The unregularized
L-defined one-secant BFGS metric is

  ThetaIdeal = mu HL^-1 - mu (HL^-1 y)(HL^-1 y)'/(y'HL^-1 y) + ss'/p
             = C M C',
  M = I - aa'/(a'a) + bb'/p.

The existing native current-primal domain gate remains mandatory. Additional
outward checks prove current half-Power primal interior and positive exact s'y;
no positive rounded dot product alone supplies that proof.
For positive p and invertible L, M is positive definite: on a-perpendicular
vectors the projector is identity, and the b direction has nonzero coupling
because a'b=p. No rank reduction is involved. The Float64 candidate constructs
M with the displayed arithmetic, Cholesky-factorizes it without shift, and
retains R with M approximately R R'. Failure refuses—there is no fallback,
clipping, regularization or precision change. Root L, R and stored scale are
kept separately; the rounded M is diagnostic, not the physical operator.

The actual declared operator is S S' with S=scale L^-T R, and its inverse is
W'W with W=R^-1 L'/scale. Four native maps have the corresponding exact adjoints:

  Sv  = scale * solve(L', Rv)
  S'v = scale * R' solve(L,v)
  Wv  = solve(R,L'v)/scale
  W'v = L solve(R',v)/scale.

No S,W,HL,physicalTheta or physicalG is materialized as numerical authority.
This is not equality to the old rounded mu*inverse_hessian construction. Exact
references separately compare the resulting operator with L-defined BFGS and
with BFGS using the true Hessian at the stored shadow.

## Full affine Newton boundary

With dsHat=W ds, dyHat=S' dy, Ahat=W A, bhat=W b and hHat=W h, h=-s,
eliminate only dsHat=hHat-dyHat. Solve the full-size bordered matrix

  [ 0       Ahat'   c       0 ]
  [ Ahat    -I      -bhat   0 ]
  [ c'      bhat'   0       1 ]
  [ 0       0       kappa   tau ]

against [rD; W rP-W h; rG; rTK]. There are n+m+2 unknowns; no Ahat'Ahat
normal equations, numerical rank truncation, or ridge is formed. The same stored
Ahat and its transpose populate both core blocks. Existing HSDNewtonRHS signs
and newton_residual! supply the physical equation authority via an explicit
FactorCone subtype, never a Theta-materializing contribution call. Recover
raw dy=W' dyHat and ds=S dsHat; do not overwrite them with roundtrip values.

The epoch owns inputs, factors, transformed coefficients and factorization.
Exact immutable word tuples detect factor/point/scale/A/core/pivot drift before
reuse. Returned RHS snapshots are independent of caller RHS storage. Only affine
h=-s is supported; combined-corrector input is rejected. Explicit other RHS values
may reuse the same frozen factorization.

## Verification and boundaries

Exact rational verification uses actual stored inputs, L/R/scale, transformed
coefficients and recovered directions. All four maps are checked on basis vectors,
actual A columns, b/s/y, RHS and recovered directions. Original five residual
groups use physical A/b/c and the same factor-defined Theta; their work is the
sum of absolute terms in each original equation, including actual Theta entries
only in the reference. The unchanged Float64 Newton ceiling is2^-17=512sqrt(eps).
No enlarged composed-factor work denominator substitutes for that definition.

Exact congruences W ThetaIdeal W'-I check the BFGS metric, including square-root
scale rounding. Comparisons to true-Hessian-based BFGS are separate. Trial17 fits
the2^-22 research metric ceiling. Trial19's third block exceeds it: its squared
Frobenius error exceeds3*(2^-22)^2, proving spectral error above that ceiling,
not merely a loose upper-bound failure. Therefore physical-equation success is
NOT sufficient to promote trial19's geometry.

This slice is reference-verified research, not complete runtime certification.
Native outward action/metric/physical-residual bounds, the remaining old matrix-
scaling gates in separate diagnostic scratch, corrector coverage and integration
into solver consumers are still pending. Original production root/inverse failures
are preserved. No public validity, line-search success, general-alpha Power,
mixed-cone liveness, sparse-memory, performance or R0 closure is claimed.
