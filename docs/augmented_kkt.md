# Dense augmented KKT formulation

This note fixes the exact Round 3 mathematical contract. It is derived from
SDPX's current Newton code, not from a generic textbook saddle-point matrix.
The route is selected explicitly by `formulation=:augmented`, or by the
Round 4 static formulation planner when `formulation=:auto` finds strong
pre-solve equality risk and the LDLT route is feasible. See
[`Formulation planner`](formulation-planner.md). Sparse, arrow, LP, and native
Q3 routes are unchanged.

## Current Newton equations

For primal vector `x`, equality multiplier `y`, and PSD block pairs
`(X_l,Y_l)`, SDPX forms

```
P_l = A_l(x) - X_l - C_l
d   = c - A*(Y) - B*y
p   = b - B'*x
R_l = target_l*I - X_l*Y_l
```

The Newton unknowns are `dx`, `dy`, and the recovered symmetric block
directions `dX_l`, `dY_l`. Their block equations are

```
dX_l = A_l(dx) + P_l
B'*dx = p
dY_l = sym(X_l^-1 * (R_l - dX_l*Y_l))
A*(dY) + B*dy = d.
```

SDPX's block code applies the final symmetric projection explicitly. These
equations also define the formulation-independent direction-recovery test.

## Dense normal equations

Eliminating `dX_l` and `dY_l`, define

```
Z_l = X_l^-1 * (P_l*Y_l - R_l)
v_i = sum_l <A_i^(l), Z_l>
r   = -(d + v).
```

The Schur matrix assembled by the existing code is

```
S_ij = sum_l tr(Y_l * A_i^(l) * X_l^-1 * A_j^(l)).
```

Equivalently it is the Gram matrix of the transformed coefficient blocks
`vec(L_X^-1*A_i*M_Y)`. Therefore the current reduced equations are

```
S*dx - B*dy = r
B'*dx        = p.
```

The mature dense route Cholesky-factorizes `S`, forms
`Q=B'*S^-1*B`, solves for `dy`, and recovers `dx`. This system is smaller and
positive definite after equality elimination, but the extra normal-equation
elimination can worsen conditioning.

## Dense equality-augmented Schur system

Negating only the equality row gives the symmetric-indefinite system

```
unknown ordering: u = [dx; dy]

K = [ S   -B ]       rhs = [ r ]
    [-B'   0 ]             [-p]

K*u = rhs.
```

This is an **equality-augmented Schur system**, not the much larger full cone
KKT matrix containing explicit `dX_l` and `dY_l` unknowns. Solving it produces
the same `dx` and `dy` as the dense normal-equation elimination. SDPX then
uses the unchanged block recovery equations above, so certification remains
in original problem coordinates and is independent of the formulation.

Only the lower triangle of `K` is authoritative. The bottom-left block stores
`-B'`; the equality diagonal is exactly zero. MFLA or BFLA performs only the
pivoted symmetric LDLT factor and vector/multiple-RHS solve. SDPX retains the
unfactored matrix for the complete equality-augmented **reduced** residual

```
rho = [r; -p] - K*[dx; dy].
```

This residual covers both rows of the reduced saddle-point system. Tests also
check all four implementation-level Newton equations, including the exact
symmetric `dY_l` recovery equation. The reduced residual is not mislabeled as
the larger full cone-variable residual. It maps the equality component back to
the established refinement convention before requesting one correction solve.
Providers do not own stopping, regularization, retry, fallback, precision, or
acceptance policy.

## Regularization and inertia

The existing Cholesky route retries with

```
S_delta = S + delta*diag(max(abs(S_ii),1)),
delta = sqrt(eps(T))*10^(attempt-1).
```

The augmented route uses the identical shift only in the primal `S` block
when its LDLT factorization fails. The equality block remains exactly zero;
adding a dual `-delta*I` would define a different Newton system. Residual and
refinement always evaluate the original unregularized `K`. The retry count
and final `delta` are recorded, and failure never switches formulation,
provider, or precision.

If `S_delta` is positive definite and `rank(B)=q`, the inertia of `K_delta`
is `(m,q,n-q)`: `m` positive, `q` negative, and `n-q` zero eigenvalues. With
full-column-rank `B`, this is `(m,n,0)`. This pattern is recorded as a fact,
not used as an automatic acceptance or fallback rule, because an indefinite
`S`, rank-deficient equalities, or numerical regularization can invalidate the
simple structural prediction.

## Numerical trade-off and experimental boundary

Normal equations are smaller, SPD, and usually faster on small
well-conditioned problems. The augmented matrix is larger and requires
pivoted LDLT, but avoids the `B'*S^-1*B` equality normal equation and may
preserve more useful conditioning on difficult equality-heavy systems.

Round 3 supports only general dense SDP and PSD-lift SOCP routes with a
provider that explicitly advertises pivoted symmetric LDLT, factor solve, and
multiple RHS. It does not support dedicated LP, sparse Schur, block-arrow,
native Q3, provider fallback, formulation fallback, or precision escalation.
Dependent equalities remain the responsibility of equality presolve/RRQR;
LDLT pivoting must not silently replace rank preprocessing. A successful LDLT
factor reporting zero inertia is rejected as rank deficient, and the route
fails closed when presolve has not produced an independent equality basis.
There is no QR, provider, precision, or normal-equation retry. Round 4 uses
this implementation unchanged: it adds only a static pre-execution choice and
no runtime formulation retry.
