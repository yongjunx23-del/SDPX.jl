# Nonsymmetric 3D scaling contract

This note fixes the production convention used by the exponential and power
cones.  It is a contract for the implementation, not an alternative cone
formulation.

## Orientation and shadows

For a primal point `s` and a dual point `y`, SDPX stores two symmetric positive
definite maps

\[
G:\mathcal K\to\mathcal K^*,\qquad
\Theta:\mathcal K^*\to\mathcal K,
\]

with `Theta` authoritative and `G` its explicitly certified diagnostic inverse.
The accepted secants are

\[
Gs=y,\qquad \Theta y=s.
\]

The logarithmically homogeneous barrier has degree three.  The Fenchel shadow
`stilde` is obtained by solving

\[
-\nabla F(\widetilde s)=y,
\]

and `ytilde = -grad F(s)`.  A strict double-secant state additionally satisfies

\[
G\widetilde s=\widetilde y,
\qquad
\Theta\widetilde y=\widetilde s.
\]

The conjugate solve owns a separately certified analytic lower factor of the
primal Hessian.  A rounded dense Hessian or a lazily materialized inverse is
diagnostic state and cannot replace that factor as numerical authority.

## Tunçel double secant in three dimensions

Let

\[
S=[s\ \widetilde s],\qquad
Y=[y\ \widetilde y],\qquad
M=Y^\mathsf T S.
\]

Logarithmic homogeneity gives the cross identities
`dot(y, stilde) = dot(ytilde, s) = 3`.  Production first certifies those
identities, symmetry of `M`, and a positive, resolvable determinant.  It then
forms the rank-two parts

\[
G_2=Y M^{-1}Y^\mathsf T,
\qquad
\Theta_2=S M^{-1}S^\mathsf T.
\]

In three dimensions the missing directions are explicit.  Define

\[
z=\frac{s\times\widetilde s}{\lVert s\times\widetilde s\rVert},
\qquad
r=\frac{y\times\widetilde y}{z^\mathsf T(y\times\widetilde y)},
\]

so `z` annihilates the primal secant plane, `r` annihilates the dual secant
plane, and `dot(z,r)=1`.  The complete inverse pair is

\[
G=G_2+t_Gzz^\mathsf T,
\qquad
\Theta=\Theta_2+t_G^{-1}rr^\mathsf T.
\]

The positive coefficient is selected from the Dahl–Andersen block-BFGS metric.
With `G0 = mu * hessian(F,s)`, production removes the action of `G0` on the
two-dimensional secant plane, adds the required rank-two secant map, and uses

\[
t_G=z^\mathsf T G_{\rm BFGS}z.
\]

All products use the one global HSD complementarity value `mu`; a product cone
must not silently substitute a per-block `dot(s,y)/3` value.

## Explicit fallback policy

Fallback is available only through
`DoubleSecantWithDualHessianFallback`.  A strict provider failure is recorded
before the provider changes.  Eligible failures include a singular second
secant Gram matrix, a degenerate axis, a nonpositive BFGS coefficient, loss of
positive definiteness, or a failed secant/inverse certificate.

The fallback starts from

\[
\Theta_0=\mu\nabla^2F_*(y)
\]

and applies a one-secant BFGS repair,

\[
\Theta=\Theta_0-
\frac{(\Theta_0y)(\Theta_0y)^\mathsf T}{y^\mathsf T\Theta_0y}
+\frac{ss^\mathsf T}{s^\mathsf Ty},
\]

so the accepted metric still satisfies `Theta*y = s`.  `G` is rebuilt by
triangular solves from the certified lower factor of `Theta`; it is not an
independent provider.  Failure at any construction or certificate gate returns
a typed non-success result.  There is no unrecorded retry with a different
provider.

## Acceptance certificates

An accepted state must pass all applicable checks:

1. finite, symmetric, positive-definite `Theta` and `G`;
2. a componentwise reconstruction certificate for `Theta = L*L'`;
3. componentwise secant certificates using the actual row arithmetic work;
4. a factor-aware inverse-column certificate for `G*Theta = I`;
5. for strict double secant, both primary and shadow secant pairs;
6. for fallback, the authoritative `Theta*y = s` secant plus the propagated
   consequence certificate for `G*s = y`;
7. exact zero-work handling: zero work accepts only an exactly zero residual;
8. common-`mu`, epoch, checkpoint, finiteness, and validity parity before a
   checkpoint can be committed or restored.

These are backward-error certificates.  No `max(1, work)` floor is permitted
where it would make the decision depend on an arbitrary scaling of otherwise
homogeneous data.

The implementation correspondence is intentionally direct:

- `_ns_scaling_double_secant!` constructs the Tunçel rank-two maps and the
  Dahl–Andersen axis coefficient;
- `_ns_scaling_dual_hessian_fallback!` applies the explicit one-secant repair;
- `_ns_scaling_theta_factor_certificate!` and
  `_ns_scaling_inverse_columns_certificate!` certify the accepted inverse pair;
- `try_nonsymmetric_higher_correction!` constructs and certifies `chi`;
- `_product_hsd_form_coupled_matrix!` freezes accepted block factors;
- `_product_coupled_prepare_factor_coordinates!` constructs `Khat`, while
  `_product_hsd_coupled_recover!` restores and certifies physical coordinates.

## Higher-order correction

For affine directions `ds_aff` and `dy_aff`, the nonsymmetric corrector uses

\[
u=\nabla^2F(s)^{-1}dy_{\rm aff},\qquad
\chi=-\tfrac12 F'''(s)[ds_{\rm aff},u].
\]

The analytic Hessian factor performs the solve.  Production independently
checks the factor identities, the triangular solve, swapped third-derivative
symmetry, and the logarithmic-homogeneity Euler identity before the correction
is admitted.  The symmetric-cone coordinate product `ds_aff .* dy_aff` is not a
valid replacement for exponential or power cones.

## HSD consumption

The product HSD layer consumes the accepted metric without changing providers.
It uses one common primal/dual step length, retains the original-coordinate
five-equation residual and `K*z-q` certificate as the final authority, and
performs exactly one factorization per KKT epoch.  Factor-coordinate transforms
may improve the numerical conditioning of the coupled system, but transformed
residuals never replace the original-coordinate certificate.

For the hybrid coupled system, collect the accepted block factors in the block
diagonal matrix `L`, where `Theta = L*L'`, and order the physical unknown as

\[
z=(dx_r,dy_N,d\tau,d\kappa).
\]

The nonsymmetric complementarity rows precede the reduced dual, gap, and scalar
rows.  Production changes only the linear-system coordinates:

\[
v=L^\mathsf Tdy_N,\qquad
R=\operatorname{diag}(L^{-1},I),\qquad
C=\operatorname{diag}(I,L^{-\mathsf T},I_2),
\]

\[
\widehat K=RKC,\qquad \widehat q=Rq.
\]

It factors `Khat` once, solves for `(dx_r,v,dtau,dkappa)`, and recovers
`dy_N = L^(-T)*v`.  Every factor copy, left/right triangular transform, RHS
transform, and recovery has a componentwise actual-work certificate with exact
zero-work handling.  The recovered vector must then pass both the retained
physical `K*z=q` certificate and the independent five Newton-equation groups.
