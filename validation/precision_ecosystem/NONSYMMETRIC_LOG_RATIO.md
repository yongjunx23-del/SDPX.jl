# Shared log-ratio branch repair — bounded scope

The value-only and work-producing helpers in `src/cones/exponential.jl` are shared
by Exp and Power. Their precondition is positive finite operands.

Float64 `n=3*2^-54,d=1` previously formed `r=fl((n-d)/d)=-1+2^-52` and evaluated
`log1p(r)`. This loses information before the logarithm: the true ratio is
`3*2^-54`, not `2^-52`, producing error approximately `log(4/3)=0.287682`.
The corresponding literal Power Phi witness violates the old6.7291e-13 floor.
This is not evidence that this point caused the historical public failure.

The repair uses the existing log1p path only for computed `r` in[-1/2,1], away
from its singular argument-1; otherwise it uses the existing separate-log path.
Both entry points share the branch predicate. Comparable-operand arithmetic and
its operation order are unchanged. Work accounting follows the branch actually
executed; no tolerance, root cap, status rule, reconstruction, factor, inverse,
secant or five-equation gate changes.

Separate logarithms can lose accuracy by cancellation under large common
scaling, even when the ratio itself is moderate. The work-producing path records
both logarithm magnitudes rather than hiding that cost. Tests use an independent
operand-log-work error goal on widened actual stored inputs, include common
scalings, and check branch/value/work consistency. They do not claim uniform
relative accuracy in the small returned logarithm or exact scaling covariance.

Tests run native arithmetic before raising MPFR reference precision; native eps
and precision are captured first. Float32/Float64, BigFloat256/512, optional
MultiFloats3.2.6 x4 (actual precision209), full nonzero limbs, branch boundaries,
near-unity values and Float64 exponent endpoints are covered. The cold Power
control independently recomputes the alpha1/2 barrier gradient from the actual
stored production shadow, not from its supplied gap.

This repairs the exhibited argument-loss defect only. The complete native Phi
error ledger remains unqualified: transcendental errors, generated input errors,
nearest-rounded interval operations, reconstruction and actual stored geometry
still need justification. Cold-control preservation is not production/root
qualification. In particular it does not license the rejected16sqrt(eps)→16eps
change or classify stalled iterations as successes. Public solves are observed
separately, never inferred from the kernel tests.
