# Disconnected half-Power Phi qualification

`power_half_phi_reference.jl` is NOT loaded by SDPX and makes no root, geometry,
provider, precision or acceptance decision. Its explicitly compensated native
Float64 expansion network differs from the old two-log network. It is not a hidden
BigFloat/MultiFloat promotion. No analytic root or exact-integer-produced value is
used by the evaluator. BigInt/MPFR occur only in independent tests.

## Exact target and domain

For exact values of stored Float64 u,v,w,c and exactly alpha=1/2:

    P=w², C=4uv,
    Q=(P-C)+(P+C)c+Pc²/4, D=C(1-c),
    Phi = (1/2)log1p(Q/D).

Require u,v,|w| in[2^-8,2^8], c in[2^-40,2^-8], finite values and an enclosing
quotient interval Z contained in[-2^-8,2^-8]. Unsupported values return a status,
not a fallback. This is an evaluation domain, NOT a dual-interiority certificate.

Runtime qualification is initially Julia1.12.6 commit15346901f0039751c5488744f1f62de7d87510a8,
aarch64 Darwin, default/user CLI math mode with no fast-math declarations in this
module, round-to-nearest/ties-to-even, gradual underflow, correctly rounded explicit
FMA. The routine checks relevant flags without changing them. Option fast_math==0
means USER mode, not absence of arbitrary @fastmath code. Pinned jloptions.c maps
both CLI `fast` and `user` to that default; the attempted `--math-mode=fast` negative
control did not exercise a different mode and is retained as a failed precondition.
These checks are not universal runtime authenticity: qualification separately
records actual EFT LLVM/native code and exact probes. Non-default IEEE mode is
conservatively outside this initial qualification. No muladd, reassociation or
fast-math is authorized by the arithmetic proof.

## Conditional arithmetic proof

All interval and EFT claims below assume the stated IEEE basic-operation/FMA
contracts. Testing cannot replace that conditional theorem or qualify arbitrary
compiler transformations/platforms.

A Float64 in the u/v/|w| domain is an integer multiple of2^-60; c is a multiple
of2^-92. Let TwoProd return h=RN(a*b), l=fma(a,b,-h). The usual product residual
is representable in binary64 when product overflow and residual underflow are
excluded; then h+l equals the exact product. TwoSum is Knuth's six-operation
error-free addition under the same nearest-rounding/no-range-loss conditions.

Those conditions hold for this complete construction, not merely its final result:

- P and uv components are multiples of2^-120; scaling uv components by4 is exact,
  giving C multiples of2^-118.
- c² components are multiples of2^-184.
- Linear product components are multiples of2^-212 (or coarser).
- Quadratic product components before scaling are multiples of2^-304; division
  by4 is exact, leaving multiples of2^-306.
- D's components are multiples of2^-210 or coarser.
- Every subsequent exact add/subtract of Q components remains on the2^-306 grid.
  RN cannot introduce a finer grid: if the grid's integer needs at most53 bits it
  is exact; otherwise RN removes low bits. Thus nonzero grow-expansion/TwoSum
  intermediates are at least2^-306, far above the normal threshold2^-1022.

Magnitude bounds are also uniform. P<=2^16 and C<=2^18. Original linear terms are
bounded by2^10 per C component and the quadratic leading product by1/4; residual
components are smaller than their rounded high products. The deliberately loose
bound2^20 for each of20 original components gives input absolute sum below2^25.
Do NOT assume TwoSum preserves that absolute sum: with unit roundoff2^-53, replacing
a,b by s,e can increase it by at most a factor1+2^-52. Over at most205 transforms,
(1+2^-52)^205<2, so transformed absolute sums stay below2^26. Successive triangle
and nearest-rounding bounds on the six expressions give the ample bound2^40 on
all temporaries (including virtual operands), far below overflow. The common grid
above excludes underflow at each step. All products/scalings have smaller bounds.
Runtime normal-or-zero checks supplement this proof; alone they would not detect
every underflowed residual.

Construct Q from4 constant,8 linear and8 quadratic components, and D from2 constant
plus4 linear components. This implementation uses13 TwoProd calls (below the design's
conservative14 bound). Grow-expansion retains every component, including zeros;
its telescoping exact TwoSum identities preserve the exact total regardless of
cancellation. Twenty and six insertions require190 and15 TwoSum calls. No adaptive
search, truncation, precision loop or unbounded expansion is used.

## Outward ledger

Each rounded primitive is immediately enclosed by its adjacent Float64 values.
For finite RN addition/multiplication/division, this encloses the real result,
including gradual underflow. Interval products/division take extrema of the four
endpoint operations, then their adjacent bounds; denominator lower endpoint must
be positive. These are bounds for individual primitives, not one final nextfloat
around a compound nearest-rounded expression. All interval arithmetic here stays
finite; unexpected unsupported arithmetic fails closed.

Exact expansion totals are enclosed by outward addition of their stored components.
Their quotient interval is checked before any series evaluation.

Degree12 Horner evaluation encloses coefficients (-1)^(k+1)/k by adjacent values
of their rounded divisions and encloses every multiply/add. For |z|<=2^-8,

    |log1p(z)-sum(k=1:12,(-1)^(k+1)z^k/k)|
      <= |z|^13/[13(1-|z|)] < 2^-107.

Adding the exact binary interval[-2^-107,2^-107] and outwardly multiplying by1/2
therefore encloses Phi. The returned estimate is a bounded interval representative;
its radius outwardly encloses both endpoint distances. Neither native log nor a
library transcendental error assumption is used.

## Qualification boundary

Tests use actual captured warm inputs and all128 source-body observed currents;
these are scalar replay inputs, not newly reconstructed HSD geometry. Independent
references use exact rational input expressions, directed MPFR logarithms and
outward combination, not the production polynomial. Exact rational sums check
actual expansion components. Discarded product tails and understated intervals
must fail cancellation controls. Runtime flags/code generation and negative domain
controls are separate evidence.

Even success qualifies only this disconnected evaluator. Production's root bracket
arithmetic, stopping and stored-point gradient/Hessian geometry still need separate
proof/review. Existing Power/mixed-Exp failures and legacy midpoint/BigFloat SPD
limitations remain unchanged. No memory-admission or performance claim is made.
