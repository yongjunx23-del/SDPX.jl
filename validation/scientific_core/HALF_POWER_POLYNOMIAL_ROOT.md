# Full-gap half-Power interval root

Experimental native Float64 root, not a production replacement. It solves the
SAME half-alpha conjugacy equation using its polynomial zero, never an analytic
root formula or a reference-generated candidate.

For P=w²,C=4uv,
  Q(c)=P(1+c/2)²-C(1-c)
      =(P-C)+(P+C)c+Pc²/4,
  Q'(c)=P+C+Pc/2.
Strict dual interior gives Q(0)<0. If w!=0, Q(1)=9P/4>0 and Q'>0, hence one
root in(0,1). For w=0 the root is exactly1, matching the existing special case.
Outward endpoint signs establish existence; interval Newton preserves the root
in the full mathematical bracket[0,1]. Clipping an intersection to this proved
root bracket is not clipping to an unproved evaluator subdomain.

## EFT domain extension

The existing fixed20-term Q construction and its exact TwoProd/TwoSum network
are used, but no log/series evaluator is called. Require u,v and nonzero|w| in
[2^-8,2^8]. Probes are zero or in[2^-40,1], including the conceptual upper
endpoint. Word grids are2^-60 and2^-92; products through w²c²/4 remain on grid
2^-306. Extending the upper c bound from2^-8 to1 increases magnitude but not
the finest grid. The raw term L1 sum is below2^24;205 growth operations have
factor(1+2^-52)^205<2, and the old conservative2^40 temporary envelope still
holds. No nonzero EFT intermediate underflows or overflows. Zero w/c terms are
exact zeros. The inherited Julia1.12.6/aarch64-Darwin/default-math/nearest/FTZ-off/
FMA contract remains required.

Q and Q' receive primitive outward interval bounds. Each update intersects the
current bracket with probe-Q(probe)/Q'(bracket). Stored midpoints are only probes.
If a midpoint would be below2^-40 while the bracket still includes that value,
2^-40 can be used as an interior/boundary probe without changing the bracket.
Roots proved below the supported positive-gap domain, or unresolved at its edge,
remain unsupported. No false root is created by clamping a returned gap.

Acceptance uses the same outward radius targets R<=tau*lower and
R<=2^-22*lower. The retained iteration/midpoint budgets remain at most64/512;
up to two fixed endpoint evaluations are separately counted. Cold seed is the
existing0.5, while an explicit supported warm seed is used unchanged. Invalid
warm data refuse rather than silently switch modes. No cap or tolerance grows.

Tests use exact rational polynomial bisection for root containment and exact
polynomial/derivative checks for every trace interval. Cold, warm, signed third
dual, zero-third dual, near-boundary roots and both endpoints are covered.
No BigInt/BigFloat/reference value enters the candidate algorithm. Factor,
corrector, line-search, default/strict and full-solver admission remain separate.
