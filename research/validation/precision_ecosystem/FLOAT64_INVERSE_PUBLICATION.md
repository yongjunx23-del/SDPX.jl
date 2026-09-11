# Float64 additional inverse selections — bounded verification

This change extends the existing BigFloat upper/lower symmetric-entry selection
policy to Float64 ONLY, after historical midpoint rejection. Candidate0 arithmetic,
ordering, checks and return behavior are unchanged. Float16/Float32/MultiFloat and
existing BigFloat behavior are not extended or requalified.

All three native post-symmetrization column certificates and the native zero-RHS
Cholesky gate still run unchanged. Each additional Float64 candidate must THEN pass
`_ns_float64_exact_spd3_veto`. Exact verification may reject; it cannot synthesize
entries, replace factors, change solves or waive another gate. Existing owned
snapshots/stores and validity/rollback responsibilities remain intact. No change to
scaling3, root/metric/progress tolerances, iteration caps, provider/configuration or
solver precision is made.

The verification arithmetic is explicitly BigInt/dyadic, not a hidden floating
precision promotion. A finite Float64 word encodes an exact integerA=2^1074*x:
subnormals use their fraction directly; normals use their53-bit significand shifted
by the biased exponent minus1. No floating rescaling, decimal parsing or BigFloat
conversion occurs. Each encoded magnitude has at most2098 bits.

For the six symmetric entries a,b,c,d,e,f, exact Sylvester signs test a, ad-b² and
`a(df-e²)-b(bf-ce)+c(be-cd)`. With largest input widthw, a fixed3w+3 intermediate bound
(at most6297) is checked against a maximum6400-bit budget BEFORE BigInt construction.
At most11 BigInt multiplications per extra candidate, at most8 candidates, no
adaptive precision/search. This is not a bound on allocator/GMP scratch or RSS.
Shape/type, finite-word, exact real-value symmetry (signed zeros equivalent), and
budget mismatches reject. Solver-owned buffers must not be concurrently mutated.

## Evidence and limits

Captured bt0/2/14 fixtures preserve actual stored native L and historical midpoint
bits. Low-level replay is an inverse-kernel input test, not reconstructed HSD or
cone-geometry authority. Native midpoint matrices at bt0/2 are genuinely indefinite;
all unchanged native gates plus exact SPD admit first selections3/5. Bt14 keeps
midpoint0. Independent rational Schur-elimination tests reuse neither the production
dyadic decoder nor its determinant expression.

The native Cholesky gate alone has false positives: matrices[2 2 0;2 c 0;0 0 1] with
c=2 or prevfloat(2). A synthetic nearby bt0 factor (L11 lowered four representable
steps) also gives an extra candidate passing all native gates but exact signs(+,+,-).
This is candidate-level evidence, not proof of midpoint rejection or upstream cone
qualification for the altered factor. A bounded parent scan found no such
midpoint-rejected case in its tested one-parameter grid; that is not a proof.

The extra-candidate veto does NOT certify unchanged midpoint/BigFloat paths,
literal-gradient accuracy, Phi/root reconstruction, Cartesian geometry, downstream
scaling or convergence. Complete memory/production qualification remains absent.
