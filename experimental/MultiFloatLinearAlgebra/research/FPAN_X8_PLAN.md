# Verified Float64x5–Float64x8 arithmetic research plan

Status: research-only. Nothing in this document is an accepted arithmetic kernel until it passes both formal verification and the SDPX numerical/performance gates.

## Objective

Develop fixed-length, branch-free arithmetic for 5–8 binary64-component floating-point expansions, with special emphasis on operations that dominate SDPX:

1. normalized addition/subtraction;
2. commutative multiplication;
3. fused multiply-add/subtract (`x*y + c`, `c - x*y`);
4. reciprocal/division and square root through precision-doubling Newton/Karp–Markstein steps;
5. SIMD-friendly linear-algebra kernels built on those primitives.

The final target is not merely good random-test accuracy. Accepted kernels must satisfy a machine-checkable contract:

- normalized/non-overlapping output expansion;
- explicit worst-case discarded-tail bound;
- commutativity where mathematically required;
- no hidden value-dependent branch in the hot arithmetic network;
- reproducible behavior under the supported IEEE-754 round-to-nearest environment.

## Two-track strategy

### Track A — correctness-first reference arithmetic

Before optimizing x8, build a slow research oracle for fixed-length expansions:

1. form error-free product/sum terms using `TwoProd`/`TwoSum`;
2. distill/renormalize without discarding information until the final compression;
3. compare against high-precision MPFR/BigFloat on random and adversarial cancellation cases;
4. use the resulting implementation only as a reference until its compression contract is formally established.

This track is intentionally allowed to be slower than the production x4 arithmetic. Its purpose is to make x5–x8 search measurable and debuggable without resurrecting the unsafe v2 algorithms.

### Track B — verified branch-free production networks

Search fixed accumulation networks and accept only candidates verified by FPANVerifier (or an equivalently rigorous checker).

Do not search N=8 from an unconstrained random network. Grow incrementally:

`N=4 proven seed -> N=5 -> N=6 -> N=7 -> N=8`.

For each N, solve addition first, then multiplication, then a direct fused MAC network.

## Search-space reduction

### 1. Weight-level skeleton

Exploit the known component ordering of a normalized expansion. Treat component `i` as living near weight `u^i` and organize product terms by `i+j`.

Candidate multiplication/FMA skeleton for N components:

- level 0 begins with `x0*y0` (and `c0` for FMA);
- level k collects product terms with `i+j=k`, error terms spilled from level `k-1`, and `ck`;
- `TwoSum` errors flow only toward lower-significance levels;
- the final stage is an N-word normalization chain.

For the N=8 FMA search, a useful *candidate* template is:

- `TwoProd(x_i,y_j)` for high-value pairs `i+j <= 6`;
- plain rounded products for the boundary `i+j == 7`;
- terms with `i+j >= 8` are candidates for formal truncation;
- symmetric pairs `(i,j)` and `(j,i)` are combined first.

These truncation rules are hypotheses, not proofs. FPANVerifier must prove that the discarded terms satisfy the target error budget before the candidate is accepted.

### 2. Enforce multiplication symmetry structurally

The search generator must pair transposed product/error terms before arbitrary accumulation. This removes non-commutative candidates from the search space instead of discovering the problem after testing.

### 3. Shrink from a valid large network

Prefer a proof-preserving/descent-style search:

1. start from a deliberately over-complete candidate;
2. remove or replace gates;
3. test quickly on a hard-case corpus;
4. formally verify promising survivors;
5. keep a Pareto frontier in `(gate_count, critical_depth, register_pressure, proof_margin)`.

This should be more productive for N=5–8 than constructing arbitrary networks from scratch.

### 4. Counterexample-guided corpus

Maintain adversarial cases separately from ordinary random tests:

- leading-limb cancellation;
- cancellation spanning multiple adjacent limbs;
- values near powers of two and midpoint/tie patterns;
- subnormal-adjacent values where supported by the arithmetic contract;
- alternating-sign dot-product patterns extracted from SDPX KKT/residual traces;
- Schur/Gram panels whose off-diagonal dot products nearly cancel.

A candidate that fails any discovered case is permanently added to the hard corpus.

### 5. Hardware-aware cost model

The production objective is not gate count alone. Record at least:

- scalar flop/EFT count;
- critical dependency depth;
- number of simultaneously live temporaries;
- FMA/TwoProd count;
- expected SIMD lane width;
- measured cycles on representative AVX2/AVX-512/Apple targets.

The formal contract is hardware-independent; kernel selection may be hardware-dependent.

## Why direct FMA matters for SDPX

SDPX hot loops overwhelmingly contain `acc += a*b` or `acc -= a*b`. A direct expansion FMA can avoid producing a fully normalized multiplication result only to feed it immediately into a second normalization network.

Therefore the production priority after `add_N` and `mul_N` is:

`fma_N / submul_N -> dot -> SYRK/GEMM -> TRSM/Cholesky`.

For cancellation-sensitive residual and certificate paths, keep a stronger reference/result-relative path unless the fused network has an adequate result-relative bound.

## x4 immediate experiment

Before x8 exists, add a research-only QW fused-MAC implementation based on the 2026 branch-free FMA construction and A/B it against current `mfmul + mfadd` in:

1. scalar MAC microbenchmarks;
2. four-lane `MultiFloatVec` dot/SYRK microkernels;
3. the real `3400 x 144` reduced CSDR panel;
4. end-to-end SDPX iterations/time/RSS/certificate gates.

Do not globally replace arithmetic. Separate at least:

- `:fma_fast` — operand-relative verified bound, intended for low-cancellation/high-throughput kernels;
- `:strong` — existing or stronger result-relative behavior for residual/refinement/certification.

## Reciprocal/division/sqrt for x8

Once verified x8 addition/multiplication exist, use precision doubling rather than running every Newton step at eight limbs:

`1 -> 2 -> 4 -> 8`.

The final quotient/root multiplication should be fused when a verified half-to-full or residual-multiply primitive becomes available. This mirrors the structure already used by modern Karp–Markstein/Newton expansion algorithms and keeps most early work at low limb count.

## Acceptance gates

A candidate becomes production-eligible only after all of the following pass:

1. formal non-overlap/error-bound verification;
2. randomized MPFR differential tests over exponent/sign distributions;
3. targeted destructive-cancellation corpus;
4. commutativity/reproducibility tests where required;
5. SIMD scalar-lane equivalence tests;
6. SDPX LP/SOCP/SDP correctness and original-coordinate certification;
7. stable performance improvement, with any >10% regression investigated before enabling by default.

## Initial milestones

- M0: freeze current x4 behavior and benchmark corpus.
- M1: x4 fused-MAC research kernel + SDPX A/B.
- M2: correctness-first x8 reference arithmetic.
- M3: verified `add5`, then `add6`, `add7`, `add8`.
- M4: verified commutative `mul5` through `mul8`.
- M5: direct `fma5` through `fma8`, prioritized by SDPX MAC cost.
- M6: x8 `inv/div/sqrt` via 1->2->4->8 precision doubling.
- M7: SIMD-packed x8 DOT/SYRK/TRSM and guarded SDPX integration.
