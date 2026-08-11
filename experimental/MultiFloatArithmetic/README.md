# MultiFloatArithmeticResearch

Standalone research core for MultiFloats arithmetic. This directory deliberately has no dependency on SDPX solver types.

Current priority:

1. implement verified branch-free fused multiply-add networks for 2-, 3-, and 4-limb `MultiFloat` / `MultiFloatVec` values;
2. validate scalar/SIMD lane agreement, commutativity, normalization, and MPFR/BigFloat error behavior;
3. build an FPAN-compatible search/verification harness for 5–8 limbs;
4. only after arithmetic is stable, build `MultiFloatLinearAlgebra` on top of it;
5. integrate the resulting backend into SDPX last.

## Arithmetic modes

`fma_fast(x, y, c)` follows the 2026 branch-free DW/TW/QW FMA networks. Their published, machine-verified error semantics are operand-relative: the error is bounded using `|x*y| + |c|`. This is the throughput-oriented path.

A future `fma_strong` path will target result-relative error for cancellation-sensitive residual/refinement work. It must not be silently substituted for `fma_fast`; callers need to choose the numerical contract explicitly.

## Rules

- Do not alter `Base.fma` or `Base.muladd` while this is research code.
- Do not enable compiler reassociation/fast-math in the verified network.
- Preserve the exact EFT and addition order of the verified network unless the modified network is re-verified.
- Scalar and `MultiFloatVec` implementations share the same tuple-level network.
- x5–x8 candidates remain research-only until a formal verifier proves the required non-overlap and error bounds.

## Sources

The initial x2/x3/x4 network is an independent Julia implementation of the algorithms described in Tomonori Kouya, *Performance evaluation of branch-free fused multiply-add algorithms for multi-component-type multiple-precision floating-point arithmetic*, arXiv:2607.11391 (2026). The underlying branch-free addition/multiplication framework and FPAN verification methodology come from David K. Zhang and Alex Aiken, SC 2025 / CAV 2025.
