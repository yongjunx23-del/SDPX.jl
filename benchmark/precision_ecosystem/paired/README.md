# Repaired paired benchmark foundation — not performance qualification

This first slice provides immutable native-input fixtures and a genuinely
independent exact oracle. The positional ABBA/BAAB timing driver, all-sample
retention/validation, untimed execution-receipt probes, and three-process
aggregation are still required before any performance submission or claim.
The cancelled old pilot remains invalid and is not reused as timing evidence.

## Exact reference

For every finite binary64 word x, `scaled(x)` is the exact integer x*2^1074.
For a normal word with exponent field e and significand q, this is q<<(e-1);
for a subnormal it is its fraction field. Signs are applied exactly. Sum all
four stored limbs to obtain each native Float64x4 input's exact integer value.

For alpha=1, each scalar product has grid 2^-2148. Accumulate products as BigInt
integers without floating arithmetic, BLAS, MFLA, MFA, matrix multiplication,
or refactorization. For beta=1/2, add C0's integer shifted left1073; beta=0 adds
zero. Persist reference integers and exact product/operand absolute sums once
per cell. No numerical approximation or convergence comparison is needed for
this reference. BigInt is reference-only, not candidate arithmetic.

The input fixture stores every binary64 limb word as sixteen lowercase hex
digits, in column-major element order and limb order1:4. Word encoding is
endian-independent; native endian metadata is separately recorded. Shapes,
alpha/beta words, type/209-bit precision, family, input ID and digest are bound.
Deserialization preserves bits, checks normalization, and rejects malformed,
nonfinite, wrong-shape or wrong-hash input. Repetition IDs never generate input.

Families: fullwidth, paired cancellation, and all-zero negative/edge-control
coverage. Paired B rows are exact copies; paired A columns differ from negation
by a full-width residual. Odd k has an actual unpaired residual-scale tail.
Cancellation is measured exactly; zero sums are counted, not assigned fabricated
finite ratios. Local fixture scope is positive dimensions at most256 each.

## Honest metrics

The retained mixed-scale gate is exactly 3.1e-61, named `mixed_max1_norm_error`:
max|error|/max(1,max|reference|). This is the existing public experimental-test
ceiling, slightly stricter than the older paired driver's 2^-201 approximation.
It is NOT floor-free relative accuracy. Gate comparisons use exact integers;
output normalization is additionally mandatory and outputs are never renormalized.

Report separately absolute error, floor-free normwise relative error,
componentwise relative distribution and zero-reference errors, and error scaled
by actual absolute product/initial-C contributions. Zero denominators remain
explicitly undefined. All-zero inputs are controls, not representative performance
workloads. Exact rational strings avoid rounded floating acceptance decisions.

## Execution and pending work

Use only a clean committed checkout and a private pinned environment. Required
PAIRED_HEAD, PAIRED_THREADS and absolute PAIRED_OUT bind the loaded harness,
thread count and output. `prepare.jl m k n family beta input_id` creates a NEW
output directory with fixture/reference/plan and before/after source receipts.
`test_exact_inputs.jl` uses an already-created empty external evidence directory.

Pins: Julia1.12.6; MultiFloats3.2.6/x4=209bits; MFLA5399c0cc/version0.4.0;
MFAcdb84680/version0.1.0; actual weak extension required. Roots come from loaded
modules, Git HEAD/cleanliness uses subprocess cwd (compatible with old Git),
and package closure/harness/environment hashes are asserted before/after.
BLAS/OMP/MKL1, one GC thread, 2G heap hint; bound Mac processes to180s.
Affinity is recorded, not falsely claimed pinned on unsupported platforms.

No production code changes, public route promotion, memory admission, speedup,
large-core scaling, or solver qualification follows from this foundation.
