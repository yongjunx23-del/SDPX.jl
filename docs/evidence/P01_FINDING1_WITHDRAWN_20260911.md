# P01 FINDING-1 withdrawn — it is a fixture defect, not a production defect

Date: 2026-09-11
Status: **corrects the commit message of `b3bd665` and the ORCHESTRATION record.**
Precedent: `BASELINE_CORRECTION_20260911.md` (retracted five wrong "plan errata")
and `PROVIDER_ENV_CORRECTION_20260911.md` (corrected a wrong provider premise).

## What was claimed

Commit `b3bd665` says, in its subject and body, that P01 found and the orchestrator
**independently confirmed** "a CONFIRMED P0 in the SDPX sparse seam" — specifically
that `src/factor_cache/routes/qdldl_sparse.jl` returns answers that are not the
solution of the system it was given. Three pieces of evidence were cited:

1. The error is bit-identical across two unrelated arithmetics
   (`MultiFloat{Float64,2}` and `BigFloat@256` both `0.20035761024843557`), so it
   cannot be provider rounding.
2. Isolated through `SparseQDLDLCache` against a dense solve: `1.9198321536764786e10`.
3. Both providers are exonerated: each provider's own cache passes the same
   fixture at `0.0` and `4.32e-78`.

The orchestrator reproduced the failure by re-running P01's contract and seeing
`0.20035761024843557` again, and treated that as independent confirmation.

## What is actually true

`SparseQDLDLCache` is **correct**. The defect is in P01's `core_structural_dense`,
which builds an asymmetric dense matrix and uses it as the oracle.

Reproduced from a from-scratch reconstruction of the specimen operator:

    residual ||K*got - rhs||_inf                = 0.183938504828963233930794706145
    ||got - dense||_inf                         = 0.081455278986826012947211571548

which looks like the reported defect — until the matrix itself is checked:

    ||K - K'||_inf                              = 0.25
    ||Symmetric(K, :U) - K||_inf                = 0.25
    residual ||Symmetric(K,:U)*got - rhs||_inf  = 2.573e-31

The cache's answer solves the **symmetric** operator defined by the stored upper
triangle to `Float64x2` rounding. It was never solving the asymmetric `K` that
P01 compared against.

### The bug

In `core_structural_dense` (test/provider_contracts/sparse_fixtures.jl):

    for size_block in CORE_BLOCK_SIZES
        for column in 1:size_block, row in 1:column
            K[nr + offset + row, nr + offset + column] = -theta_entries[offset + column]
        end
        offset += size_block
    end

`row in 1:column` writes only the **upper** triangle of each Θ block; the lower
triangle stays exactly zero. The matrix is therefore asymmetric by exactly
`theta_entries[2] = 1/4`, which is the missing mirror. The coupling loop
immediately below *does* mirror explicitly (`K[nr+i,j]` and `K[j,nr+i]`), which is
what makes the omission easy to overlook.

## Why the cited evidence was misleading

Point 1 — the identical error across two arithmetics — was the strongest-looking
piece and pointed the wrong way. An arithmetic-independent error is indeed the
signature of something structural rather than rounding, but the structural
mismatch was between **two different matrices**, not inside the provider path.
The observation was correct; the attribution was backwards.

Point 3 — that the providers are exonerated — was true and should have been the
hint: if the provider cache solves the same fixture correctly and SDPX's cache
does not, the natural next question is whether the two are being asked about the
same matrix. That question was not asked.

## Process failure, stated plainly

The orchestrator "confirmed" a P0 by **re-running a failing test and seeing the
same number**. Reproducing a failure is not the same as validating the test that
produced it. The report's oracle was never checked, and the packet's rule —
"不以实现者自报代替证据", do not substitute a claim for evidence — applies to the
orchestrator as much as to any worker. It was violated here, and the failure was
published into a commit message and the packet record before it was caught.

The catch came only from asking a different question: not "does the failure
reproduce?" but "**is the returned vector a solution of the matrix it was
actually given?**"

## Status of P01's other findings

Unchanged and not re-examined by this correction:

- **FINDING-2 (P1)** — BFLA's capability probe reports `:natural` ordering
  available while the constructor throws "ordering natural is unavailable".
- **FINDING-4 (P2)** — the same-epoch early return in `qdldl_sparse.jl` skips the
  numeric refactor without comparing values.
- **FINDING-6 (P3)** — one leg fails on the fixture's own type bookkeeping
  (`FieldError` on `colptr`).
- The four **CONTRACT** legs (`symbolic_reuse`, `numeric_refactor`, `multi_rhs`,
  `reuse_after_failure`) are unaffected by this and stand on their own evidence.
- `src/sparse_la.jl`'s `GenericSparseCholeskyFactor` remains orphaned.

P01 has been asked to fix the Θ-block loop, adopt the symmetric operator as the
oracle for every seam leg, re-run both provider legs, and re-classify FINDING-1 as
a fixture defect with the numbers rather than deleting it.

## Consequence for the release decision

The sparse seam was the packet's most alarming claimed defect and it does not
exist as described. This removes a potential release blocker — but it should also
reduce confidence in *every* cross-provider numeric claim whose oracle was not
independently checked, including claims made by the orchestrator. V01's
independent verification is the right place to settle that, and this incident is
the argument for it existing.
