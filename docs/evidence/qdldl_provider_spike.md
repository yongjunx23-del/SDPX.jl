# QDLDL 0.4.1 sparse companion inertia provider spike

Status: parallel prototype evidence, 2026-08-28, Julia 1.12.6, macos arm64.
Base: `main` @ `c7a912dd91e12a25bd4b90b3eca2c2283ee41dd5`.

Scope: validate QDLDL 0.4.1 (`osqp/QDLDL.jl`) as the symmetric
signed-regularized quasidefinite **companion inertia** provider for the
frozen HSD Newton system, per `docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md`.
Companions compared against small dense MFLA v0.3.0 (local checkout) and BFLA
v0.2.2 (local checkout) only, as evidence. No provider library was modified.

Deliverables (this branch `agent/qdldl-provider-spike`):

- `ext/SDPXQDLDLExt.jl` — narrow adapter prototype (parallel only, not
  registered in `Project.toml`; loaded by the spike driver with `include`).
- `test/provider_spikes/qdldl_sparse.jl` — validation driver (322 tests).
- This evidence document.

## Frozen contract consumed

- Five semantic Newton equations and `NewtonSystem` signs:
  `src/kkt/system.jl` (frozen at SHA `50dff568` per `docs/design/NEWTON_SYSTEM.md`).
- Exact condensed operator (nonsymmetric, `(x,tau)` skew adjoints) and the
  symmetric companion mirroring the upper x/tau coupling:
  `src/kkt/expanded_quasidefinite.jl:1-7`, `assemble_expanded_kkt!`,
  `_assemble_regularized!`, `_freeze_symmetric_companion!`.
- Expected inertia authority: `expected_expanded_inertia(system)` =
  `KKTInertia(n, m+1, 0)`, recomputed per system, never inferred from the
  observed factor.

The adapter assembles the signed-regularized companion in sparse storage:

```text
[  δI    A'    c ]
[  A    -H-δI  -b ]
[  c'   -b'   -κ/τ-δ ]
```

with `δ = sqrt(eps(T)) * scale` by default (same first rung as the dense
route's ladder). The exact nonsymmetric operator is **never** handed to QDLDL;
the adapter refuses nonsymmetric input (`:nonsymmetric`), and the spike
cross-checks that the exact-operator and companion systems are genuinely
different (companion residual `<= 1e-10` while exact-operator residual
`> 1e-3` on the same direction).

## Validation results

### Scalar type preservation and factor residual

Companion dimension 12 (`m=6, n=5`), `δ = 1e-3`, QDLDL factor + solve.
Residual `‖K*x - b‖∞` in the original scalar type. Factor data
(`workspace.D`, `L`, `Dinv`, `matrix`, `triu_matrix`) has eltype `T` in every
case; no Float64 downcast anywhere.

| scalar | initial residual | residual after update_values! + refactor! |
|---|---:|---:|
| Float64 | 1.9e-12 | 7.0e-13 |
| Float64x2 | 9.3e-29 | 3.7e-29 |
| Float64x4 | 3.1e-60 | 1.8e-60 |
| BigFloat (256-bit) | 6.4e-73 | 4.9e-73 |

BigFloat global MPFR precision stays 256 inside and after the run
(`precision(BigFloat) == 256`), including after a 256-bit factor/solve/refactor.

### Expected positive/negative inertia via D signs/counts

| scalar | positive count (`positive_inertia`) | negative count | zero | Dsigns match |
|---|---:|---:|---:|---|
| Float64 | n | m+1 | 0 | yes |
| Float64x2 | n | m+1 | 0 | yes |
| Float64x4 | n | m+1 | 0 | yes |
| BigFloat256 | n | m+1 | 0 | yes |

`companion_dsigns_match` compares every post-factor `workspace.D` sign against
`Dsigns[F.perm]` — QDLDL permutes the sign vector internally, so the
comparison must be in post-permutation order.

### Dsigns signed regularization

QDLDL 0.4.1 semantics (verified deterministically):

- pivot with `Dsigns[k]*D[k] < regularize_eps` is replaced by
  `regularize_delta * Dsigns[k]` and counted by `regularized_entries`.
- `D = [1e-20 0; 0 1e-20]`, `Dsigns=[1,1]` -> `D=[1e-7,1e-7]`, count 2.
- `Dsigns=[1,-1]` on the same matrix -> `D=[1e-7,-1e-7]`, count 2.
- wrong-sign large pivot `[2 0; 0 -3]`, `Dsigns=[1,1]` -> `D=[2,1e-7]`,
  count 1 (sign flipped, magnitude floored).
- forcing floor (`regularize_eps = T(Inf)`, `regularize_delta = 1e-3`):
  every pivot floored to `±1e-3`, count == dimension, certified inertia is
  exactly the block structure `(n, m+1, 0)` for all four scalars.
- zero signed regularization (`δ = 0`) drops the x-block diagonal pattern and
  the factor fails closed (observed `:zero_pivot`; the coupling entries keep
  the x columns structurally nonempty).

### update_values! + refactor! fixed-pattern reuse

- Symbolic pattern captured once (`entry_index`: upper-triangle `(row, col)`
  -> linear index in the input triu `nzval`).
- Value-only updates (kappa/tau diagonal and one `(x_j, y_i)` coupling) via
  `QDLDL.update_values!`, then `QDLDL.refactor!`; the updated factor solves
  the updated companion at the same type-appropriate residual (table above).
- A coordinate outside the fixed pattern fails closed with `ArgumentError`
  (no silent pattern mutation).
- Refactor allocations are dimension-independent for Float64/Float64x2/
  Float64x4 (160/192/240 bytes at dimensions 16, 41 and 82) — the symbolic
  phase does not repeat. BigFloat refactor allocations grow only through MPFR
  temporaries, not Julia factor storage.

### Vector and multi-RHS solve

Single-vector and 3-column panels solved through one factor at type residual.
The adapter loops per column because QDLDL 0.4.1 `solve!` is single-vector
only (see API gaps).

## Benchmark evidence (small dense MFLA/BFLA)

Same companion data, minimum over 5 runs after warmup (the driver
uses `minimum([... for _ in 1:5])`). QDLDL sparse factor =
symbolic + numeric; refactor = `update_values!` + `refactor!`. Dense rows are
`la_ldlt_factor!` on the dense companion through the SDPX provider seam.
Sparse A density ~80% (`sprandn(rng, T, m, n, 0.8)`); H dense (the
production cone map is dense).

### dimension 16 (m=9, n=6)

| route | factor alloc (B) | factor time | refactor alloc (B) | refactor time |
|---|---:|---:|---:|---:|
| QDLDL Float64 | 53 536 | 8.7 µs | 160 | 0.54 µs |
| QDLDL Float64x2 | 68 624 | 10.6 µs | 192 | 0.96 µs |
| QDLDL Float64x4 | 98 128 | 22.4 µs | 240 | 8.0 µs |
| QDLDL BigFloat256 | 299 888 | 135 µs | 66 048 | 39 µs |
| MFLA dense x2 (factor) | 6 048 | 3.3 µs | (full re-factor) | 3.3 µs |
| BFLA dense 256 (factor) | 16 032 | 93 µs | (full re-factor) | 93 µs |

### dimension 41 (m=24, n=16)

| route | factor alloc (B) | factor time | refactor alloc (B) | refactor time |
|---|---:|---:|---:|---:|
| QDLDL Float64 | 206 944 | 33 µs | 160 | 2.5 µs |
| QDLDL Float64x2 | 273 968 | 45 µs | 192 | 7.2 µs |
| QDLDL Float64x4 | 407 008 | 141 µs | 240 | 73 µs |
| QDLDL BigFloat256 | 1 853 936 | 817 µs | 697 504 | 399 µs |
| MFLA dense x2 (factor) | 33 664 | 28 µs | (full re-factor) | 28 µs |
| BFLA dense 256 (factor) | 85 952 | 1.08 ms | (full re-factor) | 1.08 ms |

### dimension 82 (m=49, n=32)

| route | factor alloc (B) | factor time | refactor alloc (B) | refactor time |
|---|---:|---:|---:|---:|
| QDLDL Float64 | 821 200 | 109 µs | 160 | 13 µs |
| QDLDL Float64x2 | 1 105 680 | 174 µs | 192 | 49 µs |
| QDLDL Float64x4 | 1 651 648 | 742 µs | 240 | 524 µs |
| QDLDL BigFloat256 | 9 807 920 | 4.65 ms | 5 167 200 | 3.07 ms |
| MFLA dense x2 (factor) | 122 272 | 171 µs | (full re-factor) | 171 µs |
| BFLA dense 256 (factor) | 328 368 | 7.63 ms | (full re-factor) | 7.63 ms |

Reading for these small dense sizes: at dimension 16 the dense providers are
faster for the first factor; the QDLDL fixed-pattern refactor becomes the
cheaper per-epoch path already at dimension 41 for Float64x2 (7 µs vs 28 µs)
and at dimension 82 for BigFloat256 (3.1 ms vs 7.6 ms), and the sparse first
factor overtakes dense MFLA/BFLA at dimension 82 even with 80% sparse A. This
is per-epoch evidence only; the production memory-budget gate
(`docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md` §Densification gate) still
decides dense vs sparse routing.

## Exact API gaps in QDLDL 0.4.1 (no library modification)

1. `solve!`/`solve` are **single-vector only**. A `Matrix` RHS throws
   `ReadOnlyMemoryError` on the AMD-permuted path and, with `perm=nothing`,
   silently solves only the first column (linear indexing reads only the
   first `n` entries). Multi-RHS must loop per column (adapter does).
2. `update_values!`/`scale_values!` address the matrix by **linear index into
   the input upper-triangular CSC `nzval`** (internally mapped through
   `AtoPAPt`), not by `(row, col)` coordinates; there is no exported helper
   that builds that map, so an adapter must record it at symbolic setup.
3. `Dsigns` is **permuted internally**; post-factor `workspace.D` signs must
   be compared in `Dsigns[F.perm]` order, and `Dsigns` entries are
   `+1`/`-1` integers with no documented sign convention beyond the
   regularization inequality.
4. Zero pivots under `Dsigns` are **regularized, not rejected** (floored to
   `±regularize_delta`); the unregularized path (`Dsigns = nothing`) is the
   only way to get the fail-closed `ErrorException("Zero entry in D (matrix
   is not quasidefinite)")`. Empty structural columns abort with
   `ErrorException("Input matrix is not upper triangular or has an empty
   column")` before any factorization.
5. Inertia API is a single `positive_inertia(F)` count; negative must be
   derived (`dimension - positive`), zero count is always 0 (zero pivots
   abort), and D itself is only reachable through the public
   `workspace.D`/`workspace.Dinv` fields.
6. `regularize_eps`/`regularize_delta` are **absolute thresholds in the
   scalar type**. With the QDLDL defaults (`1e-12`/`1e-7`) every pivot below
   `1e-12` is floored — for Float64x2 (`sqrt(eps) ≈ 1.5e-16`) and
   BigFloat256 (`sqrt(eps) ≈ 7.5e-39`) the natural tiny pivots of an
   `sqrt(eps(T))`-regularized companion are all floored at `1e-7`. An adapter
   must scale these parameters (the prototype exposes them; defaults are
   `regularize_eps = δ` and `regularize_delta = 10δ` where `δ =
   sqrt(eps(T))*scale`).
7. `amd_dense_scale` is downcast to `Float64` for the AMD control array — the
   only Float64 touch inside the library; value-neutral and type-preserving.
8. No matrix-shaped D accessor, no explicit triangular-solve surface, no
   transpose solve, no combined factor+inertia report: all adapter-side
   composition.

## Reviewer closeout (2026-08-28)

- `companion_update!` invalidates the factor and inertia authority **before**
  any mutation: the status drops to `QDLDL_COMPANION_READY` with
  `failure = :stale_factor`, so a solve between update and refactor fails
  closed and `companion_inertia`/`companion_dsigns_match` report nothing
  until `companion_refactor!` re-certifies. The factor object is retained
  (refactor needs it) but its authority is revoked.
- Zero numeric updates preserve the explicit CSC pattern and both linear-index
  maps: `companion_update!` writes through `nzval` at the recorded positions
  (never `setindex!`, which deletes stored entries on some Julia versions),
  and keeps both triangles of the full symmetric matrix in sync. `nnz` and
  `entry_index` are invariant under zero updates.
- The multi-RHS panel solve is safe under `dest`/`rhs` overlap: it solves
  from an owned copy of the RHS panel, so shifted/aliased views of one buffer
  cannot corrupt unconsumed RHS columns.
- The coupling fixture `_first_coupling_entry` now selects a genuine
  `(x_j, y_i)` A-coupling entry (rows `1..n`, columns `n+1..n+m`); the old
  `key[2] <= n + m` test matched the x-block diagonal `(1, 1)` instead.
- The spike fixture is deterministic: every `randn` in system construction
  draws from the seeded `MersenneTwister` (previously `H`, `b`, `c`, `shift`
  came from the global RNG, making runs non-reproducible).
- Test count is 322 + 1 (benchmark evidence), all four scalar types
  (Float64, Float64x2, Float64x4, BigFloat256).

## Findings and risks

- The adapter is genuinely generic: Float64, Float64x2, Float64x4, and
  BigFloat256 all factor/solve/refactor at type-appropriate residuals with
  correct signed inertia. The QDLDL role is exactly "sparse inertia evidence";
  it cannot and must not replace the exact nonsymmetric direction solve.
- Remaining promotion work (design doc gates 2, 5, 8, 9, 11): epoch-level
  pattern reuse with varying sparsity of `A`, the unregularized
  five-equation direction residuals through PureKLU/UMFPACK, real conic
  medium-tier and bootstrap-SDP fixtures, memory-budget routing, and
  independent review. No HSD dispatch or Project.toml wiring exists yet and
  none is claimed.
