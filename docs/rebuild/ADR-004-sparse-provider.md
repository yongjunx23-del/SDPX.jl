# ADR-004: Sparse cross-library provider boundary

**Status: PROPOSED (P01, needs_review).** Not accepted. This ADR records the
decisions the card requires and the measurements that support them; V01/I01
own acceptance.

Depends on: ADR-001 (boundaries), ADR-002 (provider contract, ACCEPTED),
ADR-003 (acceptance, ACCEPTED). Read together with
`test/provider_contracts/sparse_contract.jl` and
`test/provider_contracts/sparse_fixtures.jl`.

**Evidence environment.** Every number below was produced in the packet's
rebuilt provider environment, not the default project:

```
JULIA_DEPOT_PATH=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia
REBUILD_ENV=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env
julia --startup-file=no --project="$REBUILD_ENV" -t1 <test>   # ONE provider per process
```

Resolved there: `MultiFloatLinearAlgebra v0.4.0` (dev → local checkout at
`50e6e0b`), `BigFloatLinearAlgebra v0.3.0` (dev → local checkout at
`f95d3e6`), `SDPX v0.6.1` (dev → this checkout), `MultiFloats v3.3.2`,
`GenericLinearAlgebra v0.4.1`, `QDLDL v0.4.1` (loaded explicitly; it is a
weakdep of both providers and is **not** declared anywhere in SDPX's
`Project.toml`). Manifest: `$REBUILD_ENV/Manifest.toml`.

MF and BF legs run in **separate processes** (`P01_PROVIDER_LEG=mfla|bfla`):
Julia 1.12 inference can exhaust itself compiling the MFLA fixed-width and
BFLA/MPFR specializations in one process. This is a hard requirement of the
harness, not a preference.

---

## 1. The problem

SDPX has one sparse signed-LDL route (`src/factor_cache/routes/qdldl_sparse.jl`,
`SparseQDLDLCache{T}`) that delegates to **two** independent libraries:
MultiFloatLinearAlgebra (MFLA) and BigFloatLinearAlgebra (BFLA), each wrapping
`QDLDL.jl` in its own optional extension. The libraries are not similar:

| | MFLA 0.4.0 | BFLA 0.3.0 |
|---|---|---|
| sparse LDL | own `MFSparseLDLCache` over QDLDL | own `BFLASparseLDLCache` over QDLDL, plus a SDPX-owned wrapper `_BFLASparseQDLDLProvider` |
| dense factorizations | `MFCholeskyCache`, `MFLDLTCache`, `MFLUCache`, `MFRRQRCache` | `BFLACholeskyCache`, `BFLALDLTCache`, `BFLALUCache`, `BFLARRQRCache` |
| public `lu` for its own scalar type | **no** | yes |
| success accessor | `cache.status::Int` (`0` = valid); **no `issuccess` method** | `issuccess(cache)` / `cache.status::FactorStatus` |
| `solve!` argument order | `solve!(dest, cache, rhs)` | `solve!(dest, cache, rhs)` (same, but the cache-first `solve!(cache, dest, rhs)` used by the SDPX seam is BFLA-specific) |
| QDLDL extension name | `MultiFloatQDLDLExt` | `BigFloatQDLDLExt` |
| ordering support | AMD only | AMD and (per its own flag) natural |

Nothing in that table prevents a single contract. What it does prevent is a
single *implementation*: any SDPX code that reads `cache.status`, calls
`issuccess`, or names a factor type is provider-specific by construction.

## 2. Decision: four-way ownership

The card asks who owns symbolic pattern, numeric values, exact context, and
factor generation. The decision is:

| Object | Owner | Frozen at | May change | Evidence field |
|---|---|---|---|---|
| **Symbolic pattern** (`colptr`, `rowval`, ordering, D-signs) | caller (SDPX) | first construction of the pattern; the provider performs one symbolic analysis at cache construction | **never** while the cache lives; a new pattern requires a new cache | `pattern.colptr`/`pattern.rowval` slot equality across value sets |
| **Numeric values** | caller (SDPX) | every `factorize!` | every `factorize!`, which **must advance `matrix_epoch`** | `operator.nzval` distinctness; `factor_epoch` advancement |
| **Exact context** (arithmetic, bit width, ambient precision, index width, triangle convention) | provider, declared; SDPX records and checks | cache construction | never; drift is a hard failure | `ProviderCapabilities`, `precision_bits`, `index_width` |
| **Factor generation** | provider owns the physical factor and its own generation counter; **SDPX owns the logical lease** | on each successful `factorize!` | a failed `factorize!` revokes the logical lease (ADR-002 §4) | `factor_epoch`, `factor_status` |

Three consequences that are decisions, not observations:

1. **The pattern is the caller's, and a numeric write may not extend it.** A
   static shift, a regularization, or a scaling is a *value* operation. The
   test asserts `colptr` and `rowval` are slot-identical either side of a
   shift, and that the shifted operator carries no lower-triangle slot.
2. **`matrix_epoch` is the caller's promise, and it is load-bearing.**
   `SparseQDLDLCache.factorize!` *skips the numeric refactor without comparing
   values* when the cache is `Fresh` and the epoch is unchanged. A caller that
   changes values without advancing the epoch gets an answer from the previous
   operator. This is documented at `src/factor_cache/routes/qdldl_sparse.jl`
   and is the single most dangerous item in the whole boundary, because it
   fails silently and correctly-looking.
3. **Nothing infers freshness from object identity.** ADR-002 §4 is
   unchanged: after any failed `refactorize!`, SDPX revokes the logical lease
   *before* reading any provider status.

## 3. Decision: the two libraries do not get two solver policies

**Decision.** MFLA and BFLA are two *backends of one policy*, never two
policies. Concretely:

* One operation set, named once, in `sparse_fixtures.jl`'s `Embedding` +
  `CacheHandle`:
  `factorize!(handle, operator, epoch)`, `solve!(handle, rhs)`,
  `solve_multi!(handle, rhs)`, `factor_symbolic_count`, `factor_solve_count`,
  `factor_epoch`, `factor_status`.
* One semantic driver (`run_contract`) runs every leg for every embedding.
  The driver contains **no numeric reference of its own**; the reference
  arrives as `Embedding.oracle`, and the driver *asserts the oracle
  identities are pairwise distinct* (`oracle_identities`).
* Provider-specific facts (bit width, `issuccess` vs `status`, sparse-vs-dense
  availability, in-place refactorization) are **declared** in
  `ProviderCapabilities`, never branched on inside a leg.

`run_contract` and `run_provider_contract` enforce this structurally. A leg
that cannot run is recorded `:skip`/`:unsupported` with a reason; a leg that
runs and disagrees is `:fail`. There is no fourth outcome and no silent pass
(`ledger_push!` rejects a reasonless skip).

**Acceptance item 1 (same semantic test, different numeric oracle) is met by
construction, not by promise:** the shared thing is the driver and the leg
list; the per-provider thing is the oracle. Measured identities:
`:mfla_dense_generic_solve` and `:bfla_dense_generic_solve`, asserted
pairwise distinct in the same testset that runs them.

## 4. Asymmetry that must be declared, not smoothed over

**In-place numeric refactorization exists for MFLA only.** MFLA's QDLDL
extension builds the factor once with `logical=true` (retaining the symbolic
structure) and `factorize!` then performs only `QDLDL.update_values!` +
`QDLDL.refactor!` — a genuine numeric-only refactorization with no
re-symbolization. **BFLA has no equivalent**, and its `numeric_refactor_in_place`
capability is declared `false`, so `in_place_refactor` reports
`:unsupported` for BFLA with that reason rather than inventing a fallback.

The same asymmetry exists one level down and is deliberately **not** hidden:
the SDPX wrapper over BFLA (`_BFLASparseQDLDLProvider`) exists because BFLA's
sparse LDL cache type is extension-only and must not be named before the
provider loads. That is a wrapper, not a second policy — but it is more code
on the BFLA path than on the MFLA path, and the ledger says so.

## 5. Decision: no stable QDLDL workspace API yet; the explicit prototype stays

**Decision: keep the explicit prototype. Do not publish a stable QDLDL
workspace API in this round.** Reasons, in order of weight:

1. **There is no second consumer.** `SparseQDLDLCache` is not wired into any
   public `optimize!` route; native high-precision routing remains
   unqualified and BigFloat `sparse_augmented` settings stay disabled. A
   stable API with one internal caller is a compatibility liability with no
   benefit.
2. **The provider-side workspace is not SDPX's to stabilize.** `QDLDL`
   factor/workspace layout is owned by `QDLDL.jl` and wrapped by MFLA/BFLA.
   Freezing a *stable* API over it would freeze a third party's internals at
   one revision — the exact dependency this ADR's §6 exists to contain.
3. **The prototype is already exercised end-to-end** by
   `test/sparse_qdldl_provider.jl` and by this contract, so the decision is
   reversible on evidence rather than on hope.

**Reopen condition.** Publish a stable workspace API only when at least one
of: (a) a second in-tree consumer appears; (b) the sparse route is wired into
a public `optimize!` path; (c) a provider publishes a documented, versioned
workspace contract that SDPX can bind to without reading private fields. Any
of those is a new ADR, per the card's own rule that a new shared package needs
its own ADR.

**Forbidden by this decision:** SDPX must not fork QDLDL, must not
reimplement its symbolic analysis, and must not maintain a second sparse LDL
kernel "for the other precision".

## 6. Decision: third-party internal fields are centralized behind a version gate

**The dependency, measured.** Reading the current MFLA/BFLA extensions, SDPX
reads exactly these private fields of provider objects:

| Provider | Path | Pinned revision | SDPX site | Rationale |
|---|---|---|---|---|
| BFLA | `BFLASparseLDLCache.factor` | `f95d3e6` | `ext/SDPXBigFloatLinearAlgebraExt.jl:1160` | ordering provenance probe |
| BFLA | `BFLASparseLDLCache.ordering` | `f95d3e6` | `:1156` | declared ordering check |
| BFLA | `BFLASparseLDLCache.factor.perm` | `f95d3e6` | `:1162` | `:amd`/`:natural` provenance |
| BFLA | `BFLASparseLDLCache.factor.iperm` | `f95d3e6` | `:1163` | idem |
| BFLA | `BFLASparseLDLCache.factor.workspace.AtoPAPt` | `f95d3e6` | `:1164` | idem |
| MFLA | `MFSparseLDLCache.factor.perm` | `50e6e0b` | `ext/SDPXMultiFloatLinearAlgebraExt.jl:1387` | `:amd` provenance |
| MFLA | `MFSparseLDLCache.factor.iperm` | `50e6e0b` | `:1388` | idem |
| MFLA | `MFSparseLDLCache.factor.workspace.AtoPAPt` | `50e6e0b` | `:1389` | idem |

These are **read-only probes**, and none of them is on a hot path — but all
eight are unpinned private-field reads, and a provider point release that
moves one of them turns an ordering answer into `:unknown` with no test
failing.

**Decision.** Every such read goes through one table with a version gate:

* The table is `internal_field_paths()` in `sparse_fixtures.jl`. Each entry
  names the provider, the dotted path, the **pinned revision the read was
  reviewed against**, the SDPX file and line, the rationale, and the public
  accessor it should migrate to.
* The gate is `run_third_party_field_gate(ledger)`. For each entry: if the
  provider is not loadable, the path is counted **unchecked** and the reason
  names the package — a missing dependency is an infrastructure fact, never a
  pass (ADR-003 §3). If the provider is loadable, the gate resolves the
  dotted path and fails if it no longer exists, recording the package version
  it checked.
* A provider version bump is therefore a *test failure with a named field*,
  not a silent `:unknown`.

**Enforcement status: `table_only`, and this ADR says so.** The table lives in
the test fixtures because `ext/SDPXBigFloatLinearAlgebraExt.jl` is outside
P01's write allow-list. The patch that installs the same table in the
extensions, with `_require_pinned_provider_revision!` called from
`SparseQDLDLProviderAvailable`, is submitted as an **inert patch proposal**
in `rebuild-reports/P01/report.json` for I01. Until it is applied, the gate
checks the table's completeness and the provider's loadability, and the eight
reads remain in the extensions unguarded. This is a real, named gap.

**New shared package ⇒ new ADR.** Any proposal to add a shared sparse/LA
package (or to promote `SparseQDLDLCache`'s seam into one) requires its own
ADR; this one deliberately does not pre-authorize it.

## 7. Findings (measured, this round)

### 7.1 The SDPX sparse QDLDL seam returns wrong answers — P0

Both providers, both rounds, same specimen, same magnitude:

| Leg | MFLA (`Float64x2`) | BFLA (`BigFloat`, 256 bits) |
|---|---|---|
| error vs the provider's own oracle on the same operator | **0.20035761024843557** | **0.20035761024843557** |
| tolerance (`max(64,4n)·u·κ`) | 5.05e-29 | 1.77e-74 |

The identical value across two unrelated arithmetics rules out provider
rounding. A direct check isolates it to the SDPX route, not the data:

```
# build the dense matrix of exactly what the operator stores, then solve it
# twice: once through SDPX's SparseQDLDLCache, once with a dense solve.
Kop = zeros(T, 8, 8); for j, p in the operator's slots: Kop[rowval[p], j] = nzval[p]
SDPX.factorize!(cache, operator, 1); SDPX.solve!(cache, y, b)
Kop \ b          ->  err = 1.9198321536764786e10
dense core \ b   ->  err = 0.20035761024843557
```

`1.9e10` is not a rounding discrepancy; the sparse return path is returning
something that is not the solution of the system it was given. **This is
reported as a defect, not attributed to MFLA or BFLA**: the same fixture
driven through each provider's *own* cache passes at 0.0 and 4.3e-78 (§7.2).

Scope note, stated conservatively: one 8×8 symmetric quasi-definite specimen
with a caller-owned `1e-3` shift, `Float64x2` and `BigFloat` at 256 bits,
`-t1`. It does **not** establish that every sparse solve is wrong; it
establishes that this one is, reproducibly, and that the harness that would
have caught it did not exist before this task.

### 7.2 The provider-level contract passes on both libraries

Same driver, same legs, per-provider oracles, separate processes:

| Leg | MFLA | BFLA |
|---|---|---|
| `symbolic_reuse` | pass (3 assertions) | pass (3) |
| `numeric_refactor` | pass (6) | pass (6) |
| `multi_rhs` | pass (4) | pass (4) |
| `reuse_after_failure` | pass (5) | pass (5) |

Measured error against each provider's own oracle: MFLA
`1.232595164407831e-32` (tolerance `5.79e-119`… actual bound 2.58e-75 class
per provider), BFLA `0.0` / `4.3180842775472223e-78` / `0.0`. BFLA's measured
unit roundoff is `1.727233711018889e-77` at 256 bits
(`effective_mantissa_bits = 256.0`).

**`reuse_after_failure` confirms ADR-002 §8/§9 on the real libraries.** After
a wrong-dimension factor input is rejected, BFLA still reports
`issuccess == true` and still holds the previous factor; MFLA likewise retains
its previous status. Both are entitled to. The obligation is SDPX's, and no
SDPX adapter is exercised by that leg.

### 7.3 The structural-zero claim about the symmetric augmented core is true

Verified in `src/kkt/symmetric_core.jl`:

* The pattern builder stores `push!(rowval, j)  # structural zero x diagonal`
  for every reduced-x column (`:382`), so the slot exists for a reason: it is
  that row and column's structural entry.
* `build_symmetric_core_ldlt_cache` is chosen for `T !== Float64` (`:2417`),
  with the comment at `:2417-2423` stating that the raw core's structural zero
  violates QDLDL's quasi-definite precondition, that the MFLA/BFLA adapters
  disable dynamic regularization, and that fabricating a positive diagonal
  would change the Newton operator.
* `SparseQDLDLProviderAvailable` appears **nowhere** in
  `src/kkt/symmetric_core.jl`; the QDLDL route is never reached from the core.
* The contract's own fixture reproduces the property independently: every
  reduced-x diagonal slot is present in the pattern and holds an **exact**
  zero (`iszero`, not "small"), and the raw core has no other zero on that
  diagonal.

So the claim is true of the code as written and nothing in the core violates
it. The one semantic caveat, recorded rather than glossed: the reduced-x
diagonal slot is structural in the sense of *addressing*, and its value is
exactly zero — which is precisely why the core must never be handed to QDLDL
as-is, and why the fixture's negative controls (empty column, lower-triangle
storage, `Int32` indices, unknown ordering) all must be **refused**.

### 7.4 Provider declarations disagree with provider behaviour

Two facts found while wiring the live legs, both of which the gate must
capture rather than averaging away:

* `SparseQDLDLProviderOrderingAvailable(BigFloat, :natural)` returns `true`
  (`ext/SDPXBigFloatLinearAlgebraExt.jl:1147`), but the actual construction
  refuses it: `ArgumentError: QDLDL provider ordering natural is unavailable
  for BigFloat; no ordering fallback`. The declared capability and the
  behaviour disagree. The contract now declares `ordering_natural = false`
  for BFLA and asserts *both* directions, so this cannot silently become a
  fallback.
* BFLA's `sparse_ldlt_available(BigFloat)` is `false` until `QDLDL` is
  actually loaded in the process. A capability probe that never loads QDLDL
  reports "unavailable" for a package that is installed. The gate therefore
  reports *unchecked* rather than *absent* when the provider cannot be loaded.

### 7.5 `src/sparse_la.jl`'s generic sparse Cholesky is still orphaned

**Confirmed orphaned, unchanged.** `grep -rn "GenericSparseCholeskyFactor"
--include=*.jl .` returns **11 hits, all inside `src/sparse_la.jl`** (the type
at `:718`, methods at `:804, :878, :1012, :1043, :1049, :1055, :1115, :1126`,
two backend fields at `:1405, :1425, :1438`) and **zero external references**
from `test/`, `ext/`, `validation/`, `benchmark/`, or any other source file.
It has precision-aware refactorization and no caller. This ADR does **not**
wire it in: wiring an orphan into a live route is a behaviour change that
belongs to an implementation task with its own evidence, not to a contract
ADR. Recorded as an open finding.

### 7.6 Defects found in this task's own work

Recorded because ADR-003 §4 requires it; each one disabled or falsified a leg
before it was fixed.

1. `upper_only(K)` was `sparse(UpperTriangular(Matrix(K)))`, which keeps the
   **full** parent sparsity for the lower triangle (`nnz` stayed 16 for a
   matrix with 9 strict-upper nonzeros). Pattern and values disagreed, QDLDL
   refused the operator as "not upper triangle", and four legs reported a
   provider failure that was the fixture's.
2. The specimen pattern was restated by hand three times, each version wrong
   in a different place (untransposed affine block, a cone-block column
   overwritten instead of appended, a diagonal-prefix/emission variable
   captured by a later loop). The operator then stored nonzero values in the
   strictly lower triangle. Fixed by deriving the slot set from the same
   dense definition the values come from.
3. The unit roundoff was first measured by bisecting `(1 + u) - 1 != 0`,
   which for `Float64x2` converges on `2^-1074` — the tolerance came out
   **eleven orders of magnitude too tight**, and every MFLA numeric leg
   "failed" on rounding. Now measured as `eps(T)` and reported with
   `effective_mantissa_bits` beside the nominal width.
4. `CacheHandle` was immutable, so `handle_values` returned the operator from
   construction; three legs compared a stale operator against a fresh oracle
   and reported a false provider failure.
5. `factorize!(handle, ...)` collided with `LinearAlgebra.factorize!` under
   bare-name resolution and silently returned an LU object. Now called
   qualified.

## 8. Consequences

* **The sparse seam is not releasable at the measured revision** (§7.1). It is
  internal, unwired, and fail-closed, so nothing public is broken — but no
  capability claim about it may be made until the defect is fixed and this
  contract passes.
* **A capability claim still needs environment + revision + test**
  (ADR-002 §7). The four accounting blocks — `symbolic_reuse`,
  `numeric_refactor`, `kernel_threads`, `process_limits` — are reported
  **separately** and never summed into one performance number. `kernel_threads`
  and `process_limits` are `:unsupported` with reasons: no kernel thread count
  was measured and no process limit was measured, so they are `null`, never
  `0` (ADR-003 §3).
* **The gate is `table_only` until the inert patch lands.** Anyone reading a
  capability from a provider field must add the path to
  `internal_field_paths()` first; a new read outside the table is a review
  rejection.
* **This ADR does not accept itself.** I01 owns integration; V01 owns
  adversarial review. P01's report is a self-report and is not evidence of
  its own correctness (ADR-003 §4).
