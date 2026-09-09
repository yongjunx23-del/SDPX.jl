# R2-A: session-local symbolic reuse design

**Reference base:** `3303f1f1061b8da2e924d15216ed1ff7dd0614f0`. References below are from the prior read-only inspection. No implementation or test execution is claimed.

## Inherited decisions

- Initial scope: qualified `PreparedSolver{Float64}` execution using the **generic bordered symmetric core with actual provider `:cholmod`**. This is not general public `Model` preparation.
- Preserve cold iterates, fresh numerical workspaces, existing provider admission, original-coordinate reconstruction and certification.
- Retain at most one exclusive session-owned cache. Never put mutable factors in the global structure cache.
- Pass an explicit internal typed lease/context through the Prepared bridge. No public API expansion or task/global factor lookup.
- The supervisor approved retaining the **whole `SparseSymbolicNumericCache{Float64}`**, including unavoidable numeric storage, with numeric validity revoked between solves.

## Diagnosis and corrections to scout

The present chain creates a fresh sparse cache on every solve:

`_solve_prepared! → _bridge_sdp_solve → optimize implementation → native HSD → fresh ProductConeHSDState → fresh SymmetricCoreWorkspace → fresh SparseSymbolicNumericCache`.

Consequently the first factorization calls `ldlt` rather than retained-factor `ldlt!`. Global structure-cache hits share frozen pattern metadata, not provider analysis.

Three corrections:

1. **Retain the cache, not the workspace.** `_core_static_signature` includes `V/A/b/c` (`src/kkt/symmetric_core.jl:759-775`); its identity is valid within one solve, not across changed objectives/RHS.
2. **Global matrix-epoch monotonicity is unnecessary with unconditional revocation.** Existing core epoch handling already revokes numeric validity before refactorization. A fresh solve may start at epoch 1 provided the retained cache enters as `Prepared`, never `Fresh`, and the workspace is newly constructed and unsynchronized.
3. **Separate the measurement windows.** A fresh measured session should analyze once across 100 solves. A previously warmed measured session should analyze zero times across 100 further updates. These correct the scout’s measurement design; they do not relabel the current failing gate or change numerical tolerances.

## Implementable ownership seam

Introduce concrete internal types for:

- a session slot containing one optional retained entry;
- an entry containing the sparse cache and compatibility key;
- an exclusive attempt lease with active/closed state and checked attempt generation;
- an execution context carrying the optional lease and resolved execution/reduction descriptors.

Store the slot in `SolveState`, not `PreparedStructure`. Move ownership from slot to lease during execution; the idle slot must be empty while leased.

Retain the complete cache: CHOLMOD factor, factor-view CSC/numeric buffers, signs, regularization, signatures and cumulative cache counters. **This retains stale numeric storage, not symbolic-only memory.** Revocation makes those numerics unusable until refactorization; it does not erase them.

Construct fresh per solve:

- HSD state and cold iterate;
- pattern numeric values, cone operators and Newton system;
- symmetric-core workspace and current `V/A/b/c`;
- original-K snapshot, residual/direction buffers and homogeneous solution;
- receipts and result data.

Update the contrary ownership comments at `src/prepared.jl:1-10,87-99`.

## Exact call-chain seams

| Base location | Required change |
|---|---|
| `src/SDPX.jl:70-82` | Include internal lease/key/context definitions after sparse-cache types exist and before symmetric-core consumers. Avoid dependencies on later Prepared types. |
| `src/prepared.jl:88-99,434-446` | Add and initialize the optional typed session slot and attempt generation. |
| `src/prepared.jl:830-844` | Discard retention on structural mismatch while preserving existing error/bookkeeping. |
| `src/prepared.jl:889-954` | Acquire lease inside existing exclusive lock; pass context at bridge call `:938`; exception-safe finalize before releasing `busy`/lock. |
| `src/entrypoint_bridge.jl:302-321` | Accept/forward optional internal context. Keep fresh Model/Settings/Outputs and ordinary result adaptation. |
| `src/public/optimize.jl:375-400,421-460` | Thread context through the common internal optimize implementation. Public `optimize!` signatures remain unchanged and lease-free. |
| `src/hsd/native_hsd_public.jl:2615-2662` | Forward context; discard bordered lease before expanded restart at `:2645`; final retention eligibility depends on certified final result. |
| `src/hsd/native_hsd_public.jl:1599-1616,1700-1708,1794-1906` | Derive compatibility descriptors after existing canonical/equality/equilibration/rank gates. Early exits do not retain a factor. |
| `src/hsd/native_hsd_public.jl:2011-2022` | Forward context into generic state construction. |
| `src/hsd/product_cone_hsd.jl:225-263,468-555` | Forward context through `_product_cone_hsd_state` and `_prepare_product_hsd_symmetric_core`; specialized fixed-trace path remains ineligible. |
| `src/kkt/symmetric_core.jl:2168-2260` | Build fresh pattern/workspace; pass context into Float64 cache selection at `:2231`. Preserve preflights. |
| `src/kkt/symmetric_core.jl:2137-2156` | **Concrete reuse seam:** after ordinary disconnected-provider selection, authorize a compatible CHOLMOD entry or build a fresh sparse cache. A lease must not select the provider. |
| `src/kkt/symmetric_core.jl:895-955,1000-1077` | Preserve fresh workspace and synchronization invariants. Historical cache counters need not be zero. |
| `src/kkt/symmetric_core.jl:1135-1202` | Preserve refill/refactor/sync order; make the complete leased epoch attempt fail closed, including static validation and post-factor synchronization failures. |
| `src/factor_cache/routes/sparse_symbolic_numeric.jl:320-413,493-534` | Reuse existing numeric refactorization, detachment and revocation operations; do not weaken the same-epoch protocol. |

Keep the updated **original** problem returned at `prepared.jl:815` as the bridge input. Preserve certification through `_public_result_from_native_hsd` at `native_hsd_public.jl:2359-2454`.

## Compatibility and invalidation

Authorize reuse only after actual post-reduction structure and provider selection are known.

The key must cover:

- Prepared fingerprint and requested structural options;
- scalar type, effective precision and working-precision policy;
- requested/admitted/actual provider, actual route/core owner;
- requested thread budget and executed thread context;
- equality-reduction strategy/status/rank, row selections and pivots;
- product-rank strategy/status/rank and reduction basis/map identity;
- original/reduced/core dimensions;
- ordered cone kinds/parameters, partitions and block shapes;
- actual reduced `Ar` and lower-core CSC structure, slot maps and signatures;
- exact `dsigns`, symbolic epoch and configurable ordering policy;
- session structural generation.

Compare CSC arrays, partitions and signs exactly; a 64-bit signature alone is insufficient authorization.

Exclude ordinary `c/b`, Theta, RHS, affine particular solutions and regularization magnitude. Recompute numeric data every solve. Do not use the workspace’s `b/c`-dependent static signature as the cross-solve key.

A changed rank, reduction selection/basis, cone layout or actual CSC structure must miss even if dimensions match. Provider/precision/thread incompatibility must discard, never trigger coercion.

Preserve the scout’s conservative structure-cache clear/disable invalidation through locked generation snapshots, not a global factor registry. An active owner may finish; generation change prevents check-in. Idle entries are discarded on next access.

## Lease, epochs and failures

### Checkout

Under the existing `trylock`/`busy` guard:

1. Advance attempt generation, including failed attempts.
2. Move any entry into the lease and clear the idle slot.
3. Discard `Invalid`, `Failed` or `Factoring` entries.
4. **Unconditionally revoke a healthy retained entry to `Prepared`, with matrix epoch zero.**
5. Validate inputs and executed compatibility; mismatch discards and prepares fresh.

Register a newly created cache with the lease before subsequent construction can throw.

### Fresh workspace and first epoch

The new workspace starts unsynchronized, without receipt or homogeneous solution, and with fresh workspace epochs/buffers. Retained cache counters/factor epoch remain cumulative.

Use the existing numeric sequence:

`validate → revoke → refill current operator → recompute regularization → factorize → synchronize current original-K snapshot → solve current homogeneous system`.

The same-epoch shortcut at `sparse_symbolic_numeric.jl:348-354` only applies when admission state was `Fresh`. Checkout revocation therefore prevents a new solve’s epoch 1 from reusing an old solve’s epoch-1 numeric factor. Fresh workspace synchronization independently prevents stale snapshot/receipt reuse.

### Check-in

Retain only after final result construction yields certified `Optimal`, the executed route remains eligible, and the cache is healthy.

**Unconditionally revoke again before returning it to the slot.** Validate the attempt token, close the lease and clear lease references.

### Exceptions and fallback

- Provider numeric failure detaches the factor. Recovery must reanalyze and count again.
- Static-validation, refill, sync or homogeneous-solve failures must clear workspace acceptance artifacts and detach the leased factor.
- Construction/programming errors, interrupts, failed certification and nonoptimal results discard the entry.
- Native paths can return failure statuses without throwing; successful function return alone is not retention authorization.
- Explicitly discard/revoke the bordered lease **before expanded restart** (`native_hsd_public.jl:2635-2658`), and pass no CHOLMOD lease to expanded execution. Preserve existing fallback policy.
- Nest lease cleanup inside the existing unlock `finally`; cleanup must not strand `busy`, retain an active owner or mask the primary exception.

Receipts remain diagnostics, not reuse tickets or mathematical certificates. Results must contain no references to retained cache/lease/workspace storage.

## Required tests

Update `validation/scientific_core/test_r2a_symbolic_numeric_separation.jl` deliberately, without weakening counters or tolerances.

### Two provider-attributed 100-solve gates

- **Cold100:** warm compilation using an independent session; measure 100 solves of a fresh session. Require actual `:cholmod`, generic bordered execution, stable key, no fallback/recovery: **total symbolic delta = 1**.
- **Warm100:** after a successful solve of the measured session, measure 100 further updates: **symbolic delta = 0**.
- Require genuine numeric factorization on each solve that executes a factor epoch.
- Use a fixture whose connected components exceed the disconnected cache cutoff of four. Disconnected dense LDL is **not sparse-reuse evidence**.

### Safety and correctness controls

- Current `c/b` consumed; original-coordinate residuals/objectives/certificates satisfy existing tolerances.
- Compare against cold reference solves outside counter windows.
- Verify deterministic cold initialization despite `previous`.
- Explicit epoch-1 collision with changed operator: numeric count increases, symbolic count does not, stale solve/sync rejected before refactor.
- NaN, zero-pivot, refill/static-validation, post-factor sync, construction, certification and interrupt failures; verify detached/non-solvable state and released lease/lock.
- Recovery after detachment reanalyzes truthfully.
- Border-to-expanded restart discards retention.
- Structural changes, same-size/same-`nnz` CSC drift, signs, cones, rank/reduction/basis changes, precision/provider/thread changes and clear/disable invalidate.
- Where public Prepared validation rejects changes early, pair public rejection tests with focused internal compatibility tests.
- Prior results remain unchanged across later solves; mutating prior result arrays cannot affect subsequent solves.
- Two concurrent sessions have distinct factors/buffers and independent failure behavior. Same-session concurrent/reentrant use retains existing rejection semantics.
- At most one retained entry per session; closed leases and results retain no extra cache owners.

Measure global counters in isolated windows or aggregate concurrent expectations. Existing instrumentation increments after `ldlt` returns; distinguish pre-provider failure from returned unsuccessful factors and provider calls that throw before return.

## Risks and handoff

Main risks are stale epoch fast paths, internally swallowed failures, weak post-reduction keys, result aliases and misleading lifetime-counter diagnostics. Retention also increases session memory; make no allocation-free or immediate-RSS-release claim.

**No outstanding supervisor decision.** Implementation is warranted and remains parent-owned. No implementation, test run, symbolic-reuse success or gate pass is asserted here.
