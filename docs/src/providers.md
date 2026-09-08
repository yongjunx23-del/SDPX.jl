# Linear-algebra providers

Providers implement factor/solve operations for an already frozen SDPX
operator. They do not own canonicalization, cone algebra, route planning,
Newton equations, fallback policy, or mathematical certification.

```text
Canonical program + product-HSD state
              |
              v
        NewtonSystem
              |
              v
     SDPX route/session
              |
              v
      provider factor/solve
              |
              v
 exact-operator residual check
              |
              v
 original-coordinate certificate
```

## Provider responsibilities

A provider may supply:

- dense or sparse factor construction;
- single, transpose, and multiple right-hand-side solves;
- factorization status and numerical diagnostics;
- applicable inertia information; and
- explicit factor invalidation.

SDPX still verifies finite data, generations, solve status, backward error, the
five Newton equations, and terminal certificates.

## Available provider families

### Julia and SuiteSparse

Julia `LinearAlgebra` supplies standard dense Float64 operations. SuiteSparse
provides supported Float64 sparse factorization. These implementations are not
used as an implicit high-precision fallback.

### MultiFloatLinearAlgebra

MFLA owns fixed-width extended dense/local factor and solve kernels. SDPX uses
its public factor-handle, multi-RHS, transpose, residual, refinement, and
invalidation interfaces. Provider internals are not copied into SDPX.

### BigFloatLinearAlgebra

BFLA owns BigFloat dense/local kernels and precision-preserving factor state.
SDPX must preserve independent MPFR ownership and uniform configured precision
through every input, workspace, and destination.

### Current high-precision policy

MultiFloat and BigFloat use MFLA/BFLA dense or block-local factors. Generic
high-precision sparse factorization is not currently a production capability.
A sparse request without a supported exact provider fails closed or follows an
explicit same-arithmetic dense/bordered plan; it never selects Float64.

### Bounded internal sparse signed-LDL adapter (R3, not native routing)

`SDPX.SparseQDLDLCache` is an INTERNAL adapter only: it bridges the existing
provider-neutral cache lifecycle to the optional QDLDL-backed MFLA/BFLA
sparse-LDL providers without kernel duplication or a new backend, and it is
not wired into any public `optimize!` route. Native high-precision sparse
routing remains unqualified, and BigFloat `sparse_augmented` Settings stay
disabled (`kkt_route=:sparse_augmented` still requires Float64 CHOLMOD
arithmetic).

Caller contract: the caller supplies an explicitly eligible (e.g.
caller-shifted) symmetric quasi-definite upper-triangular operator. The raw
augmented core `K = [0 Ar'; Ar -Theta]` is NOT quasi-definite as stored
(structural zeros on the reduced-x diagonals), so it must never be sent
here; the original operator stays separate and remains the residual
authority. Construction freezes the BigFloat working precision; every
`factorize!` requires the current ambient precision and all input values
to match it before any same-epoch reuse. Wrong element/index/storage
types hit a rejecting fallback that revokes authority first. A reused
`matrix_epoch` promises an unchanged operator (the numeric refactor is
skipped without comparing values). Storage, pattern, finiteness and precision
violations throw; a changed finite operator under the same epoch violates the
caller's promise and is not detected. Any failed factorization leaves the
cache `Failed` with no stale solves.
Ordinary solves use the provider checked (slot-repairing) solve, so
arbitrary caller-owned destinations are safe and no trusted path exists
here.

Explicit `ordering=:natural` on the internal `SparseQDLDLCache` and the
unadmitted research constructor requires BFLA's ordering-capability API. It
uses the reviewed provider's `perm=nothing` path. Omission retains AMD,
including with legacy BFLA and MFLA; unsupported natural ordering rejects,
never falls back. Frozen cache ordering is checked against provider
provenance and permutation-state sanity before reuse/solve, and reported in
diagnostics. Natural ordering does not enable memory admission.

### INTERNAL EXPERIMENTAL sparse symmetric core (R3 bounded, not native routing)

`SDPX.ExperimentalSparseCoreCache` plus the experimental seams in
`src/kkt/symmetric_core.jl` form an INTERNAL, explicitly experimental
implementation over the adapter above. **Memory admission is unavailable:**
`prepare_experimental_sparse_core_state` rejects before pattern, workspace,
provider or cache construction, even with a large declared capacity.
`experimental_sparse_core_memory_inventory` returns `proven=false` and named
missing components; its hypothetical natural-order payload counts are not
byte bounds and do not qualify the current AMD provider.

Controlled numerical tests explicitly use the private, **unadmitted** research
constructor `_research_prepare_experimental_sparse_core_state`. There is no
fallback to it from the memory-admitting entry. Its mathematical scope is
BigFloat-only scalar LP, dimension at most64, identity coordinates, sparse
original `A`, a checked triangular original-A witness, semantic SPD Theta,
and an explicit positive precision-matched shift. Its local capacity
heuristic may reject research work but never certifies a memory budget.
SOC and other cones remain unsupported.

Private research construction copies pattern metadata instead of exposing
the normal core's shared read-only cache arrays. A malformed experimental
pattern therefore cannot poison another workspace or the global cache; the
default cache policy is unchanged. The wrapper owns independent frozen
map/diagonal/sign/shift/Ar-slot authority (verified before every reuse,
refactor, sync-match, and solve),
shifted upper factor values, an independent unshifted original snapshot,
last-successful epoch evidence that survives revocation, and exact owned
static snapshots (original `A`, `b`, `c` — exact BigFloat values and
precisions, never lossy sampled hashes). Static pattern Ar slots are read
back through the slot mapping against the admitted `A`, and structural x
zeros must be exact, before every factorization; every upper slot is
checked against its always-computed expectation (a missing diagonal shift
reads as a mismatch, never a match), including on direct wrapper solves.
Exact upper CSC dimensions/pointers/indices and array lengths are checked
against the inner cache's independently owned construction copies. Formation
precision AND rounding mode are frozen and required for refactor, reuse,
matching, and solves; this internal route never changes ambient settings.
Signatures avoid Float64 narrowing, but hashes can collide and are only
diagnostic: exact checks are the authority. The setup lower pattern buffer is never shifted and structural
zeros are preserved. Factorization is genuinely sparse (no dense
`K`/`Theta`/RRQR fallback); the only numeric factor entry is the pattern
seam, and direct `factorize!` calls revoke and throw. Receipts bind the
actual provider, frozen precision, declared shift, original pattern, and
both epochs with `proof_valid=false`. Same-epoch operator conflicts are
refused without refactoring; the whole epoch and the whole
solve/acceptance run inside fail-closed revocation transactions with no
manual cleanup. Every published direction passes the genuinely shared
five-equation acceptance helpers in `src/hsd/product_cone_hsd.jl` that the
production gate calls itself. The memory inventory includes named provider
shapes and dense worst-case `L` fill, but a complete simultaneous-live upper
bound is NOT yet established (ordering/solve allowances and returned-direction
and acceptance temporaries remain to be reconciled). Passing its arithmetic
gate is not proof of storage sufficiency. This remains an integration blocker;
JIT, GC, BLAS, and total-process RSS are outside its claimed scope. Existing
`_core_refine!` targets and two-correction caps are unchanged. The default
path, public Settings, and native routing are unchanged, and no
sparse-scalability claim is made.

`test/experimental_sparse_core.jl` checks unavailable memory admission;
`experimental_sparse_core_numerics.jl` explicitly exercises unadmitted numerical
research. `experimental_sparse_core_identity.jl` combines both and exercises
actual stored operators at256/512 bits, directed-rounding transitions, live CSC mutations, direct
failure revocation, and faults injected AFTER real provider factor/solve
operations inside the epoch driver. These are narrow authority regressions,
not production or memory-budget qualification.

## Factor receipts

A `FactorReceipt` binds:

- provider and route;
- operator generation;
- factor generation;
- factor attempt and certified factor outcome;
- applicable inertia evidence; and
- invalidation state.

Changing the matrix or provider invalidates the receipt immediately. Each new
right-hand side still requires finite checks and residual validation. A factor
receipt is never an optimality or infeasibility certificate.

## Selection and fallback

Provider selection is part of the frozen execution plan and depends on:

- arithmetic type and precision;
- matrix structure and symmetry;
- sparse pattern and fill estimate;
- required solve modes;
- memory limit;
- thread ownership; and
- provider availability and capability.

Package availability alone must not silently change the mathematics or narrow
arithmetic. Same-iterate fallback is authorized by route policy and recorded in
planned/executed diagnostics.

## Thread and precision ownership

Only one parallel layer should own a phase: Julia outer workers, BLAS, or the
provider. Provider threads are accounted for by the shared `ThreadBudget`.

Float64, MultiFloat, and BigFloat use the same Newton equations. A missing
high-precision provider yields an unsupported/fallback decision in the same
arithmetic; it never triggers a silent Float64 solve.

Production defaults change only after end-to-end certificate, accuracy,
allocation, RSS, and benchmark evidence on one frozen source SHA.
