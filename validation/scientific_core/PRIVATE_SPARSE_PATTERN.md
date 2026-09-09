# Private scalar-LP pattern construction

Scope: only the existing unadmitted BigFloat scalar-LP research constructor.
No default/shared pattern constructor, global cache policy, provider ordering,
rank/precision/rounding/shift/refinement/acceptance gate or memory-admitting entry
is changed. `prepare_experimental_sparse_core_state` still always refuses.

The private builder validates nonnegative checked counts and canonical BigFloat/
Int CSC lengths, endpoints, monotone pointers and strictly increasing bounded
rows before indexing witnesses or allocating structural arrays. With n columns,
m rows, d=n+m<=64, a stored entries (including explicit numerical zeros), q=a+d,
it requests exactly these seven final Int-vector lengths:

    ar_colptr n+1, ar_rowval a, colptr d+1, rowval q,
    ar_slots a, theta_slots m, x_diag_slots n.

Their visible element subtotal is q+2a+n+2d+2. Final metadata consists of m
UnitRanges and m Symbols; the final numeric vector has q BigFloat slots. Each
column first stores its primal structural diagonal, then its original-A entries
in unchanged CSC order; each dual column has its one scalar Theta slot. Thus all
q slots are filled exactly once and the signature is the same existing signature
of the copied original structure/ranges/shapes. Existing `_core_write_ar!` and
Theta refills preserve values, signs, finite checks and owned scalar semantics.

A is borrowed only while reading. Its structure is copied directly into the
final private arrays; there is no `sparse(A)` intermediate, cache lookup,
publication, returned cached template, or subsequent private-template copy.
Changing one private pattern cannot poison another or the shared cache. The
normal shared constructor remains available and retains its existing protocol.

This removes one preparation dependency on arbitrary global dictionary state
and identifiable temporary structural/CSC overlap. It does NOT prove vector
backing capacities equal requested lengths, account for all wrapper/scalar/view
objects, MPFR/GMP primitive or deepcopy scratch, replacement/GC overlap, retained
caller outputs or RSS. Preflight range/block-size/witness/SPD scratch and provider
construction/refactor temporaries remain. The old numerical-slot inventory stays
an unproved diagnostic/research rejection heuristic, never admission authority.

Tests compare every field/signature/numeric value to the unchanged shared builder
for enabled/disabled cache, dense/block product cones, 256/512bits, explicit zeros,
empty structural shapes and the maximum d=64 structural shape. Empty/rank-deficient
shape parity does not grant numerical qualification. Existing full numerical,
mutation/revocation, natural-ordering and admission-refusal tests are rerun.
