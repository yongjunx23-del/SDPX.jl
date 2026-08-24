# Equality-singleton substitution baseline

This directory contains a benchmark-local, arithmetic-generic reference for
the NativeSOC equality-singleton substitution.  It is deliberately independent
of the production elimination implementation: no `src/` file is modified and
no solve is attempted.

## Transform and fixtures

For a CSC equality matrix `B`, columns are scanned in ascending order.  A
column is eligible only when its CSC column has exactly one stored entry
(structural singleton, including explicit zeros).  At most one eligible column
is selected for each row.  For a selected `(row, column, value)`, the equality
is solved in the native arithmetic and represented as

```
x = P*u + q
```

The objective, SOC affine maps, equality system, primal map, and equality-dual
stationarity reconstruction all use this same typed `P` and `q`; no
`Float64` staging is used.  The dual oracle first scatters retained-row duals,
then recovers each pivot dual from its singleton column.  The valid fixture
has row degrees 1 and 3 and pivot values `1`, `-1`, and `1/2`.  Duplicate,
raw-near-singleton, zero-pivot, and tiny-pivot fixtures are expected to fail
closed.  The tiny arithmetic checks are repeated with one warm pass and nine
deterministic samples (the samples are oracle checks, not solver timings).

The reference guard uses an absolute `sqrt(eps(T))` floor after the structural
singleton test.  This is intentionally a diagnostic guard and is not asserted
to be identical to the production row-scaled pivot policy.

`BigFloat256` runs inside a 256-bit precision scope.  `Float64x4` is emitted as
a structured skip if the optional MultiFloats dependency is unavailable.

## nql30 structural probe

When the pinned `benchmark/data/cache/cblib/nql30.cbf.gz` checksum matches, the
driver parses it through the read-only CBF loader and applies the same column
singleton transform.  The baseline records the original/reduced dimensions,
DOF, Aeq/cone/objective nnz, pivot distributions, map and input fingerprints,
active Q columns, and dense workspace estimates.  For the canonical cache this
is 1,800 pivots (rows 1,881:3,680 and columns 2,702:4,501), with Aeq nnz
17,869 -> 14,269 and DOF 821 unchanged.  The predicted general-route main
workspace uses

```
8 * (2*n^2 + e^2 + n*e) bytes
```

for `n` variables and `e` equalities.  Dense traversal and active-column
iteration estimates are reported separately; they are planning metrics, not
claims about a sparse-aware production kernel.  The component probe is marked
`not_run_resource_bound` for nql30's large dense metric estimate, and the driver
never claims a solve succeeded.

## Reproducibility and comparison

Run from the repository root with one Julia and BLAS thread (the driver forces
BLAS threads to one):

```text
julia --project=. docs/evidence/bench/soc_equality_singleton/benchmark.jl \
  --output=work/baseline/soc_equality_singleton --samples=9
```

`results.toml` and `results.tsv` contain deterministic rows plus driver,
solver-source, commit/dirty, Project/Manifest, input, and environment hashes.
The source hash is checked before and after the run; a mismatch aborts the
baseline to avoid mixing solver revisions.  Compare rows only when arithmetic,
fixture, cache checksum, source hash, and sample count match.  The benchmark is
diagnostic: it does not measure end-to-end solve speed, RSS, or JIT-free wall
time.  Julia compilation/JIT and allocator state are process effects; RSS
would also include unrelated runtime pages, so those quantities are intentionally
not presented as solver memory or timing results.
