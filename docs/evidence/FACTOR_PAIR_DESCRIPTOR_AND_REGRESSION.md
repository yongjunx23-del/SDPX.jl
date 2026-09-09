# Factor-pair descriptor and partitioned regression

SDPX 0.6.1, tested source `93d48e4`, development/scientific-core-20260907.
No R0–R6 stage closure or public/default promotion.

## Completed slices

- Portable controls (`c82ad45`, isolated worker source `320bb24`): supported
  runtime retains positive tests; excluded runtimes assert actual refusal
  rather than skipped/broken tests. Parent reviewed the patch and included it
  in the supported-platform regression. Excluded-platform branches have not
  been executed on a second platform in this wave.
- Descriptor (`93d48e4`): `DenseFactorPairHSD` states structural counts,
  generally nonsymmetric dense core, partial-pivot LU, factor coordinates and
  a two-scalar homogeneous border. Dimension uses checked addition `n+m+2`;
  it has no numerical-rank alias.
- One opt-in planner seam now derives descriptor, payload, backend, LA,
  storage and parameter facts together. Default planner behavior is unchanged.
  Product rank is explicitly uncomputed (`rank=nothing` in diagnostics), not
  inferred from variable count or successful LU. Storage is now `:dense`;
  route remains `:factor_pair`, formulation `:dense_factor_pair_lu`.
- Failed-LU attempt metadata stays separate from factor completion/reuse.

Design review `d261e7a6` and implementation review `4cfe25a0` completed
read-only. The latter passed the bounded descriptor/default-dispatch scope at
`93d48e4`; it did not rerun tests or qualify the whole backend. Its nonblocking
symmetric-only provider-field issue and focused rank/default-metadata test gaps
were addressed in `c27e005`: 144/144 focused assertions passed, log
`/tmp/fp-metadata-c27e005.log`. This small follow-up was not independently rereviewed. Existing owner-currentness inference from Optimal,
rank/admission policy and complete peak-memory accounting remain open.

## Regression evidence

Three sequential owned Julia processes, each with a 180-second group deadline,
one Julia/BLAS/OMP/MKL thread, Julia 1.12.6 aarch64 Darwin, 2 GiB heap hint,
offline environment. Clean HEAD and loaded SDPX root asserted before/after.

At `93d48e4`:

- Part 1: 3736/3736, 108.0 seconds test time.
- Part 2: 196/196, 52.7 seconds.
- Part 3: 3323/3323, 156.9 seconds.
- Total: **7255/7255 assertions**, all 50 top-level test units assigned once;
  inventories match across processes. Common setup is repeated.

Logs: `/tmp/sdpx-suite-93d48e4-part{1,2,3}.log`.
Earlier same harness baseline `02116c7`: 7211/7211.
The exact executed harness was `/tmp/sdpx-partitioned-regression.jl`; its
selection logic is retained in `validation/scientific_core/run_partitioned_regression.jl`.

Set `SDPX_EXPECT_ROOT`, `SDPX_EXPECT_HEAD` (full clean commit), and
`SDPX_TEST_PART=1`, `2`, `3`, then run the harness with the pinned project.
Compare UNIT inventories and require PART_DONE for all three. Enforce process
limits externally. This is partitioned regression, **not** a one-process
all-suite run; it does not prove absence of cross-part process-state coupling.
The last historical one-process run remains 6825/6825 at `f74a465`.
