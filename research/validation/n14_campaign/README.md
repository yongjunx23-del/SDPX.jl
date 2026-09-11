# Bounded N14 cluster validation/solve experiment

User authorized moving remaining N14 tests AND solves to PBS on2026-09-07.
This is an experimental commit, not a promoted release. Base ca47c7d; solver
changes are the previously locally tested owned normalized-certificate candidate.
Main and deployed current remain untouched. No Power/JIT prototype included.

One8-core/64GiB PBS allocation,8h maximum, stages sequential:
1. Existing certificate/gap/layout/BigFloat/SIMD regressions, candidate/replay and
   cone checks, direct-affine canonical bit-parity checks.
2. Full package tests from a real clean Git checkout, offline dependencies.
3. Frozen CSDR guard (four threads; path substitution only) and N6 transformed
   solve with original-coordinate acceptance.
4. Build a fresh1024-bit QR-based basis from the COMPLETE1024-bit N14 source.
5. Fresh-process original and transformed N14 solves, each single-threaded,
   off equilibration, Float64x4,1e-8 tolerances,500 iterations/3600s solver limit.

N14 source: N14/L60/grid300/Q2048,9300x65, once-rounded Float64x4 rows from
1024-bit source generation. SHA19514baaf8e8a15f028d22f368c9bbd2a9765685916ef3dc6696ac91ad7d3e30.
Do NOT reuse the old256-bit-source basis for this input. The direct builder
preserves coefficients/constraints; the square triangular variable map removes
no columns or rows. Original-coordinate primal/dual/objective certification and
radius excess<=1e-8 remain authoritative for both modes.

Gate failure stops numerical dispatch. A genuine numerical no-solution exit4
is recorded and permits the other preplanned representation; an exception,
process/timeout failure stops the pipeline. No blind precision/tolerance/scaling
sweep or automatic retry. Logs, per-stage exits and scientific receipts persist.
Cold compiler/runtime/allocation/RSS observations are separate; no warmed-speed
or physical normalization/oracle/continuum claims. Source/model scalar types
are not silently downgraded.
