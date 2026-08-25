# MOI conformance status (PR11)

## What passes

The existing `test/moi.jl` suite (166 tests) covers the cones SDPX
natively supports — Nonnegatives, SecondOrderCone,
PositiveSemidefiniteConeTriangle, and scalar inequalities — through
the `MOI.copy_to` (non-incremental) interface. `supports_constraint`
declarations match the real executable capability.

## Known limitation: incremental interface

`MOI.Test.runtests` (the standard conformance suite) drives the
optimizer through the **incremental** interface (`MOI.add_variable`,
`MOI.add_constraint`, ...). SDPX implements `supports_incremental_interface
= false` and uses `MOI.copy_to` instead, so the standard conformance
suite cannot run directly on SDPX: every incremental call raises
`AddVariableNotAllowed` / `AddConstraintNotAllowed`.

## Path to full conformance

Implementing the incremental interface (or a bridging layer that
accumulates incremental calls and copies them in bulk) would let
`MOI.Test.runtests` run. This is tracked as follow-up work; it is not
required for the supported-cone functionality, which is already covered
by `test/moi.jl`.
