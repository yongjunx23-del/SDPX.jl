# experimental/support_smoothing_dual_newton

**Status: EXPERIMENTAL — unmounted. Not part of the production SDPX solver.**

This independent `support_smoothing_dual_newton` solver (with its
`test/test_general_dual_newton.jl`) was confirmed to have **zero production
references**: neither file is included by `src/SDPX.jl`, and no other source
file references its symbols (`SOCPConeBlock`, `GeneralSOCPProblem`, …).

Per the frozen architecture it must **not** form a second solver architecture.
It is isolated here rather than deleted so the code is preserved for reference.
Do not mount it into `src/SDPX.jl` or the test suite unless a Lead decision
formally adopts it.

The production conic path is the canonical-form HSD solver defined by
`docs/design/CANONICAL_FORM.md` and `docs/design/HSD_FORMULATION.md`.
