# SDPX test profiles

The fastest edit-test loop is:

```sh
julia --project=. test/runtests.jl
```

This runs the **quick** profile. `Pkg.test()` selects the same profile, but may
still spend time resolving every package listed in the full test environment.
Quick checks the LA backend, planner and public API boundaries,
canonical/problem-feature IR, prepared reuse, diagnostics, plus one small LP,
SOCP, and SDP solve.

Run the complete numerical, precision, extension, threading, integration, and
quality suite before release or after a solver-hot-path change:

```sh
SDPX_TEST_PROFILE=full julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the quick profile across its Julia, platform, and thread matrix, then
runs one full profile on the current Julia/Ubuntu/four-thread configuration.
The full profile preserves the prior test-file order and coverage; the quick
profile only changes which files run by default.
