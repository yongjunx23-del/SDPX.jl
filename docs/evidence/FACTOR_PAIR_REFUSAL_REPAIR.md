# Factor-pair refusal and execution accounting repair

SDPX 0.6.1, tested HEAD `24bacde`, development/scientific-core-20260907.
Source commits: `4772a07`, `24bacde`. Experimental opt-in only.

- Startup and iteration now share `execute_with_refusal`, which catches only
  `FactorPairNumericalRefusal`. No-state startup failures are explicit; errors
  in programming/unsupported-backend policy still propagate.
- Later refusal owns a snapshot of the last accepted point/history/generation,
  actual residual quantities, stage/reason/detail, and marks `cert_ok=false`.
  Reporting failure does not rerun cone certification.
- Solve-owned LU ledger increments attempts immediately before `lu` and
  completions after success, including provisional epochs. Public counts are
  no longer inferred from accepted iterations.
- Structured termination metadata retains refusal and accounting provenance.
- Zero time is accepted and returns TimeLimit without an accepted step on the
  canonical public control. Startup still occurs; checks remain cooperative
  between steps, NOT a hard setup+solve wall-time guarantee.

## Validation

Pinned clean source and loaded root asserted in-process; Julia 1.12.6 on
Darwin aarch64; one Julia/BLAS/OMP/MKL thread; owned-group timeout 180 seconds.

- Startup/refusal tests: 31/31 (real singular initial LU, early memory refusal,
  injected numerical refusal after one accepted step, isolation, programming
  error propagation).
- Selector/public tests: 50/50, including original-coordinate certificate and
  zero-time public request.
- Canonical adapter: 58/58, unchanged 23-step terminal numerical values.
- Internal/validation differential: 26/26.

Total across focused testsets: 165 passing assertions. Logs:
`/tmp/fp-refusal-24bacde.log` and `/tmp/fp-namespace-24bacde.log`.
The first combined harness omitted Main.TOML for the namespace test; that
harness error was corrected on the separate successful run without source edits.

Independent source review submitted as workflow `8438609b`; pending at writing.
No full-suite result is claimed for this HEAD. Admission classification,
formulation descriptor consistency, complete memory bounds, total time semantics,
platform qualification and full public failure injection remain open. The known
broad ErrorException catch inside NP.build predates this repair and remains open.
