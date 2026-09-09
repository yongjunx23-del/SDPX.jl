# Retained factor-pair owner snapshot

SDPX 0.6.1, tested source `3303f1f` (implementation `f2d4097`).
Design consultation `1e138597`; implementation review `06623322` found no
blockers in the bounded retained-owner diff. It did not rerun tests.
No R0 qualification or memory-bound claim.

## Contract

A scalar-only receipt is attached at the execution boundary. It describes the
factor-capable epoch actually retained there, not a factor stored in Result.

- No returned state / consumed pending epoch: prepared=executed=0, owner=none.
- Retained stale epoch: actual prepared dimension; executed=0, current=false.
- Retained valid epoch: actual prepared/executed dimension and current=true,
  independently of Optimal, time limit, iteration limit or typed refusal.
- Historical LU attempts/completions remain separate and unchanged.

Currentness requires owner/anchor/pair/generation lineage, state settings and
layout, shared pair/epoch integrity checks, exact copied input/scalar/cone
words and source-record binding. Deliberate input copies are compared by words,
not pointer identity. Unexpected pending object types still throw.

Operational verify wrappers retain their errors using shared failure-reason
checks. No broad exception catch, fresh factorization, terminal certification
or new retained-epoch field was introduced. step! consumption/commit behavior
and default dispatch remain unchanged.

## Checks

Pinned clean source and actual loaded root; Julia 1.12.6 aarch64 Darwin;
one Julia/BLAS/OMP/MKL thread; each owned process capped at 180 seconds.

- Factor-pair focused tests: **325/325** (including 164 owner/refusal controls,
  public zero-time/success receipts, namespace differential and unchanged
  canonical terminal).
- Default native structure diagnostics: **239/239** across 17 testsets.
- Logs: `/tmp/fp-owner-3303f1f.log`, `/tmp/default-owner-3303f1f.log`.

Controls cover startup-without-return despite completed LU, consumption without
commit, retained accepted next epoch, generation/anchor/pair mismatch, state
A/b/c/x/tau/kappa changes, epoch CSC/factor/pivot and pair/cone mutations,
operational verifier throws, and scalar receipt schema. A first test incorrectly
required Symbol-bearing receipts to be isbits; `3303f1f` instead checks their
exact immutable scalar field types. No numerical implementation adjustment was
needed for that test correction.

Separate public objective-transform tests at `ae53aa5`: **34/34**, including
bounded min(f)/max(-f), primal and dual offsets, and exact 4^3 objective scaling.
That scaling is a changed problem, not homogeneous-state invariance.

Remaining: full regression at this source,
complete original/post-reduction admission, peak-memory boundary, genuine
homogeneous-state qualification and broad ErrorException catch in NP.build.
R2-A symbolic reuse remains unimplemented; a distinct session-local design is
in progress and must not put mutable CHOLMOD factors in the global structure cache.
