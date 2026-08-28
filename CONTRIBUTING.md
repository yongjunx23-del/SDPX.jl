# Contributing to SDPX.jl

Thanks for your interest. SDPX is a numerical solver, so the bar for changes is
mostly about **evidence**: a change should come with a measurement or a test
that shows it does what it claims.

## Getting set up

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite runs in roughly two minutes. It is also run at 1 and 4 threads in CI,
because several kernels have threaded and serial paths that must agree.

```bash
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'
```

## What a good change looks like

**Performance changes need a before/after number.** Not an estimate — a measured
one, from `benchmark/`, stating the machine and thread count. Several plausible
optimisations in this codebase were measured and turned out to be slower; those
are documented in place so they are not retried. If your change replaces one of
them, say so and show the number.

**Numerical changes need a test that would fail without them.** Convergence
behaviour is easy to break in ways no existing test notices. If you change
termination, step selection, scaling, or the KKT path, add a case that pins the
behaviour you are fixing.

**Negative results are welcome.** If you measure something and it does not help,
a documented "this was tried, here is the number, it was reverted" comment is a
genuine contribution — it stops the next person repeating it.

## Things to be careful about

- **Threading.** `BigFloat` is `MPFR`-backed and mutable. Do not use
  `zeros(BigFloat, ...)` or `fill!` to allocate solver state: every slot aliases
  one object, and in-place kernels then corrupt whole arrays. Use `alloc_zeros`
  / `zero_distinct!`. This caused a long-standing "MPFR is not thread-safe" bug
  that was not, in fact, about MPFR.
- **Generic element types.** Kernels must work for `Float64`, `BigFloat`,
  `MultiFloats.Float64xN`. Avoid assuming BLAS is
  available for the element type.
- **Docstring placement.** Julia rejects a docstring placed immediately after
  another docstring; put each on its own function.

## Benchmarks

The canonical benchmark runner is `benchmark/bootstrap/runner.jl`, backed by the
schema-v7 `PhysicsBenchmarkHarness`. Problem selection and independent physics
validation live in injected catalog files; the harness owns process isolation,
measurement, serialization, and paired comparison. CI runs only the bundled
deterministic smoke catalog. Scientific catalogs and generated inputs are run
manually and are never downloaded by CI.

```bash
julia --project=. benchmark/bootstrap/runner.jl smoke \
  --problem=smoke/lp_box --arithmetic=float64 --provider=auto \
  --samples=1 --output=/tmp/sdpx-smoke.toml
```

Scientific runs must use `--catalog=/absolute/path/catalog.jl`; every catalog
entry carries provenance, an input fingerprint, a reference policy, and an
independent semantic validator. Use `benchmark/bootstrap/fresh_process_runner.jl` for at
least three fresh repetitions and `benchmark/bootstrap/compare.jl` for strict paired
comparison. See `benchmark/README.md` for the catalog and result contracts.

## Comparisons with other solvers

Do not add claims that SDPX is faster than MOSEK, SDPB, Clarabel, or any other
solver unless the comparison is reproducible from a script in `benchmark/` and
states the problem, tolerance, precision, thread count, and hardware. Solver
comparisons are extremely sensitive to all five, and an unqualified claim is
usually wrong.

## AI-assisted contributions

ChatGPT and Claude have been used as development assistants for parts of code
review, algorithm exploration, profiling, test design, documentation, and
release preparation. They are acknowledged in
[CONTRIBUTORS.md](CONTRIBUTORS.md) as tools rather than legal authors or
copyright holders.

AI-assisted pull requests are welcome, but the human contributor remains
responsible for the change:

- disclose the AI tool when it materially influenced the implementation;
- inspect every suggested code path instead of accepting generated code
  blindly;
- provide the same tests, numerical validation, and benchmark evidence required
  for a manually written change;
- verify that generated text or code does not copy material under incompatible
  terms;
- never include credentials, private benchmark data, or confidential prompts in
  an issue or pull request.

## Provenance

SDPX derives from [SDPJSolver.jl](https://github.com/FishboneChiang/SDPJSolver.jl)
(MIT, Li-Yuan Chiang). If you move or rewrite code that
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists as derived, keep that
file accurate. If you add code adapted from another project, record it there
along with its licence before opening the pull request.

## Reporting bugs

A useful report includes the element type, thread count, the solver options, and
`result.status` plus `result.termination` — the latter records *why* the solve
stopped, which is usually the first thing worth knowing. A small reproducing
problem is worth more than a description.
