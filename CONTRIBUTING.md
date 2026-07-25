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
one, from `bench/`, stating the machine and thread count. Several plausible
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
  `MultiFloats.Float64xN`, and `DoubleFloats.Double64`. Avoid assuming BLAS is
  available for the element type.
- **Docstring placement.** Julia rejects a docstring placed immediately after
  another docstring; put each on its own function.

## Benchmarks

Large benchmarks are deliberately **not** run in CI and their inputs and outputs
are not committed — the serialized problem instances run to gigabytes. Only the
`:small` tier runs as a smoke check.

```bash
julia --project=bench -e 'include("bench/run.jl"); main(tiers=(:small,))'
```

Regenerate larger instances locally with `bench/generate.jl` rather than
committing them.

## Comparisons with other solvers

Do not add claims that SDPX is faster than MOSEK, SDPB, Clarabel, or any other
solver unless the comparison is reproducible from a script in `bench/` and
states the problem, tolerance, precision, thread count, and hardware. Solver
comparisons are extremely sensitive to all five, and an unqualified claim is
usually wrong.

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
