# Preprocessing

SDPX runs a typed structural preprocessing pipeline before scaling and the
numerical solve. The pipeline operates on `SDPProblem{T}` and preserves the
original arithmetic throughout. JuMP and MathOptInterface are adapters; none
of the reduction logic depends on JuMP.

## Stages

The public entry point is:

```julia
prepared = SDPX.Experimental.preprocess(problem, options)
```

`prepared` contains the transformed problem, a typed `PreprocessPlan`, a
`ReconstructionMap`, and a structured `PreprocessReport`. The ordinary `solve`
and `solve!` entry points run the same pipeline automatically and return the
solution in the original coordinates.

The automatic stages are:

1. extract single-variable scalar PSD bounds and merge exact lower and upper
   limits;
2. eliminate only variables whose lower and upper limits are exactly equal in
   arithmetic type `T`;
3. remove exact zero, duplicate, and proportional equalities;
4. reuse the existing arithmetic-aware equality rank presolve;
5. run the existing block-aware scaling after structural reduction;
6. estimate primal and dual formulation costs;
7. estimate chordal clique and overlap costs; and
8. reconstruct and certify the result against the original problem.

Approximate fixed-variable elimination is not performed. Near-proportional
equalities are counted for diagnostics but retained. Numerical equality
dependencies are removed only after both the column relation and right-hand
side reconstruction pass arithmetic-aware checks. If a numerical relation is
ambiguous, the original equalities are retained.

Primal-to-dual conversion and chordal transformation are analysis-only. The
estimators may report a potentially cheaper alternative, but neither
transformation is selected automatically.

Sparse Schur planning is also completed before a workspace is allocated.
`problem.structure.schur_analysis` records the active-constraint overlap graph,
estimated upper-triangle nonzeros, density, and per-block active counts;
`problem.structure.schur_plan` records the deterministic `:dense`, `:sparse`,
or `:block_sparse` choice and its reason. A selected Float64 sparse plan
freezes one CSC pattern and `SchurAssemblyMap`, so later iterations update only
`nzval`. `SDPX.Experimental.sparse_schur_diagnostics(workspace)` reports
structural/numeric nonzeros, factor fill, pattern reuse, overlap/block counts,
and assembly timing. Generic MultiFloat/BigFloat sparse-Schur equality
recovery remains explicitly fail-closed: an explicit `sparse=:sparse` request
raises before workspace construction, while `sparse=:auto` may choose the
dense route when its plan does not have a supported provider.

## Bound storage

MOI supports `VariableIndex` and `ScalarAffineFunction` constraints in
`GreaterThan`, `LessThan`, and `Interval` sets. A scalar affine constraint is
recognized as a variable bound only when it has exactly one nonzero
coefficient.

Single-variable scalar blocks use `CompactScalarCoefficientVector{T}`. It
preserves the historical `Asp[block][variable]` indexing contract while
avoiding a length-`m` reference vector for each bound. Active-only ingestion,
classification, chordal analysis, scaling, LP extraction, and precision
preparation avoid scanning inactive variables.

The merged bound plan uses contiguous `Vector{T}` lower and upper arrays plus
compact activity flags and original source indices. No bound is converted
through `Float64`.

## Reconstruction

The reconstruction map records:

- retained-to-original variable, block, and equality indices;
- fixed variables and their typed values;
- the eliminated objective offset;
- removed bound constraints needed for nonnegative dual reconstruction; and
- the equality multiplier map for exact and numerical equality reductions.

Warm starts may be supplied in original coordinates. Final primal and dual
objectives, affine residuals, cone residuals, and PSD certificates are
recomputed against the original unscaled problem. A failed final certificate
downgrades an otherwise authoritative solver status.

## Options

Typical callers can leave preprocessing at its defaults:

```julia
result = solve(
    problem;
    presolve=:auto,
    scaling=:auto,
)
```

The structural controls are:

```julia
presolve = :auto                # :auto, :on, :off, true, or false
presolve_bounds = true
presolve_fixed_variables = true
presolve_zero_constraints = true
presolve_duplicate_constraints = true
presolve_dependent_equalities = true
scaling = :auto                 # :auto, :none, or :equilibrate
formulation = :auto             # :auto, :primal, :normal_equations, or :augmented
```

Exact zero, duplicate, and proportional equalities are always verified in the
problem arithmetic before they are removed. Near-proportional detection is a
diagnostic only: it never changes the model or reconstruction map. Its work is
bounded to eight equivalent passes over the retained equality matrix. If a
large dense system would exceed that budget, SDPX records an explicit warning
and skips only the approximate diagnostic; exact cleanup and the subsequent
target-arithmetic rank presolve remain enabled.

Use `result.diagnostics.presolve.preprocessing` to inspect stage timings,
allocations, dimension changes, bound counts, equality cleanup, formulation
costs, chordal estimates, and warnings.

## Target-model behavior

The maintained dense bootstrap input has 6,119 variables, 482 supplied
equalities, and 32 PSD blocks, but no scalar bound blocks. Its equality matrix
has rank 394, and the aggregate PSD patterns are 99.84% dense even though
individual coefficient matrices are sparse. Consequently:

- bound extraction and fixed-variable elimination make no change;
- equality presolve reduces 482 columns to 394;
- chordal analysis rejects decomposition before constructing clique data; and
- the existing sparse-coefficient, dense-Schur numerical path remains
  selected.

This distinction matters: individual coefficient sparsity is not evidence that
a PSD block or its final Schur complement is sparse. The canonical medium
CSDR model contains 1,700 dense-pattern `2×2` PSD blocks and no explicit
equalities or scalar bounds, so preprocessing is a no-change regression path
there; its optimized reduced-arrow and SIMD kernels remain responsible for
performance.

Sparse equilibration uses the same active-incidence representation. On the
cluster model, copying only active matrices and sharing one read-only empty
CSC matrix per block reduced the median equilibration call from 1.241 to 0.633
seconds and allocation from 685.8 to 269.7 MB; the old and new scaled-problem
checksums were identical. This optimization does not share writable
coefficient or BigFloat storage.

## Arithmetic limitations

`Float64`, `MultiFloats.Float64x4`, and `BigFloat` are covered by solve and
reconstruction tests. BigFloat copies use independently owned MPFR storage.
Preprocessing itself does not introduce shared writable values.

MultiFloats 3.x does not implement the scalar multiplication network required
by `Float64x8`, and the validated AMD EPYC nodes expose AVX2 but not AVX-512.
SDPX therefore does not claim a working `Float64x8` solver path. The type is
not selected automatically; `Float64x4` is the supported SIMD
extended-precision configuration on those nodes.
