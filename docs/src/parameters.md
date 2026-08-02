# Parameters

Most applications should configure only:

```julia
result = solve(
    problem;
    tolerance=1e-8,
    maximum_iterations=200,
    time_limit=Inf,
    threads=Threads.nthreads(),
    precision=nothing,
    verbosity=1,
    diagnostics=true,
    warm_start=nothing,
)
```

The default controller adapts centering, fraction-to-boundary values,
backtracking, and refinement from the measured Newton iteration. Presolve,
scaling, kernel selection, working precision, and scheduling also default to
automatic policies.

Expert code can construct typed options with ASCII aliases:

```julia
options = SolverOptions(
    BigFloat;
    tolerance=big"1e-30",
    maximum_iterations=400,
    beta=big"0.1",
    gamma=big"0.9",
    precision_bits=384,
    expert_mode=true,
)
```

The original fields `β`, `γ`, `Ωp`, `Ωd`, `ϵ_gap`, `ϵ_primal`, `ϵ_dual`, and
`iter_max` remain supported. Supplying both an ASCII alias and its Unicode
field is an error, preventing silent precedence mistakes.

See the
[complete parameter reference](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/parameters.md)
and
[adaptive policy report](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/adaptive-parameter-policy.md)
for meanings, bounds, fallbacks, and benchmark evidence.
