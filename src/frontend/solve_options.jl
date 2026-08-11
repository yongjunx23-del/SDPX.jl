"""
    SolveOptions(; kwargs...)

Small, policy-oriented user interface for SDPX solves.

Every field defaults to `:auto`.  `SolveOptions` is intentionally separate
from [`SolverOptions`](@ref): `SolverOptions` remains the fully resolved,
expert interior-point configuration, while `SolveOptions` is the stable
frontend contract intended for ordinary users, CLI tools, JuMP adapters and
benchmark runners.

The important consequence is that new backend tuning knobs do not have to
become public API.  The midend resolves this policy object once, records the
resolved choices in diagnostics, and hands a concrete `SolverOptions{T}` to
the numerical backend.

`precision` accepts `:auto`, a fixed arithmetic symbol (`:float64`,
`:float64x2`, `:float64x3`, `:float64x4`, `:bigfloat`) or a positive integer
bit count.  An integer bit count means BigFloat working precision.  For an
already-ingested problem the arithmetic type cannot be recovered from rounded
input, so an explicit precision request must be compatible with the problem's
stored arithmetic.  The CLI parses the input at the requested precision before
constructing the problem and therefore does not have that limitation.
"""
Base.@kwdef struct SolveOptions
    precision::Any = :auto
    duality_gap_threshold::Any = :auto
    primal_error_threshold::Any = :auto
    dual_error_threshold::Any = :auto
    maximum_iterations::Any = :auto
    max_runtime::Any = :auto
    threads::Any = :auto
    verbosity::Any = :auto
    presolve::Any = :auto
    scaling::Any = :auto
    algorithm::Any = :auto
    sparse::Any = :auto
    formulation::Any = :auto
    chordal_decomposition::Any = :auto
    equality_solver::Any = :auto
    working_precision_policy::Any = :auto
    diagnostics::Any = :auto
    timing::Any = :auto
    certification::Any = :auto
end
