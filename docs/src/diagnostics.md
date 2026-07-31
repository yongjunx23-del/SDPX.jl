# Diagnostics and certificates

`SDPResult` contains the terminal status, primal and dual solutions,
objectives, residuals, iteration counts, timings, and adaptive-parameter
history. `solve_summary` assembles the public information contract into one
named tuple.

```julia
summary = solve_summary(problem, result, options)
summary.status
summary.objective_value
summary.primal_residual
summary.dual_residual
summary.relative_gap
summary.certificate.valid
summary.warnings
```

`result_certificate` recomputes objectives, affine residuals, componentwise
backward errors, complementarity, and PSD checks in the original coordinates.
An authoritative solver status is downgraded when this independent check
fails.

For a failed optimize-mode run, `infeasibility_diagnosis` checks normalized
homogeneous rays:

```julia
ray = SDPX.infeasibility_diagnosis(problem, result, options)
ray.kind
ray.primal_infeasibility
ray.dual_infeasibility
```

`kind=:primal_infeasible` means the returned dual direction passed PSD,
homogeneous stationarity, and positive contradiction-margin checks.
`kind=:dual_infeasible_or_primal_unbounded` means the returned primal
direction passed homogeneous equality, PSD, and negative objective checks.
`kind=:undetermined` is not a feasibility claim.

In v0.3 an eligible failed solve is promoted to `PrimalInfeasible` or
`DualInfeasible` only when the corresponding ray passes the independent check.
The normalized ray is returned in the relevant primal or dual fields,
`result_certificate` verifies it again, and the MathOptInterface wrapper
reports `INFEASIBLE` or `DUAL_INFEASIBLE` with
`INFEASIBILITY_CERTIFICATE` on the certificate side.

The current Newton iteration still does not carry homogeneous self-dual
embedding variables `τ` and `κ`. Its formal certificate generator is recorded
as `:direct_primal_dual`; this makes the present guarantee precise while
leaving a compatible boundary for an HSD generator that can find certificates
more reliably. Eligible stopped solves attach the report at
`result.termination.infeasibility_diagnosis` even when verbose diagnostics are
disabled, because the report justifies the terminal status itself.
