# Optimize-mode infeasibility certificates and HSD migration

Status: SDPX 0.3 implements the certificate boundary and direct homogeneous-ray
generation. A full homogeneous self-dual (HSD) Newton embedding is not yet
implemented.

## Certificate contract

SDPX solves

```text
minimize    c'x
subject to  A_l(x) - C_l = X_l >= 0
            B'x = b.
```

A dual ray certifies primal infeasibility when

```text
Y_l >= 0
A*Y + B y = 0
sum_l C_l • Y_l + b'y > 0.
```

A primal ray certifies dual infeasibility when

```text
A_l(x) >= 0
B'x = 0
c'x < 0.
```

SDPX normalizes a candidate by its infinity norm, recomputes every equation in
the original problem coordinates, verifies each PSD block, checks the strict
objective margin and finite values, and only then returns
`PrimalInfeasible` or `DualInfeasible`. The certificate is checked a second
time by `result_certificate`. MathOptInterface exposes one result containing
the ray on the certificate side.

The direct generator has no feasible-problem overhead during the Newton loop:
it examines only eligible failed terminal iterates. Equality-only LPs use the
stronger analytic null-space ray already available from their least-squares
projection.

## Why this is not called HSD

The ray equations are the homogeneous limits used by HSD solvers, but SDPX's
current Newton state does not contain embedding scalars `tau` and `kappa`.
Consequently, a mathematically infeasible model may still finish with
`Stalled`, `IterLimit`, or a numerical status when the ordinary primal-dual
iterate does not approach a usable ray. Certificate metadata records
`certificate_generator=:direct_primal_dual` and
`homogeneous_self_dual_embedding=false`.

This distinction follows the public descriptions of
[Clarabel's homogeneous embedding](https://clarabel.org/stable/) and
[MathOptInterface's infeasibility-certificate semantics](https://jump.dev/MathOptInterface.jl/stable/background/infeasibility_certificates/).
No proprietary solver behavior is assumed.

## Full HSD implementation plan

1. Canonicalize the native SDP and equality operators into one signed conic
   map. Property tests must derive both native ray equations from that map;
   sign conventions must not be duplicated by hand in the solver and wrapper.
2. Add an internal embedding state containing `tau`, `kappa`, their residual,
   and their complementarity contribution. Keep these fields outside
   `SDPResult` until the algorithm is validated.
3. Extend the predictor and corrector right-hand sides with the embedding
   scalar equation while retaining the current HRVW/KSH/M PSD direction.
   Recover the scalar Newton step through a bordered solve so the established
   Schur and block-arrow factors remain reusable.
4. Select termination from scale-invariant ratios: solution candidates use
   `tau`; infeasibility candidates use `kappa`. Normalize the recovered ray,
   pass it through the existing original-coordinate certificate boundary, and
   fall back to an ordinary stopped status if validation fails.
5. Add exact small LP/SDP fixtures for primal infeasible, dual infeasible,
   jointly infeasible, feasible ill-conditioned, and nearly infeasible cases
   in Float64, Float64x4, and BigFloat.
6. Benchmark feasible overhead and infeasible detection. HSD becomes the
   default only if feasible Task_Low08 and medium CSDR runs retain their
   objective, residual, PSD, and iteration behavior with a small, stable
   runtime penalty. Otherwise it remains an automatic retry after an eligible
   direct solve failure.

## Acceptance gates

- No infeasibility status without a valid original-coordinate ray.
- `Optimal` behavior and certificates are unchanged on feasible regressions.
- MOI reports `INFEASIBLE` with a dual certificate or `DUAL_INFEASIBLE` with
  a primal certificate, never an unqualified `UNBOUNDED` claim.
- Scaling, presolve, and reconstruction preserve the ray equations.
- Checkpoint compatibility is versioned before embedding state is serialized.
- HSD-specific allocation and factorization overhead is measured separately
  from final certificate validation.
