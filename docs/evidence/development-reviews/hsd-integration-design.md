# HSD integration design — wiring the tau/kappa embedding into the LP solve

Branch: refactor/zeroalloc-factorcache-hsd. Phase 6. This is the design + a
minimal bordered-system prototype for driving the LP interior-point loop
through the homogeneous self-dual (HSD) embedding. It deliberately does NOT
modify src/lp_solver.jl or the interior-point hot loop; it is the blueprint
for a dedicated integration session.

## Current state
- src/hsd.jl provides: hsd_skew_embedding(A,b,c), primal/dual infeasibility
  certificates (Farkas / double-ray), hsd_status, hsd_classify, the
  hsd_bordered_system prototype, and - since this round - a WORKING dense
  path-following Newton solver **hsd_lp_solve(A,b,c)** that drives the
  tau/kappa embedding to a classified status (optimal / primal-infeasible)
  end-to-end on small LPs, verified across Float64/x2/x3/x4/BigFloat.
- The main LP solver still detects infeasibility via ray certificates; its
  termination flag homogeneous_self_dual_embedding is currently false. The
  remaining integration step is wiring hsd_lp_solve (or the bordered-system
  KKT-reuse form below) into that loop.

## Proposed solver integration (next steps)

1. Variable ordering. For LP min c'x s.t. Ax=b, x>=0 (dual A'y+s=c, s>=0),
   the HSD variable vector is z = (x, s, tau, kappa, y) and the bordered
   Newton system is assembled from the skew embedding M plus the
   complementarity diagonal X^{-1}S and the tau/kappa barrier terms.

2. Bordered / augmented linear system. The per-iteration Newton direction
   solves a system of the form
       [ M_skew + D       r ]
       [ r'            -kappa ]
   where D is the (diagonal) slack scaling and the border carries the
   tau/kappa coupling. hsd_bordered_system builds the skew-plus-diagonal core
   and returns dimensions (m, n, dim) so the solver can allocate once.

3. Reuse of the main KKT factor. The augmented HSD system is solved via a
   Schur/arrow update on the existing SDP KKT factor (src/kkt_backend.jl,
   src/kkt.jl): factor the (1+m+n) skew-plus-diagonal matrix once per outer
   iteration, then apply the bordered direction solves (predictor/corrector)
   reusing that factor - exactly the FactorCache reuse pattern from Phase 3.

4. Convergence / certificate rules. After each iteration:
   - kappa -> 0 with tau positive: optimal; report x/tau, y/tau, primal
     objective c'x/tau.
   - tau -> 0 with kappa > 0: infeasible; the sign structure of the limiting
     dual/slack selects primal-infeasible (Farkas, b'y>0, A'y<=0) vs
     dual-infeasible/unbounded (x>=0, Ax=0, c'x<0). Use the certificate
     functions in hsd.jl to validate in original coordinates.
   - Set the solver's homogeneous_self_dual_embedding flag true and populate
     the termination reason (:hsd_optimal / :hsd_infeasible).

5. Zero-allocation contract. Preallocate the bordered system matrix and the
   factor in the LPWorkspace (Phase-2 cold-state pattern), so the HSD Newton
   step allocates zero steady-state, matching the Phase-4b contract.

## Deliverable committed with this doc
- src/hsd.jl: hsd_bordered_system (prototype: skew embedding + diagonal
  barrier, with dimensions).
- test/hsd.jl: bordered-system assertions (14 HSD assertions total).

