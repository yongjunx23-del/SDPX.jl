# 4D S-matrix provenance — EXPERIMENTAL SPEC ONLY

**Primary source (verified from the v2 arXiv source archive):** Joan Elias
Miro, Andrea Guerrieri and Mehmet Asim Gumus, *Bridging Positivity and
S-matrix Bootstrap Bounds*, arXiv:2210.01502v2 (JHEP 05 (2023) 001).
The paper's relevant equations are Eq. (2.6) (triple-rho crossing ansatz),
Eq. (2.8) (4D partial-wave projection), Eq. (2.9) (full partial-wave disk),
Eq. (2.10) (finite primal min/max), and Section 4 (linearised-unitarity/
dual bounds). The ansatz and normalization are cross-checked against the
source of Paulos et al., arXiv:1708.06765v1, but the Miro et al. paper is the
main provenance for this experimental design.

The proposed typed specification has three intentionally distinct formulation
values:

- `:primal_full_unitarity`: sampled triple-rho amplitude, exact Eq. (2.9)
  disk, lowerable as Q3 or equivalent 2x2 real PSD blocks;
- `:dual_linearized`: the paper's Section 4 linearised positivity/unitarity
  formulation, which requires its own derivative/spin/grid/cutoff data and is
  not interchangeable with the primal;
- `:finite_conic_dual`: the exact finite-dimensional conic dual of the sampled
  primal, useful only for implementation weak/strong-duality checks.

The current branch contains only an unregistered diagnostic implementation.
No formulation is in the production physics catalog, and no optimization is
run or claimed.

## Exact conventions

For metric `(+---)`, identical mass-`m` scalars obey
`s+t+u=4m^2`. In the physical s-channel,
`s>4m^2`, `t=-(s-4m^2)(1-z)/2`, `u=-(s-4m^2)(1+z)/2`, and
`rho_phase=sqrt((s-4m^2)/s)`. Miro Eq. (2.8) uses

```text
f_l(s) = 1/(32*pi) integral[-1,1] P_l(z) M(s,t,u) dz,
S_l(s) = 1 + i*rho_phase*f_l(s).
```

Eq. (2.9) is `|S_l|^2<=1`, imposed only on the declared finite sampled
energy/spin grid by the diagnostic. The affine cone row is exactly
`[1, 1-rho_phase*Im(f_l), rho_phase*Re(f_l)] in Q3`.

Crossing is imposed on the full amplitude before projection through fully
symmetric triple-rho orbit coefficients. Individual partial waves are not
crossing invariant. Energy samples are strictly above threshold; the
threshold limit is a separate test.

## Witness stop-ship

`f_l=0` / `M=0` gives `S_l=1`, so the free amplitude is a boundary witness,
never a strict interior witness. The diagnostic records the deterministic
least-squares witness search and a max-margin SOCP constructor. Current
numerical evidence is: tiny `(24,35)` reaches only approximately machine-zero
margin (`~8.9e-16` in the original default diagnostic), while small `(120,165)`
and medium `(576,455)` have negative least-squares rows; an independent HiGHS
check of `A*v>=1` reported infeasible. This is numerical evidence only, not a
dual infeasibility proof. Until a certified `delta>0` witness exists, no
catalog registration, paper claim, objective, or solve is permitted.

The optional complex/coupled-channel tier is deliberately not implemented:
there is no matched 4D primary reference plus independent witness in this
scope.
