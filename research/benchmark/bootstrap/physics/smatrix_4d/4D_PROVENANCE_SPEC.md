# 4D S-matrix provenance — EXPERIMENTAL SPEC ONLY

**Latest primary numerical source:** Joan Elias Miro, Andrea Guerrieri and
Mehmet Asim Gumus, *The Phases of the Scalar S-Matrix Island*,
arXiv:2605.06613v1, Appendix A. Predecessor/formalism references are
arXiv:2210.01502v2, arXiv:1708.06765v1, and arXiv:2106.10257.
The implementation remains experimental and unregistered: it uses the A1
partial-wave normalization and sampled A5--A6 primal rows. Its implemented
basis is a single-anchor fully symmetric triple-rho orbit basis; the A2--A4
multi-wavelet, threshold term and A7 t-grid are provenance-only metadata. The
A4 center-power rule is stored explicitly as `N_(20/3)=Nmax` and
`N_sigma=Nmax-2` elsewhere (offsets `(0,-2,-2,-2,-2,-2,-2,-2)`), so custom
overrides change the artifact identity and cannot retain a paper-default flag.
A8--A15 is recorded as a fixed-t dual placeholder and remains fail-closed. It
does not claim the paper's numerical bounds.

The proposed typed specification has three intentionally distinct formulation
values. Its current diagnostic lowering uses one anchor triple-rho basis; the
Appendix-A multi-wavelet centers are explicit metadata, not an unclaimed
complete implementation:

- `:primal_full_unitarity`: sampled triple-rho amplitude, exact Eq. (2.9)
  disk, lowerable as Q3 or equivalent 2x2 real PSD blocks;
- `:dual_linearized`: the paper's Section 4 linearised positivity/unitarity
  formulation, which requires its own derivative/spin/grid/cutoff data and is
  not interchangeable with the primal;
- `:finite_conic_dual`: the exact finite-dimensional conic dual of the sampled
  primal, useful only for implementation weak/strong-duality checks.

The current branch contains only an unregistered spec-only diagnostic
implementation. No formulation is in the production physics catalog, and no
optimization is run or claimed. arXiv:2605.06613 is an S-matrix reference
only and is unrelated to CSDR.

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
canonical widths are degree 4 -> 11, degree 8 -> 41, degree 12 -> 102, degree
16 -> 204. Evidence is: tiny `(24,11)` reaches only approximately machine-zero
margin (recorded 2.48e-13 for the canonical `(24,11)` diagnostic), while small `(120,41)`
and medium `(576,102)` have negative least-squares rows; an independent HiGHS
check of `A*v>=1` reported infeasible. This is numerical evidence only, not a
dual infeasibility proof. Until a certified `delta>0` witness exists, no
catalog registration, paper claim, objective, or solve is permitted.

The optional complex/coupled-channel tier is deliberately not implemented:
there is no matched 4D primary reference plus independent witness in this
scope.
