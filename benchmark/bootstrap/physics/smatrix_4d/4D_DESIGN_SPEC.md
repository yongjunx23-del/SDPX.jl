# 4D identical-scalar S-matrix bootstrap — reviewed SPEC ONLY

**Status: STOP-SHIP / NOT REGISTERED.** This directory intentionally contains
no `catalog.jl` or loader entry. Its `spec_only/` code is an experimental
representation/diagnostic layer, not a registered production implementation
or benchmark. The existing `smatrix_soc/` catalog remains
the only registered S-matrix catalog.

## Exact primary reference and conventions

The latest primary numerical source is:

> Joan Elias Miro, Andrea Guerrieri and Mehmet Asim Gumus, *The Phases of the
> Scalar S-Matrix Island*, arXiv:2605.06613v1 (2026), Appendix A.

Predecessor/formalism references are Miro et al. arXiv:2210.01502v2,
Paulos et al. arXiv:1708.06765v1, and Guerrieri--Sever arXiv:2106.10257.

The triple-rho ansatz and historical four-dimensional conventions are also
cross-checked against Paulos et al., *The S-matrix Bootstrap III: Higher
Dimensional Amplitudes*, arXiv:1708.06765v1, Eqs. (6)--(16).

Its Eqs. (6)--(8) use `(+---)`, identical real scalars, and
`s=(p1+p2)^2`, `t=(p1-p3)^2`, `u=(p1-p4)^2`, `s+t+u=4m^2`. In the physical
s-channel, `s>=4m^2`, `z=cos(theta)`,

```text
t=-(s-4m^2)(1-z)/2, u=-(s-4m^2)(1+z)/2.
```

For four spacetime dimensions, Miro et al. Eq. (2.8) gives (consistent with
the `d=3` normalization of Paulos et al. Eq. (10)--(11))

```text
S_l(s) = 1 + i*sqrt((s-4m^2)/s)
              * integral[-1,1] P_l(z)/(32*pi) * M(s,t(z),u(z)) dz.
```

Equivalently, writing `f_l` for the integral, `S_l=1+i*beta*f_l` and
`M=16*pi*sum_l (2l+1)*f_l*P_l`. Eq. (12) imposes `|S_l|<=1` for `s>=4m^2`
and even `l`; it does **not** impose the non-convex equality in a finite
inelasticity-allowed relaxation.

## Proposed finite convex lowering (not yet accepted)

Use the source's Eqs. (14)--(16): the subtraction point is
`s*=t*=u*=4m^2/3`, and

```text
rho(x)=(sqrt(4m^2-s*)-sqrt(4m^2-x)) /
       (sqrt(4m^2-s*)+sqrt(4m^2-x)).
```

The upper-rim branch is fixed as
`sqrt(4m^2-(s+i0))=-i*sqrt(s-4m^2)`. A real coefficient multiplies each
fully symmetric orbit of `rho(s)^a rho(t)^b rho(u)^c`, with `a+b+c<=N`.
Crossing is imposed on `M(s,t,u)` **before** partial-wave projection. The
finite projection would use frozen Gauss--Legendre nodes and weights:

```text
f_l(s_i) ~= (1/(32*pi))*sum_q w_q*P_l(z_q)*M(s_i,t_i(z_q),u_i(z_q)).
```

For each sampled `(s_i,l)`, the exact affine disk lowering is

```text
[1, 1-beta_i*Im(f_l), beta_i*Re(f_l)] in Q3,
```

which is equivalent to `|1+i*beta_i*f_l|<=1`. The energy grid must satisfy
`beta_min>0`; threshold is only a separately tested limit because
`beta -> 0` collapses the disk to `S_l=1`. Only even spins are sampled for
identical bosons, but individual partial waves are not crossing invariant.

The spec records the Appendix-A numerical defaults explicitly: `Nmax` in
`{10,12,14,16,18,20}`, `Lmax` in `{16,18}`, `smax=300`, wavelet centers
`{20/3,10,20,30,40,50,60,86}`, threshold branch `alpha_th`, and the A7
`t` grid. It also parameterizes smaller diagnostic tiers, quadrature, spin
sets, formulation (`:primal_full_unitarity`, `:dual_linearized`, or
`:finite_conic_dual`), witness mode, and rho-map anchor. The current
spec-only builder uses one anchor triple-rho diagnostic basis; the
multi-wavelet centers are recorded metadata and are not silently presented
as a complete A2--A4 implementation. The dual A8--A15
parameters remain fail-closed placeholders; neither the paper dual nor an
objective is implemented here. These are not benchmark contracts until the
witness gate below succeeds.

## Stop-ship witness result

The zero amplitude is **not** a strict witness: `M=0` gives `f_l=0` and
`S_l=1`, exactly on every disk boundary. A valid strict witness would require
one real ansatz coefficient vector with positive disk margin at every sampled
energy and even spin. The evidence-only diagnostic uses the proposed
projection matrices and a deterministic least-squares search for positive
imaginary partial waves:

```text
scale=tiny:   matrix=(24,35),  min margin approximately 8.9e-16
scale=small:  matrix=(120,165), least-squares candidate has negative rows;
              independent HiGHS diagnostic A*v>=1: infeasible
scale=medium: matrix=(576,455), least-squares candidate has negative rows;
              independent HiGHS diagnostic A*v>=1: infeasible
```

The HiGHS results are **numerical evidence, not an exact infeasibility proof**;
no dual Farkas certificate is included. Tiny's margin is at machine epsilon
and therefore also fails the robustness gate. Consequently no 4D artifact, objective, solve, or paper-equivalent claim is
registered. The diagnostic can lower the sampled primal rows to SOCP/PSD
representations, but it does not solve or register them.

Run the bounded diagnostic with an isolated environment if dependencies are
needed:

```bash
JULIA_NUM_THREADS=1 julia --project=<isolated-env> \
  benchmark/bootstrap/physics/smatrix_4d/spec_only/witness_diagnostic.jl
```

## Legitimate next steps only

1. Reproduce the source's full dispersive/partial-wave ansatz and derive a
   known absorptive physical interior amplitude that is represented by it;
2. enlarge or change the crossing-compatible dispersive ansatz using the exact
   4D primary equations, with an independently checked strict witness;
3. add coupled dispersive/spectral variables only after deriving all channel
   crossing matrices, Hermiticity, realification, and a strict witness.

It is not legitimate to add independent unconstrained `S_l` variables, move
SOC centers/radii, relax `|S_l|<=1`, call the free amplitude strict, or relabel
finite sampling as continuous unitarity.
