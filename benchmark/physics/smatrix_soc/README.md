# Sampled 1+1D elastic S-matrix SOCP benchmark

This directory provides deterministic, build-only second-order cone programs
derived from Miguel F. Paulos, Joao Penedones, Jonathan Toledo, Balt C. van
Rees, and Pedro Vieira, *The S-matrix Bootstrap II: Two Dimensional
Amplitudes*, [arXiv:1607.06110v2](https://arxiv.org/abs/1607.06110).

For identical scalar particles of mass `m=1`, the paper gives crossing and
elastic unitarity in Eqs. (1)–(2), and in rapidity form in Eq. (10):

```text
s = 4m^2 cosh(theta/2)^2,
S(theta) = S(i*pi-theta),
|S(theta)| <= 1 for real theta.
```

The benchmark uses the finite analytic ansatz

```text
z(theta) = sinh(theta)/(sinh(theta)+i),
S_d(theta) = sum_{n=0}^d a_n z(theta)^n,  a_n real.
```

Because `sinh(i*pi-theta)=sinh(theta)`, crossing is structural. For real
rapidity, `z(-theta)=conj(z(theta))`, giving real analyticity. The denominator
has no zero in the physical strip. This basis is a benchmark-derived analytic
truncation; it is not the paper's dispersion-spline numerical ansatz.

At each sampled rapidity the model creates one native three-dimensional
Lorentz cone

```text
[1, Re S_d(theta_k), Im S_d(theta_k)] in Q3.
```

| scale | ansatz degree | energy samples / Q3 cones |
|---|---:|---:|
| tiny | 2 | 16 |
| small | 4 | 64 |
| medium | 8 | 256 |
| stress | 12 | 1024 |

The zero-coefficient amplitude is an analytic strict-interior witness with
margin one at every sample. Constant `S=+1` and `S=-1` are stored boundary
witnesses corresponding to free elastic amplitudes.

Sampled unitarity is not a continuous-domain proof. No pole spectrum,
residue optimization, paper numerical bound, spin cutoff, or `Lmax` parameter
is introduced. The artifact fixes `reference_status=:sampled_build_only` and
`paper_equivalent=false`; tests construct but never solve it.
The injected catalog likewise uses the runner's construction-only path until a
nontrivial paper-grounded optimization objective is added and independently
validated.
