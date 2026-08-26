# Kazakov–Zheng finite-N lattice bootstrap benchmark

This directory contains a deterministic, build-first affine SDP benchmark
derived from Vladimir Kazakov and Zechuan Zheng, *Bootstrap for finite N
lattice Yang-Mills theory*, arXiv:2404.16925v4 / JHEP 03 (2025) 099.

Implemented physics:

- finite-N `SU(2)`, `D=2`, with the paper's `Tr = tr/N` and
  `lambda = 2N^2/beta` conventions;
- exact adjacent backtrack cancellation, cyclic trace equivalence, charge
  conjugation, and all eight `B2 ≃ D4` signed lattice symmetries;
- the Hermitian-conjugation Gram condition in eqs. (3.2)–(3.3). The first
  scale has the identity plus the eight origin-based length-four Wilson lines,
  hence the paper's `9×9` structure in eq. (3.7);
- an audited edge-simple subset of the single-trace loop equations,
  constructed directly as `Aid + Avar = 0`: the master identity is eq. (2.31),
  `Aid` and `Avar` are eqs. (2.32) and (2.33), and the compact `SU(2)` equation
  is (2.36). The tiny row is the lowest single-trace equation (2.37). Eqs.
  (2.26)–(2.30) are derivation context, not implemented-row labels. A loop
  with repeated geometric links is not given incomplete split/rejoin terms
  from eqs. (2.34)–(2.35);
- the exact `SU(2)` plaquette oracle obtained by differentiating Appendix A:
  `I0(8/lambda)/I1(8/lambda) - lambda/4`;
- a sparse lowering to the ordinary `SDPX.SDPProblem` form at each fixed
  coupling. Every coefficient is affine in Wilson moments.

The scalable `:based_length` hierarchy is intentionally not called the
paper's recursive `Lambda` hierarchy. The published `Lambda=3` census
(`8335` loops, `14591` equations, `1044` free moments and the 18 sizes in
eq. (4.5)) is stored as reference metadata, with
`independently_reproduced=false`. `assert_paper_lambda3_reproduced!()` fails
closed. This prevents a synthetic or padded model from being reported as the
paper benchmark.

The artifact fingerprint commits to the declared provenance and semantic
counts as well as the operator basis, affine equations, Gram entries,
objective, and exact-oracle metadata. Validation rebuilds those semantics
from the declared spec before lowering.

Reflection-positivity matrices, irrep projectors, repeated-link split/rejoin
terms, and the exact paper hierarchy remain explicit follow-on work.

The injected catalog is deliberately `:build_only`. Its smoke/scaling suites
measure deterministic model construction and lowering, not an optimization or
a claim that the based-length objective reproduces a paper bound.
