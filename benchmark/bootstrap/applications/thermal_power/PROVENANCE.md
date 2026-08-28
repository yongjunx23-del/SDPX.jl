# Provenance and claim boundary

Primary source: Giacomo Giudice, Asli Cakan, J. Ignacio Cirac, and Mari Carmen
Banuls, *Renyi free energy and variational approximations to thermal states*,
arXiv:2012.12848v2; Physical Review B 103, 205128 (2021).

The implementation uses these source statements:

- Eq. (5) defines the Renyi free energy for a density operator.
- Eq. (7) writes the maximum-Renyi problem in an energy eigenbasis, with
  probability normalization and fixed mean energy.
- Section II.1 explains why the optimizing state may be taken diagonal in the
  Hamiltonian eigenbasis, reducing the commuting problem to probabilities.

The following choices belong to this benchmark rather than to the paper:

- Renyi order four is selected so every quartic epigraph is one native
  `PowerCone(1/4)` block. The paper's numerical work emphasizes order two.
- The energy spectrum is an equally spaced deterministic grid.
- The target mean energy is chosen to make the uniform distribution the
  analytic optimum, enabling an independent correctness oracle.
- A redundant-looking explicit `p >= 0` block is retained because `p`
  represents physical probabilities; the third power-cone coordinate alone
  is sign-symmetric.

No tensor-network algorithm, Ising-model spectrum, thermodynamic-limit
statement, local observable, or numerical value from the paper is claimed.
The artifact is construction-only until the native Power-cone route passes
its solver and certificate gates.
