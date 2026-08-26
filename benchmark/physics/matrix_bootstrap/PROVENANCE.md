# Matrix-bootstrap benchmark provenance

This directory is a build-first benchmark source model.  It is intentionally
not a claim that the repository reproduces the authors' production SDPA-GMP
input files or their post-quotient free-variable basis.

## Primary source

- Henry W. Lin and Zechuan Zheng, *High-Precision Bootstrap of Multimatrix
  Quantum Mechanics*, arXiv:2507.21007v3 (11 June 2026; the arXiv HTML page
  displays the manuscript date 24 August 2026),
  <https://arxiv.org/abs/2507.21007> and
  <https://arxiv.org/html/2507.21007>.
- The source TeX was obtained from the arXiv e-print on 26 August 2026 and
  inspected from `BosonicBFSS_arxiv.tex` and `Sup_body.tex`.
- The authors' example generator is the Mathematica notebook at
  <https://github.com/Canonical111/O2massiveBootstrap>.  It is not a
  machine-readable level-14 quotient or a fixed-slice solver reference.

The implementation follows the source equations directly:

1. For `D=2`, `Z=(X_1+iX_2)/sqrt(2)`, `Zbar=(X_1-iX_2)/sqrt(2)`, with the
   corresponding momenta.  The charge map is `Z,P:+1` and `Zbar,Pbar:-1`;
   reflection exchanges the two signs.  We use `Pi=-iP` and `Pibar=-iPbar`
   so the real word algebra has letter levels `ell(Z)=ell(Zbar)=1` and
   `ell(Pi)=ell(Pibar)=2` (main text Eq. (2), hierarchy paragraph).
2. The source Hamiltonian is
   `H = tr(P Pbar) + M^2 tr(Z Zbar) + 1/2 tr([Z,Zbar]^2)`.
   In the real `Pi` convention, the stationary derivation used here is
   `delta(Z)=Pi`, `delta(Zbar)=Pibar`,
   `delta(Pi)=M^2 Z + [Z,[Zbar,Z]]`, and
   `delta(Pibar)=M^2 Zbar + [Zbar,[Z,Zbar]]`.
   Expanding the nested commutators gives the exact coefficients
   `(2,-1,-1)` for each cubic derivative.  The duplicate-looking positive
   terms in the source file are intentional and are the `2*Z*Zbar*Z` (or
   reflected) term, not a guessed extra interaction.
3. Stationarity is imposed as the linear residual
   ` <tr delta(O)> = 0`, equivalent to the source's
   `<[H,O]>=0` (main text Eq. (4)) for the formal word derivation.  Coordinate
   words use cyclic trace symmetry.  Momentum-containing words do not: their
   ordering is preserved up to O(2) reflection and signed reversal, because a
   cyclic move generates canonical-commutator contractions.  The affine E34
   reduction gives `<tr Z Pibar>=1/2` and `<tr Pibar Z>=-1/2`; a genuinely
   self-reversing odd-momentum orbit is zero.
4. Inner-product positivity follows main text Eq. (5) for the generated word
   basis.  The trustworthy low-order part of ground-state positivity (main
   text Eq. (6)) is Appendix E42, which becomes
   `[[1,-2p],[-2p,2x+M^2]] >= 0` for
   `p=<tr Pi Pibar>` and `x=<tr Z Zbar>`.  This is included as a dedicated
   block; the full ground-state hierarchy is not claimed.
5. Appendix E40 fixes the energy convention used by `scan_observable=:energy`:
   `E=-3/2 <tr Pi Pibar> + M^2/2 <tr Z Zbar>`.
6. The nonlinear relaxation follows Appendix B Eqs. (relax1)--(relax2):
   `[1 x'; x Q] >= 0` and `M-Q >= 0`.  `Q` is an independent affine variable;
   the generator never inserts `x*x` as a linear coefficient.

The implemented kinematic quotient covers charge neutrality, O(2)
reflection, Hermiticity/time reversal, and the E34 vacuum canonical/gauge
normalization.  General momentum commutators can generate factorized
multi-trace terms.  The full such quotient, the full gauge Ward tower, and the
authors' indexed-irrep basis are not implemented.  For that reason the public
builder fails closed above levels 4/6/8 and every Lin--Zheng artifact records
`paper_equivalent=false` and `publication_claim=:none`.

## Published counts and intervals

Table 1 of the source reports post-quotient `D=2` counts (free/all) of
`(3,14)`, `(8,94)`, `(22,614)`, `(77,4086)`, `(326,27830)`, and
`(1569,192374)` at levels 4, 6, 8, 10, 12, and 14 respectively.  These are
retained as provenance metadata only, with
`published_census_status=:metadata_only_fail_closed`; the low-order
relaxation's counts are deliberately different.

Table 2 reports the massive `M^2=1,D=2` suite intervals
`E=[1.172098376,1.172098408]`,
`<tr X^2>=[0.77800898,0.77800934]`, and
`<tr(Z^2 Zbar^2)>=[0.15850588,0.15850607]`; the massless intervals are
`E=[0.707832,0.707868]`,
`<tr X^2>=[1.15420,1.15460]`, and
`<tr(Z^2 Zbar^2)>=[0.37055,0.37085]`.  They are suite-level scan results, not
fixed-slice objective oracles.  The artifact therefore defaults to
`reference_status=:build_only`, with `objective=nothing` and
`reference_objective=nothing`.  No numerical solve is used as a correctness
proof.

## Independent small cross-check

`build_kazakov_zheng_lambda4` labels the optional fixture as
`source=:kazakov_zheng_lambda4`, source
`arXiv:2108.04830 Appendix E (Lambda=4 relaxation)`, and
`reference_status=:cross_check_only`.  It is the Kazakov--Zheng Lambda=4
relaxation associated with the earlier proposal; it is not the 2025
Lin--Zheng model and its metadata/fingerprint are intentionally distinct.
The Appendix-E `g=h=1` model has 20 symmetry-reduced moments, 14 exact loop
equations, three correlation blocks of sizes 11, 10, and 5, and one
relaxation block of size 2.  Its paper reports the Lambda=4 interval
`0.393566 <= Tr(A^2) <= 0.431148`; that interval is stored as provenance
metadata only, with no objective or solve certificate.
