# Massless EFT pole-augmented bootstrap slice

This is the first reviewed, build-only bootstrap slice. It implements the
external pole-plus-symmetric-pair ansatz and finite partial-wave projection,
but it is **not** a paper-equivalent continuum oracle and does not select a
solver/profile route.

## Normalization (resolved explicitly)

The external rows use an amplitude `tau` and the PSD block

```
[2-Im(tau)  Re(tau); Re(tau)  Im(tau)] >= 0.
```

Therefore the exact native cone is `[1, Re(tau), 1-Im(tau)] in Q3`, or
`abs(1+i*tau) <= 1`. The source prose instead writes
`abs(1+4*i*T_physical) <= 1`; this implementation resolves that discrepancy by
explicitly declaring `tau = 4*T_physical`. No factor four is silently inserted
in generated rows.

## Objective contract

`g0 = 3 alpha_00 - 3 alpha_pole`.
For the generated `Phi_11` and the stated `sigma_2`, the reviewed Taylor map is
`g2 = alpha_10/2 + alpha_20/4 - alpha_11/32 - 3 alpha_pole/8`.
The source's `-alpha_11/16` is retained only as a documented discrepancy and
is never used by this implementation. The prototype's unrelated `g2` formula
is not used.

Smoke, train, and production suites are all `:sampled_build_only`. The N=14
external SDPB interval is provenance metadata only: no independent objective,
strict witness, SDPX certificate, or continuum/endpoint/spin-tail claim is
asserted. Held-out audits are regenerated independently on a 599-node grid and
are diagnostic only.
