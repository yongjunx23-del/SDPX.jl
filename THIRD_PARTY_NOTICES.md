# Third-party notices

SDPX.jl is distributed under the MIT License (see `LICENSE`). It also contains
material derived from, or designed against, third-party software whose own
terms are reproduced or referenced below. The top-level MIT file does **not**
cover the third-party portions listed here.

---

## Clarabel.jl (Apache-2.0)

**Upstream:** https://github.com/oxfordcontrol/Clarabel.jl
**Reviewed fork:** https://github.com/yongjunx23-del/Clarabel.jl
**Reviewed revision:** `fbc5dd01576d47fda53861a992a02e93c8dbd03e` (2026-04-14)
**Reviewed version:** 0.11.1
**License:** Apache License 2.0
**Permitted use:** SDPX borrows *design* and *algebra* — the structured sparse
metric representation, the lean numerical factor lifecycle, and the KKT-start
and update protocol — and adapts them into SDPX's own unified HSD engine.

### Status of the borrowed material

As of the revision recorded in `source_manifest.json`, **no Clarabel source
file has been copied verbatim into SDPX.** The borrowings landed so far are:

| SDPX artifact | Clarabel source | Category |
|---|---|---|
| `validation/clarabel_borrowing/soc_rank2_gate.jl` (A/C gates) | `src/cones/coneops_socone.jl` (`update_scaling!` rank-2 block), `src/kktsolvers/direct-ldl/directldl_datamaps.jl` (`SOCExpansionMap`) | **Independent reimplementation** from the published formulas, in a self-contained gate that does not call SDPX production code. |
| SDPX SOC rank-2 adapter (PR-02, when implemented) | same | **Adapted** — will carry an in-file Apache-2.0 attribution note naming the Clarabel source and revision. |
| `src/hsd/initialize.jl` startup reduction (PR-04A) | `src/kktsolvers/kktsolvers_directldl.jl` (single-factor authority) | **Independent** — SDPX reuses its *own* `GenericPivotedLDL`; no Clarabel code copied. |
| SDPX five-equation KKT structure | `src/kktsystem.jl` (design comparison only) | **Independent** — SDPX structure predates this work. |

### Notices and attribution obligations

Clarabel.jl ships no upstream `NOTICE` file at the reviewed revision, so none is
reproduced. Any file that in future contains copied or substantially adapted
Clarabel code must carry, at the top of that file:

```text
Portions derived from Clarabel.jl (https://github.com/oxfordcontrol/Clarabel.jl),
revision fbc5dd01576d47fda53861a992a02e93c8dbd03e, licensed under the Apache
License, Version 2.0. See THIRD_PARTY_NOTICES.md and LICENSE-APACHE-2.0.
```

The full Apache-2.0 text is reproduced in `LICENSE-APACHE-2.0`.

---

## QDLDL.jl (indirect, via Clarabel)

QDLDL is a separate project with its own license, and is **not** covered by
Clarabel's Apache-2.0 terms. SDPX does not currently vendor QDLDL source: where
SDPX exercises a QDLDL-backed route it does so through the
`SparseQDLDLCache` boundary in `src/factor_cache/routes/qdldl_sparse.jl`, using
the installed package. If QDLDL source is ever vendored into this repository,
its license and attribution must be reviewed and recorded separately here.

---

## Other dependencies

SDPX's Julia dependencies (MathOptInterface, MutableArithmetics, the
`MultiFloats`/`BigFloatLinearAlgebra`/`MultiFloatLinearAlgebra` provider
packages, and the Julia standard library) retain their own licenses. They are
consumed as ordinary package dependencies, are not vendored, and are not
covered by this file.
