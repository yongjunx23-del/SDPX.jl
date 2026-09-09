# Reproducibility (candidate release material — no push/release authorized)
`scripts/check_reproducible_delivery.jl` (stdlib-only: `SHA`, `TOML`)
prints the delivery fingerprint, one `name=value` line each:
git `HEAD`, Julia `VERSION`, SDPX version (`Project.toml`),
MPFR/GMP versions (`Base.MPFR`/`Base.GMP`), SDPX extension load
state (`Base.get_extension`), SHA1 of `Project.toml`/`Manifest.toml`.
Any failed read prints `name=READ_FAILED ...` and the script exits
non-zero (`reproducible_delivery_check=FAIL`); all green prints `PASS`.
Run (single bounded process, ≤180 s):
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PKG_OFFLINE=true julia \
--startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G \
--project=/tmp/sdpx-r4r5r6-env-1788946769 \
scripts/check_reproducible_delivery.jl
```
Evidence the fingerprint points at:
- `docs/evidence/` — committed qualification records.
- `local-archives/high-precision-ecosystem-20260908/` — frozen
  high-precision ecosystem records (outside git; verify presence locally).
Known release blocker: `Manifest.toml` is not committed, so the
script fail-closes on `manifest_sha1` until a manifest is committed.
