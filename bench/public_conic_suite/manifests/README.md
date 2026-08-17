# Manifest notes

- `downloads.toml`: direct public URLs for external LP, SOCP, and SDP data.
- `pathological_suites.toml`: deterministic parameter grids for generated
  numerical stress tests.

Downloaded files are not committed. The downloader records the source URL,
size, and SHA-256 digest in `data/download_lock.toml` so a local benchmark
campaign can pin the exact inputs it used.
