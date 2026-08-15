# External benchmark cache

This directory contains metadata only. `cache/` is ignored and is populated
only by an explicit runner request:

```sh
julia --project=. benchmark/runner.jl representative --prepare --problem=netlib/afiro
```

Neither `Pkg.test()`, the solver, nor an ordinary benchmark run accesses the
network. A cached file is accepted only when its SHA-256 matches the registry.
External formats without a current SDPX loader remain structured skips even
when the data file is cached.

The registered Full-unitarity-EFT payload is installed manually as:

```text
<cache-root>/csdr/full-unitarity-eft-j40-na15-nmu200-nx2-nalpha2-v1.bin
```

The registry pins SHA-256
`ae66d61cdf2b00d46fd6ab83438c4e07bce3134a0fcd54519b7f7d5fce2533e8`.
The runner validates this digest before loading Julia Serialization data. Treat
the payload as a trusted, versioned artifact; do not deserialize untrusted
files. Larger ladder entries remain metadata-only until separately generated
artifacts and checksums exist.
