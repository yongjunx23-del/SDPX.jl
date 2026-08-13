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
