#!/usr/bin/env julia

"""Resolve the release-specific cluster environment on a login node.

The compute-node jobs use the resulting manifest in offline mode. MKL is an
optional Linux/x86_64 benchmark backend and is not a dependency of SDPX.
"""

using Pkg

source = get(ENV, "SDPX_RELEASE_SOURCE", "")
isempty(source) &&
    error("SDPX_RELEASE_SOURCE must name the immutable release source")

Pkg.develop(PackageSpec(path=source))
Pkg.add([
    PackageSpec(name="Aqua", version="0.8"),
    PackageSpec(name="Convex", version="0.16"),
    PackageSpec(name="DoubleFloats", version="1"),
    PackageSpec(name="JLD2"),
    PackageSpec(name="JSON"),
    PackageSpec(name="JuMP", version="1"),
    PackageSpec(name="MultiFloats", version="3"),
    PackageSpec(name="StableRNGs"),
])

enable_mkl =
    lowercase(get(ENV, "SDPX_ENABLE_MKL", "false")) in
    ("1", "true", "yes")
if enable_mkl
    Sys.islinux() && Sys.ARCH === :x86_64 ||
        error("the optional MKL benchmark requires Linux x86_64")
    Pkg.add(PackageSpec(name="MKL"))
end

enable_blis =
    lowercase(get(ENV, "SDPX_ENABLE_BLIS", "false")) in
    ("1", "true", "yes")
if enable_blis
    Sys.islinux() && Sys.ARCH === :x86_64 ||
        error("the optional BLIS benchmark requires Linux x86_64")
    Pkg.add(PackageSpec(name="BLISBLAS"))
end

Pkg.instantiate()
Pkg.precompile()
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
