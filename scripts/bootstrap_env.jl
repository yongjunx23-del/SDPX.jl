#=====================================================================#
#  scripts/bootstrap_env.jl — build the provider-capable REBUILD_ENV
#
#  `docs/rebuild/baseline.md` §2 records that the frozen `Manifest.toml`
#  resolves no provider: `MultiFloats`, `MultiFloatLinearAlgebra`,
#  `BigFloatLinearAlgebra` and `QDLDL` are `[weakdeps]`, so the default
#  `Pkg.test()` path exercises Float64 only and every MF/BF capability claim
#  is unverified there.  The same section records that the packet's
#  `scripts/bootstrap_env.jl` "would be a third such environment and has
#  **not been executed**".  This file is that script.
#
#  It exists so that every task whose acceptance depends on MF/BF can state
#  HOW its environment is produced and record the resulting Manifest, which
#  §2 makes mandatory.
#
#  Recipe
#  ------
#  Identical in substance to the supported `scripts/provider_smoke.sh`:
#  `Pkg.develop` the three local checkouts and `Pkg.add` the two registered
#  arithmetic packages.  The checkouts are dev'ed, never cloned, because
#  MFLA is unregistered and because the packet pins both providers to frozen
#  local revisions (`50e6e0b` = MFLA v0.4.0, `f95d3e6` = BFLA v0.3.0).  A
#  registry version would silently violate that pin: the stale `extenv` in
#  the workspace depot resolves MFLA 0.2.0 and BFLA 0.1.1.
#
#  Usage
#  -----
#      REBUILD_ENV=/abs/path/to/rebuild-env \
#        julia --startup-file=no scripts/bootstrap_env.jl
#
#  or pass the environment path as the first argument.  The script activates
#  (creating if needed) that environment and leaves a resolvable
#  `Project.toml` + `Manifest.toml` behind.
#
#  The environment is NOT precompiled here on purpose: precompiling SDPX with
#  both provider extensions present is the step that must be split across two
#  processes (see the note at the end of this file).
#=====================================================================#

using Pkg

const ROOT = normpath(joinpath(@__DIR__, ".."))
const WORKSPACE = normpath(joinpath(ROOT, ".."))

env_path = if !isempty(ARGS)
    ARGS[1]
elseif haskey(ENV, "REBUILD_ENV") && !isempty(ENV["REBUILD_ENV"])
    ENV["REBUILD_ENV"]
else
    ""
end

if isempty(env_path)
    println(stderr, """
        bootstrap_env.jl: no environment given.

        Pass it as the first argument or set REBUILD_ENV, e.g.

            REBUILD_ENV=$WORKSPACE/rebuild-env \\
              julia --startup-file=no $ROOT/scripts/bootstrap_env.jl
        """)
    exit(2)
end

mfla = get(ENV, "SDPX_MFLA_PROJECT", joinpath(WORKSPACE, "MultiFloatLinearAlgebra.jl"))
bfla = get(ENV, "SDPX_BFLA_PROJECT", joinpath(WORKSPACE, "BigFloatLinearAlgebra.jl"))

for (label, path) in (("SDPX", ROOT), ("MFLA", mfla), ("BFLA", bfla))
    isdir(path) || (println(stderr, "bootstrap_env.jl: $label checkout not found at $path"); exit(1))
    isfile(joinpath(path, "Project.toml")) ||
        (println(stderr, "bootstrap_env.jl: $label has no Project.toml at $path"); exit(1))
end

mkpath(env_path)
Pkg.activate(env_path)
Pkg.develop([PackageSpec(path=ROOT), PackageSpec(path=mfla), PackageSpec(path=bfla)])
Pkg.add(["MultiFloats", "GenericLinearAlgebra"])
Pkg.instantiate()

println()
println("REBUILD_ENV = ", env_path)
println("Manifest    = ", joinpath(env_path, "Manifest.toml"))
println()

# Record what was actually resolved.  The provider versions are the load
# bearing part: they must be 0.4.x (MFLA) and 0.3.x (BFLA) to match the
# packet's frozen revisions.
deps = Pkg.dependencies()
for (uuid, dep) in deps
    dep.name in ("MultiFloatLinearAlgebra", "BigFloatLinearAlgebra", "MultiFloats",
                 "GenericLinearAlgebra", "SDPX") || continue
    source = dep.source === nothing ? "" : string(dep.source)
    println(rpad(dep.name, 26), rpad(string(dep.version), 10), source)
end

println()
println("""
    Next: verify the extensions load.  Julia 1.12 can exhaust its inference
    compiler when the MFLA fixed-width specializations and the BFLA/MPFR
    specialization are compiled in the SAME process, so load them in two
    separate processes (`-t1`), exactly as `scripts/provider_smoke.sh` does:

        julia --startup-file=no --project="$env_path" -t1 -e \\
          'using SDPX, MultiFloats, MultiFloatLinearAlgebra; \\
           println(Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt))'

        julia --startup-file=no --project="$env_path" -t1 -e \\
          'using SDPX, BigFloatLinearAlgebra; \\
           println(Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt))'
    """)
