# identity.jl -- emit identity.toml for one evidence run (P0-00/P0-02).
#
# Usage:
#   julia --project=<env> identity.jl --repo-root ABSOLUTE_PATH \
#       --out ABSOLUTE_PATH [--input-sha HEX] [--threads N] [--policy TEXT]
#
# Records: source SHA + tree, pathof(SDPX), Julia version/commit,
# Sys.MACHINE/Sys.KERNEL, CPU brand, thread config (Julia threads, requested
# solve threads, BLAS env + effective BLAS threads), provider package
# versions/SHAs (MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra,
# GenericLinearAlgebra), and the input SHA. Any field that cannot be
# determined is written as the literal string "missing" -- never guessed.

using SDPX
using Pkg
using TOML
import LinearAlgebra

function _argval(argv, key, default=nothing)
    for i in 1:length(argv)
        if argv[i] == key && i < length(argv)
            return argv[i + 1]
        end
    end
    return default
end

function _try(f::Function)
    try
        v = f()
        return v === nothing ? "missing" : v
    catch
        return "missing"
    end
end

function _git(repo::AbstractString, args::String...)
    out = try
        strip(read(Cmd(["git", "-C", repo, args...]), String))
    catch
        return "missing"
    end
    return isempty(out) ? "missing" : out
end

function _provider_info(names::Vector{String})
    deps = try
        Pkg.dependencies()
    catch
        return Dict{String,Any}(n => "missing" for n in names)
    end
    info = Dict{String,Any}()
    for (uuid, dep) in deps
        dep.name in names || continue
        tree = try
            h = getproperty(dep, :tree_hash)
            h === nothing ? "missing" : string(h)
        catch
            "missing"
        end
        info[dep.name] = Dict{String,Any}(
            "version" => dep.version === nothing ? "missing" : string(dep.version),
            "tree_sha" => tree,
            "path" => try
                String(pathof(Base.require(Main, Symbol(dep.name))))
            catch
                "missing"
            end,
        )
    end
    for n in names
        haskey(info, n) || (info[n] = "missing")
    end
    return info
end

function main(argv::Vector{String})
    if any(a -> a == "--help" || a == "-h", argv)
        println("usage: identity.jl --repo-root PATH --out PATH [--input-sha HEX] [--threads N] [--policy TEXT]")
        exit(0)
    end
    repo = _argval(argv, "--repo-root", nothing)
    out = _argval(argv, "--out", nothing)
    (repo === nothing || out === nothing) &&
        (println(stderr, "identity.jl: --repo-root and --out are required"); exit(2))
    input_sha = _argval(argv, "--input-sha", "missing")
    req_threads = _argval(argv, "--threads", "missing")
    policy = _argval(argv, "--policy", "missing")

    cpu_brand = _try() do
        infos = Sys.cpu_info()
        isempty(infos) ? "missing" : String(infos[1].model)
    end
    julia_commit = _try(() -> Base.GIT_VERSION_INFO.commit)
    blas_env = Dict{String,Any}(
        "OPENBLAS_NUM_THREADS" => get(ENV, "OPENBLAS_NUM_THREADS", "missing"),
        "OMP_NUM_THREADS" => get(ENV, "OMP_NUM_THREADS", "missing"),
        "MKL_NUM_THREADS" => get(ENV, "MKL_NUM_THREADS", "missing"),
        "JULIA_NUM_THREADS" => get(ENV, "JULIA_NUM_THREADS", "missing"),
        "effective_blas_threads" => _try(() -> LinearAlgebra.BLAS.get_num_threads()),
    )
    doc = Dict{String,Any}(
        "source" => Dict{String,Any}(
            "repo_root" => String(repo),
            "sha" => _git(repo, "rev-parse", "HEAD"),
            "tree" => _git(repo, "rev-parse", "HEAD^{tree}"),
            "status_porcelain" => _git(repo, "status", "--porcelain=v1"),
        ),
        "sdpx" => Dict{String,Any}(
            "pathof" => _try(() -> String(pathof(SDPX))),
            "version" => _try(() -> string(Pkg.dependencies()[Base.UUID(
                "9c19f76d-03c5-4610-b403-7c8fdd8897fd")].version)),
        ),
        "julia" => Dict{String,Any}(
            "version" => string(VERSION),
            "commit" => julia_commit,
        ),
        "host" => Dict{String,Any}(
            "machine" => _try(() -> String(Sys.MACHINE)),
            "kernel" => _try(() -> String(Sys.KERNEL)),
            "cpu_brand" => cpu_brand,
        ),
        "threads" => Dict{String,Any}(
            "julia_threads" => Threads.nthreads(),
            "requested_solve_threads" => req_threads,
            "blas" => blas_env,
        ),
        "providers" => _provider_info([
            "MultiFloats", "MultiFloatLinearAlgebra",
            "BigFloatLinearAlgebra", "GenericLinearAlgebra",
        ]),
        "input" => Dict{String,Any}("sha" => input_sha),
        "policy" => policy,
    )
    mkpath(dirname(out))
    open(out, "w") do io
        TOML.print(io, doc)
    end
    println("identity.jl: wrote $out")
end

main(ARGS)
