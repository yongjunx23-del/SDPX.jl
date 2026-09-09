# Reproducible-delivery check (R6-D scaffolding, stdlib-only).
#
# Prints the delivery fingerprint and exits non-zero if ANY read fails:
#   1. git HEAD of this checkout
#   2. Julia VERSION
#   3. SDPX version from Project.toml
#   4. MPFR / GMP versions from Base.MPFR / Base.GMP
#   5. loaded SDPX extension list via Base.get_extension
#   6. SHA1 of Project.toml and Manifest.toml
#
# Run with the bounded-process contract, e.g.:
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#   JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PKG_OFFLINE=true \
#   julia --startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G \
#     --project=/tmp/sdpx-r4r5r6-env-1788946769 scripts/check_reproducible_delivery.jl

using SHA
using TOML

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const FAILURES = String[]

function report!(name::String, thunk::Function)
    value = try
        thunk()
    catch error
        push!(FAILURES, name)
        println(name, "=READ_FAILED ", sprint(showerror, error))
        return nothing
    end
    println(name, "=", value)
    return value
end

function main()
    report!("repo_root", () -> REPO_ROOT)
    report!("git_head", () -> strip(
        read(`git -C $REPO_ROOT rev-parse HEAD`, String),
    ))
    report!("julia_version", () -> string(VERSION))
    report!("sdpx_version", () -> begin
        project = TOML.parsefile(joinpath(REPO_ROOT, "Project.toml"))
        haskey(project, "version") ||
            error("Project.toml has no version field")
        project["version"]
    end)
    # Base.MPFR / Base.GMP always ship with Julia; no new dependency.
    report!("mpfr_version", () -> string(Base.MPFR.version()))
    report!("gmp_version", () -> string(Base.GMP.version()))
    report!("sdpx_extensions", () -> begin
        sdpx = Base.require(Main, :SDPX)
        names = (
            :SDPXAppleAccelerateExt,
            :SDPXBigFloatLinearAlgebraExt,
            :SDPXGenericLinearAlgebraExt,
            :SDPXJLD2Ext,
            :SDPXMultiFloatLinearAlgebraExt,
            :SDPXMultiFloatsExt,
        )
        join(
            (
                string(name) * ":" *
                string(Base.get_extension(sdpx, name) !== nothing)
                for name in names
            ),
            ",",
        )
    end)
    report!("project_sha1", () -> bytes2hex(SHA.sha1(
        read(joinpath(REPO_ROOT, "Project.toml")),
    )))
    report!("manifest_sha1", () -> bytes2hex(SHA.sha1(
        read(joinpath(REPO_ROOT, "Manifest.toml")),
    )))
    if isempty(FAILURES)
        println("reproducible_delivery_check=PASS")
        return 0
    end
    println(
        "reproducible_delivery_check=FAIL failures=",
        join(FAILURES, ","),
    )
    return 1
end

exit(main())
