# QDLDL/BFLA independent bounded provider qualification (R3, NOT native-route integration).
#
# Scope: qualifies the OPTIONAL QDLDL-backed BigFloat sparse-LDL extension in
# isolation. It does NOT connect SDPX sparse high-precision routes and makes no
# claim about SDPX native integration.
#
# Bounds: matrices <= 48 (this file uses n = 12 only), single Julia/GC/BLAS
# thread, `--heap-size-hint=2G`, each Julia command documented with exit code.
# Run with the fresh private env (QDLDL pinned 0.4.1, BFLA dev'ed):
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
#   julia --heap-size-hint=2G --project=/private/tmp/qdldl-bfla-qual-env \
#     validation/precision_ecosystem/qdldl_bfla_qualification.jl
#
# Lanes:
#   A. Existing upstream BFLA QDLDL extension tests (verbatim include).
#   B. Independent bounded qualification at 256/512/1024 bits with
#      ORIGINAL-matrix residuals, fixed-pattern numeric refactor, repeated RHS,
#      input/cache/workspace limb isolation, and rejection controls.

using Test
using Base.Threads
using LinearAlgebra
using SparseArrays
using SHA
using TOML
import BigFloatLinearAlgebra
import QDLDL

const BFLA = BigFloatLinearAlgebra

# Hash the code actually loaded, not an unrelated expected checkout. These
# defaults pin this qualification; a different revision requires explicit input.
const BFLA_WORKTREE = realpath(pkgdir(BFLA))
const BFLA_EXT_FILE = joinpath(BFLA_WORKTREE, "ext/BigFloatQDLDLExt.jl")
const BFLA_QDLDL_TEST = joinpath(BFLA_WORKTREE, "test/qdldl_extension.jl")
const QDLDL_PKGDIR = realpath(pkgdir(QDLDL))
const QDLDL_SRC = realpath(pathof(QDLDL))
const EXPECTED_BFLA_HEAD = get(ENV, "SDPX_QUAL_BFLA_HEAD",
    "aaa71f33252ce712dbdb0a798d9328a442700726")
const EXPECTED_QDLDL_SHA = get(ENV, "SDPX_QUAL_QDLDL_SHA256",
    "8759d60e456578d709b869cb66d4f7796f4defd10cc19186dd2b7cdb5c3c8fde")

_sha256_file(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function _qualification_manifest()
    string(Base.PkgId(BFLA).uuid) == "44d352a4-380e-4c6a-9c2a-31e5bfe329aa" ||
        error("unexpected BFLA package identity")
    string(Base.PkgId(QDLDL).uuid) == "bfc457fd-c171-5ab7-bd9e-d5dbfc242d63" ||
        error("unexpected QDLDL package identity")
    Base.pkgversion(BFLA) == v"0.3.0" || error("unexpected BFLA version")
    Base.pkgversion(QDLDL) == v"0.4.1" || error("unexpected QDLDL version")
    realpath(pathof(BFLA)) == realpath(joinpath(BFLA_WORKTREE,"src/BigFloatLinearAlgebra.jl")) ||
        error("BFLA entrypoint/root mismatch")
    QDLDL_SRC == realpath(joinpath(QDLDL_PKGDIR,"src/QDLDL.jl")) ||
        error("QDLDL entrypoint/root mismatch")
    head = readchomp(`git -C $BFLA_WORKTREE rev-parse HEAD`)
    head == EXPECTED_BFLA_HEAD || error("unexpected BFLA source revision")
    isempty(readchomp(`git -C $BFLA_WORKTREE status --porcelain`)) ||
        error("BFLA source is dirty")
    _sha256_file(QDLDL_SRC) == EXPECTED_QDLDL_SHA ||
        error("unexpected QDLDL source contents")
    Base.get_extension(BFLA,:BigFloatQDLDLExt) !== nothing || error("extension absent")
    BFLA.sparse_ldlt_available(BigFloat) || error("sparse provider unavailable")
    hashes = Dict{String,String}()
    for (name,root) in (("BFLA",BFLA_WORKTREE),("QDLDL",QDLDL_PKGDIR))
        hashes[name*"/Project.toml"] = _sha256_file(joinpath(root,"Project.toml"))
        for sub in ("src","ext")
            isdir(joinpath(root,sub)) || continue
            for (dir,_,files) in walkdir(joinpath(root,sub)), file in files
                path=joinpath(dir,file)
                hashes[name*"/"*relpath(path,root)] = _sha256_file(path)
            end
        end
    end
    hashes["included_upstream_test"] = _sha256_file(BFLA_QDLDL_TEST)
    hashes["harness"] = _sha256_file(@__FILE__)
    project = Base.active_project()
    project === nothing && error("explicit qualification project required")
    return Dict("julia_version"=>string(VERSION),"julia_threads"=>Threads.nthreads(),
        "blas_threads"=>LinearAlgebra.BLAS.get_num_threads(),
        "ambient_precision_bits"=>precision(BigFloat),"bfla_root"=>BFLA_WORKTREE,
        "bfla_head"=>head,"qdldl_root"=>QDLDL_PKGDIR,"qdldl_entrypoint"=>QDLDL_SRC,
        "project"=>project,"project_sha256"=>_sha256_file(project),
        "manifest_sha256"=>_sha256_file(joinpath(dirname(project),"Manifest.toml")),
        "harness_head"=>readchomp(`git -C $(@__DIR__) rev-parse HEAD`),
        "harness_status"=>readchomp(`git -C $(@__DIR__) status --porcelain`),
        "sha256"=>hashes)
end

function _print_manifest()
    data = _qualification_manifest()
    println("=== QUAL MANIFEST ===")
    TOML.print(stdout,data;sorted=true)
    println("\n=== END QUAL MANIFEST ===")
    flush(stdout)
    return data
end

# --- deterministic quasi-definite fixture (upper-triangle CSC, BigFloat) ---
# Kfull = [Dx Ar'; Ar -Dy], Dx, Dy positive diagonal; Kupper stored.
function _build_quasidefinite(p::Int; n::Int=12, shift::String="0")
    @assert n <= 48
    nr = n ÷ 2
    m = n - nr
    setprecision(BigFloat, p) do
        K = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        for i in 1:nr
            K[i, i] = BigFloat(2; precision=p) +
                      BigFloat("0.125"; precision=p) * BigFloat(((i * 7) % 5); precision=p) +
                      BigFloat(shift; precision=p)
        end
        for i in 1:m
            K[nr + i, nr + i] = -(BigFloat(2; precision=p) +
                                  BigFloat("0.0625"; precision=p) * BigFloat(((i * 3) % 5); precision=p))
        end
        for i in 1:m, j in 1:nr
            v = BigFloat(((-1)^(i + j)) * (0.25 + 0.0625 * ((i * 3 + j * 5) % 7)); precision=p)
            K[nr + i, j] = v
            K[j, nr + i] = v
        end
        # Upper-triangle CSC (only stored triangle fed to QDLDL).
        rows = Int[]
        cols = Int[]
        vals = BigFloat[]
        for col in 1:n, row in 1:col
            if !iszero(K[row, col])
                push!(rows, row)
                push!(cols, col)
                push!(vals, BigFloat(K[row, col]; precision=p))
            end
        end
        # Every structural column nonempty holds by nonzero diagonal.
        A = sparse(rows, cols, vals, n, n)
        A = SparseMatrixCSC(n, n, copy(A.colptr), copy(A.rowval),
                            BFLA.owned_copy(A.nzval; precision_bits=p))
        dsigns = vcat(fill(1, nr), fill(-1, m))
        b = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        for i in 1:n
            b[i] = BigFloat(((-1)^i) * (1.0 + 0.125 * i); precision=p)
        end
        return K, A, dsigns, b, nr
    end
end

function _dense_of_upper(A::SparseMatrixCSC{BigFloat}, p::Int)
    n = size(A, 1)
    return setprecision(BigFloat, p) do
        K = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        for col in axes(A, 2), ptr in nzrange(A, col)
            row = A.rowval[ptr]
            K[row, col] = A.nzval[ptr]
            K[col, row] = A.nzval[ptr]
        end
        K
    end
end

const _QUALIFICATION_BEFORE = _print_manifest()
@testset "QDLDL/BFLA bounded provider qualification" begin
    @test Threads.nthreads() == 1

    # ---- Lane A: existing upstream extension tests (verbatim) ----
    @testset "lane A: upstream qdldl_extension.jl" begin
        @test BFLA.sparse_ldlt_available(BigFloat)
        @test Base.get_extension(BFLA, :BigFloatQDLDLExt) !== nothing
        include(BFLA_QDLDL_TEST)
    end

    # ---- Lane B: independent bounded qualification ----
    for p in (256, 512, 1024)
        @testset "lane B: independent qualification p=$p" begin
            setprecision(BigFloat, p) do
                K0, A0, dsigns, b0, nr = _build_quasidefinite(p)
                n = size(A0, 1)
                @test n <= 48

                # Extension truly loaded (not mere adapter presence).
                @test BFLA.sparse_ldlt_available(BigFloat)
                ext = Base.get_extension(BFLA, :BigFloatQDLDLExt)
                @test ext !== nothing

                cache = BFLA.sparse_ldlt_cache(A0; precision_bits=p, dsigns=dsigns, nrhs=2)
                @test cache.precision_bits == p
                @test cache.n == n
                factor0 = something(cache.factor)
                @test factor0 isa QDLDL.QDLDLFactorisation{BigFloat}
                @test eltype(factor0.L) == BigFloat
                @test eltype(factor0.Dinv.diag) == BigFloat
                # Actual factor scalar precision/ownership at construction.
                @test all(precision(v) == p for v in factor0.L.nzval)
                @test all(precision(v) == p for v in factor0.Dinv.diag)
                @test factor0.workspace.Dsigns === nothing
                @test iszero(factor0.workspace.regularize_eps)
                @test iszero(factor0.workspace.regularize_delta)
                @test precision(factor0.workspace.regularize_eps) == p
                # Input limbs are not aliased by the cache authority.
                @test all(c !== i for c in cache.matrix.nzval for i in A0.nzval)
                # Hidden QDLDL triuA owns limbs independently of input/cache.
                @test all(h !== c for h in factor0.workspace.triuA.nzval for c in cache.matrix.nzval)
                @test all(h !== i for h in factor0.workspace.triuA.nzval for i in A0.nzval)

                # Mutating the caller input after construction changes nothing.
                input_snapshot = BFLA.owned_copy(cache.matrix.nzval; precision_bits=p)
                A0.nzval[1] += BigFloat(1; precision=p)
                @test cache.matrix.nzval == input_snapshot
                A0.nzval[1] -= BigFloat(1; precision=p)

                BFLA.factorize!(cache, A0)
                @test BFLA.issuccess(cache)
                diag = BFLA.factor_diagnostics(cache)
                @test diag.provider === :qdldl
                @test diag.kind === :sparse_ldlt
                @test diag.symbolic_count == 1
                @test diag.numeric_factor_count == 1
                @test diag.positive_inertia == nr
                @test diag.regularized_entries == 0
                @test diag.precision_bits == p
                nnz_l_first = diag.nnz_l
                @test nnz_l_first > 0
                factor = something(cache.factor)
                @test all(precision(v) == p && isfinite(v) for v in factor.L.nzval)
                @test all(precision(v) == p && isfinite(v) for v in factor.Dinv.diag)
                @test all(precision(v) == p && isfinite(v) for v in factor.workspace.triuA.nzval)

                # Independent ORIGINAL-matrix residual (dense K0, not factor echo).
                x = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                BFLA.solve_trusted!(x, cache, b0)
                @test all(precision(v) == p for v in x)
                res = norm(K0 * x - b0, Inf)
                bound = BigFloat(2; precision=p)^(32 - p)
                @test res <= bound

                # Repeated RHS: second vector solve + multi-RHS width 2.
                b1 = BFLA.owned_copy(b0; precision_bits=p)
                b1[1] += BigFloat("0.5"; precision=p)
                x1 = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                BFLA.solve_trusted!(x1, cache, b1)
                @test norm(K0 * x1 - b1, Inf) <= bound
                B = BFLA.owned_zeros(BigFloat, n, 2; precision_bits=p)
                BFLA.copy_owned!(view(B, :, 1), b0)
                BFLA.copy_owned!(view(B, :, 2), b1)
                X = BFLA.owned_zeros(BigFloat, n, 2; precision_bits=p)
                BFLA.solve_trusted!(X, cache, B)
                @test norm(K0 * X - B, Inf) <= bound

                # Fixed-pattern numeric refactor: same pattern, shifted values.
                _, A2, _, _, _ = _build_quasidefinite(p; shift="0.125")
                @test A2.colptr == cache.frozen_colptr
                @test A2.rowval == cache.frozen_rowval
                K2 = _dense_of_upper(A2, p)
                BFLA.factorize!(cache, A2)
                @test BFLA.issuccess(cache)
                diag2 = BFLA.factor_diagnostics(cache)
                # Symbolic/fill reuse distinguished from adapter presence:
                # symbolic count stays 1, numeric count advances, L fill unchanged.
                @test diag2.symbolic_count == 1
                @test diag2.numeric_factor_count == 2
                @test diag2.nnz_l == nnz_l_first
                @test diag2.positive_inertia == nr
                @test diag2.regularized_entries == 0
                x2 = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                BFLA.solve_trusted!(x2, cache, b0)
                @test norm(K2 * x2 - b0, Inf) <= bound

                # Workspace limb isolation: factor storage rejects alias destinations.
                @test_throws ArgumentError BFLA.solve_trusted!(factor.Dinv.diag, cache, b0)

                # ---- rejection controls ----
                # Singular (all-zero values, same pattern): fail closed.
                Z = SparseMatrixCSC(n, n, copy(A2.colptr), copy(A2.rowval),
                                    BFLA.owned_zeros(BigFloat, length(A2.nzval); precision_bits=p))
                BFLA.factorize!(cache, Z; check=false)
                @test !BFLA.issuccess(cache)
                @test BFLA.factor_status(cache).kind === :pivot_failure
                @test BFLA.factor_diagnostics(cache).positive_inertia == -1
                BFLA.factorize!(cache, A2)
                @test BFLA.issuccess(cache)
                @test_throws Exception BFLA.factorize!(cache, Z)

                # Invalid: empty structural column rejects at construction.
                bad_colptr = copy(A2.colptr)
                bad_colptr[2] = bad_colptr[1] # empty column 1
                bad_empty = SparseMatrixCSC(n, n, bad_colptr, copy(A2.rowval),
                                            BFLA.owned_copy(A2.nzval; precision_bits=p))
                @test_throws ArgumentError BFLA.sparse_ldlt_cache(
                    bad_empty; precision_bits=p, dsigns=dsigns, nrhs=1)

                # Invalid: lower-triangle storage rejects.
                Klow = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
                for col in 1:n, row in col:n
                    Klow[row, col] = K2[row, col]
                end
                @test_throws ArgumentError BFLA.sparse_ldlt_cache(
                    sparse(Klow); precision_bits=p, dsigns=dsigns, nrhs=1)

                # Invalid: D-sign vector rejects (length / values).
                @test_throws DimensionMismatch BFLA.sparse_ldlt_cache(
                    A2; precision_bits=p, dsigns=vcat(dsigns, Int[1]), nrhs=1)
                @test_throws ArgumentError BFLA.sparse_ldlt_cache(
                    A2; precision_bits=p, dsigns=zeros(Int, n), nrhs=1)

                # Invalid: nonfinite value rejects.
                bad_nf = SparseMatrixCSC(n, n, copy(A2.colptr), copy(A2.rowval),
                                         BFLA.owned_copy(A2.nzval; precision_bits=p))
                bad_nf.nzval[1] = BigFloat(NaN; precision=p)
                @test_throws DomainError BFLA.factorize!(cache, bad_nf)

                # Invalid: ambient precision mismatch revokes solve authority.
                setprecision(BigFloat, 64) do
                    @test_throws BFLA.PrecisionMismatch BFLA.factorize!(cache, A2)
                end
                @test BFLA.factor_status(cache).kind === :unprepared
                BFLA.factorize!(cache, A2)
                @test BFLA.issuccess(cache)

                # No dynamic regularization observed anywhere in this lane.
                @test BFLA.factor_diagnostics(cache).regularized_entries == 0
                @test something(cache.factor).workspace.regularize_count[1] == 0
            end
        end
    end
end
@assert _qualification_manifest() == _QUALIFICATION_BEFORE "qualification source/environment changed"
println("QUALIFICATION_SOURCE_UNCHANGED")
