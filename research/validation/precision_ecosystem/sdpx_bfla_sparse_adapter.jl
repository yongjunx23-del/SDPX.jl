# SDPX BigFloat sparse signed-LDL INTERNAL adapter validation (R3, bounded).
#
# Scope: exercises the INTERNAL `SDPX.SparseQDLDLCache{BigFloat}` adapter that
# bridges the existing SDPX cache lifecycle to the reviewed BFLA QDLDL
# extension (BFLA aaa71f3) over QDLDL 0.4.1.  This is NOT native-route
# integration: no public `optimize!` path is touched, BigFloat
# `sparse_augmented` Settings stay disabled, and native high-precision routing
# remains unqualified.
#
# Reuse (parent direction): the genuine-core construction, the borrowed
# expanded-route rung-0 shift policy (`delta = sqrt(eps)*scale`), the
# ORIGINAL-operator residual authority, and the QDLDL qualification bounds
# mirror `validation/precision_ecosystem/signed_kkt_provider_reference.jl`
# and `validation/precision_ecosystem/qdldl_bfla_qualification.jl`.  The
# difference under test is the seam: every factor/solve here goes through the
# SDPX cache (`SDPX.factorize!` / `SDPX.solve!` / `SDPX.solve_multi!` /
# `SDPX.refine_once!`), never through `BFLA.sparse_ldlt_cache` directly.
#
# Bounds: genuine full-rank fixtures with n = 14 <= 32 (nr = 6 <= m = 8,
# explicit triangular minor as the structural rank certificate), single
# Julia/GC/BLAS thread, `--heap-size-hint=2G`.  Caller-owned shifts only; no
# rank deletion, no hidden precision, no dynamic regularization.
#
# Run (private env only: approved BFLA worktree + QDLDL 0.4.1 + this SDPX
# worktree dev'ed; each command is single-threaded and heap-bounded):
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
#   julia --startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G \
#     --project=/private/tmp/sdpx-bfla-adapter-env \
#     validation/precision_ecosystem/sdpx_bfla_sparse_adapter.jl

using Test
using Base.Threads
using LinearAlgebra
using SparseArrays
using SHA
import BigFloatLinearAlgebra
import QDLDL
using SDPX

const BFLA = BigFloatLinearAlgebra

const EXPECTED_BFLA_HEAD = get(ENV, "SDPX_QUAL_BFLA_HEAD",
    "aaa71f33252ce712dbdb0a798d9328a442700726")
const EXPECTED_QDLDL_VERSION = v"0.4.1"

# Bind the harness to the LOADED SDPX source, not just the script checkout:
# a different SDPX checkout could otherwise execute while this checkout
# receives the receipt.  Both roots must agree as real paths.
function _assert_sdpx_source(script_root::AbstractString, loaded_root::AbstractString)
    realpath(script_root) == realpath(loaded_root) || error(
        "harness script checkout $(realpath(script_root)) disagrees with " *
        "loaded SDPX source $(realpath(loaded_root))",
    )
    return realpath(script_root)
end

function _adapter_manifest()
    string(Base.PkgId(BFLA).uuid) == "44d352a4-380e-4c6a-9c2a-31e5bfe329aa" ||
        error("unexpected BFLA package identity")
    string(Base.PkgId(QDLDL).uuid) == "bfc457fd-c171-5ab7-bd9e-d5dbfc242d63" ||
        error("unexpected QDLDL package identity")
    Base.pkgversion(BFLA) == v"0.3.0" || error("unexpected BFLA version")
    Base.pkgversion(QDLDL) == EXPECTED_QDLDL_VERSION ||
        error("unexpected QDLDL version")
    bfla_root = realpath(pkgdir(BFLA))
    head = readchomp(`git -C $bfla_root rev-parse HEAD`)
    head == EXPECTED_BFLA_HEAD || error("unexpected BFLA source revision")
    isempty(readchomp(`git -C $bfla_root status --porcelain`)) ||
        error("BFLA source is dirty")
    Base.get_extension(BFLA, :BigFloatQDLDLExt) !== nothing ||
        error("BFLA QDLDL extension absent")
    BFLA.sparse_ldlt_available(BigFloat) || error("sparse provider unavailable")
    SDPX.SparseQDLDLProviderAvailable(BigFloat) ||
        error("SDPX BFLA sparse adapter unavailable")
    sdpx_root = _assert_sdpx_source(
        joinpath(@__DIR__, "..", ".."), pkgdir(SDPX),
    )
    return Dict(
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "bfla_root" => bfla_root,
        "bfla_head" => head,
        "qdldl_version" => string(Base.pkgversion(QDLDL)),
        "sdpx_root" => sdpx_root,
        "sdpx_loaded" => realpath(pkgdir(SDPX)),
        "sdpx_head" => readchomp(`git -C $sdpx_root rev-parse HEAD`),
        "sdpx_status" => readchomp(`git -C $sdpx_root status --porcelain`),
    )
end

# --- deterministic genuine full-rank ORIGINAL core K0 = [0 A'; A -Theta] ----
# nr <= m with an explicit triangular minor: rows/cols 1..nr are
# lower-triangular with nonzero diagonal (structural rank certificate), the
# remaining rows arbitrary, Theta diagonal positive.  Hence rank(A) = nr and
# rank(K0) = m + nr = n exactly (null(K0) = {(z,0) : Az = 0}).
function _genuine_core(p::Int)
    nr, m = 6, 8
    n = nr + m
    @assert n <= 32
    return setprecision(BigFloat, p) do
        A = zeros(BigFloat, m, nr)
        for i in 1:m, j in 1:nr
            if j > i
                @assert iszero(A[i, j])
            elseif j == i
                A[i, j] = BigFloat(1; precision=p) +
                    BigFloat("0.125"; precision=p) * BigFloat((3 * i) % 4; precision=p)
            else
                A[i, j] = BigFloat(((-1)^(i + j)) * (0.5 + 0.0625 * ((3 * i + 5 * j) % 9)); precision=p)
            end
        end
        Theta = zeros(BigFloat, m)
        for j in 1:m
            Theta[j] = BigFloat(1; precision=p) +
                BigFloat("0.125"; precision=p) * BigFloat((7 * j) % 5; precision=p)
        end
        K0 = zeros(BigFloat, n, n)
        for i in 1:m, j in 1:nr
            K0[j, nr + i] = A[i, j]
            K0[nr + i, j] = A[i, j]
        end
        for j in 1:m
            K0[nr + j, nr + j] = -Theta[j]
        end
        for j in 1:nr
            @assert iszero(K0[j, j])  # raw core is NOT quasi-definite as stored
        end
        x_true = zeros(BigFloat, n)
        for i in 1:n
            x_true[i] = BigFloat(((-1)^i) * (1.0 + 0.125 * ((5 * i) % 7)); precision=p)
        end
        b = K0 * x_true
        (K0=K0, b=b, x_true=x_true, nr=nr, m=m, n=n)
    end
end

function _operator_scale(K0::Matrix{BigFloat}, p::Int)
    return setprecision(BigFloat, p) do
        scale = BigFloat(0; precision=p)
        for row in axes(K0, 1)
            s = BigFloat(0; precision=p)
            for col in axes(K0, 2)
                s += abs(K0[row, col])
            end
            scale = max(scale, s)
        end
        max(scale, BigFloat(1; precision=p))
    end
end

# Explicit caller-owned signed static shift into a FRESH copy: K0 is never
# modified.  `copy` + `setindex!` replacement never mutates shared limbs.
function _shifted_operator(K0::Matrix{BigFloat}, p::Int, nr::Int)
    return setprecision(BigFloat, p) do
        scale = _operator_scale(K0, p)
        delta = sqrt(eps(BigFloat)) * scale
        Kd = copy(K0)
        for j in 1:nr
            Kd[j, j] = Kd[j, j] + delta
        end
        for j in (nr + 1):size(K0, 1)
            Kd[j, j] = Kd[j, j] - delta
        end
        (Kd=Kd, dx=delta, dy=delta, scale=scale)
    end
end

# Upper-triangle CSC of the SHIFTED operator only.  Raw K0 (structural-zero
# x-diagonals) is never converted to a factor input.
function _upper_csc(Kd::Matrix{BigFloat}, p::Int)
    n = size(Kd, 1)
    return setprecision(BigFloat, p) do
        rows = Int[]
        cols = Int[]
        vals = BigFloat[]
        for col in 1:n, row in 1:col
            if !iszero(Kd[row, col])
                push!(rows, row)
                push!(cols, col)
                push!(vals, Kd[row, col])
            end
        end
        sparse(rows, cols, vals, n, n)
    end
end

function _dense_of_upper(U::SparseMatrixCSC{BigFloat,Int}, p::Int)
    n = size(U, 1)
    return setprecision(BigFloat, p) do
        K = zeros(BigFloat, n, n)
        for col in axes(U, 2), ptr in nzrange(U, col)
            row = U.rowval[ptr]
            K[row, col] = U.nzval[ptr]
            K[col, row] = U.nzval[ptr]
        end
        K
    end
end

const _ADAPTER_BEFORE = _adapter_manifest()
println("=== SDPX BFLA SPARSE ADAPTER MANIFEST ===")
for (k, v) in sort(collect(_ADAPTER_BEFORE); by=first)
    println(k, " = ", v)
end
println("=== END MANIFEST ===")
flush(stdout)

@testset "SDPX BFLA sparse signed-LDL internal adapter (R3 bounded)" begin
    @test Threads.nthreads() == 1
    # BigFloat sparse_augmented Settings stay disabled; no route default moved.
    @test_throws ArgumentError SDPX.Settings(BigFloat; kkt_route=:sparse_augmented)

    for p in (256, 512, 1024)
        @testset "precision=$p" begin
            setprecision(BigFloat, p) do
                core = _genuine_core(p)
                K0, b, x_true, nr, m, n = core.K0, core.b, core.x_true, core.nr, core.m, core.n
                @test n <= 32
                K0_snapshot = copy(K0)
                tol = BigFloat(2; precision=p)^(32 - p)
                dsigns = vcat(fill(1, nr), fill(-1, m))

                # Caller-owned shift; original stays separate.
                sh = _shifted_operator(K0, p, nr)
                Kd, dx, dy = sh.Kd, sh.dx, sh.dy
                @test K0 == K0_snapshot
                @test Kd != K0
                U = _upper_csc(Kd, p)

                # --- construction gates (before any factor) ---
                @test SDPX.SparseQDLDLProviderAvailable(BigFloat) === true
                cache = SDPX.SparseQDLDLCache{BigFloat}(U, dsigns; nrhs=1)
                @test SDPX.factor_status(cache) === SDPX.Prepared
                @test SDPX.factor_epoch(cache) == 0
                # Frozen construction precision + the one real symbolic
                # build performed by provider construction (P1/P2).
                @test cache.precision_bits == p
                @test cache.symbolic_count == 1
                @test BFLA.factor_diagnostics(cache.provider.inner).symbolic_count == 1
                # Rejections: sign length/values, empty column, lower triangle,
                # non-finite values.
                @test_throws DimensionMismatch SDPX.SparseQDLDLCache{BigFloat}(
                    U, vcat(dsigns, Int[1]); nrhs=1)
                @test_throws ArgumentError SDPX.SparseQDLDLCache{BigFloat}(
                    U, zeros(Int, n); nrhs=1)
                bad_colptr = copy(U.colptr)
                bad_colptr[2] = bad_colptr[1]
                bad_empty = SparseMatrixCSC(n, n, bad_colptr, copy(U.rowval), copy(U.nzval))
                @test_throws ArgumentError SDPX.SparseQDLDLCache{BigFloat}(
                    bad_empty, dsigns; nrhs=1)
                Klow = zeros(BigFloat, n, n)
                for col in 1:n, row in col:n
                    Klow[row, col] = Kd[row, col]
                end
                @test_throws ArgumentError SDPX.SparseQDLDLCache{BigFloat}(
                    sparse(Klow), dsigns; nrhs=1)
                bad_nf = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval), copy(U.nzval))
                bad_nf.nzval[1] = BigFloat(NaN; precision=p)
                @test_throws ArgumentError SDPX.SparseQDLDLCache{BigFloat}(
                    bad_nf, dsigns; nrhs=1)
                # Construction copies: mutating the caller input afterwards
                # changes nothing about the frozen authority.
                U.nzval[1] += BigFloat(1; precision=p)
                @test cache.colptr == U.colptr && cache.rowval == U.rowval
                U.nzval[1] -= BigFloat(1; precision=p)

                # --- factorize / vector + multi-RHS solves ---
                req = SDPX.FactorRequirements(n, 1)
                SDPX.prepare!(cache, req)
                @test SDPX.factor_status(cache) === SDPX.Prepared
                SDPX.factorize!(cache, U, 1)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                @test SDPX.factor_epoch(cache) == 1
                @test SDPX.factor_matrix_epoch(cache) == 1
                # Ordinary solves use the provider CHECKED solve (slot
                # repairing), so arbitrary caller-owned destinations are
                # safe; owned buffers are used here for the tight bounds.
                x = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                SDPX.solve!(cache, x, b)
                # Single-solve bound is sqrt(tol): one QDLDL triangular
                # pass is not backward stable to eps level on the KKT
                # operator.  The harness tolerance tol is met after
                # original-residual correction below.
                @test norm(Kd * x - b, Inf) <= sqrt(tol)
                @test cache.solve_count == 1
                # Repeated RHS + multi-RHS (width 2 and 3) through the cache.
                b1 = BFLA.owned_copy(b; precision_bits=p)
                b1[1] += BigFloat("0.5"; precision=p)
                x1 = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                SDPX.solve!(cache, x1, b1)
                @test norm(Kd * x1 - b1, Inf) <= sqrt(tol)
                B = BFLA.owned_zeros(BigFloat, n, 3; precision_bits=p)
                BFLA.copy_owned!(view(B, :, 1), b)
                BFLA.copy_owned!(view(B, :, 2), b1)
                for i in eachindex(b)
                    B[i, 3] = b[i] + b1[i]
                end
                X = BFLA.owned_zeros(BigFloat, n, 3; precision_bits=p)
                SDPX.solve_multi!(cache, X, B)
                @test norm(Kd * X - B, Inf) <= sqrt(tol)
                @test cache.solve_count == 2 + 3

                # --- arbitrary destination ownership (P1): shared-slot ---
                # `fill` destinations are repaired in place by the checked
                # solve; residual authority is the factored operator.
                xf = fill(BigFloat(0; precision=p), n)
                @test all(v -> v === xf[1], xf)  # shared slot, by construction
                SDPX.solve!(cache, xf, b)
                @test norm(Kd * xf - b, Inf) <= sqrt(tol)
                Xf = fill(BigFloat(0; precision=p), n, 2)
                Bf = BFLA.owned_zeros(BigFloat, n, 2; precision_bits=p)
                BFLA.copy_owned!(view(Bf, :, 1), b)
                BFLA.copy_owned!(view(Bf, :, 2), b1)
                SDPX.solve_multi!(cache, Xf, Bf)
                @test norm(Kd * Xf - Bf, Inf) <= sqrt(tol)
                # Shared-slot RHS is read-only source: safe as well.
                SDPX.solve!(cache, xf, fill(b[1], n))
                @test norm(Kd * xf - fill(b[1], n), Inf) <= sqrt(tol)
                # Source/sibling aliasing rejects WITHOUT revoking the
                # valid factor: the rejection is a caller-buffer bug, not
                # a factor failure.
                alias_buf = BFLA.owned_copy(b; precision_bits=p)
                @test_throws ArgumentError SDPX.solve!(cache, alias_buf, alias_buf)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                @test_throws ArgumentError SDPX.solve_multi!(cache, Bf, Bf)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                # The factor still solves after alias rejections.
                SDPX.solve!(cache, xf, b)
                @test norm(Kd * xf - b, Inf) <= sqrt(tol)

                # --- same-epoch unchanged-operator promise ---
                numeric_before = cache.numeric_count
                SDPX.factorize!(cache, U, 1)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                @test cache.numeric_count == numeric_before
                x_again = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                SDPX.solve!(cache, x_again, b)
                @test x_again == x
                # Same epoch but drifted values still skips numeric work BY
                # PROMISE (caller must bump the epoch on value change).
                U_changed = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval), copy(U.nzval))
                U_changed.nzval[2] += BigFloat("0.25"; precision=p)
                SDPX.factorize!(cache, U_changed, 1)
                @test cache.numeric_count == numeric_before
                x_promise = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                SDPX.solve!(cache, x_promise, b)
                @test x_promise == x  # previous operator authority, by promise
                # ... while a new epoch refactors and tracks the new operator.
                SDPX.factorize!(cache, U_changed, 2)
                @test cache.numeric_count == numeric_before + 1
                @test SDPX.factor_epoch(cache) == 2
                K_changed = _dense_of_upper(U_changed, p)
                x_new = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                SDPX.solve!(cache, x_new, b)
                @test norm(K_changed * x_new - b, Inf) <= sqrt(tol)
                @test x_new != x

                # --- ORIGINAL-vs-shifted residual authority ---
                # Bounded correction driven by the Kd factor, judged on the
                # ORIGINAL K0 residual only (mirrors the signed reference).
                SDPX.factorize!(cache, U, 3)
                xw = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                SDPX.solve!(cache, xw, b)
                orig_res = norm(K0 * xw - b, Inf)
                iters = 0
                while orig_res > tol && iters < 12
                    r = b - K0 * xw
                    d = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                    SDPX.refine_once!(cache, r, d)
                    xw .+= d
                    orig_res = norm(K0 * xw - b, Inf)
                    iters += 1
                end
                println("RECEIPT adapter p=$p scale=$(sh.scale) dx=$dx dy=$dy " *
                    "iters=$iters orig_res=$orig_res fwd_err=$(norm(xw - x_true, Inf)) " *
                    "factor_epoch=$(SDPX.factor_epoch(cache)) numeric=$(cache.numeric_count) " *
                    "symbolic=$(cache.symbolic_count) solves=$(cache.solve_count) " *
                    "refines=$(cache.refine_count) frozen_bits=$(cache.precision_bits)")
                @test iters <= 12
                @test orig_res <= tol
                @test norm(xw - x_true, Inf) <= sqrt(tol)
                @test cache.refine_count >= 1
                @test K0 == K0_snapshot  # original operator never modified
                # Non-finite / wrong-size refinement inputs reject.
                bad_r = b - K0 * xw
                bad_r[1] = BigFloat(NaN; precision=p)
                @test_throws ArgumentError SDPX.refine_once!(
                    cache, bad_r, BFLA.owned_zeros(BigFloat, n; precision_bits=p))
                @test_throws DimensionMismatch SDPX.refine_once!(
                    cache, BFLA.owned_zeros(BigFloat, n; precision_bits=p),
                    BFLA.owned_zeros(BigFloat, n + 1; precision_bits=p))

                # --- failure revocation: no stale solves ---
                Z = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval), zeros(BigFloat, length(U.nzval)))
                epoch_before = SDPX.factor_epoch(cache)
                numeric_before = cache.numeric_count
                @test_throws Exception SDPX.factorize!(cache, Z, 4)
                @test SDPX.factor_status(cache) === SDPX.Failed
                @test SDPX.factor_epoch(cache) == epoch_before
                @test cache.numeric_count == numeric_before
                @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                    cache, BFLA.owned_zeros(BigFloat, n; precision_bits=p), b)
                @test_throws SDPX.FactorCacheStateError SDPX.solve_multi!(
                    cache, BFLA.owned_zeros(BigFloat, n, 1; precision_bits=p),
                    reshape(BFLA.owned_copy(b; precision_bits=p), n, 1))
                # Recovery by a fresh factorize! at a new epoch.
                SDPX.factorize!(cache, U, 5)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                xr = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                SDPX.solve!(cache, xr, b)
                @test norm(Kd * xr - b, Inf) <= sqrt(tol)

                # --- shape/pattern/nonfinite preflight revokes on entry ---
                wrong = zeros(BigFloat, n + 1, n + 1)
                @test_throws DimensionMismatch SDPX.factorize!(
                    cache, sparse(UpperTriangular(wrong)), 6)
                @test SDPX.factor_status(cache) === SDPX.Failed
                SDPX.factorize!(cache, U, 7)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                # Genuine pattern change: move the single column-1 entry
                # (the x-diagonal) from row 1 to row 2.
                @test U.rowval[1] == 1
                drift = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval), copy(U.nzval))
                drift.rowval[1] = 2
                @test drift.rowval != U.rowval
                @test_throws ArgumentError SDPX.factorize!(cache, drift, 8)
                @test SDPX.factor_status(cache) === SDPX.Failed
                SDPX.factorize!(cache, U, 9)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                bad_vals = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval), copy(U.nzval))
                bad_vals.nzval[3] = BigFloat(Inf; precision=p)
                @test_throws ArgumentError SDPX.factorize!(cache, bad_vals, 10)
                @test SDPX.factor_status(cache) === SDPX.Failed
                SDPX.factorize!(cache, U, 11)
                @test SDPX.factor_status(cache) === SDPX.Fresh

                # --- wrong-typed factor inputs revoke via fallback (P1) ---
                # Wrong element type: Float64 storage into a BigFloat cache.
                U64elem = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval),
                    Float64.(U.nzval))
                @test_throws ArgumentError SDPX.factorize!(cache, U64elem, 20)
                @test SDPX.factor_status(cache) === SDPX.Failed
                @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                    cache, BFLA.owned_zeros(BigFloat, n; precision_bits=p), b)
                SDPX.factorize!(cache, U, 21)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                # Wrong index type: Int32-indexed BigFloat storage.
                U32 = SparseMatrixCSC{BigFloat,Int32}(n, n,
                    Int32.(U.colptr), Int32.(U.rowval), copy(U.nzval))
                @test_throws ArgumentError SDPX.factorize!(cache, U32, 22)
                @test SDPX.factor_status(cache) === SDPX.Failed
                @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                    cache, BFLA.owned_zeros(BigFloat, n; precision_bits=p), b)
                SDPX.factorize!(cache, U, 23)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                # Wrong storage: dense matrix with identical values.
                @test_throws ArgumentError SDPX.factorize!(
                    cache, Matrix(_dense_of_upper(U, p)), 24)
                @test SDPX.factor_status(cache) === SDPX.Failed
                @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                    cache, BFLA.owned_zeros(BigFloat, n; precision_bits=p), b)
                SDPX.factorize!(cache, U, 25)
                @test SDPX.factor_status(cache) === SDPX.Fresh

                # --- invalidation revokes; diagnostics report lifecycle ---
                SDPX.invalidate!(cache)
                @test SDPX.factor_status(cache) === SDPX.Invalid
                @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                    cache, BFLA.owned_zeros(BigFloat, n; precision_bits=p), b)
                SDPX.factorize!(cache, U, 12)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                diag = SDPX.factor_diagnostics(cache)
                @test diag.n == n
                @test diag.matrix_epoch == 12
                @test diag.status === SDPX.Fresh
                @test diag.numeric_count >= 1
                @test diag.solve_count >= 1
                @test diag.symbolic_count == 1  # one real symbolic build
                @test diag.precision_bits == p  # frozen at construction
                @test diag.refine_count == cache.refine_count
            end
        end
    end
end

@testset "adapter precision-drift control" begin
    p = 256
    setprecision(BigFloat, p) do
        core = _genuine_core(p)
        K0, b, nr, m, n = core.K0, core.b, core.nr, core.m, core.n
        sh = _shifted_operator(K0, p, nr)
        U = _upper_csc(sh.Kd, p)
        dsigns = vcat(fill(1, nr), fill(-1, m))
        cache = SDPX.SparseQDLDLCache{BigFloat}(U, dsigns; nrhs=1)
        @test cache.precision_bits == p
        SDPX.prepare!(cache, SDPX.FactorRequirements(n, 0))
        SDPX.factorize!(cache, U, 1)
        @test SDPX.factor_status(cache) === SDPX.Fresh
        # P1 regression: 256-bit construction, then ambient 64 with 64-bit
        # SAME-PATTERN input at the SAME epoch must NOT reuse the stale
        # 256-bit factor — frozen precision rejects before same-epoch reuse.
        setprecision(BigFloat, 64) do
            U64 = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval),
                BigFloat.(U.nzval; precision=64))
            @test_throws Exception SDPX.factorize!(cache, U64, 1)
            @test SDPX.factor_status(cache) === SDPX.Failed
            @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                cache, BFLA.owned_zeros(BigFloat, n; precision_bits=64),
                BFLA.owned_copy(b; precision_bits=64))
        end
        # A different ambient precision cannot factor at a new epoch either.
        setprecision(BigFloat, 64) do
            U64 = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval),
                BigFloat.(U.nzval; precision=64))
            @test_throws Exception SDPX.factorize!(cache, U64, 2)
            @test SDPX.factor_status(cache) === SDPX.Failed
        end
        # Restored precision recovers with a fresh factorize!.
        SDPX.factorize!(cache, U, 3)
        @test SDPX.factor_status(cache) === SDPX.Fresh
        x = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        SDPX.solve!(cache, x, b)
        single_tol = sqrt(BigFloat(2; precision=p)^(32 - p))
        @test norm(_dense_of_upper(U, p) * x - b, Inf) <= single_tol
    end
end

@testset "adapter source binding (P2)" begin
    # The manifest binds the script checkout to the LOADED SDPX source.
    @test realpath(joinpath(@__DIR__, "..", "..")) == realpath(pkgdir(SDPX))
    # Wrong-source negative test: a different checkout must be rejected.
    @test_throws ErrorException _assert_sdpx_source(mktempdir(), pkgdir(SDPX))
    @test _assert_sdpx_source(joinpath(@__DIR__, "..", ".."), pkgdir(SDPX)) ==
        realpath(pkgdir(SDPX))
end

@assert _adapter_manifest() == _ADAPTER_BEFORE "adapter source/environment changed"
println("ADAPTER_SOURCE_UNCHANGED")
