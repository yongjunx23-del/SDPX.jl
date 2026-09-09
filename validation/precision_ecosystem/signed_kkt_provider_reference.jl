# Signed-KKT (HSD core) rank-corrected provider reference, validation ONLY.
#
# Repair of c016b82 following the rank/attribution review. All original-K
# fixtures of the previous revision are singular (rank(K) = m + rank(A) with
# nr > m), so they are retained here as correctly named singular
# compatible/incompatible controls with exact certificates, and GENUINE
# nonsingular fixtures (nr <= m, structural triangular rank certificate) are
# added. No native-route wiring, no rank deletion, no precision downgrade.
#
# Rank theorem used (comment, not code): for K = [0 A'; A -Theta] with
# diagonal Theta > 0, K[z;w] = 0 gives w = Theta^{-1}Az and then
# A'w = 0, i.e. ||Theta^{-1/2}Az||^2 = 0, so Az = 0 and w = 0. Hence
# null(K) = {(z,0) : Az = 0} and rank(K) = m + rank(A). Shifted inertia can
# never establish original rank; rank/compatibility here are certified by
# structural zero patterns plus EXACT Rational{BigInt} arithmetic, never by
# Float64 rank guesses.
#
# Shift policy labeling (corrected): the tested shift
# delta = sqrt(eps(BigFloat)) * scale is BORROWED from the EXPANDED route
# (src/kkt/expanded_quasidefinite.jl:952, static rung k = 0 of :966-977) and
# paired with a HARNESS-SPECIFIC absolute acceptance 2^(32-p) and at most 12
# same-precision corrections. That is a different reference algorithm from the
# actual symmetric-core policy, which is documented (not equated) here:
# symmetric_core.jl is Float64-only for the shifted path with
# delta = 64*eps*current_scale (:1169-1187, unregularized path otherwise),
# normalized residual eta = ||r||/(K_scale*||x||+||rhs||) (:1397-1440),
# normalized refinement floor 256*eps(one(T)) in `_core_refine!` (:1467),
# at most TWO corrections with strict contraction (:1443+), then a production
# five-equation gate. Both policy files are hashed in the manifest; no
# substring assertion is presented as proof of anything.
#
# Bounds: n <= 32 (this file uses n = 14 and n = 18), single Julia/GC/BLAS
# thread, `--heap-size-hint=2G`, each Julia command documented with exit code.
# Run with the fresh private env (BFLA dev'ed at aaa71f3, QDLDL pinned 0.4.1):
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
#   julia --startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G \
#     --project=/private/tmp/qdldl-bfla-qual-env \
#     validation/precision_ecosystem/signed_kkt_provider_reference.jl

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
const QDLDL_PKGDIR = realpath(pkgdir(QDLDL))
const QDLDL_SRC = realpath(pathof(QDLDL))
const SDPX_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SDPX_EXPANDED_POLICY = joinpath(SDPX_ROOT, "src/kkt/expanded_quasidefinite.jl")
const SDPX_SYMMETRIC_POLICY = joinpath(SDPX_ROOT, "src/kkt/symmetric_core.jl")
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
    realpath(pathof(BFLA)) == realpath(joinpath(BFLA_WORKTREE, "src/BigFloatLinearAlgebra.jl")) ||
        error("BFLA entrypoint/root mismatch")
    QDLDL_SRC == realpath(joinpath(QDLDL_PKGDIR, "src/QDLDL.jl")) ||
        error("QDLDL entrypoint/root mismatch")
    head = readchomp(`git -C $BFLA_WORKTREE rev-parse HEAD`)
    head == EXPECTED_BFLA_HEAD || error("unexpected BFLA source revision")
    isempty(readchomp(`git -C $BFLA_WORKTREE status --porcelain`)) ||
        error("BFLA source is dirty")
    _sha256_file(QDLDL_SRC) == EXPECTED_QDLDL_SHA ||
        error("unexpected QDLDL source contents")
    Base.get_extension(BFLA, :BigFloatQDLDLExt) !== nothing || error("extension absent")
    BFLA.sparse_ldlt_available(BigFloat) || error("sparse provider unavailable")
    hashes = Dict{String,String}()
    for (name, root) in (("BFLA", BFLA_WORKTREE), ("QDLDL", QDLDL_PKGDIR))
        hashes[name * "/Project.toml"] = _sha256_file(joinpath(root, "Project.toml"))
        for sub in ("src", "ext")
            isdir(joinpath(root, sub)) || continue
            for (dir, _, files) in walkdir(joinpath(root, sub)), file in files
                path = joinpath(dir, file)
                hashes[name * "/" * relpath(path, root)] = _sha256_file(path)
            end
        end
    end
    # Both SDPX policy sources are hashed and documented; neither hash nor any
    # substring check is presented as equivalence with this harness's borrowed
    # expanded-rung/harness-tolerance reference algorithm.
    hashes["SDPX/src/kkt/expanded_quasidefinite.jl"] = _sha256_file(SDPX_EXPANDED_POLICY)
    hashes["SDPX/src/kkt/symmetric_core.jl"] = _sha256_file(SDPX_SYMMETRIC_POLICY)
    hashes["harness"] = _sha256_file(@__FILE__)
    project = Base.active_project()
    project === nothing && error("explicit qualification project required")
    return Dict(
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "ambient_precision_bits" => precision(BigFloat),
        "bfla_root" => BFLA_WORKTREE,
        "bfla_head" => head,
        "qdldl_root" => QDLDL_PKGDIR,
        "qdldl_entrypoint" => QDLDL_SRC,
        "sdpx_expanded_policy" => SDPX_EXPANDED_POLICY,
        "sdpx_symmetric_policy" => SDPX_SYMMETRIC_POLICY,
        "project" => project,
        "project_sha256" => _sha256_file(project),
        "manifest_sha256" => _sha256_file(joinpath(dirname(project), "Manifest.toml")),
        "harness_head" => readchomp(`git -C $(@__DIR__) rev-parse HEAD`),
        "harness_status" => readchomp(`git -C $(@__DIR__) status --porcelain`),
        "sha256" => hashes,
    )
end

function _print_manifest(tag::AbstractString)
    data = _qualification_manifest()
    println("=== QUAL MANIFEST [$tag] ===")
    TOML.print(stdout, data; sorted=true)
    println("\n=== END QUAL MANIFEST [$tag] ===")
    flush(stdout)
    return data
end

# --- ownership primitives -----------------------------------------------
# Base `BigFloat(x; precision)` ALIASES x when precisions match (verified:
# `===`), and `copy(::BigFloat)` is shallow, so neither is an ownership
# primitive. Fresh scalars below go through BFLA's deep value copy.
function _fresh_scalar(v::BigFloat, p::Int)
    return only(BFLA.owned_copy([v]; precision_bits=p))
end

# True in-place MPFR mutation of one limb (verified via ccall against libmpfr).
function _mpfr_add_inplace!(z::BigFloat, x::BigFloat)
    ccall((:mpfr_add, Base.MPFR.libmpfr), Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Int32),
        z, z, x, Int32(0))
    return z
end

# --- exact-rational certificates -----------------------------------------
# All conversions below are bit-exact (Rational{BigInt} of a BigFloat is the
# exact dyadic value), so zero/nonzero verdicts are theorems, not estimates.
function _exact_rank(R::Matrix{Rational{BigInt}})
    M = copy(R)
    m, n = size(M)
    r = 0
    for c in 1:n
        # NOTE: findfirst over a UnitRange returns the 1-based POSITION, not
        # the row number; an explicit loop keeps the pivot row unambiguous.
        piv = nothing
        for i in (r + 1):m
            if M[i, c] != 0
                piv = i
                break
            end
        end
        piv === nothing && continue
        r += 1
        if piv != r
            tmp = copy(M[r, :])
            M[r, :] = M[piv, :]
            M[piv, :] = tmp
        end
        for i in 1:m
            (i == r || M[i, c] == 0) && continue
            f = M[i, c] // M[r, c]
            for j in c:n
                M[i, j] = M[i, j] - f * M[r, j]
            end
        end
    end
    return r
end

# Exact reduced-row-echelon null basis (all arithmetic in QQ). Returns
# (pivot_columns, basis). Empty basis iff full column rank.
function _exact_null_basis(R::Matrix{Rational{BigInt}})
    M = copy(R)
    m, n = size(M)
    pivots = Int[]
    row = 0
    for c in 1:n
        piv = nothing
        for i in (row + 1):m
            if M[i, c] != 0
                piv = i
                break
            end
        end
        piv === nothing && continue
        row += 1
        if piv != row
            tmp = copy(M[row, :])
            M[row, :] = M[piv, :]
            M[piv, :] = tmp
        end
        for i in 1:m
            (i == row || M[i, c] == 0) && continue
            f = M[i, c] // M[row, c]
            for j in c:n
                M[i, j] = M[i, j] - f * M[row, j]
            end
        end
        push!(pivots, c)
    end
    basis = Vector{Vector{Rational{BigInt}}}()
    for f in 1:n
        f in pivots && continue
        z = zeros(Rational{BigInt}, n)
        z[f] = 1
        for (ri, pc) in enumerate(pivots)
            z[pc] = -M[ri, f] // M[ri, pc]
        end
        push!(basis, z)
    end
    return pivots, basis
end

# --- deterministic ORIGINAL HSD cores K = [0 A'; A -Theta] ----------------
# Genuine nonsingular cases use nr <= m with an explicit triangular minor:
# rows 1..nr are lower-triangular with nonzero diagonal (structural rank
# certificate), remaining rows arbitrary. Singular controls reuse the
# previous-revision formulas (all nr > m) with exact null vectors verified
# at runtime.
function _triangular_A(p::Int, nr::Int, m::Int, row_exp::Function, diag_exp::Function)
    return setprecision(BigFloat, p) do
        A = BFLA.owned_zeros(BigFloat, m, nr; precision_bits=p)
        for i in 1:min(nr, m), j in 1:min(nr, m)
            if j > i
                @assert iszero(A[i, j])
            elseif j == i
                d = BigFloat(1; precision=p) +
                    BigFloat("0.125"; precision=p) * BigFloat((3 * i) % 4; precision=p)
                A[i, j] = d * BigFloat(10; precision=p)^diag_exp(i)
            else
                v = BigFloat(((-1)^(i + j)) * (0.5 + 0.0625 * ((3 * i + 5 * j) % 9)); precision=p)
                A[i, j] = v * BigFloat(10; precision=p)^row_exp(i)
            end
        end
        for i in (min(nr, m) + 1):m, j in 1:nr
            v = BigFloat(((-1)^(i * j + i)) * (0.5 + 0.0625 * ((3 * i + 5 * j) % 9)); precision=p)
            A[i, j] = v * BigFloat(10; precision=p)^row_exp(i)
        end
        A
    end
end

function _assemble_K0(A::Matrix{BigFloat}, Theta::Vector{BigFloat}, p::Int)
    # Every K0 entry is a FRESH scalar object (owned_copy deep step); K0
    # shares no limb with A/Theta. Plain setindex! would alias.
    return setprecision(BigFloat, p) do
        m, nr = size(A)
        n = nr + m
        K0 = BFLA.owned_zeros(BigFloat, n, n; precision_bits=p)
        for i in 1:m, j in 1:nr
            K0[j, nr + i] = _fresh_scalar(A[i, j], p)
            K0[nr + i, j] = _fresh_scalar(A[i, j], p)
        end
        for j in 1:m
            K0[nr + j, nr + j] = _fresh_scalar(-Theta[j], p)
        end
        for j in 1:nr
            @assert iszero(K0[j, j])
        end
        K0
    end
end

function _explicit_xtrue(p::Int, n::Int)
    return setprecision(BigFloat, p) do
        x = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        for i in 1:n
            x[i] = BigFloat(((-1)^i) * (1.0 + 0.125 * ((5 * i) % 7)); precision=p)
        end
        x
    end
end

function _genuine_core(p::Int, illscaled::Bool)
    # nr <= m with triangular minor: A (8x6) full column rank by structure.
    nr, m = 6, 8
    n = nr + m
    @assert n <= 32
    return setprecision(BigFloat, p) do
        if illscaled
            # Varying exponents (guarded in-test against the old constant bug).
            A = _triangular_A(p, nr, m, i -> ((i * 3) % 5) - 2, i -> ((i * 7) % 5) - 2)
            Theta = BFLA.owned_zeros(BigFloat, m; precision_bits=p)
            for j in 1:m
                Theta[j] = BigFloat(10; precision=p)^(((3 * j) % 7) - 3)
            end
        else
            A = _triangular_A(p, nr, m, i -> 0, i -> 0)
            Theta = BFLA.owned_zeros(BigFloat, m; precision_bits=p)
            for j in 1:m
                Theta[j] = BigFloat(1; precision=p) +
                           BigFloat("0.125"; precision=p) * BigFloat((7 * j) % 5; precision=p)
            end
        end
        K0 = _assemble_K0(A, Theta, p)
        x_true = _explicit_xtrue(p, n)
        b = K0 * x_true
        (K0=K0, A=A, Theta=Theta, b=b, x_true=x_true, nr=nr, m=m)
    end
end

function _singular_compatible(p::Int)
    # Previous "fullrank" fixture, honestly named: A is 6x8 of rank 6, so
    # rank(K) = 12 < 14. Exact null vectors (x-part) verified at runtime:
    # z1 = e3-2e5+e7, z2 = e2-e4-e6+e8, both with z'b_x = 0 (compatible).
    nr, m = 8, 6
    return setprecision(BigFloat, p) do
        A = BFLA.owned_zeros(BigFloat, m, nr; precision_bits=p)
        for i in 1:m, j in 1:nr
            A[i, j] = BigFloat(((-1)^(i * j + i)) * (0.5 + 0.0625 * ((3 * i + 5 * j) % 9)); precision=p)
        end
        Theta = BFLA.owned_zeros(BigFloat, m; precision_bits=p)
        for j in 1:m
            Theta[j] = BigFloat(1; precision=p) +
                       BigFloat("0.125"; precision=p) * BigFloat((7 * j) % 5; precision=p)
        end
        K0 = _assemble_K0(A, Theta, p)
        n = nr + m
        b = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        for i in 1:n
            b[i] = BigFloat(((-1)^i) * (1.0 + 0.125 * i); precision=p)
        end
        z1 = [0, 0, 1, 0, -2, 0, 1, 0]
        z2 = [0, 1, 0, -1, 0, -1, 0, 1]
        (K0=K0, A=A, Theta=Theta, b=b, x_true=nothing, nr=nr, m=m, nulls=(z1, z2), zdot=(0 // 1, 0 // 1))
    end
end

function _singular_incompatible(p::Int)
    # Previous "singular" fixture: duplicated A rows give rank(A) <= 4, so
    # rank(K) <= 10 < 14. The exact nullspace/RHS certificate is computed
    # below. One valid witness is (-1,-39,-7,36,8,0,0,0): Aw=0,
    # w'b_x=3 and ||w||_1=91, giving an original residual floor of 3/91.
    nr, m = 8, 6
    return setprecision(BigFloat, p) do
        A = BFLA.owned_zeros(BigFloat, m, nr; precision_bits=p)
        for i in 1:4, j in 1:nr
            A[i, j] = BigFloat(((-1)^(i * j + i)) * (0.5 + 0.0625 * ((3 * i + 5 * j) % 9)); precision=p)
        end
        for j in 1:nr
            A[5, j] = _fresh_scalar(A[1, j], p)
            A[6, j] = _fresh_scalar(A[1, j], p)
        end
        Theta = BFLA.owned_zeros(BigFloat, m; precision_bits=p)
        for j in 1:m
            Theta[j] = BigFloat(1; precision=p)
        end
        K0 = _assemble_K0(A, Theta, p)
        n = nr + m
        b = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        for i in 1:n
            b[i] = BigFloat(((-1)^i) * (1.0 + 0.125 * i); precision=p)
        end
        # NOTE: no hardcoded null vector is trusted here (a previously
        # claimed vector fails row 2 exactly); the null basis is COMPUTED
        # in-test via exact rational RREF below.
        (K0=K0, A=A, Theta=Theta, b=b, x_true=nothing, nr=nr, m=m, nulls=(), zdot=())
    end
end

function _singular_incompatible_illscaled(p::Int)
    # Previous "illscaled" fixture, honestly named: A[:,10] = -A[:,1] exactly
    # (same row factor, same formula value, same sign), so z = e1+e10 has
    # Az = 0 while z'b_x = 9/8: every candidate has ||r||_inf >= 9/16.
    # The old constant-exponent bug ((i*5)%5 is identically 0) is corrected to
    # varying exponents; the null identity is row-local, hence unaffected.
    nr, m = 10, 8
    return setprecision(BigFloat, p) do
        exps = [((i * 7) % 5) - 2 for i in 1:m]
        @assert length(unique(exps)) > 1
        A = BFLA.owned_zeros(BigFloat, m, nr; precision_bits=p)
        for i in 1:m, j in 1:nr
            rs = BigFloat(10; precision=p)^exps[i]
            A[i, j] = rs * BigFloat(((-1)^(i + j)) * (0.5 + 0.0625 * ((3 * i + 7 * j) % 9)); precision=p)
        end
        Theta = BFLA.owned_zeros(BigFloat, m; precision_bits=p)
        for j in 1:m
            Theta[j] = BigFloat(10; precision=p)^(-6 + 2 * ((3 * j) % 7))
        end
        K0 = _assemble_K0(A, Theta, p)
        n = nr + m
        b = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        for i in 1:n
            b[i] = BigFloat(((-1)^i) * (1.0 + 0.125 * i); precision=p)
        end
        z = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        (K0=K0, A=A, Theta=Theta, b=b, x_true=nothing, nr=nr, m=m, nulls=(z,), zdot=(9 // 8,), floor=9 // 16)
    end
end

# Infinity-norm scale of the ORIGINAL operator (borrowed expanded-route
# convention: max row sum, floor 1).
function _operator_scale(K0::AbstractMatrix{BigFloat}, p::Int)
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

# Borrowed EXPANDED rung k = 0: uniform delta = sqrt(eps) * scale.
function _stated_shift(K0::AbstractMatrix{BigFloat}, p::Int)
    return setprecision(BigFloat, p) do
        scale = _operator_scale(K0, p)
        delta = sqrt(eps(BigFloat)) * scale
        (dx=delta, dy=delta, scale=scale)
    end
end

# Experimental balance-motivated alternative (NOT a default change): per-block
# shifts tied to each block's own max row sum.
function _balanced_shift(K0::AbstractMatrix{BigFloat}, p::Int, nr::Int)
    return setprecision(BigFloat, p) do
        sx = BigFloat(0; precision=p)
        for row in 1:nr
            s = BigFloat(0; precision=p)
            for col in axes(K0, 2)
                s += abs(K0[row, col])
            end
            sx = max(sx, s)
        end
        sy = BigFloat(0; precision=p)
        for row in (nr + 1):size(K0, 1)
            s = BigFloat(0; precision=p)
            for col in axes(K0, 2)
                s += abs(K0[row, col])
            end
            sy = max(sy, s)
        end
        sq = sqrt(eps(BigFloat))
        (dx=sq * max(sx, BigFloat(1; precision=p)),
         dy=sq * max(sy, BigFloat(1; precision=p)),
         scale=_operator_scale(K0, p))
    end
end

# Explicit signed-static-shift operator Kdelta from the ORIGINAL K0.
# K0 is never modified; the shift is assembled into a deep owned copy.
function _shifted_operator(K0::AbstractMatrix{BigFloat}, p::Int, nr::Int, dx::BigFloat, dy::BigFloat)
    return setprecision(BigFloat, p) do
        Kd = BFLA.owned_copy(K0; precision_bits=p)
        for j in 1:nr
            Kd[j, j] = Kd[j, j] + dx
        end
        for j in (nr + 1):size(K0, 1)
            Kd[j, j] = Kd[j, j] - dy
        end
        Kd
    end
end

# Upper-triangle CSC of the SHIFTED operator only. Raw K0 (structural-zero
# x-diagonals) is never converted to a factor input. Limb independence of U
# comes from the owned_copy deep step (elementwise _mpfr_set!), not from the
# BigFloat() constructor (which aliases on matching precision).
function _upper_csc(Kd::AbstractMatrix{BigFloat}, p::Int)
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
        U = sparse(rows, cols, vals, n, n)
        SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval),
            BFLA.owned_copy(U.nzval; precision_bits=p))
    end
end

# Bounded iterative correction using the Kdelta factor as preconditioner.
# Residuals are evaluated against the ORIGINAL K0 at working precision.
# Returns a receipt; convergence is judged on the ORIGINAL residual only.
function _refine_original!(cache, K0::AbstractMatrix{BigFloat},
        Kd::AbstractMatrix{BigFloat}, b::AbstractVector{BigFloat},
        p::Int, tol::BigFloat, x_true; maxit::Int=12)
    n = size(K0, 1)
    return setprecision(BigFloat, p) do
        x = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        BFLA.solve_trusted!(x, cache, b)
        d = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        r = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
        orig_res = norm(K0 * x - b, Inf)
        shifted_res = norm(Kd * x - b, Inf)
        orig_res_0 = orig_res
        shifted_res_0 = shifted_res
        iters = 0
        while orig_res > tol && iters < maxit
            BFLA.copy_owned!(r, b)
            mul!(r, K0, x, -one(BigFloat), one(BigFloat))
            @assert all(precision(v) == p for v in r)
            BFLA.solve_trusted!(d, cache, r)
            x .+= d
            @assert all(precision(v) == p for v in x)
            orig_res = norm(K0 * x - b, Inf)
            shifted_res = norm(Kd * x - b, Inf)
            iters += 1
        end
        fwd = x_true === nothing ? nothing : norm(x - x_true, Inf)
        (x=x, iters=iters, orig_res=orig_res, shifted_res=shifted_res,
         orig_res_0=orig_res_0, shifted_res_0=shifted_res_0,
         fwd_err=fwd, converged=orig_res <= tol)
    end
end

# Hölder bound for the STORED working-precision residual. This is a
# diagnostic identity, not an incompatibility certificate: that separately
# requires the exact residual K*x-b and its pairing with the original RHS.
# z covers only the x-block (its y-block is zero).
function _certified_floor(r::AbstractVector{BigFloat}, z::AbstractVector)
    Rr = Rational{BigInt}.(r)
    s = sum(Rr[i] * z[i] for i in eachindex(z))
    floor_lo = abs(s) // sum(abs, z)
    return s, floor_lo
end

const _QUALIFICATION_BEFORE = _print_manifest("before")

@testset "signed-KKT rank-corrected reference (validation only)" begin
    @test Threads.nthreads() == 1
    for p in (256, 512, 1024)
        # ---- genuine nonsingular cores: manufactured explicit RHS ----
        for (kind, illscaled) in (("genuine-fullrank", false), ("genuine-illscaled", true))
            @testset "case=$kind p=$p" begin
                setprecision(BigFloat, p) do
                    core = _genuine_core(p, illscaled)
                    K0, A, Theta, b, x_true, nr, m = core.K0, core.A, core.Theta, core.b, core.x_true, core.nr, core.m
                    n = nr + m
                    @test n <= 32
                    # Structural rank certificate: triangular minor with
                    # nonzero diagonal, positive Theta diagonal.
                    for i in 1:nr, j in (i + 1):nr
                        @test iszero(A[i, j])
                    end
                    for i in 1:nr
                        @test !iszero(A[i, i])
                    end
                    for j in 1:m
                        @test Theta[j] > 0
                    end
                    if illscaled
                        @test length(unique([((i * 3) % 5) - 2 for i in 1:m])) > 1
                        @test length(unique([((i * 7) % 5) - 2 for i in 1:nr])) > 1
                    end
                    # Exact-rational rank cross-check (theorem: rank(K)=m+rank(A)).
                    @test _exact_rank(Rational{BigInt}.(A)) == nr
                    @test _exact_rank(Rational{BigInt}.(K0)) == n
                    # Ownership: A/Theta/K0/b/x_true share no limb.
                    stores = (A, K0, reshape(Theta, length(Theta), 1), reshape(b, n, 1), reshape(x_true, n, 1))
                    for (ai, aj) in ((1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5))
                        @test all(u !== v for u in stores[ai] for v in stores[aj])
                    end
                    @test all(A[i, j] !== K0[i2, j2] for i in axes(A, 1), j in axes(A, 2),
                        i2 in axes(K0, 1), j2 in axes(K0, 2))
                    # True in-place MPFR mutation of one A limb leaves the
                    # retained K0 copies (values) unchanged, then is restored.
                    a_obj = A[1, 1]
                    k_before = (Rational{BigInt}(K0[1, nr + 1]), Rational{BigInt}(K0[nr + 1, 1]))
                    a_before = Rational{BigInt}(a_obj)
                    _mpfr_add_inplace!(a_obj, BigFloat(1; precision=p))
                    @test Rational{BigInt}(a_obj) == a_before + 1
                    @test Rational{BigInt}(K0[1, nr + 1]) == k_before[1]
                    @test Rational{BigInt}(K0[nr + 1, 1]) == k_before[2]
                    _mpfr_add_inplace!(a_obj, BigFloat(-1; precision=p))
                    @test Rational{BigInt}(a_obj) == a_before

                    tol = BigFloat(2; precision=p)^(32 - p)
                    dsigns = vcat(fill(1, nr), fill(-1, m))
                    for (policy, sh) in (
                            ("stated-borrowed-expanded-r0", _stated_shift(K0, p)),
                            ("balanced-experiment", _balanced_shift(K0, p, nr)),
                            )
                        @test sh.dx > 0 && sh.dy > 0
                        @test precision(sh.dx) == p && precision(sh.dy) == p
                        Kd = _shifted_operator(K0, p, nr, sh.dx, sh.dy)
                        @test K0 != Kd
                        U = _upper_csc(Kd, p)
                        cache = BFLA.sparse_ldlt_cache(U; precision_bits=p, dsigns=dsigns, nrhs=1)
                        factor0 = something(cache.factor)
                        @test factor0 isa QDLDL.QDLDLFactorisation{BigFloat}
                        @test factor0.workspace.Dsigns === nothing
                        @test iszero(factor0.workspace.regularize_eps)
                        @test iszero(factor0.workspace.regularize_delta)
                        BFLA.factorize!(cache, U)
                        @test BFLA.issuccess(cache)
                        diag = BFLA.factor_diagnostics(cache)
                        @test diag.provider === :qdldl
                        @test diag.symbolic_count == 1
                        @test diag.numeric_factor_count == 1
                        @test diag.positive_inertia == nr
                        @test diag.regularized_entries == 0
                        @test something(cache.factor).workspace.regularize_count[1] == 0
                        nnz_l_first = diag.nnz_l
                        rec = _refine_original!(cache, K0, Kd, b, p, tol, x_true)
                        println("RECEIPT case=$kind p=$p policy=$policy " *
                            "scale=$(sh.scale) dx=$(sh.dx) dy=$(sh.dy) " *
                            "iters=$(rec.iters) orig_res=$(rec.orig_res) " *
                            "shifted_res=$(rec.shifted_res) fwd_err=$(rec.fwd_err) " *
                            "nnz_k=$(diag.nnz_k) nnz_l=$(diag.nnz_l) " *
                            "symbolic=$(diag.symbolic_count) numeric=$(diag.numeric_factor_count)")
                        @test rec.converged
                        @test rec.iters <= 12
                        @test rec.orig_res <= tol
                        @test rec.fwd_err <= (illscaled ? sqrt(tol) : tol)
                        # Fixed-pattern refactor + matrix-shift receipt.
                        K02 = BFLA.owned_copy(K0; precision_bits=p)
                        # Sign-preserving multiplicative perturbation: nonzero
                        # stays nonzero, so the sparsity pattern is invariant
                        # by construction (triangular structural zeros untouched).
                        for i in 1:m, j in 1:nr
                            if !(i <= nr && j <= nr && j > i)
                                sgn = BigFloat(((-1)^(i + j)); precision=p)
                                K02[j, nr + i] = K02[j, nr + i] *
                                    (BigFloat(1; precision=p) + sgn * BigFloat("0.015625"; precision=p))
                                K02[nr + i, j] = K02[j, nr + i]
                            end
                        end
                        sh2 = policy == "stated-borrowed-expanded-r0" ? _stated_shift(K02, p) :
                            _balanced_shift(K02, p, nr)
                        Kd2 = _shifted_operator(K02, p, nr, sh2.dx, sh2.dy)
                        U2 = _upper_csc(Kd2, p)
                        @test U2.colptr == cache.frozen_colptr
                        @test U2.rowval == cache.frozen_rowval
                        BFLA.factorize!(cache, U2)
                        @test BFLA.issuccess(cache)
                        diag2 = BFLA.factor_diagnostics(cache)
                        @test diag2.symbolic_count == 1
                        @test diag2.numeric_factor_count == 2
                        @test diag2.nnz_l == nnz_l_first
                        @test diag2.positive_inertia == nr
                        rec2 = _refine_original!(cache, K02, Kd2, b, p, tol, nothing)
                        println("RECEIPT case=$kind-shifted-matrix p=$p policy=$policy " *
                            "scale=$(sh2.scale) dx=$(sh2.dx) dy=$(sh2.dy) " *
                            "iters=$(rec2.iters) orig_res=$(rec2.orig_res) " *
                            "shifted_res=$(rec2.shifted_res) nnz_l=$(diag2.nnz_l) " *
                            "symbolic=$(diag2.symbolic_count) numeric=$(diag2.numeric_factor_count)")
                        @test rec2.converged
                        @test rec2.orig_res <= tol
                    end
                end
            end
        end
        # ---- singular controls with exact certificates ----
        for (kind, builder) in (("singular-compatible", _singular_compatible),
                ("singular-incompatible", _singular_incompatible),
                ("singular-incompatible-illscaled", _singular_incompatible_illscaled))
            @testset "case=$kind p=$p" begin
                setprecision(BigFloat, p) do
                    core = builder(p)
                    K0, A, Theta, b, nr, m = core.K0, core.A, core.Theta, core.b, core.nr, core.m
                    n = nr + m
                    @test n <= 32
                    RA = Rational{BigInt}.(A)
                    RK = Rational{BigInt}.(K0)
                    rankA = _exact_rank(RA)
                    @test _exact_rank(RK) == m + rankA
                    @test rankA < nr
                    # Exact certificates, per-case shape (all in QQ, no rank guesses).
                    # cert === nothing  -> compatible (convergence expected).
                    # cert == (z, q)    -> proven incompatible with |z'b| = |q|.
                    cert = nothing
                    if kind == "singular-compatible"
                        @test length(core.nulls) == 2
                        for (z, q) in zip(core.nulls, core.zdot)
                            @test RA * z == zeros(Rational{BigInt}, m)
                            full_z = vcat(Rational{BigInt}.(z), zeros(Rational{BigInt}, m))
                            @test RK * full_z == zeros(Rational{BigInt}, n)
                            @test sum(Rational{BigInt}(b[i]) * z[i] for i in 1:nr) == q
                        end
                        # Nullity 2 plus two independent verified nulls whose
                        # RHS products both vanish: compatibility certified.
                        @test nr - rankA == 2
                        @test _exact_rank(hcat(Rational{BigInt}.(core.nulls[1]),
                            Rational{BigInt}.(core.nulls[2]))) == 2
                    elseif kind == "singular-incompatible-illscaled"
                        z = core.nulls[1]
                        q = core.zdot[1]
                        # Column antisymmetry is row-local (same row factor,
                        # same formula value, matching sign), hence exact.
                        @test RA[:, 10] + RA[:, 1] == zeros(Rational{BigInt}, m)
                        @test RA * z == zeros(Rational{BigInt}, m)
                        full_z = vcat(Rational{BigInt}.(z), zeros(Rational{BigInt}, m))
                        @test RK * full_z == zeros(Rational{BigInt}, n)
                        @test sum(Rational{BigInt}(b[i]) * z[i] for i in 1:nr) == q
                        @test abs(q) // sum(abs, z) == core.floor
                        cert = (z=z, q=q)
                    else
                        # No hardcoded null vector is trusted here: the null
                        # basis is COMPUTED by exact rational RREF.
                        _, basis = _exact_null_basis(RA)
                        @test length(basis) == nr - rankA
                        @test !isempty(basis)
                        for w in basis
                            @test RA * w == zeros(Rational{BigInt}, m)
                            full_w = vcat(w, zeros(Rational{BigInt}, m))
                            @test RK * full_w == zeros(Rational{BigInt}, n)
                        end
                        dots = [sum(Rational{BigInt}(b[i]) * w[i] for i in 1:nr) for w in basis]
                        nz = findfirst(q -> q != 0, dots)
                        # A mislabeled (actually compatible) case fails loudly here.
                        @test nz !== nothing
                        cert = (z=basis[something(nz)], q=dots[something(nz)])
                    end
                    tol = BigFloat(2; precision=p)^(32 - p)
                    dsigns = vcat(fill(1, nr), fill(-1, m))
                    for (policy, sh) in (
                            ("stated-borrowed-expanded-r0", _stated_shift(K0, p)),
                            ("balanced-experiment", _balanced_shift(K0, p, nr)),
                            )
                        Kd = _shifted_operator(K0, p, nr, sh.dx, sh.dy)
                        U = _upper_csc(Kd, p)
                        cache = BFLA.sparse_ldlt_cache(U; precision_bits=p, dsigns=dsigns, nrhs=1)
                        BFLA.factorize!(cache, U)
                        @test BFLA.issuccess(cache)
                        diag = BFLA.factor_diagnostics(cache)
                        @test diag.positive_inertia == nr
                        @test diag.regularized_entries == 0
                        rec = _refine_original!(cache, K0, Kd, b, p, tol, nothing)
                        analytic = cert === nothing ? nothing : abs(cert.q) // sum(abs, cert.z)
                        println("RECEIPT case=$kind p=$p policy=$policy " *
                            "scale=$(sh.scale) dx=$(sh.dx) dy=$(sh.dy) " *
                            "iters=$(rec.iters) orig_res=$(rec.orig_res) " *
                            "shifted_res=$(rec.shifted_res) analytic_floor=$analytic " *
                            "nnz_k=$(diag.nnz_k) nnz_l=$(diag.nnz_l) " *
                            "symbolic=$(diag.symbolic_count) numeric=$(diag.numeric_factor_count)")
                        if cert === nothing
                            # Compatible singular system: refinement can still
                            # succeed; success certifies nothing about rank.
                            @test rec.converged
                            @test rec.orig_res <= tol
                        else
                            # Independently evaluate the exact residual of
                            # the ACTUAL stored iterate against the retained
                            # original operator/RHS. Do not round this oracle
                            # back to the working precision before comparison.
                            r_exact = RK * Rational{BigInt}.(rec.x) - Rational{BigInt}.(b)
                            full_z = vcat(Rational{BigInt}.(cert.z), zeros(Rational{BigInt}, m))
                            @test sum(full_z .* r_exact) == -cert.q
                            @test maximum(abs, r_exact) >= analytic
                            r_stored = K0 * rec.x - b
                            accumulation_error = maximum(abs, Rational{BigInt}.(r_stored) - r_exact)
                            @test Rational{BigInt}(rec.orig_res) + accumulation_error >= analytic
                            # The separate stored-vector bound diagnoses its
                            # arithmetic, without standing in for the RHS proof.
                            s, _ = _certified_floor(r_stored, cert.z)
                            @test Rational{BigInt}(rec.orig_res) * sum(abs, cert.z) >= abs(s)
                            @test !rec.converged
                            @test rec.iters == 12
                        end
                    end
                end
            end
        end
        # ---- failure/revocation/incompatible controls (genuine core) ----
        @testset "controls p=$p" begin
            setprecision(BigFloat, p) do
                core = _genuine_core(p, false)
                K0, nr, m, b = core.K0, core.nr, core.m, core.b
                n = nr + m
                tol = BigFloat(2; precision=p)^(32 - p)
                dsigns = vcat(fill(1, nr), fill(-1, m))
                sh = _stated_shift(K0, p)
                Kd = _shifted_operator(K0, p, nr, sh.dx, sh.dy)
                U = _upper_csc(Kd, p)
                cache = BFLA.sparse_ldlt_cache(U; precision_bits=p, dsigns=dsigns, nrhs=1)
                BFLA.factorize!(cache, U)
                @test BFLA.issuccess(cache)
                # Genuine alias-destination rejection (factor workspace limb).
                F = something(cache.factor)
                @test length(F.Dinv.diag) == n
                try
                    BFLA.solve_trusted!(F.Dinv.diag, cache, b)
                    @test false
                catch e
                    @test e isa ArgumentError
                    @test occursin("alias", e.msg)
                end
                # Nonfinite factor input revokes solve authority.
                bad = SparseMatrixCSC(n, n, copy(U.colptr), copy(U.rowval),
                    BFLA.owned_copy(U.nzval; precision_bits=p))
                bad.nzval[1] = BigFloat(NaN; precision=p)
                BFLA.factorize!(cache, bad; check=false)
                @test BFLA.factor_status(cache).kind === :unprepared
                @test_throws ArgumentError BFLA.solve_trusted!(
                    BFLA.owned_zeros(BigFloat, n; precision_bits=p), cache, b)
                BFLA.factorize!(cache, U)
                @test BFLA.issuccess(cache)
                xw = BFLA.owned_zeros(BigFloat, n; precision_bits=p)
                @test_throws DimensionMismatch BFLA.solve_trusted!(
                    xw, cache, BFLA.owned_zeros(BigFloat, n + 1; precision_bits=p))
                bad_rhs = BFLA.owned_copy(b; precision_bits=p)
                bad_rhs[1] = BigFloat(Inf; precision=p)
                @test_throws DomainError BFLA.solve_trusted!(xw, cache, bad_rhs)
                @test BFLA.factor_diagnostics(cache).regularized_entries == 0
                @test something(cache.factor).workspace.regularize_count[1] == 0
            end
        end
    end
end
@assert _qualification_manifest() == _QUALIFICATION_BEFORE "qualification source/environment changed"
println("QUALIFICATION_SOURCE_UNCHANGED")
