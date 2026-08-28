#=====================================================================
    Provider spike: PureKLU 1.4.1 — exact general/nonsymmetric sparse LU
    adapter prototype (ext/SDPXPureKLUExt.jl).

    Scope (docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md promotion
    gates 1–7, 9):
      * manufactured exact nonsymmetric Newton operators in Float64,
        Float64x2, Float64x4, BigFloat256;
      * scalar preservation / no downcast / no global precision mutation;
      * klu analyze / factor / refactor reuse (symbolic + numeric
        workspace identity);
      * vector, multi-RHS, transpose solves through one factor;
      * factor residual + unregularized five-equation semantic Newton
        residual in the original scalar type;
      * stale factor / stale pattern rejection;
      * singular / condition failure behavior;
      * allocation / fill / wall-time observations.

    Explicit non-goals enforced here: no inertia claim (supports_inertia
    is false and the module has no inertia symbol), no certificate or
    terminal-status authority, no LinearSolve/SciMLBase loaded, and no
    HSD dispatch. This file is not part of the ordinary Pkg.test target;
    it is an extension-gated spike (see docs/evidence/
    pureklu_provider_spike.md for the driver).

    Run (spike environment with SDPX dev'd + PureKLU + MultiFloats):
      julia --project=/tmp/sdpx-pureklu-env \
            test/provider_spikes/pureklu_sparse.jl
=====================================================================#
using Test
using LinearAlgebra
using SparseArrays
using Random
using Printf
using SDPX
using MultiFloats: Float64x2, Float64x4, MultiFloat

if !isdefined(Main, :SDPXPureKLUExt)
    include(joinpath(@__DIR__, "..", "..", "ext", "SDPXPureKLUExt.jl"))
end
const PK = Main.SDPXPureKLUExt
const PUREKLU_PKG = PK.PUREKLU_VERSION

# Julia 1.12 (SparseArrays 1.12.0) emits 1-based `colptr` from `sparse()` while
# earlier Julia emits 0-based. Everything here goes through `nzrange`/API or
# whole-vector pattern comparison, so both conventions are covered; this is
# recorded as an environment finding in the evidence document.
const _SPARSE_COLPTR_BASE = try
    S = sparse([1], [1], [1.0], 1, 1)
    first(S.colptr)
catch
    0
end

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

"""Block-diagonal positive-definite cone linearization with `m` rows."""
function _cone_fixture(::Type{T}, rng, m::Int, block_sizes::Vector{Int}) where {T<:AbstractFloat}
    sum(block_sizes) == m || throw(ArgumentError("block sizes must cover m"))
    contributions = SDPX.LocalConeLinearization{T,Matrix{T},Vector{T}}[]
    start = 1
    for size in block_sizes
        rows = start:(start + size - 1)
        M = T.(randn(rng, size, size))
        M = (M + transpose(M)) / 2
        for i in 1:size
            M[i, i] += T(size + 2)
        end
        push!(contributions, SDPX.LocalConeLinearization(
            rows, M, zeros(T, size),
        ))
        start += size
    end
    return SDPX.assemble_cone_linearization(T, m, contributions)
end

"""
Manufacture an exact HSD Newton system whose expected direction solves the
five frozen equations exactly in `T` (the RHS is *derived*, never
approximated). `A` is sparse with a forced nonzero per column so the exact
expanded operator is nonsingular.
"""
function manufactured_system(
    ::Type{T}, rng; n::Int=3, m::Int=5,
    block_sizes::Vector{Int}=[2, 3],
) where {T<:AbstractFloat}
    A = spzeros(T, m, n)
    for i in 1:m, j in 1:n
        rand(rng) < 0.55 && (A[i, j] = T(randn(rng)))
    end
    for j in 1:n
        A[mod1(j, m), j] = T(randn(rng) + 5)   # full column rank
    end
    b = T.(randn(rng, m))
    c = T.(randn(rng, n))
    cone = _cone_fixture(T, rng, m, block_sizes)
    tau = T(1.25)
    kappa = T(0.8)
    expected = SDPX.NewtonDirection(
        T.(randn(rng, n)), T.(randn(rng, m)), T.(randn(rng, m)),
        T(randn(rng)), T(randn(rng)),
    )
    H = cone.operator
    primal = A * expected.dx + expected.ds - b * expected.dtau
    dual = transpose(A) * expected.dy + c * expected.dtau
    gap = -dot(c, expected.dx) - dot(b, expected.dy) + expected.dkappa
    cone_rhs = expected.ds + H * expected.dy
    scalar_rhs = kappa * expected.dtau + tau * expected.dkappa
    rhs = SDPX.HSDNewtonRHS(primal, dual, gap, cone_rhs, scalar_rhs)
    system = SDPX.NewtonSystem(A, b, c, cone, tau, kappa, rhs)
    return system, expected
end

"""Same-precision normwise backward error mirroring src/kkt/refinement.jl."""
function _kkt_backward_error(K, x, rhs)
    T = eltype(K)
    residual = K * x - rhs
    residual_norm = norm(residual, Inf)
    denominator = max(norm(K, Inf) * norm(x, Inf) + norm(rhs, Inf), one(T))
    return residual_norm / denominator
end

function _relative_direction_error(expected, direction)
    T = eltype(expected.dx)
    num = max(
        norm(expected.dx - direction.dx, Inf), norm(expected.dy - direction.dy, Inf),
        norm(expected.ds - direction.ds, Inf), abs(expected.dtau - direction.dtau),
        abs(expected.dkappa - direction.dkappa),
    )
    den = max(
        norm(expected.dx, Inf), norm(expected.dy, Inf), norm(expected.ds, Inf),
        abs(expected.dtau), abs(expected.dkappa), one(T),
    )
    return num / den
end

"""Build a same-pattern perturbed operator: values scaled by (1 + δ·randn)."""
function _perturbed(::Type{T}, rng, A::SparseMatrixCSC{T,Int}, δ::Float64) where {T<:AbstractFloat}
    V = A.nzval .* (1 .+ T(δ) .* T.(randn(rng, length(A.nzval))))
    return SparseMatrixCSC(size(A, 1), size(A, 2), copy(A.colptr), copy(A.rowval), V)
end

# ---------------------------------------------------------------------------
# exact expanded route — one scalar at a time
# ---------------------------------------------------------------------------

function _check_exact_expanded_route(::Type{T}, rng; label::String) where {T<:AbstractFloat}
    @testset "exact expanded route ($label)" begin
        system, expected = manufactured_system(T, rng)
        m, n = size(system.A)
        K = PK.assemble_expanded_kkt_sparse(system)
        dimension = n + m + 1
        @test size(K) == (dimension, dimension)
        @test eltype(K) === T
        # genuine nonsymmetry: (x,τ) skew blocks and (y,τ) −b blocks
        @test !issymmetric(Matrix(K))

        session = PK.PureKLUSession(T, dimension)
        @test !PK.supports_inertia(session)
        @test !PK.is_analyzed(session)
        PK.set_operator!(session, K)
        @test PK.analyze!(session)
        @test PK.is_analyzed(session)
        @test !PK.is_factored(session)
        @test PK.factor!(session)
        @test PK.is_factored(session)
        @test PK.is_fresh(session)
        symbolic_before = session.factor.symbolic
        numeric_before = session.factor.numeric
        metrics = PK.fill_metrics(session)
        @test metrics.lnz >= 0 && metrics.unz >= 0 && metrics.nblocks >= 1
        @test metrics.status_code == 0          # KLU_OK; never an inertia claim

        # --- scalar preservation / no downcast ---------------------------
        @test session.factor.nzval isa Vector{T}
        @test eltype(session.factor.nzval) === T

        # --- vector solve + factor residual -------------------------------
        rhs = PK.expanded_rhs_vector(system)
        @test eltype(rhs) === T
        x = PK.solve(session, rhs)
        @test x !== nothing
        @test x isa Vector{T}
        be = _kkt_backward_error(K, x, rhs)
        @test be <= T(256) * eps(T)
        absres, relres = PK.factor_residual(session, x, rhs)
        @test relres <= T(256) * eps(T)
        @test absres >= zero(T)

        # --- unregularized five-equation semantic Newton residual ----------
        direction = PK.recover_expanded_direction(system, x)
        semantic = PK.semantic_max_residual(system, direction)
        @test semantic <= T(256) * eps(T)
        # and the recovered direction matches the manufactured one
        @test _relative_direction_error(expected, direction) <= T(1024) * eps(T)

        # --- multi-RHS through the SAME factor ----------------------------
        rhs_panel = hcat(rhs, 2 * rhs)
        x_panel = PK.solve(session, rhs_panel)
        @test x_panel !== nothing
        @test size(x_panel) == size(rhs_panel)
        @test x_panel[:, 1] ≈ x atol=T(1024) * eps(T)
        @test x_panel[:, 2] ≈ 2 .* x atol=T(1024) * eps(T)
        @test session.factor.numeric === numeric_before   # no re-factor

        # --- transpose solve (KLU tsolve, same factor) --------------------
        yt = PK.solve_transpose(session, rhs)
        @test yt !== nothing
        @test eltype(yt) === T
        @test _kkt_backward_error(transpose(K), yt, rhs) <= T(256) * eps(T)

        # --- refactor reuse (same pattern, new values) --------------------
        A2 = _perturbed(T, rng, K, 1e-6)
        @test PK.refactor!(session, A2)
        @test PK.is_factored(session) && PK.is_fresh(session)
        @test session.factor.symbolic === symbolic_before   # analyze reused
        @test session.factor.numeric === numeric_before     # workspace reused
        @test session.matrix === A2
        x2 = PK.solve(session, rhs)
        @test x2 !== nothing
        @test _kkt_backward_error(A2, x2, rhs) <= T(256) * eps(T)
        yt2 = PK.solve_transpose(session, rhs)
        @test yt2 !== nothing
        @test _kkt_backward_error(transpose(A2), yt2, rhs) <= T(256) * eps(T)

        # --- stale factor rejection ---------------------------------------
        A3 = _perturbed(T, rng, K, 1e-4)          # same pattern, new values
        PK.set_operator!(session, A3)
        @test !PK.is_fresh(session)
        @test PK.solve(session, rhs) === nothing  # rejected, not silently wrong
        @test session.status == PK.PUREKLU_STALE
        @test PK.refactor!(session, A3)           # recovers through refactor
        x3 = PK.solve(session, rhs)
        @test x3 !== nothing
        @test _kkt_backward_error(A3, x3, rhs) <= T(256) * eps(T)

        # --- stale pattern rejection --------------------------------------
        A4 = SparseMatrixCSC(
            dimension, dimension, copy(A3.colptr), copy(A3.rowval),
            copy(A3.nzval),
        )
        # drop one structural entry: different pattern, same dimension.
        # NOTE: `nzrange` is used instead of raw colptr arithmetic because
        # Julia 1.12's SparseArrays 1.12.0 emits a *1-based* colptr while
        # earlier versions emit 0-based; `nzrange` is the canonical accessor.
        col = findfirst(c -> !isempty(nzrange(A4, c)), 1:dimension)
        drop_pos = first(nzrange(A4, col))
        for p in drop_pos:(length(A4.rowval) - 1)   # shift the whole tail down
            A4.rowval[p] = A4.rowval[p + 1]
            A4.nzval[p] = A4.nzval[p + 1]
        end
        resize!(A4.rowval, length(A4.rowval) - 1)
        resize!(A4.nzval, length(A4.nzval) - 1)
        for c in (col + 1):(dimension + 1)
            A4.colptr[c] -= 1
        end
        @assert all(c -> isempty(nzrange(A4, c)) ||
                          issorted(view(A4.rowval, nzrange(A4, c))), 1:dimension)
        @test !PK.refactor!(session, A4)          # pattern change is rejected
        @test session.last_reason == :pattern_changed
        @test PK.solve(session, rhs) === nothing  # and solves stay fail-closed
        PK.set_operator!(session, A4)
        @test PK.factor!(session)                 # full re-analyze recovers
        @test PK.is_fresh(session)
        x4 = PK.solve(session, rhs)
        @test x4 !== nothing
        @test _kkt_backward_error(A4, x4, rhs) <= T(256) * eps(T)

        return (be=Float64(be), semantic=Float64(semantic))
    end
end

# ---------------------------------------------------------------------------
# singular / condition failure
# ---------------------------------------------------------------------------

function _check_failure_ladder()
    @testset "singular / condition failure" begin
        # exact singularity -> KLU_SINGULAR, no exception, solve rejected
        S = sparse([1.0 1.0 0.0; 1.0 1.0 0.0; 0.0 0.0 2.0])
        session = PK.PureKLUSession(Float64, 3)
        PK.set_operator!(session, S)
        @test !PK.factor!(session)
        @test PK.is_singular(session)
        @test session.status_code == 1            # KLU_SINGULAR
        @test session.numerical_rank >= 0         # rank diagnosis (not inertia)
        @test session.singular_col >= 0
        @test PK.solve(session, ones(3)) === nothing
        @test !PK.is_fresh(session)

        # near-singular: tiny-but-nonzero pivot (1e-14) passes KLU, and the
        # forward error is dominated by condition in Float64 while the same
        # system in BigFloat256 solves accurately. The factor residual alone
        # is backward-stable in both — exactly why the route must keep the
        # original-scalar semantic gate and must not downcast.
        M = [1.0 1.0 0.0; 1.0 1.0 + 1e-14 0.0; 0.0 0.0 2.0]
        b = [1.0, 2.0, 1.0]
        # exact reference computed in BigFloat 256 (not Float64 LU: two Float64
        # LU implementations agree with each other at eps, hiding the failure)
        xref = setprecision(BigFloat, 256) do
            BigFloat.(M) \ BigFloat.(b)
        end
        setprecision(BigFloat, 256) do
            # Float64: factor succeeds (pivot 1e-14 is nonzero) but the
            # direction is condition-dominated — forward error ≫ eps.
            ses = PK.PureKLUSession(Float64, 3)
            PK.set_operator!(ses, sparse(Float64.(M)))
            @test PK.factor!(ses)
            xs = PK.solve(ses, Float64.(b))
            @test xs !== nothing
            @test norm(xs .- Float64.(xref), Inf) > 1e-8     # condition failure

            # BigFloat256: same system, exact scalar — accurate direction.
            sesB = PK.PureKLUSession(BigFloat, 3)
            PK.set_operator!(sesB, sparse(BigFloat.(M)))
            @test PK.factor!(sesB)
            xb = PK.solve(sesB, BigFloat.(b))
            @test xb !== nothing
            @test eltype(xb) === BigFloat
            # observed difference ~1.6e-50 (two independent 256-bit rounding
            # paths, both within the κ·eps·‖x‖ ≈ 1.7e-49 bound); gate at 1e-40
            @test norm(xb - xref, Inf) < big"1e-40"          # no downcast wins
        end
    end
end

# ---------------------------------------------------------------------------
# reduced-Schur-shaped operator (nonsymmetric (c−g) vs (c+g))
# ---------------------------------------------------------------------------

function _check_reduced_schur_shape()
    @testset "reduced-Schur-shaped nonsymmetric operator" begin
        T = Float64
        rng = MersenneTwister(0x52)
        # genuinely diagonal block H (1:2 and 3:5) so H⁻¹ is trivial and the
        # reduced (c−g) vs (c+g) signs are validated against src exactly
        cone = SDPX.assemble_cone_linearization(T, 5, [
            SDPX.LocalConeLinearization(1:2, Diagonal(T[2, 3]), zeros(T, 2)),
            SDPX.LocalConeLinearization(3:5, Diagonal(T[4, 5, 6]), zeros(T, 3)),
        ])
        H = cone.operator
        @test all(==(0.0), H[i, j] for i in 1:5, j in 1:5 if i != j)  # diagonal
        A = spzeros(T, 5, 3)
        for i in 1:5, j in 1:3
            rand(rng) < 0.6 && (A[i, j] = T(randn(rng)))
        end
        for j in 1:3
            A[mod1(j, 5), j] = T(randn(rng) + 5)
        end
        b = T.(randn(rng, 5)); c = T.(randn(rng, 3))
        tau = T(1.25); kappa = T(0.8)
        expected = SDPX.NewtonDirection(
            T.(randn(rng, 3)), T.(randn(rng, 5)), T.(randn(rng, 5)),
            T(randn(rng)), T(randn(rng)),
        )
        primal = A * expected.dx + expected.ds - b * expected.dtau
        dual = transpose(A) * expected.dy + c * expected.dtau
        gap = -dot(c, expected.dx) - dot(b, expected.dy) + expected.dkappa
        cone_rhs = expected.ds + H * expected.dy
        scalar_rhs = kappa * expected.dtau + tau * expected.dkappa
        system = SDPX.NewtonSystem(
            A, b, c, cone, tau, kappa,
            SDPX.HSDNewtonRHS(primal, dual, gap, cone_rhs, scalar_rhs),
        )

        # src/kkt/reduced_schur.jl signs (code-authoritative):
        #   [ M        c − g  ]         M = A'H⁻¹A, g = A'H⁻¹b, q = b'H⁻¹b
        #   [ (c + g)'  κ/τ − q]   ρ = κ/τ − q
        D = diag(H) .^ (-1)
        M = transpose(A) * Diagonal(D) * A
        g = transpose(A) * (D .* b)
        q = dot(b, D .* b)
        atc = transpose(A) * (D .* (cone_rhs .- primal))
        btc = dot(cone_rhs .- primal, D .* b)
        S = zeros(T, 4, 4)
        S[1:3, 1:3] = M
        for j in 1:3
            S[j, 4] = c[j] - g[j]
            S[4, j] = c[j] + g[j]
        end
        S[4, 4] = kappa / tau - q
        rhs4 = vcat(dual .- atc, -gap - btc + scalar_rhs / tau)

        session = PK.PureKLUSession(T, 4)
        PK.set_operator!(session, sparse(S))
        @test PK.factor!(session)
        sol = PK.solve(session, rhs4)
        @test sol !== nothing
        @test _kkt_backward_error(sparse(S), sol, rhs4) <= T(256) * eps(T)

        dx = sol[1:3]; dtau = sol[4]
        rC_minus_rP = cone_rhs .- primal
        rhs_dy = rC_minus_rP + A * dx - b * dtau
        dy = D .* rhs_dy
        ds = cone_rhs .- H * dy
        dkappa = (scalar_rhs - kappa * dtau) / tau
        direction = SDPX.NewtonDirection(dx, dy, ds, dtau, dkappa)
        @test PK.semantic_max_residual(system, direction) <= T(256) * eps(T)
        @test _relative_direction_error(expected, direction) <= T(1024) * eps(T)
    end
end

# ---------------------------------------------------------------------------
# allocation / fill / wall time observations
# ---------------------------------------------------------------------------

function _check_metrics_and_time()
    @testset "allocation / fill / wall time" begin
        rng = MersenneTwister(0x53)
        n = 60
        system, _ = manufactured_system(Float64, rng; n=n, m=72, block_sizes=[72])
        K = PK.assemble_expanded_kkt_sparse(system)
        dimension = size(K, 1)
        session = PK.PureKLUSession(Float64, dimension)
        PK.set_operator!(session, K)
        PK.analyze!(session)
        PK.factor!(session)
        mf = PK.fill_metrics(session)
        @test mf.lnz + mf.unz > 0
        @test mf.nblocks >= 1
        @test mf.maxblock <= dimension

        rhs = PK.expanded_rhs_vector(system)
        dest = zeros(Float64, dimension)
        panel = hcat(rhs, 2rhs, 3rhs)
        dest_panel = zeros(Float64, dimension, 3)
        yt = zeros(Float64, dimension)
        PK.solve!(session, dest, rhs)      # warm-up (JIT)
        PK.solve!(session, dest_panel, panel)
        PK.solve_transpose!(session, yt, rhs)
        alloc_solve = @allocated PK.solve!(session, dest, rhs)
        alloc_multi = @allocated PK.solve!(session, dest_panel, panel)
        alloc_transpose = @allocated PK.solve_transpose!(session, yt, rhs)

        t_analyze = @elapsed begin
            s2 = PK.PureKLUSession(Float64, dimension)
            PK.set_operator!(s2, K); PK.analyze!(s2)
        end
        t_factor = @elapsed begin
            s2 = PK.PureKLUSession(Float64, dimension)
            PK.set_operator!(s2, K); PK.factor!(s2)
        end
        t_refactor = @elapsed begin
            K2 = _perturbed(Float64, rng, K, 1e-6)
            PK.refactor!(session, K2)
        end
        t_solve = @elapsed PK.solve!(session, dest, rhs)

        @printf("  [allocation/fill] n=%d dim=%d nnz(A)=%d lnz=%d unz=%d nzoff=%d blocks=%d maxblock=%d\n",
            n, dimension, nnz(K), mf.lnz, mf.unz, mf.nzoff, mf.nblocks, mf.maxblock)
        @printf("  [allocation] solve-inplace=%d B multi-rhs(3)=%d B transpose=%d B\n",
            alloc_solve, alloc_multi, alloc_transpose)
        @printf("  [wall time] analyze=%.3f ms factor=%.3f ms refactor=%.3f ms solve=%.3f ms\n",
            1e3 * t_analyze, 1e3 * t_factor, 1e3 * t_refactor, 1e3 * t_solve)

        # coarse sanity only; precise numbers are evidence-doc material
        @test alloc_solve < 1 << 16
        @test alloc_transpose < 1 << 16
        @test t_solve < 10.0
        @test t_refactor < 10.0
    end
end

# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

@testset "PureKLU provider spike ($PUREKLU_PKG)" begin
    @test PUREKLU_PKG == "1.4.1"
    @info "sparse colptr base on this Julia/SparseArrays" base=_SPARSE_COLPTR_BASE

    # never claims inertia or certificate authority
    @test :inertia ∉ names(PK, all=true)
    @test :certificate ∉ names(PK, all=true)
    ses = PK.PureKLUSession(Float64, 4)
    @test !PK.supports_inertia(ses)

    # never loads LinearSolve/SciMLBase
    @test Base.get_extension(SDPX, :SDPXLinearSolveExt) === nothing
    @test !isdefined(Main, :LinearSolve)
    @test !isdefined(Main, :SciMLBase)

    results = Dict{String,Any}()
    setprecision(BigFloat, 256) do
        @test precision(BigFloat) == 256
        rng64 = MersenneTwister(0x5001)
        results["Float64"] = _check_exact_expanded_route(
            Float64, rng64; label="Float64",
        )
        rng2 = MersenneTwister(0x5002)
        results["Float64x2"] = _check_exact_expanded_route(
            Float64x2, rng2; label="Float64x2",
        )
        rng4 = MersenneTwister(0x5004)
        results["Float64x4"] = _check_exact_expanded_route(
            Float64x4, rng4; label="Float64x4",
        )
        rngB = MersenneTwister(0x5008)
        results["BigFloat256"] = _check_exact_expanded_route(
            BigFloat, rngB; label="BigFloat256",
        )
        # BigFloat precision must survive untouched inside the route
        @test precision(BigFloat) == 256
    end
    @test precision(BigFloat) == 256   # no global precision mutation

    _check_failure_ladder()
    _check_reduced_schur_shape()
    _check_metrics_and_time()

    @info "PureKLU spike observed backward errors" results
end
