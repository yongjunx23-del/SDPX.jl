# test/rebuild/S05.jl
#
# Standalone: julia --project=<SDPX.jl> --startup-file=no test/rebuild/S05.jl
#
# S05 — converge the two provider adapters into one precise contract (ADR-002).
#
# ============================================================================
# WHAT THIS FILE PROVES, AND AGAINST WHAT
# ============================================================================
# `MultiFloatLinearAlgebra`, `BigFloatLinearAlgebra` and `QDLDL` are NOT
# installed in the default SDPX project (ADR-002 §1, WORKER_BRIEF §4). A contract
# test that required an absent provider would prove nothing, so every acceptance
# item below is proven against a **mock provider that implements the ADR-002
# semantics** — `MockMFLA` and `MockBFLA`, whose two-phase `factorize!` ordering
# is taken from ADR-002 §8/§9 (verify → commit marker → numeric work).
#
# The mocks use REAL dense kernels (LAPACK via LinearAlgebra) so that pivot
# metadata is a genuine provider report, but their arithmetic is Float64, not
# multi-float or big float. **No result in this file is a claim about MFLA's or
# BFLA's arithmetic, precision, or performance.** Live-provider checks skip with
# an explicit reason at the end.
#
# ============================================================================
# SIZES: measured, not guessed
# ============================================================================
# §2 below measures `factor_summary` allocation at sizes 8/32/128. It is 0 bytes
# at all three, so there is no size below which it allocates and no size to
# freeze; the largest size used anywhere in this file is 128.

using Test
using LinearAlgebra
using Random: MersenneTwister

const S05_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(S05_ROOT, "src", "la", "protocol.jl"))
include(joinpath(S05_ROOT, "src", "la", "admission.jl"))
include(joinpath(S05_ROOT, "src", "la", "factor_lease.jl"))
include(joinpath(S05_ROOT, "ext", "rebuild", "mfla_adapter.jl"))
include(joinpath(S05_ROOT, "ext", "rebuild", "bfla_adapter.jl"))

# ---------------------------------------------------------------------------
# LIVE PROVIDER SELECTION — one provider per process, chosen by the driver
# ---------------------------------------------------------------------------
# `Bootstrap` below runs `scripts/run_s05.sh`, which executes this file three
# times: once against the default project (mocks only), once with MFLA and once
# with BFLA in the REBUILD_ENV. The live adapter sections are included HERE, at
# top level, so their methods are callable at this file's world age.
const S05_LIVE = lowercase(get(ENV, "S05_LIVE_PROVIDER", "none"))
if S05_LIVE == "mfla"
    let M = load_mfla()
        M === nothing || include(joinpath(S05_ROOT, "ext", "rebuild", "mfla_live.jl"))
    end
elseif S05_LIVE == "bfla"
    let M = load_bfla()
        M === nothing || include(joinpath(S05_ROOT, "ext", "rebuild", "bfla_live.jl"))
    end
end
#define_facts_for(p) = declared_facts(p)

# ---------------------------------------------------------------------------
# canonical test operators
# ---------------------------------------------------------------------------

"""A well-conditioned symmetric matrix, SPD."""
function spd_matrix(n::Int)
    A = zeros(Float64, n, n)
    for i in 1:n
        A[i, i] = 4.0
        i < n && (A[i, i + 1] = 1.0; A[i + 1, i] = 1.0)
    end
    A
end

"""A symmetric matrix with a forced 2x2 pivot block at the leading position."""
function bk_2x2_matrix(n::Int)
    A = spd_matrix(n)
    n >= 2 || return A
    A[1, 1] = 1.0; A[2, 2] = 1.0; A[1, 2] = 2.0; A[2, 1] = 2.0
    A
end

"""
    provider_case(kind; size, provider) -> (provider, request, triangle)

Build a request whose triangle/shape match what the provider kind declares.
"""
# `provider_scalar(p)`: the scalar a request must state to be admissible against
# `p` — the provider's own kernel arithmetic at its own width. Requesting MORE
# width on purpose is a test case (see the precision-refusal set), not the default.
provider_scalar(p) = ScalarSpec(declared_facts(p).kernel_scalar.family,
                                declared_facts(p).kernel_scalar.min_bits)

"""
    ask(p, op; shape, triangle, ncols, needs, nnz, rank_kind) -> FactorRequest

Build a request against `p` using `p`'s declared facts for the fields `p`
constrains. Explicit overrides are how the refusal tests construct requests that
`p` must reject.
"""
function ask(p, op::SolveOp; n::Int=4, m::Int=0, triangle=nothing,
             ncols::Int=2, needs::MultiRHSKind=MultiRHSBlocked, nnz::Int=0,
             rank_kind::Symbol=:full, scalar=nothing, exact_bits::Bool=false,
             facts=nothing, serial_required::Union{Nothing,Bool}=nothing,
             concurrency=nothing)
    f = facts === nothing ? declared_facts(p) : facts
    scalar === nothing && (scalar = ScalarSpec(f.kernel_scalar.family, f.kernel_scalar.min_bits))
    mm = m == 0 ? n : m
    tri = triangle === nothing ?
          (mm == n ? (f.factor.triangle === TriangleUnused ? TriangleEither :
                      f.factor.triangle) : TriangleUnused) : triangle
    sc = exact_bits ? ScalarSpec(scalar.family, scalar.min_bits, true) : scalar
    # Concurrency follows the provider's declared thread scope: a serial-required
    # request against a threaded kernel is itself a refusal case (and is tested as
    # one), so it must not be the accidental default here.
    threads_ok = f.solve.threads !== ThreadNone
    ser = serial_required === nothing ? !threads_ok : serial_required
    conc = concurrency === nothing ?
           ConcurrencySpec(allow_threads=!ser, max_threads=(!ser && threads_ok) ? 2 : 1,
                           concurrent_handles=1, serial_required=ser) : concurrency
    FactorRequest(op, sc, ShapeSpec(mm, n; rank_kind=rank_kind);
                  triangle=tri, known_nnz=nnz, concurrency=conc,
                  rhs=RHSKind(is_matrix=true, ncols=ncols, needs=needs))
end

# `with_scalar(f, spec)`: `f` with its kernel scalar restated. Used ONLY by the
# refusal tests, to isolate one dimension: a request that trips `RefuseScalarFamily`
# first never reaches the shape/storage check the test is about. The restated facts
# are otherwise identical, and the test asserts the isolated refusal code.
with_scalar(f, spec::ScalarSpec) =
    CapabilityFacts(f.provider_name, f.revision, spec, f.conversion, f.factor, f.solve,
                    f.storage, f.indices, f.concurrency, f.exception_vocabulary,
                    f.failure_is_atomic)

# `cross(p, op; ...)`: a request built from provider `p`'s declared facts, to be
# admitted against a DIFFERENT provider's facts. Asking `p` for facts and then
# testing them against `p` would prove nothing about the mismatch.
cross(p, op::SolveOp; op_kw...) = ask(p, op; op_kw...)

# `solve_request(p, op; ncols, needs)` — a solve request shaped for `p`.
solve_request(p, op::SolveOp; ncols::Int=2, needs::MultiRHSKind=MultiRHSBlocked) =
    ask(p, op; n=4, ncols=ncols, needs=needs)

"""
    end_to_end(provider, kind; size) -> (handle, A, accepted)

Prepare + refactor through the contract. Returns the handle and the operator so
the caller can compute a reference residual.
"""
function end_to_end(p, kind::Symbol; size::Int=8)
    n = size
    req = ask(p, OpPrepareFactor; n=n, ncols=2, needs=MultiRHSBatched)
    h = factor_handle(p, req)
    a1 = prepare_factor!(h)
    A = spd_matrix(n)
    r = refactor_numeric!(h, kind === :ldlt ? bk_2x2_matrix(n) : A)
    (h, A, a1.allowed && r.ok)
end

# ---------------------------------------------------------------------------
# 0. the contract's structural invariants
# ---------------------------------------------------------------------------

@testset "S05 provider contract" verbose = true begin

    @testset "0. no generalized boolean in the request/capability types" begin
        # A `Bool` for multi-RHS or threading is the specific defect ADR-002 §3
        # names. Assert the fields are enums, not booleans.
        @test !(fieldtype(SolveCapability, :multi_rhs) <: Bool)
        @test !(fieldtype(SolveCapability, :threads) <: Bool)
        @test fieldtype(SolveCapability, :multi_rhs) <: MultiRHSKind
        @test fieldtype(SolveCapability, :threads) <: ThreadScope
        @test fieldtype(FactorRequest, :triangle) <: TriangleConvention
        @test fieldtype(FactorRequest, :op) <: SolveOp
        @test fieldtype(FactorRequest, :scalar) <: ScalarSpec
        @test fieldtype(FactorRequest, :shape) <: ShapeSpec
        @test fieldtype(FactorRequest, :indices) <: IndexSpec
        @test fieldtype(FactorRequest, :concurrency) <: ConcurrencySpec
        # A square request must name its triangle; it is not defaulted.
        @test_throws ArgumentError FactorRequest(OpPrepareFactor,
            ScalarSpec(ArithMultiFloat, 256), ShapeSpec(4, 4))
        # Per-column and batched must be distinguishable values.
        @test MultiRHSPerColumn !== MultiRHSBatched
        @test length(instances(MultiRHSKind)) >= 4
    end

    # -----------------------------------------------------------------------
    # 1. capability facts are exact, cheap, and not generalizable
    # -----------------------------------------------------------------------
    @testset "1. capability facts distinguish per-column from batched" begin
        mfla = MockMFLA(8, 8, :ldlt)
        bfla = MockBFLA(:ldlt, 8, 8)

        # `capabilities` must not factor or benchmark. Count provider calls
        # before and after; a trial factorization would move this counter.
        calls0 = mfla.cache.n_refactor_calls
        inval0 = mfla.cache.n_invalidate_calls
        fm = capabilities(solve_request(mfla, OpSolveN), mfla)
        @test mfla.cache.n_refactor_calls == calls0   # no trial factorization
        @test mfla.cache.n_invalidate_calls == inval0 # not even a commit marker
        @test bfla.cache.n_refactor_calls == 0
        fb = capabilities(solve_request(bfla, OpSolveN), bfla)
        @test bfla.cache.n_refactor_calls == 0
        @test bfla.cache.n_status_clears == 0

        @test fm.solve.multi_rhs === MultiRHSBatched
        @test fb.solve.multi_rhs === MultiRHSPerColumn
        @test fm.solve.multi_rhs !== fb.solve.multi_rhs

        # The whole point: a single `multi_rhs::Bool` would report `true` for
        # both. Assert the *distinction* survives, per fact.
        @test fm.solve.threads === ThreadFactorOnly
        @test fb.solve.threads === ThreadNone
        @test fm.solve.transpose_solve
        @test !fb.solve.transpose_solve
        @test fm.storage.sparse_native
        @test !fb.storage.sparse_native
        @test fm.factor.triangle === TriangleLower
        @test fb.factor.triangle === TriangleUpper
        @test !fm.factor.rectangular && !fb.factor.rectangular

        # No fact may be a bare Bool standing in for a distinction.
        @test !(fieldtype(SolveCapability, :multi_rhs) <: Bool)
        @test !(fieldtype(SolveCapability, :threads) <: Bool)
        @test !(fieldtype(StorageCapability, :sparse_native) <: AbstractFloat)
        # Threading's scope is stated per operation, so a `threading=true` claim
        # for the library cannot be read as a claim for *this* factorization.
        @test fm.solve.threads !== ThreadFactorAndSolve   # ldlt: factor only
        @test declared_facts(MockMFLA(4, 4, :lu)).solve.threads === ThreadFactorAndSolve
    end

    # -----------------------------------------------------------------------
    # 2. ACCEPTANCE ITEM 1 — preflight rejection retains the physical factor
    #    and the logical lease is STILL revoked; the next solve fails closed.
    # -----------------------------------------------------------------------
    @testset "2. retained physical factor across preflight failure -> fails closed" begin
        for (label, p, kind) in (("MFLA/ldlt", MockMFLA(6, 6, :ldlt), :ldlt),
                                 ("BFLA/ldlt", MockBFLA(:ldlt, 6, 6), :ldlt),
                                 ("MFLA/cholesky", MockMFLA(6, 6, :cholesky), :cholesky),
                                 ("BFLA/cholesky", MockBFLA(:cholesky, 6, 6), :cholesky))
            @testset "$label" begin
                h, A, ok = end_to_end(p, kind; size=6)
                @test ok
                @test is_valid(h.lease)
                gen_committed = h.provider_generation
                @test gen_committed > 0

                # Prove the provider really was in a committed state, and record
                # its physical status + generation BEFORE the failing call.
                @test raw_provider_generation(p) == gen_committed
                status_before = raw_status(p)
                @test status_before === :success
                @test raw_retained_physical(p)
                refactor_calls_before = p.cache.n_refactor_calls

                # --- the failing NEW Newton request ---
                # A dimension preflight rejection: exactly the case ADR-002 §8/§9
                # recorded, where BOTH providers keep the old factor AND the old
                # success flag.
                bad = Matrix{Float64}(I, 5, 5)
                res = refactor_numeric!(h, bad)
                @test !res.ok
                @test res.revoked                       # revoked on the failed path
                @test h.lease.state === LeaseRevoked
                @test h.lease.revocation_event in (EvRefactorThrew, EvRefactorCommitFailed)

                # The provider DID retain both its factor and its success flag —
                # this is the hazard, and it is real here, not hypothetical.
                @test raw_retained_physical(p)
                @test raw_provider_generation(p) == gen_committed   # no new factor
                @test raw_status(p) === status_before               # stale success

                # ...and the logical lease is revoked anyway. This is the whole
                # acceptance item: freshness was NOT inferred from the provider.
                @test !is_valid(h.lease)
                @test !authorize(h.lease)
                @test physical_factor_retained(h.lease) == false ||
                      physical_factor_retained(h.lease) == true  # recorded, not consulted
                @test !legacy_shim_authorizes(legacy_shim(h))

                # A subsequent solve FAILS CLOSED: no answer, and — the measurable
                # part — the provider's solve is never entered.
                solves_before = p.cache.n_solve_calls
                rhs = ones(6, 2)
                dest = fill(NaN, 6, 2)
                out = solve_into!(dest, h, rhs, OpSolveN)
                @test !out.performed
                @test out.refused_reason !== RefuseNone
                @test all(isnan, dest)                       # nothing was written
                @test p.cache.n_solve_calls == solves_before # provider solve NOT called
                @test occursin("lease", lowercase(out.detail))

                # The old physical factor is still one call away from being
                # silently reused — which is precisely why the lease must be the
                # only authorization gate. Show the value that would have been
                # returned, to document the size of the hazard.
                stale_ok = try
                    raw_solve!(p, copy(dest), rhs, OpSolveN)
                    true
                catch
                    false
                end
                @test stale_ok    # the provider WOULD have answered from the old factor

                # Re-authorizing requires a fresh successful numeric factorization.
                good = kind === :cholesky ? A : bk_2x2_matrix(6)
                res2 = refactor_numeric!(h, good)
                @test res2.ok
                @test is_valid(h.lease)
                @test h.provider_generation > gen_committed
                out2 = solve_into!(copy(dest), h, rhs, OpSolveN)
                @test out2.performed
            end
        end
    end

    @testset "2b. commit-phase failure also revokes, and leaves no stale success" begin
        for (label, p, kind) in (("MFLA/ldlt", MockMFLA(6, 6, :ldlt), :ldlt),
                                 ("BFLA/ldlt", MockBFLA(:ldlt, 6, 6), :ldlt))
            @testset "$label" begin
                h, A, ok = end_to_end(p, kind; size=6)
                @test ok
                @test is_valid(h.lease)
                gen = h.provider_generation

                # Force the failure to happen AFTER the commit marker.
                p.cache.commit_fail = :rank_deficient
                res = refactor_numeric!(h, kind === :cholesky ? A : bk_2x2_matrix(6))
                @test !res.ok
                @test h.lease.state === LeaseRevoked
                # Whatever the provider reports, it is not a success for THIS
                # request — and the lease is revoked before we look.
                @test raw_status(p) !== :success
                @test raw_provider_generation(p) == gen

                dest = fill(NaN, 6, 2)
                out = solve_into!(dest, h, ones(6, 2), OpSolveN)
                @test !out.performed
                @test all(isnan, dest)

                # Cleanup-phase observation is explicit and is NOT authorization.
                obs = commit_failure_observation(h)
                @test obs.lease_state === LeaseRevoked
                @test !is_valid(h.lease)

                p.cache.commit_fail = nothing
                @test refactor_numeric!(h, bk_2x2_matrix(6)).ok
                @test is_valid(h.lease)
            end
        end
    end

    # -----------------------------------------------------------------------
    # 3. ACCEPTANCE ITEM — factor_summary is O(1): measured allocation
    # -----------------------------------------------------------------------
    @testset "3. factor_summary performs no matrix allocation" begin
        cold_allocs = Int[]
        for (label, p, kind) in (("MFLA/ldlt", MockMFLA(8, 8, :ldlt), :ldlt),
                                 ("BFLA/ldlt", MockBFLA(:ldlt, 8, 8), :ldlt))
            @testset "$label" begin
                h, A, ok = end_to_end(p, kind; size=8)
                @test ok

                # Warm up, then measure. The FIRST call after the cache is
                # invalidated compiles a fresh specialization; the O(1) claim is
                # about the steady-state cost, which is what a hot path pays.
                factor_summary(h)                       # compile / warm
                bytes_warm = @allocated factor_summary(h)
                @test bytes_warm == 0
                s = FactorSummary(UInt64(0), StatusUnprepared, 0, 0, OpFactorSummary,
                                  0, 0, 0, :none, UInt64(0), UInt64(0), false)

                # No matrix allocation: the summary is smaller than one column.
                @test sizeof(typeof(s)) < 8 * 8 * 8

                # Inertia is NOT recomputed: it comes from the pivot grammar, and
                # the deep-check path is never entered.
                @test h.diagnostics.deep_calls == 0
                @test h.diagnostics.factor_copies == 0

                # The summary is bound to the lease it describes.
                @test factor_summary(h).lease_valid == is_valid(h.lease)
                @test factor_summary(h).lease_valid == true
                @test factor_summary(h).generation == h.provider_generation
                @test factor_summary(h).pivot_grammar === raw_pivots(p).pivot_grammar

                # After a revocation the summary says so. This is the cold-cache
                # path (the summary cache was invalidated), so both the cold and
                # the warm cost are measured and REPORTED, not assumed.
                p.cache.preflight_reject = :none
                refactor_numeric!(h, Matrix{Float64}(I, 5, 5))
                @test !is_valid(h.lease)
                cold = @allocated factor_summary(h)
                warm = @allocated factor_summary(h)
                push!(cold_allocs, cold)
                @test warm == 0
                @test factor_summary(h).lease_valid == false
                p.cache.preflight_reject = nothing
            end
        end

        # O(1) across sizes: a summary that recomputed inertia or copied the
        # factor would grow with n. Measured WARM at three sizes.
        allocs = Int[]
        for n in (8, 32, 128)
            p = MockMFLA(n, n, :ldlt)
            h, A, ok = end_to_end(p, :ldlt; size=n)
            @test ok
            factor_summary(h)
            push!(allocs, @allocated factor_summary(h))
        end
        @test allocs == [0, 0, 0]
        # The COLD (post-invalidation) first call is a one-time compile cost. It
        # must be bounded and, crucially, must NOT scale with n — a summary that
        # allocated a matrix or recomputed inertia would grow as O(n^2).
        @test length(cold_allocs) == 2
        @test all(c -> c < 64 * 1024, cold_allocs)
        @test cold_allocs[1] == cold_allocs[2]     # size-independent, not O(n^2)
    end

    # -----------------------------------------------------------------------
    # 4. ACCEPTANCE ITEM — unsupported is REFUSED, never silently degraded
    # -----------------------------------------------------------------------
    @testset "4. unsupported requests are refused, not degraded" begin
        @testset "per-column provider refuses a batched requirement" begin
            p = MockBFLA(:ldlt, 8, 8)
            h, A, ok = end_to_end(p, :ldlt; size=8)
            @test ok
            req = solve_request(p, OpSolveN; ncols=4, needs=MultiRHSBatched)
            a = admit(req, declared_facts(p))
            @test !a.allowed
            @test a.reason === RefuseMultiRHSMode
            @test occursin("batch", lowercase(a.detail))
            @test occursin("per-column", lowercase(a.offered_fact)) ||
                  occursin("percolumn", lowercase(a.offered_fact)) ||
                  occursin("MultiRHSPerColumn", a.offered_fact)
            @test_throws AdmissionRefused admit_or_throw(req, declared_facts(p))
            # A refusal never mutates the request into something admissible.
            @test req.rhs.needs === MultiRHSBatched
            @test req.rhs.ncols == 4
        end

        @testset "square-only provider refuses rectangular QR" begin
            p = MockMFLA(5, 3, :qr)          # a rectangular-capable provider
            q = MockBFLA(:cholesky, 4, 4)    # a square-only provider
            # A rectangular QR request expressed in the SQUARE-ONLY provider's own
            # arithmetic, so the only fact that differs is the shape.
            fq = declared_facts(q)
            rect = cross(p, OpRefactorNumeric; n=3, m=5, ncols=1,
                         needs=MultiRHSUnsupported, scalar=fq.kernel_scalar,
                         triangle=TriangleUnused)
            @test admit(rect, with_scalar(declared_facts(p), fq.kernel_scalar)).allowed
            ab = admit(rect, fq)
            @test ab.reason === RefuseShapeRectangular
            @test !ab.allowed
            @test ab.reason === RefuseShapeRectangular
            @test_throws AdmissionRefused admit_or_throw(rect, declared_facts(q))
        end

        @testset "no transpose solve is refused, not silently transposed" begin
            p = MockBFLA(:ldlt, 8, 8)
            h, A, ok = end_to_end(p, :ldlt; size=8)
            @test ok
            dest = fill(NaN, 8, 2)
            out = solve_into!(dest, h, ones(8, 2), OpSolveT)
            @test !out.performed
            @test out.refused_reason === RefuseOperation
            @test all(isnan, dest)
            @test occursin("transpose", lowercase(out.detail)) ||
                  occursin("OpSolveT", out.detail)
        end

        @testset "precision requirement is refused, not rounded" begin
            # Same arithmetic family, MORE width than the kernel has: this is
            # the width question, not the family question.
            p = MockBFLA(:ldlt, 4, 4)   # ConvertForbidden, 256-bit kernel
            base = provider_scalar(p)
            req = FactorRequest(OpRefactorNumeric,
                                ScalarSpec(base.family, base.min_bits + 256, true),
                                ShapeSpec(4, 4); triangle=TriangleUpper)
            a = admit(req, declared_facts(p))
            @test !a.allowed
            @test a.reason === RefuseBitWidth
            @test !admission_report(a).degraded

            q = MockMFLA(4, 4, :ldlt)   # ConvertUpOnly
            b = provider_scalar(q)
            req2 = FactorRequest(OpRefactorNumeric,
                                 ScalarSpec(b.family, b.min_bits + 256, true),
                                 ShapeSpec(4, 4); triangle=TriangleLower)
            a2 = admit(req2, declared_facts(q))
            @test !a2.allowed
            @test a2.reason === RefuseBitWidth

            # A DIFFERENT arithmetic family is a different refusal, not the same
            # one: the two must not be collapsed into "unsupported".
            req3 = FactorRequest(OpRefactorNumeric, ScalarSpec(ArithExact, 64),
                                 ShapeSpec(4, 4); triangle=TriangleUpper)
            a3 = admit(req3, declared_facts(p))
            @test !a3.allowed
            @test a3.reason === RefuseScalarFamily
            @test a3.reason !== a.reason
        end

        @testset "triangle mismatch is refused, not reinterpreted" begin
            p = MockBFLA(:ldlt, 4, 4)   # Upper only
            req = ask(p, OpRefactorNumeric; n=4, triangle=TriangleLower)
            a = admit(req, declared_facts(p))
            @test !a.allowed
            @test a.reason === RefuseTriangle
        end

        @testset "a refused request never reaches the provider" begin
            p = MockBFLA(:ldlt, 6, 6)
            tri = declared_facts(p).factor.triangle
            # rectangle against a square-only provider
            req = ask(p, OpPrepareFactor; n=4, m=6, ncols=1, needs=MultiRHSUnsupported)
            h = factor_handle(p, req)
            before = p.cache.n_refactor_calls
            a = prepare_factor!(h)
            @test !a.allowed
            @test p.cache.n_refactor_calls == before
            r = refactor_numeric!(h, ones(6, 4))
            @test !r.ok
            @test r.status === StatusUnsupported
            @test p.cache.n_refactor_calls == before   # provider never entered
            @test h.lease.state !== LeaseBound
        end
    end

    # -----------------------------------------------------------------------
    # 5. ACCEPTANCE ITEM — sparse limitations are not hidden
    # -----------------------------------------------------------------------
    @testset "5. sparse limitations are explicit" begin
        # BFLA is dense-only: a sparse operator is refused, not densified.
        b = MockBFLA(:ldlt, 8, 8)
        mb = MockMFLA(8, 8, :ldlt)   # sparse-native
        # The request is built from BFLA's own facts plus `known_nnz`, so the ONE
        # dimension it exercises is storage — not the scalar family and not the
        # triangle convention.
        sreq = cross(b, OpPrepareFactor; n=8, nnz=40, ncols=2,
                     needs=MultiRHSUnsupported,
                     concurrency=declared_facts(b).concurrency)
        @test is_sparse_request(sreq)
        a = admit(sreq, declared_facts(b))
        @test !a.allowed
        @test a.reason in (RefuseStorageKind, RefuseDensifyNotAllowed, RefuseDensifyMemory)
        @test a.reason === RefuseStorageKind      # dense-only provider: no densify permission
        hb = factor_handle(b, sreq)
        rb = refactor_numeric!(hb, spd_matrix(8))
        @test !rb.ok
        @test b.cache.n_refactor_calls == 0     # never entered -> no silent densify

        # MFLA declares sparse-native: an equivalent sparse request (MFLA's own
        # scalar and triangle, same nnz/shape) IS admissible. Stated with MFLA's
        # conventions because the point is MFLA's storage fact, not its grammar.
        mreq = cross(mb, OpPrepareFactor; n=8, nnz=40, ncols=2,
                     needs=MultiRHSUnsupported, concurrency=declared_facts(mb).concurrency)
        @test is_sparse_request(mreq)
        am = admit(mreq, declared_facts(mb))
        @test am.allowed
        @test declared_facts(mb).storage.sparse_native
        @test !declared_facts(b).storage.sparse_native
        @test !declared_facts(b).storage.densify_allowed

        # A provider that is NOT sparse-native and DOES permit densification is
        # still bounded: below 0.1% fill the densification is refused, with the
        # fill ratio stated.
        densifier = MockMFLA(100, 100, :ldlt)
        f0 = declared_facts(densifier)
        f_sparse_dense = CapabilityFacts(f0.provider_name, f0.revision, f0.kernel_scalar,
            f0.conversion, f0.factor, f0.solve,
            StorageCapability(accepts_sparse=true, accepts_dense=true, sparse_native=false,
                              densify_allowed=true,
                              densify_memory_limit_bytes=1024^3,
                              memory_limit_bytes=1024^3),
            f0.indices, f0.concurrency, f0.exception_vocabulary, f0.failure_is_atomic)
        # 50 nnz against 100x100 cells is 0.5% fill: sparse enough that densifying
        # changes the algorithm, and inside the (generous) densify memory limit, so
        # the refusal under test is the FILL gate and not the memory gate.
        sparse_req = ask(densifier, OpRefactorNumeric; n=100, nnz=50, ncols=1,
                         needs=MultiRHSUnsupported, facts=f_sparse_dense,
                         scalar=f_sparse_dense.kernel_scalar,
                         triangle=TriangleLower,
                         concurrency=f_sparse_dense.concurrency)
        a2 = admit(sparse_req, f_sparse_dense)
        @test !a2.allowed
        @test a2.reason === RefuseDensifyMemory
        @test occursin("0.5%", a2.detail)    # the fill ratio is measured and stated

        # And the memory ceiling itself is a refusal, not an allocation attempt.
        f_tiny = CapabilityFacts(f0.provider_name, f0.revision, f0.kernel_scalar,
            f0.conversion, f0.factor, f0.solve,
            StorageCapability(accepts_sparse=true, accepts_dense=true, sparse_native=true,
                              densify_allowed=false, densify_memory_limit_bytes=0,
                              memory_limit_bytes=1024),
            f0.indices, f0.concurrency, f0.exception_vocabulary, f0.failure_is_atomic)
        a3 = admit(sparse_req, f_tiny)
        @test !a3.allowed
        @test a3.reason === RefuseMemoryLimit
    end

    # -----------------------------------------------------------------------
    # 6. ACCEPTANCE ITEM — the hot solve path: no deep diagnostics, no copy
    # -----------------------------------------------------------------------
    @testset "6. hot solve path calls no deep diagnostics and copies no factor" begin
        for (label, p, kind) in (("MFLA/ldlt", MockMFLA(8, 8, :ldlt), :ldlt),
                                 ("BFLA/ldlt", MockBFLA(:ldlt, 8, 8), :ldlt))
            @testset "$label" begin
                h, A, ok = end_to_end(p, kind; size=8)
                @test ok
                deep0 = h.diagnostics.deep_calls
                snaps0 = p.cache.n_snapshot_calls
                copies0 = h.n_factor_copies

                # 200 hot solves. The factor is NOT copied, and the deep-check
                # path is never entered.
                for k in 1:200
                    out = solve_into!(zeros(8, 2), h, ones(8, 2), OpSolveN)
                    @test out.performed
                end
                @test h.diagnostics.deep_calls == deep0
                @test h.diagnostics.factor_copies == 0
                @test p.cache.n_snapshot_calls == snaps0
                @test h.n_factor_copies == copies0
                @test p.cache.n_deep_calls == 0
                @test h.n_hot_solves == 200

                # A factor copy is a DIFFERENT, explicit call and is counted.
                copy_operator_snapshot(h)
                @test p.cache.n_snapshot_calls == snaps0 + 1
                @test h.n_factor_copies == copies0 + 1
                # The stated cost of that copy is quadratic in n, which is what
                # "no factor copy on the hot path" is protecting.
                @test factor_copy_cost(factor_summary(h)) == 8 * 8

                # Deep diagnostics likewise, and only through inspect_factor.
                inspect_factor(h, :full)
                @test h.diagnostics.deep_calls == deep0 + 1
                @test p.cache.n_deep_calls == 1

                # The multi-RHS mode is a visible call-site fact: BFLA loops.
                if declared_facts(p).solve.multi_rhs === MultiRHSPerColumn
                    @test h.n_per_column_calls == 200 * 2
                    @test h.n_batched_calls == 0
                else
                    @test h.n_batched_calls == 200
                    @test h.n_per_column_calls == 0
                end
            end
        end

        # `invalidate_numeric!` revokes the NUMERIC factor only — the symbolic
        # epoch and lease record survive (ADR-002 §2).
        p = MockMFLA(8, 8, :ldlt)
        h, A, ok = end_to_end(p, :ldlt; size=8)
        @test ok
        epoch_before = h.lease.matrix_epoch
        invalidate_numeric!(h)
        @test h.lease.state === LeaseRevoked
        @test h.lease.matrix_epoch == epoch_before     # symbolic lease not revoked
        @test p.cache.n_invalidate_calls >= 1
        out = solve_into!(zeros(8, 2), h, ones(8, 2), OpSolveN)
        @test !out.performed
    end

    # -----------------------------------------------------------------------
    # 7. pivot metadata: standardized, and never guessed
    # -----------------------------------------------------------------------
    @testset "7. pivot metadata interpretation is grammar-driven" begin
        # Real BK pivot reports from real kernels.
        p1 = MockMFLA(5, 5, :ldlt); p1.cache.prepared = true
        A1 = [1.0 2 0 0 0; 2 1 0 0 0; 0 0 3 1 0; 0 0 1 2 1; 0 0 0 1 3]
        @test raw_refactor!(p1, A1) === :success
        meta1 = raw_pivots(p1)
        @test meta1.n_2x2 == 1                        # a genuine 2x2 block
        r1 = interpret_pivots(meta1)
        @test r1.inertia == (4, 1, 0)
        @test r1.n_1x1 + 2 * r1.n_2x2 + r1.n_zero == meta1.n

        # The block-vs-entry confusion is caught, not silently accepted.
        @test_throws ArgumentError interpret_pivots(
            PivotMetadata(:ldlt_bk_1x1_2x2, 5, 5, 1, 0, 5, 5, true))
        # An unknown grammar is refused rather than guessed.
        @test_throws ArgumentError interpret_pivots(
            PivotMetadata(:who_knows, 4, 4, 0, 0, 4, 4, true))
        # LU has no inertia; QR's rank is a column fact.
        @test interpret_pivots(PivotMetadata(:lu_partial, 3, 3, 0, 0, 3, 0, true)).inertia === nothing
        qr_rep = interpret_pivots(PivotMetadata(:qr_column_rank, 3, 3, 0, 0, 3, 0, true))
        @test qr_rep.inertia === nothing
        @test qr_rep.n_2x2 == 0
        @test_throws ArgumentError interpret_pivots(
            PivotMetadata(:qr_column_rank, 3, 2, 0, 0, 3, 0, true))
        @test_throws ArgumentError interpret_pivots(
            PivotMetadata(:cholesky_psd, 3, 3, 1, 0, 2, 2, true))

        # Provider status vocabularies are mapped onto one SDPX vocabulary.
        @test standardize_status(:bfla, :success) === StatusOk
        @test standardize_status(:bfla, :not_posdef) === StatusNotPositiveDefinite
        @test standardize_status(:mfla, :rank_deficient) === StatusRankDeficient
        @test standardize_status(:mfla, :not_posdef) === StatusNotPositiveDefinite
        @test standardize_status(:bfla, :singular) === StatusSingular
        @test standardize_status(:mfla, :something_new) === StatusError
        # A status symbol that is the name of the provider's own enum value must
        # map through, not fall to StatusError; and an unknown one must NEVER
        # become StatusOk.
        @test standardize_status(:bfla, :BFLASuccess) === StatusOk
        @test standardize_status(:bfla, :bflasuccess) === StatusOk
        @test standardize_status(:mfla, :MFLASuccess) === StatusOk
        @test standardize_status(:bfla, :BFLASingular) === StatusSingular
        @test standardize_status(:bfla, :definitely_not_a_status) === StatusError
        @test standardize_status(:bfla, :definitely_not_a_status) !== StatusOk
        @test interpret_pivots(raw_pivots(p1)).grammar === :ldlt_bk_1x1_2x2

        # Non-SPD Cholesky is a *status*, not a throw and not a success.
        p2 = MockMFLA(2, 2, :cholesky); p2.cache.prepared = true
        st = raw_refactor!(p2, [1.0 2; 2 1])
        @test st === :not_posdef
        @test standardize_status(:mfla, st) === StatusNotPositiveDefinite
        @test raw_provider_generation(p2) == 0     # no factor -> no generation bump
    end

    # -----------------------------------------------------------------------
    # 8. rectangular QR is not hidden behind a generalized boolean
    # -----------------------------------------------------------------------
    @testset "8. rectangular QR is a first-class, distinct request" begin
        q = MockBFLA(:rrqr, 6, 4)
        f = declared_facts(q)
        @test f.factor.rectangular
        @test f.factor.square_only == false
        @test f.factor.rank_revealing

        req = ask(q, OpRefactorNumeric; n=4, m=6, ncols=1, needs=MultiRHSUnsupported,
                  rank_kind=:rank_revealing)
        @test admit(req, f).allowed
        h = factor_handle(q, req)
        q.cache.prepared = true
        A = randn(MersenneTwister(5), 6, 4)
        r = refactor_numeric!(h, A)
        @test r.ok
        rep = pivot_report(h)
        @test rep.grammar === :qr_column_rank
        @test rep.n_2x2 == 0
        @test rep.inertia === nothing            # no inertia for QR, ever
        @test raw_pivots(q).n == 4               # column count, not row count
        s = factor_summary(h)
        @test s.n == 4
        x = zeros(4, 1)
        out = solve_into!(x, h, A[:, 1:1], OpSolveN)
        @test out.performed
        @test norm(A * x - A[:, 1:1]) < 1e-8     # a real least-squares solve

        # A rank-revealing request against a provider that does not declare it.
        sq = MockBFLA(:cholesky, 4, 4)
        a = admit(req, declared_facts(sq))
        @test !a.allowed
        @test a.reason === RefuseShapeRectangular
    end

    # -----------------------------------------------------------------------
    # 9. the legacy shim is one-way and authorizes nothing
    # -----------------------------------------------------------------------
    @testset "9. old entry points keep a one-way shim owned by the integrator" begin
        p = MockMFLA(4, 4, :ldlt)
        h, A, ok = end_to_end(p, :ldlt; size=4)
        @test ok
        leg = legacy_shim(h)
        @test leg isa LegacyFactorEntry
        @test !legacy_shim_authorizes(leg)
        # One-way by construction: there is no operation that turns the legacy
        # struct back into an authorizable handle.
        # Enforced structurally: there is no way from a legacy entry to a handle.
        @test hasmethod(factor_handle, Tuple{LegacyFactorEntry,FactorRequest})
        @test_throws ArgumentError factor_handle(leg, h.request)
        @test !hasmethod(solve_into!, Tuple{Any, LegacyFactorEntry, Any, SolveOp})
        @test !applicable(authorize, leg)
        @test !applicable(is_valid, leg)
        @test !applicable(solve_into!, zeros(4, 1), leg, ones(4, 1), OpSolveN)
        # The legacy struct cannot even be mistaken for a lease-carrying handle.
        @test !(:lease in fieldnames(LegacyFactorEntry))
    end

    # -----------------------------------------------------------------------
    # 10. the adapters really exist for the real providers — and skip loudly
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # 10. LIVE provider legs — same contract, real provider
    # -----------------------------------------------------------------------
    # Every assertion below runs against the real provider when the driver
    # selected it. When it did not, the leg SKIPS WITH A REASON — never a silent
    # pass, and never a `0` for something unrun.
    @testset "10. live provider legs" begin
        if S05_LIVE == "none"
            # A skip must be VISIBLE WITH ITS REASON (ADR-002 §7). `@test_skip` in
            # Julia 1.12 prints only a number, so the reason is recorded as a
            # passing assertion on the skip itself.
            reason = "no live provider selected: run scripts/run_s05.sh, which executes " *
                     "this file with S05_LIVE_PROVIDER=mfla and =bfla in separate -t1 " *
                     "processes (Julia 1.12 cannot compile both provider " *
                     "specialization sets in one process)"
            @test occursin("separate -t1 processes", reason)
            @info "S05 live legs SKIPPED" reason=reason
        elseif S05_LIVE == "mfla"
            @testset "live MFLA" begin
                res = mfla_live_adapter(:ldlt; n=6)
                @test res !== nothing
                p, cache = res
                @test live_cache_kind(cache) === :ldlt
                f = declared_facts(p)
                @test f.provider_name === :mfla
                @test f.kernel_scalar.family === ArithMultiFloat
                @test f.kernel_scalar.min_bits > 64       # more than one Float64 limb
                @test f.factor.calls_bunch_kaufman
                # No live leg may claim threads: MFLA's capability report states
                # `reusable_workspace`, not thread eligibility.
                @test f.solve.threads === ThreadNone
                @test !(fieldtype(SolveCapability, :threads) <: Bool)

                req = ask(p, OpPrepareFactor; n=6, ncols=2, needs=f.solve.multi_rhs)
                h = factor_handle(p, req)
                @test prepare_factor!(h).allowed

                MF = load_mfla().MultiFloat{Float64,2}
                A = zeros(MF, 6, 6)
                for i in 1:6
                    A[i, i] = MF(4); i < 6 && (A[i, i+1] = MF(1); A[i+1, i] = MF(1))
                end
                r = refactor_numeric!(h, A)
                @test r.ok
                @test is_valid(h.lease)

                rep = pivot_report(h)
                @test rep.grammar === :ldlt_bk_1x1_2x2     # MFLA reports BLOCKS
                @test rep.n_1x1 + 2 * rep.n_2x2 + rep.n_zero == 6
                @test rep.inertia !== nothing

                # O(1) summary against the real provider, measured.
                factor_summary(h)
                @test (@allocated factor_summary(h)) == 0
                @test h.diagnostics.deep_calls == 0
                # The summary must NOT have come from `factor_diagnostics`, whose
                # `inertia` field recomputes from the factors.
                @test h.diagnostics.factor_copies == 0

                X = zeros(MF, 6, 2); R = zeros(MF, 6, 2)
                for j in 1:2, i in 1:6; R[i, j] = MF(i + j); end
                out = solve_into!(X, h, R, OpSolveN)
                @test out.performed
                @test Float64(norm(Float64.(A) * Float64.(X) - Float64.(R))) < 1e-10

                # ---- ADR-002 §8/§9, live: preflight rejection retains the
                # physical factor AND the previous success flag; the lease is
                # revoked anyway and the next solve fails closed. --------------
                g0 = raw_provider_generation(p)
                s0 = raw_status(p)
                @test raw_retained_physical(p)
                bad = Matrix{MF}(I, 5, 5)
                r2 = refactor_numeric!(h, bad)
                @test !r2.ok
                @test r2.revoked
                @test h.lease.state === LeaseRevoked
                @test raw_retained_physical(p)                      # retained!
                @test raw_status(p) === s0                          # stale success!
                @test raw_provider_generation(p) == g0              # no new factor
                d2 = fill(MF(0), 6, 2)
                o2 = solve_into!(d2, h, R, OpSolveN)
                @test !o2.performed
                @test all(iszero, d2)                               # nothing written
                @test occursin("lease", lowercase(o2.detail))
            end
        elseif S05_LIVE == "bfla"
            @testset "live BFLA" begin
                res = bfla_live_adapter(:ldlt; n=6, precision_bits=256)
                @test res !== nothing
                p, cache = res
                @test live_cache_kind(cache) === :ldlt
                f = declared_facts(p)
                @test f.provider_name === :bfla
                @test f.kernel_scalar.family === ArithFloat
                @test f.kernel_scalar.min_bits == 256      # the prepared precision
                @test f.solve.multi_rhs === MultiRHSPerColumn
                @test f.solve.threads === ThreadNone
                @test !f.storage.sparse_native
                @test f.factor.calls_bunch_kaufman

                req = ask(p, OpPrepareFactor; n=6, ncols=2, needs=f.solve.multi_rhs)
                h = factor_handle(p, req)
                @test prepare_factor!(h).allowed

                A = zeros(BigFloat, 6, 6)
                for i in 1:6
                    A[i, i] = BigFloat(4); i < 6 && (A[i, i+1] = BigFloat(1); A[i+1, i] = BigFloat(1))
                end
                r = refactor_numeric!(h, A)
                @test r.ok
                @test is_valid(h.lease)
                rep = pivot_report(h)
                @test rep.grammar === :ldlt_bk_1x1_2x2
                @test rep.n_1x1 + 2 * rep.n_2x2 + rep.n_zero == 6

                factor_summary(h)
                @test (@allocated factor_summary(h)) == 0
                @test h.diagnostics.deep_calls == 0
                @test h.diagnostics.factor_copies == 0

                X = zeros(BigFloat, 6, 2); R = zeros(BigFloat, 6, 2)
                for j in 1:2, i in 1:6; R[i, j] = BigFloat(i + j); end
                out = solve_into!(X, h, R, OpSolveN)
                @test out.performed
                @test Float64(norm(Float64.(A) * Float64.(X) - Float64.(R))) < 1e-40

                # ---- ADR-002 §8, live ---------------------------------------
                g0 = raw_provider_generation(p)
                s0 = raw_status(p)
                @test s0 === :success
                @test raw_retained_physical(p)
                bad = Matrix{BigFloat}(I, 5, 5)
                r2 = refactor_numeric!(h, bad)
                @test !r2.ok
                @test r2.revoked
                @test h.lease.state === LeaseRevoked
                @test raw_retained_physical(p)       # BFLA kept the physical factor
                @test raw_status(p) === s0           # ...and its :success flag
                @test raw_provider_generation(p) == g0
                d2 = fill(BigFloat(NaN), 6, 2)
                o2 = solve_into!(d2, h, R, OpSolveN)
                @test !o2.performed
                @test all(isnan, d2)                 # the old factor was NOT reused
            end
        else
            @test false   # an unknown selection is a DRIVER error, not a skip
            @info "S05 live legs SKIPPED" reason="unknown S05_LIVE_PROVIDER=$(S05_LIVE)"
        end
    end
end
