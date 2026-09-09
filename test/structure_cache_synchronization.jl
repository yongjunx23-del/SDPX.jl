# Structure-cache synchronization tests (suite-included and standalone).
#
# Included by `test/runtests.jl`; it can also run against an isolated env copy:
#
#   JULIA_NUM_THREADS=2 JULIA_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     julia --heap-size-hint=2G --project=<isolated-env> \
#       test/structure_cache_synchronization.jl
#
# Scope: read/write consistency of the process-global symmetric-core
# structure cache (`src/factor_cache/structure_cache.jl` and the lookup /
# publication sections of `src/kkt/symmetric_core.jl` only).  No factor
# policy, numeric, precision, or provider behavior is changed or claimed
# here; the concurrency tests assert correctness invariants only, never
# speedups.

using Test
using SDPX
using SparseArrays
using LinearAlgebra

# Evidence-grade provenance: printed INTO the log by the test process itself.
println("provenance: pkgdir=", Base.pkgdir(SDPX))
println("provenance: worktree HEAD=",
    chomp(read(`git -C $(Base.pkgdir(SDPX)) rev-parse HEAD`, String)))
println("provenance: Threads.nthreads()=", Base.Threads.nthreads())
println("provenance: BLAS threads=", LinearAlgebra.BLAS.get_num_threads())

const _SYNC_M = 8
const _SYNC_NR = 4
const _SYNC_RANGES = [1:3, 4:8]
const _SYNC_SHAPES = [:dense_lower, :dense_lower]

"""Deterministic small `Ar` fixture (`m × nr`, fixed sparsity pattern)."""
function _sync_Ar(::Type{T}, m::Int, nr::Int; shift::Int=0) where {T}
    rows = Int[]
    cols = Int[]
    vals = T[]
    for j in 1:nr
        for k in 0:1
            i = mod(j * 2 + k + shift - 1, m) + 1
            push!(rows, i)
            push!(cols, j)
            push!(vals, T(0.5) + T(0.1) * T(j) + T(0.01) * T(k))
        end
    end
    return sparse(rows, cols, vals, m, nr)
end

_sync_build(::Type{T}, Ar; ranges=_SYNC_RANGES, shapes=_SYNC_SHAPES) where {T} =
    SDPX.SymmetricCorePattern{T}(Ar, ranges, shapes)

function _sync_with_fresh_cache(f)
    SDPX.set_structure_cache_enabled!(true)
    SDPX.clear_structure_cache!()
    try
        return f()
    finally
        SDPX.set_structure_cache_enabled!(true)
        SDPX.clear_structure_cache!()
    end
end

@testset "disabled cache bypasses lookup, counters, and publication" begin
    _sync_with_fresh_cache() do
        base = SDPX.structure_cache_stats()
        @test base.enabled
        @test base.hits == 0 && base.misses == 0 && base.entries == 0

        SDPX.set_structure_cache_enabled!(false)
        @test SDPX.structure_cache_stats().enabled == false

        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR)
        p1 = _sync_build(Float64, Ar)
        st1 = SDPX.structure_cache_stats()
        @test st1.entries == 0
        @test st1.hits == 0 && st1.misses == 0

        # A second disabled build of the same structure is a fully owned
        # rebuild: same signature, but no shared arrays and still no stats.
        p2 = _sync_build(Float64, sparse(Ar))
        @test SDPX.symmetric_core_signature(p2) == SDPX.symmetric_core_signature(p1)
        @test p2.colptr !== p1.colptr
        @test p2.rowval !== p1.rowval
        @test all(iszero, p2.nzval)
        st2 = SDPX.structure_cache_stats()
        @test st2.entries == 0
        @test st2.hits == 0 && st2.misses == 0

        # Re-enabling repopulates on the next miss.
        SDPX.set_structure_cache_enabled!(true)
        p3 = _sync_build(Float64, sparse(Ar))
        st3 = SDPX.structure_cache_stats()
        @test st3.enabled
        @test st3.misses == 1 && st3.hits == 0 && st3.entries == 1
        @test p3.colptr !== p1.colptr  # rebuilt, not shared with disabled era
    end
end

@testset "clear drops entries and resets hit/miss counters" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR)
        p1 = _sync_build(Float64, Ar)
        @test SDPX.structure_cache_stats().misses == 1
        p2 = _sync_build(Float64, sparse(Ar))
        st = SDPX.structure_cache_stats()
        @test st.hits == 1 && st.misses == 1 && st.entries == 1
        @test p2.colptr === p1.colptr

        SDPX.clear_structure_cache!()
        cleared = SDPX.structure_cache_stats()
        @test cleared.entries == 0
        @test cleared.hits == 0 && cleared.misses == 0

        # Post-clear build is an owned rebuild, not a resurrection of the
        # dropped arrays.
        p3 = _sync_build(Float64, sparse(Ar))
        @test SDPX.structure_cache_stats().misses == 1
        @test p3.colptr !== p1.colptr
        @test SDPX.symmetric_core_signature(p3) == SDPX.symmetric_core_signature(p1)
    end
end

@testset "hit shares frozen structure and owns a fresh zero numeric buffer" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR)
        p1 = _sync_build(Float64, Ar)
        p2 = _sync_build(Float64, sparse(Ar))
        @test SDPX.structure_cache_stats().hits == 1
        for field in (:colptr, :rowval, :ar_slots, :theta_slots, :x_diag_slots)
            @test getproperty(p2, field) === getproperty(p1, field)
        end
        @test p2.nzval !== p1.nzval
        @test all(iszero, p1.nzval)
        @test all(iszero, p2.nzval)

        # Numeric workspace isolation through the real refill path: writing
        # `-Theta` into the hit pattern must not leak into the first owner
        # or into later hits.
        Theta = Matrix{Float64}(I, _SYNC_M, _SYNC_M)
        SDPX.refill!(p2, sparse(Ar), Theta)
        @test all(iszero, p1.nzval)
        @test any(!iszero, p2.nzval)
        p3 = _sync_build(Float64, sparse(Ar))
        @test all(iszero, p3.nzval)
        @test p3.colptr === p1.colptr

        # Key discrimination is unchanged: a different arithmetic type and a
        # different sparsity pattern both miss.
        before = SDPX.structure_cache_stats()
        pf32 = _sync_build(Float32, _sync_Ar(Float32, _SYNC_M, _SYNC_NR))
        after_type = SDPX.structure_cache_stats()
        @test after_type.misses == before.misses + 1
        @test pf32.colptr !== p1.colptr
        pshift = _sync_build(Float64, _sync_Ar(Float64, _SYNC_M, _SYNC_NR; shift=1))
        after_shift = SDPX.structure_cache_stats()
        @test after_shift.misses == after_type.misses + 1
        @test SDPX.symmetric_core_signature(pshift) != SDPX.symmetric_core_signature(p1)
    end
end

@testset "disable suppresses in-flight publication (gated publish)" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR)
        p_seed = _sync_build(Float64, Ar)  # miss=1; entries == 1
        @test SDPX.structure_cache_stats().entries == 1
        key = (Float64, SDPX.symmetric_core_signature(p_seed))
        entry = (
            a_colptr=copy(p_seed.ar_colptr),
            a_rowval=copy(p_seed.ar_rowval),
            block_ranges=UnitRange{Int}[r for r in _SYNC_RANGES],
            block_shapes=Symbol[s for s in _SYNC_SHAPES],
            colptr=copy(p_seed.colptr),
            rowval=copy(p_seed.rowval),
            ar_slots=copy(p_seed.ar_slots),
            theta_slots=copy(p_seed.theta_slots),
            x_diag_slots=copy(p_seed.x_diag_slots),
        )
        # A fresh eligible token for the seeded key (also records a hit).
        _, eligible0, token0 = SDPX._structure_cache_lookup_and_account!(key)
        @test eligible0
        @test token0 isa UInt64

        # Disabling atomically drops the seeded entry.
        SDPX.set_structure_cache_enabled!(false)
        @test SDPX.structure_cache_stats().entries == 0

        # Stale token plus disabled flag: no resurrection.
        @test SDPX._structure_cache_try_publish!(key, entry, token0) == false
        @test SDPX.structure_cache_stats().entries == 0

        # Re-enable does NOT revive the stale token: the disable advanced
        # the invalidation generation, so the old token mismatches.
        SDPX.set_structure_cache_enabled!(true)
        @test SDPX._structure_cache_try_publish!(key, entry, token0) == false
        @test SDPX.structure_cache_stats().entries == 0

        # A post-re-enable lookup mints a live token; its publication wins
        # exactly once (first-wins), and the next real build is a hit.
        _, eligible1, token1 = SDPX._structure_cache_lookup_and_account!(key)
        @test eligible1
        @test token1 != token0
        @test SDPX._structure_cache_try_publish!(key, entry, token1) == true
        @test SDPX.structure_cache_stats().entries == 1
        @test SDPX._structure_cache_try_publish!(key, entry, token1) == false
        p_hit = _sync_build(Float64, sparse(Ar))
        st = SDPX.structure_cache_stats()
        @test st.hits == 2 && st.misses == 2 && st.entries == 1
        @test all(iszero, p_hit.nzval)
    end
end

@testset "concurrent same-structure builds account exactly once per build" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR)
        n_tasks = 4
        n_per_task = 25
        total = n_tasks * n_per_task
        signer = zeros(UInt64, total)
        allzero_ok = zeros(Bool, total)
        @sync for t in 1:n_tasks
            Base.Threads.@spawn begin
                for k in 1:n_per_task
                    idx = (t - 1) * n_per_task + k
                    p = _sync_build(Float64, sparse(Ar))
                    signer[idx] = SDPX.symmetric_core_signature(p)
                    allzero_ok[idx] = all(iszero, p.nzval)
                end
            end
        end
        @test all(==(signer[1]), signer)
        @test all(allzero_ok)
        st = SDPX.structure_cache_stats()
        @test st.entries == 1
        # Exactly one accounting event per build: no lost counter updates.
        @test st.hits + st.misses == total
        @test st.misses >= 1
        # A post-join build is a clean hit on the single cached entry.
        hits_before = st.hits
        p_last = _sync_build(Float64, sparse(Ar))
        @test SDPX.structure_cache_stats().hits == hits_before + 1
        @test all(iszero, p_last.nzval)
    end
end

@testset "concurrent toggle/build/read stays consistent" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR)
        Ar_b = _sync_Ar(Float64, _SYNC_M, _SYNC_NR; shift=1)
        n_builders = 3
        builds_each = 30
        # Builders hammer two structures while a toggler flips the flag,
        # clears, and reads stats.  Only invariant checks run concurrently;
        # determinism is established by the ordered tail below.
        @sync begin
            for _ in 1:n_builders
                Base.Threads.@spawn begin
                    for k in 1:builds_each
                        p = isodd(k) ? _sync_build(Float64, sparse(Ar)) :
                            _sync_build(Float64, sparse(Ar_b))
                        @assert length(p.nzval) > 0 && length(p.colptr) > 0
                        @assert all(iszero, p.nzval)
                    end
                end
            end
            Base.Threads.@spawn begin
                for _ in 1:60
                    SDPX.set_structure_cache_enabled!(true)
                    st = SDPX.structure_cache_stats()
                    @assert st.entries >= 0 && st.hits >= 0 && st.misses >= 0
                    SDPX.set_structure_cache_enabled!(false)
                    @assert SDPX.structure_cache_stats().entries == 0
                    SDPX.clear_structure_cache!()
                end
            end
        end
        # Ordered deterministic tail: after the storm, a final clear leaves
        # zero entries; a disabled build publishes nothing; re-enable + two
        # same-structure builds give exactly one miss then one hit sharing
        # the frozen arrays.
        SDPX.set_structure_cache_enabled!(false)
        SDPX.clear_structure_cache!()
        tail0 = SDPX.structure_cache_stats()
        @test tail0.entries == 0
        pd = _sync_build(Float64, sparse(Ar))
        @test SDPX.structure_cache_stats().entries == 0
        SDPX.set_structure_cache_enabled!(true)
        SDPX.clear_structure_cache!()
        q1 = _sync_build(Float64, sparse(Ar))
        q2 = _sync_build(Float64, sparse(Ar))
        tail = SDPX.structure_cache_stats()
        @test tail.misses == 1 && tail.hits == 1 && tail.entries == 1
        @test q2.colptr === q1.colptr
        @test q2.nzval !== q1.nzval
        @test pd.colptr !== q1.colptr
    end
end

# Channel-rendezvous regression for the lookup/build/publish protocol.
#
# Each test below drives the PRIVATE production-used helper
# `_structure_cache_lookup_build_publish!` (the only production caller is
# `SymmetricCorePattern` construction) with closures that rendezvous on
# Channels: the structure closure signals entry (which proves its lookup
# already returned, since the helper invokes it only post-lookup) and then
# blocks pre-publication while the main task invalidates.  No sleeps,
# timestamps, or counter polling are used as proof anywhere in this file.
# Constructor integration is verified behaviorally in the testsets above
# (real miss publishes, real hit shares frozen arrays with a fresh numeric
# buffer, real disabled build publishes nothing) and in the integration
# tail of the first rendezvous test below, which replays the same key
# through the real constructor.
_fake_protocol_built() = (
    colptr=[1], rowval=Int[], ar_slots=Int[], theta_slots=Int[],
    x_diag_slots=Int[], nnz=0,
)
_fake_protocol_entry() = (
    a_colptr=[1], a_rowval=Int[],
    block_ranges=UnitRange{Int}[1:1], block_shapes=[:dense_lower],
    colptr=[1], rowval=Int[], ar_slots=Int[], theta_slots=Int[],
    x_diag_slots=Int[],
)
_sync_protocol_key(::Type{T}, Ar) where {T} = (T,
    SDPX._symmetric_core_structure_signature(
        size(Ar, 2), size(Ar, 1), Ar.colptr, Ar.rowval,
        _SYNC_RANGES, _SYNC_SHAPES,
    ))

@testset "blocked eligible build rejects stale publication across disable/clear/re-enable" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR; shift=601)
        key = _sync_protocol_key(Float64, Ar)
        entered = Channel{Bool}(1)
        release = Channel{Bool}(1)
        payload_ran = Ref(false)
        task = Base.Threads.@spawn SDPX._structure_cache_lookup_build_publish!(
            key,
            function ()
                # Helper invoked us only post-lookup: this signal proves
                # the eligible lookup returned a miss and publication is
                # still pending.
                put!(entered, true)
                take!(release)  # block pre-publication
                return _fake_protocol_built()
            end,
            function (built)
                payload_ran[] = true
                return _fake_protocol_entry()
            end,
        )
        @test take!(entered)
        SDPX.set_structure_cache_enabled!(false)
        SDPX.clear_structure_cache!()
        SDPX.set_structure_cache_enabled!(true)
        put!(release, true)
        res = fetch(task)
        @test res.enabled === true  # the lookup itself was eligible
        @test res.hit === nothing
        # The eligible build assembled its payload, but the disable+clear
        # advanced the generation, so the stale token was rejected.
        @test payload_ran[]
        st = SDPX.structure_cache_stats()
        @test st.entries == 0 && st.misses == 0 && st.hits == 0

        # Integration tail: the same key through the REAL constructor
        # misses, publishes, then hits with shared structure and a fresh
        # numeric buffer — the constructor is wired to this protocol.
        SDPX.clear_structure_cache!()
        q1 = _sync_build(Float64, sparse(Ar))
        q2 = _sync_build(Float64, sparse(Ar))
        st2 = SDPX.structure_cache_stats()
        @test st2.misses == 1 && st2.hits == 1 && st2.entries == 1
        @test q2.colptr === q1.colptr
        @test q2.nzval !== q1.nzval && all(iszero, q2.nzval)
    end
end

@testset "blocked eligible build rejects stale publication across disable/re-enable" begin
    # Same rendezvous without the clear: the disable alone advances the
    # generation, and counters are preserved (disable never resets them).
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR; shift=602)
        key = _sync_protocol_key(Float64, Ar)
        entered = Channel{Bool}(1)
        release = Channel{Bool}(1)
        task = Base.Threads.@spawn SDPX._structure_cache_lookup_build_publish!(
            key,
            function ()
                put!(entered, true)
                take!(release)
                return _fake_protocol_built()
            end,
            function (built)
                return _fake_protocol_entry()
            end,
        )
        @test take!(entered)
        SDPX.set_structure_cache_enabled!(false)
        SDPX.set_structure_cache_enabled!(true)
        put!(release, true)
        res = fetch(task)
        @test res.enabled === true && res.hit === nothing
        st = SDPX.structure_cache_stats()
        @test st.misses == 1 && st.hits == 0 && st.entries == 0
    end
end

@testset "blocked eligible build rejects stale publication across bare clear" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR; shift=605)
        key = _sync_protocol_key(Float64, Ar)
        entered = Channel{Bool}(1)
        release = Channel{Bool}(1)
        payload_ran = Ref(false)
        task = Base.Threads.@spawn SDPX._structure_cache_lookup_build_publish!(
            key,
            function ()
                put!(entered, true)
                take!(release)
                return _fake_protocol_built()
            end,
            function (built)
                payload_ran[] = true
                return _fake_protocol_entry()
            end,
        )
        @test take!(entered)
        # No disable: this specifically depends on clear advancing the token.
        SDPX.clear_structure_cache!()
        put!(release, true)
        res = fetch(task)
        @test res.enabled === true && res.hit === nothing
        @test payload_ran[]
        st = SDPX.structure_cache_stats()
        @test st.enabled && st.entries == 0 && st.misses == 0 && st.hits == 0
    end
end

@testset "blocked disabled build constructs no payload and publishes nothing" begin
    # Exact P1-1 interleaving, channel-ordered: the disabled lookup
    # completes, the build blocks, the main task re-enables mid-build, and
    # the captured ineligibility still skips payload construction and
    # publication.  The pre-fix protocol (publish gated only on the live
    # flag) would store here and fail the entries check.
    _sync_with_fresh_cache() do
        SDPX.set_structure_cache_enabled!(false)
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR; shift=603)
        key = _sync_protocol_key(Float64, Ar)
        entered = Channel{Bool}(1)
        release = Channel{Bool}(1)
        structure_ran = Ref(false)
        payload_ran = Ref(false)
        task = Base.Threads.@spawn SDPX._structure_cache_lookup_build_publish!(
            key,
            function ()
                put!(entered, true)  # disabled lookup done; building
                take!(release)       # block; re-enable lands here
                structure_ran[] = true
                return _fake_protocol_built()
            end,
            function (built)
                payload_ran[] = true
                return _fake_protocol_entry()
            end,
        )
        @test take!(entered)
        SDPX.set_structure_cache_enabled!(true)
        put!(release, true)
        res = fetch(task)
        @test res.enabled === false
        @test res.hit === nothing && res.built !== nothing
        @test structure_ran[] && !payload_ran[]
        st = SDPX.structure_cache_stats()
        @test st.misses == 0 && st.hits == 0 && st.entries == 0
    end
end

@testset "hit path invokes neither build closure" begin
    _sync_with_fresh_cache() do
        Ar = _sync_Ar(Float64, _SYNC_M, _SYNC_NR; shift=604)
        p = _sync_build(Float64, Ar)  # real miss publishes a real entry
        key = _sync_protocol_key(Float64, Ar)
        structure_ran = Ref(false)
        payload_ran = Ref(false)
        res = SDPX._structure_cache_lookup_build_publish!(
            key,
            function ()
                structure_ran[] = true
                return _fake_protocol_built()
            end,
            function (built)
                payload_ran[] = true
                return _fake_protocol_entry()
            end,
        )
        @test res.enabled === true
        @test res.hit !== nothing && res.hit.colptr === p.colptr
        @test res.built === nothing
        @test !structure_ran[] && !payload_ran[]
    end
end

@testset "generation boundary fails closed on a local instance" begin
    # The 2^64 boundary is unreachable in-process by design; its rule is
    # pinned down deterministically on a LOCAL cache instance, so global
    # state is never polluted.  Boundary semantics under test: advancing
    # ONTO max succeeds and stays usable; only ANOTHER advance while
    # already at max fails closed (forced disabled, entries dropped,
    # counter pinned, never wraps).
    local_cache = SDPX.SymmetricCoreStructureCache()
    @test local_cache.generation == UInt64(0)
    local_cache.patterns[(Float64, UInt64(0x1234))] = 1
    local_cache.generation = typemax(UInt64) - UInt64(1)
    @test SDPX._structure_cache_advance_generation!(local_cache) ==
        typemax(UInt64)
    @test local_cache.enabled == true  # onto max: still usable
    @test SDPX._structure_cache_advance_generation!(local_cache) ==
        typemax(UInt64)  # advance AT max: pinned, never wraps
    @test local_cache.enabled == false  # fail closed: forced disabled
    @test isempty(local_cache.patterns)  # fail closed: entries dropped
    # Global cache untouched by the local exercise.
    @test SDPX.structure_cache_stats().enabled == true
end
