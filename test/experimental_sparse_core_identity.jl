# Parent-owned regressions for formation context and CSC coordinate authority.
# Standalone, UNADMITTED LP-only numerical research. Memory admission rejects;
# no public route or production qualification is enabled by these tests.
include("experimental_sparse_core_numerics.jl")
include("experimental_sparse_core.jl")

function _identity_fixture(; factored=true, ordering=:amd)
    system, V = _lp_predictor()
    bits = precision(BigFloat)
    delta = BigFloat(bits >= 512 ? "1e-60" : "1e-30")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, bits, _LIMIT, _RSS; ordering=ordering)
    factored && SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    return ws, system, ctx
end

function _identity_attempt(entry, ws, system, ctx, destination)
    if entry === :driver
        SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    elseif entry === :factor
        SDPX.factorize_symmetric_core_pattern!(ws.cache, ws.pattern, 1)
    elseif entry === :solve
        SDPX.solve!(ws.cache, destination, ones(BigFloat, ws.dimension))
    elseif entry === :multi
        SDPX.solve_multi!(ws.cache, reshape(destination, :, 1),
            ones(BigFloat, ws.dimension, 1))
    elseif entry === :refine
        SDPX.refine_once!(ws.cache, ones(BigFloat, ws.dimension), destination)
    elseif entry === :caller
        SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
    else
        error("unknown test entry")
    end
end

function _identity_revoked(ws; whole_workspace=false, last_epoch=1)
    @test SDPX.factor_status(ws.cache) !== SDPX.Fresh
    @test SDPX.factor_status(ws.cache.inner) !== SDPX.Fresh
    @test ws.cache.last_valid
    @test ws.cache.last_matrix_epoch == last_epoch
    if whole_workspace
        @test ws.factor_receipt === nothing
        @test !ws.synchronized
        @test ws.homogeneous_epoch == -1
    end
end

function _identity_corrupt!(w, kind)
    if kind === :row
        slot = w.upper_colptr[3]
        @assert w.upper_rowval[slot] == 1
        w.upper_rowval[slot] = 2 # valid, sorted upper column rows2,3
    elseif kind === :pointer
        w.upper_colptr[3] += 1
    elseif kind === :dimension
        w.n += 1
    elseif kind === :row_length
        pop!(w.upper_rowval)
    elseif kind === :pointer_length
        pop!(w.upper_colptr)
    elseif kind === :value_length
        pop!(w.upper_nzval)
    elseif kind === :diagonal_length
        pop!(w.upper_diag)
    else
        error("unknown test corruption")
    end
end

# Test-only provider proxy. Both numeric factorization and solves delegate to
# the REAL loaded provider before a deterministic fault is injected. No
# fabricated factor, receipt, solution, or production hook is involved.
mutable struct _IdentityFaultProvider{P,W}
    payload::P
    workspace::W
    phase::Symbol
    enabled::Bool
    factors::Int
    solves::Int
    observed_synchronized::Bool
end
SDPX._qdldl_provider_ordering(::Type{BigFloat}, p::_IdentityFaultProvider) =
    SDPX._qdldl_provider_ordering(BigFloat, p.payload)

function SDPX._qdldl_provider_factorize!(p::_IdentityFaultProvider, A::SparseMatrixCSC{BigFloat,Int})
    result = SDPX._qdldl_provider_factorize!(p.payload, A)
    p.factors += 1
    if p.enabled && p.phase === :synchronization
        _identity_corrupt!(p.workspace.cache, :row)
    end
    return result
end
function SDPX._qdldl_provider_solve!(p::_IdentityFaultProvider,
    destination::AbstractVector{BigFloat}, rhs::AbstractVector{BigFloat})
    result = SDPX._qdldl_provider_solve!(p.payload, destination, rhs)
    p.solves += 1
    if p.enabled && p.phase === :homogeneous
        p.observed_synchronized = p.workspace.synchronized &&
                                  p.workspace.factor_receipt !== nothing
        error("intentional post-provider homogeneous fault")
    end
    return result
end
function _identity_attach_fault!(ws, phase)
    original = ws.cache.inner
    proxy = _IdentityFaultProvider(original.provider, ws, phase, true, 0, 0, false)
    fields = ntuple(fieldcount(typeof(original))) do i
        name = fieldname(typeof(original), i)
        name === :provider ? proxy : getfield(original, name)
    end
    ws.cache.inner = SDPX.SparseQDLDLCache{BigFloat,typeof(proxy)}(fields...)
    return proxy
end

@testset "parent sparse-core authority regressions" begin
    @testset "formation mode is separate from precision" begin
        for bits in (256, 512)
            setprecision(BigFloat, bits) do
                # Positive control: all three formation modes work when
                # consumed in their own context. Oracle rounds the exact sum
                # of the actual stored operands, not a rebuilt ideal factor.
                for mode in (RoundNearest, RoundUp, RoundDown)
                    setrounding(BigFloat, mode) do
                        ws, system, ctx = _identity_fixture()
                        w = ws.cache
                        @test w.formation_rounding == mode
                        @test SDPX.factor_diagnostics(w).formation_rounding == string(mode)
                        @test SDPX._core_factor_matches_pattern(w, ws.pattern)
                        for j in 1:w.n
                            slot = findfirst(==(w.upper_diag[j]), w.lower_to_upper)
                            exact = setprecision(BigFloat, bits + 256) do
                                BigFloat(w.snapshot_lower[slot]) +
                                    BigFloat(w.dsigns[j]) * BigFloat(w.delta)
                            end
                            expected = BigFloat(exact, mode; precision=bits)
                            @test w.upper_nzval[w.upper_diag[j]] == expected
                            @test precision(w.upper_nzval[w.upper_diag[j]]) == bits
                        end
                        @test SDPX.factor_status(w) === SDPX.Fresh
                    end
                end
                for mode in (RoundUp, RoundDown), entry in
                    (:driver, :factor, :solve, :multi, :refine, :caller)
                    ws, system, ctx = _identity_fixture()
                    before = deepcopy(ws.cache.upper_nzval)
                    destination = fill(BigFloat(77), ws.dimension)
                    setrounding(BigFloat, mode) do
                        @test precision(BigFloat) == bits # only mode changed
                        @test !SDPX._core_factor_matches_pattern(ws.cache, ws.pattern)
                        @test_throws ArgumentError _identity_attempt(
                            entry, ws, system, ctx, destination)
                        @test ws.cache.factor_epoch == 1
                        @test ws.cache.upper_nzval == before
                        @test all(==(BigFloat(77)), destination)
                        _identity_revoked(ws; whole_workspace=entry in (:driver, :caller))
                    end
                    @test rounding(BigFloat) == RoundNearest
                    # Restoring the frozen context permits a new valid epoch.
                    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 2)
                    @test SDPX.factor_status(ws.cache) === SDPX.Fresh
                    @test ws.cache.last_matrix_epoch == 2
                end
                # Reverse context change is rejected too (not hard-coded to
                # require Nearest regardless of the actual creation mode).
                for mode in (RoundUp, RoundDown)
                    ws, system, ctx = setrounding(BigFloat, mode) do
                        _identity_fixture()
                    end
                    @test !SDPX._core_factor_matches_pattern(ws.cache, ws.pattern)
                    @test_throws ArgumentError SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
                    _identity_revoked(ws; whole_workspace=true)
                end
            end
        end
    end

    @testset "actual CSC coordinates and shapes, not just slot values" begin
        for kind in (:row, :pointer, :dimension, :row_length, :pointer_length,
                     :value_length, :diagonal_length), entry in (:factor, :solve, :caller)
            ws, system, ctx = _identity_fixture()
            w = ws.cache
            frozen_rows = copy(w.inner.rowval)
            frozen_ptrs = copy(w.inner.colptr)
            _identity_corrupt!(w, kind)
            before = deepcopy(w.upper_nzval)
            destination = fill(BigFloat(77), ws.dimension)
            @test !SDPX._core_factor_matches_pattern(w, ws.pattern)
            @test_throws ArgumentError _identity_attempt(entry, ws, system, ctx, destination)
            @test w.inner.rowval == frozen_rows
            @test w.inner.colptr == frozen_ptrs
            @test w.upper_nzval == before
            @test w.factor_epoch == 1
            @test all(==(BigFloat(77)), destination)
            _identity_revoked(ws; whole_workspace=entry === :caller)
        end
        for entry in (:multi, :refine)
            ws, system, ctx = _identity_fixture()
            _identity_corrupt!(ws.cache, :row)
            @test_throws ArgumentError _identity_attempt(entry, ws, system, ctx,
                fill(BigFloat(77), ws.dimension))
            _identity_revoked(ws)
        end
    end

    @testset "experimental metadata is not shared cache payload" begin
        first_ws, system, ctx = _identity_fixture()
        second_ws, _, _ = _identity_fixture()
        cached = lock(SDPX._SYMMETRIC_CORE_STRUCTURE_LOCK) do
            SDPX._SYMMETRIC_CORE_STRUCTURE_CACHE.patterns[(BigFloat, first_ws.pattern.signature)]
        end
        for name in (:colptr, :rowval, :ar_slots, :theta_slots, :x_diag_slots,
                     :block_ranges, :block_shapes)
            a, b, c = getproperty(first_ws.pattern, name), getproperty(second_ws.pattern, name), getproperty(cached, name)
            @test a == b == c
            @test a !== b && a !== c && b !== c
        end
        first_ws.pattern.theta_slots[1], first_ws.pattern.theta_slots[2] =
            first_ws.pattern.theta_slots[2], first_ws.pattern.theta_slots[1]
        @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(first_ws, system, 2)
        @test second_ws.pattern.theta_slots == cached.theta_slots
        @test SDPX._core_factor_matches_pattern(second_ws.cache, second_ws.pattern)
        third_ws, _, _ = _identity_fixture()
        @test SDPX.factor_status(third_ws.cache) === SDPX.Fresh
    end

    @testset "refill maps are checked before they can write" begin
        for kind in (:theta_swap, :theta_to_ar, :x_empty, :x_to_theta)
            ws, system, ctx = _identity_fixture()
            p = ws.pattern
            before = deepcopy(p.nzval)
            if kind === :theta_swap
                p.theta_slots[1], p.theta_slots[2] = p.theta_slots[2], p.theta_slots[1]
            elseif kind === :theta_to_ar
                p.theta_slots[1] = p.ar_slots[1]
            elseif kind === :x_empty
                empty!(p.x_diag_slots)
            else
                p.x_diag_slots[1] = p.theta_slots[1]
            end
            @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(ws, system, 2)
            @test p.nzval == before
            @test ws.cache.factor_epoch == 1
            _identity_revoked(ws; whole_workspace=true)
        end
        ws, system, ctx = _identity_fixture()
        pop!(system.A.nzval)
        @test_throws DimensionMismatch SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
        _identity_revoked(ws; whole_workspace=true)
    end

    @testset "rounded shift equality is not a structural-zero proof" begin
        ws, system, ctx = _identity_fixture(; factored=false)
        w = ws.cache
        slot = ws.pattern.x_diag_slots[1]
        w.snapshot_lower[slot] = BigFloat("1e-400")
        @test w.snapshot_lower[slot] + w.delta == w.delta
        @test !SDPX._experimental_upper_ok(w)
    end

    @testset "same-epoch reuse and direct failures revoke" begin
        ws, system, ctx = _identity_fixture()
        u = ws.cache.upper_diag[3]
        ws.cache.upper_nzval[u] = BigFloat(-2) # missing the declared shift
        @test_throws ArgumentError SDPX.factorize_symmetric_core_pattern!(ws.cache, ws.pattern, 1)
        _identity_revoked(ws)
        for entry in (:solve, :multi, :refine)
            ws, system, ctx = _identity_fixture()
            @test_throws DimensionMismatch if entry === :solve
                SDPX.solve!(ws.cache, zeros(BigFloat, 4), ones(BigFloat, 5))
            elseif entry === :multi
                SDPX.solve_multi!(ws.cache, zeros(BigFloat, 4, 1), ones(BigFloat, 5, 1))
            else
                SDPX.refine_once!(ws.cache, ones(BigFloat, 5), zeros(BigFloat, 4))
            end
            _identity_revoked(ws)
        end
        ws, system, ctx = _identity_fixture()
        @test_throws ArgumentError SDPX.prepare!(ws.cache, SDPX.FactorRequirements(6))
        _identity_revoked(ws)
    end

    @testset "real post-factor driver failures revoke every layer" begin
        for phase in (:synchronization, :homogeneous)
            ws, system, ctx = _identity_fixture(; factored=false)
            proxy = _identity_attach_fault!(ws, phase)
            caught = try
                SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
                nothing
            catch e
                e
            end
            msg = caught === nothing ? "" : sprint(showerror, caught)
            @test proxy.factors == 1 # real provider numeric factor completed
            if phase === :synchronization
                @test caught isa ArgumentError
                @test occursin("do not match the fresh factor operator", msg)
                @test proxy.solves == 0
            else
                @test occursin("intentional post-provider homogeneous fault", msg)
                @test proxy.solves == 1 # real checked provider solve completed
                @test proxy.observed_synchronized
            end
            _identity_revoked(ws; whole_workspace=true)
            proxy.enabled = false
            ws.cache.upper_rowval .= ws.cache.inner.rowval
            SDPX.factor_experimental_sparse_core_epoch!(ws, system, 2)
            @test SDPX.factor_status(ws.cache) === SDPX.Fresh
            @test ws.cache.last_matrix_epoch == 2
            @test proxy.factors == 2
            direction, _ = SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
            @test SDPX.experimental_sparse_core_accept(system, direction, ctx.families)
        end
    end
end
