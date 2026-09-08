# Standalone integration against a BFLA pin that explicitly supports natural
# ordering. Memory admission remains unavailable; core tests use research only.
include("experimental_sparse_core_identity.jl")
import MultiFloats, MultiFloatLinearAlgebra

function _adapter_original_error(U, x, b, bits)
    setprecision(BigFloat, max(1024, 2bits)) do
        worst = BigFloat(0)
        for i in eachindex(b)
            acc, work = BigFloat(0), abs(BigFloat(b[i]))
            for j in eachindex(x)
                term = BigFloat(U[min(i, j), max(i, j)]) * BigFloat(x[j])
                acc += term
                work += abs(term)
            end
            err = abs(acc - BigFloat(b[i]))
            worst = max(worst, iszero(work) ? (iszero(err) ? BigFloat(0) : BigFloat(Inf)) : err / work)
        end
        worst
    end
end

@testset "explicit SDPX natural-order adapter" begin
    @test SDPX.SparseQDLDLProviderOrderingAvailable(BigFloat, :natural)
    @test SDPX.SparseQDLDLProviderOrderingAvailable(BigFloat, :amd)
    @test !SDPX.SparseQDLDLProviderOrderingAvailable(BigFloat, :unknown)
    @test !SDPX.SparseQDLDLProviderOrderingAvailable(MultiFloats.Float64x4, :natural)
    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            U = sparse(BigFloat[2 1 0; 0 -2 0.25; 0 0 -1.5])
            signs, b = [1, -1, -1], BigFloat[1, 2, -1]
            natural = SDPX.SparseQDLDLCache{BigFloat}(U, signs; ordering=:natural)
            amd = SDPX.SparseQDLDLCache{BigFloat}(U, signs)
            for cache in (natural, amd)
                @test SDPX.factor_diagnostics(cache).provider_ordering === cache.ordering
                f = something(cache.provider.inner.factor)
                @test (f.perm === nothing) == (cache.ordering === :natural)
                @test (f.iperm === nothing) == (cache.ordering === :natural)
                @test (f.workspace.AtoPAPt === nothing) == (cache.ordering === :natural)
                SDPX.factorize!(cache, U, 1)
                x = fill(BigFloat(0), 3)
                SDPX.solve!(cache, x, b)
                @test _adapter_original_error(U, x, b, bits) <= 128 * eps(BigFloat)
                @test x[1] !== x[2]
                @test all(precision(v) == bits for v in x)
                SDPX.factorize!(cache, U, 1)
                @test cache.numeric_count == 1 && cache.symbolic_count == 1
            end
            @test_throws ErrorException setfield!(natural, :ordering, :amd)
            @test_throws ArgumentError SDPX.SparseQDLDLCache{BigFloat}(U, signs; ordering=:unknown)
            # Replacing the actual provider handle cannot change a cache's
            # frozen ordering, including through same-epoch reuse.
            for entry in (:reuse, :solve, :multi, :refine)
                saved = natural.provider
                natural.provider = amd.provider
                x = fill(BigFloat(77), 3)
                if entry === :reuse
                    @test_throws ArgumentError SDPX.factorize!(natural, U, natural.matrix_epoch)
                elseif entry === :solve
                    @test_throws ArgumentError SDPX.solve!(natural, x, b)
                elseif entry === :multi
                    @test_throws ArgumentError SDPX.solve_multi!(natural, reshape(x, :, 1), reshape(b, :, 1))
                else
                    @test_throws ArgumentError SDPX.refine_once!(natural, b, x)
                end
                @test SDPX.factor_status(natural) !== SDPX.Fresh
                @test all(==(BigFloat(77)), x)
                natural.provider = saved
                SDPX.factorize!(natural, U, natural.factor_epoch + 1)
                @test SDPX.factor_status(natural) === SDPX.Fresh
            end
        end
    end
end

@testset "natural-order unadmitted core preserves original acceptance" begin
    for bits in (256, 512)
        setprecision(BigFloat, bits) do
            ws, system, ctx = _identity_fixture(; ordering=:natural)
            @test ws.cache.ordering === :natural
            @test ws.cache.inner.ordering === :natural
            @test SDPX.factor_diagnostics(ws.cache).provider_ordering === :natural
            f = something(ws.cache.inner.provider.inner.factor)
            @test f.perm === nothing && f.iperm === nothing && f.workspace.AtoPAPt === nothing
            direction, _ = SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
            @test SDPX.experimental_sparse_core_accept(system, direction, ctx.families)
            @test ws.cache.inner.symbolic_count == 1
            # A working natural-order numerical primitive is NOT memory admission.
            caught = try
                SDPX.prepare_experimental_sparse_core_state(system, ws.V, ctx.families,
                    ctx.witness_rows, ctx.delta, bits, typemax(Int), 0; ordering=:natural)
                nothing
            catch e
                e
            end
            @test caught isa ArgumentError
            @test occursin("owned_storage_bound_unavailable", caught === nothing ? "" : sprint(showerror, caught))
        end
    end
end
