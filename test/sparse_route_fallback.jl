# PR-03: sparse-route fallback must be recorded, never silent.
#
# The plan's PR-03 imposes two rules this file pins:
#
#   * "所有 fallback 由计划明确授权并记录" -- every fallback must be authorized
#     and recorded;
#   * "禁止 silently sparse→dense 越过内存上限" -- no silent sparse-to-dense
#     substitution that evades the memory limit.
#
# `kkt_route=:sparse_schur` is the route where this matters: the sparse UMFPACK
# Schur path can fail and the executor then retries the same iterate through the
# dense `:expanded` executor, then possibly `:bordered`. That ladder is
# legitimate, but only if the receipt distinguishes requested from executed.
# A receipt that reported the requested route as if it had executed would be
# exactly the silent densification the plan forbids.
#
# This file also pins the public boundary for the sparse route by arithmetic
# type: `:sparse_augmented` is Float64/CHOLMOD only, and every other precision
# must be refused before any route is planned.
using Test
using SDPX

@testset "Sparse route fallback is recorded, never silent" begin
    function _soc_model()
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        return model
    end

    function _run(route::Symbol)
        result = SDPX.optimize!(_soc_model(); settings=SDPX.Settings(Float64;
            kkt_route=route, verbosity=0,
            limits=SDPX.Limits(iterations=300, time=120.0, threads=1)))
        return result, SDPX.diagnostics(result).selected_algorithms
    end

    @testset "every public route executes or reports why it did not" begin
        for route in (:bordered, :expanded, :sparse_schur, :sparse_augmented)
            result, selected = _run(route)
            # The solve must succeed on this analytic case regardless of route,
            # so a failure here is a route regression, not a toleranced outcome.
            @test SDPX.status(result) === :optimal
            @test SDPX.certificate(result).valid
            @test isapprox(SDPX.primal_objective(result), -1.0; atol=1e-8)

            # The requested route is always reported verbatim.
            @test selected.requested_kkt_route === route

            # The executed route must be a real route or the explicit
            # `:not_executed` sentinel -- never an unset/empty value.
            @test selected.executed_kkt_route isa Symbol
            @test selected.executed_kkt_route !== Symbol("")
            @test selected.executed_kkt_route in
                  (:bordered, :expanded, :sparse_schur, :sparse_augmented, :not_executed)

            # Storage and fallback facts must be present and typed, so a
            # consumer can tell densification happened.
            @test selected.executed_kkt_storage isa Symbol
            @test selected.fallback_reason isa Symbol
        end
    end

    @testset "a requested route that did not execute says so" begin
        # If the requested route differs from the executed one, the receipt must
        # carry a fallback reason. `:none` alongside a route change would be the
        # silent densification this test exists to prevent.
        for route in (:bordered, :expanded, :sparse_schur, :sparse_augmented)
            _, selected = _run(route)
            executed = selected.executed_kkt_route
            if executed !== route && executed !== :not_executed
                @test selected.fallback_reason !== :none
            end
            # And the converse: a fallback reason implies either a route change
            # or an explicit non-execution.
            if selected.fallback_reason !== :none
                @test executed !== route || executed === :not_executed
            end
        end
    end

    @testset "sparse_schur densification is explicit when it happens" begin
        # On a small analytic SOC the sparse UMFPACK Schur path is not viable
        # and the executor falls back. Pin the observable facts so the behaviour
        # cannot change silently: either it executes the sparse route with
        # sparse storage, or it reports a fallback AND dense storage.
        _, selected = _run(:sparse_schur)
        if selected.executed_kkt_route === :sparse_schur
            @test selected.executed_kkt_storage === :sparse
            @test selected.fallback_reason === :none
        else
            @test selected.fallback_reason !== :none
            @test selected.executed_kkt_storage !== :sparse
        end
    end

    @testset "storage claim is consistent with the executed route" begin
        # `:expanded` is the dense executor; claiming sparse storage for it
        # would misreport the factorization that actually ran.
        _, expanded = _run(:expanded)
        @test expanded.executed_kkt_route === :expanded
        @test expanded.executed_kkt_storage === :dense

        _, bordered = _run(:bordered)
        @test bordered.executed_kkt_route === :bordered
        @test bordered.executed_kkt_storage === :sparse
        @test bordered.fallback_reason === :none
    end

    @testset "sparse_augmented is Float64/CHOLMOD only, fail-closed" begin
        # These assertions duplicate the ones beside the sparse-augmented E2E
        # test on purpose: they are the PR-03 public-boundary contract, and a
        # reader of the sparse-route file should not have to find them
        # elsewhere. `Settings` must refuse before any route is planned.
        @test_throws ArgumentError SDPX.Settings{BigFloat}(
            kkt_route=:sparse_augmented,
        )
        mf_types = try
            mf = Base.require(Base.PkgId(
                Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"), "MultiFloats",
            ))
            (mf.Float64x2, mf.Float64x3, mf.Float64x4)
        catch
            nothing
        end
        if mf_types === nothing
            @test_skip "MultiFloats provider unavailable"
        else
            for T in mf_types
                @test_throws ArgumentError SDPX.Settings{T}(
                    kkt_route=:sparse_augmented,
                )
            end
        end
        # The sparse-augmented route, where it does run, must be sparse.
        _, selected = _run(:sparse_augmented)
        @test selected.executed_kkt_route === :sparse_augmented
        @test selected.executed_kkt_storage === :sparse
        @test selected.la_executed_provider === :cholmod
    end
end
