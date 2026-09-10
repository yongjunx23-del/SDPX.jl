# PR-03: the optional QDLDL sparse signed-LDL provider seam, wired into CI.
#
# Prior state (verified before this file existed): the only coverage of
# `SparseQDLDLCache` lived in `validation/qdldl_sparse_provider.jl`, which is
# referenced by no workflow, no script and not by `test/runtests.jl`. The seam
# therefore had ZERO CI protection, and an audit found it also never covered
# `Float64x3` (its loops ran `(Float64x2, Float64x4)` only).
#
# The seam needs `MultiFloatLinearAlgebra` and `QDLDL`, which are NOT in the
# default project environment (only `MultiFloats` is). So this file probes for
# them and, when they are absent, SKIPS WITH A REASON rather than passing
# silently. A skip is visible in the test summary; a missing test is not.
#
# When the providers ARE present (the `scripts/provider_smoke.sh` environment,
# or any CI job that installs them), the seam is exercised for all three
# MultiFloat widths.
#
# Provider contract this pins -- documented upstream and by the adapters:
#   * symmetric quasi-definite operator, upper triangle only in CSC;
#   * every structural column nonempty;
#   * a +1/-1 D-sign vector;
#   * NO dynamic regularization, so the caller owns any explicit signed shift;
#   * the SDPX symmetric augmented core is deliberately NOT quasi-definite as
#     stored (structural zeros on the reduced-x diagonal), so it must not be
#     sent here. It is sent only explicitly shifted operators.
using Test
using SDPX
using LinearAlgebra
using SparseArrays

const _QDLDL_PROVIDER = try
    Base.require(Base.PkgId(
        Base.UUID("642d9d30-8e28-45ca-9d81-256429ea358f"),
        "MultiFloatLinearAlgebra",
    ))
    Base.require(Base.PkgId(
        Base.UUID("bfc457fd-c171-5ab7-bd9e-d5dbfc242d63"), "QDLDL",
    ))
    true
catch
    false
end

@testset "SparseQDLDL provider seam (CI-wired)" begin
    if !_QDLDL_PROVIDER
        @test_skip "MultiFloatLinearAlgebra/QDLDL not installed in this " *
                   "environment; the QDLDL sparse seam is UNTESTED here"
    else
        # Resolve the widths from the package rather than naming them bare, so
        # this file still parses in an environment without the provider.
        mf = Base.require(Base.PkgId(
            Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"), "MultiFloats",
        ))
        widths = (mf.Float64x2, mf.Float64x3, mf.Float64x4)

        @testset "declared precision widths, including the previously unrun x3" begin
            for T in widths
                n = 8
                nr = n - 4
                Ar = T.(randn(4, nr))
                Theta = T.(Diagonal(1.0 .+ rand(4)))
                Kfull = zeros(T, n, n)
                for j in 1:nr
                    # Explicit quasi-definite x diagonal: the caller-owned
                    # signed shift the contract requires.
                    Kfull[j, j] = T(1e-8)
                end
                Kfull[nr+1:n, 1:nr] .= Ar
                Kfull[1:nr, nr+1:n] .= transpose(Ar)
                Kfull[nr+1:n, nr+1:n] .= -Theta
                Kup = sparse(UpperTriangular(Matrix(Symmetric(Kfull, :U))))
                dsigns = vcat(fill(1, nr), fill(-1, 4))

                cache = SDPX.SparseQDLDLCache{T}(Kup, dsigns; nrhs=1)
                SDPX.factorize!(cache, Kup, 1)
                rhs = T.(randn(n))
                expected = Kfull \ rhs
                got = zeros(T, n)
                SDPX.solve!(cache, got, rhs)
                @test norm(got - expected) < 1e-10

                # Batched solve reuses the one symbolic factorization.
                rhs2 = T.(randn(n, 3))
                got2 = zeros(T, n, 3)
                SDPX.solve_multi!(cache, got2, rhs2)
                @test norm(got2 - (Kfull \ rhs2)) < 1e-10

                # Same-epoch reuse must not re-symbolize, and must still solve.
                SDPX.factorize!(cache, Kup, 1)
                got3 = zeros(T, n)
                SDPX.solve!(cache, got3, rhs)
                @test norm(got3 - expected) < 1e-10

                # Fail closed on a non-quasi-definite operator (zero x
                # diagonal). Accepting it would be the silent-regularization
                # failure the contract forbids.
                bad = zeros(T, n, n)
                bad[nr+1:n, 1:nr] .= Ar
                bad[1:nr, nr+1:n] .= transpose(Ar)
                bad[nr+1:n, nr+1:n] .= -Theta
                bad_up = sparse(UpperTriangular(Matrix(Symmetric(bad, :U))))
                @test_throws Exception SDPX.SparseQDLDLCache{T}(
                    bad_up, dsigns; nrhs=1,
                )
            end
        end
    end

    @testset "provider availability is a declared fact, not an assumption" begin
        # Whether or not the provider is installed here, the seam's availability
        # predicate must answer deterministically and must never claim support
        # for an arithmetic it has no method for.
        for T in (Float64, BigFloat)
            available = SDPX.SparseQDLDLProviderAvailable(T)
            @test available isa Bool
            # Float64 uses CHOLMOD and BigFloat has no QDLDL adapter at all;
            # neither may report an available QDLDL provider.
            @test !available
        end
        if _QDLDL_PROVIDER
            mf = Base.require(Base.PkgId(
                Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"), "MultiFloats",
            ))
            for T in (mf.Float64x2, mf.Float64x3, mf.Float64x4)
                @test SDPX.SparseQDLDLProviderAvailable(T)
            end
        end
    end
end
