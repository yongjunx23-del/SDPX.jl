using Test, SparseArrays

@testset "Certificate scratch preserves owned zero coordinates" begin
    for bits in (128, 256, 512)
        setprecision(BigFloat, bits) do
            # Rows 1 and 3 remain zero after -A*v. Sharing their MPFR storage
            # corrupts subsequent Newton writes even though -A*v was correct.
            A = sparse([2], [1], BigFloat[-1], 3, 1)
            v = BigFloat[1]
            out = SDPX.alloc_zeros(BigFloat, 3)
            for repeat in 1:3
                @test SDPX._at_negmul!(out, A, v) === out
                @test out == BigFloat[0, 1, 0]
                SDPX._store_owned_scalar!(out, 1, BigFloat(2))
                @test out == BigFloat[2, 1, 0]
                SDPX._store_owned_scalar!(out, 3, BigFloat(3))
                @test out == BigFloat[2, 1, 3]
            end
            fresh = SDPX._at_negmul(A, v)
            @test fresh == BigFloat[0, 1, 0]
            SDPX._store_owned_scalar!(fresh, 1, BigFloat(4))
            @test fresh == BigFloat[4, 1, 0]
            @test v == BigFloat[1]
            @test nonzeros(A) == BigFloat[-1]
        end
    end
end

# The public solve requires the optional BigFloat factorization provider.
# The ownership regression above is unconditional and does not require it.
if Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) !== nothing
    @testset "BigFloat SOC survives rejected ray checks" begin
        setprecision(BigFloat, 256) do
            model = SDPX.Model(BigFloat)
            x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
            SDPX.constraint!(model, :disk, Any[BigFloat(1), x[1], x[2]], SDPX.LorentzCone())
            SDPX.objective!(model, SDPX.Minimize(), -x[1])
            tol = BigFloat(1e-8)
            result = SDPX.optimize!(model; settings=SDPX.Settings(BigFloat;
                verbosity=0,
                tolerances=SDPX.Tolerances(BigFloat; primal=tol, dual=tol, gap=tol),
                limits=SDPX.Limits(iterations=150, time=120.0, threads=1)))
            @test SDPX.status(result) === :optimal
            @test SDPX.certificate(result).valid
            @test abs(SDPX.certificate(result).primal_objective + 1) <= tol
        end
    end
end
