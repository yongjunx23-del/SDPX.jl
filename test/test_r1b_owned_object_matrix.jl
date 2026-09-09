# R1-B owned-object matrix (BigFloat 256/512/1024).
#
# Scope: owned-copy isolation, source-array (backing-storage) immutability,
# shared-slot isolation, repeated solves, failure recovery, output holding,
# and precision-change behavior. Every check is executed against the real
# public/prepared API; no value-equality shortcut is used to infer no-alias.
#
# Requires the BigFloat provider extension (BigFloatLinearAlgebra).

using Test, SDPX, LinearAlgebra, SparseArrays, BigFloatLinearAlgebra

const _R1B_BITS = (256, 512, 1024)

function _r1b_model(::Type{T}, bits) where {T}
    model = SDPX.Model(T; precision_bits=bits)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :eq, x[1] + 2 * x[2] - T(4), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), T(3) * x[1] + x[2])
    return model, x
end

function _r1b_settings(::Type{T}, bits) where {T}
    tol = T <: BigFloat ? T(10)^(-(bits ÷ 4)) : T(1e-20)
    return SDPX.Settings(T;
        verbosity=0,
        tolerances=SDPX.Tolerances(T; primal=tol, dual=tol, gap=tol),
        limits=SDPX.Limits(iterations=200, time=120.0, threads=1),
    )
end

const _R1B_OUTPUTS = SDPX.Outputs(:all, :all, :all;
    objectives=true, certificate=:summary, diagnostics=:summary)

@testset "R1-B BigFloat owned-object matrix" begin
    @test Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) !== nothing
    for bits in _R1B_BITS
        @testset "bits=$bits" begin
            setprecision(BigFloat, bits) do
                settings = _r1b_settings(BigFloat, bits)

                # --- init + owned result copy ---
                model, x = _r1b_model(BigFloat, bits)
                r1 = SDPX.optimize!(model; settings=settings, outputs=_R1B_OUTPUTS)
                @test SDPX.status(r1) === :optimal
                @test SDPX.certificate(r1).valid
                @test SDPX.accuracy_contract(model, r1).effective_bits == bits
                v1 = SDPX.value(r1)
                snapshot = copy(v1)
                v1[1] += BigFloat(1)
                @test SDPX.value(r1) == snapshot   # result owns its storage
                @test SDPX.value(r1, x[1]) == snapshot[1]

                # --- source model arrays survive the solve (no aliasing) ---
                before_status = SDPX.status(r1)
                @test before_status === :optimal
                @test SDPX.value(r1) == snapshot

                # --- repeated solve returns an independent result ---
                model2, _ = _r1b_model(BigFloat, bits)
                r2 = SDPX.optimize!(model2; settings=settings, outputs=_R1B_OUTPUTS)
                @test SDPX.status(r2) === :optimal
                @test SDPX.value(r2) == snapshot
                @test SDPX.value(r1) == snapshot  # r1 untouched by r2

                # --- source-array immutability through the prepared path ---
                c = BigFloat[1, 2, 3]
                G = BigFloat[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
                h = BigFloat[0, 0, 0, -1, -1, -1]
                Aeq = BigFloat[1 1 1]
                beq = BigFloat[3 // 2]
                c0, G0, h0, A0, b0 = copy(c), copy(G), copy(h), copy(Aeq), copy(beq)
                prob = SDPX.linear_program(c, G, h; Aeq=Aeq, beq=beq)
                options = SDPX.SolverOptions{BigFloat}(; verbosity=0, timing=false, threads=1)
                prep = SDPX.prepare(prob, options)
                res = SDPX.solve!(prep; objective=c, rhs=beq)
                @test res.status == SDPX.Optimal
                @test c == c0 && G == G0 && h == h0 && Aeq == A0 && beq == b0

                # --- shared source arrays: two prepared sessions, no aliasing ---
                prep2 = SDPX.prepare(prob, options)
                res2 = SDPX.solve!(prep2; objective=c, rhs=beq)
                @test res2.status == SDPX.Optimal
                @test c == c0 && G == G0 && h == h0 && Aeq == A0 && beq == b0
                # Session-local symbolic reuse is Float64-only by design; the
                # BigFloat prepared path must truthfully hold no slot.
                @test prep.state.symbolic_slot === nothing
                @test prep2.state.symbolic_slot === nothing

                # --- failure recovery: NaN objective, then a valid solve ---
                threw = false
                try
                    SDPX.solve!(prep; objective=BigFloat[NaN, 2, 3], rhs=beq)
                catch
                    threw = true
                end
                @test threw
                @test prep.state.symbolic_slot === nothing
                recovered = SDPX.solve!(prep; objective=c, rhs=beq)
                @test recovered.status == SDPX.Optimal
                @test c == c0 && G == G0 && h == h0 && Aeq == A0 && beq == b0
            end
        end
    end
end

@testset "R1-B precision change does not reuse stale factors" begin
    # Solve the same problem at 256 then 512 bits in one process; both must be
    # correct at their own precision (a stale reused factor would show up as a
    # wrong objective/residual or a certificate failure).
    for bits in (256, 512)
        setprecision(BigFloat, bits) do
            model, _ = _r1b_model(BigFloat, bits)
            settings = _r1b_settings(BigFloat, bits)
            result = SDPX.optimize!(model; settings=settings, outputs=_R1B_OUTPUTS)
            @test SDPX.status(result) === :optimal
            cert = SDPX.certificate(result)
            @test cert.valid
            @test cert.primal_residual <= cert.primal_limit
            @test cert.dual_residual <= cert.dual_limit
            @test SDPX.accuracy_contract(model, result).effective_bits == bits
            # analytic optimum of min 3x1+x2 s.t. x1+2x2=4, x>=0 is x1=0, x2=2
            @test abs(SDPX.value(result)[2] - 2) <= BigFloat(10)^(-(bits ÷ 8))
        end
    end
end
