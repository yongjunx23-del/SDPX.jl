# R1 full qualification (real public API only):
# R1-A: AccuracyContract fields verified against the loaded runtime
# R1-B: owned-object mutation isolation (results are independent copies;
#       mutating one result never corrupts another result, the model, or a
#       subsequent solve) across Float64 and BigFloat 256/512
# R1-D: stable public outputs — original-coordinate evaluators, truthful
#       original-coordinate certificate summary, typed failure symbols
#
# R1-A is covered in depth by test/accuracy_contract.jl; this file adds the
# end-to-end solve-level ownership and public-output gates.

using Test, SDPX, LinearAlgebra, SparseArrays

function _r1_lp_model(::Type{T}; precision_bits::Union{Nothing,Int}=nothing) where {T}
    model = precision_bits === nothing ? SDPX.Model(T) :
        SDPX.Model(T; precision_bits=precision_bits)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :eq, x[1] + 2 * x[2] - T(4), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), T(3) * x[1] + x[2])
    return model, x
end

@testset "R1-A: AccuracyContract matches runtime at solve level" begin
    model, _ = _r1_lp_model(Float64)
    settings = SDPX.Settings(Float64; verbosity=0)
    c = SDPX.accuracy_contract(model, settings)
    @test c.storage_type === Float64
    @test c.effective_bits == 53
    @test c.working_precision_bits == 53
    @test c.verification_precision_bits == 53
    @test c.rounding === RoundNearest
    @test c.finite_required === true

    result = SDPX.optimize!(model; settings=settings)
    @test SDPX.status(result) === :optimal
    rc = SDPX.accuracy_contract(model, result)
    @test rc.storage_type === Float64
    @test rc.effective_bits == 53
end

@testset "R1-B: owned results are independent copies (Float64)" begin
    # Prepared-session repeated solves return independent result objects.
    c0 = Float64[1.0, 2.0, 3.0]
    G = Float64[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
    h = Float64[0.0, 0.0, 0.0, -1.0, -1.0, -1.0]
    Aeq = Float64[1.0 1.0 1.0]
    prob = SDPX.linear_program(c0, G, h; Aeq=Aeq, beq=Float64[1.5])
    options = SDPX.SolverOptions{Float64}(; verbosity=0, timing=false, threads=1)
    prep = SDPX.prepare(prob, options)
    r1 = SDPX.solve!(prep; objective=c0, rhs=Float64[1.5])
    r2 = SDPX.solve!(prep; objective=Float64[1.1, 2.0, 3.0], rhs=Float64[1.5])
    @test r1.status == SDPX.Optimal && r2.status == SDPX.Optimal
    x1_orig = copy(r1.x)
    y1_orig = copy(r1.y)
    # Mutating the first result must not touch the second result nor state.
    r1.x .= 999.0
    r1.y .= -999.0
    @test r2.x != r1.x
    # objective c=[1.1,2,3], sum(x)=1.5, 0<=x<=1: optimal x1=1, x2=0.5
    @test r2.x[1] ≈ 1.0 atol=1e-6
    @test r2.x[2] ≈ 0.5 atol=1e-6
    # A subsequent solve is unaffected by the mutation.
    r3 = SDPX.solve!(prep; objective=c0, rhs=Float64[1.5])
    @test r3.status == SDPX.Optimal
    @test r3.x ≈ x1_orig atol=1e-8
    @test r3.y ≈ y1_orig atol=1e-8

    # Model interface: value(result) is an owned copy.
    model, x = _r1_lp_model(Float64)
    outputs = SDPX.Outputs(:all, :all, :all;
        objectives=true, certificate=:summary, diagnostics=:summary)
    result = SDPX.optimize!(model;
        settings=SDPX.Settings(Float64; verbosity=0), outputs=outputs)
    @test SDPX.status(result) === :optimal
    v = SDPX.value(result)
    v[1] += 100.0
    @test SDPX.value(result)[1] != v[1]
    @test SDPX.value(result, x[1]) == SDPX.value(result)[1]

    # Mutating the source model after the solve never changes the result:
    # snapshot the retained coordinates before the source mutation and
    # compare against that snapshot afterwards (not a getter-vs-getter test).
    before_mutation = copy(SDPX.value(result))
    y = SDPX.variable!(model, :y, 1; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :extra, y[1] - 0.5, SDPX.ZeroCone())
    @test SDPX.value(result) == before_mutation
    @test SDPX.status(result) === :optimal
end

@testset "R1-B: BigFloat 256/512 owned results" begin
    if Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) === nothing
        @test_skip "BigFloat provider extension not loaded"
    else
        for bits in (256, 512)
            setprecision(BigFloat, bits) do
                model, x = _r1_lp_model(BigFloat; precision_bits=bits)
                settings = SDPX.Settings(BigFloat;
                    verbosity=0,
                    tolerances=SDPX.Tolerances(BigFloat;
                        primal=BigFloat(1e-20), dual=BigFloat(1e-20),
                        gap=BigFloat(1e-20)),
                    limits=SDPX.Limits(iterations=200, time=120.0, threads=1))
                outputs = SDPX.Outputs(:all, :all, :all;
                    objectives=true, certificate=:summary, diagnostics=:summary)
                r1 = SDPX.optimize!(model; settings=settings, outputs=outputs)
                @test SDPX.status(r1) === :optimal
                @test SDPX.certificate(r1).valid
                rc = SDPX.accuracy_contract(model, r1)
                @test rc.storage_type === BigFloat
                @test rc.effective_bits == bits

                # Owned result arrays: mutation is local to the result.
                v1 = SDPX.value(r1)
                snapshot = copy(v1)
                v1[1] += BigFloat(1)
                @test SDPX.value(r1) == snapshot

                # A second solve returns a fresh, independent result.
                model2, _ = _r1_lp_model(BigFloat; precision_bits=bits)
                r2 = SDPX.optimize!(model2; settings=settings, outputs=outputs)
                @test SDPX.status(r2) === :optimal
                @test SDPX.value(r2) == snapshot
            end
        end
    end
end

@testset "R1-D: truthful original-coordinate certificate summary" begin
    model, x = _r1_lp_model(Float64)
    result = SDPX.optimize!(model;
        settings=SDPX.Settings(Float64; verbosity=0),
        outputs=SDPX.Outputs(:all, :all, :all;
            objectives=true, certificate=:summary, diagnostics=:summary))
    @test SDPX.status(result) === :optimal
    cert = SDPX.certificate(result)
    @test cert.available
    @test cert.valid
    @test cert.method isa Symbol
    @test cert.reason isa Symbol
    @test isfinite(cert.primal_residual) && cert.primal_residual <= cert.primal_limit
    @test isfinite(cert.dual_residual) && cert.dual_residual <= cert.dual_limit
    @test isfinite(cert.relative_gap) && cert.relative_gap <= cert.gap_limit
    # Original-coordinate objectives agree with the primal value.
    v = SDPX.value(result)
    @test cert.primal_objective ≈ 3 * v[1] + v[2] rtol=1e-6
    @test cert.primal_objective ≈ cert.dual_objective atol=1e-6
end

@testset "R1-D: typed public failure symbols" begin
    # Infeasible LP: x >= 0 with x[1] + x[2] + 1 == 0.
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :eq, x[1] + x[2] + 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    result = SDPX.optimize!(model; settings=SDPX.Settings(Float64; verbosity=0))
    s = SDPX.status(result)
    @test s === :primal_infeasible
    # Check the solver-reported original-coordinate infeasibility certificate:
    # available, valid, the documented ray method, and its reported dual
    # residual inside its own limit.  This validates the reported certificate
    # facts; it does NOT independently recompute the returned ray equations.
    cert = SDPX.certificate(result)
    @test cert.available
    @test cert.valid
    @test cert.method === :original_coordinate_primal_infeasibility_ray
    @test cert.reason === :valid
    @test cert.dual_residual <= cert.dual_limit
    @test SDPX.termination(result).status === SDPX.PrimalInfeasible
end