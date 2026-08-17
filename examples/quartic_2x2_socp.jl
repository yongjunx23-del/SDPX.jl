using SDPX
using LinearAlgebra

"""Exact Lorentz-cone form of the 2x2 quartic-moment SDP."""

const GOLDEN_RATIO_CONJUGATE = (sqrt(5.0) - 1.0) / 2.0

function quartic_socp_model()
    model = Model(Float64; name="quartic_2x2_socp")
    w = variable!(model, :w, 3; domain=Reals()) # [W_0, W_2, W_4]
    constraint!(model, :normalization, w[1] - 1.0, ZeroCone())
    constraint!(model, :recurrence, w[1] - w[2] - w[3], ZeroCone())

    # [a b; b c] is PSD iff (a+c, a-c, 2b) belongs to Q_3.
    cone = (w[1] + w[3], w[1] - w[3], 2.0 * w[2])
    constraint!(model, :moment_lorentz, cone, LorentzCone())
    objective!(model, Maximize(), w[2])
    return model, w
end

function run_quartic_socp()
    model, w = quartic_socp_model()
    settings = Settings(
        model;
        algorithm=:socp,
        formulation=:auto,
        provider=:auto,
        scaling=:auto,
        limits=Limits(iterations=200, time=60.0, threads=1),
        verbosity=0,
        timing=true,
        diagnostics=:summary,
        certification=true,
    )
    outputs = Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
    )
    result = optimize!(model; settings=settings, outputs=outputs)
    status(result) === :optimal || error("quartic SOCP ended with status $(status(result))")
    cert = certificate(result)
    cert.valid || error("invalid original-coordinate SOCP certificate: $(cert.reason)")

    moments = value(result, w)
    lorentz_vector = [
        moments[1] + moments[3],
        moments[1] - moments[3],
        2.0 * moments[2],
    ]
    cone_margin = lorentz_vector[1] - norm(lorentz_vector[2:3])
    objective = primal_objective(result)
    abs(objective - GOLDEN_RATIO_CONJUGATE) <= 2e-7 || error(
        "SOCP value $objective disagrees with the analytic optimum $GOLDEN_RATIO_CONJUGATE",
    )
    return (
        objective=objective,
        moments=moments,
        lorentz_vector=lorentz_vector,
        cone_margin=cone_margin,
        certificate=cert,
        plan=execution_plan(result),
        result=result,
    )
end

function main(args=ARGS)
    isempty(args) || error("quartic_2x2_socp.jl takes no arguments")
    record = run_quartic_socp()
    println("quartic 2x2 SOCP")
    println("  W2 = ", record.objective)
    println("  analytic W2 = ", GOLDEN_RATIO_CONJUGATE)
    println("  Lorentz margin = ", record.cone_margin)
    println("  certificate residuals = (",
        record.certificate.primal_residual, ", ",
        record.certificate.dual_residual, ", ",
        record.certificate.relative_gap, ")")
    return record
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
