# Checking a solution instead of trusting it.
#
# A solver reporting `Optimal` is a claim about its own iteration, not proof
# that the returned point solves the problem you posed. `result_certificate`
# re-derives the optimality conditions in the *original* coordinates -- after
# any scaling, equilibration, or presolve has been undone -- and reports what
# it can and cannot confirm.
#
# This example asks for three tolerances on the same problem and shows the
# certificate agreeing with the solver twice and disagreeing once.
#
# Run with:  julia --project=examples examples/04_certificates.jl

using LinearAlgebra
using Printf
using SDPX

coefficients = zeros(2, 2, 2)
coefficients[1, 1, 1] = 1.0
coefficients[2, 2, 2] = 1.0
constant = [0.0 1.0; 1.0 0.0]
objective = [2.0, 3.0]

outcomes = Dict{Float64,Any}()
for tolerance in (1e-8, 1e-10, 1e-12)
    problem = ingest(
        objective,
        [coefficients],
        [constant],
        Matrix{Float64}(undef, 2, 0),
        Float64[],
    )
    # Certify against the *same* options the solve used. Certifying against a
    # different tolerance than was requested compares a result to a standard
    # it was never aiming for.
    options = SolverOptions{Float64}(
        verbosity=0,
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
    )
    result = solve!(problem, options)
    certificate = result_certificate(problem, result, options)
    outcomes[tolerance] = (result=result, certificate=certificate)

    @printf("tolerance %.0e -> status %-8s certificate %-6s\n",
        tolerance, result.status, certificate.valid ? "valid" : "INVALID")
    @printf("    duality gap %.3e   primal residual %.3e   dual residual %.3e\n",
        certificate.gap, certificate.primal_residual, certificate.dual_residual)
    @printf("    primal PSD %s   dual PSD %s\n",
        certificate.primal_psd.ok, certificate.dual_psd.ok)
    isempty(certificate.failures) ||
        @printf("    unmet conditions: %s\n", certificate.failures)
    println()
end

println("""Both 1e-8 and 1e-10 are reached and certified. At 1e-12 the solve
stalls -- Float64 has no accuracy left to give on this problem -- and the
certificate says so specifically, naming the dual residual and the duality gap
rather than reporting a bare failure. That is the signal to widen the
arithmetic; `02_extended_precision.jl` shows what happens when you do.""")

# The certificate must agree with the solver where the solve succeeded, and
# must refuse to certify where it did not.
for tolerance in (1e-8, 1e-10)
    outcome = outcomes[tolerance]
    outcome.result.status == SDPX.Optimal ||
        error("tolerance $(tolerance): expected Optimal, got $(outcome.result.status)")
    outcome.certificate.valid ||
        error("tolerance $(tolerance): solve reported Optimal but certificate did not hold")
end
outcomes[1e-12].certificate.valid &&
    error("tolerance 1e-12 was expected to be uncertifiable in Float64")
isempty(outcomes[1e-12].certificate.failures) &&
    error("an invalid certificate must name the conditions it could not meet")
