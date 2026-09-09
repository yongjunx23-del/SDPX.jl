# Parent de-risking check for the R0-P runtime fix.
#
# Question: at the captured failing Power points, does the reviewed compensated
# half-Power factor produce a DENSE Theta that satisfies the runtime's own
# dense metric contract (secant Theta*y = s, SPD, finite inverse pair), with
# metric error inside the unchanged 2^-22 target?  If yes, the smallest fix is
# an experimental scaling provider that materializes the certified dense Theta
# and leaves the existing consumers unchanged.
#
# Read-only experiment: no production source is modified.
using Test, TOML, LinearAlgebra, SDPX
include(joinpath(@__DIR__, "..", "factor_preserving_affine.jl"))
include(joinpath(@__DIR__, "..", "factor_affine_reference.jl"))
include(joinpath(@__DIR__, "..", "native_factor_affine_certificate.jl"))
const FA = FactorPreservingAffine
const FAR = FactorAffineReference
const NC = NativeFactorAffineCertificate
const Q = Rational{BigInt}
const RESULTS = Any[]

inside(I, x) = Q(I.lo) <= x <= Q(I.hi)

@testset "dense Theta from certified compensated factor" begin
    for id in (17, 19)
        row = TOML.parsefile(joinpath(@__DIR__, "..", "fixtures", "factor_affine_trial_$id.toml"))
        epoch = FA.build(row; factor_mode = :compensated_half_candidate)
        for (block, construction) in zip(epoch.cone.blocks, epoch.construction)
            rows = block.offset:block.offset+2
            # exact dense transform from the certified factor representation
            S = [FA.transform(block, [i == j ? 1.0 : 0.0 for i in 1:3], :S) for j in 1:3]
            Sdense = hcat(S...)
            Wdense = [FA.transform(block, [i == j ? 1.0 : 0.0 for i in 1:3], :W) for j in 1:3]
            Wdense = hcat(Wdense...)
            Theta = Sdense * transpose(Sdense)
            G = transpose(Wdense) * Wdense
            y = block.dual
            s = block.primal
            secant = Theta * y - s
            secant_rel = maximum(abs, secant) / max(maximum(abs, s), eps())
            inverse_pair = G * Theta - Matrix{Float64}(I, 3, 3)
            inverse_rel = maximum(abs, inverse_pair)
            spd = try
                cholesky(Symmetric(Theta); check = true)
                true
            catch
                false
            end
            # exact rational metric error against the true inverse Hessian
            H = FAR.true_hessian(block.shadow)
            Hinv = FAR.exact_solve(H, Matrix{Q}(I, 3, 3))
            target = Q(block.mu) * Hinv
            error_matrix = Q.(Theta) - target
            metric_error = sqrt(sum(abs2, error_matrix))
            # normalized against the target magnitude
            metric_rel = metric_error / sqrt(sum(abs2, target))
            ok = spd && secant_rel <= 1e-12 && inverse_rel <= 1e-12 &&
                 metric_error <= Q(FA.HalfPowerFactorCertificate.KAPPA)
            println("DENSE_THETA id=", id, " offset=", block.offset,
                " spd=", spd, " secant=", secant_rel,
                " inverse=", inverse_rel, " metric_abs=", Float64(metric_error),
                " metric_rel=", Float64(metric_rel))
            push!(RESULTS, (; id, offset = block.offset, spd, secant_rel,
                inverse_rel, metric_error = Float64(metric_error),
                metric_rel = Float64(metric_rel)))
            @test spd
            @test secant_rel <= 1e-12
            @test inverse_rel <= 1e-12
        end
    end
end
