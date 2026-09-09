using Test, LinearAlgebra, SDPX

@testset "Exp Fenchel reconstruction retains the dual point" begin
    setprecision(BigFloat, 256) do
        T = BigFloat
        y = parse(T, "1e5")
        rho = parse(T, "1e-8")
        log_ratio = parse(T, "0.3")
        z = y * exp(log_ratio)
        x = SDPX._nonsymmetric_stable_fma(
            y, log_ratio - rho, zero(T),
        )
        primal = (x, y, z)
        dual = .-collect(SDPX._exp_logarithmic_gradient_values(primal))
        dual_before = deepcopy(dual)
        shadow = Vector{T}(undef, 3)

        result = SDPX.exp_logarithmic_conjugate!(shadow, dual)
        @test result.iterations > 0
        @test dual == dual_before
        @test norm(shadow - T[primal...], Inf) <=
              T("1e-64") * norm(T[primal...], Inf)
        replay_gradient = T[SDPX._exp_logarithmic_gradient_values(shadow)...]
        @test norm(replay_gradient + dual, Inf) <=
              T("1e-64") * max(one(T), norm(dual, Inf))

        workspace = SDPX.NonsymmetricConjugateWorkspace(T;
            max_iterations=64, max_bisections=256,
            residual_tolerance=T("1e-24"),
        )
        status = SDPX.conjugate_shadow!(
            workspace, SDPX.ExpConjugateTag(), dual,
        )
        @test status.status === SDPX.NS_CONJUGATE_SUCCESS
        @test workspace.valid && workspace.hessian_factor_valid

        values = SDPX._exp_logarithmic_hessian_values(workspace.shadow)
        hessian = [values[1] values[2] values[3];
                   values[4] values[5] values[6];
                   values[7] values[8] values[9]]
        factor = workspace.hessian_factor
        @test norm(factor * transpose(factor) - hessian, Inf) <=
              T(2000) * eps(T) * max(one(T), norm(hessian, Inf))
    end
end

@testset "Exp reconstruction rejects an unresolved margin" begin
    setprecision(BigFloat, 256) do
        T = BigFloat
        y = parse(T, "1e5")
        rho = parse(T, "1e-40")
        l = parse(T, "0.3")
        z = y * exp(l)
        x = SDPX._nonsymmetric_stable_fma(y, l - rho, zero(T))
        primal = (x, y, z)
        dual = .-collect(SDPX._exp_logarithmic_gradient_values(primal))
        output = fill(T(-1), 3)
        @test_throws DomainError SDPX.exp_logarithmic_conjugate!(output, dual)
        @test output == fill(T(-1), 3)
    end
end
