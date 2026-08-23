using SDPX
using LinearAlgebra
using SparseArrays
using MultiFloats: Float64x4
using Test

# N7: the LP Newton direction gate validates every accepted predictor and
# corrector direction against the unregularized KKT convention
# `K0 = [H -B; B' 0]` with the factor-side shift `Eδ = δI`. Acceptance is
# decided on `rhoδ = rho0 - qδ = r - (K0 + Eδ)*d` against the
# regularization-aware scale `sδ`; `tau` never carries `δ` itself.

function dense_kkt(
    H::Matrix{T},
    B::Matrix{T},
    delta::T,
) where {T}
    variables = size(H, 1)
    equalities = size(B, 2)
    K = [copy(H) -B; transpose(B) delta * Matrix{T}(I, equalities, equalities)]
    @inbounds for index in 1:variables
        K[index, index] += delta
    end
    return K
end

function lp_gate_options(::Type{T}) where {T}
    return SDPX.SolverOptions{T}(
        ϵ_primal=T(1e-8),
        ϵ_dual=T(1e-8),
        verbosity=0,
    )
end

function run_gate(
    workspace::SDPX.LPWorkspace{T},
    G,
    B,
    rhs::Vector{T},
    dx::Vector{T},
    dy::Vector{T},
    delta::T,
    opts::SDPX.SolverOptions{T};
    phase::Symbol=:affine,
) where {T}
    return SDPX._lp_direction_accuracy_gate!(
        workspace,
        G,
        B,
        rhs,
        dx,
        dy,
        delta,
        workspace.correction_rhs,
        opts;
        phase=phase,
    )
end

@testset "N7 LP direction accuracy gate" begin
    @testset "dense LU route (equalities > 0) — $T" for T in (
        Float64,
        Float64x4,
    )
        variables, equalities = 2, 1
        G = Matrix{T}(I, variables, variables)
        B = reshape(T[1, 1], variables, equalities)
        H = T[2 0; 0 3]
        rhs = T[4, 9, 1]
        opts = lp_gate_options(T)
        workspace = SDPX.LPWorkspace(
            T,
            variables,
            variables,
            equalities;
            packed_hessian=false,
        )
        workspace.backend_formulation = :dense_lu
        workspace.H .= H

        # Exact δ = 0 direction: rho0 = rhoδ ≈ 0, gate passes.
        K0 = dense_kkt(H, B, zero(T))
        direction = K0 \ rhs
        saved = copy(rhs)
        saved_dx = copy(view(direction, 1:variables))
        saved_dy = copy(view(direction, (variables + 1):(variables + equalities)))
        rhs_reference = copy(rhs)
        dx_reference = copy(saved_dx)
        dy_reference = copy(saved_dy)
        dedicated_scratch = copy(workspace.correction_rhs)
        exact = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(saved_dx),
            copy(saved_dy),
            zero(T),
            opts,
        )
        @test exact.ok
        @test exact.reason === :none
        @test exact.eta_fact <= exact.tolerance
        @test exact.eta0 ≈ zero(T) atol=exact.tolerance
        @test exact.eta_fact ≈ exact.eta0
        @test exact.tau == SDPX._lp_direction_acceptance_tolerance(opts)
        @test exact.tau == max(sqrt(eps(T)), T(1e-8), T(1e-8))
        @test exact.eta_fact_normalized ≈
              exact.eta_fact / max(exact.scale_sigma, one(T))
        # Operator norm reconstruction matches the explicit K0.
        @test exact.k0_infinity ≈ opnorm(Matrix(K0), Inf)
        @test rhs == saved
        @test view(direction, 1:variables) == saved_dx
        @test view(direction, (variables + 1):(variables + equalities)) == saved_dy
        # The gate never mutates the RHS or the direction. The scratch is
        # caller-owned residual storage and intentionally holds `rho0`/`rhoδ`
        # when it returns.
        @test workspace.correction_rhs != dedicated_scratch
        @test rhs == rhs_reference
        @test saved_dx == dx_reference
        @test saved_dy == dy_reference
        # The gate never mutates the accepted factor's regularization state.
        @test workspace.direction_gate === exact

        # Known-shift exact solve: d solves Kδ, so rhoδ ≈ 0 while rho0 ≈ δd.
        delta = T(1e-2)
        Kδ = dense_kkt(H, B, delta)
        shifted = Kδ \ rhs
        accepted = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(view(shifted, 1:variables)),
            copy(view(shifted, (variables + 1):(variables + equalities))),
            delta,
            opts,
        )
        @test accepted.ok
        @test accepted.eta_fact <= accepted.tolerance
        @test accepted.eta0 ≈ accepted.eta_reg rtol=T(1e-6)
        @test accepted.eta_reg == abs(delta) * maximum(abs, shifted)
        @test accepted.scale_s0 ==
              SDPX.knrmInf(rhs) + accepted.k0_infinity * maximum(abs, shifted)
        @test accepted.scale_sigma ≈
              accepted.scale_s0 + abs(delta) * maximum(abs, shifted) rtol=T(1e-12)
        # The unregularized residual is dominated by the shift, and tau is δ-free.
        @test accepted.eta0 > accepted.eta_fact
        @test accepted.tau == exact.tau
        @test accepted.eta_fact_normalized ≈
              accepted.eta_fact / accepted.scale_sigma
        @test accepted.eta0_normalized ≈
              accepted.eta0 / accepted.scale_s0
        @test accepted.eta_reg_normalized ≈
              accepted.eta_reg / accepted.scale_s0

        # Injected true error: perturb one primal coordinate of the exact
        # shifted direction; rhoδ is now O(1e-2), far above tau*sδ.
        corrupted = copy(view(shifted, 1:variables))
        corrupted[1] += T(1e-2)
        rejected = run_gate(
            workspace,
            G,
            B,
            rhs,
            corrupted,
            copy(view(shifted, (variables + 1):(variables + equalities))),
            delta,
            opts,
        )
        @test !rejected.ok
        @test rejected.reason === :direction_residual_exceeded
        @test rejected.residual > rejected.tolerance
        @test rejected.eta_fact_normalized > rejected.tau
        @test workspace.direction_gate === rejected
    end

    @testset "dense LU route — BigFloat" begin
        setprecision(BigFloat, 256) do
            T = BigFloat
            variables, equalities = 2, 1
            G = Matrix{T}(I, variables, variables)
            B = reshape(T[1, 1], variables, equalities)
            H = T[2 0; 0 3]
            rhs = T[4, 9, 1]
            opts = lp_gate_options(T)
            workspace = SDPX.LPWorkspace(
                T,
                variables,
                variables,
                equalities;
                packed_hessian=false,
            )
            workspace.backend_formulation = :dense_lu
            workspace.H .= H

            delta = T("1e-2")
            Kδ = dense_kkt(H, B, delta)
            shifted = Kδ \ rhs
            accepted = run_gate(
                workspace,
                G,
                B,
                rhs,
                copy(view(shifted, 1:variables)),
                copy(view(shifted, (variables + 1):(variables + equalities))),
                delta,
                opts,
            )
            @test accepted.ok
            @test accepted.eta0 ≈ accepted.eta_reg rtol=T("1e-30")

            corrupted = copy(view(shifted, 1:variables))
            corrupted[1] += T("1e-2")
            rejected = run_gate(
                workspace,
                G,
                B,
                rhs,
                corrupted,
                copy(view(shifted, (variables + 1):(variables + equalities))),
                delta,
                opts,
            )
            @test !rejected.ok
            @test rejected.residual > rejected.tolerance
        end
    end

    @testset "equality-free dense Cholesky route — $T" for T in (
        Float64,
        Float64x4,
    )
        variables, equalities = 2, 0
        G = Matrix{T}(I, variables, variables)
        B = Matrix{T}(undef, variables, 0)
        H = T[2 0.5; 0.5 3]
        rhs = T[4, 9]
        opts = lp_gate_options(T)
        workspace = SDPX.LPWorkspace(
            T,
            variables,
            variables,
            equalities;
            packed_hessian=false,
        )
        workspace.backend_formulation = :positive_definite_cholesky
        workspace.H .= H

        delta = T(1e-2)
        Kδ = dense_kkt(H, B, delta)
        direction = Kδ \ rhs
        accepted = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(direction),
            T[],
            delta,
            opts,
        )
        @test accepted.ok
        @test accepted.eta_fact <= accepted.tolerance
        @test accepted.eta0 ≈ accepted.eta_reg rtol=T(1e-6)
        @test accepted.eta_fact_normalized <= accepted.tau
        @test accepted.k0_infinity ≈ opnorm(Matrix(Kδ - delta * I), Inf)

        corrupted = copy(direction)
        corrupted[2] += T(1e-2)
        rejected = run_gate(
            workspace,
            G,
            B,
            rhs,
            corrupted,
            T[],
            delta,
            opts,
        )
        @test !rejected.ok
        @test rejected.residual > rejected.tolerance
    end

    @testset "reduced standard-form route — $T" for T in (Float64, Float64x4)
        variables, equalities = 4, 2
        variable_for_row = [2, 1, 4, 3]
        row_for_variable = invperm(variable_for_row)
        values = T[1, 2, 3, 4]
        G = SDPX.LPDiagonalMatrix(values, variable_for_row, row_for_variable)
        B = T[1 0; 0 1; 1 1; 1 -1]
        weights = T[2, 3, 4, 5]
        rhs = T[1, 2, 3, 4, 5, 6]
        opts = lp_gate_options(T)
        workspace = SDPX.LPWorkspace(
            T,
            variables,
            variables,
            equalities;
            packed_hessian=false,
            reduced_standard_form=true,
        )
        workspace.backend_formulation = :diagonal_reduced_cholesky
        workspace.standard_system = SDPX.LPStandardFormSystem(
            G,
            B,
            1,
            :threaded_blocked_syrk,
        )
        workspace.weights .= weights

        H = Matrix{T}(I, variables, variables)
        @inbounds for variable in 1:variables
            row = row_for_variable[variable]
            H[variable, variable] = weights[row] * values[row] * values[row]
        end
        delta = T(1e-3)
        K0 = SDPX.alloc_zeros(T, variables + equalities, variables + equalities)
        SDPX._lp_populate_kkt!(K0, H, B, zero(T))
        Kδ = SDPX.alloc_zeros(T, variables + equalities, variables + equalities)
        SDPX._lp_populate_kkt!(Kδ, H, B, delta)
        direction = Kδ \ rhs
        accepted = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(view(direction, 1:variables)),
            copy(view(direction, (variables + 1):(variables + equalities))),
            delta,
            opts,
        )
        @test accepted.ok
        @test accepted.eta_fact <= accepted.tolerance
        @test accepted.eta0 ≈ accepted.eta_reg rtol=T(1e-5)
        @test accepted.eta_fact_normalized <= accepted.tau
        @test accepted.k0_infinity ≈ opnorm(Matrix(K0), Inf)

        corrupted = copy(view(direction, 1:variables))
        corrupted[3] += T(1e-2)
        rejected = run_gate(
            workspace,
            G,
            B,
            rhs,
            corrupted,
            copy(view(direction, (variables + 1):(variables + equalities))),
            delta,
            opts,
        )
        @test !rejected.ok
        @test rejected.residual > rejected.tolerance
    end

    @testset "sparse equality-free route" begin
        T = Float64
        variables, rows = 4, 6
        I = Int[]
        J = Int[]
        V = Float64[]
        for row in 1:rows
            first = mod(row - 1, variables) + 1
            second = mod(row, variables) + 1
            push!(I, row)
            push!(J, first)
            push!(V, 1.0)
            push!(I, row)
            push!(J, second)
            push!(V, 1.0)
        end
        G = sparse(I, J, V, rows, variables)
        B = spzeros(T, variables, 0)
        weights = collect(T, 1:rows) ./ T(7)
        rhs = fill(T(2), variables)
        opts = lp_gate_options(T)
        workspace = SDPX.LPWorkspace(
            T,
            rows,
            variables,
            0;
            packed_hessian=false,
            sparse_storage=true,
        )
        workspace.backend_formulation = :sparse_normal
        H0 = transpose(G) * (Diagonal(weights) * G)
        delta = T(1e-2)
        Kδ = SDPX._lp_sparse_assemble(
            H0,
            B,
            delta,
        )
        workspace.sparse_system = SDPX.LPSparseSystem{T}(
            G,
            B,
            Kδ,
            SDPX.GenericSparseCholeskyBackend(T),
            :sparse_normal,
            variables,
            0,
            false,
            nothing,
            nothing,
        )
        direction = Matrix(Kδ) \ rhs
        accepted = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(direction),
            T[],
            delta,
            opts,
        )
        @test accepted.ok
        @test accepted.eta_fact <= accepted.tolerance
        @test accepted.eta0 ≈ accepted.eta_reg rtol=1e-6
        @test accepted.eta_fact_normalized <= accepted.tau
        @test accepted.k0_infinity ≈ opnorm(Matrix(H0), Inf)

        # The sparse operator norm reads the authoritative lower triangle and
        # reuses caller-owned row-sum scratch.
        row_sums = zeros(T, variables)
        SDPX._lp_sparse_k0_infinity_norm(Kδ, delta, row_sums)
        norm_allocated = @allocated SDPX._lp_sparse_k0_infinity_norm(
            Kδ, delta, row_sums,
        )
        @test norm_allocated == 0

        # The sparse action accumulates in place without forming any dense
        # operator or per-direction array (no generic fallback is hidden).
        expected_action = rhs - Matrix(Kδ) * direction
        action_scratch = copy(rhs)
        SDPX._lp_sparse_regularized_action!(
            action_scratch,
            Kδ,
            direction,
            -1.0,
            1.0,
        )
        @test action_scratch ≈ expected_action atol=1e-12

        # Production frozen storage retains only the lower triangle. The
        # result and operator norm must remain identical, and an inactive
        # poisoned upper triangle must never be consulted.
        lower_Kδ = sparse(tril(Kδ))
        lower_scratch = copy(rhs)
        SDPX._lp_sparse_regularized_action!(
            lower_scratch,
            lower_Kδ,
            direction,
            -1.0,
            1.0,
        )
        @test lower_scratch ≈ expected_action atol=1e-12
        @test SDPX._lp_sparse_k0_infinity_norm(
            lower_Kδ, delta, row_sums,
        ) ≈ opnorm(Matrix(H0), Inf)

        corrupted = copy(direction)
        corrupted[1] += 1e-2
        rejected = run_gate(
            workspace,
            G,
            B,
            rhs,
            corrupted,
            T[],
            delta,
            opts,
        )
        @test !rejected.ok
        @test rejected.residual > rejected.tolerance

        # δ = 0 through the same retained-operator measurement.
        K0_sparse = SDPX._lp_sparse_assemble(H0, B, zero(T))
        workspace.sparse_system = SDPX.LPSparseSystem{T}(
            G,
            B,
            K0_sparse,
            SDPX.GenericSparseCholeskyBackend(T),
            :sparse_normal,
            variables,
            0,
            false,
            nothing,
            nothing,
        )
        unshifted = Matrix(K0_sparse) \ rhs
        exact = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(unshifted),
            T[],
            zero(T),
            opts,
        )
        @test exact.ok
        @test exact.eta0 ≈ 0.0 atol=exact.tolerance
        @test exact.eta_fact ≈ exact.eta0
        @test exact.k0_infinity ≈ opnorm(Matrix(K0_sparse), Inf)
    end

    @testset "sparse equality-free route — BigFloat" begin
        setprecision(BigFloat, 256) do
            T = BigFloat
            variables, rows = 4, 6
            I = Int[]
            J = Int[]
            V = T[]
            for row in 1:rows
                first = mod(row - 1, variables) + 1
                second = mod(row, variables) + 1
                push!(I, row)
                push!(J, first)
                push!(V, T(1))
                push!(I, row)
                push!(J, second)
                push!(V, T(1))
            end
            G = sparse(I, J, V, rows, variables)
            B = spzeros(T, variables, 0)
            weights = T.(1:rows) ./ T(7)
            rhs = fill(T(2), variables)
            opts = lp_gate_options(T)
            workspace = SDPX.LPWorkspace(
                T,
                rows,
                variables,
                0;
                packed_hessian=false,
                sparse_storage=true,
            )
            workspace.backend_formulation = :sparse_normal
            H0 = transpose(G) * (Diagonal(weights) * G)
            delta = T("1e-2")
            Kδ = SDPX._lp_sparse_assemble(H0, B, delta)
            workspace.sparse_system = SDPX.LPSparseSystem{T}(
                G,
                B,
                Kδ,
                SDPX.GenericSparseCholeskyBackend(T),
                :sparse_normal,
                variables,
                0,
                false,
                nothing,
                nothing,
            )
            direction = Matrix(Kδ) \ rhs
            accepted = run_gate(
                workspace,
                G,
                B,
                rhs,
                copy(direction),
                T[],
                delta,
                opts,
            )
            @test accepted.ok
            @test accepted.eta_fact <= accepted.tolerance
            @test accepted.eta0 ≈ accepted.eta_reg rtol=T("1e-30")
            @test accepted.eta_fact_normalized <= accepted.tau
            @test accepted.k0_infinity ≈ opnorm(Matrix(H0), Inf)

            corrupted = copy(direction)
            corrupted[2] += T("1e-2")
            rejected = run_gate(
                workspace,
                G,
                B,
                rhs,
                corrupted,
                T[],
                delta,
                opts,
            )
            @test !rejected.ok
            @test rejected.residual > rejected.tolerance
        end
    end

    @testset "fail closed on unknown, non-finite, and sparse-equality routes" begin
        T = Float64
        variables, equalities = 2, 1
        G = Matrix{T}(I, variables, variables)
        B = reshape(T[1, 1], variables, equalities)
        H = T[2 0; 0 3]
        rhs = T[4, 9, 1]
        delta = T(1e-2)
        opts = lp_gate_options(T)
        direction = dense_kkt(H, B, delta) \ rhs

        workspace = SDPX.LPWorkspace(
            T,
            variables,
            variables,
            equalities;
            packed_hessian=false,
        )
        workspace.H .= H
        unknown = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(view(direction, 1:variables)),
            copy(view(direction, (variables + 1):(variables + equalities))),
            delta,
            opts,
        )
        @test !unknown.ok
        @test unknown.reason === :unknown_lp_route
        @test !isfinite(unknown.tolerance)

        workspace.backend_formulation = :dense_lu
        nonfinite = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(view(direction, 1:variables)),
            copy(view(direction, (variables + 1):(variables + equalities))),
            T(Inf),
            opts,
        )
        @test !nonfinite.ok
        @test nonfinite.reason === :nonfinite_direction_data

        workspace.H[1, 1] = T(Inf)
        nonfinite_operator = run_gate(
            workspace,
            G,
            B,
            rhs,
            copy(view(direction, 1:variables)),
            copy(view(direction, (variables + 1):(variables + equalities))),
            delta,
            opts,
        )
        @test !nonfinite_operator.ok
        @test nonfinite_operator.reason === :nonfinite_operator_data
        @test !isfinite(nonfinite_operator.tolerance)

        # Exact zero RHS and zero direction: both denominators are zero and
        # the exact zero residual must be accepted with zero ratios.
        workspace.H[1, 1] = T(2)
        zero_gate = run_gate(
            workspace,
            G,
            B,
            zeros(T, variables + equalities),
            zeros(T, variables),
            zeros(T, equalities),
            delta,
            opts,
        )
        @test zero_gate.ok
        @test zero_gate.reason === :none
        @test zero_gate.eta_fact == zero(T)
        @test zero_gate.scale_s0 == zero(T)
        @test zero_gate.scale_sigma == zero(T)
        @test zero_gate.eta_fact_normalized == zero(T)
        @test zero_gate.eta0_normalized == zero(T)
        @test zero_gate.eta_reg_normalized == zero(T)

        sparse_workspace = SDPX.LPWorkspace(
            T,
            variables,
            variables,
            equalities;
            packed_hessian=false,
            sparse_storage=true,
        )
        sparse_workspace.backend_formulation = :sparse_normal
        sparse_workspace.sparse_system = SDPX.LPSparseSystem{T}(
            sparse(Matrix{T}(I, variables, variables)),
            sparse(B),
            spdiagm(0 => fill(T(1), variables + equalities)),
            SDPX.GenericSparseCholeskyBackend(T),
            :sparse_normal,
            variables,
            equalities,
            false,
            nothing,
            nothing,
        )
        unsupported = run_gate(
            sparse_workspace,
            G,
            B,
            rhs,
            copy(view(direction, 1:variables)),
            copy(view(direction, (variables + 1):(variables + equalities))),
            delta,
            opts,
        )
        @test !unsupported.ok
        @test unsupported.reason === :sparse_equality_unsupported
    end

    @testset "default regularized LP trajectory is unchanged" begin
        problem = SDPX.linear_program(
            [1.0, 2.0],
            [1.0 0.0; 0.0 1.0; 1.0 1.0],
            [1.0, 1.0, 3.0];
            Aeq=[1.0 1.0],
            beq=[3.0],
            sparse=false,
            verbosity=0,
        )
        result = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:lp,
                presolve=false,
                scaling=:none,
                diagnostics=true,
                verbosity=0,
            ),
        )
        @test result.status == SDPX.Optimal
        @test result.x ≈ [2.0, 1.0] rtol=1e-7
        @test result.pObj ≈ 4.0 rtol=1e-7
        @test result.regularizations == 0
        @test result.iterations > 0
        @test length(result.parameter_history) == result.iterations
        @test all(
            row -> row.factorization_quality == 1.0 &&
                   row.refinement_count == 0 &&
                   row.regularization == SDPX._lp_regularization_floor(Float64) &&
                   row.fallback_reason === :none,
            result.parameter_history,
        )
        @test result.termination.reason === :none
        @test result.termination.stage === :none
        @test result.termination.executed.fallback_reason === :none
        # The automatic cold start still factors exactly once and solves two
        # RHS; the Newton loop performs one factor plus two RHS solves per
        # accepted iteration with no gate-induced correction/refinement.
        initialization = result.termination.executed.initialization
        @test initialization.applied
        @test initialization.path === :phase2_kkt_cold_start
        @test initialization.factorization_attempts == 1
        @test initialization.factorization_count == 1
        @test initialization.rhs_solve_count == 2
        @test result.diagnostics.selected_algorithms.kkt === :dense_lu
        @test result.diagnostics.selected_algorithms.la_factorization === :lu
    end
end
