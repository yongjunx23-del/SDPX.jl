using Test
using LinearAlgebra
using TOML
using SDPX

isdefined(@__MODULE__, :SDPXBenchmarkRegistry) || include(joinpath(
    @__DIR__, "..", "benchmark", "SDPXBenchmarkRegistry.jl",
))
const BR = SDPXBenchmarkRegistry
const AB = BR.AB

@testset "analytic benchmark inventory" begin
    @test AB.family_names() == (
        :chebyshev_lp,
        :weighted_minimum_norm_socp,
        :basel_soc_chain,
        :spectral_path_sdp,
        :odd_cycle_maxcut_sdp,
        :rational_moment_sdp,
    )
    for family in AB.family_names()
        @test AB.build_problem(family; tier=:tier1).input_fingerprint isa String
    end
    for degree in (8, 16, 24, 32, 48, 64, 96, 128),
        basis in (:chebyshev, :monomial)
        @test BR.benchmark_spec(
            "analytic/chebyshev/n$(degree)/$(basis)",
        ).loader === :analytic
    end
    for vertices in (5, 7, 15, 31, 63, 127, 255), form in (:clean, :redundant)
        @test BR.benchmark_spec(
            "analytic/maxcut/n$(vertices)/$(form)",
        ).loader === :analytic
    end
    for rho_power in (1, 2, 4, 8, 16, 32), order in (4, 8, 12, 16, 24, 32)
        @test BR.benchmark_spec(
            "analytic/moment/m$(rho_power)/order$(order)/lower",
        ).loader === :analytic
    end
    @test AB.analytic_objective(:chebyshev_lp; tier=:tier2) == 2.0^(1 - 32)
    @test AB.analytic_objective(:basel_soc_chain; tier=:tier2) ≈
          AB.build_problem(:basel_soc_chain; tier=:tier2).oracle.objective
    @test AB.analytic_objective(:rational_moment_sdp; m=8) ≈
          AB.analytic_objective(:rational_moment_sdp; rho_power=8)
end

@testset "Chebyshev LP oracle" begin
    for basis in (:chebyshev, :monomial)
        case = AB.build_problem(:chebyshev_lp; degree=6, basis)
        n = case.parameters.degree
        t = case.oracle.primal[end]
        @test t == 2.0^(1 - n)
        for j in 0:n
            x = cos(pi * j / n)
            values = basis === :monomial ? [x^k for k in 0:(n - 1)] :
                     [cos(k * acos(x)) for k in 0:(n - 1)]
            polynomial = dot(values, case.oracle.primal[1:n])
            @test polynomial + t + 1e-12 >= x^n
            @test -polynomial + t + 1e-12 >= -x^n
        end
    end
end

@testset "MultiFloat analytic constants use extended precision" begin
    if haskey(BR.MULTIFLOAT_TYPES, :float64x4)
        T = BR.MULTIFLOAT_TYPES[:float64x4]
        spec = BR.benchmark_spec("analytic/chebyshev/n8/chebyshev")
        built = BR.build_problem(spec, T)
        node = AB._cos_t(T, T(pi) / T(8))
        node_reference = setprecision(BigFloat, 600) do
            cos(BigFloat(pi) / BigFloat(8))
        end
        maxcut = AB.build_problem(
            :odd_cycle_maxcut_sdp; T=T, vertices=5,
        )
        gram_coefficient = maxcut.oracle.X[1, 2]
        gram_reference = setprecision(BigFloat, 600) do
            cos(BigFloat(pi) - BigFloat(pi) / BigFloat(5))
        end

        @test built.input_generation_precision_bits ==
              2 * SDPX.sig_bits(T) + 64
        @test abs(BigFloat(node) - node_reference) <
              BigFloat(2)^-SDPX.sig_bits(T)
        @test abs(BigFloat(gram_coefficient) - gram_reference) <
              BigFloat(2)^-SDPX.sig_bits(T)
    else
        @info "MultiFloats unavailable; skipping extended-precision constant check"
    end
end

@testset "SOCP and PSD equivalent oracles" begin
    weighted = AB.build_problem(
        :weighted_minimum_norm_socp; dimension=8, spread=4,
    )
    @test isapprox(sum(weighted.oracle.x_star), 1; atol=1e-14)
    @test isapprox(
        weighted.oracle.objective,
        AB.analytic_objective(
            :weighted_minimum_norm_socp; dimension=8, spread=4,
        );
        rtol=1e-14,
    )

    native = AB.build_problem(
        :basel_soc_chain; terms=8, representation=:native,
    )
    lifted = AB.build_problem(
        :basel_soc_chain; terms=8, representation=:psd2,
    )
    scaled = AB.build_problem(
        :basel_soc_chain; terms=8, representation=:native, spread=40,
    )
    @test isapprox(native.oracle.objective, lifted.oracle.objective; rtol=1e-14)
    @test isapprox(native.oracle.objective, scaled.oracle.objective; rtol=1e-14)
end

@testset "SDP analytic answers" begin
    spectral = AB.build_problem(:spectral_path_sdp; path_length=8)
    degenerate = AB.build_problem(
        :spectral_path_sdp; path_length=8, delta="0", near_degenerate=true,
    )
    near = AB.build_problem(
        :spectral_path_sdp; path_length=8, delta_power=40,
        near_degenerate=true,
    )
    @test isapprox(spectral.oracle.objective, 2cos(pi / 9); rtol=1e-14)
    @test spectral.problem.dims.n == 1
    @test degenerate.problem.dims.k == [16]
    @test near.problem.dims.k == [16]
    @test near.parameters.delta == exp2(-40)
    @test isapprox(degenerate.oracle.objective, spectral.oracle.objective; rtol=1e-14)
    @test isapprox(near.oracle.objective, spectral.oracle.objective; rtol=1e-14)

    clean = AB.build_problem(:odd_cycle_maxcut_sdp; vertices=7)
    redundant = AB.build_problem(
        :odd_cycle_maxcut_sdp; vertices=7, redundant=true,
    )
    @test isapprox(
        clean.oracle.objective,
        7 / 2 * (1 + cos(pi / 7));
        rtol=1e-14,
    )
    @test all(isapprox(value, 0.5; atol=0, rtol=0)
              for value in clean.problem.c if !iszero(value))
    @test redundant.problem.dims.n == 8
    @test redundant.oracle.X[1, 1] == 1
end

@testset "moment hierarchy oracle" begin
    lower = AB.build_problem(
        :rational_moment_sdp; order=4, bound=:lower, rho_power=8,
    )
    upper = AB.build_problem(
        :rational_moment_sdp; order=4, bound=:upper, rho_power=8,
    )
    exact = -log(1 - lower.parameters.rho) / lower.parameters.rho
    @test isapprox(lower.oracle.exact_integral, exact; rtol=1e-14)
    @test lower.oracle.primal[1] == upper.oracle.primal[1]
    @test upper.oracle.physical_objective(upper.oracle.primal) ==
          upper.oracle.exact_integral
    for k in 0:(length(lower.oracle.primal) - 2)
        @test isapprox(
            lower.oracle.primal[k + 1] -
            lower.parameters.rho * lower.oracle.primal[k + 2],
            1 / (k + 1);
            rtol=1e-12,
        )
    end

    @test lower.problem.c[1] == 1
    @test upper.problem.c[1] == -1
    @test lower.problem.C == upper.problem.C
    @test lower.problem.B == upper.problem.B
    @test lower.problem.b == upper.problem.b
    @test lower.problem.cons.Asp == upper.problem.cons.Asp
    @test isapprox(
        lower.problem.B' * lower.oracle.primal,
        lower.problem.b;
        atol=1e-11,
    )
    moment_blocks = (
        AB._moment_block(Float64, 5, 10, :moment),
        AB._moment_block(Float64, 4, 10, :x),
        AB._moment_block(Float64, 4, 10, :one_minus_x),
    )
    for block in moment_blocks
        matrix = sum(
            block[variable] * lower.oracle.primal[variable]
            for variable in eachindex(block)
        )
        @test isfinite(minimum(eigvals(Symmetric(Matrix(matrix)))))
        @test minimum(eigvals(Symmetric(Matrix(matrix)))) > -1e-9
    end
    @test_throws ArgumentError AB.build_problem(
        :rational_moment_sdp; order=4, rho_power=54,
    )
end

function _fake_result_row(;
    problem_id,
    objective,
    direction=:exact,
    equivalence_group=missing,
    monotonic_group=missing,
    bound_group=missing,
    classification=:PASS,
)
    values = Dict{Symbol,Any}(field => missing for field in BR.RESULT_COLUMNS)
    merge!(values, Dict(
        :problem_id => problem_id,
        :arithmetic => :float64,
        :requested_provider => :auto,
        :precision_bits => 53,
        :status => :Optimal,
        :certificate_valid => true,
        :objective => string(objective),
        :analytic_direction => direction,
        :analytic_equivalence_group => equivalence_group,
        :analytic_monotonic_group => monotonic_group,
        :analytic_bound_group => bound_group,
        :analytic_absolute_tolerance => "0.0",
        :analytic_relative_tolerance => "1e-8",
        :classification => classification,
        :semantic_pass => classification === :PASS,
        :eligible_for_performance => classification === :PASS,
        :semantic_failures => "",
        :group_failures => "",
    ))
    return NamedTuple{BR.RESULT_COLUMNS}(
        Tuple(values[field] for field in BR.RESULT_COLUMNS),
    )
end

@testset "analytic group gates" begin
    equivalent = BR._apply_analytic_group_gates([
        _fake_result_row(
            problem_id="analytic/chebyshev/n8/chebyshev",
            objective=2.0^-7,
            equivalence_group="chebyshev/n8",
        ),
        _fake_result_row(
            problem_id="analytic/chebyshev/n8/monomial",
            objective=2.0^-7 * (1 + 1e-10),
            equivalence_group="chebyshev/n8",
        ),
    ])
    @test all(row -> row.equivalence_gate_valid === true, equivalent)
    @test all(row -> row.eligible_for_performance, equivalent)

    mismatch = BR._apply_analytic_group_gates([
        _fake_result_row(
            problem_id="analytic/chebyshev/n8/chebyshev",
            objective=2.0^-7,
            equivalence_group="chebyshev/n8",
        ),
        _fake_result_row(
            problem_id="analytic/chebyshev/n8/monomial",
            objective=2.0^-6,
            equivalence_group="chebyshev/n8",
        ),
    ])
    @test all(row -> row.equivalence_gate_valid === false, mismatch)
    @test all(row -> row.classification === :FAIL, mismatch)
    @test all(row -> !row.eligible_for_performance, mismatch)

    monotone = BR._apply_analytic_group_gates([
        _fake_result_row(
            problem_id="analytic/moment/m4/order4/lower",
            objective=1.0, direction=:lower,
            monotonic_group="moment/m4/lower",
        ),
        _fake_result_row(
            problem_id="analytic/moment/m4/order8/lower",
            objective=1.1, direction=:lower,
            monotonic_group="moment/m4/lower",
        ),
    ])
    @test all(row -> row.monotonicity_gate_valid === true, monotone)

    bracket = BR._apply_analytic_group_gates([
        _fake_result_row(
            problem_id="analytic/moment/m4/order4/lower",
            objective=1.0, direction=:lower,
            bound_group="moment/m4/order4",
        ),
        _fake_result_row(
            problem_id="analytic/moment/m4/order4/upper",
            objective=1.2, direction=:upper,
            bound_group="moment/m4/order4",
        ),
    ])
    @test all(row -> row.bound_pair_gate_valid === true, bracket)

    nonfinite = BR._apply_analytic_group_gates([
        _fake_result_row(
            problem_id="analytic/moment/m4/order4/lower",
            objective=Inf, direction=:lower,
            bound_group="moment/m4/order4",
        ),
        _fake_result_row(
            problem_id="analytic/moment/m4/order4/upper",
            objective=1.2, direction=:upper,
            bound_group="moment/m4/order4",
        ),
    ])
    @test all(row -> row.bound_pair_gate_valid === false, nonfinite)
    @test all(row -> !row.eligible_for_performance, nonfinite)
end

@testset "canonical analytic gate rejects false optimal" begin
    output = tempname() * ".toml"
    run = BR.run_suite(
        :analytic_numerical;
        problem="analytic/chebyshev/n32/chebyshev",
        arithmetic=:float64,
        provider=:auto,
        output=output,
        warmup=false,
        allow_semantic_failures=true,
    )
    row = only(run.rows)
    @test row.status === :Optimal
    @test row.certificate_valid
    @test parse(Float64, row.objective_relative_error) >
          parse(Float64, row.analytic_relative_tolerance)
    @test row.classification === :FAIL
    @test !row.eligible_for_performance
    @test row.b_correct < 0
    @test occursin("objective", row.semantic_failures)
    @test run.failure_map !== nothing
end

@testset "augmented KKT policy is a structured capability skip" begin
    spec = BR.benchmark_spec(
        "analytic/spectral/n64/single/formulation-augmented",
    )
    @test :capability_skip in spec.tags
    @test spec.parameters.capability_requirement ===
          :general_dense_augmented_kkt
    output = tempname() * ".toml"
    run = BR.run_suite(
        :analytic_numerical;
        problem=spec.id,
        arithmetic=:float64,
        provider=:auto,
        output=output,
        warmup=false,
    )
    row = only(run.rows)
    @test row.status === :skipped
    @test row.skip_reason === :augmented_backend_capability_unavailable
    @test row.classification === :UNRESOLVED
    @test !row.eligible_for_performance
    @test row.group_failures == ""
end

@testset "canonical certificate and precision facts" begin
    spec = BR.benchmark_spec("analytic/spectral/n8/single")
    built = BR.build_problem(spec, Float64)
    certificate = BR._safe_certificate(built.problem, nothing, Float64, built)
    @test certificate.value === nothing
    @test certificate.error isa AbstractString

    output = tempname() * ".toml"
    run = BR.run_suite(
        :analytic_fast;
        problem=spec.id,
        output=output,
        warmup=false,
        allow_semantic_failures=true,
    )
    row = only(run.rows)
    @test row.certificate_valid
    @test row.certificate_kind === :optimality
    @test row.validation_precision_bits == 52
    @test row.complementarity !== missing
    @test row.relative_complementarity !== missing
    @test row.dual_cone_violation !== missing
    @test row.nonzeros isa Integer
    @test row.cone_composition == "psd8x1"
    @test row.workspace_bytes isa Integer
    @test row.independent_validation_seconds >= 0
    @test row.end_to_end_seconds >= row.total_seconds
end
