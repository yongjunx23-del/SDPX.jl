# Analytic benchmark adapter. The family builders own the mathematics; this
# file only describes cases for the canonical registry and runner.
include(joinpath(ROOT, "analytic", "AnalyticFamilies.jl"))
const AB = SDPXAnalyticBenchmarks

_analytic_suite(tier::Symbol) = tier === :tier1 ? :analytic_fast :
                                tier === :tier2 ? :analytic_numerical :
                                :analytic_stress

function _analytic_spec(
    id::AbstractString,
    analytic_family::Symbol,
    family::Symbol,
    problem_type::Symbol,
    tier::Symbol,
    purpose::Symbol,
    analytic_parameters::NamedTuple;
    tags=(),
    size=(;),
    equivalence_group=nothing,
    monotonic_group=nothing,
    bound_group=nothing,
    solve_settings=(;),
    oracle_tolerance=nothing,
    capability_requirement=nothing,
    skip_reason=nothing,
)
    parameters = (
        analytic_family=analytic_family,
        tier=tier,
        analytic_parameters=analytic_parameters,
        source_parameters=analytic_parameters,
        equivalence_group=equivalence_group,
        monotonic_group=monotonic_group,
        bound_group=bound_group,
        solve_settings=solve_settings,
        oracle_tolerance=oracle_tolerance,
        capability_requirement=capability_requirement,
        skip_reason=skip_reason,
    )
    return BenchmarkSpec(
        String(id),
        replace(String(id), '/' => ' '),
        family,
        problem_type,
        :synthetic,
        (_analytic_suite(tier),),
        (:analytic, analytic_family, tags...),
        purpose,
        nothing,
        :analytic,
        parameters,
        BenchmarkReference(
            :optimal,
            nothing,
            1.0e-8,
            1.0e-8,
            "independent analytic oracle; original-coordinate certificate required",
        ),
        size,
        nothing,
    )
end

function _push_analytic!(specs, args...; kwargs...)
    push!(specs, _analytic_spec(args...; kwargs...))
    return specs
end

const ANALYTIC_SPECS = let specs = BenchmarkSpec[]
    # Chebyshev LP: every requested degree in stable and pathological bases.
    for degree in (8, 16, 24, 32, 48, 64, 96, 128)
        tier = degree <= 16 ? :tier1 : degree <= 64 ? :tier2 : :tier3
        for basis in (:chebyshev, :monomial)
            _push_analytic!(
                specs,
                "analytic/chebyshev/n$(degree)/$(basis)",
                :chebyshev_lp, :lp, :linear_program, tier,
                :chebyshev_precision,
                (degree=degree, basis=basis);
                equivalence_group="chebyshev/n$(degree)",
                tags=basis === :chebyshev ? (:stable_basis,) :
                     (:monomial_basis, :pathology),
                size=(variables=degree + 1, degree=degree),
            )
        end
    end

    # One growing Lorentz cone. `spread` is the total base-2 range after the
    # harmless symmetric normalization used by the builder.
    for dimension in (32, 128, 512, 2048, 8192), spread in (0, 10, 20, 40)
        tier = dimension <= 128 ? :tier1 : dimension <= 2048 ? :tier2 : :tier3
        _push_analytic!(
            specs,
            "analytic/weighted_socp/n$(dimension)/spread$(spread)",
            :weighted_minimum_norm_socp, :socp,
            :second_order_cone_program, tier, :single_large_soc,
            (dimension=dimension, spread=spread);
            tags=spread == 0 ? (:large_cone,) : (:large_cone, :pathology),
            size=(variables=dimension + 1, cone_dimension=dimension + 1),
        )
    end

    # Basel chain: formulation and homogeneous rescaling are mathematically
    # equivalent, so all representations at a fixed N share one group.
    for terms in (10, 100, 1_000, 10_000),
        representation in (:native, :psd2), spread in (0, 10, 20, 40)
        tier = terms <= 10 ? :tier1 : terms <= 1_000 ? :tier2 : :tier3
        kind = representation === :native ? :socp : :sdp
        problem_type = representation === :native ?
                       :second_order_cone_program : :semidefinite_program
        _push_analytic!(
            specs,
            "analytic/basel/n$(terms)/$(representation)/spread$(spread)",
            :basel_soc_chain, kind, problem_type, tier,
            representation === :native ? :many_small_cones : :many_small_psd2,
            (terms=terms, representation=representation, spread=spread);
            equivalence_group="basel/n$(terms)",
            tags=(representation, spread == 0 ? :unscaled : :pathology),
            size=(variables=terms, cones=terms, block_dimension=3),
        )
    end

    # Spectral SDP: the single block and every direct-sum delta have the same
    # largest eigenvalue. `delta_power=p` means delta=2^-p exactly in T.
    for path_length in (8, 16, 32, 64, 128, 256)
        tier = path_length <= 16 ? :tier1 : path_length <= 128 ? :tier2 : :tier3
        group = "spectral/n$(path_length)"
        _push_analytic!(
            specs,
            "analytic/spectral/n$(path_length)/single",
            :spectral_path_sdp, :sdp, :semidefinite_program, tier,
            :large_psd_few_equalities,
            (path_length=path_length, delta="0", near_degenerate=false);
            equivalence_group=group,
            tags=(:spectral,),
            size=(psd_dimension=path_length, equalities=1),
        )
        _push_analytic!(
            specs,
            "analytic/spectral/n$(path_length)/direct_sum/delta0",
            :spectral_path_sdp, :sdp, :semidefinite_program, tier,
            :near_degenerate_sdp,
            (path_length=path_length, delta="0", near_degenerate=true);
            equivalence_group=group,
            tags=(:spectral, :exact_degeneracy),
            size=(psd_dimension=2path_length, equalities=1),
        )
        for power in (10, 20, 40, 80, 160)
            _push_analytic!(
                specs,
                "analytic/spectral/n$(path_length)/direct_sum/delta2m$(power)",
                :spectral_path_sdp, :sdp, :semidefinite_program, tier,
                :near_degenerate_sdp,
                (path_length=path_length, delta_power=power,
                 near_degenerate=true);
                equivalence_group=group,
                tags=(:spectral, :near_degenerate),
                size=(psd_dimension=2path_length, equalities=1),
            )
        end
    end

    # Odd-cycle MaxCut with and without one exact redundant equality.
    for vertices in (5, 7, 15, 31, 63, 127, 255), redundant in (false, true)
        tier = vertices <= 7 ? :tier1 : vertices <= 63 ? :tier2 : :tier3
        form = redundant ? :redundant : :clean
        _push_analytic!(
            specs,
            "analytic/maxcut/n$(vertices)/$(form)",
            :odd_cycle_maxcut_sdp, :sdp, :semidefinite_program, tier,
            :odd_cycle_degeneracy,
            (vertices=vertices, redundant=redundant);
            equivalence_group="maxcut/n$(vertices)",
            tags=redundant ? (:low_rank, :redundant_equality) : (:low_rank,),
            size=(psd_dimension=vertices,
                  equalities=vertices + (redundant ? 1 : 0)),
        )
    end

    # Moment hierarchy: full requested rho/order grid, with independent lower
    # and upper rows plus series metadata for post-run tightening checks.
    for rho_power in (1, 2, 4, 8, 16, 32),
        order in (4, 8, 12, 16, 24, 32), bound in (:lower, :upper)
        tier = order <= 8 && rho_power <= 4 ? :tier1 :
               order <= 24 && rho_power <= 16 ? :tier2 : :tier3
        _push_analytic!(
            specs,
            "analytic/moment/m$(rho_power)/order$(order)/$(bound)",
            :rational_moment_sdp, :sdp, :semidefinite_program, tier,
            :moment_hierarchy,
            (order=order, rho_power=rho_power, bound=bound);
            monotonic_group="moment/m$(rho_power)/$(bound)",
            bound_group="moment/m$(rho_power)/order$(order)",
            tags=(:moment, Symbol("bound_$(bound)")),
            size=(order=order, moments=2order + 2),
        )
    end

    # A small, explicit policy matrix. These are ordinary SolveOptions routed
    # through the same planner; they do not create another decision system.
    for scaling in (:none, :equilibrate)
        _push_analytic!(
            specs,
            "analytic/chebyshev/n32/monomial/scaling-$(scaling)",
            :chebyshev_lp, :lp, :linear_program, :tier2,
            :scaling_policy,
            (degree=32, basis=:monomial);
            equivalence_group="chebyshev/n32",
            solve_settings=(scaling=scaling,),
            tags=(:monomial_basis, :scaling_comparison),
            size=(variables=33, degree=32),
        )
        _push_analytic!(
            specs,
            "analytic/weighted_socp/n512/spread40/scaling-$(scaling)",
            :weighted_minimum_norm_socp, :socp,
            :second_order_cone_program, :tier2, :scaling_policy,
            (dimension=512, spread=40);
            solve_settings=(scaling=scaling,),
            tags=(:large_cone, :pathology, :scaling_comparison),
            size=(variables=513, cone_dimension=513),
        )
    end
    for formulation in (:primal, :augmented)
        _push_analytic!(
            specs,
            "analytic/spectral/n64/single/formulation-$(formulation)",
            :spectral_path_sdp, :sdp, :semidefinite_program, :tier2,
            :kkt_policy,
                (path_length=64, delta="0", near_degenerate=false);
                equivalence_group="spectral/n64",
                solve_settings=(formulation=formulation,),
                tags=formulation === :augmented ?
                     (:spectral, :kkt_comparison, :capability_skip) :
                     (:spectral, :kkt_comparison),
                # The planner deliberately rejects augmented sparse/block-arrow
                # routes today. Keep this row visible as a capability request
                # rather than turning it into an execution error.
                capability_requirement=:general_dense_augmented_kkt,
                skip_reason=:augmented_backend_capability_unavailable,
                size=(psd_dimension=64, equalities=1),
            )
        end
    specs
end

function _build_analytic_problem(
    ::Type{T};
    analytic_family,
    tier=:tier1,
    analytic_parameters=(;),
    source_parameters=nothing,
    solve_settings=(;),
    equivalence_group=nothing,
    monotonic_group=nothing,
    bound_group=nothing,
    oracle_tolerance=nothing,
    capability_requirement=nothing,
    skip_reason=nothing,
) where {T}
    case = AB.build_problem(
        analytic_family; T=T, tier=tier, analytic_parameters...,
    )
    oracle = case.oracle
    kind = get(oracle, :objective_kind, :exact)
    direction = kind === :bound ? get(oracle, :bound, :exact) : :exact
    reference = kind === :bound ? get(oracle, :exact_integral, nothing) :
                get(oracle, :objective, nothing)
    expected = kind === :exact ? reference : nothing

    default_tolerance = T === Float64 ? "1e-8" : "1e-20"
    settings = merge(
        (tolerance=default_tolerance, maximum_iterations=200, max_time=Inf),
        solve_settings,
    )
    requested = parse(
        T,
        string(oracle_tolerance === nothing ? settings.tolerance : oracle_tolerance),
    )
    floor = reference === nothing ? eps(one(T)) :
            (iszero(reference) ? eps(one(T)) : eps(abs(reference)))
    contract = (
        family=case.family,
        kind=kind,
        direction=direction,
        reference=reference,
        absolute_tolerance=requested * floor,
        relative_tolerance=requested,
        residual_tolerance=parse(T, string(settings.tolerance)),
        equivalence_group=equivalence_group,
        monotonic_group=monotonic_group,
        bound_group=bound_group,
    )
    return (
        problem=case.problem,
        expected=expected,
        kind=case.kind,
        physical_objective=get(oracle, :physical_objective, identity),
        analytic_contract=contract,
        source_parameters=source_parameters === nothing ?
                          case.parameters : source_parameters,
        solve_settings=settings,
        benchmark_scale=tier,
        input_generation_precision_bits=T === BigFloat ?
                                        precision(BigFloat) :
                                        (T === Float64 ? 53 :
                                         2 * SDPX.sig_bits(T) + 64),
    )
end
