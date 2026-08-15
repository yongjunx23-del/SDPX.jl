const FULL_UNITARITY_EFT_REFERENCE = (
    lower="30.4732058529286002611344264526887472",
    upper="30.4732058529379858416561499463130361",
)

const FULL_UNITARITY_EFT_BASE_PARAMETERS = (
    l_max=40,
    N_a=15,
    N_mu=200,
    N_x=2,
    N_alpha=2,
)

function _full_unitarity_pending_rung(scale::Int)
    parameters = (
        l_max=40 * scale,
        N_a=15 * scale,
        N_mu=200 * scale,
        N_x=2 * scale,
        N_alpha=2 * scale,
    )
    suffix = "j$(parameters.l_max)_na$(parameters.N_a)_nmu$(parameters.N_mu)" *
             "_nx$(parameters.N_x)_nalpha$(parameters.N_alpha)"
    return BenchmarkSpec(
        "csdr/full_unitarity_eft_" * suffix,
        "Full-unitarity EFT " * uppercasefirst(suffix),
        :socp,
        :second_order_cone_program,
        :csdr,
        (:heavy,),
        (:application, :bootstrap, :fixed_trace_q3, :parameter_ladder,
         :register_only, :pending_artifact),
        :full_unitarity_scaling_ladder,
        nothing,
        :csdr_fixed_trace_reduced_v1,
        (
            benchmark_scale=scale,
            source_parameters=parameters,
            generation_rule=:regenerate_source_model,
        ),
        BenchmarkReference(
            :unknown,
            nothing,
            Inf,
            Inf,
            "Register-only source-model rung. Generate and pin an independent " *
            "neutral payload before execution; never duplicate J40 matrices.",
        ),
        (local_available=false, geometry=:pending_regeneration),
        ExternalSource(
            "Full-unitarity-EFT",
            "",
            "full-unitarity-eft-$(suffix)-v1.bin",
            :csdr_fixed_trace_reduced_v1,
            nothing,
            "Private scientific input; metadata only until an independently " *
            "generated neutral payload is approved.",
        ),
    )
end

const FULL_UNITARITY_EFT_SPECS = BenchmarkSpec[
    BenchmarkSpec(
        "csdr/full_unitarity_eft_j40_na15_nmu200_nx2_nalpha2",
        "Full-unitarity EFT J40 Na15 Nmu200 Nx2 Nalpha2",
        :socp,
        :second_order_cone_program,
        :csdr,
        (:large, :heavy),
        (:application, :bootstrap, :fixed_trace_q3, :native_soc,
         :parameter_ladder, :certified_anchor, :cluster_only),
        :full_unitarity_fixed_trace_application,
        nothing,
        :csdr_fixed_trace_reduced_v1,
        (
            benchmark_scale=1,
            source_parameters=FULL_UNITARITY_EFT_BASE_PARAMETERS,
            input_generation_precision_bits=256,
            source_model_sha256=
                "823005149e46760d45d4488e75261e2b2956993e4056a792feb65ec013ac5475",
            fixed_physics=(
                dimension=4,
                subtraction=:zero_subtraction,
                alpha_set=("0", "-1/2"),
            ),
            original_equalities=90,
            reduced_equalities=84,
            solve_settings=(
                tolerance="1e-12",
                maximum_iterations=200,
                max_time=900.0,
                specialization=:fixed_trace,
            ),
            objective_interval=FULL_UNITARITY_EFT_REFERENCE,
        ),
        BenchmarkReference(
            :optimal,
            nothing,
            1.0e-12,
            1.0e-12,
            "Archived BigFloat-256 physical objective interval; require the " *
            "original-coordinate Lorentz certificate.",
        ),
        (
            variables=8400,
            soc_blocks=4200,
            cone_dimension=3,
            equalities=84,
            source_psd2_blocks=4200,
        ),
        ExternalSource(
            "Full-unitarity-EFT",
            "",
            "full-unitarity-eft-j40-na15-nmu200-nx2-nalpha2-v1.bin",
            :csdr_fixed_trace_reduced_v1,
            "ae66d61cdf2b00d46fd6ab83438c4e07bce3134a0fcd54519b7f7d5fce2533e8",
            "Private scientific neutral payload. Copy it explicitly into the " *
            "cache; the runner never downloads it.",
        ),
    ),
    _full_unitarity_pending_rung(2),
    _full_unitarity_pending_rung(4),
]
