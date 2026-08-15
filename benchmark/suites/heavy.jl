const HEAVY_SUITE = SuiteEntry[
    SuiteEntry(spec.id, :registered_only, :none) for spec in
        vcat(HEAVY_SPECS, FULL_UNITARITY_EFT_SPECS)
]
