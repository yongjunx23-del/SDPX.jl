const _LOCAL_FULL_PUBLIC_IDS = String[
    spec.id for spec in PUBLIC_SPECS if :local_full in spec.tiers
]

const _LOCAL_FULL_SYNTHETIC_IDS = String[
    spec.id for spec in SYNTHETIC_SPECS if :local_full in spec.tiers
]

const LOCAL_FULL_SUITE = let entries = SuiteEntry[]
    for id in sort!(vcat(_LOCAL_FULL_PUBLIC_IDS, _LOCAL_FULL_SYNTHETIC_IDS))
        push!(entries, SuiteEntry(id, :float64, :auto))
    end
    # Deliberate sampling, never a problem × precision × provider product.
    append!(entries, SuiteEntry[
        SuiteEntry("synthetic/lp_box", :float64x2, :auto),
        SuiteEntry("synthetic/lp_eq_near_1e8", :float64x2, :auto),
        SuiteEntry("synthetic/soc_q3", :float64x2, :auto),
        SuiteEntry("synthetic/sdp_dense", :float64x2, :auto),
        SuiteEntry("synthetic/sdp_equality_heavy", :float64x2, :auto),
        SuiteEntry("synthetic/sdp_small_eig_1e8", :float64x2, :auto),
        SuiteEntry("synthetic/lp_eq_near_1e8", :float64x3, :auto),
        SuiteEntry("synthetic/sdp_dense", :float64x3, :auto),
        SuiteEntry("synthetic/sdp_small_eig_1e12", :float64x3, :auto),
        SuiteEntry("synthetic/lp_eq_near_1e12", :float64x4, :auto),
        SuiteEntry("synthetic/sdp_equality_heavy", :float64x4, :auto),
        SuiteEntry("synthetic/sdp_small_eig_1e12", :float64x4, :auto),
        SuiteEntry("synthetic/lp_box", :bigfloat256, :auto),
        SuiteEntry("synthetic/lp_eq_near_1e12", :bigfloat256, :auto),
        SuiteEntry("synthetic/sdp_dense", :bigfloat256, :auto),
        SuiteEntry("synthetic/sdp_small_eig_1e12", :bigfloat256, :auto),
        SuiteEntry("synthetic/sdp_dense", :float64x4, :multifloat),
        SuiteEntry("synthetic/sdp_dense", :bigfloat256, :bfla),
    ])
    entries
end
