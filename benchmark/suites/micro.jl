const MICRO_SUITE = SuiteEntry[
    SuiteEntry("synthetic/lp_box", :float64, :auto),
    SuiteEntry("synthetic/lp_eq_full", :float64, :auto),
    SuiteEntry("synthetic/lp_eq_exact_deficient", :float64, :auto),
    SuiteEntry("synthetic/soc_q3", :float64, :auto),
    SuiteEntry("synthetic/soc_mixed", :float64, :auto),
    SuiteEntry("synthetic/sdp_dense", :float64, :auto),
    SuiteEntry("synthetic/sdp_equality_heavy", :float64, :auto),
    SuiteEntry("synthetic/sdp_small_eig_1e8", :float64, :auto),
    SuiteEntry("pathological/lp_degenerate_scaled", :float64, :auto),
    SuiteEntry("pathological/socp_near_tangent", :float64, :auto),
    SuiteEntry("pathological/sdp_small_eigenvalue", :float64, :auto),
]
