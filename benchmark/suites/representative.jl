const REPRESENTATIVE_SUITE = SuiteEntry[
    MICRO_SUITE...,
    SuiteEntry("synthetic/lp_sparse", :float64, :auto),
    SuiteEntry("synthetic/lp_eq_near_1e2", :float64, :auto),
    SuiteEntry("synthetic/lp_eq_near_1e4", :float64, :auto),
    SuiteEntry("synthetic/lp_scale_1e4", :float64, :auto),
    SuiteEntry("synthetic/soc_q8", :float64, :auto),
    SuiteEntry("synthetic/soc_scale_1e4", :float64, :auto),
    SuiteEntry("synthetic/sdp_many_small", :float64, :auto),
    SuiteEntry("synthetic/sdp_sparse", :float64, :auto),
    SuiteEntry("synthetic/sdp_block_arrow", :float64, :auto),
    SuiteEntry("synthetic/sdp_small_eig_1e4", :float64, :auto),
    # Public instances are registered here but become structured skips until
    # their format loader and local cache are both available.
    SuiteEntry("netlib/afiro", :float64, :auto),
    SuiteEntry("netlib/degen2", :float64, :auto),
    SuiteEntry("sdplib/control1", :float64, :auto),
    SuiteEntry("sdplib/truss1", :float64, :auto),
    SuiteEntry("sdplib/theta1", :float64, :auto),
    SuiteEntry("dimacs/nql30", :float64, :auto),
    SuiteEntry("dimacs/qssp30", :float64, :auto),
    SuiteEntry("dimacs/filter48_socp", :float64, :auto),
    # Optional providers are loaded only for these explicit checks. Missing
    # local packages become structured `provider_unavailable` skips.
    SuiteEntry("synthetic/sdp_dense", :float64x4, :multifloat),
    SuiteEntry("synthetic/sdp_dense", :bigfloat256, :bfla),
]
