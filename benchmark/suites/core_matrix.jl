# Fixed LP/SOCP/SDP × Float64/Float64x4/BigFloat256 campaign.
#
# The extended-precision rows request their specialized providers explicitly.
# If MFLA/BFLA are not installed, the canonical runner records a structured
# `provider_unavailable` skip rather than silently comparing a legacy fallback.
const CORE_MATRIX_SUITE = SuiteEntry[
    SuiteEntry("synthetic/lp_box", :float64, :auto),
    SuiteEntry("synthetic/lp_box", :float64x4, :multifloat),
    SuiteEntry("synthetic/lp_box", :bigfloat256, :bfla),
    SuiteEntry("synthetic/soc_q3", :float64, :auto),
    SuiteEntry("synthetic/soc_q3", :float64x4, :multifloat),
    SuiteEntry("synthetic/soc_q3", :bigfloat256, :bfla),
    SuiteEntry("synthetic/sdp_dense", :float64, :auto),
    SuiteEntry("synthetic/sdp_dense", :float64x4, :multifloat),
    SuiteEntry("synthetic/sdp_dense", :bigfloat256, :bfla),
]
