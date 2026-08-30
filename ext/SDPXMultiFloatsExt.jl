# MultiFloats arithmetic traits for the single product-HSD solver.
#
# The former reduced-arrow / legacy-Workspace specializations were retired
# with the standalone KKT engine.  Fixed-trace product-HSD x4 kernels now
# live in SDPXMultiFloatLinearAlgebraExt and dispatch only through the
# provider protocol.
module SDPXMultiFloatsExt

using SDPX
using MultiFloats: Float64x4, MultiFloat

SDPX.is_multifloat_arithmetic(::Type{<:MultiFloat}) = true
SDPX.is_supported_arithmetic(::Type{<:MultiFloat}) = true

# MultiFloat has Float64's finite exponent range and maps infinities to NaN;
# product-HSD therefore keeps its non-finite-iterate guard enabled.
SDPX.dynamic_range_limited(::Type{<:MultiFloat}) = true

# This is an arithmetic-kernel preference, not a solver or fallback choice.
SDPX.default_extended_precision_blas(::Type{Float64x4}) = :auto
SDPX.default_mixed_precision_condition_limit(::Type{Float64x4}) = 1.0e14

end
