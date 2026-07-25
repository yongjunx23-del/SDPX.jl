#=====================================================================
    MultiFloats.jl as a first-class backend (§4.2). The IPM hot path
    needs only {+,−,×,/,sqrt,comparisons,abs}, all native and
    branch-free on MultiFloat — verified during development that the
    *generic* kernel path (kernels/generic.jl, built on Base
    LinearAlgebra) already works correctly for Float64x2/x4/etc. with
    zero extension-specific code: `cholesky!`, `ldiv!`, `mul!` all
    dispatch to Base's generic dense algorithms, and — being a
    bitstype — MultiFloat has none of BigFloat's mutable-reference
    aliasing hazards (copyto! is a true value copy). So this extension
    is deliberately small: it only supplies the one behavioral trait
    the core solver can't infer on its own (§4.2's dynamic-range
    guard), not new kernels.
=====================================================================#
module SDPXMultiFloatsExt

using SDPX
using MultiFloats: MultiFloat

# MultiFloat inherits Float64's ~10±308 exponent range and collapses ±Inf to
# NaN (no dedicated infinity bit pattern) — solve! runs the non-finite-iterate
# guard and caps restart escalation for these types (see solve.jl).
SDPX.dynamic_range_limited(::Type{<:MultiFloat}) = true

end
