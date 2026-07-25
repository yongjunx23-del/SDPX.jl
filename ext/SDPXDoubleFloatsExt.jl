#=====================================================================
    DoubleFloats.jl (`Double64`) as a first-class backend (§4.3) —
    comes essentially free from the same genericity as MultiFloats:
    verified during development that `cholesky!`/`ldiv!`/`mul!` all
    work generically for `Double64`, and (unlike MultiFloat) `Inf`
    stays `Inf` rather than collapsing to `NaN`. It still has a
    bounded exponent range matching `Float64` (`floatmax(Double64) ≈
    1.8e308`), so the same non-finite-iterate guard and restart-
    escalation cap apply for the same underlying reason (raw
    high-degree-polynomial bootstrap data can overflow).
=====================================================================#
module SDPXDoubleFloatsExt

using SDPX
using DoubleFloats: DoubleFloat

SDPX.dynamic_range_limited(::Type{<:DoubleFloat}) = true

end
