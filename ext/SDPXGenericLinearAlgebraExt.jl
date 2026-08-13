#=
    Optional GenericLinearAlgebra bridge.

GenericLinearAlgebra deliberately extends Julia's generic LinearAlgebra
implementation.  SDPX does not call package internals: loading this extension
only records that the standard high-precision backend is running with the GLA
method set available.  All numerical calls remain behind `la_*` and use the
stable LinearAlgebra interface.
=#
module SDPXGenericLinearAlgebraExt

using SDPX
using GenericLinearAlgebra

SDPX.generic_la_provider_implementation(::Val{:generic_linear_algebra}) =
    :julia_generic_with_gla_loaded

end
