# src/cones/symmetric/SymmetricCones.jl
#
# Mutating symmetric-cone algebra for the three canonical self-dual cones:
# Nonnegative, SOC (Lorentz) and PSDTriangle (packed-lower PSD).
#
# Zero-allocation hot path. The cone descriptors carry all scratch needed by
# the listed hot operations (`membership`, `jordan_product!`, `inverse!`,
# `sqrt!`, `nt_scaling!`, `scaling_apply!`), so once a call has been warmed
# (compiled) it allocates **zero** Julia heap bytes on a Float64 (or MultiFloat)
# buffer. See `docs/design/CANONICAL_FORM.md` §5 for the zero-gate rule.
#
# PSD eigendecomposition uses a cyclic-Jacobi iteration with a hard convergence
# check (throws `_SymmetricEigenFailed` if the budget is exhausted), never a
# fixed-iteration loop that silently accepts a non-converged result. Float64
# matrix products route through BLAS gemm. No Kronecker matrices are used.

module SymmetricCones

using LinearAlgebra
using LinearAlgebra: mul!, BLAS
import Base: eltype, length, sqrt

include("types.jl")
include("eigen.jl")
include("nonnegative.jl")
include("soc.jl")
include("psd.jl")

# ---------------------------------------------------------------------------
# Allocating convenience wrappers (reference/analysis; not the zero-alloc path)
# ---------------------------------------------------------------------------
jordan_product(cone, x, y) = jordan_product!(cone, similar(x), x, y)
inverse(cone, x) = inverse!(cone, similar(x), x)
sqrt(cone, x) = sqrt!(cone, similar(x), x)
nt_scaling(cone, x) = nt_scaling!(cone, similar(x), x)
scaling_apply(cone, W, x) = scaling_apply!(cone, similar(x), W, x)
scaling_inverse_apply(cone, W, x) = scaling_inverse_apply!(cone, similar(x), W, x)
barrier_gradient(cone, x) = barrier_gradient!(cone, similar(x), x)
barrier_hessian_product(cone, x, d) = barrier_hessian_product!(cone, similar(x), x, d)
third_order_correction(cone, d1, d2, d3) = third_order_correction!(cone, similar(d1), d1, d2, d3)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
export NonnegativeCone, SOCone, PSDTriangleCone
export membership, dual_membership
export identity!, identity_element
export jordan_product!, inverse!, sqrt!, nt_scaling!
export scaling_apply!, scaling_inverse_apply!
export boundary_step!
export barrier_gradient!, barrier_hessian_product!, third_order_correction!
export jordan_product, inverse, sqrt, nt_scaling
export scaling_apply, scaling_inverse_apply
export barrier_gradient, barrier_hessian_product, third_order_correction
export spectrum, primitive_idempotents
export dim, stored_length

end # module SymmetricCones
