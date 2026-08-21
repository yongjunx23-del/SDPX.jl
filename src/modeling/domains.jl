#=====================================================================#
#    Public mathematical domain types for the v0.5 Model frontend.
#
#    These types are pure mathematical values: they describe the set a
#    variable or product-cone block lives in, and the sense of an
#    objective. They carry no dimension, no data, and no state. A PSD
#    cone is deliberately dimension-less at this level: the matrix
#    dimension `n` belongs to the variable / block shape and is recorded
#    by `SDPX.NativeBlock` (see src/ir/types.jl), never by `PSDCone`
#    itself. No free/± split, no scalarization, no SOC→PSD lift is
#    performed anywhere in this file.
#
#    Every concrete type below is a singleton immutable value; instances
#    compare equal exactly when their types are identical.
#
#    Include order: this file must be included before refs.jl and
#    types.jl (which reference these types only via docstrings) and
#    before ir/types.jl, which stores them in block descriptors.
#=====================================================================#

"""
    SDPX.Reals

The full real line. A scalar variable (or vector block) constrained to
`Reals` is unconstrained. This type intentionally mirrors the
`MathOptInterface.Reals` mathematical set but is SDPX-owned so the
frontend never depends on MOI's type identity.
"""
struct Reals end

"""
    SDPX.Nonnegative

The closed nonnegative orthant `{x >= 0}`. As a product-cone block it
is one vector block; it is never split into scalar blocks.
"""
struct Nonnegative end

"""
    SDPX.Nonpositive

The closed nonpositive orthant `{x <= 0}`. As a product-cone block it
is one vector block.
"""
struct Nonpositive end

"""
    SDPX.ZeroCone

The zero / equality subspace `{x == 0}`. As a product-cone block it is
one vector block.
"""
struct ZeroCone end

"""
    SDPX.LorentzCone

The second-order cone of *vector dimension* `n`:
`{ (t, x) : ||x||_2 <= t }` with `t` first, `n >= 1`. The dimension `n`
belongs to the block shape (`SDPX.NativeBlock`), not to this type.
"""
struct LorentzCone end

"""
    SDPX.RotatedLorentzCone

The rotated second-order cone of *vector dimension* `n`:
`{ (u, v, x) : 2uv >= ||x||_2^2, u >= 0, v >= 0 }`, `n >= 3`. The
dimension `n` belongs to the block shape.
"""
struct RotatedLorentzCone end

"""
    SDPX.ExponentialCone

The 3-dimensional exponential cone
`K_exp = { (x, y, z) : y * exp(x / y) <= z, y > 0 }` together with its
limit `{ (0, y, z) : y >= 0, z >= 0 }`. The dimension is fixed at 3 and
is validated at block construction: an exponential block always has
vector shape `n == 3`, never more and never less. This type carries no
data and remains one block — it is never split or lifted.
"""
struct ExponentialCone end

"""
    SDPX.PSDCone

The positive semidefinite cone over real symmetric matrices. This type
carries **no dimension**: an `n × n` PSD block records its matrix
dimension `n` and lower-authoritative packed storage metadata in its
`SDPX.NativeBlock` descriptor and remains ONE block — never
`n(n+1)/2` unrelated scalar blocks.
"""
struct PSDCone end

# ---------------------------------------------------------------------------
# Objective sense
# ---------------------------------------------------------------------------

"""
    SDPX.Minimize

Objective sense marker: minimize `c'x + constant`.
"""
struct Minimize end

"""
    SDPX.Maximize

Objective sense marker: maximize `c'x + constant`.
"""
struct Maximize end

# Type unions used by the native IR for well-typed fields. Affine cone
# blocks use the same mathematical domains as product-variable blocks.
const ProductConeDomain = Union{Reals,Nonnegative,Nonpositive,ZeroCone,LorentzCone,RotatedLorentzCone,PSDCone,ExponentialCone}
const AffineConeDomain = ProductConeDomain

is_product_cone(domain) = domain isa ProductConeDomain

"""
    is_affine_cone(domain)

Whether `domain` is a valid affine-cone domain for the native IR.
PSD affine blocks retain their matrix dimension and packed-lower
storage metadata just like PSD product blocks.
"""
is_affine_cone(::AffineConeDomain) = true
is_affine_cone(domain) = false

"""
    affine_dimension(domain, n)

Intrinsic affine dimension of a block of shape `n` for `domain`. For
PSD blocks the contract stores matrix dimension `n` and the *packed*
variable count `n(n+1)/2`; for all other domains the block variable
count equals the affine dimension. This is a pure shape helper; it
never allocates or inspects coefficient data.
"""
affine_dimension(::PSDCone, n::Integer) = variable_length(PSDCone(), n)
affine_dimension(domain, n::Integer) = Int(n)

"""
    variable_length(domain, n)

Number of scalar variables stored by a block of shape `n` in `domain`.
A PSD `n × n` block stores `n(n+1)/2` entries in lower-triangle packed
form (column-major, i.e. entries `(i, j)` with `i >= j` ordered by `j`
then `i`), matching `PSDPackedStorage.packed_length`.
"""
variable_length(::PSDCone, n::Integer) = Int(n * (n + 1) ÷ 2)
variable_length(domain, n::Integer) = Int(n)
