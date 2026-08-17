# Sparse LP policy and backend selection helpers.
#
# Numeric sparse factors live in `sparse_la.jl`; this file deliberately has no
# legacy sparse LDL/Cholesky backend.  Sparse execution is the Round-6/7
# frozen-CSC provider seam only.

"""The provider-neutral sparse layer supports Float64 through CHOLMOD."""
supports_sparse_backend(::Type{Float64}) = true
supports_sparse_backend(::Type) = false

const LP_SPARSE_NNZ_PER_ROW = 13.0
const LP_SPARSE_MINIMUM_DIMENSION = 200
function select_lp_formulation(
    ; dimension::Integer,
    nonzeros::Integer,
    equalities::Integer,
    arithmetic::Type=Float64,
    storage::Union{Bool,Symbol}=:auto,
)
    requested = storage isa Bool ? (storage ? :sparse : :dense) : storage
    requested in (:auto, :dense, :sparse) || throw(ArgumentError(
        "sparse storage must be :auto, :dense, or :sparse",
    ))
    requested === :dense && return :dense_lu
    # Generic sparse arithmetic remains explicit-first.  Its factorization is
    # correct, but Round 6 did not establish an automatic crossover for
    # BigFloat/MultiFloat, so package availability must not silently change an
    # `:auto` execution plan.
    supports_sparse_generic(arithmetic) && requested !== :sparse &&
        return :dense_lu
    supports_sparse_execution(arithmetic) || return :dense_lu
    if requested !== :sparse
        dimension >= LP_SPARSE_MINIMUM_DIMENSION || return :dense_lu
        dimension > 0 && nonzeros / dimension < LP_SPARSE_NNZ_PER_ROW ||
            return :dense_lu
    end
    equalities > 0 && requested === :sparse && throw(ArgumentError(
        "storage=:sparse supports only SPD normal equations (equalities=0); " *
        "the sparse augmented LDL route is retired and no sparse pivoted LDLT is implemented",
    ))
    equalities == 0 ? :sparse_normal : :dense_lu
end

function formulation_backend(formulation::Symbol)
    formulation === :sparse_normal && return CHOLMODSparseCholeskyBackend()
    formulation === :dense_lu && return LPLUBackend()
    throw(ArgumentError("unknown LP formulation $(formulation)"))
end

inertia_available(::KKTBackend) = false
