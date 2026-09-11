#=====================================================================
    Mixed-precision KKT planning shim.

    Only the planning surface remains: `_mixed_precision_storage_bytes`
    (Float64 workspace formula) and `_mixed_precision_workspace_decision`
    (refusal-first admission: unsupported arithmetic, :disabled,
    below-auto-dimension, memory_unknown, memory_budget). The guarded
    factorization machinery was dead and uncallable (it referenced
    never-defined names) and has been removed.
=====================================================================#

const MIXED_KKT_MINIMUM_AUTO_DIMENSION = 256


function _mixed_precision_storage_bytes(m::Int, n::Int)
    elements =
        m * m + m * n + n * n +
        3m + 3n
    return Base.checked_mul(elements, sizeof(Float64))
end

function _mixed_precision_workspace_decision(
    prob::SDPProblem{T},
    mode::Symbol,
    memory_fraction::Float64;
    available_memory_bytes::Integer=_available_memory_bytes(),
) where {T}
    arithmetic = _arithmetic_class(T)
    arithmetic in (:bigfloat, :fixed_extended) || return (
        enabled=false,
        reason=:unsupported_arithmetic,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    mode === :off && return (
        enabled=false,
        reason=:disabled,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    prob.structure.schur_backend === :dense_cholesky || return (
        enabled=false,
        reason=:unsupported_schur_backend,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    _, m, n, _ = prob.dims
    m > 0 || return (
        enabled=false,
        reason=:empty_system,
        required_bytes=0,
        memory_limit_bytes=0,
    )
    required = try
        _mixed_precision_storage_bytes(m, n)
    catch error
        error isa OverflowError || rethrow()
        return (
            enabled=false,
            reason=:storage_overflow,
            required_bytes=typemax(Int),
            memory_limit_bytes=0,
        )
    end
    available = Int(available_memory_bytes)
    memory_limit = available > 0 ?
                   floor(Int, available * memory_fraction) : 0
    (memory_limit > 0 && required <= memory_limit) || return (
        enabled=false,
        reason=available > 0 ? :memory_budget : :memory_unknown,
        required_bytes=required,
        memory_limit_bytes=memory_limit,
    )
    mode === :auto && m < MIXED_KKT_MINIMUM_AUTO_DIMENSION && return (
        enabled=false,
        reason=:below_auto_dimension,
        required_bytes=required,
        memory_limit_bytes=memory_limit,
    )
    return (
        enabled=true,
        reason=:selected,
        required_bytes=required,
        memory_limit_bytes=memory_limit,
    )
end
