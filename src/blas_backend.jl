#=====================================================================
    BLAS/LAPACK backend control.

    libblastrampoline exposes a common thread API for most backends, but
    Apple's Accelerate threading mode is controlled by BLASSetThreading and
    is not changed by LinearAlgebra.BLAS.set_num_threads. Optional package
    extensions can register the correct controller here without making a
    platform-specific backend a hard dependency of SDPX.
=====================================================================#

const _blas_thread_getter =
    Ref{Function}(() -> Int(LinearAlgebra.BLAS.get_num_threads()))
const _blas_thread_setter = Ref{Function}(
    count -> begin
        LinearAlgebra.BLAS.set_num_threads(count)
        Int(LinearAlgebra.BLAS.get_num_threads())
    end,
)
const _blas_backend = Ref{Symbol}(:libblastrampoline)

"""
    blas_threads() -> Int

Return the effective thread count of the active dense BLAS/LAPACK backend.
Unlike `LinearAlgebra.BLAS.get_num_threads`, this also reports the actual
Accelerate mode when the optional AppleAccelerate extension is loaded.
"""
blas_threads() = Int(_blas_thread_getter[]())

"""
    set_blas_threads!(count) -> Int

Set the effective dense BLAS/LAPACK thread count and return the value reported
by the active backend. On Apple Accelerate, `1` selects strict single-threaded
mode and values greater than one select Accelerate-managed multithreading.
"""
function set_blas_threads!(count::Integer)
    count >= 1 ||
        throw(ArgumentError("BLAS thread count must be at least one"))
    return Int(_blas_thread_setter[](Int(count)))
end

"""Return the registered dense BLAS/LAPACK backend controller."""
blas_backend() = _blas_backend[]

function _register_blas_thread_controller!(
    getter::Function,
    setter::Function,
    backend::Symbol,
)
    _blas_thread_getter[] = getter
    _blas_thread_setter[] = setter
    _blas_backend[] = backend
    return nothing
end
