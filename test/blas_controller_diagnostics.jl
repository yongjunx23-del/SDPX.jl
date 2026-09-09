# BLAS-controller diagnostics regression (tests only).
#
# `_native_hsd_diagnostics` must report the registered read-only BLAS
# controller (`blas_threads()` / `blas_backend()`), not the raw
# `LinearAlgebra.BLAS` state.  A synthetic controller returning a
# distinguishable count (7) and label (`:test_controller`) must flow into
# `selected_algorithms.ambient_blas_threads/backend` on a tiny LP solve
# without mutating actual BLAS threading, and the previous controller
# must be restored even on error.
using Test, SDPX, LinearAlgebra

@testset "blas controller diagnostics" begin
    old_getter = SDPX._blas_thread_getter[]
    old_setter = SDPX._blas_thread_setter[]
    old_backend = SDPX._blas_backend[]
    try
        SDPX._register_blas_thread_controller!(
            () -> 7,
            _ -> error("BLAS setter must not be called during diagnostics"),
            :test_controller,
        )
        @test SDPX.blas_threads() == 7
        @test SDPX.blas_backend() == :test_controller
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
        SDPX.constraint!(model, :lo, x[1] - 1.0, SDPX.Nonnegative())
        SDPX.constraint!(model, :hi, 2.0 - x[1], SDPX.Nonnegative())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
        outputs = SDPX.Outputs(
            :all, :all, :all;
            objectives=true,
            certificate=:summary,
            diagnostics=:full,
            history=false,
            trace=false,
        )
        result = SDPX.optimize!(
            model;
            settings=SDPX.Settings(Float64; verbosity=0),
            outputs,
        )
        @test SDPX.status(result) === :optimal
        @test SDPX.certificate(result).valid
        selected = SDPX.diagnostics(result).selected_algorithms
        @test selected.ambient_blas_threads == 7
        @test selected.ambient_blas_backend == :test_controller
        @test LinearAlgebra.BLAS.get_num_threads() == 1
    finally
        SDPX._register_blas_thread_controller!(
            old_getter, old_setter, old_backend)
    end
    @test SDPX._blas_backend[] === old_backend
    @test SDPX.blas_threads() == Int(LinearAlgebra.BLAS.get_num_threads())
end
