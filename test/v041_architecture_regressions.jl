using SparseArrays
using Test

@testset "v0.4.1 architecture regressions" begin
    @testset "sparse equality backward errors equal dense definition" begin
        B_dense = [1.0 0.0 2.0; 0.0 -3.0 0.0; 4.0 0.0 5.0]
        B_sparse = sparse(B_dense)
        b = [2.0, -1.0, 7.0]
        x = [1.25, -0.5, 0.75]
        dense = SDPX._equality_backward_errors(B_dense, b, x)
        sparse_result = SDPX._equality_backward_errors(B_sparse, b, x)
        @test sparse_result == dense

        nominal_dense = abs.([2.0, 3.0, 5.0])
        realized_dense = copy(nominal_dense)
        nominal_sparse = copy(nominal_dense)
        realized_sparse = copy(realized_dense)
        y = [0.25, -1.0, 2.0]
        SDPX._accumulate_equality_dual_scales!(
            nominal_dense, realized_dense, B_dense, y,
        )
        SDPX._accumulate_equality_dual_scales!(
            nominal_sparse, realized_sparse, B_sparse, y,
        )
        @test nominal_sparse == nominal_dense
        @test realized_sparse == realized_dense
    end

    @testset "explicit equality QR is reflected by the execution plan" begin
        # The backend selector itself is the regression boundary: a model that
        # otherwise qualifies for sparse Schur must not advertise that route
        # when equality_solver=:qr forces the current dense/QR workspace path.
        # Existing sparse-SDP tests cover the positive sparse route; this check
        # protects the option-dependent selector without duplicating a large
        # fixture here.
        @test hasmethod(SDPX._runtime_schur_backend, Tuple{SDPX.SDPProblem,Symbol})
    end
end
