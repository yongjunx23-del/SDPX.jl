#!/usr/bin/env julia
# Evidence-only diagnostic. This is intentionally not a benchmark catalog and
# is never loaded by PhysicsBenchmarkHarness. It records finite-ansatz witness
# behavior; failure is not an exact infeasibility theorem.
include(joinpath(@__DIR__, "diagnostic_impl.jl"))
using LinearAlgebra

for scale in (:tiny, :small, :medium)
    artifact = SMatrix4DSpecDiagnostic.build_smatrix_4d(scale)
    A = artifact.projection_imag
    candidate = A \ ones(eltype(A), size(A, 1))
    values = A * candidate
    println("scale=", scale, " matrix=", size(A),
            " least_squares_min_imag=", minimum(values),
            " least_squares_max_imag=", maximum(values),
            " strict_margin_recorded=", minimum(artifact.strict_margins))
end
stress = getproperty(SMatrix4DSpecDiagnostic.smatrix_4d_specs(Float64), :stress)
stress_rows = stress.energy_samples * (stress.spin_max ÷ 2 + 1)
stress_width = length(SMatrix4DSpecDiagnostic._basis_indices(stress.ansatz_degree))
stress_ops = stress_rows * stress.quadrature_order * stress_width
println("stress_counts rows=", stress_rows, " quadrature=", stress.quadrature_order,
        " variables=", stress_width,
        " orbit_evaluations=", stress_ops,
        " mode=build/profile-only bounded; no stress build attempted")
println("environment=julia", VERSION, " threads=", Threads.nthreads(),
        " arithmetic=Float64; external LP/HiGHS is optional and not an exact proof")
println("interpretation=diagnostic evidence only; no exact infeasibility claim")
