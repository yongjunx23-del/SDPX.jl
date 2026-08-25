using SDPX
using LinearAlgebra
using .PhysicsBenchmarkHarness

const _SMOKE_SPEC = PhysicsBenchmarkSpec(
    id="smoke/lp_box",
    name="deterministic LP smoke",
    family=:lp,
    problem_type=:linear_program,
    source=:fixture,
    purpose=:harness_smoke,
    parameters=(variables=2,),
    reference=PhysicsBenchmarkReference(
        status=:optimal,
        objective=3.0,
        absolute_tolerance=1.0e-7,
        relative_tolerance=1.0e-7,
        note="harness execution and serialization smoke",
    ),
    fingerprint="smoke-lp-box-v1:x-in-[1,3]:c=[1,2]",
)

function _build_smoke_problem(spec, ::Type{T}) where {T}
    spec.id == _SMOKE_SPEC.id || throw(KeyError(spec.id))
    identity = Matrix{T}(I, 2, 2)
    problem = SDPX.linear_program(
        T[1, 2],
        vcat(identity, -identity),
        T[1, 1, -3, -3];
        T,
        sparse=false,
        verbosity=0,
    )
    return (problem=problem, expected=T(3), kind=:lp)
end

function physics_benchmark_catalog()
    return PhysicsBenchmarkCatalog(
        :smoke,
        "1",
        [_SMOKE_SPEC],
        Dict(:smoke => [
            PhysicsBenchmarkEntry(_SMOKE_SPEC.id, :float64, :auto),
        ]),
        _build_smoke_problem,
    )
end
