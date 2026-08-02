# JuMP and MathOptInterface

SDPX provides a non-incremental `MathOptInterface` optimizer. JuMP builds the
model and SDPX converts it to the same typed block-SDP representation used by
the native interface.

```julia
using JuMP, LinearAlgebra, SDPX

model = Model(() -> SDPX.Optimizer(
    sparse=:auto,
    tolerance=1e-8,
    threads=4,
    verbosity=0,
))

@variable(model, x[1:2])
@constraint(
    model,
    Symmetric([x[1] -1.0; -1.0 x[2]]) in PSDCone(),
)
@objective(model, Min, 2x[1] + 3x[2])
optimize!(model)
```

Use `GenericModel{T}` and `SDPX.Optimizer{T}` for extended coefficient types
such as `MultiFloats.Float64x4`.

The wrapper supports affine equality and bound constraints, second-order
cones through a PSD lift, and positive-semidefinite triangle cones. Consult
the
[complete JuMP/MOI guide](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/julia-interface.md)
for supported attributes, typed examples, and option names.
