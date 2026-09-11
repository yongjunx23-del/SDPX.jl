# Observe the real MOI result boundary; no sign count is supplied by production.
module PublicSignBoundaryTests
using Test, SDPX
const MOI = SDPX.MOI

function audit_signs(raw, observed; atol=1e-8)
    length(raw) == length(observed) || throw(DimensionMismatch())
    changed = count(!isapprox(a, b; atol=atol, rtol=atol) for (a, b) in zip(raw, observed))
    flipped = count(abs(a) > atol && isapprox(-a, b; atol=atol, rtol=atol)
                    for (a, b) in zip(raw, observed))
    return (; checked=length(raw), changed, flipped)
end

@testset "MOI public boundary preserves owned dual signs" begin
    total_checked = 0
    total_flipped = 0
    for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE), coefficient in (1.0, -1.0),
        representation in (:bounds, :interval, :equality)
        source = MOI.Utilities.Model{Float64}()
        x = MOI.add_variable(source)
        f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
        constraints = if representation === :bounds
            [MOI.add_constraint(source, f, MOI.GreaterThan(1.0)),
             MOI.add_constraint(source, f, MOI.LessThan(2.0))]
        elseif representation === :interval
            [MOI.add_constraint(source, f, MOI.Interval(1.0, 2.0))]
        else
            [MOI.add_constraint(source, f, MOI.EqualTo(1.0)),
             MOI.add_constraint(source, f, MOI.GreaterThan(0.0))]
        end
        MOI.set(source, MOI.ObjectiveSense(), sense)
        objective = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(coefficient, x)], 0.0)
        MOI.set(source, MOI.ObjectiveFunction{typeof(objective)}(), objective)
        optimizer = SDPX.Optimizer()
        MOI.set(optimizer, MOI.Silent(), true)
        mapping = MOI.copy_to(optimizer, source)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        result = optimizer.public_result
        @test SDPX.certificate(result).valid
        normalized_c = sense == MOI.MIN_SENSE ? coefficient : -coefficient
        expected_x = representation === :equality || normalized_c > 0 ? 1.0 : 2.0
        @test MOI.get(optimizer, MOI.VariablePrimal(), mapping[x]) ≈ expected_x atol=1e-7
        @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ coefficient * expected_x atol=1e-7
        raw = Float64[]
        observed = Float64[]
        for constraint in constraints
            index = mapping[constraint]
            info = optimizer.model_constraint_records[SDPX._moi_constraint_key(index)]
            value = SDPX.dual(result, info.refs[1])
            # MOI interval dual is the sum of its two bound duals, not a sign patch.
            info.kind === :interval && (value += SDPX.dual(result, info.aux_refs[1]))
            push!(raw, value)
            push!(observed, MOI.get(optimizer, MOI.ConstraintDual(), index))
        end
        audit = audit_signs(raw, observed)
        @test audit.changed == 0
        @test audit.flipped == 0
        @test sum(observed) ≈ normalized_c atol=1e-7
        if representation === :bounds
            @test observed[1] >= -1e-8
            @test observed[2] <= 1e-8
        end
        # A deliberate boundary negation must fail the same observation.
        damaged = audit_signs(raw, -observed)
        @test damaged.changed > 0
        @test damaged.flipped > 0
        total_checked += audit.checked
        total_flipped += audit.flipped
    end
    println("MOI_SIGN_OBSERVATION solves=12 checked=", total_checked,
            " net_boundary_sign_flips=", total_flipped,
            " negative_control=detected")
end
end
