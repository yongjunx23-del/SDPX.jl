# Pure correctness gates shared by the Round 4 scoreboard and its lightweight
# contract tests.  Keep this file free of solver/provider imports.

function _round4_objective_valid(actual, expected, absolute_tolerance, relative_tolerance)
    error = abs(actual - expected)
    allowed = absolute_tolerance +
              relative_tolerance * max(one(error), abs(expected))
    return isfinite(error) && error <= allowed
end

_round4_severe_false_negative(auto_valid, normal_valid, augmented_valid) =
    !auto_valid && (normal_valid || augmented_valid)
