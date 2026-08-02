# Solver implementation manifest.
#
# SDPX 0.3 begins the staged decomposition of the former monolithic solve.jl.
# The numerical implementation is moved intact first, preserving definition
# order and keeping the release diff reviewable. Subsequent minor releases can
# extract initialization, iteration control, and pipeline orchestration behind
# this stable include boundary with focused equivalence tests.

include("solver/interior_point.jl")
