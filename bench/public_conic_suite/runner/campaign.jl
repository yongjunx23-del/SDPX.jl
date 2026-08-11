module GeneratedPathologicalCampaign

"""
P0 generated pathological campaign matrix.

The scope is a bounded selection of `manifests/pathological_suites.toml`:
every generator family is represented at a severity that exercises the audit
gates without turning one P0 job into a multi-hour release run.  Cases are
built entirely from repository source, so the runner needs no downloaded data
and can compute offline.

`min_bits` marks cases whose analytic margin is below the precision of an
arithmetic type.  Float64 (53 bits) skips those cases so the suite remains
passable; Float64x4 (209 bits) and BigFloat>=256 run the full matrix.  The
column is advisory only: the runner filters with `campaign_rows_for(bits)`.

Cluster resource contract:
- regular PBS: `ppn=5`, Julia `-t 4`, solver threads 4, BLAS threads 1,
  runs Float64 and Float64x4;
- bigfloat PBS: `ppn=1`, Julia `-t 1`, solver threads 1, BLAS threads 1,
  runs BigFloat (default 256 bits).
"""

using SHA

export CAMPAIGN, CAMPAIGN_VERSION, RESOURCE_MATRIX
export campaign_input_hash, campaign_family_counts, campaign_rows_for

const CAMPAIGN_VERSION = "p0-v4"

const CAMPAIGN = (
    # LP
    (family=:lp, case=:lp_near_dependent, severity="n16_eps1e-8",
     kwargs=(n=16, epsilon="1e-8"), expected_status=:optimal, min_bits=53),
    (family=:lp, case=:lp_near_dependent, severity="n16_eps1e-16",
     kwargs=(n=16, epsilon="1e-16"), expected_status=:optimal, min_bits=209),
    (family=:lp, case=:lp_row_scaling, severity="n16_decades6",
     kwargs=(n=16, decades=6), expected_status=:optimal, min_bits=53),
    (family=:lp, case=:lp_row_scaling, severity="n16_decades16",
     kwargs=(n=16, decades=16), expected_status=:optimal, min_bits=209),
    (family=:lp, case=:lp_infeasible_margin, severity="eps1e-8",
     kwargs=(epsilon="1e-8",), expected_status=:infeasible, min_bits=53),
    (family=:lp, case=:lp_infeasible_margin, severity="eps1e-16",
     kwargs=(epsilon="1e-16",), expected_status=:infeasible, min_bits=209),
    (family=:lp, case=:lp_klee_minty, severity="n8_eps1e-2",
     kwargs=(n=8, epsilon="1e-2"), expected_status=:optimal, min_bits=53),
    # SOCP
    (family=:socp, case=:socp_near_tangent, severity="eps1e-8",
     kwargs=(epsilon="1e-8",), expected_status=:optimal, min_bits=53),
    (family=:socp, case=:socp_near_tangent, severity="eps1e-16",
     kwargs=(epsilon="1e-16",), expected_status=:optimal, min_bits=209),
    (family=:socp, case=:socp_near_infeasible, severity="eps1e-4",
     kwargs=(epsilon="1e-4",), expected_status=:infeasible, min_bits=53),
    (family=:socp, case=:socp_near_infeasible, severity="eps1e-12",
     kwargs=(epsilon="1e-12",), expected_status=:infeasible, min_bits=209),
    (family=:socp, case=:socp_many_tiny, severity="ncones1000_eps1e-4",
     kwargs=(ncones=1000, epsilon="1e-4"), expected_status=:optimal,
     min_bits=53),
    # SDP
    (family=:sdp, case=:sdp_weak_infeasible_2x2, severity="delta1e-8",
     kwargs=(delta="1e-8",), expected_status=:optimal, min_bits=53),
    (family=:sdp, case=:sdp_weak_infeasible_2x2, severity="delta0",
     kwargs=(delta="0",), expected_status=:weakly_infeasible, min_bits=53),
    (family=:sdp, case=:sdp_hilbert, severity="n8",
     kwargs=(n=8,), expected_status=:optimal, min_bits=53),
    (family=:sdp, case=:sdp_congruence_scaling, severity="decades8",
     kwargs=(decades=8,), expected_status=:optimal, min_bits=53),
    (family=:sdp, case=:sdp_congruence_scaling, severity="decades16",
     kwargs=(decades=16,), expected_status=:optimal, min_bits=209),
    (family=:sdp, case=:sdp_small_eigenvalue, severity="n8_eps1e-8",
     kwargs=(n=8, epsilon="1e-8"), expected_status=:optimal, min_bits=53),
    (family=:sdp, case=:sdp_small_eigenvalue, severity="n8_eps1e-16",
     kwargs=(n=8, epsilon="1e-16"), expected_status=:optimal, min_bits=209),
)

const RESOURCE_MATRIX = (
    regular=(
        ppn=5,
        julia_threads=4,
        solver_threads=4,
        blas_threads=1,
        default_arithmetic="float64,float64x4",
    ),
    bigfloat=(
        ppn=1,
        julia_threads=1,
        solver_threads=1,
        blas_threads=1,
        default_arithmetic="bigfloat256",
    ),
)

"""Rows executable at `bits` of arithmetic precision."""
campaign_rows_for(bits::Int) =
    filter(row -> get(row, :min_bits, 53) <= bits, CAMPAIGN)

"""
    campaign_input_hash(case, kwargs) -> String

Deterministic SHA-256 over the generator case name and its canonical keyword
arguments.  The hash deliberately excludes arithmetic and tolerances so the
same mathematical input can be compared across the Float64/Float64x4/BigFloat
precision ladder.
"""
function campaign_input_hash(case::Symbol, kwargs::NamedTuple)
    names = sort!(collect(keys(kwargs)))
    parts = String[]
    for name in names
        push!(parts, string(name), "=", repr(getproperty(kwargs, name)))
    end
    payload = join(
        (CAMPAIGN_VERSION, "SDPXPathologicalBenchmarks", string(case), join(parts, "|")),
        "|",
    )
    return bytes2hex(sha256(payload))
end

function campaign_family_counts()
    counts = Dict{Symbol,Int}()
    for row in CAMPAIGN
        counts[row.family] = get(counts, row.family, 0) + 1
    end
    return counts
end

end # module
