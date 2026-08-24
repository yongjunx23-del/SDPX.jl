# Per-phase allocation breakdown of one full predictor-corrector Newton step
# (SDP route, Float64). Uses Profile.Allocs to attribute steady-state
# allocations to the producing phase. Phase-1b 'staged profile' deliverable.

using SDPX
using LinearAlgebra
import Profile

function _prob(T::Type)
    k = 3
    m = k * (k + 1) ÷ 2
    c = zeros(T, m); c[1] = -one(T)
    A = zeros(T, m, k, k)
    A[1,1,1]=one(T); A[2,2,2]=one(T); A[3,3,3]=one(T)
    A[4,1,2]=one(T); A[4,2,1]=one(T)
    A[5,1,3]=one(T); A[5,3,1]=one(T)
    A[6,2,3]=one(T); A[6,3,2]=one(T)
    B = zeros(T, m, 1); B[1,1]=one(T); B[2,1]=one(T); B[3,1]=one(T)
    return SDPX.ingest(c, [A], [zeros(T,k,k)], B, T[3]; T=T, sparse=false, verbosity=0)
end

function _classify(stacktrace)
    for frame in stacktrace
        func = frame.func === nothing ? "" : string(frame.func)
        occursin("schur", func) && return "KKT assembly"
        occursin("build", func) && return "KKT assembly"
        occursin("accumulate_v", func) && return "KKT assembly"
        occursin("compute_residual", func) && return "residuals"
        occursin("factorize", func) && return "factorize"
        occursin("_factor", func) && return "factorize"
        occursin("factor_blocks", func) && return "block factor"
        occursin("corrector_rhs", func) && return "RHS build"
        occursin("predictor_corrector_rhs", func) && return "RHS build"
        occursin("solve_direction", func) && return "KKT solve"
        occursin("_solve_dense", func) && return "KKT solve"
        occursin("refine", func) && return "refinement"
        occursin("line_search", func) && return "line search"
        occursin("direction_blocks", func) && return "block update"
        occursin("backtrack", func) && return "line search"
    end
    return "orchestration/diagnostics"
end

function _report()
    prob = _prob(Float64)
    opts = SDPX.SolverOptions{Float64}(algorithm=:sdp, presolve=false, scaling=:none, verbosity=0, iter_max=200)
    ws = SDPX.Workspace(prob; thread_count=1)
    init = SDPX._kkt_cold_start_initialization(ws, prob, opts)
    x,X,y,Y,mu = init.x, init.X, init.y, init.Y, init.μ
    SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=1)
    Profile.Allocs.@profile sample_rate=1.0 SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2)
    totals = Dict{String,Int}()
    for a in Profile.Allocs.fetch().allocs
        phase = _classify(a.stacktrace)
        totals[phase] = get(totals, phase, 0) + a.size
    end
    return totals
end

totals = _report()
total = sum(values(totals))
println("Per-phase allocation (Float64, one full Newton iteration)")
for (phase, bytes) in sort(collect(totals); by=x->-x[2])
    println(rpad(phase, 28), rpad(string(bytes), 9), "(", round(100*bytes/max(total,1); digits=1), "%)")
end
println("total ", total, " bytes")
exit()