# Wave A-4: per-site allocation audit of the v2 hot loop (new main).
using SDPX
using Profile

function _small_sdp_problem(::Type{T}) where {T}
    k = 3
    m = k * (k + 1) ÷ 2
    c = zeros(T, m)
    c[1] = -one(T)
    A = zeros(T, m, k, k)
    A[1, 1, 1] = one(T)
    A[2, 2, 2] = one(T)
    A[3, 3, 3] = one(T)
    A[4, 1, 2] = one(T); A[4, 2, 1] = one(T)
    A[5, 1, 3] = one(T); A[5, 3, 1] = one(T)
    A[6, 2, 3] = one(T); A[6, 3, 2] = one(T)
    B = zeros(T, m, 1)
    B[1, 1] = one(T); B[2, 1] = one(T); B[3, 1] = one(T)
    return SDPX.ingest(c, [A], [zeros(T, k, k)], B, T[3]; T=T, sparse=false, verbosity=0)
end

function _audit(::Type{T}) where {T}
    prob = _small_sdp_problem(T)
    opts = SDPX.SolverOptions{T}(algorithm=:sdp, presolve=false, scaling=:none, verbosity=0, iter_max=200)
    ws = SDPX.Workspace(prob; thread_count=1)
    init = SDPX._kkt_cold_start_initialization(ws, prob, opts)
    init.ok || return
    x, X, y, Y, mu = init.x, init.X, init.y, init.Y, init.μ
    SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=1)
    Profile.Allocs.clear()
    Profile.Allocs.start(sample_rate=1.0)
    SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2)
    Profile.Allocs.stop()
    results = Profile.Allocs.fetch()
    total = sum(a.size for a in results.allocs)
    println("=== $T per-site allocation (one newton_step) total=", total, " B ===")
    by_site = Dict{String,Int}()
    for a in results.allocs
        frame = nothing
        for f in a.stacktrace
            file = string(f.file)
            if occursin("SDPX", file) && !occursin("Allocs", file)
                frame = f
                break
            end
        end
        frame === nothing && (frame = a.stacktrace[1])
        key = string(frame.file, ":", frame.line)
        by_site[key] = get(by_site, key, 0) + a.size
    end
    for (site, bytes) in sort(collect(by_site); by=x->-x[2])[1:min(12, length(by_site))]
        println(rpad(site, 50), bytes, " B")
    end
end

_audit(Float64)