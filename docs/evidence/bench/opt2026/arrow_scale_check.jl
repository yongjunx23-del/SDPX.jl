# The arrow KKT path is exercised almost only by the CSDR model (n=0 + arrow
# structure); Task_Low08 uses the standard path. The existing equivalence test
# covers 3 blocks / 1 shared variable, which is too small for the threaded
# reduction or the per-block local pivots to engage. Re-check arrow == dense at
# a size where they do.
using LinearAlgebra, MultiFloats, Printf, Random, SparseArrays, SDPX
using MultiFloats: Float64x4
const T = Float64x4

function arrow_problem(blocks, shared; seed=11)
    rng = MersenneTwister(seed)
    m = shared + blocks
    coeff = [Vector{SparseMatrixCSC{T,Int}}(undef, m) for _ in 1:blocks]
    for l in 1:blocks, i in 1:m
        if i <= shared || i == shared + l
            v = T.(randn(rng, 3))
            coeff[l][i] = sparse([1, 2, 2], [1, 1, 2], [v[1], v[2], v[3]], 2, 2)
            coeff[l][i] = sparse(Symmetric(Matrix(coeff[l][i]), :L))
        else
            coeff[l][i] = spzeros(T, 2, 2)
        end
    end
    C = [Matrix{T}(Symmetric(T.(randn(rng, 2, 2)))) for _ in 1:blocks]
    c = T.(randn(rng, m))
    return (c=c, A=coeff, C=C, B=zeros(T, m, 0), b=T[])
end

function check(blocks, shared)
    d = arrow_problem(blocks, shared)
    dense = SDPX.ingest(d.c, [Array{T}(undef, 0, 0, 0) for _ in 1:0]; ) # placeholder
end

function run(blocks, shared)
    d = arrow_problem(blocks, shared)
    sp = SDPX.ingest(d.c, d.A, d.C, d.B, d.b; sparse=true, verbosity=0)
    dn = SDPX.ingest(d.c, d.A, d.C, d.B, d.b; sparse=false, verbosity=0)
    rng = MersenneTwister(7)
    X = [Matrix{T}(Symmetric(T.([2.0 0.2; 0.2 1.8]) .+ T(0.1) .* T.(randn(rng, 2, 2)))) for _ in 1:blocks]
    Y = [Matrix{T}(Symmetric(T.([1.7 0.1; 0.1 2.1]) .+ T(0.1) .* T.(randn(rng, 2, 2)))) for _ in 1:blocks]
    ws, wd = SDPX.Workspace(sp), SDPX.Workspace(dn)
    @assert ws.arrow !== nothing "expected arrow structure"
    SDPX.factor_blocks!(ws, X, Y) || error("sparse factor_blocks failed")
    SDPX.factor_blocks!(wd, X, Y) || error("dense factor_blocks failed")
    Sd = copy(SDPX.schur_build!(wd, dn, dn.cons, X, Y))
    SDPX.schur_build!(ws, sp, sp.cons, X, Y)
    Ss = zeros(T, size(Sd))
    SDPX.materialize_schur!(Ss, ws)
    schur_err = Float64(maximum(abs, Ss .- Sd) / max(maximum(abs, Sd), one(T)))

    o = SDPX.SolverOptions{T}(verbosity=0)
    ok_s = SDPX.factor_kkt!(ws, sp, o).ok
    ok_d = SDPX.factor_kkt!(wd, dn, o).ok
    m = sp.dims.m
    rhs = T.(collect(range(0.2, 1.6; length=m)))
    dxs, dxd = zeros(T, m), zeros(T, m)
    SDPX.solve_kkt!(ws, 0, copy(rhs), T[], dxs, T[])
    SDPX.solve_kkt!(wd, 0, copy(rhs), T[], dxd, T[])
    solve_err = Float64(maximum(abs, dxs .- dxd) / max(maximum(abs, dxd), one(T)))
    # Independent check: does the arrow solution actually satisfy S*dx = rhs?
    resid = Float64(maximum(abs, Sd * dxs .- rhs) / max(maximum(abs, rhs), one(T)))
    @printf("blocks=%-5d shared=%-4d m=%-5d fused=%-5s ok=%s/%s  schur_err=%.3e  dx_err=%.3e  S*dx-rhs=%.3e\n",
        blocks, shared, m, ws.fused_arrow, ok_s, ok_d, schur_err, solve_err, resid)
    flush(stdout)
end

println("threads=", Threads.nthreads(), "  eps(Float64x4)=", Float64(eps(T)))
for (b, s) in ((3, 1), (20, 4), (100, 12), (400, 48), (1000, 96))
    run(b, s)
end
