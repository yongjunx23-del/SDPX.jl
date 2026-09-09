# Clarabel.jl cross-checks on the frozen R0-E exponential-cone cases
# (benchmark/general/exp.jl, :small tier).
#
#   exp_entropy_small   : min sum(r) s.t. p >= 0, sum(p)=1,
#                         (-r_i, p_i, 1) in K_exp (i=1..3)
#                         known optimum -log(3)
#   exp_logsumexp_small : min t s.t. (c_i - t, 1, z_i) in K_exp (sum z <= 1),
#                         c from Xoshiro(0x0e0002)/0.4*randn
#                         known optimum logsumexp(c)
#
# Native Clarabel 0.11.1, canonical form Ax + s = b (no SDPX state used),
# presolve/equilibrate disabled so no conic presolve can alter the problem.
# Results (Float64, direct_solve_method=:qdldl):
#   exp_entropy_small   : SOLVED obj = -1.0986122892423997  (known -1.0986122886681098)
#   exp_logsumexp_small : SOLVED obj =  1.4040073736800012  (known  1.4040073747450363)
# Both problems are well-posed, feasible and bounded; both are solved to
# ~1e-9 by Clarabel while SDPX dev reports numerical_breakdown (cert invalid).
using Clarabel, LinearAlgebra, SparseArrays, Random

function exp_entropy(n=3)
    nv = 2n; q = [zeros(n); ones(n)]; P = spzeros(nv, nv)
    nr = n + 1 + 3n
    rows = Int[]; cols = Int[]; vals = Float64[]; b = zeros(nr)
    for i in 1:n
        push!(rows, i); push!(cols, i); push!(vals, -1.0)   # s = p
    end
    for i in 1:n; push!(rows, n+1); push!(cols, i); push!(vals, 1.0); end
    b[n+1] = 1.0                                            # sum p = 1
    for i in 1:n
        base = n+1 + 3*(i-1)
        push!(rows, base+1); push!(cols, n+i); push!(vals, 1.0)   # s1 = -r_i
        push!(rows, base+2); push!(cols, i); push!(vals, -1.0)    # s2 = p_i
        b[base+3] = 1.0                                     # s3 = 1
    end
    A = sparse(rows, cols, vals, nr, nv)
    cones = vcat([Clarabel.NonnegativeConeT(n), Clarabel.ZeroConeT(1)],
        fill(Clarabel.ExponentialConeT(), n))
    (P, q, A, b, cones)
end

function exp_logsumexp(n=3, seed=0x0e0002)
    rng = Random.Xoshiro(seed); c = 0.4 .* randn(rng, n)
    nv = 1 + n; q = [1.0; zeros(n)]; P = spzeros(nv, nv)
    nr = 3n + 1
    rows = Int[]; cols = Int[]; vals = Float64[]; b = zeros(nr)
    for i in 1:n
        base = 3*(i-1)
        push!(rows, base+1); push!(cols, 1); push!(vals, 1.0); b[base+1] = c[i]  # s1 = c_i - t
        b[base+2] = 1.0                                                          # s2 = 1
        push!(rows, base+3); push!(cols, 1+i); push!(vals, -1.0)                 # s3 = z_i
    end
    b[3n+1] = 1.0
    for i in 1:n; push!(rows, 3n+1); push!(cols, 1+i); push!(vals, 1.0); end     # sum z <= 1
    A = sparse(rows, cols, vals, nr, nv)
    cones = vcat(fill(Clarabel.ExponentialConeT(), n), [Clarabel.NonnegativeConeT(1)])
    mv = maximum(c); known = mv + log(sum(exp.(c .- mv)))
    (P, q, A, b, cones, c, known)
end

function solve(P, q, A, b, cones)
    settings = Clarabel.Settings{Float64}(verbose=false, direct_solve_method=:qdldl,
        equilibrate_enable=false, presolve_enable=false)
    solver = Clarabel.Solver(P, q, A, b, cones, settings)
    Clarabel.solve!(solver)
    (solver.solution.status, solver.solution.obj_val)
end

P,q,A,b,cones = exp_entropy()
st, obj = solve(P,q,A,b,cones)
println("exp_entropy_small  : ", st, " obj=", obj, " (known ", -log(3.0), ")")
P,q,A,b,cones,c,known = exp_logsumexp()
st, obj = solve(P,q,A,b,cones)
println("exp_logsumexp_small: ", st, " obj=", obj, " (known ", known, ")")
