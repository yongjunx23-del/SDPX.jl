# Clarabel.jl cross-check on the identical canonical Power-cone problem.
#
# Problem (validation/scientific_core/fixtures/factor_affine_trial_17.toml):
#   min sum(t)   s.t.  t_i >= 0  (prefix orthant)
#                      (t_i, 1, a_i) in POW3^{0.5,0.5}     (i = 1..3)
# Exact optimum: sum a_i^2.  No SDPX state is used as problem data.
#
# Two exactly equivalent forms:
#   :power - native Clarabel.PowerConeT(0.5) with activity (t, 1, a)
#   :soc   - SecondOrderConeT(3) with the EXACT integer map
#            (s1+s2, s1-s2, 2*s3) for activity (t,1,a), i.e. (t+1, t-1, 2a).
#            SOC membership (t+1)^2 >= (t-1)^2 + (2a)^2 <=> t >= a^2, with no
#            sqrt(2) rounding.  Clarabel 0.11.1 has no rotated-SOC cone type.
#            Original power primal is recovered by the inverse map; original
#            power duals by the transpose map.
#
# Float64 and BigFloat (256/512) at default and tightened tolerances.
# The audit reconstructs every returned value exactly as Rational{BigInt}.

using Clarabel, LinearAlgebra, SparseArrays, Printf, TOML, SHA

const FIXTURE = realpath(ARGS[1])
const OUT = abspath(ARGS[2])
mkpath(OUT)

floatword(s) = reinterpret(Float64, parse(UInt64, s; base = 16))
row = TOML.parsefile(FIXTURE)
a64 = [floatword(row["b_bits"][k]) for k in (6, 9, 12)]  # 1-based: a components

exact(x::AbstractFloat) = Rational{BigInt}(x)
const A_EXACT = sum(exact(v)^2 for v in a64)

# mapped activity s = b - A x, per form
function build_data(::Type{T}, form::Symbol) where {T}
    # rows 1..3 prefix orthant; block i occupies rows 3i+1, 3i+2, 3i+3
    rows = Int[]; cols = Int[]; vals = T[]
    for i in 1:3
        push!(rows, i); push!(cols, i); push!(vals, -one(T))
        push!(rows, 3 * i + 1); push!(cols, i); push!(vals, -one(T))
        if form === :soc
            push!(rows, 3 * i + 2); push!(cols, i); push!(vals, -one(T))
        end
    end
    A = sparse(rows, cols, vals, 12, 3)
    b = zeros(T, 12)
    for (i, v) in enumerate(a64)
        if form === :power
            b[3 * i + 2] = one(T)          # s2 = 1
            b[3 * i + 3] = T(v)            # s3 = a_i
        else
            b[3 * i + 1] = one(T)          # s1 = t + 1
            b[3 * i + 2] = -one(T)         # s2 = t - 1
            b[3 * i + 3] = T(2) * T(v)     # s3 = 2 a_i
        end
    end
    P = spzeros(T, 3, 3)
    q = ones(T, 3)
    A, b, P, q
end

power_cones() = Clarabel.SupportedCone[
    Clarabel.NonnegativeConeT(3),
    Clarabel.PowerConeT(0.5), Clarabel.PowerConeT(0.5), Clarabel.PowerConeT(0.5),
]

soc_cones() = Clarabel.SupportedCone[
    Clarabel.NonnegativeConeT(3),
    Clarabel.SecondOrderConeT(3), Clarabel.SecondOrderConeT(3), Clarabel.SecondOrderConeT(3),
]

function settings_for(::Type{T}, tight::Bool) where {T}
    tight || return Clarabel.Settings{T}(verbose = false, direct_solve_method = :qdldl)
    hi = T <: BigFloat
    tol = hi ? BigFloat(10)^-30 : T(1e-12)
    Clarabel.Settings{T}(
        verbose = false,
        tol_gap_abs = tol, tol_gap_rel = tol, tol_feas = tol,
        tol_infeas_abs = tol, tol_infeas_rel = tol, tol_ktratio = tol,
        static_regularization_constant = hi ? BigFloat(10)^-30 : T(1e-12),
        static_regularization_proportional = eps(T)^2,
        dynamic_regularization_eps = hi ? BigFloat(10)^-30 : T(1e-13),
        dynamic_regularization_delta = hi ? BigFloat(10)^-24 : T(2e-7),
        iterative_refinement_reltol = hi ? BigFloat(10)^-28 : T(1e-13),
        iterative_refinement_abstol = hi ? BigFloat(10)^-28 : T(1e-12),
        min_terminate_step_length = hi ? BigFloat(10)^-12 : T(1e-4),
        max_iter = hi ? 1000 : 500,
        equilibrate_enable = false, presolve_enable = false,
        chordal_decomposition_enable = false,
        direct_solve_method = :qdldl,
    )
end

function run_case(::Type{T}, form::Symbol, tight::Bool) where {T}
    A, b, P, q = build_data(T, form)
    settings = settings_for(T, tight)
    solver = Clarabel.Solver(P, q, A, b, form === :power ? power_cones() : soc_cones(), settings)
    Clarabel.solve!(solver)
    sol = solver.solution
    info = solver.info
    (; x = copy(sol.x), z = copy(sol.z), s = copy(sol.s), status = string(sol.status),
        obj = sol.obj_val, obj_dual = sol.obj_val_dual,
        iters = info.iterations, r_prim = info.res_primal, r_dual = info.res_dual,
        gap_abs = info.gap_abs, gap_rel = info.gap_rel, ktratio = info.ktratio,
        precision = string(T), form = string(form), tight)
end

function audit(rec)
    m = 12
    (length(rec.x) == 3 && length(rec.s) == m && length(rec.z) == m) || return (; ok = false)
    x = [exact(v) for v in rec.x]
    s = [exact(v) for v in rec.s]
    z = [exact(v) for v in rec.z]
    A, b, _, _ = build_data(eltype(rec.x), Symbol(rec.form))
    Ax = zeros(Rational{BigInt}, m)
    for j in 1:3, i in 1:m
        A[i, j] == 0 && continue
        Ax[i] += exact(A[i, j]) * x[j]
    end
    prim_res = maximum(abs(Ax[i] + s[i] - exact(b[i])) for i in 1:m)
    dual_res = maximum(abs(sum(exact(A[i, j]) * z[i] for i in 1:m) + (j <= 3 ? big(1) : big(0))) for j in 1:3)
    obj = sum(x)
    obj_err = abs(obj - A_EXACT)
    primal_min = big(0)
    dual_min = big(0)
    # rows 1-3 are the prefix orthant; the three Power/SOC blocks are rows
    # 4-6, 7-9 and 10-12.  Audit the three cone blocks, not the orthant.
    for i in 0:2
        s1, s2, s3 = s[3i + 4], s[3i + 5], s[3i + 6]
        z1, z2, z3 = z[3i + 4], z[3i + 5], z[3i + 6]
        if rec.form == "power"
            primal_min = min(primal_min, s1 * s2 - s3^2)
            dual_min = min(dual_min, 4 * z1 * z2 - z3^2)
        else
            primal_min = min(primal_min, s1^2 - s2^2 - s3^2)
            dual_min = min(dual_min, z1^2 - z2^2 - z3^2)
        end
    end
    comp = sum(s[i] * z[i] for i in 1:m)
    (; ok = true, prim_res, dual_res, obj, obj_err, rel_obj_err = obj_err / A_EXACT,
        primal_min, dual_min, comp, x, s, z)
end

results = NamedTuple[]
for T in (Float64, BigFloat)
    for prec in (T <: BigFloat ? (256, 512) : (53,))
        setprecision(BigFloat, prec) do
            for form in (:power, :soc), tight in (false, true)
                rec = run_case(T, form, tight)
                aud = audit(rec)
                push!(results, (; rec..., audit = aud, precision_bits = prec))
                @printf("%-8s %-4d %-6s tight=%-5s %-14s obj=%.17g err=%s iters=%d\n",
                    string(T), prec, rec.form, string(rec.tight), rec.status,
                    Float64(aud.obj), aud.ok ? string(Float64(aud.obj_err)) : "audit_fail", rec.iters)
            end
        end
    end
end

open(joinpath(OUT, "clarabel.json"), "w") do io
    println(io, "{")
    println(io, "  \"solver\": \"clarabel.jl\", \"version\": \"", pkgversion(Clarabel), "\",")
    println(io, "  \"fixture\": \"", FIXTURE, "\", \"fixture_sha256\": \"", bytes2hex(sha256(read(FIXTURE))), "\",")
    println(io, "  \"exact_optimum\": \"", A_EXACT, "\",")
    println(io, "  \"results\": [")
    for (k, r) in enumerate(results)
        print(io, "    {\"precision\": \"", r.precision, "\", \"precision_bits\": ", r.precision_bits,
            ", \"form\": \"", r.form, "\", \"tight\": ", r.tight, ", \"status\": \"", r.status,
            "\", \"iters\": ", r.iters, ", \"res_primal\": \"", r.r_prim, "\", \"res_dual\": \"", r.r_dual,
            "\", \"gap_abs\": \"", r.gap_abs, "\", \"gap_rel\": \"", r.gap_rel, "\", \"ktratio\": \"", r.ktratio, "\",")
        print(io, " \"x\": [", join(("\"" * string(v) * "\"" for v in r.x), ","), "],")
        print(io, " \"s\": [", join(("\"" * string(v) * "\"" for v in r.s), ","), "],")
        print(io, " \"z\": [", join(("\"" * string(v) * "\"" for v in r.z), ","), "],")
        a = r.audit
        if a.ok
            print(io, " \"audit\": {\"prim_res\": \"", a.prim_res, "\", \"dual_res\": \"", a.dual_res,
                "\", \"obj_err\": \"", a.obj_err, "\", \"rel_obj_err\": \"", a.rel_obj_err,
                "\", \"primal_cone_min\": \"", a.primal_min, "\", \"dual_cone_min\": \"", a.dual_min,
                "\", \"complementarity\": \"", a.comp, "\"}")
        else
            print(io, " \"audit\": {\"ok\": false}")
        end
        print(io, "}")
        k < length(results) && print(io, ",")
        println(io)
    end
    println(io, "  ]")
    println(io, "}")
end
