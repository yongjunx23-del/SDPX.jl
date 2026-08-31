#!/usr/bin/env julia
# Compare equality preprocessing strategies for CSDR alpha3 (fixed-trace route).
# Each variant: warmup + 3 timed runs, determinism + certificate checks.
using SDPX, MultiFloats, MultiFloatLinearAlgebra
using LinearAlgebra, SparseArrays, Serialization, SHA

const ST = Float64x4
const INPUT = "/tmp/csdr-alpha9-twice/solve-alpha3.bin"
const EXPECTED_OBJECTIVE = -31.672155970636578
const VARIANT = get(ENV, "PRECOND_VARIANT", "baseline")  # baseline|colnorm|elim|ruiz
const SCALING = Symbol(get(ENV, "PRECOND_SCALING", "auto"))  # auto|none|equilibrate
raw = deserialize(INPUT)
converted = (
    c=ST.(raw.c), B=sparse(ST.(raw.B)), b=ST.(raw.b),
    objective_constant=ST(raw.objective_constant),
)
c, B, b = converted.c, converted.B, converted.b
nv = length(c)
me = size(B, 2)

# ---------- preprocessing ----------
B_use, b_use, nv_use, elim = B, b, nv, nothing
if VARIANT == "colnorm"
    Bd = Matrix(B)
    coln = [norm(Bd[:, j]) for j in 1:me]
    B_use = sparse(Bd ./ coln')
    b_use = b ./ coln
elseif VARIANT == "elim"
    # RRQR pivots of B' -> 42 pivot rows; x_p = -P^{-1} R x_r
    Bd = Matrix(B)
    F = qr(Bd', ColumnNorm())
    piv = F.p[1:me]
    rest = setdiff(1:nv, piv)
    P = Bd[piv, :]                 # 42x42 pivot submatrix (rows=pivot vars)
    R = Bd[rest, :]                # 8358x42
    X = P' \ R'                    # 42x8358: x_p = -X * x_r  (b = 0)
    # reduced objective: c_r' - c_p' X
    c_red = c[rest] - X' * c[piv]
    # reduced cones: substitute pivot vars into each cone
    # cone k uses vars (2k-1, 2k); build dense rows for rest-vars
    nv_use = length(rest)
    B_use = nothing; b_use = nothing
    elim = (piv=piv, rest=rest, X=X, c_red=c_red)
end

# ---------- model builders ----------
function build_model_full(B_mat, b_vec)
    model = SDPX.Model(ST; name="precond_$(VARIANT)")
    spectral = SDPX.variable!(model, :spectral, nv; domain=SDPX.Reals())
    for equality in axes(B_mat, 2)
        terms = Any[-b_vec[equality]]
        for pointer in nzrange(B_mat, equality)
            push!(terms, B_mat.nzval[pointer] * spectral[B_mat.rowval[pointer]])
        end
        SDPX.constraint!(model, Symbol(:sum_rule_, equality), sum(terms), SDPX.ZeroCone())
    end
    for cell in 1:(nv÷2)
        r = spectral[2cell-1]; q = spectral[2cell]
        SDPX.constraint!(model, Symbol(:unitarity_, cell),
            Any[one(ST), q - one(ST), r], SDPX.LorentzCone())
    end
    objective_terms = Any[converted.objective_constant]
    for index in eachindex(c)
        iszero(c[index]) || push!(objective_terms, c[index] * spectral[index])
    end
    SDPX.objective!(model, SDPX.Minimize(), sum(objective_terms))
    return model
end

function build_model_elim()
    X = elim.X; c_red = elim.c_red; rest = elim.rest; piv = elim.piv
    model = SDPX.Model(ST; name="precond_elim")
    xr = SDPX.variable!(model, :xr, nv_use; domain=SDPX.Reals())
    # cones: cone k uses (2k-1, 2k). If either is a pivot var, substitute.
    # x_p = -X * x_r  =>  x_i = -sum_j X[i_local, j] * xr[j]
    # Build per-cone dense row over rest vars.
    for cell in 1:(nv÷2)
        i1 = 2cell-1; i2 = 2cell
        # q - 1 = x_{i2} - 1 ; r = x_{i1}
        # cone: (1, q-1, r) in Lorentz cone
        # express q-1 and r as affine forms of xr
        qm1 = Any[-one(ST)]   # constant -1
        rv  = Any[zero(ST)]
        for (idx, coeff) in ((i1, one(ST)), (i2, one(ST)))
            if idx in piv
                local_j = findfirst(==(idx), piv)
                for (jj, rr) in enumerate(rest)
                    a = -X[local_j, jj]
                    iszero(a) || push!(qm1, a * xr[jj])
                    iszero(a) || push!(rv, a * xr[jj])
                end
            else
                jj = findfirst(==(idx), rest)
                push!(qm1, one(ST) * xr[jj])
                push!(rv, zero(ST))
            end
        end
        # careful: q-1 gets +1*x_{i2}, r gets +1*x_{i1}
        # rebuild properly below
    end
    return model
end

# Simpler correct elimination builder: precompute dense cone rows.
function build_model_elim2()
    X = elim.X; c_red = elim.c_red; rest = elim.rest; piv = elim.piv
    pivset = Set(piv)
    model = SDPX.Model(ST; name="precond_elim")
    xr = SDPX.variable!(model, :xr, nv_use; domain=SDPX.Reals())
    for cell in 1:(nv÷2)
        i1 = 2cell-1; i2 = 2cell
        # r = x_{i1}, q = x_{i2}; cone (1, q-1, r)
        # q-1 = -1 + sum_j a_j xr_j ; r = sum_j b_j xr_j
        a = zeros(ST, nv_use); b = zeros(ST, nv_use)
        for (idx, target) in ((i1, b), (i2, a))
            if idx in pivset
                local_j = findfirst(==(idx), piv)
                for jj in 1:nv_use
                    v = -X[local_j, jj]
                    iszero(v) || (target[jj] += v)
                end
            else
                jj = findfirst(==(idx), rest)
                target[jj] += one(ST)
            end
        end
        qm1 = Any[-one(ST)]
        rv = Any[zero(ST)]
        for jj in 1:nv_use
            iszero(a[jj]) || push!(qm1, a[jj] * xr[jj])
            iszero(b[jj]) || push!(rv, b[jj] * xr[jj])
        end
        SDPX.constraint!(model, Symbol(:cone_, cell), Any[one(ST), sum(qm1), sum(rv)], SDPX.LorentzCone())
    end
    objective_terms = Any[converted.objective_constant]
    for jj in 1:nv_use
        iszero(c_red[jj]) || push!(objective_terms, c_red[jj] * xr[jj])
    end
    SDPX.objective!(model, SDPX.Minimize(), sum(objective_terms))
    return model
end

settings = SDPX.Settings{ST}(
    tolerances=SDPX.Tolerances{ST}(primal=ST(1e-8), dual=ST(1e-8), gap=ST(1e-8)),
    limits=SDPX.Limits(iterations=500, time=600.0, threads=4),
    kkt_route=:bordered, verbosity=0,
    scaling=SCALING,
    equilibration=VARIANT == "ruiz" ? :ruiz : :off,
)
outputs = SDPX.Outputs(:all, :all, :all; objectives=true,
    certificate=:summary, diagnostics=:full, history=false, trace=false)

function solve_once(model)
    timed = @timed SDPX.optimize!(model; settings, outputs)
    result = timed.value
    certificate = SDPX.certificate(result)
    SDPX.status(result) === :optimal || error("status=$(SDPX.status(result))")
    certificate.valid || error("invalid certificate: $(certificate.reason)")
    isapprox(Float64(certificate.primal_objective), EXPECTED_OBJECTIVE;
        atol=1e-10, rtol=1e-12) || error("objective drift: $(certificate.primal_objective)")
    return (seconds=timed.time, bytes=timed.bytes, iterations=result.iterations,
        objective=certificate.primal_objective, valid=certificate.valid)
end

model = VARIANT == "elim" ? build_model_elim2() : build_model_full(B_use, b_use)
solve_once(model)  # warmup
rows = [solve_once(model) for _ in 1:3]
all(r -> r.iterations == rows[1].iterations && r.objective == rows[1].objective, rows) ||
    error("not deterministic")
median_seconds = sort([r.seconds for r in rows])[2]
median_bytes = sort([r.bytes for r in rows])[2]
println("VARIANT=$VARIANT status=optimal cert=true median_time_s=$median_seconds " *
    "median_bytes=$median_bytes iterations=$(rows[1].iterations) " *
    "objective=$(rows[1].objective)")
println("METRIC solver_seconds=$median_seconds")
println("METRIC allocation_bytes=$median_bytes")
println("METRIC iterations=$(rows[1].iterations)")
