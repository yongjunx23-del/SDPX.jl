# SDPX on the exact canonical small Power-cone problem used by MOSEK/Clarabel.
#
#   min sum(t)  s.t.  x_i = a_i (ZeroCone), t_i >= 0,
#                     (t_i, 1, x_i) in PowerCone(0.5)
#
# Exact optimum sum a_i^2 ~ 1.1242390986454483421.  This only reads the public
# SDPX API; no fixture state is injected.
using SDPX, Printf, SHA, TOML
import BigFloatLinearAlgebra

const FIXTURE = realpath(ARGS[1])
const OUT = abspath(ARGS[2])
floatword(s) = reinterpret(Float64, parse(UInt64, s; base = 16))
row = TOML.parsefile(FIXTURE)
a = [floatword(row["b_bits"][k]) for k in (6, 9, 12)]

function run_once(::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :fixed_signal, 3; domain = SDPX.Reals())
    t = SDPX.variable!(model, :power_epigraph, 3; domain = SDPX.Nonnegative())
    for i in 1:3
        SDPX.constraint!(model, Symbol(:fix_signal_, i), x[i] - T(a[i]), SDPX.ZeroCone())
        SDPX.constraint!(model, Symbol(:power_term_, i), (t[i], one(T), x[i]), SDPX.PowerCone(T(0.5)))
    end
    SDPX.objective!(model, SDPX.Minimize(), sum(t))
    settings = SDPX.Settings{T}(limits = SDPX.Limits(time = 120.0, threads = 1), verbosity = 0, certification = true)
    result = SDPX.optimize!(model; settings = settings)
    result
end

mkpath(OUT)
open(joinpath(OUT, "sdpx.txt"), "w") do io
    for prec in (53, 256, 512)
        if prec == 53
            r = run_once(Float64)
            cert = SDPX.certificate(r)
            @printf("Float64      status=%s primal_obj=%.17g primal_res=%.3e dual_res=%.3e\n",
                SDPX.status(r), cert.primal_objective, cert.primal_residual, cert.dual_residual)
            println(io, "Float64 status=", SDPX.status(r), " primal_obj=", cert.primal_objective,
                " primal_res=", cert.primal_residual, " dual_res=", cert.dual_residual)
        else
            setprecision(BigFloat, prec) do
                r = run_once(BigFloat)
                cert = SDPX.certificate(r)
                @printf("BigFloat%-4d status=%s primal_obj=%s primal_res=%s dual_res=%s\n",
                    prec, SDPX.status(r), string(cert.primal_objective), string(cert.primal_residual), string(cert.dual_residual))
                println(io, "BigFloat", prec, " status=", SDPX.status(r), " primal_obj=", cert.primal_objective,
                    " primal_res=", cert.primal_residual, " dual_res=", cert.dual_residual)
            end
        end
    end
end
