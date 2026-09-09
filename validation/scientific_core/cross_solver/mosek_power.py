"""MOSEK cross-check on the exact canonical Power-cone problem.

Problem (identical to validation/scientific_core/fixtures/factor_affine_trial_17.toml):
    min sum(t_i)
    s.t. t_i >= 0                                  (prefix orthant)
         (t_i, 1, a_i) in POW3^{0.5,0.5}           i = 1..3
    equivalently t_i >= a_i^2, exact optimum sum a_i^2.

Both the native power-cone form and the exactly equivalent rotated-SOC form are
solved.  No SDPX state is used; only A/b/c of the fixture define the problem.
"""
from __future__ import annotations

import json
import struct
import sys
import tomllib
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import mosek

FIXTURE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FIXTURE = FIXTURE.resolve()


def bits_to_float(word: str) -> float:
    return struct.unpack(">d", bytes.fromhex(word))[0]


def load_fixture(path: Path):
    data = tomllib.loads(path.read_text())
    a = [bits_to_float(data["b_bits"][k]) for k in (5, 8, 11)]
    return data, a


def exact_optimum(a):
    q = [Fraction.from_float(v) for v in a]
    return sum(v * v for v in q)


def solve_power(a, tight: bool):
    """Native power-cone form.  Returns a receipt dict."""
    with mosek.Env() as env, env.Task(0, 0) as task:
        task.putintparam(mosek.iparam.log, 0)
        task.putintparam(mosek.iparam.num_threads, 1)
        if tight:
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, 1e-12)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, 1e-12)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1e-12)
        task.appendvars(3)
        task.putvarboundslice(0, 3, [mosek.boundkey.lo] * 3, [0.0] * 3, [0.0] * 3)
        task.putclist([0, 1, 2], [1.0, 1.0, 1.0])
        task.putobjsense(mosek.objsense.minimize)
        # 9 affine expressions: (t_i, 1, a_i)
        task.appendafes(9)
        rows, cols, vals = [], [], []
        for i in range(3):
            rows.append(3 * i)
            cols.append(i)
            vals.append(1.0)
        task.putafefentrylist(rows, cols, vals)
        g = [0.0] * 9
        for i in range(3):
            g[3 * i + 1] = 1.0
            g[3 * i + 2] = a[i]
        task.putafegslice(0, 9, g)
        doms = [task.appendprimalpowerconedomain(3, [0.5, 0.5]) for _ in range(3)]
        for i, d in enumerate(doms):
            task.appendacc(d, [3 * i, 3 * i + 1, 3 * i + 2], None)
        task.optimize()
        sol = mosek.soltype.itr
        status = task.getsolsta(sol)
        x = [0.0] * 3
        task.getxx(sol, x)
        s = [0.0] * 12
        for i, d in enumerate(doms):
            act = task.evaluateacc(sol, i)
            s[3 * i : 3 * i + 3] = list(act)
        z = []
        for i in range(3):
            z.extend(list(task.getaccdoty(sol, i)))
        return {
            "form": "power",
            "tight": tight,
            "solsta": str(status),
            "prosta": str(task.getprosta(sol)),
            "x": x,
            "s": s,
            "z": z,
            "objective": task.getprimalobj(sol),
            "dual_objective": task.getdualobj(sol),
        }


def solve_rotated_soc(a, tight: bool):
    """Exactly equivalent rotated-SOC form: (t_i, 0.5, a_i) in Qr^3."""
    with mosek.Env() as env, env.Task(0, 0) as task:
        task.putintparam(mosek.iparam.log, 0)
        task.putintparam(mosek.iparam.num_threads, 1)
        if tight:
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, 1e-12)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, 1e-12)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1e-12)
        task.appendvars(3)
        task.putvarboundslice(0, 3, [mosek.boundkey.lo] * 3, [0.0] * 3, [0.0] * 3)
        task.putclist([0, 1, 2], [1.0, 1.0, 1.0])
        task.putobjsense(mosek.objsense.minimize)
        task.appendafes(9)
        rows, cols, vals = [], [], []
        for i in range(3):
            rows.append(3 * i)
            cols.append(i)
            vals.append(1.0)
        task.putafefentrylist(rows, cols, vals)
        g = [0.0] * 9
        for i in range(3):
            g[3 * i + 1] = 0.5
            g[3 * i + 2] = a[i]
        task.putafegslice(0, 9, g)
        doms = [task.appendrquadraticconedomain(3) for _ in range(3)]
        for i, d in enumerate(doms):
            task.appendacc(d, [3 * i, 3 * i + 1, 3 * i + 2], None)
        task.optimize()
        sol = mosek.soltype.itr
        status = task.getsolsta(sol)
        x = [0.0] * 3
        task.getxx(sol, x)
        s = [0.0] * 12
        for i, d in enumerate(doms):
            act = task.evaluateacc(sol, i)
            s[3 * i : 3 * i + 3] = list(act)
        z = []
        for i in range(3):
            z.extend(list(task.getaccdoty(sol, i)))
        return {
            "form": "rotated_soc",
            "tight": tight,
            "solsta": str(status),
            "prosta": str(task.getprosta(sol)),
            "x": x,
            "s": s,
            "z": z,
            "objective": task.getprimalobj(sol),
            "dual_objective": task.getdualobj(sol),
        }


def audit(rec, a, exact_opt):
    getcontext().prec = 60
    x = [Fraction(v) for v in rec["x"]]
    s = [Fraction(v) for v in rec["s"]]
    obj = sum(x)
    dets = []
    for i in range(3):
        s1, s2, s3 = s[3 * i], s[3 * i + 1], s[3 * i + 2]
        if rec["form"] == "power":
            dets.append({"primal_det": str(float(s1 * s2 - s3 * s3))})
        else:
            # MOSEK rotated quadratic cone: 2 x0 x1 >= sum_{i>=2} x_i^2
            dets.append({"primal_det": str(float(2 * s1 * s2 - s3 * s3))})
    # objective error against exact optimum
    err = abs(obj - exact_opt)
    rec["audit"] = {
        "objective": float(obj),
        "exact_optimum": str(exact_opt),
        "exact_optimum_float": float(exact_opt),
        "abs_objective_error": str(err),
        "abs_objective_error_float": float(err),
        "rel_objective_error": str(err / exact_opt),
        "cone_dets": dets,
    }
    if "z" in rec and rec["z"]:
        z = [Fraction(v) for v in rec["z"]]
        dual_dets = []
        for i in range(3):
            z1, z2, z3 = z[3 * i], z[3 * i + 1], z[3 * i + 2]
            if rec["form"] == "power":
                dual_dets.append({"dual_det": str(float(4 * z1 * z2 - z3 * z3))})
            else:
                dual_dets.append({"dual_det": str(float(2 * z1 * z2 - z3 * z3))})
        comp = sum(s[k] * z[k] for k in range(9))
        rec["audit"]["dual_dets"] = dual_dets
        rec["audit"]["complementarity"] = str(float(comp))
    return rec


def main():
    data, a = load_fixture(FIXTURE)
    exact_opt = exact_optimum(a)
    results = []
    for tight in (False, True):
        results.append(audit(solve_power(a, tight), a, exact_opt))
        results.append(audit(solve_rotated_soc(a, tight), a, exact_opt))
    payload = {
        "solver": "mosek",
        "version": ".".join(str(v) for v in mosek.Env.getversion()),
        "fixture": str(FIXTURE),
        "a": a,
        "exact_optimum": str(exact_opt),
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mosek.json").write_text(json.dumps(payload, indent=2))
    for r in results:
        print(
            f"{r['form']:12s} tight={str(r['tight']):5s} {r['solsta']:12s} "
            f"obj={r['audit']['objective']:.17g} err={r['audit']['abs_objective_error_float']:.3e}"
        )


if __name__ == "__main__":
    main()
