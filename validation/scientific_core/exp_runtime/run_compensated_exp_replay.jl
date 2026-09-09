# R0-E compensated Exp replay driver — bounded validation-only experiment.
#
# Run (own env, bounded process):
#   JULIA_PROJECT=/tmp/r0e-comp-exp-env julia --startup-file=no --threads=1 \
#     --gcthreads=1 --heap-size-hint=2G run_compensated_exp_replay.jl
# with BLAS/OMP/MKL threads = 1. Each run must finish in <= 180 s.
#
# What it does:
#   1. Asserts in-process identity: loaded SDPX root == this worktree, git
#      HEAD, clean tracked checkout (only the three research deliverables may
#      be untracked), VERSION, single-thread BLAS, and before/after hashes of
#      every file it did not modify (src/**, test/**, Project/Manifest).
#   2. Loads the frozen hex-word records, runs the research evaluator on the
#      stored words, audits pairings, and cross-checks against the separate
#      BigFloat/MPFR independent verifier (never fed back into candidates).
#   3. Writes a machine-readable TOML report and prints a concise summary.
#
# Validation-only: no production file is modified; the production Float64 Exp
# path is exercised read-only for identity purposes only.

using LinearAlgebra
using SHA
using TOML
using Dates

const WORKTREE = realpath(joinpath(@__DIR__, "..", "..", ".."))
const RUNDIR = @__DIR__
# Exact-source mode: when the driver exports the expected root/HEAD, identity
# is asserted strictly.  A plain integration run records the observed state
# (the /tmp <-> /private/tmp alias makes naive path equality unreliable).
const EXPECT_ROOT = get(ENV, "SDPX_EXPECT_ROOT", "")
const EXPECT_HEAD = get(ENV, "SDPX_EXPECT_HEAD", "")

# --- bounded-process assertions -------------------------------------------
@assert Threads.nthreads() == 1
BLAS.set_num_threads(1)
@assert BLAS.get_num_threads() == 1
@assert VERSION == v"1.12.6"

using SDPX
include(joinpath(RUNDIR, "compensated_exp_reference.jl"))
using .CompensatedExpReference
const CER = CompensatedExpReference
const IND = CER.Independent

# --- in-process identity ----------------------------------------------------
sdpx_root = realpath(pkgdir(SDPX))
if !isempty(EXPECT_ROOT)
    @assert sdpx_root == realpath(EXPECT_ROOT) "SDPX root $sdpx_root != $(EXPECT_ROOT)"
end
HEAD_BEFORE = strip(read(`git -C $WORKTREE rev-parse HEAD`, String))
if !isempty(EXPECT_HEAD)
    @assert HEAD_BEFORE == EXPECT_HEAD "HEAD $HEAD_BEFORE != $EXPECT_HEAD"
end
println("SDPX_ROOT=", sdpx_root)
println("GIT_HEAD=", HEAD_BEFORE)
println("VERSION=", VERSION, " BLAS_THREADS=", BLAS.get_num_threads())

if !isempty(EXPECT_ROOT) || !isempty(EXPECT_HEAD)
    tracked_clean = strip(read(`git -C $WORKTREE status --porcelain -- src test Project.toml Manifest.toml`, String))
    @assert isempty(tracked_clean) "tracked files modified: $tracked_clean"
end
println("TRACKED_CLEAN=src,test,Project.toml,Manifest.toml")

function shahex(path)
    isfile(path) || return "absent"
    return bytes2hex(sha256(read(path)))
end
const PINNED_FILES = ["Project.toml", "Manifest.toml"]
hashes_before = Dict(f => shahex(joinpath(WORKTREE, f)) for f in PINNED_FILES)

f64(hex::String) = reinterpret(Float64, parse(UInt64, hex, base = 16))
hexof(x::Float64) = string(reinterpret(UInt64, x), base = 16, pad = 16)

const REC = Dict{String, Any}(
    "A4" => Dict("iter" => 11, "block" => 4,
        "s" => ["3ff5bfae30dcdde5", "3ff3cf74a3baf989", "400db161f421c165"],
        "y" => ["c00db1453507976d", "3fd7fe25cb0a2f71", "3ff3bfae8d5f1e85"],
        "oldshadow" => ["40c4e5c0df117f10", "40c2fb3760a25caa", "40dc8a15bd37352e"]),
    "A7" => Dict("iter" => 11, "block" => 7,
        "s" => ["3ff5bfae30df9e1f", "3ff3cf74a3cc49ed", "400db161f421c165"],
        "y" => ["c00db1453507976d", "3fd7fe25cb0a1f8b", "3ff3bfae8d6056aa"],
        "oldshadow" => ["40c4e5c0848c8718", "40c2fb370e6b4416", "40dc8a154198eae0"]),
    "A10" => Dict("iter" => 11, "block" => 10,
        "s" => ["3ff5be86ed75af00", "3ff3c4141eee2aa4", "400db161f421c165"],
        "y" => ["c00db14535074013", "3fd7fe180cb2d8c2", "3ff3bfd5c2dc2182"],
        "oldshadow" => ["40bb4ee488152a53", "40b8ce0c77f11783", "40d2a5bd790576e7"]),
    "B4" => Dict("iter" => 16, "block" => 4,
        "s" => ["4004ca298eaa7b5c", "4002ecabff56082f", "401c62a1e772366c"],
        "y" => ["c01c62a1355cffc5", "3fe663b154e08ffe", "4002ec817c08a6e0"],
        "oldshadow" => nothing),
    "B7" => Dict("iter" => 16, "block" => 7,
        "s" => ["4004ca2934c84025", "4002eca86b8b7686", "401c62a1e772366c"],
        "y" => ["c01c62a1355cffe4", "3fe663b154dd54ce", "4002ec817c4bccbb"],
        "oldshadow" => nothing),
    "B10" => Dict("iter" => 16, "block" => 10,
        "s" => ["4004ca17c9e21bfe", "4002ebf0c5e69583", "401c62a1e772366c"],
        "y" => ["c01c62a1355d34b6", "3fe663b14c2da9f8", "4002ec80739c6c1c"],
        "oldshadow" => nothing),
)

report = Dict{String, Any}(
    "meta" => Dict{String, Any}(
        "worktree" => WORKTREE, "git_head_before" => HEAD_BEFORE,
        "julia_version" => string(VERSION), "blas_threads" => 1,
        "threads" => 1, "created_utc" => string(now(UTC)),
        "design" => "validation/scientific_core/exp_runtime/COMPENSATED_EXP_DESIGN.md",
        "capture" => "exp-triple-capture iterations 11 (A) and 16 (B) hex triples; " *
            "iterations 17-24 contribute REJ metadata rows only (no further hex words)",
    ),
    "records" => Dict{String, Any}(),
)

setprecision(512) do
    for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
        spec = REC[nm]
        s = Tuple(f64(h) for h in spec["s"])
        d = Tuple(f64(h) for h in spec["y"])
        r = CER.evaluate_conjugate(d[1], d[2], d[3])
        entry = Dict{String, Any}(
            "iter" => spec["iter"], "block" => spec["block"],
            "s_hex" => spec["s"], "d_hex" => spec["y"],
            "status" => string(r.status),
            "reason" => string(r.reason),
            "stage" => string(r.stage),
        )
        if r.status !== :refused
            entry["l0"] = Dict("h" => r.l0.h, "l" => r.l0.l, "E" => r.l0.E,
                "dexp" => r.l0.dexp, "q" => r.l0.q,
                "remainder" => r.l0.remainder, "target" => r.l0.target)
            entry["D"] = Dict("h" => r.D.h, "l" => r.D.l, "E" => r.D.E,
                "lo" => r.D.lo, "hi" => r.D.hi)
            entry["L"] = Dict("h" => r.L.h, "l" => r.L.l, "E" => r.L.E,
                "dexp" => r.L.dexp, "q" => r.L.q,
                "remainder" => r.L.remainder, "target" => r.L.target)
            entry["root"] = Dict("rho" => r.root.rho,
                "residual" => r.root.residual, "Rmax" => r.root.Rmax,
                "threshold" => r.root.threshold, "lo" => r.root.lo,
                "hi" => r.root.hi, "iterations" => r.root.iterations,
                "E_rho" => r.root.E_rho)
            entry["shadow_hex"] = [r.out_hex.X, r.out_hex.Y, r.out_hex.Z]
            entry["recon_errors"] = Dict("e_X" => r.recon_errors.e_X,
                "e_Y" => r.recon_errors.e_Y, "e_Z" => r.recon_errors.e_Z)
            entry["P"] = Dict("value" => r.P.value, "E" => r.P.E,
                "E_P" => r.P.E_P, "Pmin" => r.P.Pmin, "p_star" => r.P.p_star)
            entry["replay"] = Dict("B1" => r.replay.B1, "B2" => r.replay.B2,
                "B3" => r.replay.B3, "E_YZ" => r.replay.E_YZ,
                "E_L" => r.replay.E_L, "L0" => r.replay.L0,
                "Ymin" => r.replay.Ymin, "Zmin" => r.replay.Zmin)
            entry["gradient"] = [Dict("g" => first(r.gradient[i]),
                "E" => r.gradient[i].E) for i in 1:3]
            entry["ops"] = Dict("two_prod" => r.ops.two_prod,
                "two_sum" => r.ops.two_sum, "divisions" => r.ops.divisions,
                "series_evals" => r.ops.series_evals)
            # Independent-reference deltas (MPFR hulls; verifier only).
            lo, hi = IND.log_ratio(d[3], -d[1])
            entry["indep_l0"] = Dict(
                "delta" => Float64(abs((BigFloat(r.l0.h) + BigFloat(r.l0.l)) -
                    (BigFloat(lo) + BigFloat(hi)) / 2)),
                "hull_width" => Float64(abs(hi - lo)))
            Dref = BigFloat(1) - BigFloat(d[2]) / BigFloat(d[1]) +
                   (BigFloat(lo) + BigFloat(hi)) / 2
            entry["indep_D"] = Dict(
                "delta" => Float64(abs((BigFloat(r.D.h) + BigFloat(r.D.l)) - Dref)),
                "hull_width" => Float64(abs(hi - lo)))
            rref, _ = IND.root(d[1], d[2], d[3])
            entry["indep_rho"] = Dict(
                "delta" => Float64(abs(BigFloat(r.root.rho) - rref)),
                "ref" => Float64(rref))
            X, Y, Z = r.out_words.X, r.out_words.Y, r.out_words.Z
            plo, phi = IND.replay_P(X, Y, Z)
            entry["indep_P"] = Dict(
                "delta" => Float64(abs(BigFloat(r.P.value) -
                    (BigFloat(plo) + BigFloat(phi)) / 2)),
                "hull_width" => Float64(abs(phi - plo)))
            # Pairing audit on the reconstructed shadow + primal gradient.
            g = CER.compensated_gradient_words(s[1], s[2], s[3])
            @assert g.status === :ok
            a = CER.audit_pairings(s_trial = s, d_trial = d,
                shadow = (X, Y, Z), grad_primal = g.words)
            entry["audit"] = Dict(
                "m12_lo" => a.m12.lo, "m12_hi" => a.m12.hi,
                "m12_gate" => string(a.m12.gate),
                "m21_lo" => a.m21.lo, "m21_hi" => a.m21.hi,
                "m21_gate" => string(a.m21.gate),
                "cross_gate" => string(a.cross.gate),
                "verdict" => string(a.reason))
            if spec["oldshadow"] !== nothing
                so = Tuple(f64(h) for h in spec["oldshadow"])
                ao = CER.audit_pairings(s_trial = s, d_trial = d, shadow = so,
                    grad_primal = g.words)
                entry["oldshadow_audit"] = Dict(
                    "m12_gate" => string(ao.m12.gate),
                    "m21_gate" => string(ao.m21.gate),
                    "verdict" => string(ao.reason))
            end
        end
        report["records"][nm] = entry
    end
end

# --- concise summary --------------------------------------------------------
println("---")
for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
    e = report["records"][nm]
    if e["status"] == "refused"
        println(nm, ": REFUSED reason=", e["reason"], " stage=", e["stage"])
    else
        println(nm, ": ", e["status"],
            " Rmax=", e["root"]["Rmax"], " thr=", e["root"]["threshold"],
            " E_P=", e["P"]["E_P"], " B1=", e["replay"]["B1"],
            " m12=", e["audit"]["m12_gate"],
            " m21=", e["audit"]["m21_gate"],
            " cross=", e["audit"]["cross_gate"],
            " indep_dl0=", e["indep_l0"]["delta"],
            " indep_dP=", e["indep_P"]["delta"])
    end
end

# --- in-process acceptance of the required outcomes -------------------------
@assert report["records"]["A7"]["oldshadow_audit"]["m12_gate"] == "fail"
@assert report["records"]["A7"]["audit"]["m12_gate"] == "pass"
@assert report["records"]["A7"]["audit"]["m21_gate"] == "fail"
for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
    @assert !occursin("unrepresentable", sprint(show, report["records"][nm]))
end

# --- machine-readable report + after-hashes ---------------------------------
open(joinpath(RUNDIR, "compensated_exp_replay_report.toml"), "w") do io
    TOML.print(io, report)
end
println("REPORT=compensated_exp_replay_report.toml")

HEAD_AFTER = strip(read(`git -C $WORKTREE rev-parse HEAD`, String))
@assert HEAD_AFTER == HEAD_BEFORE
tracked_after = strip(read(`git -C $WORKTREE status --porcelain -- src test Project.toml Manifest.toml`, String))
@assert isempty(tracked_after)
hashes_after = Dict(f => shahex(joinpath(WORKTREE, f)) for f in PINNED_FILES)
@assert hashes_after == hashes_before
println("HEAD_AFTER=", HEAD_AFTER, " HASHES_UNCHANGED=true")
println("REPLAY_DONE")