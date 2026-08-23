#!/usr/bin/env julia

# Build one reduced CSDR PSD dual in Float64x4 and serialize the exact arrays
# consumed by both Clarabel and SDPX.

using Serialization
using SHA

const REFERENCE_ROOT = normpath(get(
    ENV,
    "CLARABEL_CSDR_ROOT",
    get(ENV, "CLARABEL_CSDR_ROOT", joinpath(@__DIR__, "reference-data")),
))
const REFERENCE_RUNNER =
    joinpath(REFERENCE_ROOT, "scripts", "jump_clarabel_double64.jl")

isfile(REFERENCE_RUNNER) ||
    error("reference CSDR runner not found at $REFERENCE_RUNNER")
include(REFERENCE_RUNNER)

env_int(name, default) = parse(Int, get(ENV, name, string(default)))

function canonical_psd_dual(workspace)
    T = eltype(workspace.p.LE)
    ncell = workspace.ncell
    nfree = size(workspace.null_basis, 2)
    m = nfree + ncell

    u_particular = workspace.p.Au * workspace.lambda_particular
    v_particular = workspace.p.Av * workspace.lambda_particular
    Au_free = workspace.p.Au * workspace.null_basis
    Av_free = workspace.p.Av * workspace.null_basis

    c = zeros(T, m)
    A = [zeros(T, m, 2, 2) for _ in 1:ncell]
    C = [zeros(T, 2, 2) for _ in 1:ncell]

    for cell in 1:ncell
        scale = workspace.cone_scale[cell]
        u0 = scale * u_particular[cell]
        v0 = scale * v_particular[cell]

        # SDPX uses X_l = sum_i x_i A_i^l - C_l.  The desired block is
        # [[r, u], [u, r - 2v]], including the affine particular solution.
        C[cell][1, 2] = -u0
        C[cell][2, 1] = -u0
        C[cell][2, 2] = 2v0

        for free_index in 1:nfree
            variable_scale = workspace.free_scale[free_index]
            coefficient_u =
                scale * Au_free[cell, free_index] * variable_scale
            coefficient_v =
                scale * Av_free[cell, free_index] * variable_scale
            A[cell][free_index, 1, 2] = coefficient_u
            A[cell][free_index, 2, 1] = coefficient_u
            A[cell][free_index, 2, 2] = -2coefficient_v
        end

        support_index = nfree + cell
        A[cell][support_index, 1, 1] = one(T)
        A[cell][support_index, 2, 2] = one(T)
        c[support_index] = workspace.p.weights[cell] / scale
    end

    B = Matrix{T}(undef, m, 0)
    b = T[]
    return (; c, A, C, B, b, ncell, nfree, m)
end

function main(args)
    length(args) == 1 || error("usage: prepare_problem.jl OUTPUT.bin")
    output = abspath(args[1])
    mkpath(dirname(output))

    J = env_int("CSDR_J", 4)
    K = env_int("CSDR_K", 1)
    na = env_int("CSDR_NA", 2)
    nmu = env_int("CSDR_NMU", 40)
    start_m = env_int("CSDR_START_M", 0)
    bits = env_int("CSDR_BITS", 256)

    options = parse_options([
        "--J=$J",
        "--K=$K",
        "--na=$na",
        "--nmu=$nmu",
        "--nalpha=2",
        "--lambda-values=-0.5,0",
        "--start-m=$start_m",
        "--energy-sampling=uniform_theta",
        "--energy-power=1",
        "--bits=$bits",
        "--scalar=float64x4",
        "--target=g0",
        "--sense=max",
        "--formulation=direct",
        "--dual-basis=reduced",
        "--cone-backend=psd",
        "--cone-scale=balanced",
        "--functional-scale=0",
        "--presolve=0",
        "--regularization-exp=24",
        "--static-regularization=0",
        "--dynamic-regularization=0",
        "--tol-exp=7",
        "--time-limit=600",
        "--iter-limit=1000",
    ])

    data = setprecision(BigFloat, bits) do
        alpha_values = BigFloat[-1 // 2, 0]
        base = build_problem(
            BigFloat;
            J,
            K,
            na,
            nmu,
            start_m,
            energy_sampling = :uniform_theta,
            energy_power = 1,
            alpha_values,
        )
        workspace = build_direct_model(Float64x4, base, options)
        canonical = canonical_psd_dual(workspace)
        metadata = (;
            J,
            K,
            na,
            nmu,
            start_m,
            bits,
            scalar = "Float64x4",
            formulation = "reduced direct PSD dual",
            physical_normalization = 2,
        )
        (; canonical..., metadata)
    end

    open(output, "w") do io
        serialize(io, data)
    end
    digest = bytes2hex(open(sha256, output))
    println(
        "prepared path=$output sha256=$digest blocks=$(data.ncell) " *
        "free=$(data.nfree) variables=$(data.m)",
    )
end

main(ARGS)
