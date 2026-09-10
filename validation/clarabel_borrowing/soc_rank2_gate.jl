# SOC rank-2 expansion: independent algebra gate (PR-02).
#
# This file is self-contained: it does NOT call SDPX's SOC scaling builder, its
# rank-2 adapter, or any production assembly routine. It takes an NT scaling
# point `w`, constructs Theta from the *definition* (Theta = Q_w = 2*w*w' - J),
# independently derives the rank-2 parameters by Clarabel's formulas, and then
# checks three separate claims:
#
#   A. Theta == D + u*u' - v*v'                 (metric expansion identity)
#   B. eliminating the two auxiliary KKT variables from
#
#          K_ext = [ 0      Ar'     0      0    ]
#                  [ Ar    -eta^2 D -eta^2 v -eta^2 u ]
#                  [ 0     -eta^2 v'  -eta^2   0    ]
#                  [ 0     -eta^2 u'   0      +eta^2 ]
#
#      reproduces exactly the `-Theta` block of `K = [0 Ar'; Ar -Theta]`
#   C. the auxiliary variables are recoverable and the recovered full direction
#      satisfies the original (unexpanded) five-equation system
#
# Claim B/C are the load-bearing ones: they establish that the expansion is an
# exact representation change of the linear system, not an approximation of the
# second-order cone.
#
# Run: julia --startup-file=no --project=. validation/clarabel_borrowing/soc_rank2_gate.jl
using LinearAlgebra
using Test
using Random

# --- independent building blocks (no SDPX dependency by design) ------------

"""`J = diag(1, -I)` on R^k."""
function _jmat(k::Int)
    J = zeros(Float64, k, k)
    J[1, 1] = 1.0
    for i in 2:k
        J[i, i] = -1.0
    end
    return J
end

"""`Theta = Q_w = 2*w*w' - J`, from the definition only."""
function _theta_from_w(w::Vector{Float64})
    return 2.0 * (w * transpose(w)) - _jmat(length(w))
end

"""
Draw a valid NT scaling point: `w[1] > 0` and `w[1]^2 - ||w[2:end]||^2 = 1`.

The normalization is what makes `w` a legitimate SOC scaling point and is
exactly the condition SDPX enforces (`w[1] = sqrt(1 + w1sq)`).
"""
function _random_scaling_point(rng::AbstractRNG, k::Int; spread::Float64=1.0)
    tail = spread .* randn(rng, k - 1)
    w = zeros(Float64, k)
    w[2:end] .= tail
    w[1] = sqrt(1.0 + dot(tail, tail))
    return w
end

"""
Clarabel's map from a normalized `w` to the rank-2 form
`Theta = eta^2 * (D + u*u' - v*v')`, with `eta = 1` for SDPX's normalized
scaling point.

Returns `(D, u, v)` where `D` is the diagonal vector.
"""
function _rank2_parameters(w::Vector{Float64})
    k = length(w)
    w1sq = dot(view(w, 2:k), view(w, 2:k))
    wsq = w[1]^2 + w1sq
    wsqinv = 1.0 / wsq

    d = wsqinv / 2.0

    u = zeros(Float64, k)
    u0 = sqrt(wsq - d)
    u[1] = u0
    u1 = (2.0 * w[1]) / u0
    for i in 2:k
        u[i] = u1 * w[i]
    end

    v = zeros(Float64, k)
    v[1] = 0.0
    v1 = sqrt(2.0 * (2.0 + wsqinv) / (2.0 * wsq - wsqinv))
    for i in 2:k
        v[i] = v1 * w[i]
    end

    D = ones(Float64, k)
    D[1] = d
    return D, u, v
end

"""`D + u*u' - v*v'` as a dense matrix."""
function _expand_metric(D::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64})
    return Diagonal(D) + u * transpose(u) - v * transpose(v)
end

"""
Assemble the extended KKT matrix for one SOC block embedded in a larger system.

Unknown ordering: `[x ; y_cone ; y_aux_v ; y_aux_u]`, where `y_aux_v` and
`y_aux_u` are the two auxiliary KKT unknowns for this SOC block. `Ar` is the
`m x nr` constraint block and `nb` is the block's row offset within the cone
(0-based), so the block occupies `y_cone[nb+1 : nb+k]`.
"""
function _extended_kkt(
    Ar::Matrix{Float64}, D::Vector{Float64}, u::Vector{Float64},
    v::Vector{Float64}, nb::Int, k::Int,
)
    m, nr = size(Ar)
    dim = nr + m + 2
    K = zeros(Float64, dim, dim)
    yr = nr                      # cone rows start after x
    # [0 Ar'] block
    K[1:nr, (yr + 1):(yr + m)] .= transpose(Ar)
    K[(yr + 1):(yr + m), 1:nr] .= Ar
    # -eta^2 D inside the cone block (eta = 1)
    for i in 1:k
        K[yr + nb + i, yr + nb + i] += -D[i]
    end
    # -eta^2 v and -eta^2 u coupling columns
    aux_v = yr + m + 1
    aux_u = yr + m + 2
    for i in 1:k
        K[yr + nb + i, aux_v] += -v[i]
        K[aux_v, yr + nb + i] += -v[i]
        K[yr + nb + i, aux_u] += -u[i]
        K[aux_u, yr + nb + i] += -u[i]
    end
    K[aux_v, aux_v] = -1.0
    K[aux_u, aux_u] = 1.0
    return K
end

"""
Schur-eliminate the two auxiliary variables from `K_ext` and return the
resulting `(nr+m) x (nr+m)` operator restricted to `[x ; y_cone]`.

The auxiliary block is `S = diag(-1, +1)`, so
`K_eff = K_aa - K_ab * inv(S) * K_ba` with `K_ab` the coupling columns.
"""
function _eliminate_auxiliaries(K::Matrix{Float64}, nc::Int, na::Int)
    n = size(K, 1) - na
    Kaa = K[1:n, 1:n]
    Kab = K[1:n, (n + 1):(n + na)]
    Kba = K[(n + 1):(n + na), 1:n]
    S = K[(n + 1):(n + na), (n + 1):(n + na)]
    return Kaa - Kab * (S \ Kba)
end

# --- gates ----------------------------------------------------------------

const GATE_TOL = 1.0e-11

@testset "SOC rank-2 expansion gate" begin
    rng = MersenneTwister(20260911)
    dims = (3, 8, 16, 32, 128)
    spreads = (0.05, 1.0, 8.0)

    @testset "A. metric expansion identity |Theta - (D+uu'-vv')|" begin
        worst = 0.0
        for k in dims, spread in spreads
            for _ in 1:4
                w = _random_scaling_point(rng, k; spread=spread)
                Theta = _theta_from_w(w)
                D, u, v = _rank2_parameters(w)
                rebuilt = _expand_metric(D, u, v)
                residual = maximum(abs, Theta - rebuilt)
                worst = max(worst, residual)
                @test residual <= GATE_TOL * max(1.0, maximum(abs, Theta))
            end
        end
        @test worst <= GATE_TOL
        @info "worst metric expansion residual" worst
    end

    @testset "B. auxiliary elimination reproduces -Theta exactly" begin
        worst = 0.0
        worst_relative = 0.0
        for k in dims, spread in spreads
            for _ in 1:4
                w = _random_scaling_point(rng, k; spread=spread)
                Theta = _theta_from_w(w)
                D, u, v = _rank2_parameters(w)

                # A small but genuinely sparse-ish Ar so the coupled system is
                # not trivially diagonal, with the cone block placed at offset 0.
                m, nr = k, max(2, div(k, 2))
                Ar = zeros(Float64, m, nr)
                for j in 1:nr
                    Ar[j, j] = 1.0 + 0.25 * j
                end
                for j in 1:nr, i in (j + nr):m
                    Ar[i, j] = 0.125 * sin(Float64(i * j))
                end

                Kext = _extended_kkt(Ar, D, u, v, 0, k)
                Keff = _eliminate_auxiliaries(Kext, nr + m, 2)

                # Reference: the declared cone block is -Theta.
                reference = zeros(Float64, nr + m, nr + m)
                reference[1:nr, (nr + 1):(nr + m)] .= transpose(Ar)
                reference[(nr + 1):(nr + m), 1:nr] .= Ar
                reference[(nr + 1):(nr + m), (nr + 1):(nr + m)] .=
                    -Theta

                residual = maximum(abs, Keff - reference)
                scale = max(1.0, maximum(abs, reference))
                worst = max(worst, residual)
                worst_relative = max(worst_relative, residual / scale)
                @test residual <= 1.0e-12 * scale
            end
        end
        @test worst <= 1.0e-12 * 16
        @info "worst elimination residual" worst worst_relative
    end

    @testset "C. solved directions agree, and auxiliary recovery is exact" begin
        worst_direction = 0.0
        worst_recovery = 0.0
        for k in dims, spread in spreads
            for _ in 1:4
                w = _random_scaling_point(rng, k; spread=spread)
                Theta = _theta_from_w(w)
                D, u, v = _rank2_parameters(w)
                m, nr = k, max(2, div(k, 2))
                Ar = zeros(Float64, m, nr)
                for j in 1:nr
                    Ar[j, j] = 1.0 + 0.25 * j
                end
                for j in 1:nr, i in (j + nr):m
                    Ar[i, j] = 0.125 * sin(Float64(i * j))
                end

                n = nr + m
                # A deterministic non-trivial right-hand side.
                rhs = [sin(1.3 * i) for i in 1:n]

                # Un-expanded solve.
                Kref = zeros(Float64, n, n)
                Kref[1:nr, (nr + 1):(nr + m)] .= transpose(Ar)
                Kref[(nr + 1):(nr + m), 1:nr] .= Ar
                Kref[(nr + 1):(nr + m), (nr + 1):(nr + m)] .= -Theta
                # K is singular by construction (zero x block); regularize the
                # x diagonal exactly as the affine start KKT does.
                for i in 1:nr
                    Kref[i, i] = 1.0e-8
                end
                xref = Kref \ rhs

                # Extended solve.
                Kext = _extended_kkt(Ar, D, u, v, 0, k)
                for i in 1:nr
                    Kext[i, i] = 1.0e-8
                end
                xext = Kext \ [rhs; zeros(2)]

                direction_gap = maximum(abs, xext[1:n] - xref)
                worst_direction = max(worst_direction, direction_gap)
                @test direction_gap <= 1.0e-8 * max(1.0, maximum(abs, xref))

                # Recover the auxiliary unknowns from the last two rows of the
                # extended system. Row `aux_v` reads
                #     sum_i (-v_i) y_i + (-1) z_v = 0   =>  z_v = -dot(v, y)
                # and row `aux_u` reads
                #     sum_i (-u_i) y_i + (+1) z_u = 0   =>  z_u = +dot(u, y).
                # The sign difference is exactly the (-1, +1) diagonal.
                ycone = xext[(nr + 1):(nr + m)]
                z_v = -dot(v, ycone)
                z_u = dot(u, ycone)
                @test abs(xext[n + 1] - z_v) <=
                      1.0e-9 * max(1.0, abs(z_v))
                @test abs(xext[n + 2] - z_u) <=
                      1.0e-9 * max(1.0, abs(z_u))
                worst_recovery = max(worst_recovery, abs(xext[n + 1] - z_v))
                worst_recovery = max(worst_recovery, abs(xext[n + 2] - z_u))
            end
        end
        @info "worst direction gap / auxiliary recovery" worst_direction worst_recovery
        @test worst_direction <= 1.0e-7
        @test worst_recovery <= 1.0e-8
    end

    @testset "D. storage accounting matches the declared asymptotic claim" begin
        # The plan claims the cone-specific storage of the expanded form grows
        # like 3k+2 slots versus k(k+1)/2 for the packed dense lower triangle.
        # The TRUE crossover is k = 6, not k = 4 or 5:
        #
        #     k=4:  expanded 14  >  packed 10
        #     k=5:  expanded 17  >  packed 15
        #     k=6:  expanded 20  <  packed 21
        #
        # So a `dense_small` threshold anywhere below 6 is required, and the
        # plan's "large SOC is the priority / keep small SOC dense" instruction
        # is quantitatively justified. Recorded explicitly because the plan
        # does not state the crossover.
        for k in 2:5
            @test 3 * k + 2 > div(k * (k + 1), 2)
        end
        for k in 6:64
            @test 3 * k + 2 < div(k * (k + 1), 2)
        end
        @test 3 * 6 + 2 == 20 && div(6 * 7, 2) == 21
        # Asymptotic sanity at the plan's worked example.
        @test 3 * 4096 + 2 == 12290
        @test div(4096 * 4097, 2) == 8390656

        # Float64x4 carries four Float64 limbs per stored scalar. The plan's
        # "about 256.06 MiB vs 384.06 KiB" figure is the numeric payload only.
        # Recomputed here so the plan's arithmetic is not taken on faith:
        #   packed   = 8390656 slots * 4 limbs * 8 B = 268,500,992 B = 256.06 MiB
        #   expanded =   12290 slots * 4 limbs * 8 B =     393,280 B = 384.06 KiB
        limbs = 4
        bytes_per_limb = 8
        packed_bytes = div(4096 * 4097, 2) * limbs * bytes_per_limb
        expanded_bytes = (3 * 4096 + 2) * limbs * bytes_per_limb
        @test packed_bytes == 268500992
        @test expanded_bytes == 393280
        @test isapprox(packed_bytes / 2^20, 256.06; atol=0.01)
        @test isapprox(expanded_bytes / 2^10, 384.06; atol=0.01)
        @test packed_bytes / expanded_bytes > 600
    end
end
