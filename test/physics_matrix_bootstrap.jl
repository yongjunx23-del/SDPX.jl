using Test
using SparseArrays
using LinearAlgebra
using SDPX

include(joinpath(
    @__DIR__, "..", "benchmark", "bootstrap", "physics",
    "matrix_bootstrap", "matrix_bootstrap.jl",
))
using .MatrixBootstrap

function _entry_signature(entry::AffineEntry)
    return (entry.constant,
        Dict(term.variable => term.coefficient for term in entry.terms))
end

function _entry_equal(a::AffineEntry, b::AffineEntry)
    return _entry_signature(a) == _entry_signature(b)
end

function _row_coefficients(artifact::SDPSliceArtifact, row::Int)
    return Dict(column => artifact.equalities[row, column]
        for column in axes(artifact.equalities, 2)
        if !iszero(artifact.equalities[row, column]))
end

@testset "matrix-bootstrap ordered O(2) moment algebra and E34" begin
    coordinate = word(:Z, :Z, :Zbar, :Zbar)
    @test canonical_moment(coordinate) ==
          canonical_moment(word(:Z, :Zbar, :Zbar, :Z))

    # A quantum momentum cannot be moved around a matrix trace for free.
    momentum_word = word(:Z, :Pi, :Zbar, :Pibar)
    cyclic_rotation = word(:Pi, :Zbar, :Pibar, :Z)
    @test canonical_trace_word(momentum_word) == canonical_trace_word(cyclic_rotation)
    @test canonical_moment(momentum_word)[1] != canonical_moment(cyclic_rotation)[1]

    forward = canonical_moment(word(:Z, :Pibar))
    backward = canonical_moment(word(:Pibar, :Z))
    @test forward == (word(:Z, :Pibar), Rat(1))
    @test backward == (word(:Z, :Pibar), Rat(-1))
    @test canonical_moment(word(:Pi))[2] == Rat(0)

    # Independent Appendix-E E34 check in the complex basis.  Time reversal
    # gives f+b=0 and [Z,Pibar]=1 gives f-b=1.
    f = affine_moment_reduction(word(:Z, :Pibar))
    b = affine_moment_reduction(word(:Pibar, :Z))
    @test f.relation == b.relation == :canonical_commutator_E34
    @test f.representative === nothing
    @test b.representative === nothing
    @test f.constant + b.constant == Rat(0)
    @test f.constant - b.constant == Rat(1)
    @test f.constant == Rat(1, 2)
    @test b.constant == Rat(-1, 2)

    reflected_f = affine_moment_reduction(word(:Zbar, :Pi)).constant
    reflected_b = affine_moment_reduction(word(:Pi, :Zbar)).constant
    @test reflected_f == Rat(1, 2)
    @test reflected_b == Rat(-1, 2)
    # The D=2 gauge generator is [Z,Pibar]+[Zbar,Pi]-2.
    @test (f.constant - b.constant) + (reflected_f - reflected_b) - Rat(2) == Rat(0)

    @test word_charge(momentum_word) == 0
    @test word_level(momentum_word) == 6
    @test enumerate_words(4; charge=0, trace=true)[1] == Word(())
    @test all(word_charge(w) == 0 for w in enumerate_words(8; charge=0, trace=true))
    @test issorted(enumerate_words(8; trace=false), by=w -> (word_level(w), w))
end

@testset "matrix-bootstrap Eq. (4) formal derivation" begin
    raw = MatrixBootstrap._letter_derivative(UInt8(3), Rat(1))
    coefficients = Dict{Word,Rat}()
    for (coefficient, w) in raw
        coefficients[w] = get(coefficients, w, Rat(0)) + coefficient
    end
    # d Pi = M^2 Z + [Z,[Zbar,Z]].
    @test coefficients == Dict(
        word(:Z) => Rat(1),
        word(:Z, :Zbar, :Z) => Rat(2),
        word(:Z, :Z, :Zbar) => Rat(-1),
        word(:Zbar, :Z, :Z) => Rat(-1),
    )

    terms = eom_terms(word(:Pi, :Zbar), 1)
    @test !isempty(terms)
    @test all(isempty(w) || word_charge(w) == 0 for w in keys(terms))
    reflected = eom_terms(word(:Pibar), 1)
    @test length(reflected) == length(eom_terms(word(:Pi), 1))
    @test all(isempty(w) || abs(word_charge(w)) == 1 for w in keys(reflected))
end

@testset "matrix-bootstrap honest low-order build scopes" begin
    a = build_lin_zheng(level=4)
    b = build_lin_zheng(level=4)
    @test a.fingerprint == b.fingerprint
    @test validate_artifact(a)
    @test a.model == :lin_zheng_o2_low_order_relaxation
    @test a.reference_status == :build_only
    @test a.objective === nothing
    @test a.reference_objective === nothing
    @test a.metadata.is_affine
    @test !a.metadata.paper_equivalent
    @test a.metadata.publication_claim == :none
    @test a.metadata.primary_doi == "10.1103/cyq8-4sd7"
    @test a.metadata.published_census_status == :metadata_only_fail_closed
    @test a.metadata.kinematic_constraints.canonical_commutator ==
          :appendix_E34_affine_reduction
    @test :full_gauge_Ward_tower in a.metadata.omitted_constraints
    @test :published_Table_1_2_reproduction in a.metadata.omitted_constraints
    @test !haskey(a.moment_indices, word(:Z, :Pibar))
    zero_moments = zeros(Rat, length(a.variables))
    @test moment_value(zero_moments, a, word(:Z, :Pibar)) == Rat(1, 2)
    @test moment_value(zero_moments, a, word(:Pibar, :Z)) == Rat(-1, 2)
    charge_one = only(filter(block -> block.name == :M_charge_1, a.psd_blocks))
    @test charge_one.basis == Word[word(:Z), word(:Pi)]
    @test _entry_signature(charge_one.entries[1, 2]) ==
          (Rat(1, 2), Dict{Int,Rat}())
    @test _entry_signature(charge_one.entries[2, 1]) ==
          (Rat(1, 2), Dict{Int,Rat}())

    @test published_counts().free ==
          Dict(4 => 3, 6 => 8, 8 => 22, 10 => 77, 12 => 326, 14 => 1569)
    @test published_counts().all ==
          Dict(4 => 14, 6 => 94, 8 => 614, 10 => 4086, 12 => 27830,
               14 => 192374)
    @test published_intervals()[(:massive, :energy)] ==
          (Rat(1172098376, 1000000000), Rat(1172098408, 1000000000))

    @test !isempty(a.variables)
    @test size(a.equalities, 2) == length(a.variables)
    @test all(size(block, 1) == size(block, 2) for block in a.psd_blocks)
    for block in a.psd_blocks, i in axes(block.entries, 1), j in axes(block.entries, 2)
        @test _entry_equal(block.entries[i, j], block.entries[j, i])
    end
    # Do not silently retain 0=0 rows after affine moment reduction.
    @test all(nnz(a.equalities[row, :]) > 0 || !iszero(a.rhs[row])
              for row in eachindex(a.rhs))

    for (level, scope) in ((4, :tiny), (6, :small), (8, :medium))
        artifact = build_lin_zheng(level=level)
        @test artifact.level == level
        @test artifact.metadata.supported_scope == scope
        @test artifact.metadata.published_free_variables[level] > 0
        @test artifact.metadata.published_all_variables[level] >=
              artifact.metadata.published_free_variables[level]
        @test all(variable.index == i for (i, variable) in enumerate(artifact.variables))
        @test count(block -> block.kind == :ground_state_positivity,
                    artifact.psd_blocks) == 1
    end
    for level in (10, 12, 14)
        @test_throws ArgumentError build_lin_zheng(level=level)
    end

    scan = build_lin_zheng(level=6, scan_observable=:x2,
                           scan_value=77800898 // 100000000)
    @test :fixed_scan in scan.equality_labels
    @test scan.metadata.scan_is_slice
    @test scan.spec.scan_value == Rat(77800898, 100000000)
    @test scan.fingerprint != build_lin_zheng(level=6).fingerprint
end

@testset "matrix-bootstrap Appendix E40 and ground-state E42" begin
    mass2 = Rat(3, 2)
    scan_value = Rat(7, 5)
    artifact = build_lin_zheng(level=4, mass2=mass2,
        scan_observable=:energy, scan_value=scan_value)

    xkey = canonical_moment(word(:Z, :Zbar))[1]
    pkey = canonical_moment(word(:Pi, :Pibar))[1]
    xid = artifact.moment_indices[xkey]
    pid = artifact.moment_indices[pkey]

    # E40: E=-3/2 <Pi Pibar> + M^2/2 <Z Zbar>.
    scan_row = only(findall(==(:fixed_scan), artifact.equality_labels))
    @test artifact.rhs[scan_row] == scan_value
    @test _row_coefficients(artifact, scan_row) ==
          Dict(pid => Rat(-3, 2), xid => Rat(3, 4))
    @test artifact.metadata.scan_expression ==
          (PiPibar=Rat(-3, 2), ZZbar=Rat(3, 4),
           source_equation=:appendix_E40)

    # E42 at D=2: [[1,-2p],[-2p,2x+M^2]] >= 0.
    ground = only(filter(block -> block.kind == :ground_state_positivity,
                         artifact.psd_blocks))
    @test ground.metadata.source_equations == (:main_eq_6, :appendix_E42)
    @test ground.metadata.equation_exact
    @test !ground.metadata.hierarchy_complete
    @test _entry_signature(ground.entries[1, 1]) == (Rat(1), Dict{Int,Rat}())
    @test _entry_signature(ground.entries[1, 2]) ==
          (Rat(0), Dict(pid => Rat(-2)))
    @test _entry_signature(ground.entries[2, 1]) ==
          (Rat(0), Dict(pid => Rat(-2)))
    @test _entry_signature(ground.entries[2, 2]) ==
          (mass2, Dict(xid => Rat(2)))

    witness = zeros(Rat, length(artifact.variables))
    witness[pid] = Rat(-1, 2)
    witness[xid] = Rat(1)
    exact_ground = affine_matrix(ground, Rat; values=witness)
    @test exact_ground == Rat[1 1; 1 7//2]
    @test exact_ground[1, 1] > 0
    @test det(exact_ground) == Rat(5, 2)

    float_ground = affine_matrix(ground, Float64; values=Float64.(witness))
    @test float_ground == [1.0 1.0; 1.0 3.5]
    @test eigmin(Symmetric(float_ground)) > 0
    setprecision(BigFloat, 256) do
        high_ground = affine_matrix(ground, BigFloat; values=BigFloat.(witness))
        @test high_ground[1, 1] == BigFloat(1)
        @test high_ground[1, 2] == BigFloat(1)
        @test high_ground[2, 2] == BigFloat(7) / BigFloat(2)
        @test det(high_ground) == BigFloat(5) / BigFloat(2)
    end
end

function _lowering_fixture()
    spec = MatrixBootstrapSpec()
    variables = [AffineVariable(1, "x", :witness, nothing, nothing)]
    entry = AffineEntry(Rat(2), [AffineTerm(1, Rat(1))])
    block = PSDBlock(:lowering_oracle, :oracle, Word[Word(())],
                     reshape([entry], 1, 1),
                     (source_equations=(:independent_lowering_oracle,),))
    equalities = sparse([1], [1], Rat[1], 1, 1)
    return SDPSliceArtifact(spec, :lin_zheng_o2, "arXiv:2507.21007v3",
        :lowering_oracle, 4, (;), Word[Word(())], variables, Dict{Word,Int}(),
        zeros(Int, 0, 0), equalities, Rat[1], Symbol[:witness], PSDBlock[block],
        :build_only, nothing, nothing, (;), "lowering-oracle")
end

@testset "matrix-bootstrap SDPX A(x)-C lowering sign" begin
    fixture = _lowering_fixture()
    # Artifact convention: F(x)=2+x >= 0 and x=1.  The independent feasible
    # witness is x=1, X=3.  SDPX must therefore store C=-2.
    for T in (Float64, BigFloat)
        if T === BigFloat
            setprecision(BigFloat, 256)
        end
        problem = to_sdp_problem(fixture, T)
        @test problem.C[1][1, 1] == -T(2)
        x = T[1]
        X = [reshape(T[3], 1, 1)]
        y = T[0]
        Y = [zeros(T, 1, 1)]
        primal, dual = SDPX.solution_residuals(problem, x, X, y, Y)
        @test primal == zero(T)
        @test dual == zero(T)
        @test X[1][1, 1] >= zero(T)
    end
end

@testset "matrix-bootstrap provenance whitelist" begin
    @test_throws ArgumentError build_matrix_bootstrap(source=:unknown_source)
    @test_throws ArgumentError build_lin_zheng(source_version="arXiv:2507.21007v2")
    @test_throws ArgumentError build_lin_zheng(reference_status=:published_interval)
    @test_throws ArgumentError build_lin_zheng(reference_status=:cross_check_only)

    # Mutable artifact storage must not be silently accepted after hashing.
    tampered = build_lin_zheng(level=4)
    tampered.rhs[1] += Rat(1)
    @test_throws ArgumentError validate_artifact(tampered; rebuild=false)
end

@testset "matrix-bootstrap injected build-only catalog" begin
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl"))
    end
    harness = Main.PhysicsBenchmarkHarness
    catalog = harness.load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "bootstrap", "applications", "matrix_bootstrap", "catalog.jl",
    ))
    @test catalog.name == :lin_zheng26_matrix_bootstrap
    @test length(harness.catalog_entries(catalog, :scaling)) == 3
    entry = only(harness.catalog_entries(catalog, :smoke))
    spec = harness.catalog_spec(catalog, entry.problem_id)
    @test spec.reference.status == :build_only
    @test spec.parameters.primary_doi == "10.1103/cyq8-4sd7"
    built = harness.build_problem(catalog, spec, Float64)
    @test built.kind == :sdp
    @test built.problem isa SDPX.SDPProblem
    @test built.external_checksum == built.artifact.fingerprint == spec.fingerprint
    @test isempty(harness.validate_result(catalog, spec, built, nothing, nothing))
end
