using Test, Serialization
include("julia112_logical_storage.jl")
const JLS = Julia112LogicalStorage

@testset "pinned layout facts, not allocator evidence" begin
    facts = JLS.runtime_facts()
    @test !ismutabletype(BigFloat)
    @test Base.allocatedinline(BigFloat)
    @test sizeof(BigFloat) == 8
    @test facts.custom_bytes == (32, 64, 128)
    for changed in ((word_bits=32,), (kernel=:Linux,), (mpfr_version=v"4.2.1",),
                    (julia_commit="different",), (bigfloat_inline=false,),
                    (mpfr_offsets=(0, 8, 16, 24, 40),), (mpfr_patches=("local",),))
        @test_throws ArgumentError JLS.check_facts(merge(facts, changed))
    end
    fingerprints = JLS.runtime_fingerprints()
    @test all(x -> isfile(x.path) && length(x.sha256) == 64, values(fingerprints))
    @test_throws ArgumentError JLS.independent_array_charge(128, 1)
    @test_throws ArgumentError JLS.independent_array_charge(256, -1)
    @test_throws ArgumentError JLS.independent_array_charge(256, 1, 3)
    @test_throws OverflowError JLS.independent_array_charge(256, typemax(Int))
end

@testset "actual BigFloat Memory, aliases, and inline array slots" begin
    for (bits, payload, per_slot) in ((256, 64, 104), (512, 96, 136), (1024, 160, 200))
        setprecision(BigFloat, bits) do
            a = [BigFloat(i) for i in 1:4]
            actual = [getfield(x, :d) for x in a]
            @test all(m -> length(m) * 8 == payload, actual)
            @test all(i == j || actual[i] !== actual[j] for i in 1:4, j in 1:4)
            before = deepcopy(a)
            r = JLS.selected_array_inventory((a,), bits)
            @test sizeof(a) == 8 * length(a) # inline eight-byte wrappers, not boxed targets
            @test r.distinct_arrays == r.distinct_backings == 1
            @test r.distinct_limb_memories == 4
            @test only(r.entries).capacity == 4
            @test r.logical_subtotal_bytes == 64 + 4 * per_slot
            @test r.logical_subtotal_bytes == JLS.independent_array_charge(bits, 4)
            @test r.complete_owned_bound === false && r.memory_admission === false
            @test r.reason === :owned_storage_bound_unavailable
            @test a == before && all(getfield(a[i], :d) === actual[i] for i in 1:4)

            # Numerically equal, genuinely independent Memory is not deduplicated.
            x = BigFloat(1); y = deepcopy(x)
            @test x == y && getfield(x, :d) !== getfield(y, :d)
            equal_values = JLS.selected_array_inventory(([x, y],), bits)
            @test equal_values.distinct_limb_memories == 2
            shared_values = JLS.selected_array_inventory((fill(x, 4),), bits)
            @test shared_values.distinct_limb_memories == 1
            @test shared_values.logical_subtotal_bytes == 64 + 4 * 8 + (32 + payload)

            # Repeating an Array does not add a shell; reshape adds a shell only.
            repeated = JLS.selected_array_inventory((a, a), bits)
            @test repeated.logical_subtotal_bytes == r.logical_subtotal_bytes
            matrix = reshape(a, 2, 2)
            @test getfield(getfield(matrix, :ref), :mem) === getfield(getfield(a, :ref), :mem)
            both = JLS.selected_array_inventory((a, matrix), bits)
            @test both.distinct_arrays == 2 && both.distinct_backings == 1
            @test both.distinct_limb_memories == 4
            @test both.logical_subtotal_bytes == r.logical_subtotal_bytes + 40
            @test JLS.selected_array_inventory((matrix,), bits).logical_subtotal_bytes == 72 + 4 * per_slot
            offset_array = Base.wrap(Array, memoryref(getfield(getfield(a, :ref), :mem), 2), 2)
            @test offset_array == a[2:3]
            offset_r = JLS.selected_array_inventory((offset_array,), bits)
            @test only(offset_r.entries).logical_length == 2
            @test offset_r.logical_subtotal_bytes == r.logical_subtotal_bytes # whole backing is reachable
            println("LAYOUT bits=", bits, " payload=", payload, " logical_vector_upper_bytes=", r.logical_subtotal_bytes,
                    " limbs=", r.distinct_limb_memories, " admitted=", r.memory_admission)
        end
    end
end

@testset "deserialized descriptors remain byte-for-byte unchanged" begin
    slot = Base.MPFR.offset_d ÷ 8 + 1
    for bits in (256, 512, 1024)
        io = IOBuffer()
        serialize(io, [BigFloat(i; precision=bits) for i in 1:2])
        seekstart(io); restored = deserialize(io)
        # Do not call precision, equality, or formatting on these BigFloats.
        memories = map(x -> getfield(x, :d), restored)
        before = map(copy, memories)
        @test all(before[i][slot] != UInt(pointer(memories[i]) + Base.MPFR.offset_p) for i in 1:2)
        r = JLS.selected_array_inventory((restored,), bits)
        @test r.distinct_limb_memories == 2
        @test r.memory_admission === false
        @test map(copy, memories) == before
        @test all(memories[i][slot] != UInt(pointer(memories[i]) + Base.MPFR.offset_p) for i in 1:2)
    end
    # Rejection must not repair a same-capacity precision mismatch either.
    io = IOBuffer(); serialize(io, [BigFloat(1; precision=255)]); seekstart(io)
    wrong_precision = deserialize(io)
    raw = getfield(only(wrong_precision), :d); before = copy(raw)
    @test before[slot] != UInt(pointer(raw) + Base.MPFR.offset_p)
    @test_throws ArgumentError JLS.selected_array_inventory((wrong_precision,), 256)
    @test raw == before
end

@testset "whole capacity, hidden references, and unsupported owners" begin
    setprecision(BigFloat, 256) do
        a = [BigFloat(1), BigFloat(2)]
        sizehint!(a, 16)
        mem = getfield(getfield(a, :ref), :mem)
        @test length(mem) > length(a)
        before = JLS.selected_array_inventory((a,), 256)
        @test only(before.entries).capacity == length(mem)
        @test before.logical_subtotal_bytes == 64 + 8 * length(mem) + 2 * 96
        # A reachable pointer beyond visible length must still be counted.
        mem[end] = BigFloat(9)
        after = JLS.selected_array_inventory((a,), 256)
        @test length(a) == 2
        @test after.logical_subtotal_bytes == before.logical_subtotal_bytes + 96
        @test after.distinct_limb_memories == 3
        mem[end] = BigFloat(9; precision=512)
        @test_throws ArgumentError JLS.selected_array_inventory((a,), 256)
        @test_throws ArgumentError JLS.selected_array_inventory((BigFloat[1, 2],), 512)
        almost = BigFloat(1; precision=255)
        @test length(getfield(almost, :d)) == length(getfield(BigFloat(1), :d))
        caught = try
            JLS.selected_array_inventory(([almost],), 256)
            nothing
        catch e
            e
        end
        @test caught isa ArgumentError
        @test occursin("precision does not match", sprint(showerror, caught))

        # Same precision/value, but a nonstandard enlarged native limb Memory.
        x = BigFloat(1)
        raw = getfield(x, :d)
        enlarged = Memory{Base.GMP.Limb}(undef, length(raw) + 1)
        copyto!(enlarged, 1, raw, 1, length(raw)); enlarged[end] = 0
        oversized = Base.MPFR._BigFloat(enlarged)
        @test oversized == x && precision(oversized) == 256
        @test_throws ArgumentError JLS.selected_array_inventory(([oversized],), 256)
        GC.@preserve raw begin
            borrowed_memory = unsafe_wrap(Memory{Base.GMP.Limb}, pointer(raw), length(raw); own=false)
            borrowed_value = Base.MPFR._BigFloat(borrowed_memory)
            @test borrowed_value == x && precision(borrowed_value) == 256
            @test_throws ArgumentError JLS.selected_array_inventory(([borrowed_value],), 256)
        end

        source = Int64[1, 2]
        GC.@preserve source begin
            borrowed = unsafe_wrap(Vector{Int64}, pointer(source), length(source); own=false)
            @test_throws ArgumentError JLS.selected_array_inventory((borrowed,), 256)
        end
        # Ordinary large owned Memory reaches the separate gc-owned-data branch.
        owned = collect(Int64, 1:4096)
        owned_r = JLS.selected_array_inventory((owned,), 256)
        @test only(owned_r.entries).memory_style === :gc_owned
        @test owned_r.logical_subtotal_bytes == 64 + 8 * 4096
        @test JLS.selected_array_inventory((Bool[true, false],), 256).logical_subtotal_bytes == 66
        @test JLS.selected_array_inventory((BigFloat[],), 256).distinct_limb_memories == 0
        @test JLS.selected_array_inventory((), 256).logical_subtotal_bytes == 0
        @test_throws ArgumentError JLS.selected_array_inventory((zeros(Int64, 4097),), 256)
        @test_throws ArgumentError JLS.selected_array_inventory(Tuple(fill(source, 129)), 256)
        @test_throws ArgumentError JLS.selected_array_inventory((view(source, :),), 256)
        @test_throws ArgumentError JLS.selected_array_inventory((Any[x],), 256)
        @test_throws ArgumentError JLS.selected_array_inventory((zeros(Int64, 1, 1, 1),), 256)
        @test_throws ArgumentError JLS.selected_array_inventory((Float64[1],), 256)
    end
end
