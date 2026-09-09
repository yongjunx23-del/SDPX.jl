"""
Versioned, read-only diagnostics for selected concrete Julia arrays.

NOT a workspace/phase/allocator/RSS bound and never memory-admission authority.
Inputs must be quiescent. Only the pinned Darwin/aarch64 Julia 1.12.6 layout,
Vector/Matrix{BigFloat,Int64,Bool}, and 256/512/1024-bit BigFloats are covered.
"""
module Julia112LogicalStorage

using SHA, Libdl

const CONTRACT_ID = :julia1126_darwin_aarch64_selected_arrays_v1
const PRECISIONS = (256, 512, 1024)
const PAYLOAD_BYTES = (64, 96, 160) # MPFR descriptor AND significand
const MAX_CAPACITY = 4096
const MAX_ROOTS = 128
const EXPECTED_FACTS = (
    julia_version=v"1.12.6",
    julia_commit="15346901f0039751c5488744f1f62de7d87510a8",
    kernel=:Darwin, arch=:aarch64, word_bits=64,
    int_bytes=8, clong_bytes=8, cint_bytes=4, limb_bytes=8,
    mpfr_version=v"4.2.2", gmp_version=v"6.3.0", mpfr_patches=(),
    bigfloat_mutable=false, bigfloat_inline=true, bigfloat_bytes=8,
    bigfloat_fields=(:d,), bigfloat_memory_field=true, mpfr_offsets=(0, 8, 16, 24, 32),
    custom_bytes=(32, 64, 128),
)

function check_facts(facts)
    for name in keys(EXPECTED_FACTS)
        hasproperty(facts, name) && getproperty(facts, name) == getproperty(EXPECTED_FACTS, name) ||
            throw(ArgumentError("unsupported logical-storage runtime fact: $name"))
    end
    return nothing
end

function runtime_facts()
    # Reject before touching version-specific fields or reading object headers.
    VERSION == EXPECTED_FACTS.julia_version &&
    Base.GIT_VERSION_INFO.commit == EXPECTED_FACTS.julia_commit &&
    Sys.KERNEL === :Darwin && Sys.ARCH === :aarch64 && Sys.WORD_SIZE == 64 ||
        throw(ArgumentError("unsupported logical-storage runtime/platform"))
    facts = (
        julia_version=VERSION, julia_commit=Base.GIT_VERSION_INFO.commit,
        kernel=Sys.KERNEL, arch=Sys.ARCH, word_bits=Sys.WORD_SIZE,
        int_bytes=sizeof(Int), clong_bytes=sizeof(Clong), cint_bytes=sizeof(Cint),
        limb_bytes=sizeof(Base.GMP.Limb), mpfr_version=Base.MPFR.version(),
        gmp_version=Base.GMP.version(),
        mpfr_patches=Tuple(filter(!isempty, Base.MPFR.patches())),
        bigfloat_mutable=ismutabletype(BigFloat),
        bigfloat_inline=Base.allocatedinline(BigFloat), bigfloat_bytes=sizeof(BigFloat),
        bigfloat_fields=fieldnames(BigFloat),
        bigfloat_memory_field=fieldtype(BigFloat, :d) === Memory{Base.GMP.Limb},
        mpfr_offsets=(Base.MPFR.offset_prec, Base.MPFR.offset_sign, Base.MPFR.offset_exp,
                      Base.MPFR.offset_d, Base.MPFR.offset_p),
        custom_bytes=ntuple(i -> Int(ccall((:mpfr_custom_get_size, Base.MPFR.libmpfr),
            Csize_t, (Clong,), PRECISIONS[i])), 3),
    )
    check_facts(facts)
    return facts
end

"""Actual installed source/library fingerprints; evidence, not a C-scratch proof."""
function runtime_fingerprints()
    root = realpath(joinpath(Sys.BINDIR, ".."))
    paths = (
        mpfr_source=joinpath(root, "share/julia/base/mpfr.jl"),
        boot_source=joinpath(root, "share/julia/base/boot.jl"),
        array_source=joinpath(root, "share/julia/base/array.jl"),
        runtime_header=joinpath(root, "include/julia/julia.h"),
        mpfr_library=realpath(Libdl.dlpath(Base.MPFR.libmpfr)),
        gmp_library=realpath(Libdl.dlpath(Base.GMP.libgmp)),
    )
    return map(p -> (path=p, sha256=bytes2hex(sha256(read(p)))), paths)
end

function _payload_bytes(bits::Int)
    i = findfirst(==(bits), PRECISIONS)
    i === nothing && throw(ArgumentError("unsupported logical-storage precision $bits"))
    return PAYLOAD_BYTES[i]
end

# Logical declared fields + GC tag + conservative optional owner field.
# Excludes allocator rounding, separately allocated buffer metadata and GC reserve.
_memory_charge(payload::Int) = Base.checked_add(32, payload)
_array_shell(rank::Int) = Base.checked_add(24, Base.checked_mul(8, rank))

"""Independent-value formula, not actual capacity discovery or an admission gate."""
function independent_array_charge(bits::Int, capacity::Int, rank::Int=1)
    capacity >= 0 && rank in (1, 2) || throw(ArgumentError("invalid array shape"))
    payload = _payload_bytes(bits)
    per_value = Base.checked_add(8, _memory_charge(payload))
    return Base.checked_add(Base.checked_add(_array_shell(rank), 32),
        Base.checked_mul(capacity, per_value))
end

# julia.h: jl_genericmemory_t and jl_genericmemory_how. Called ONLY after
# runtime_facts checks the supported ABI. No header or MPFR mutation occurs.
function _owned_memory_style(mem)
    GC.@preserve mem begin
        body = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), mem)
        data = Base.unsafe_convert(Ptr{Cvoid}, mem)
        # Validate the two declared body fields before interpreting owner storage.
        unsafe_load(Ptr{UInt}(body)) == UInt(length(mem)) ||
            throw(ArgumentError("unexpected Memory length layout"))
        unsafe_load(Ptr{Ptr{Cvoid}}(body + 8)) == data ||
            throw(ArgumentError("unexpected Memory pointer layout"))
        data == body + 16 && return :inline
        owner = unsafe_load(Ptr{Ptr{Cvoid}}(body + 16))
        owner == body && return :gc_owned
        throw(ArgumentError("external or unaccounted Memory owner"))
    end
end

"""
    selected_array_inventory(arrays::Tuple, precision_bits::Int)

Inspect actual backing capacity and all assigned BigFloat references, including
those outside logical Array length. Deduplicate Array and Memory identities,
never BigFloat numerical values. Reject unsupported layouts/owners/precision.

The root tuple, inspector scratch, scalar/CSC/wrapper shells, other live roots,
construction overlap, C scratch and retained results are NOT inventoried.
The returned logical subtotal cannot authorize any solve or allocation.
"""
function selected_array_inventory(arrays::Tuple, precision_bits::Int)
    facts = runtime_facts()
    payload = _payload_bytes(precision_bits)
    length(arrays) <= MAX_ROOTS || throw(ArgumentError("too many diagnostic roots"))
    array_seen = IdDict{Any,Nothing}()
    backing_seen = IdDict{Any,Nothing}()
    limbs_seen = IdDict{Any,Nothing}()
    entries = NamedTuple[]
    total = 0
    for (root_index, A) in pairs(arrays)
        A isa Array && ndims(A) in (1, 2) && eltype(A) in (BigFloat, Int64, Bool) ||
            throw(ArgumentError("unsupported diagnostic array layout"))
        haskey(array_seen, A) && continue
        array_seen[A] = nothing
        T = eltype(A)
        mem = getfield(getfield(A, :ref), :mem)
        typeof(mem) === Memory{T} || throw(ArgumentError("unsupported backing Memory kind"))
        capacity = length(mem)
        capacity <= MAX_CAPACITY || throw(ArgumentError("diagnostic capacity limit"))
        capacity >= length(A) || throw(ArgumentError("inconsistent backing capacity"))
        style = _owned_memory_style(mem)
        shell_bytes = _array_shell(ndims(A))
        backing_bytes = 0
        limb_bytes = 0
        new_limbs = 0
        if !haskey(backing_seen, mem)
            backing_seen[mem] = nothing
            backing_bytes = _memory_charge(Base.checked_mul(capacity, sizeof(T)))
            if T === BigFloat
                for i in eachindex(mem)
                    isassigned(mem, i) || continue
                    x = mem[i]
                    limbs = getfield(x, :d) # NOT x.d (BigFloatData without descriptor)
                    Base.checked_mul(length(limbs), sizeof(Base.GMP.Limb)) == payload ||
                        throw(ArgumentError("unsupported BigFloat backing size"))
                    _owned_memory_style(limbs)
                    # precision(x) enters MPFR and can repair a deserialized
                    # significand pointer. The pinned property accessor only reads.
                    x.prec == precision_bits ||
                        throw(ArgumentError("BigFloat precision does not match diagnostic contract"))
                    if !haskey(limbs_seen, limbs)
                        limbs_seen[limbs] = nothing
                        limb_bytes = Base.checked_add(limb_bytes, _memory_charge(payload))
                        new_limbs += 1
                    end
                end
            end
        end
        subtotal = Base.checked_add(shell_bytes, Base.checked_add(backing_bytes, limb_bytes))
        total = Base.checked_add(total, subtotal)
        push!(entries, (root_index=root_index, element_type=string(T), rank=ndims(A),
            logical_length=length(A), capacity=capacity, memory_style=style,
            shell_bytes=shell_bytes, new_backing_bytes=backing_bytes,
            new_limb_memories=new_limbs, new_limb_bytes=limb_bytes, subtotal=subtotal))
    end
    return (contract=CONTRACT_ID, metric=:logical_selected_array_storage_upper,
        runtime=facts, precision_bits=precision_bits, logical_subtotal_bytes=total,
        distinct_arrays=length(array_seen), distinct_backings=length(backing_seen),
        distinct_limb_memories=length(limbs_seen), entries=entries,
        complete_owned_bound=false, memory_admission=false,
        reason=:owned_storage_bound_unavailable,
        unresolved=(:other_roots_and_shells, :inspector_scratch, :operation_scratch,
                    :construction_overlap, :retained_results, :allocator_and_gc))
end

end # module
