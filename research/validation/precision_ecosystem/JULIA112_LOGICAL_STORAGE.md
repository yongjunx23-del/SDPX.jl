# Selected-array logical storage — reference only

`julia112_logical_storage.jl` is not loaded by SDPX. Its result is a diagnostic
subtotal for specified, quiescent concrete arrays, **not** an allocation, phase,
workspace or RSS bound. `complete_owned_bound` and `memory_admission` remain false.

Supported layout: Julia1.12.6 commit15346901f0039751c5488744f1f62de7d87510a8,
aarch64 Darwin, MPFR4.2.2 without patches, GMP6.3.0/64-bit limbs. Only vectors and
matrices of BigFloat/Int64/Bool, capacity≤4096 per array, ≤128 roots, and
BigFloat precision256/512/1024 are inspected. Other platforms need their own
source/ABI review; do not apply this Darwin contract to HPC Linux.

## Source-derived quantities

Installed `share/julia/base/mpfr.jl:123–170` defines BigFloat as an immutable,
eight-byte wrapper around `Memory{Limb}`. Concrete BigFloat array slots inline
that wrapper; **they do not reference separately boxed BigFloat objects**.
The Memory payload contains the32-byte MPFR descriptor and32/64/128 significand
bytes, for64/96/160 total bytes. The descriptor's significand pointer points
inside this same allocation. BigFloat's public `.d` is a different view;
the inspector uses `getfield(x, :d)` to include the descriptor.

`include/julia/julia.h` declares an8-byte GC tag,16-byte Memory body, an optional
8-byte owner field,16-byte embedded MemoryRef and8 bytes per Array dimension.
`julia112_layout_probe.c` independently checks these installed header layouts.
`boot.jl:603–657` shows Array's separate backing Memory. The conservative logical
charges used here are:

- Memory:32+payload bytes (includes an unused owner allowance for inline data).
- Array shell:24+8N bytes, including GC tag; MemoryRef/dimension storage is inline.
- Array backing:32+sizeof(T)×capacity bytes.
- Each distinct BigFloat limb Memory:96/128/192 bytes.

Thus a fresh independent-value BigFloat vector of capacityC has the logical
upper subtotal64+(104/136/200)C; a matrix adds8. This is **not** a physical
allocation formula. A genuinely boxed BigFloat, if reached outside these
concrete arrays, needs additional accounting and is outside this inspector.

## Actual-object checks

The inspector reads the actual backing Memory capacity, including assigned
references outside visible Array length. It deduplicates Array and Memory
identities, not numerical values. Source-defined Memory headers distinguish
inline or self-owned GC storage from borrowed/other-owner storage, which rejects.
Version/ABI checks precede these read-only header accesses. Nonstandard enlarged
BigFloat buffers, precision mismatch, abstract/union layouts, views, oversized
capacity and arithmetic overflow reject. The caller must not mutate inputs
concurrently; this is not a coherent concurrent snapshot protocol. Precision is
read through the pinned `x.prec` accessor: `precision(x)` would call MPFR and may
repair a deserialized descriptor's embedded pointer. Deserialization tests compare
all raw words on both successful inspection and precision-mismatch rejection.

The validation runner separately captures installed source/library fingerprints
and unchanged source/input evidence. Layout/version agreement or those hashes do
not prove arbitrary runtime build settings or C-scratch bounds.

## Still missing

The root tuple, inspector dictionaries, scalar/CSC/wrapper shells, unlisted live
objects, construction/replacement overlap, retained results and MPFR/GMP scratch
are not included. Allocator rounding/buffer metadata, GC reserves and process
RSS are also excluded. A sum of all finite allocation events could conservatively
avoid precise lifetime analysis, but every event and primitive scratch bound
would still need evidence. No multiplier supplies those missing facts.

The SDPX memory-admitting entry remains unavailable; this reference cannot enable
it. Natural ordering removes AMD machinery only under that explicit mode; the
shape-only sparse-core inventory still does not establish actual provider mode.
