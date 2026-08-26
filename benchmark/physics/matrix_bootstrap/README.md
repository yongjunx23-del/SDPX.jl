# O(2) matrix-bootstrap build artifact

`matrix_bootstrap.jl` provides a deterministic, exact-rational, low-order
relaxation for the D=2 bosonic matrix bootstrap of Lin--Zheng.  It is
build-first: each fixed `scan_value` is one affine SDP slice, while the
nonlinear scan is never exposed as a single linear objective.

```julia
include("benchmark/physics/matrix_bootstrap/matrix_bootstrap.jl")
using .MatrixBootstrap

artifact = build_lin_zheng(
    level=8,
    mass2=1,
    scan_observable=:x2,
    scan_value=77800898 // 100000000,
)
artifact.reference_status        # :build_only
artifact.objective               # nothing
artifact.fingerprint              # SHA-256
artifact.equalities               # SparseMatrixCSC{Rational{BigInt},Int}
artifact.psd_blocks               # exact affine PSD metadata
artifact.metadata.supported_scope # :medium
artifact.metadata.paper_equivalent # false
validate_artifact(artifact)       # true, including deterministic rebuild
```

The supported build scopes are level 4 (`:tiny`), level 6 (`:small`), and
level 8 (`:medium`).  Momentum-containing traces remain ordered: moving a
momentum around the trace is not treated as a free cyclic rotation.  The
low-order quotient implements the Appendix-E E34 canonical pair exactly, the
E40 energy slice, and the E42 ground-state positivity block.  The full gauge
Ward tower, high-order momentum/multi-trace quotient, complete indexed-irrep
decomposition, and paper Tables 1--2 reproduction are deliberately absent.

The raw word counts are therefore not the published post-quotient counts.
Those counts and intervals are retained only as fail-closed provenance
metadata; they are not acceptance oracles.  See
[`PROVENANCE.md`](PROVENANCE.md) for the paper equations, source versions,
stable PRL citation, and interval semantics.

`catalog.jl` registers levels 4/6/8 with the shared benchmark harness. Their
reference status is `:build_only`: the runner measures construction and
validates the exact artifact fingerprint without invoking a solver.
