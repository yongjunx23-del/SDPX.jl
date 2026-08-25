# v2 allocation / type audit (Wave A-4)

Branch refactor/zeroalloc-factorcache-hsd-v2 (new main). Per-site Julia heap
allocation of one full predictor-corrector Newton step (SDP route, Float64),
measured with Profile.Allocs (sample_rate=1.0) via
benchmark/v2_allocation_audit.jl.

## Per-iteration total
- @allocated (min-of-3): 8224 B
- Profile.Allocs total: 8424 B (includes profiler overhead)

## Top allocation sites (Float64, one newton_step)
| site | bytes | likely source |
|---|---|---|
| src/step.jl:951 | 2800 | newton_step return NamedTuple / IterationDiagnostics |
| src/step.jl:686 | 1680 | factorize! result boxing (phase timings / status tuple) |
| src/step.jl:680 | 816 | schur build / closure |
| src/step.jl:808 | 720 | IterationDiagnostics construction |
| src/kkt.jl:921 | 464 | equality Gram / factor path |
| src/kkt.jl:1695 | 448 | factorize internals |
| src/kkt.jl:1857 | 240 | factorize result NamedTuple |
| src/step.jl:695 | 120 | kkt_phases destructure |
| src/schur.jl:939 | 96 | schur build reshape |
| src/step.jl:944 | 96 | predictor rhs |
| src/schur.jl:829 | 96 | schur build reshape |

## Wave C targets (in priority order)
1. step.jl:951 return NamedTuple + IterationDiagnostics (~2.8 KB) -> relocate to
   preallocated cold-state record (HotSolverState / ColdDiagnostics split).
2. step.jl:686 factorize! result boxing (~1.7 KB) -> write status/phase times
   into preallocated workspace fields; return small status enum.
3. step.jl:680/808 closures + IterationDiagnostics (~1.5 KB) -> inline closures,
   preallocate diagnostics.
4. kkt.jl equality-gram / factor path (~1.1 KB) -> reuse factor handle storage.

## Type-stability notes
- Workspace.backend is typed (Union{Nothing,KKTBackend}) on new main.
- Remaining Any/abstract hot-state fields to audit: sparse_kkt, DenseAugmented
  factor-diagnostics fields (cold, low priority).

## Hard gate (Wave C)
Float64/x2/x3/x4/BigFloat256 one_full_iteration == 0 Julia bytes.

