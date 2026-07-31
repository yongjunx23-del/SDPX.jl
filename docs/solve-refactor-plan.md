# Staged solver-file decomposition

The 0.3 release begins a staged split of the former 2,600-line `src/solve.jl`.
The goal is reviewability and testable ownership boundaries, not a cosmetic
shuffle or a simultaneous rewrite of the numerical algorithm.

## Stage 1 — complete in 0.3

`src/solve.jl` is now a small implementation manifest. The existing method
order is preserved intact in `src/solver/interior_point.jl`. This changes no
dispatch, workspace lifetime, or floating-point operation order and gives
later stages a stable directory boundary.

## Stage 2 — low-risk support extraction

Move checkpoint I/O, option replacement/rerounding, warm-start validation,
printing, and initial-point policy into focused files. Each move is mechanical
and is accepted only after source-compatibility tests, checkpoint round trips,
and the full numerical suite pass.

## Stage 3 — iteration state and termination

Introduce an internal iteration-state record and extract:

- residual and merit evaluation;
- best-iterate retention;
- adaptive-parameter history;
- restart and stagnation decisions;
- terminal-result construction.

This stage may change ownership and is therefore a separate minor-version
change. Allocation gates and bitwise Float64 small fixtures accompany each
extraction.

## Stage 4 — orchestration and precision retries

Move presolve/planning/reconstruction, staged BigFloat retries, and the simple
public overloads into separate frontend files. The core iteration receives an
already prepared execution plan and returns a raw result; original-coordinate
certification remains a mandatory outer boundary.

## Stage 5 — optional algorithm modules

Only after the earlier boundaries are stable should HSD state, alternate
search directions, or distributed Schur backends become independent solver
modules. Task_Low08, medium CSDR, LP/MOI, Float64x4, and BigFloat regression
gates remain shared so modularity cannot silently create divergent solvers.
