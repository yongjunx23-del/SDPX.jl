# Qualified internals

These names support inspection, custom preprocessing, and solver research.
Use them with explicit `SDPX.name` qualification; they are intentionally not
exported and their result layouts may change before 1.0.

```@docs
SDPX
```

## Problem and execution descriptions

```@docs
SDPX.SDPProblem
SDPX.SolveMode
SDPX.ActiveSparseCoefficientVector
SDPX.StructureAnalysis
SDPX.ProblemClassification
SDPX.PresolveReport
SDPX.ExecutionPlan
SDPX.SolveDiagnostics
```

## Adaptive controller

```@docs
SDPX.FixedParameterPolicy
SDPX.AdaptiveParameterPolicy
SDPX.IterationDiagnostics
SDPX.IterationParameters
SDPX.select_parameters
```

## Preprocessing

```@docs
SDPX.preprocess
SDPX.PreprocessReport
SDPX.PreprocessStageReport
SDPX.ReconstructionMap
```

## Formulation and sparse planning

The static formulation planner and provider-neutral sparse layer are exposed
through their diagnostic entry points; see [architecture.md](architecture.md),
[providers.md](providers.md), and [sparse-execution.md](sparse-execution.md).

## Spectrum and BLAS inspection

```@docs
SDPX.SpectrumResult
SDPX.blas_backend
SDPX.blas_threads
SDPX.set_blas_threads!
```
