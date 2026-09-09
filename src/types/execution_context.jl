# R2-A execution context for session-local symbolic reuse and internal options.
# Explicitly passed through the Prepared bridge; never exposed in public API.

struct NativeExecutionContext
    symbolic_lease::Union{Nothing, SessionSymbolicLease}
    structure_generation::UInt64
    prepared_fingerprint::UInt64
end

NativeExecutionContext() = NativeExecutionContext(nothing, UInt64(0), UInt64(0))
